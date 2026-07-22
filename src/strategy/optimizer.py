"""Leakage-safe threshold optimization for the Track A signal engine."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np
import pandas as pd

from src.evaluation.deflated_sharpe import deflated_sharpe_ratio
from src.models.volatility_forecaster import QRWVolatilityForecaster
from src.strategy.signal_engine import QRWSignalEngine


class QRWStrategyOptimizer:
    """Optimize signal thresholds in-sample and apply them out-of-sample."""

    DEFAULT_GRID = np.round(np.arange(0.52, 0.75, 0.01), 2).tolist()

    def __init__(
        self,
        signal_engine: QRWSignalEngine | None = None,
        vol_forecaster: QRWVolatilityForecaster | None = None,
    ) -> None:
        self.signal_engine = signal_engine or QRWSignalEngine()
        self.vol_forecaster = vol_forecaster or QRWVolatilityForecaster(None)
        self.best_params: dict[str, dict] = {}
        self.search_surface = pd.DataFrame()

    def _regimes(self, tick_df: pd.DataFrame, probability_df: pd.DataFrame) -> pd.Series:
        prices = tick_df["price"].to_numpy(dtype=float)
        log_returns = pd.Series(np.diff(np.log(prices), prepend=np.log(prices[0])))
        causal_vol = log_returns.rolling(30, min_periods=10).std(ddof=0).shift(1)
        causal_vol = causal_vol.fillna(0.0) * self.vol_forecaster.ANNUALIZE
        rows = probability_df["row"].to_numpy(dtype=int)
        return pd.Series(
            [self.vol_forecaster.vol_regime(float(causal_vol.iloc[row])) for row in rows],
            index=probability_df.index,
            dtype="object",
        )

    def _deflate(
        self,
        frame: pd.DataFrame,
        selected: dict,
        regime_rows: list[dict],
        n_candidates: int,
    ) -> dict[str, float]:
        """Judge the winner against what the luckiest of N trials would show.

        Selection happens before this runs, so the point is not to change which
        thresholds are chosen but to say honestly whether the winner survives
        the fact that it was the best of ``n_candidates`` noisy trials.
        """
        engine = QRWSignalEngine(
            theta_buy=selected["theta_buy"],
            theta_sell=selected["theta_sell"],
            transaction_cost=self.signal_engine.transaction_cost,
        )
        pnl = engine.backtest_from_probabilities(frame)["pnl"].to_numpy(dtype=float)
        trial_sharpes = np.array(
            [row.get("sharpe_per_observation", np.nan) for row in regime_rows],
            dtype=float,
        )
        try:
            return deflated_sharpe_ratio(
                pnl, n_trials=n_candidates, trial_sharpes=trial_sharpes
            )
        except ValueError as error:
            # A flat or degenerate P&L series has no Sharpe to deflate.
            return {"deflated_sharpe_ratio": float("nan"), "reason": str(error)}

    def _metrics_for_thresholds(
        self,
        frame: pd.DataFrame,
        theta_buy: float,
        theta_sell: float,
    ) -> dict[str, float | int]:
        # Delegates to the signal engine rather than recomputing. The local
        # copy this replaces had drifted in three ways: it charged no
        # transaction cost at all (so the grid was optimised as if trading were
        # free, which at 5bps against ~4e-7 tick moves inverts the answer), it
        # counted bars-in-position as trades, and it reported a t-statistic
        # under the name "sharpe".
        engine = QRWSignalEngine(
            theta_buy=theta_buy,
            theta_sell=theta_sell,
            transaction_cost=self.signal_engine.transaction_cost,
        )
        backtest = engine.backtest_from_probabilities(frame)
        return engine.compute_signal_metrics(backtest)

    @staticmethod
    def _objective(metrics: dict[str, float | int], objective: str) -> float:
        if objective in {"sharpe", "t_stat"}:
            # "sharpe" is kept as an accepted alias because callers and saved
            # configs use it, but the quantity optimised is the t-statistic.
            return float(metrics["t_stat"])
        if objective == "hit_rate":
            return float(metrics["hit_rate"])
        if objective == "profit_factor":
            return float(metrics["profit_factor"])
        if objective == "net_pnl":
            # The only objective denominated in money after costs. hit_rate and
            # profit_factor can both improve while net P&L falls, because
            # neither charges for the trades taken to achieve them.
            return float(metrics["net_pnl"])
        raise ValueError(
            "objective must be sharpe/t_stat, hit_rate, profit_factor, or net_pnl"
        )

    def grid_search(
        self,
        train_df: pd.DataFrame,
        qrw_model=None,
        theta_grid: list[float] | None = None,
        objective: str = "sharpe",
        *,
        min_trades: int = 10,
        max_drawdown: float = 0.15,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, dict]:
        """Search only ``train_df`` and retain every evaluated combination."""
        grid = [float(value) for value in (theta_grid or self.DEFAULT_GRID)]
        if not grid or any(not 0.5 <= value < 1.0 for value in grid):
            raise ValueError("theta_grid values must be in [0.5, 1.0)")
        if int(min_trades) < 0 or max_drawdown <= 0:
            raise ValueError("constraints must be non-negative")
        probabilities = self.signal_engine.build_probability_frame(train_df, qrw_model)
        probabilities["regime"] = self._regimes(train_df, probabilities)
        groups: dict[str, pd.DataFrame] = {"ALL": probabilities}
        for regime in ("LOW", "MID", "HIGH"):
            subset = probabilities[probabilities["regime"] == regime]
            if len(subset):
                groups[regime] = subset

        total = len(groups) * len(grid) ** 2
        n_candidates = len(grid) ** 2
        completed = 0
        surface_rows: list[dict] = []
        best: dict[str, dict] = {}
        for regime, frame in groups.items():
            candidates: list[tuple[float, dict]] = []
            for theta_buy, theta_sell in product(grid, grid):
                metrics = self._metrics_for_thresholds(frame, theta_buy, theta_sell)
                # Selection is by the objective alone. The ad hoc penalty that
                # used to sit here -- sqrt(log(n_candidates))/sqrt(n_trades)
                # scaled by uncalibrated 0.5/0.1 constants -- has been replaced
                # by deflating the winner's Sharpe after the search, which is
                # the published treatment of selection bias and accounts for
                # skew, kurtosis and the variance of the trial Sharpes.
                score = self._objective(metrics, objective)

                eligible = (
                    metrics["n_trades"] >= min_trades
                    and metrics["max_drawdown"] >= -float(max_drawdown)
                    and np.isfinite(score)
                )
                surface_rows.append(
                    {
                        "regime": regime,
                        "theta_buy": theta_buy,
                        "theta_sell": theta_sell,
                        "score": score,
                        "eligible": bool(eligible),
                        **metrics,
                    }
                )
                if eligible:
                    candidates.append((score, {"theta_buy": theta_buy, "theta_sell": theta_sell, "score": score, "metrics": metrics}))
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)
            if candidates:
                selected = max(candidates, key=lambda item: (item[0], -item[1]["theta_buy"], -item[1]["theta_sell"]))[1]
                selected["constraint_relaxed"] = False
            else:
                regime_rows = [row for row in surface_rows if row["regime"] == regime]
                fallback = max(regime_rows, key=lambda row: row["score"])
                selected = {
                    "theta_buy": fallback["theta_buy"],
                    "theta_sell": fallback["theta_sell"],
                    "score": fallback["score"],
                    "metrics": {
                        key: fallback[key]
                        for key in (
                            "hit_rate",
                            "profit_factor",
                            "net_pnl",
                            "max_drawdown",
                            "n_trades",
                            "t_stat",
                        )
                    },
                    "constraint_relaxed": True,
                }
            selected["n_evaluated"] = len(grid) ** 2
            selected["deflated_sharpe"] = self._deflate(
                frame,
                selected,
                [row for row in surface_rows if row["regime"] == regime],
                n_candidates,
            )
            best[regime] = selected

        self.search_surface = pd.DataFrame(surface_rows)
        self.best_params = best
        return best

    def evaluate_out_of_sample(
        self,
        test_df: pd.DataFrame,
        qrw_model=None,
        best_params: dict[str, dict] | None = None,
    ) -> dict:
        """Apply frozen parameters using only the supplied test period."""
        parameters = best_params or self.best_params
        if not parameters:
            raise ValueError("best_params are required before OOS evaluation")
        probabilities = self.signal_engine.build_probability_frame(test_df, qrw_model)
        probabilities["regime"] = self._regimes(test_df, probabilities)

        regimes = probabilities["regime"].to_numpy()
        theta_buy = np.empty(len(probabilities), dtype=np.float64)
        theta_sell = np.empty(len(probabilities), dtype=np.float64)
        for regime in pd.unique(regimes):
            params = parameters.get(str(regime), parameters.get("ALL"))
            if not params:
                raise ValueError(f"no parameters available for regime {regime}")
            mask = regimes == regime
            theta_buy[mask] = params["theta_buy"]
            theta_sell[mask] = params["theta_sell"]

        # Vectorized re-implementation of QRWSignalEngine.backtest_from_probabilities
        # with a per-row theta_buy/theta_sell (regime parameters can differ
        # row to row). The prior per-row loop scored each row via its own
        # single-row DataFrame, which made trade_size an isolated
        # abs(position) rather than a true diff against the previous row --
        # preserved here exactly (this is a speed rewrite, not a behavior
        # change), along with QRWSignalEngine's default transaction_cost,
        # which the per-row loop always used since only theta_buy/theta_sell
        # were ever passed to its constructor.
        default_transaction_cost = QRWSignalEngine().transaction_cost
        p_up = probabilities["p_up"].to_numpy(dtype=float)
        p_down = probabilities["p_down"].to_numpy(dtype=float)
        ret_1step = probabilities["ret_1step"].to_numpy(dtype=float)

        buy = p_up > theta_buy
        sell = (~buy) & (p_down > theta_sell)
        positions = np.where(buy, 1.0, np.where(sell, -1.0, 0.0))
        trade_size = np.abs(positions)

        backtest = probabilities.copy()
        backtest["theta_buy"] = theta_buy
        backtest["theta_sell"] = theta_sell
        backtest["signal"] = np.where(buy, "BUY", np.where(sell, "SELL", "HOLD"))
        backtest["confidence"] = np.clip(np.maximum(p_up, p_down) - 0.5, 0.0, 0.5)
        backtest["pnl"] = positions * ret_1step - trade_size * default_transaction_cost
        backtest["correct"] = np.where(
            positions == 0, "HOLD", np.where(positions * ret_1step > 0, "WIN", "LOSS")
        )
        backtest["momentum"] = p_up - p_down
        metrics = QRWSignalEngine.compute_signal_metrics(backtest)
        equity = backtest["pnl"].cumsum()
        drawdown = equity - equity.cummax().clip(lower=0.0)
        return {
            **metrics,
            "backtest": backtest,
            "equity_curve": equity.to_numpy(),
            "drawdown_curve": drawdown.to_numpy(),
        }

    def regime_adaptive_signal(
        self,
        current_vol: float,
        amplitude_state: np.ndarray,
    ) -> dict[str, float | str]:
        regime = self.vol_forecaster.vol_regime(current_vol)
        params = self.best_params.get(regime)
        if params is None:
            import warnings
            warnings.warn(f"No params for regime {regime}, using ALL fallback")
            params = self.best_params.get("ALL", {})
        theta_buy = float(params.get("theta_buy", self.signal_engine.theta_buy))
        theta_sell = float(params.get("theta_sell", self.signal_engine.theta_sell))
        result = QRWSignalEngine(theta_buy, theta_sell).amplitude_to_signal(amplitude_state)
        result.update({"regime_used": regime, "theta_buy": theta_buy, "theta_sell": theta_sell})
        return result

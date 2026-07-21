"""Leakage-safe threshold optimization for the Track A signal engine."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np
import pandas as pd

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

    @staticmethod
    def _metrics_for_thresholds(
        frame: pd.DataFrame,
        theta_buy: float,
        theta_sell: float,
    ) -> dict[str, float | int]:
        p_up = frame["p_up"].to_numpy(dtype=float)
        p_down = frame["p_down"].to_numpy(dtype=float)
        future = frame["ret_1step"].to_numpy(dtype=float)
        buy = p_up > theta_buy
        sell = (~buy) & (p_down > theta_sell)
        position = np.where(buy, 1.0, np.where(sell, -1.0, 0.0))
        pnl = position * future
        traded = position != 0
        trade_pnl = pnl[traded]
        gross_profit = float(trade_pnl[trade_pnl > 0].sum())
        gross_loss = float(-trade_pnl[trade_pnl < 0].sum())
        cumulative = np.cumsum(pnl)
        peak = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
        drawdown = cumulative - peak
        pnl_std = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
        return {
            "hit_rate": float(np.mean(trade_pnl > 0)) if len(trade_pnl) else 0.0,
            "profit_factor": gross_profit / max(gross_loss, 1e-12),
            "net_pnl": float(pnl.sum()),
            "max_drawdown": float(drawdown.min(initial=0.0)),
            "n_trades": int(traded.sum()),
            "sharpe": float(np.mean(pnl) / pnl_std * np.sqrt(len(pnl))) if pnl_std > 0 else 0.0,
        }

    @staticmethod
    def _objective(metrics: dict[str, float | int], objective: str) -> float:
        if objective == "sharpe":
            return float(metrics["sharpe"])
        if objective == "hit_rate":
            return float(metrics["hit_rate"])
        if objective == "profit_factor":
            return float(metrics["profit_factor"])
        raise ValueError("objective must be sharpe, hit_rate, or profit_factor")

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
                raw_score = self._objective(metrics, objective)
                
                # Ad hoc multiple-testing penalty: grows with log(n_candidates)
                # tried in the grid search and shrinks with sqrt(n_trades).
                # This is NOT the Deflated Sharpe Ratio (Bailey & Lopez de
                # Prado) -- it does not account for skew, kurtosis, or the
                # variance of the maximum Sharpe under repeated trials. The
                # 0.5/0.1 coefficients below are uncalibrated constants.
                penalty = np.sqrt(np.log(n_candidates)) / max(np.sqrt(metrics["n_trades"]), 1.0)
                if objective == "sharpe":
                    score = raw_score - penalty * 0.5
                elif objective == "hit_rate":
                    score = raw_score - penalty * 0.1
                elif objective == "profit_factor":
                    score = raw_score * np.exp(-penalty)
                else:
                    score = raw_score

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
                    "metrics": {key: fallback[key] for key in ("hit_rate", "profit_factor", "net_pnl", "max_drawdown", "n_trades", "sharpe")},
                    "constraint_relaxed": True,
                }
            selected["n_evaluated"] = len(grid) ** 2
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

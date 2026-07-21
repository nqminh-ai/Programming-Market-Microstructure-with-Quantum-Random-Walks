"""Causal QRW volatility forecasts and deterministic GARCH benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.coin_operators import obi_adaptive_coin


class QRWVolatilityForecaster:
    """Estimate volatility from the spread of a QRW distribution.

    A forecast at row ``i`` only uses rows before ``i``.  The QRW's coin is
    refreshed from the trailing OBI/tick-direction signal before each run, so
    the resulting lattice spread depends on realized market state rather than
    only on ``window`` — otherwise ``reset()+run(window)`` is deterministic
    and the "forecast" degenerates into a constant rescaling of the realized
    volatility it is being compared against.
    """

    ANNUALIZE = float(np.sqrt(252 * 6.5 * 3600))

    def __init__(
        self,
        qrw_model,
        window_sizes: list[int] | None = None,
        *,
        alpha_obi: float = 1.0,
        alpha_direction: float = 0.0,
    ) -> None:
        self.qrw = qrw_model
        self.windows = list(window_sizes or [5, 15, 30])
        if not self.windows or any(int(window) < 2 for window in self.windows):
            raise ValueError("window sizes must contain integers >= 2")
        if not np.isfinite(alpha_obi) or alpha_obi < 0.0:
            raise ValueError("alpha_obi must be finite and non-negative")
        if not np.isfinite(alpha_direction):
            raise ValueError("alpha_direction must be finite")
        # Uncalibrated sensitivity constants controlling how strongly the
        # lattice coin responds to OBI/direction signal — not fit from data;
        # override with calibrated values if/when they become available.
        self.alpha_obi = float(alpha_obi)
        self.alpha_direction = float(alpha_direction)

    @staticmethod
    def _probability(state: np.ndarray) -> np.ndarray:
        values = np.asarray(state).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("state must be a non-empty finite array")
        if np.iscomplexobj(values):
            probability = np.abs(values) ** 2
        else:
            probability = values.astype(np.float64, copy=True)
            if np.any(probability < 0):
                raise ValueError("a real state must contain probabilities >= 0")
        total = float(probability.sum())
        if total <= 0:
            raise ValueError("state must have positive probability mass")
        return probability / total

    def compute_distribution_variance(self, amplitude_state: np.ndarray) -> float:
        """Return the normalized position variance of an amplitude/probability."""
        probability = self._probability(amplitude_state)
        positions = np.arange(len(probability), dtype=np.float64) - len(probability) // 2
        mean = float(positions @ probability)
        variance = float(((positions - mean) ** 2) @ probability)
        return max(variance, 0.0)

    @staticmethod
    def _trailing_market_signal(
        tick_df: pd.DataFrame,
        window: int,
        end_index: int,
    ) -> tuple[float, float]:
        """Return ``(mean_obi, net_direction)`` for the trailing ``window``
        events ending just before ``end_index``.

        Uses real ``obi``/``tick_direction`` columns when present; otherwise
        derives both from the trailing price direction, matching the same
        fallback convention already used by the dashboard
        (``sign(price.diff())``, rolling-averaged for OBI).
        """
        start_index = max(end_index - window, 0)
        if "tick_direction" in tick_df.columns:
            direction_window = tick_df["tick_direction"].to_numpy(
                dtype=np.float64
            )[start_index:end_index]
        else:
            prices = tick_df["price"].to_numpy(dtype=np.float64)[
                start_index:end_index
            ]
            direction_window = (
                np.sign(np.diff(prices, prepend=prices[0]))
                if prices.size
                else prices
            )
        if "obi" in tick_df.columns:
            obi_window = tick_df["obi"].to_numpy(dtype=np.float64)[
                start_index:end_index
            ]
        elif direction_window.size:
            obi_window = (
                pd.Series(direction_window)
                .rolling(20, min_periods=1)
                .mean()
                .to_numpy()
            )
        else:
            obi_window = direction_window
        mean_obi = (
            float(np.clip(np.mean(obi_window), -1.0, 1.0))
            if obi_window.size
            else 0.0
        )
        net_direction = (
            float(np.sign(np.sum(direction_window)))
            if direction_window.size
            else 0.0
        )
        return mean_obi, net_direction

    def _refresh_market_coin(
        self,
        tick_df: pd.DataFrame,
        window: int,
        end_index: int,
    ) -> None:
        """Set the QRW's coin from realized market state before a run."""
        if not hasattr(self.qrw, "set_coin"):
            raise RuntimeError(
                "qrw_model must support set_coin() so the forecast can "
                "depend on realized market state, not only on window"
            )
        mean_obi, net_direction = self._trailing_market_signal(
            tick_df, window, end_index
        )
        self.qrw.set_coin(
            obi_adaptive_coin(
                mean_obi,
                self.alpha_obi,
                tick_direction=net_direction,
                alpha_direction=self.alpha_direction,
            )
        )

    @staticmethod
    def _validate_prices(tick_df: pd.DataFrame, window: int) -> np.ndarray:
        if "price" not in tick_df.columns:
            raise ValueError("tick_df must contain a price column")
        if int(window) < 2:
            raise ValueError("window must be >= 2")
        prices = tick_df["price"].to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError("prices must be finite and positive")
        return prices

    def rolling_vol_forecast(
        self,
        tick_df: pd.DataFrame,
        window: int = 15,
    ) -> pd.Series:
        """Return one strictly causal QRW forecast for each row after warm-up."""
        if self.qrw is None:
            raise RuntimeError("qrw_model must be provided for rolling_vol_forecast()")
        prices = self._validate_prices(tick_df, window)
        if len(prices) <= window:
            return pd.Series(dtype=float, name="vol_qrw")

        if hasattr(self.qrw, "n_positions"):
            max_steps = (int(self.qrw.n_positions) - 1) // 2
            if window > max_steps:
                raise ValueError(
                    f"QRW lattice is too small for window={window}; "
                    f"maximum safe steps are {max_steps}"
                )

        values: list[float] = []
        for index in range(window, len(prices)):
            trailing_prices = prices[index - window:index]
            trailing_returns = np.diff(np.log(trailing_prices))
            trailing_std = float(np.std(trailing_returns, ddof=0))

            if hasattr(self.qrw, "reset"):
                self.qrw.reset()
            self._refresh_market_coin(tick_df, window, index)
            probability = self.qrw.run(window)
            lattice_variance = self.compute_distribution_variance(probability)
            shape_multiplier = np.sqrt(lattice_variance) / max(float(window), 1.0)
            per_tick_vol = shape_multiplier * trailing_std * np.sqrt(window)
            values.append(float(per_tick_vol * self.ANNUALIZE))

        return pd.Series(
            values,
            index=tick_df.index[window:],
            dtype=float,
            name="vol_qrw",
        )

    def latest_vol_forecast(
        self,
        tick_df: pd.DataFrame,
        window: int = 15,
    ) -> dict[str, float | int | object]:
        """Forecast the next tick directly from an in-memory rolling buffer."""
        if self.qrw is None:
            raise RuntimeError("qrw_model must be provided for latest_vol_forecast()")
        prices = self._validate_prices(tick_df, window)
        if len(prices) < window:
            raise ValueError(f"at least {window} ticks are required")
        if hasattr(self.qrw, "n_positions"):
            max_steps = (int(self.qrw.n_positions) - 1) // 2
            if window > max_steps:
                raise ValueError(
                    f"QRW lattice is too small for window={window}; "
                    f"maximum safe steps are {max_steps}"
                )

        trailing_prices = prices[-window:]
        trailing_returns = np.diff(np.log(trailing_prices))
        realized = float(np.std(trailing_returns, ddof=0) * self.ANNUALIZE)
        if hasattr(self.qrw, "reset"):
            self.qrw.reset()
        self._refresh_market_coin(tick_df, window, len(prices))
        probability = self.qrw.run(window)
        lattice_variance = self.compute_distribution_variance(probability)
        shape_multiplier = np.sqrt(lattice_variance) / max(float(window), 1.0)
        per_tick_vol = shape_multiplier * float(np.std(trailing_returns, ddof=0)) * np.sqrt(window)
        forecast = float(per_tick_vol * self.ANNUALIZE)
        timestamp = tick_df.iloc[-1].get("timestamp", tick_df.index[-1])
        return {
            "timestamp": timestamp,
            "vol_qrw": forecast,
            "vol_realized": realized,
            "regime": self.vol_regime(forecast),
            "window": int(window),
            "observations": int(len(tick_df)),
        }

    def _realized_trailing_vol(
        self,
        tick_df: pd.DataFrame,
        window: int,
    ) -> pd.Series:
        prices = self._validate_prices(tick_df, window)
        output = pd.Series(np.nan, index=tick_df.index, dtype=float, name="vol_realized")
        for index in range(window, len(prices)):
            trailing = np.diff(np.log(prices[index - window:index]))
            output.iloc[index] = float(np.std(trailing, ddof=0) * self.ANNUALIZE)
        return output

    def _fixed_origin_garch_vol(
        self,
        tick_df: pd.DataFrame,
        window: int,
    ) -> pd.Series:
        """Fit once on an initial prefix, then forecast the remaining rows."""
        from src.models.garch_model import GARCHBaseline

        prices = self._validate_prices(tick_df, window)
        output = pd.Series(np.nan, index=tick_df.index, dtype=float, name="vol_garch")
        if len(prices) < 25:
            return output

        log_returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
        fit_end = min(max(25, window + 5), len(prices) - 1)
        training = log_returns[1:fit_end]
        if len(training) < 20 or float(np.std(training)) <= 1e-14:
            return output

        model = GARCHBaseline()
        parameters = model.fit(training)
        mu = float(parameters["mu"])
        omega = float(parameters["omega"])
        alpha = float(parameters["alpha"])
        beta = float(parameters["beta"])
        variance = max(float(np.var(training)), omega / max(1 - alpha - beta, 1e-8))
        previous_residual = float(training[-1] - mu)

        MIN_ANNUALIZED_VOL = 0.001  # 0.1% minimum
        for index in range(fit_end, len(prices)):
            variance = omega + alpha * previous_residual**2 + beta * variance
            if variance < 1e-10:
                import warnings
                warnings.warn(f"GARCH degenerate: variance={variance:.2e}, using realized vol as fallback")
                realized = self._realized_trailing_vol(tick_df, window)
                output.iloc[index:] = realized.iloc[index:]
                break
            
            output.iloc[index] = max(
                float(np.sqrt(variance) * self.ANNUALIZE),
                MIN_ANNUALIZED_VOL
            )
            previous_residual = float(log_returns[index] - mu)
        return output

    def compare_with_garch(
        self,
        tick_df: pd.DataFrame,
        window: int = 15,
    ) -> pd.DataFrame:
        """Return aligned QRW, fixed-origin GARCH, and realized volatility."""
        comparison = pd.concat(
            [
                self.rolling_vol_forecast(tick_df, window=window),
                self._fixed_origin_garch_vol(tick_df, window=window),
                self._realized_trailing_vol(tick_df, window=window),
            ],
            axis=1,
        )
        return comparison.dropna().sort_index()

    def vol_regime(
        self,
        vol_value: float,
        low_thresh: float = 0.10,
        high_thresh: float = 0.25,
    ) -> str:
        if not np.isfinite(vol_value) or low_thresh < 0 or high_thresh <= low_thresh:
            raise ValueError("volatility and thresholds must be finite and ordered")
        if vol_value < low_thresh:
            return "LOW"
        if vol_value < high_thresh:
            return "MID"
        return "HIGH"

    @staticmethod
    def benchmark_metrics(comparison_df: pd.DataFrame) -> dict[str, float | bool]:
        """Compute deterministic forecast errors and directional accuracy."""
        required = {"vol_qrw", "vol_garch", "vol_realized"}
        missing = required.difference(comparison_df.columns)
        if missing:
            raise ValueError(f"comparison_df is missing columns: {sorted(missing)}")
        frame = comparison_df[list(required)].replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 2:
            raise ValueError("at least two aligned observations are required")

        realized = frame["vol_realized"].to_numpy(dtype=float)
        qrw = frame["vol_qrw"].to_numpy(dtype=float)
        garch = frame["vol_garch"].to_numpy(dtype=float)
        mae_qrw = float(np.mean(np.abs(realized - qrw)))
        mae_garch = float(np.mean(np.abs(realized - garch)))
        rmse_qrw = float(np.sqrt(np.mean((realized - qrw) ** 2)))
        rmse_garch = float(np.sqrt(np.mean((realized - garch) ** 2)))
        realized_direction = np.sign(np.diff(realized))
        
        qrw_changes = np.diff(qrw)
        if np.std(qrw_changes) < 1e-8:
            import warnings
            warnings.warn("QRW vol series almost constant — direction accuracy meaningless")
            direction_qrw = np.nan
        else:
            direction_qrw = float(np.mean(np.sign(qrw_changes) == realized_direction))
            
        garch_changes = np.diff(garch)
        if np.std(garch_changes) < 1e-8:
            direction_garch = np.nan
        else:
            direction_garch = float(np.mean(np.sign(garch_changes) == realized_direction))
        return {
            "mae_qrw": mae_qrw,
            "mae_garch": mae_garch,
            "rmse_qrw": rmse_qrw,
            "rmse_garch": rmse_garch,
            "direction_qrw": direction_qrw,
            "direction_garch": direction_garch,
            "qrw_wins_mae": mae_qrw < mae_garch,
            "qrw_wins_rmse": rmse_qrw < rmse_garch,
        }

    def benchmark_metrics_by_window(
        self,
        tick_df: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> dict[str, dict[str, float | bool]]:
        results: dict[str, dict[str, float | bool]] = {}
        for window in windows or self.windows:
            comparison = self.compare_with_garch(tick_df, window=int(window))
            results[f"window_{int(window)}"] = self.benchmark_metrics(comparison)
        return results

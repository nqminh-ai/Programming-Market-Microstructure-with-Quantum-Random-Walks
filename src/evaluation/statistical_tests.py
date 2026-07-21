"""Statistical validation for empirical and simulated market paths."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class StatisticalTestSuite:
    """Run Phase 5 tests on one empirical holdout and common simulations."""

    PRIMARY_HORIZONS = (1, 5, 10, 20, 50, 100, 200, 500)
    SCALING_HORIZONS = (1, 5, 10, 20, 50, 100, 200, 500)
    BOOTSTRAP_SCORECARD_METRICS = (
        "mean_crps",
        "mean_absolute_error",
        "mean_direction_brier_score",
        "mean_direction_log_loss",
        "coverage_90",
        "coverage_90_absolute_error",
        "mean_interval_90_width",
    )
    BOOTSTRAP_SELECTION_RULE = (
        "primary=mean_crps; tiebreak=mean_direction_log_loss; "
        "remaining_metrics=diagnostic_only"
    )
    # Numerical floor/ceiling to keep probabilities and variances away from
    # 0 (avoids log(0)/division-by-zero without biasing scores in practice).
    _PROBABILITY_EPSILON = 1e-12
    # 90% central interval used for both coverage checks and reported CIs.
    _INTERVAL_90_QUANTILES = (0.05, 0.95)

    def __init__(
        self,
        empirical_prices: np.ndarray,
        forecast_samples: Mapping[str, np.ndarray],
        *,
        sample_semantics: Mapping[str, str] | None = None,
        rolling_one_step_losses: Mapping[str, np.ndarray] | None = None,
        rolling_one_step_timestamps: np.ndarray | None = None,
        random_seed: int = 2026,
        bootstrap_iterations: int = 1_000,
        max_lag: int = 20,
    ) -> None:
        prices = np.asarray(empirical_prices, dtype=np.float64).reshape(-1)
        if len(prices) < 30:
            raise ValueError("empirical_prices must contain at least 30 values")
        if not np.isfinite(prices).all() or np.any(prices <= 0.0):
            raise ValueError("empirical_prices must be finite and positive")
        if not forecast_samples:
            raise ValueError("forecast_samples cannot be empty")
        if bootstrap_iterations < 50:
            raise ValueError("bootstrap_iterations must be at least 50")
        if max_lag < 1:
            raise ValueError("max_lag must be positive")

        paths: dict[str, np.ndarray] = {}
        step_counts: set[int] = set()
        for model, values in forecast_samples.items():
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 2 or array.shape[0] < 20 or array.shape[1] < 3:
                raise ValueError(
                    f"{model} paths must have shape (n_paths, n_steps + 1)"
                )
            if not np.isfinite(array).all() or np.any(array <= 0.0):
                raise ValueError(f"{model} paths must be finite and positive")
            paths[str(model)] = array
            step_counts.add(array.shape[1] - 1)
        if len(step_counts) != 1:
            raise ValueError("all simulated models must use the same steps")

        self.empirical_prices = prices
        self.forecast_samples = paths
        self.simulated_paths = self.forecast_samples
        supplied_semantics = sample_semantics or {
            model: "path_trajectories" for model in paths
        }
        if set(supplied_semantics) != set(paths):
            raise ValueError("sample_semantics must identify every forecast model")
        allowed_semantics = {"fixed_origin_marginals", "path_trajectories"}
        invalid_semantics = set(supplied_semantics.values()) - allowed_semantics
        if invalid_semantics:
            raise ValueError(
                f"unsupported sample semantics: {sorted(invalid_semantics)}"
            )
        self.sample_semantics = dict(supplied_semantics)
        self.n_steps = step_counts.pop()
        if len(prices) < self.n_steps + 1:
            raise ValueError(
                "empirical_prices must cover the simulated path horizon"
            )
        self.comparison_prices = prices[: self.n_steps + 1]
        self.random_seed = int(random_seed)
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.max_lag = min(int(max_lag), self.n_steps - 2)
        self.rolling_one_step_losses = self._validated_rolling_losses(
            rolling_one_step_losses
        )
        self.rolling_one_step_timestamps = (
            self._validated_rolling_timestamps(rolling_one_step_timestamps)
        )
        self.acf_profiles: dict[str, np.ndarray] = {}
        self.pacf_profiles: dict[str, np.ndarray] = {}
        self.scaling_points: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _validated_rolling_losses(
        self,
        values: Mapping[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray] | None:
        if values is None:
            return None
        if set(values) != set(self.forecast_samples):
            raise ValueError(
                "rolling_one_step_losses must identify every forecast model"
            )
        result: dict[str, np.ndarray] = {}
        lengths: set[int] = set()
        for model, loss in values.items():
            array = np.asarray(loss, dtype=np.float64).reshape(-1)
            if len(array) < 20 or not np.isfinite(array).all():
                raise ValueError(
                    f"{model} rolling one-step losses must contain at least "
                    "20 finite observations"
                )
            if np.any(array < 0.0):
                raise ValueError("rolling one-step losses cannot be negative")
            result[str(model)] = array
            lengths.add(len(array))
        if len(lengths) != 1:
            raise ValueError("rolling one-step losses must be time aligned")
        return result

    def _validated_rolling_timestamps(
        self,
        values: np.ndarray | None,
    ) -> np.ndarray | None:
        if self.rolling_one_step_losses is None:
            if values is not None:
                raise ValueError(
                    "rolling timestamps require rolling one-step losses"
                )
            return None
        if values is None:
            raise ValueError(
                "rolling_one_step_timestamps are required for DM alignment"
            )
        timestamps = np.asarray(values).reshape(-1)
        expected = len(next(iter(self.rolling_one_step_losses.values())))
        if len(timestamps) != expected:
            raise ValueError(
                "rolling timestamps must match every one-step loss sequence"
            )
        index = pd.Index(timestamps)
        if index.hasnans or index.has_duplicates:
            raise ValueError("rolling timestamps must be finite and unique")
        if not index.is_monotonic_increasing:
            raise ValueError("rolling timestamps must be strictly increasing")
        return timestamps.copy()

    @staticmethod
    def _log_returns(prices: np.ndarray) -> np.ndarray:
        values = np.diff(np.log(np.asarray(prices, dtype=np.float64)))
        return values[np.isfinite(values)]

    @staticmethod
    def _write_csv(output: str | Path | None, frame: pd.DataFrame) -> None:
        if output is None:
            return
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)

    @staticmethod
    def _benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
        pvalues = np.asarray(list(values), dtype=np.float64)
        adjusted = np.full_like(pvalues, np.nan)
        valid = np.isfinite(pvalues)
        if not np.any(valid):
            return adjusted
        selected = pvalues[valid]
        order = np.argsort(selected)
        ranked = selected[order]
        count = len(ranked)
        corrected = ranked * count / np.arange(1, count + 1)
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        restored = np.empty(count, dtype=np.float64)
        restored[order] = np.clip(corrected, 0.0, 1.0)
        adjusted[valid] = restored
        return adjusted

    def _horizon_returns(
        self,
        prices: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        starts = self._horizon_starts(horizon)
        return np.log(prices[starts + horizon] / prices[starts])

    def _horizon_starts(self, horizon: int) -> np.ndarray:
        starts = np.arange(0, self.n_steps - horizon + 1, horizon)
        available = self.n_steps - horizon + 1
        target = min(5, available)
        if len(starts) < target:
            starts = np.unique(
                np.linspace(0, available - 1, target, dtype=int)
            )
        return starts

    @staticmethod
    def _sample_crps(samples: np.ndarray, observation: float) -> float:
        values = np.sort(np.asarray(samples, dtype=np.float64))
        count = len(values)
        weights = 2.0 * np.arange(1, count + 1) - count - 1.0
        return float(
            np.mean(np.abs(values - observation))
            - (weights @ values) / (count * count)
        )

    @staticmethod
    def _fixed_origin_returns(samples: np.ndarray, horizon: int) -> np.ndarray:
        return np.log(samples[:, horizon] / samples[:, 0])

    def marginal_score_tests(
        self,
        *,
        output: str | Path | None = None,
        horizons: Iterable[int] = PRIMARY_HORIZONS,
    ) -> pd.DataFrame:
        """Score fixed-origin marginal forecasts with proper scoring rules."""
        rows: list[dict[str, float | int | str]] = []
        valid_horizons = [
            int(value)
            for value in horizons
            if 1 <= int(value) <= self.n_steps
        ]
        for model, samples in self.forecast_samples.items():
            for horizon in valid_horizons:
                observation = float(
                    np.log(
                        self.comparison_prices[horizon]
                        / self.comparison_prices[0]
                    )
                )
                simulated = self._fixed_origin_returns(samples, horizon)
                probability_up = float(np.mean(simulated > 0.0))
                target_up = float(observation > 0.0)
                probability = float(
                    np.clip(
                        probability_up,
                        self._PROBABILITY_EPSILON,
                        1.0 - self._PROBABILITY_EPSILON,
                    )
                )
                lower, upper = np.quantile(simulated, self._INTERVAL_90_QUANTILES)
                rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "sample_size": len(simulated),
                        "sample_semantics": self.sample_semantics[model],
                        "observation_log_return": observation,
                        "crps": self._sample_crps(simulated, observation),
                        "mean_absolute_error": float(
                            np.mean(np.abs(simulated - observation))
                        ),
                        "probability_up": probability_up,
                        "direction_target_up": int(target_up),
                        "direction_brier_score": (
                            probability_up - target_up
                        ) ** 2,
                        "direction_log_loss": float(
                            -np.log(probability)
                            if target_up
                            else -np.log1p(-probability)
                        ),
                        "interval_90_low": float(lower),
                        "interval_90_high": float(upper),
                        "interval_90_width": float(upper - lower),
                        "interval_90_covered": int(
                            lower <= observation <= upper
                        ),
                    }
                )

        frame = pd.DataFrame(rows)
        self._write_csv(output, frame)
        return frame

    def distribution_tests(
        self,
        *,
        output: str | Path | None = None,
        horizons: Iterable[int] = PRIMARY_HORIZONS,
    ) -> pd.DataFrame:
        """Compatibility wrapper for the marginal-score protocol."""
        warnings.warn(
            "distribution_tests now returns fixed-origin marginal scores",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.marginal_score_tests(output=output, horizons=horizons)

    @staticmethod
    def _fit_log_variance(
        horizons: np.ndarray,
        variances: np.ndarray,
    ) -> dict[str, float]:
        x = np.log(horizons.astype(np.float64))
        y = np.log(variances.astype(np.float64))
        design = np.column_stack([np.ones(len(x)), x])
        intercept, beta = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = intercept + beta * x
        residual = y - fitted
        residual_sum = float(np.sum(residual**2))
        total_sum = float(np.sum((y - np.mean(y)) ** 2))
        degrees = len(x) - 2
        if degrees > 0:
            sigma2 = residual_sum / degrees
            covariance = sigma2 * np.linalg.inv(design.T @ design)
            beta_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
        else:
            beta_se = float("nan")
        if np.isfinite(beta_se) and beta_se > 0.0:
            statistic = (float(beta) - 1.0) / beta_se
            pvalue = float(stats.t.sf(statistic, df=degrees))
        else:
            statistic = float("nan")
            pvalue = float("nan")
        return {
            "intercept": float(intercept),
            "beta": float(beta),
            "beta_se": beta_se,
            "t_statistic_beta_gt_1": statistic,
            "pvalue_beta_gt_1": pvalue,
            "r_squared": (
                1.0 - residual_sum / total_sum
                if total_sum > 0.0
                else float("nan")
            ),
        }

    def _variance_samples(
        self,
        prices_or_paths: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        values = np.asarray(prices_or_paths, dtype=np.float64)
        if values.ndim == 1:
            starts = np.arange(0, len(values) - horizon, horizon)
            return np.log(values[starts + horizon] / values[starts])
        return self._fixed_origin_returns(values, horizon)

    def _bootstrap_betas(
        self,
        values: np.ndarray,
        horizons: np.ndarray,
        rng: np.random.Generator,
        *,
        sample_semantics: str,
    ) -> np.ndarray:
        dataset = np.asarray(values, dtype=np.float64)
        bootstrap_values = np.empty(
            self.bootstrap_iterations,
            dtype=np.float64,
        )
        if dataset.ndim == 1:
            one_step_returns = np.diff(np.log(dataset))
            block_size = min(
                max(5, int(np.sqrt(len(one_step_returns)))),
                len(one_step_returns),
            )
            block_count = int(
                np.ceil(len(one_step_returns) / block_size)
            )
        else:
            sampled_path_count = min(dataset.shape[0], 500)

        for iteration in range(self.bootstrap_iterations):
            if dataset.ndim == 1:
                starts = rng.integers(
                    0,
                    len(one_step_returns) - block_size + 1,
                    size=block_count,
                )
                sampled_returns = np.concatenate(
                    [
                        one_step_returns[start : start + block_size]
                        for start in starts
                    ]
                )[: len(one_step_returns)]
                sampled_values = np.exp(
                    np.concatenate(
                        [[0.0], np.cumsum(sampled_returns)]
                    )
                )
            elif sample_semantics == "path_trajectories":
                selected_paths = rng.integers(
                    0,
                    dataset.shape[0],
                    size=sampled_path_count,
                )
                sampled_values = dataset[selected_paths]
            else:
                sampled_values = dataset

            if dataset.ndim == 2 and sample_semantics == "fixed_origin_marginals":
                variances = np.asarray(
                    [
                        np.var(
                            self._fixed_origin_returns(
                                sampled_values[
                                    rng.integers(
                                        0,
                                        sampled_values.shape[0],
                                        size=sampled_path_count,
                                    )
                                ],
                                int(horizon),
                            ),
                            ddof=1,
                        )
                        for horizon in horizons
                    ],
                    dtype=np.float64,
                )
            else:
                variances = np.asarray(
                    [
                        np.var(
                            self._variance_samples(
                                sampled_values, int(horizon)
                            ),
                            ddof=1,
                        )
                        for horizon in horizons
                    ],
                    dtype=np.float64,
                )
            valid = np.isfinite(variances) & (variances > 0.0)
            if np.count_nonzero(valid) < 3:
                bootstrap_values[iteration] = np.nan
                continue
            bootstrap_values[iteration] = self._fit_log_variance(
                horizons[valid],
                variances[valid],
            )["beta"]
        return bootstrap_values[np.isfinite(bootstrap_values)]

    def variance_scaling_tests(
        self,
        *,
        output: str | Path | None = None,
        figure_output: str | Path | None = None,
        horizons: Iterable[int] = SCALING_HORIZONS,
    ) -> pd.DataFrame:
        """Estimate log-variance scaling and a bootstrap confidence interval."""
        datasets: dict[str, np.ndarray] = {
            "Empirical": self.comparison_prices,
            **self.forecast_samples,
        }
        seed_sequence = np.random.SeedSequence(self.random_seed + 1)
        children = iter(seed_sequence.spawn(len(datasets)))
        rows: list[dict[str, float | int | str]] = []
        requested = sorted({int(value) for value in horizons if int(value) > 0})

        for model, values in datasets.items():
            max_horizon = (
                len(values) - 2
                if values.ndim == 1
                else values.shape[1] - 1
            )
            selected_horizons: list[int] = []
            variances: list[float] = []
            for horizon in requested:
                if horizon > max_horizon:
                    continue
                sample = self._variance_samples(values, horizon)
                sample = sample[np.isfinite(sample)]
                if len(sample) < 3:
                    continue
                variance = float(np.var(sample, ddof=1))
                if variance <= 0.0:
                    continue
                selected_horizons.append(horizon)
                variances.append(variance)
            if len(selected_horizons) < 3:
                raise ValueError(
                    f"{model} has fewer than three usable scaling horizons"
                )

            horizon_array = np.asarray(selected_horizons, dtype=np.float64)
            variance_array = np.asarray(variances, dtype=np.float64)
            fit = self._fit_log_variance(horizon_array, variance_array)
            bootstrap = self._bootstrap_betas(
                values,
                horizon_array,
                np.random.default_rng(next(children)),
                sample_semantics=(
                    "empirical_time_series"
                    if model == "Empirical"
                    else self.sample_semantics[model]
                ),
            )
            if len(bootstrap) < max(30, self.bootstrap_iterations // 2):
                raise RuntimeError(f"bootstrap failed for {model}")
            self.scaling_points[model] = (horizon_array, variance_array)
            rows.append(
                {
                    "model": model,
                    **fit,
                    "beta_ci_low": float(np.quantile(bootstrap, 0.025)),
                    "beta_ci_high": float(np.quantile(bootstrap, 0.975)),
                    "bootstrap_probability_beta_le_1": float(
                        np.mean(bootstrap <= 1.0)
                    ),
                    "bootstrap_iterations": len(bootstrap),
                    "bootstrap_method": (
                        "moving_block_returns"
                        if values.ndim == 1
                        else (
                            "path_resampling"
                            if self.sample_semantics[model]
                            == "path_trajectories"
                            else "independent_marginal_resampling"
                        )
                    ),
                    "horizons_used": len(horizon_array),
                    "min_horizon": int(horizon_array.min()),
                    "max_horizon": int(horizon_array.max()),
                }
            )

        frame = pd.DataFrame(rows)
        frame["pvalue_beta_gt_1_bh"] = self._benjamini_hochberg(
            frame["pvalue_beta_gt_1"]
        )
        self._write_csv(output, frame)
        if figure_output is not None:
            self._plot_variance_scaling(Path(figure_output), frame)
        return frame

    def _plot_variance_scaling(
        self,
        output: Path,
        results: pd.DataFrame,
    ) -> None:
        figure, axis = plt.subplots(figsize=(9, 6))
        for _, row in results.iterrows():
            model = str(row["model"])
            horizons, variances = self.scaling_points[model]
            axis.plot(
                np.log(horizons),
                np.log(variances),
                marker="o",
                linewidth=1.2,
                label=f"{model} (beta={row['beta']:.2f})",
            )
        axis.set_xlabel("log(horizon)")
        axis.set_ylabel("log(variance of log-price displacement)")
        axis.set_title("Variance Scaling")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)

    @staticmethod
    def _acf(values: np.ndarray, max_lag: int) -> np.ndarray:
        series = np.asarray(values, dtype=np.float64).reshape(-1)
        series = series[np.isfinite(series)]
        centered = series - np.mean(series)
        denominator = float(np.dot(centered, centered))
        result = np.ones(max_lag + 1, dtype=np.float64)
        if denominator <= 0.0:
            result[1:] = 0.0
            return result
        for lag in range(1, max_lag + 1):
            result[lag] = float(
                np.dot(centered[:-lag], centered[lag:]) / denominator
            )
        return result

    @staticmethod
    def _ensemble_acf(returns: np.ndarray, max_lag: int) -> np.ndarray:
        centered = returns - np.mean(returns, axis=1, keepdims=True)
        denominator = float(np.sum(centered**2))
        result = np.ones(max_lag + 1, dtype=np.float64)
        if denominator <= 0.0:
            result[1:] = 0.0
            return result
        for lag in range(1, max_lag + 1):
            result[lag] = float(
                np.sum(centered[:, :-lag] * centered[:, lag:])
                / denominator
            )
        return result

    @staticmethod
    def _pacf_from_acf(acf: np.ndarray) -> np.ndarray:
        max_lag = len(acf) - 1
        pacf = np.ones(max_lag + 1, dtype=np.float64)
        for lag in range(1, max_lag + 1):
            matrix = np.fromfunction(
                lambda i, j: acf[np.abs(i - j).astype(int)],
                (lag, lag),
                dtype=int,
            )
            try:
                coefficients = np.linalg.solve(matrix, acf[1 : lag + 1])
                pacf[lag] = float(coefficients[-1])
            except np.linalg.LinAlgError:
                logger.warning(
                    "PACF: Yule-Walker system singular at lag {}, setting NaN", lag
                )
                pacf[lag] = float("nan")
        return pacf

    @staticmethod
    def _ljung_box_pvalue(
        acf: np.ndarray,
        sample_size: int,
        lag: int,
    ) -> tuple[float, float]:
        used = np.arange(1, lag + 1)
        statistic = sample_size * (sample_size + 2.0) * np.sum(
            acf[used] ** 2 / (sample_size - used)
        )
        return float(statistic), float(stats.chi2.sf(statistic, lag))

    def autocorrelation_tests(
        self,
        *,
        output: str | Path | None = None,
        figure_output: str | Path | None = None,
    ) -> pd.DataFrame:
        """Compare ACF profiles and run Ljung-Box diagnostics."""
        empirical = self._log_returns(self.comparison_prices)
        empirical_acf = self._acf(empirical, self.max_lag)
        self.acf_profiles = {"Empirical": empirical_acf}
        self.pacf_profiles = {
            "Empirical": self._pacf_from_acf(empirical_acf)
        }
        rows: list[dict[str, float | int | str]] = []
        datasets: list[tuple[str, np.ndarray, int]] = [
            ("Empirical", empirical_acf, len(empirical))
        ]

        for model, paths in self.forecast_samples.items():
            if self.sample_semantics[model] != "path_trajectories":
                continue
            returns = np.diff(np.log(paths), axis=1)
            profile = self._ensemble_acf(returns, self.max_lag)
            self.acf_profiles[model] = profile
            self.pacf_profiles[model] = self._pacf_from_acf(profile)
            # _ensemble_acf pools products/energy across all n_paths
            # trajectories, so the Ljung-Box sample size must reflect that
            # pooled observation count, not just the per-path step count.
            effective_sample_size = int(paths.shape[0] * self.n_steps)
            datasets.append((model, profile, effective_sample_size))

        for model, profile, sample_size in datasets:
            row: dict[str, float | int | str] = {
                "model": model,
                "evaluation_scope": "trajectory_only",
                "sample_size": sample_size,
                "acf_mse": (
                    0.0
                    if model == "Empirical"
                    else float(
                        np.mean(
                            (
                                profile[1 : self.max_lag + 1]
                                - empirical_acf[1 : self.max_lag + 1]
                            )
                            ** 2
                        )
                    )
                ),
            }
            for lag in (1, 5, 10):
                if lag <= self.max_lag:
                    statistic, pvalue = self._ljung_box_pvalue(
                        profile,
                        sample_size,
                        lag,
                    )
                else:
                    statistic, pvalue = float("nan"), float("nan")
                row[f"ljung_box_stat_lag_{lag}"] = statistic
                row[f"ljung_box_pvalue_lag_{lag}"] = pvalue
            rows.append(row)

        frame = pd.DataFrame(rows)
        for lag in (1, 5, 10):
            column = f"ljung_box_pvalue_lag_{lag}"
            frame[f"{column}_bh"] = self._benjamini_hochberg(frame[column])
        self._write_csv(output, frame)
        if figure_output is not None:
            self._plot_acf(Path(figure_output))
        return frame

    def _plot_acf(self, output: Path) -> None:
        figure, axis = plt.subplots(figsize=(10, 6))
        lags = np.arange(1, self.max_lag + 1)
        for model, profile in self.acf_profiles.items():
            axis.plot(lags, profile[1:], marker="o", markersize=3, label=model)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("Lag")
        axis.set_ylabel("Autocorrelation")
        axis.set_title("Return ACF Comparison")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)

    @staticmethod
    def _hill_tail_index(values: np.ndarray) -> tuple[float, int]:
        magnitudes = np.abs(np.asarray(values, dtype=np.float64))
        magnitudes = np.sort(magnitudes[np.isfinite(magnitudes)])
        magnitudes = magnitudes[magnitudes > 0.0][::-1]
        if len(magnitudes) < 10:
            return float("nan"), 0
        count = min(max(int(np.floor(0.1 * len(magnitudes))), 5), len(magnitudes) - 1)
        threshold = magnitudes[count]
        gamma = float(np.mean(np.log(magnitudes[:count] / threshold)))
        if gamma <= 0.0:
            return float("inf"), count
        return 1.0 / gamma, count

    @staticmethod
    def _tail_metrics(values: np.ndarray) -> dict[str, float | int]:
        returns = np.asarray(values, dtype=np.float64)
        returns = returns[np.isfinite(returns)]
        tail_index, tail_count = StatisticalTestSuite._hill_tail_index(
            returns
        )
        if len(returns) >= 20 and np.std(returns) > 0.0:
            kurtosis_result = stats.kurtosistest(returns)
            kurtosis_statistic = float(kurtosis_result.statistic)
            kurtosis_pvalue = float(kurtosis_result.pvalue)
        else:
            kurtosis_statistic = float("nan")
            kurtosis_pvalue = float("nan")
        losses = -returns
        result: dict[str, float | int] = {
            "sample_size": len(returns),
            "kurtosis": float(
                stats.kurtosis(returns, fisher=False, bias=False)
            ),
            "kurtosis_test_statistic": kurtosis_statistic,
            "kurtosis_test_pvalue": kurtosis_pvalue,
            "tail_index": tail_index,
            "tail_observations": tail_count,
        }
        for level in (0.95, 0.99):
            label = int(level * 100)
            value_at_risk = float(np.quantile(losses, level))
            exceedance = losses[losses >= value_at_risk]
            result[f"var_{label}"] = value_at_risk
            result[f"cvar_{label}"] = float(np.mean(exceedance))
        return result

    def tail_analysis(
        self,
        *,
        output: str | Path | None = None,
    ) -> pd.DataFrame:
        """Estimate kurtosis, Hill tail index, VaR, and expected shortfall."""
        rows: list[dict[str, float | int | str]] = [
            {
                "model": "Empirical",
                "evaluation_scope": "trajectory_only",
                **self._tail_metrics(
                    self._log_returns(self.comparison_prices)
                ),
            }
        ]
        trajectory_models = [
            model
            for model in self.forecast_samples
            if self.sample_semantics[model] == "path_trajectories"
        ]
        children = iter(
            np.random.SeedSequence(self.random_seed + 2).spawn(
                len(trajectory_models)
            )
        )
        for model in trajectory_models:
            paths = self.forecast_samples[model]
            rng = np.random.default_rng(next(children))
            starts = self._horizon_starts(1)
            path_index = rng.integers(0, paths.shape[0], size=len(starts))
            sample = np.log(
                paths[path_index, starts + 1] / paths[path_index, starts]
            )
            rows.append(
                {
                    "model": model,
                    "evaluation_scope": "trajectory_only",
                    **self._tail_metrics(sample),
                }
            )
        frame = pd.DataFrame(rows)
        frame["kurtosis_test_pvalue_bh"] = self._benjamini_hochberg(
            frame["kurtosis_test_pvalue"]
        )
        self._write_csv(output, frame)
        return frame

    def diebold_mariano_tests(
        self,
        *,
        output: str | Path | None = None,
    ) -> pd.DataFrame:
        """Compare aligned rolling-origin one-step loss sequences."""
        if self.rolling_one_step_losses is None:
            raise ValueError(
                "Diebold-Mariano requires aligned rolling_one_step_losses; "
                "fixed-origin marginal columns are not a time series"
            )
        models = list(self.rolling_one_step_losses)
        rows: list[dict[str, float | int | str | bool]] = []
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i >= j:
                    continue
                d = (
                    self.rolling_one_step_losses[model1]
                    - self.rolling_one_step_losses[model2]
                )
                mean_d = np.mean(d)
                n = len(d)
                gamma0 = np.var(d, ddof=1)
                variance_d = gamma0
                centered = d - mean_d
                for lag in range(1, self.max_lag + 1):
                    # Stop one lag before len(d) - 1, not at it: the
                    # denominator below is len(d) - lag - 1, which hits
                    # exactly 0 (silent inf/nan, not an exception) when
                    # lag == len(d) - 1. That previously depended on an
                    # unenforced coupling between self.max_lag and the
                    # rolling-loss array length happening to avoid it.
                    if lag >= len(d) - 1:
                        break
                    # Autocovariance must center both arms on the shared
                    # sample mean_d, not np.cov's own per-argument means,
                    # or the HAC variance estimate is subtly biased.
                    gamma_lag = float(
                        np.sum(centered[:-lag] * centered[lag:])
                        / (len(d) - lag - 1)
                    )
                    weight = 1.0 - lag / (self.max_lag + 1)
                    variance_d += 2 * weight * gamma_lag
                variance_d = max(variance_d, self._PROBABILITY_EPSILON)
                stat = float(mean_d / np.sqrt(variance_d / n))
                pvalue = float(2.0 * stats.norm.sf(np.abs(stat)))
                rows.append(
                    {
                        "model1": model1,
                        "model2": model2,
                        "dm_statistic": stat,
                        "dm_pvalue": pvalue,
                        "mean_loss_differential": float(mean_d),
                        "observations": n,
                        "loss_type": "absolute_price_error",
                        "forecast_alignment": "rolling_origin_one_step",
                        "timestamp_alignment": (
                            "strictly_increasing_common_timestamp_index"
                        ),
                        "model1_better": stat < 0.0,
                    }
                )

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["dm_pvalue_bh"] = self._benjamini_hochberg(frame["dm_pvalue"])
        self._write_csv(output, frame)
        return frame

    def run_bootstrap_scorecard_ci(
        self,
        *,
        output: str | Path | None = None,
    ) -> pd.DataFrame:
        """Resample seven valid marginal metrics and rerank each replicate."""
        models = list(self.forecast_samples)
        horizons = np.asarray(
            [h for h in self.PRIMARY_HORIZONS if h <= self.n_steps],
            dtype=int,
        )
        if len(horizons) < 2:
            raise ValueError("at least two primary horizons are required")
        results: list[dict[str, float | int | str]] = []
        rng = np.random.default_rng(self.random_seed + 3)
        # `observation` depends only on horizon (fixed comparison prices),
        # not on the bootstrap iteration or model -- hoisting it out of the
        # iteration/model/horizon triple loop avoids recomputing the same
        # log-ratio tens of thousands of times without touching the RNG
        # draw sequence (and therefore without changing any bootstrap
        # result).
        observation_by_horizon = {
            int(horizon): float(
                np.log(self.comparison_prices[horizon] / self.comparison_prices[0])
            )
            for horizon in horizons
        }
        for iteration in range(self.bootstrap_iterations):
            selected_horizons = rng.choice(
                horizons, size=len(horizons), replace=True
            )
            iteration_rows: list[dict[str, float | int | str]] = []
            for model in models:
                samples = self.forecast_samples[model]
                crps_values: list[float] = []
                absolute_errors: list[float] = []
                brier_scores: list[float] = []
                log_losses: list[float] = []
                coverage: list[float] = []
                interval_widths: list[float] = []
                for horizon in selected_horizons:
                    indices = rng.integers(
                        0, samples.shape[0], size=samples.shape[0]
                    )
                    simulated = self._fixed_origin_returns(
                        samples[indices], int(horizon)
                    )
                    observation = observation_by_horizon[int(horizon)]
                    crps_values.append(
                        self._sample_crps(simulated, observation)
                    )
                    absolute_errors.append(
                        float(np.mean(np.abs(simulated - observation)))
                    )
                    probability_up = float(np.mean(simulated > 0.0))
                    target_up = float(observation > 0.0)
                    probability = float(
                        np.clip(
                            probability_up,
                            self._PROBABILITY_EPSILON,
                            1.0 - self._PROBABILITY_EPSILON,
                        )
                    )
                    brier_scores.append((probability_up - target_up) ** 2)
                    log_losses.append(
                        float(
                            -np.log(probability)
                            if target_up
                            else -np.log1p(-probability)
                        )
                    )
                    lower, upper = np.quantile(simulated, self._INTERVAL_90_QUANTILES)
                    coverage.append(float(lower <= observation <= upper))
                    interval_widths.append(float(upper - lower))

                mean_coverage = float(np.mean(coverage))
                iteration_rows.append(
                    {
                        "bootstrap_iteration": iteration,
                        "model": model,
                        "mean_crps": float(np.mean(crps_values)),
                        "mean_absolute_error": float(
                            np.mean(absolute_errors)
                        ),
                        "mean_direction_brier_score": float(
                            np.mean(brier_scores)
                        ),
                        "mean_direction_log_loss": float(
                            np.mean(log_losses)
                        ),
                        "coverage_90": mean_coverage,
                        "coverage_90_absolute_error": abs(
                            mean_coverage - 0.90
                        ),
                        "mean_interval_90_width": float(
                            np.mean(interval_widths)
                        ),
                    }
                )

            ranked = pd.DataFrame(iteration_rows).sort_values(
                ["mean_crps", "mean_direction_log_loss", "model"],
                kind="stable",
            )
            ranked["overall_rank"] = np.arange(1, len(ranked) + 1)
            results.extend(ranked.to_dict(orient="records"))

        frame = pd.DataFrame(results)
        summary_rows: list[dict[str, float | str | int]] = []
        for model, group in frame.groupby("model", sort=False):
            summary: dict[str, float | str | int] = {
                "model": model,
                "bootstrap_iterations": len(group),
                "resampled_metric_count": len(
                    self.BOOTSTRAP_SCORECARD_METRICS
                ),
                "mean_overall_rank": float(group["overall_rank"].mean()),
                "overall_rank_ci_low": float(
                    group["overall_rank"].quantile(0.025)
                ),
                "overall_rank_ci_high": float(
                    group["overall_rank"].quantile(0.975)
                ),
                "probability_rank_1": float(
                    np.mean(group["overall_rank"] == 1)
                ),
                "rank_selection_rule": self.BOOTSTRAP_SELECTION_RULE,
                "resampling_unit": "horizons_and_marginal_draws",
            }
            for metric in self.BOOTSTRAP_SCORECARD_METRICS:
                summary[metric] = float(group[metric].mean())
                summary[f"{metric}_ci_low"] = float(
                    group[metric].quantile(0.025)
                )
                summary[f"{metric}_ci_high"] = float(
                    group[metric].quantile(0.975)
                )
            summary_rows.append(summary)
        summary = pd.DataFrame(summary_rows)
        self._write_csv(output, summary)
        return summary

    def run_all(
        self,
        *,
        results_dir: str | Path = "results",
        figures_dir: str | Path = "figures",
    ) -> dict[str, pd.DataFrame]:
        """Run marginal tests plus explicitly trajectory-only diagnostics."""
        results_path = Path(results_dir)
        figures_path = Path(figures_dir)
        return {
            "marginal_scores": self.marginal_score_tests(
                output=results_path / "marginal_score_tests.csv",
            ),
            "variance_scaling": self.variance_scaling_tests(
                output=results_path / "variance_scaling_results.csv",
                figure_output=figures_path / "variance_scaling.png",
            ),
            "trajectory_autocorrelation": self.autocorrelation_tests(
                output=results_path / "trajectory_autocorrelation_tests.csv",
                figure_output=figures_path / "acf_comparison.png",
            ),
            "trajectory_tail": self.tail_analysis(
                output=results_path / "trajectory_tail_analysis.csv",
            ),
            "diebold_mariano": self.diebold_mariano_tests(
                output=results_path / "diebold_mariano_tests.csv",
            ),
            "scorecard_ci": self.run_bootstrap_scorecard_ci(
                output=results_path / "scorecard_bootstrap_ci.csv",
            ),
        }

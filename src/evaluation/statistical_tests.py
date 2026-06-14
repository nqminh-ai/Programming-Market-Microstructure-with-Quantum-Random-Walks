"""Statistical validation for empirical and simulated market paths."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class StatisticalTestSuite:
    """Run Phase 5 tests on one empirical holdout and common simulations."""

    DISTRIBUTION_HORIZONS = (1, 10, 50, 100)
    SCALING_HORIZONS = (1, 5, 10, 20, 50, 100, 200, 500)

    def __init__(
        self,
        empirical_prices: np.ndarray,
        simulated_paths: Mapping[str, np.ndarray],
        *,
        random_seed: int = 2026,
        bootstrap_iterations: int = 1_000,
        max_lag: int = 20,
    ) -> None:
        prices = np.asarray(empirical_prices, dtype=np.float64).reshape(-1)
        if len(prices) < 30:
            raise ValueError("empirical_prices must contain at least 30 values")
        if not np.isfinite(prices).all() or np.any(prices <= 0.0):
            raise ValueError("empirical_prices must be finite and positive")
        if not simulated_paths:
            raise ValueError("simulated_paths cannot be empty")
        if bootstrap_iterations < 50:
            raise ValueError("bootstrap_iterations must be at least 50")
        if max_lag < 1:
            raise ValueError("max_lag must be positive")

        paths: dict[str, np.ndarray] = {}
        step_counts: set[int] = set()
        for model, values in simulated_paths.items():
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
        self.simulated_paths = paths
        self.n_steps = step_counts.pop()
        if len(prices) < self.n_steps + 1:
            raise ValueError(
                "empirical_prices must cover the simulated path horizon"
            )
        self.comparison_prices = prices[: self.n_steps + 1]
        self.random_seed = int(random_seed)
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.max_lag = min(int(max_lag), self.n_steps - 2)
        self.acf_profiles: dict[str, np.ndarray] = {}
        self.pacf_profiles: dict[str, np.ndarray] = {}
        self.scaling_points: dict[str, tuple[np.ndarray, np.ndarray]] = {}

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

    def _simulated_horizon_returns(
        self,
        paths: np.ndarray,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        starts = self._horizon_starts(horizon)
        path_index = rng.integers(0, paths.shape[0], size=len(starts))
        return np.log(
            paths[path_index, starts + horizon]
            / paths[path_index, starts]
        )

    def distribution_tests(
        self,
        *,
        output: str | Path | None = None,
        horizons: Iterable[int] = DISTRIBUTION_HORIZONS,
    ) -> pd.DataFrame:
        """Compare empirical and simulated marginal return distributions."""
        rows: list[dict[str, float | int | str]] = []
        seed_sequence = np.random.SeedSequence(self.random_seed)
        children = iter(
            seed_sequence.spawn(len(self.simulated_paths) * 8)
        )
        valid_horizons = [
            int(value)
            for value in horizons
            if 1 <= int(value) <= self.n_steps
        ]
        for model, paths in self.simulated_paths.items():
            for horizon in valid_horizons:
                empirical = self._horizon_returns(
                    self.comparison_prices,
                    horizon,
                )
                rng = np.random.default_rng(next(children))
                simulated = self._simulated_horizon_returns(
                    paths,
                    horizon,
                    rng,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ad_result = stats.anderson_ksamp(
                        [empirical, simulated],
                    )
                if horizon == 1:
                    ks_result = stats.ks_2samp(
                        empirical,
                        simulated,
                        alternative="two-sided",
                        method="auto",
                    )
                    ks_statistic = float(ks_result.statistic)
                    ks_pvalue = float(ks_result.pvalue)
                else:
                    ks_statistic = float("nan")
                    ks_pvalue = float("nan")
                rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "sample_size_empirical": len(empirical),
                        "sample_size_simulated": len(simulated),
                        "ks_statistic": ks_statistic,
                        "ks_pvalue": ks_pvalue,
                        "anderson_statistic": float(ad_result.statistic),
                        "anderson_pvalue": float(ad_result.pvalue),
                        "wasserstein_distance": float(
                            stats.wasserstein_distance(
                                empirical,
                                simulated,
                            )
                        ),
                    }
                )

        frame = pd.DataFrame(rows)
        frame["ks_pvalue_bh"] = self._benjamini_hochberg(
            frame["ks_pvalue"]
        )
        frame["anderson_pvalue_bh"] = np.nan
        for _, indices in frame.groupby("horizon").groups.items():
            frame.loc[indices, "anderson_pvalue_bh"] = (
                self._benjamini_hochberg(
                    frame.loc[indices, "anderson_pvalue"]
                )
            )
        self._write_csv(output, frame)
        return frame

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
        starts = np.arange(0, values.shape[1] - horizon, horizon)
        return np.log(
            values[:, starts + horizon] / values[:, starts]
        ).reshape(-1)

    def _bootstrap_betas(
        self,
        values: np.ndarray,
        horizons: np.ndarray,
        rng: np.random.Generator,
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
            else:
                selected_paths = rng.integers(
                    0,
                    dataset.shape[0],
                    size=sampled_path_count,
                )
                sampled_values = dataset[selected_paths]

            variances = np.asarray(
                [
                    np.var(
                        self._variance_samples(sampled_values, int(horizon)),
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
            **self.simulated_paths,
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
                        else "path_resampling"
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

        for model, paths in self.simulated_paths.items():
            returns = np.diff(np.log(paths), axis=1)
            profile = self._ensemble_acf(returns, self.max_lag)
            self.acf_profiles[model] = profile
            self.pacf_profiles[model] = self._pacf_from_acf(profile)
            datasets.append((model, profile, self.n_steps))

        for model, profile, sample_size in datasets:
            row: dict[str, float | int | str] = {
                "model": model,
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
                **self._tail_metrics(
                    self._log_returns(self.comparison_prices)
                ),
            }
        ]
        children = iter(
            np.random.SeedSequence(self.random_seed + 2).spawn(
                len(self.simulated_paths)
            )
        )
        for model, paths in self.simulated_paths.items():
            sample = self._simulated_horizon_returns(
                paths,
                1,
                np.random.default_rng(next(children)),
            )
            rows.append(
                {
                    "model": model,
                    **self._tail_metrics(sample),
                }
            )
        frame = pd.DataFrame(rows)
        frame["kurtosis_test_pvalue_bh"] = self._benjamini_hochberg(
            frame["kurtosis_test_pvalue"]
        )
        self._write_csv(output, frame)
        return frame

    def run_all(
        self,
        *,
        results_dir: str | Path = "results",
        figures_dir: str | Path = "figures",
    ) -> dict[str, pd.DataFrame]:
        """Run all four Phase 5 test categories and persist their outputs."""
        results_path = Path(results_dir)
        figures_path = Path(figures_dir)
        return {
            "distribution": self.distribution_tests(
                output=results_path / "distribution_tests.csv",
            ),
            "variance_scaling": self.variance_scaling_tests(
                output=results_path / "variance_scaling_results.csv",
                figure_output=figures_path / "variance_scaling.png",
            ),
            "autocorrelation": self.autocorrelation_tests(
                output=results_path / "autocorrelation_tests.csv",
                figure_output=figures_path / "acf_comparison.png",
            ),
            "tail": self.tail_analysis(
                output=results_path / "tail_analysis.csv",
            ),
        }

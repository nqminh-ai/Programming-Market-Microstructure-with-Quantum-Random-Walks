"""Fair chronological benchmark for QRW and classical market baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from src.models.adaptive_market_qrw import AdaptiveDecoherenceQRW
from src.models.classical_rw import ClassicalRandomWalk
from src.models.garch_model import GARCHBaseline
from src.models.gbm_model import GBMBaseline
from src.models.qrw_core import QuantumRandomWalk
from src.models.qrw_heavy_tail import HeavyTailAdaptiveQRW


class BenchmarkSuite:
    """Fit all models on a common past window and score one later path."""

    PROTOCOL_VERSION = "fixed_origin_ex_ante_zero_inflated_v2"
    REQUIRED_COLUMNS = AdaptiveDecoherenceQRW.REQUIRED_COLUMNS
    METRICS = (
        "wasserstein_path_mae",
        "variance_ratio",
        "return_kurtosis",
        "hit_rate_h1",
        "hit_rate_h5",
        "hit_rate_h10",
        "mean_direction_log_likelihood",
    )

    def __init__(
        self,
        market_data: pd.DataFrame,
        *,
        train_fraction: float = 0.6,
        n_steps: int = 500,
        n_paths: int = 5_000,
        random_seed: int = 2026,
    ) -> None:
        missing = sorted(self.REQUIRED_COLUMNS.difference(market_data.columns))
        if missing:
            raise ValueError(f"market_data is missing columns: {missing}")
        if not 0.5 <= train_fraction <= 0.8:
            raise ValueError("train_fraction must be between 0.5 and 0.8")
        if n_steps < 10:
            raise ValueError("n_steps must be at least 10")
        if n_paths < 100:
            raise ValueError("n_paths must be at least 100")

        frame = market_data.sort_values(
            "timestamp",
            kind="stable",
        ).reset_index(drop=True).copy()
        numeric_columns = [
            "price",
            "tick_direction",
            "obi",
            "trade_intensity",
        ]
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("market data must contain finite numeric values")
        if (numeric["price"] <= 0.0).any():
            raise ValueError("prices must be positive")
        frame[numeric_columns] = numeric

        cut = int(len(frame) * train_fraction)
        available_steps = len(frame) - cut - 1
        if cut < 100 or available_steps < 10:
            raise ValueError("dataset is too short for the requested split")
        self.train_fraction = float(train_fraction)
        self.requested_n_steps = int(n_steps)
        self.n_steps = min(int(n_steps), available_steps)
        self.n_paths = int(n_paths)
        self.random_seed = int(random_seed)
        self.train = frame.iloc[:cut].copy().reset_index(drop=True)
        self.holdout = frame.iloc[cut:].copy().reset_index(drop=True)
        self.test = self.holdout.iloc[
            : self.n_steps + 1
        ].copy().reset_index(drop=True)
        self.initial_price = float(self.test["price"].iloc[0])
        self.tick_size = self._infer_tick_size(self.train["price"].to_numpy())
        train_directions = self._price_directions(self.train)
        if len(train_directions) == 0:
            raise ValueError("training data has no within-segment price changes")
        self.movement_probability = float(
            np.mean(train_directions != 0.0)
        )
        self.results: pd.DataFrame | None = None
        self.model_comparison: pd.DataFrame | None = None
        self.simulated_paths: dict[str, np.ndarray] = {}
        self.diagnostics: dict[str, Any] = {}

    @staticmethod
    def _infer_tick_size(price: np.ndarray) -> float:
        changes = np.abs(np.diff(np.asarray(price, dtype=np.float64)))
        changes = changes[changes > 1e-12]
        if len(changes) == 0:
            raise ValueError("cannot infer tick size from a constant price series")
        rounded = np.round(changes, decimals=10)
        return float(np.min(rounded[rounded > 0.0]))

    @staticmethod
    def _log_returns(frame: pd.DataFrame) -> np.ndarray:
        price = frame["price"].to_numpy(dtype=np.float64)
        values = np.diff(np.log(price))
        if "segment_id" in frame:
            segment = frame["segment_id"].to_numpy(copy=False)
            values = values[segment[:-1] == segment[1:]]
        return values[np.isfinite(values)]

    @staticmethod
    def _price_directions(frame: pd.DataFrame) -> np.ndarray:
        price = frame["price"].to_numpy(dtype=np.float64)
        direction = np.sign(np.diff(price))
        if "segment_id" in frame:
            segment = frame["segment_id"].to_numpy(copy=False)
            direction = direction[segment[:-1] == segment[1:]]
        return direction[np.isfinite(direction)]

    @staticmethod
    def _seed_values(seed: int, count: int) -> list[int]:
        sequence = np.random.SeedSequence(seed)
        return [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in sequence.spawn(count)
        ]

    def _fit_qrw(self) -> tuple[AdaptiveDecoherenceQRW, dict[str, Any]]:
        calibration = AdaptiveDecoherenceQRW(
            self.train,
            {"n_positions": 101},
        )
        parameters = calibration.calibrate_two_stage(None)
        return calibration, parameters

    def _simulate_qrw(
        self,
        model: AdaptiveDecoherenceQRW,
        *,
        seed: int,
    ) -> np.ndarray:
        probability_up = float(model.predict_probability()[-1])
        rng = np.random.default_rng(seed)
        moving = (
            rng.random((self.n_paths, self.n_steps))
            < self.movement_probability
        )
        direction = np.where(
            rng.random((self.n_paths, self.n_steps)) < probability_up,
            1.0,
            -1.0,
        )
        increments = np.where(moving, self.tick_size * direction, 0.0)
        paths = np.empty(
            (self.n_paths, self.n_steps + 1),
            dtype=np.float64,
        )
        paths[:, 0] = self.initial_price
        paths[:, 1:] = self.initial_price + np.cumsum(increments, axis=1)
        return paths

    def _fit_qrw_heavy_tail(
        self,
    ) -> tuple[HeavyTailAdaptiveQRW, dict[str, Any]]:
        model = HeavyTailAdaptiveQRW(self.train, {"n_positions": 101})
        parameters = model.calibrate_two_stage(output_path=None)
        return model, parameters

    def _training_jump_cap(self) -> float:
        """Largest absolute price move observed in training (within a segment).

        Sampled Pareto jumps are capped here so the simulation never produces a
        move larger than anything actually seen in the data, which also keeps
        prices finite and positive.
        """
        change = np.abs(np.diff(self.train["price"].to_numpy(dtype=np.float64)))
        valid = change > 1e-12
        if "segment_id" in self.train:
            segment = self.train["segment_id"].to_numpy(copy=False)
            valid &= segment[:-1] == segment[1:]
        observed = change[valid]
        if observed.size == 0:
            return self.tick_size
        return float(np.max(observed))

    def _simulate_qrw_heavy_tail(
        self,
        model: HeavyTailAdaptiveQRW,
        *,
        seed: int,
    ) -> np.ndarray:
        probability_up = float(model.predict_probability()[-1])
        jump_scale = max(float(model.jump_scale), 1e-12)
        tail_index = max(float(model.tail_index), 1e-3)
        jump_cap = self._training_jump_cap()

        rng = np.random.default_rng(seed)
        moving = (
            rng.random((self.n_paths, self.n_steps))
            < self.movement_probability
        )
        direction = np.where(
            rng.random((self.n_paths, self.n_steps)) < probability_up,
            1.0,
            -1.0,
        )
        # Inverse-transform sampling of a discrete Pareto jump magnitude.
        uniform = rng.random((self.n_paths, self.n_steps))
        jumps = jump_scale / np.power(1.0 - uniform, 1.0 / tail_index)
        jumps = np.round(jumps / jump_scale) * jump_scale
        jumps = np.minimum(jumps, jump_cap)
        increments = np.where(moving, jumps * direction, 0.0)
        paths = np.empty(
            (self.n_paths, self.n_steps + 1),
            dtype=np.float64,
        )
        paths[:, 0] = self.initial_price
        paths[:, 1:] = self.initial_price + np.cumsum(increments, axis=1)
        # Heavy-tailed down-moves can in principle exceed the initial price;
        # floor at one tick so prices stay strictly positive for downstream tests.
        np.maximum(paths, self.tick_size, out=paths)
        return paths

    def _score_paths(
        self,
        model_name: str,
        paths: np.ndarray,
    ) -> tuple[list[dict[str, float | str]], float]:
        expected_shape = (self.n_paths, self.n_steps + 1)
        if paths.shape != expected_shape:
            raise ValueError(
                f"{model_name} returned {paths.shape}, expected {expected_shape}"
            )
        if not np.isfinite(paths).all():
            raise ValueError(f"{model_name} produced non-finite paths")

        realized = self.test["price"].to_numpy(dtype=np.float64)
        path_errors = np.mean(
            np.abs(paths[:, 1:] - realized[None, 1:]),
            axis=0,
        )
        terminal_ticks = (
            paths[:, -1] - paths[:, 0]
        ) / self.tick_size
        terminal_centered_square = (
            terminal_ticks - terminal_ticks.mean()
        ) ** 2
        variance_ratio = float(
            np.mean(terminal_centered_square) / self.n_steps
        )
        variance_ratio_se = float(
            np.std(terminal_centered_square, ddof=1)
            / np.sqrt(self.n_paths)
            / self.n_steps
        )

        increments = np.diff(paths, axis=1)
        relative_returns = increments / np.maximum(paths[:, :-1], 1e-12)
        flattened_returns = relative_returns.reshape(-1)
        tail_kurtosis = float(
            kurtosis(
                flattened_returns,
                fisher=False,
                bias=False,
            )
        )
        kurtosis_se = float(np.sqrt(24.0 / len(flattened_returns)))
        rows: list[dict[str, float | str]] = [
            {
                "model": model_name,
                "metric": "wasserstein_path_mae",
                "value": float(np.mean(path_errors)),
                "std": float(np.std(path_errors, ddof=1)),
            },
            {
                "model": model_name,
                "metric": "variance_ratio",
                "value": variance_ratio,
                "std": variance_ratio_se,
            },
            {
                "model": model_name,
                "metric": "return_kurtosis",
                "value": tail_kurtosis,
                "std": kurtosis_se,
            },
        ]

        for horizon in (1, 5, 10):
            predicted_change = np.mean(
                paths[:, horizon:] - paths[:, :-horizon],
                axis=0,
            )
            realized_change = realized[horizon:] - realized[:-horizon]
            valid = np.abs(realized_change) > 1e-12
            if np.any(valid):
                hit = np.sign(predicted_change[valid]) == np.sign(
                    realized_change[valid]
                )
                value = float(np.mean(hit))
                standard_error = float(
                    np.sqrt(value * (1.0 - value) / len(hit))
                )
            else:
                value = float("nan")
                standard_error = float("nan")
            rows.append(
                {
                    "model": model_name,
                    "metric": f"hit_rate_h{horizon}",
                    "value": value,
                    "std": standard_error,
                }
            )

        simulated_moving = np.abs(increments) > 1e-12
        moving_count = simulated_moving.sum(axis=0)
        model_up_probability = np.divide(
            (increments > 0.0).sum(axis=0),
            moving_count,
            out=np.full(self.n_steps, 0.5, dtype=np.float64),
            where=moving_count > 0,
        )
        realized_change = np.diff(realized)
        valid = np.abs(realized_change) > 1e-12
        target = realized_change[valid] > 0.0
        probability = np.clip(
            model_up_probability[valid],
            1e-12,
            1.0 - 1e-12,
        )
        log_probability = np.where(
            target,
            np.log(probability),
            np.log1p(-probability),
        )
        rows.append(
            {
                "model": model_name,
                "metric": "mean_direction_log_likelihood",
                "value": float(np.mean(log_probability)),
                "std": float(
                    np.std(log_probability, ddof=1)
                    / np.sqrt(len(log_probability))
                ),
            }
        )
        return rows, float(np.std(flattened_returns, ddof=0))

    def _coherent_qrw_diagnostic(self, n_steps: int = 200) -> dict[str, Any]:
        steps = min(int(n_steps), self.n_steps)
        model = QuantumRandomWalk(2 * steps + 3)
        model.run(steps)
        qrw_ratio = float(model.variance() / steps)
        crw_ratio = 1.0
        return {
            "n_steps": steps,
            "qrw_variance_ratio": qrw_ratio,
            "crw_theoretical_variance_ratio": crw_ratio,
            "ratio_multiple": qrw_ratio / crw_ratio,
            "passed": bool(qrw_ratio > 1.3 * crw_ratio),
        }

    def run(
        self,
        *,
        benchmark_output: str | Path | None = None,
        comparison_output: str | Path | None = None,
        garch_output: str | Path | None = None,
        diagnostics_output: str | Path | None = None,
    ) -> pd.DataFrame:
        """Fit, simulate, score, and optionally persist every model."""
        train_returns = self._log_returns(self.train)
        train_directions = self._price_directions(self.train)
        moving_train_directions = train_directions[train_directions != 0.0]
        seeds = self._seed_values(self.random_seed, 6)
        metric_rows: list[dict[str, float | str]] = []
        comparison_rows: list[dict[str, float | int | str]] = []

        qrw, qrw_parameters = self._fit_qrw()
        qrw_paths = self._simulate_qrw(qrw, seed=seeds[0])
        self.simulated_paths["QRW Adaptive"] = qrw_paths
        rows, model_volatility = self._score_paths("QRW Adaptive", qrw_paths)
        metric_rows.extend(rows)
        train_change = np.diff(
            self.train["price"].to_numpy(dtype=np.float64)
        )
        train_valid = np.abs(train_change) > 1e-12
        if "segment_id" in self.train:
            train_segment = self.train["segment_id"].to_numpy(copy=False)
            train_valid &= train_segment[:-1] == train_segment[1:]
        if "obi_valid" in self.train:
            train_valid &= (
                self.train["obi_valid"].astype(bool).to_numpy()[:-1]
            )
        train_target = train_change[train_valid] > 0.0
        train_qrw_probability = np.clip(
            AdaptiveDecoherenceQRW(
                self.train.iloc[:-1].copy(),
                {
                    "n_positions": 101,
                    "gamma_base": qrw_parameters["gamma"],
                    "obi_bias": qrw_parameters["obi_bias"],
                    "alpha_obi": qrw_parameters["alpha_obi"],
                    "alpha_direction": qrw_parameters["alpha_direction"],
                    "alpha_obi_change": qrw_parameters["alpha_obi_change"],
                    "alpha_abs_obi": qrw_parameters["alpha_abs_obi"],
                    "gamma_intensity": qrw_parameters["gamma_intensity"],
                    "feature_mean": qrw_parameters["feature_mean"],
                    "feature_scale": qrw_parameters["feature_scale"],
                },
            ).predict_probability()[train_valid],
            1e-12,
            1.0 - 1e-12,
        )
        qrw_log_likelihood = float(
            np.sum(
                np.where(
                    train_target,
                    np.log(train_qrw_probability),
                    np.log1p(-train_qrw_probability),
                )
            )
        )
        qrw_parameter_count = 6
        comparison_rows.append(
            self._comparison_row(
                "QRW Adaptive",
                log_likelihood=qrw_log_likelihood,
                parameter_count=qrw_parameter_count,
                observations=len(train_target),
                empirical_volatility=float(np.std(train_returns)),
                model_volatility=model_volatility,
                likelihood_type="directional_bernoulli",
            )
        )

        heavy_tail_qrw, _ = self._fit_qrw_heavy_tail()
        heavy_tail_paths = self._simulate_qrw_heavy_tail(
            heavy_tail_qrw,
            seed=seeds[0],
        )
        self.simulated_paths["QRW Heavy-Tail"] = heavy_tail_paths
        rows, heavy_tail_volatility = self._score_paths(
            "QRW Heavy-Tail",
            heavy_tail_paths,
        )
        metric_rows.extend(rows)
        # The directional coin is identical to QRW Adaptive, so the Bernoulli
        # log-likelihood matches; the Pareto tail adds one parameter (tail index).
        comparison_rows.append(
            self._comparison_row(
                "QRW Heavy-Tail",
                log_likelihood=qrw_log_likelihood,
                parameter_count=qrw_parameter_count + 1,
                observations=len(train_target),
                empirical_volatility=float(np.std(train_returns)),
                model_volatility=heavy_tail_volatility,
                likelihood_type="directional_bernoulli",
            )
        )

        for index, kind in enumerate(("simple", "biased", "correlated"), 1):
            model = ClassicalRandomWalk(
                kind=kind,
                initial_position=self.initial_price,
                step_size=self.tick_size,
            )
            model.fit(train_directions)
            paths = model.simulate(
                self.n_steps,
                self.n_paths,
                random_state=seeds[index],
            )
            self.simulated_paths[model.model_name] = paths
            rows, model_volatility = self._score_paths(
                model.model_name,
                paths,
            )
            metric_rows.extend(rows)
            log_probability = model.direction_log_probabilities(
                moving_train_directions
            )
            parameter_count = {"simple": 0, "biased": 1, "correlated": 2}[
                kind
            ]
            comparison_rows.append(
                self._comparison_row(
                    model.model_name,
                    log_likelihood=float(np.sum(log_probability)),
                    parameter_count=parameter_count,
                    observations=len(log_probability),
                    empirical_volatility=float(np.std(train_returns)),
                    model_volatility=model_volatility,
                    likelihood_type="directional_bernoulli",
                )
            )

        garch = GARCHBaseline(initial_price=self.initial_price)
        garch_parameters = garch.fit(
            train_returns,
            output_path=garch_output,
        )
        garch_paths = garch.simulate(
            self.n_steps,
            self.n_paths,
            random_state=seeds[4],
        )
        self.simulated_paths["GARCH(1,1)"] = garch_paths
        rows, model_volatility = self._score_paths(
            "GARCH(1,1)",
            garch_paths,
        )
        metric_rows.extend(rows)
        comparison_rows.append(
            {
                "model": "GARCH(1,1)",
                "aic": garch.aic,
                "bic": garch.bic,
                "log_likelihood": garch.log_likelihood_value,
                "parameter_count": 4,
                "observations": len(train_returns),
                "empirical_volatility": float(np.std(train_returns)),
                "model_volatility": model_volatility,
                "likelihood_type": "continuous_gaussian",
            }
        )

        gbm = GBMBaseline(initial_price=self.initial_price)
        gbm_parameters = gbm.fit(train_returns)
        gbm_paths = gbm.simulate(
            self.n_steps,
            self.n_paths,
            random_state=seeds[5],
        )
        self.simulated_paths["GBM"] = gbm_paths
        rows, model_volatility = self._score_paths("GBM", gbm_paths)
        metric_rows.extend(rows)
        comparison_rows.append(
            {
                "model": "GBM",
                "aic": gbm.aic,
                "bic": gbm.bic,
                "log_likelihood": gbm.log_likelihood_value,
                "parameter_count": 2,
                "observations": len(train_returns),
                "empirical_volatility": float(np.std(train_returns)),
                "model_volatility": model_volatility,
                "likelihood_type": "continuous_gaussian",
            }
        )

        self.results = pd.DataFrame(metric_rows)
        self.model_comparison = pd.DataFrame(comparison_rows)
        simple_ratio = float(
            self.results.loc[
                (self.results["model"] == "CRW Simple")
                & (self.results["metric"] == "variance_ratio"),
                "value",
            ].iloc[0]
        )
        self.diagnostics = {
            "protocol_version": self.PROTOCOL_VERSION,
            "train_fraction": self.train_fraction,
            "requested_n_steps": self.requested_n_steps,
            "train_rows": int(len(self.train)),
            "test_rows": int(len(self.test)),
            "n_steps": self.n_steps,
            "n_paths": self.n_paths,
            "tick_size": self.tick_size,
            "random_seed": self.random_seed,
            "model_count": int(self.results["model"].nunique()),
            "metric_count": int(self.results["metric"].nunique()),
            "simple_crw_variance_ratio": simple_ratio,
            "simple_crw_target": self.movement_probability,
            "simple_crw_passed": bool(
                abs(simple_ratio - self.movement_probability) < 0.05
            ),
            "roadmap_simple_crw_target_0_5_corrected": True,
            "movement_probability": self.movement_probability,
            "forecast_protocol": (
                "fixed_origin_ex_ante_last_observed_features"
            ),
            "uses_holdout_features_for_simulation": False,
            "qrw_forecast_up_probability": float(
                qrw.predict_probability()[-1]
            ),
            "garch_convergence_flag": int(
                garch_parameters["convergence_flag"]
            ),
            "garch_converged": bool(
                garch_parameters["convergence_flag"] == 0
            ),
            "coherent_no_decoherence": self._coherent_qrw_diagnostic(),
            "qrw_calibration": qrw_parameters,
            "gbm_parameters": gbm_parameters,
        }
        self.diagnostics["checkpoint_passed"] = bool(
            self.diagnostics["model_count"] >= 4
            and self.diagnostics["metric_count"] >= 5
            and self.diagnostics["simple_crw_passed"]
            and self.diagnostics["garch_converged"]
            and self.diagnostics["coherent_no_decoherence"]["passed"]
        )

        self._write_csv(benchmark_output, self.results)
        self._write_csv(comparison_output, self.model_comparison)
        if diagnostics_output is not None:
            destination = Path(diagnostics_output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(self.diagnostics, indent=2),
                encoding="utf-8",
            )
        return self.results.copy()

    @staticmethod
    def _comparison_row(
        model: str,
        *,
        log_likelihood: float,
        parameter_count: int,
        observations: int,
        empirical_volatility: float,
        model_volatility: float,
        likelihood_type: str,
    ) -> dict[str, float | int | str]:
        return {
            "model": model,
            "aic": 2 * parameter_count - 2 * log_likelihood,
            "bic": (
                np.log(max(observations, 1)) * parameter_count
                - 2 * log_likelihood
            ),
            "log_likelihood": log_likelihood,
            "parameter_count": parameter_count,
            "observations": observations,
            "empirical_volatility": empirical_volatility,
            "model_volatility": model_volatility,
            "likelihood_type": likelihood_type,
        }

    @staticmethod
    def _write_csv(
        output: str | Path | None,
        frame: pd.DataFrame,
    ) -> None:
        if output is None:
            return
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)

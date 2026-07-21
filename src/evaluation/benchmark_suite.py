"""Fair chronological benchmark for QRW and classical market baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.qrw_market_sim import MarketQRW
from src.models.coin_operators import su2_market_coin
from src.models.classical_rw import ClassicalRandomWalk
from src.models.garch_model import GARCHBaseline
from src.models.gbm_model import GBMBaseline
from src.models.qrw_core import DensityMatrixQRW, QuantumRandomWalk


class BenchmarkSuite:
    """Fit all models on a common past window and score one later path."""

    PROTOCOL_VERSION = "fixed_origin_marginal_density_matrix_ar1_obi_v4"
    REQUIRED_COLUMNS = MarketQRW.REQUIRED_COLUMNS
    METRICS = (
        "mean_marginal_crps",
        "mean_marginal_absolute_error",
        "terminal_variance_per_step",
        "direction_brier_score",
        "direction_log_loss",
        "coverage_90",
        "mean_interval_width_90",
    )

    def __init__(
        self,
        market_data: pd.DataFrame,
        *,
        holdout_data: pd.DataFrame | None = None,
        train_fraction: float = 0.6,
        n_steps: int = 500,
        n_paths: int = 5_000,
        random_seed: int = 2026,
    ) -> None:
        if holdout_data is None and not 0.5 <= train_fraction <= 0.8:
            raise ValueError("train_fraction must be between 0.5 and 0.8")
        if n_steps < 10:
            raise ValueError("n_steps must be at least 10")
        if n_paths < 100:
            raise ValueError("n_paths must be at least 100")

        frame = self._validated_market_frame(market_data, name="market_data")
        if holdout_data is None:
            cut = int(len(frame) * train_fraction)
            train = frame.iloc[:cut].copy().reset_index(drop=True)
            holdout = frame.iloc[cut:].copy().reset_index(drop=True)
            split_strategy = "internal_chronological_fraction"
        else:
            train = frame
            holdout = self._validated_market_frame(
                holdout_data,
                name="holdout_data",
            )
            if train["timestamp"].iloc[-1] > holdout["timestamp"].iloc[0]:
                raise ValueError(
                    "explicit holdout_data cannot start before market_data ends"
                )
            cut = len(train)
            split_strategy = "explicit_chronological_holdout"

        available_steps = len(holdout)
        if cut < 100 or available_steps < 10:
            raise ValueError("dataset is too short for the requested split")
        self.train_fraction = float(cut / (cut + len(holdout)))
        self.split_strategy = split_strategy
        self.requested_n_steps = int(n_steps)
        self.n_steps = min(int(n_steps), available_steps)
        self.n_paths = int(n_paths)
        self.random_seed = int(random_seed)
        self.train = train
        self.holdout = holdout
        # Forecast from the last observed training price.  The holdout supplies
        # only future targets; none of its price or feature rows initializes the
        # simulator.
        self.test = pd.concat(
            [self.train.iloc[[-1]], self.holdout.iloc[: self.n_steps]],
            ignore_index=True,
        )
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
        self.forecast_samples: dict[str, np.ndarray] = {}
        # Kept as a read-compatible alias while callers migrate terminology.
        # Every downstream evaluator receives ``sample_semantics`` and must not
        # interpret QRW rows as jointly sampled trajectories.
        self.simulated_paths = self.forecast_samples
        self.sample_semantics: dict[str, str] = {}
        self.rolling_one_step_losses: dict[str, np.ndarray] = {}
        self.diagnostics: dict[str, Any] = {}
        self._qrw_forecast_diagnostics: dict[str, Any] = {}

    @classmethod
    def _validated_market_frame(
        cls,
        frame: pd.DataFrame,
        *,
        name: str,
    ) -> pd.DataFrame:
        missing = sorted(cls.REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} cannot be empty")
        validated = frame.sort_values(
            "timestamp",
            kind="stable",
        ).reset_index(drop=True).copy()
        numeric_columns = [
            "price",
            "tick_direction",
            "obi",
            "trade_intensity",
        ]
        numeric = validated[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("market data must contain finite numeric values")
        if (numeric["price"] <= 0.0).any():
            raise ValueError("prices must be positive")
        if (numeric["trade_intensity"] < 0.0).any():
            raise ValueError("trade_intensity must be non-negative")
        validated[numeric_columns] = numeric
        validated["obi"] = validated["obi"].clip(-1.0, 1.0)
        return validated

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

    def _fit_qrw(self) -> tuple[MarketQRW, dict[str, Any]]:
        calibration = MarketQRW(
            self.train,
            {
                "n_positions": 2 * self.n_steps + 3,
                "coin_type": "obi_adaptive",
                "quantum_calibration_max_events": 5000,
                "quantum_window_size": 5,
            },
        )
        parameters = calibration.calibrate("results/benchmark_params.json")
        return calibration, parameters

    @staticmethod
    def _fit_obi_ar1(frame: pd.DataFrame) -> dict[str, float | int]:
        """Fit a stable AR(1) OBI forecast using training pairs only."""
        obi = frame["obi"].to_numpy(dtype=np.float64)
        valid = np.ones(max(len(obi) - 1, 0), dtype=bool)
        if "segment_id" in frame:
            segment = frame["segment_id"].to_numpy(copy=False)
            valid &= segment[:-1] == segment[1:]
        if "obi_valid" in frame:
            observed = frame["obi_valid"].astype(bool).to_numpy()
            valid &= observed[:-1] & observed[1:]
        previous = obi[:-1][valid]
        following = obi[1:][valid]
        if len(previous) < 2 or np.var(previous) <= 1e-15:
            phi = 0.0
            intercept = float(np.mean(following)) if len(following) else float(obi[-1])
        else:
            centered_previous = previous - previous.mean()
            phi = float(
                centered_previous @ (following - following.mean())
                / (centered_previous @ centered_previous)
            )
            phi = float(np.clip(phi, -0.99, 0.99))
            intercept = float(following.mean() - phi * previous.mean())
        residual = following - (intercept + phi * previous)
        return {
            "intercept": intercept,
            "phi": phi,
            "residual_std": float(np.std(residual, ddof=0)) if len(residual) else 0.0,
            "training_pairs": int(len(previous)),
        }

    def _forecast_qrw_features(
        self,
        model: MarketQRW,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        """Create a causal fixed-origin feature path for the QRW horizon."""
        ar1 = self._fit_obi_ar1(self.train)
        features = np.empty((self.n_steps, 2), dtype=np.float64)
        obi_history = model.tick_data["obi"].to_numpy(dtype=np.float64)
        direction_history = model.tick_data["tick_direction"].to_numpy(dtype=np.float64)
        features[0] = (obi_history[-1], direction_history[-1])
        obi = float(features[0, 0])
        direction = float(features[0, 1])
        for index in range(1, self.n_steps):
            forecast_obi = float(
                np.clip(ar1["intercept"] + ar1["phi"] * obi, -1.0, 1.0)
            )
            features[index] = (
                forecast_obi,
                direction,
            )
            obi = forecast_obi
        return features, ar1

    def _simulate_qrw(
        self,
        model: MarketQRW,
        *,
        seed: int,
    ) -> np.ndarray:
        """Evolve a density matrix and sample each fixed-origin marginal.

        The samples are forecast-horizon marginals, not repeatedly observed
        classical trajectories.  Keeping that distinction avoids destroying
        interference through an implicit position measurement after every step.
        """
        rng = np.random.default_rng(seed)
        features, ar1 = self._forecast_qrw_features(model)
        engine = DensityMatrixQRW(2 * self.n_steps + 3)
        movement_probability = float(model.movement_probability)
        marginals = np.empty(
            (self.n_paths, self.n_steps + 1),
            dtype=np.float64,
        )
        marginals[:, 0] = self.initial_price
        quantum_variance = np.empty(self.n_steps, dtype=np.float64)
        for index, feature in enumerate(features):
            previous_rho = engine.rho.copy()
            event_gamma = model.gamma
            coin = su2_market_coin(
                float(feature[0]),
                float(feature[1]),
                bias=model.obi_bias,
                alpha_obi=model.alpha_obi,
                alpha_direction=model.alpha_direction,
                alpha_phase=getattr(model, "alpha_phase", 0.0),
                window=getattr(model, "quantum_window_size", 1),
            )
            engine.step_with_decoherence(event_gamma, coin_matrix=coin)
            if movement_probability < 1.0:
                engine.rho = (
                    movement_probability * engine.rho
                    + (1.0 - movement_probability) * previous_rho
                )
                engine.rho = (engine.rho + engine.rho.conj().T) / 2.0
            trace = engine.trace()
            if not np.isfinite(trace) or trace <= 0.0:
                raise RuntimeError("QRW density evolution lost normalization")
            engine.rho /= trace
            probability = engine.get_probability()
            total = float(probability.sum())
            if not np.isfinite(total) or total <= 0.0:
                raise RuntimeError("QRW density evolution lost normalization")
            probability = probability / total
            cumulative = np.cumsum(probability)
            cumulative[-1] = 1.0
            sampled_index = np.searchsorted(
                cumulative,
                rng.random(self.n_paths),
                side="right",
            )
            sampled_position = engine.positions[sampled_index]
            marginals[:, index + 1] = (
                self.initial_price + self.tick_size * sampled_position
            )
            mean_position = float(probability @ engine.positions)
            quantum_variance[index] = float(
                probability @ (engine.positions - mean_position) ** 2
            )
        positive_variance = quantum_variance > 0.0
        variance_scaling_exponent = (
            float(
                np.polyfit(
                    np.log(np.arange(1, self.n_steps + 1)[positive_variance]),
                    np.log(quantum_variance[positive_variance]),
                    1,
                )[0]
            )
            if np.count_nonzero(positive_variance) >= 2
            else None
        )
        self._qrw_forecast_diagnostics = {
            "state_evolution": "density_matrix_adaptive_coin_dephasing",
            "sampling_protocol": "fixed_origin_position_marginals",
            "future_obi_model": "train_only_ar1_conditional_mean",
            "obi_ar1": ar1,
            "movement_channel": "identity_quantum_step_mixture",
            "movement_probability": movement_probability,
            "position_variance_by_horizon": quantum_variance.tolist(),
            "variance_scaling_exponent": variance_scaling_exponent,
            "terminal_position_variance": float(quantum_variance[-1]),
            "terminal_variance_per_step": float(
                quantum_variance[-1] / self.n_steps
            ),
        }
        return marginals

    @staticmethod
    def _sample_crps(samples: np.ndarray, observation: float) -> float:
        """Return the empirical CRPS without a quadratic pairwise matrix."""
        values = np.sort(np.asarray(samples, dtype=np.float64))
        count = len(values)
        if count == 0:
            raise ValueError("CRPS requires at least one forecast sample")
        weights = 2.0 * np.arange(1, count + 1) - count - 1.0
        pairwise_half = float(weights @ values) / (count * count)
        return float(np.mean(np.abs(values - observation)) - pairwise_half)

    def _score_marginals(
        self,
        model_name: str,
        samples: np.ndarray,
    ) -> tuple[list[dict[str, float | str]], float]:
        expected_shape = (self.n_paths, self.n_steps + 1)
        if samples.shape != expected_shape:
            raise ValueError(
                f"{model_name} returned {samples.shape}, expected {expected_shape}"
            )
        if not np.isfinite(samples).all():
            raise ValueError(f"{model_name} produced non-finite samples")

        realized = self.test["price"].to_numpy(dtype=np.float64)
        absolute_errors = np.mean(
            np.abs(samples[:, 1:] - realized[None, 1:]),
            axis=0,
        )
        crps = np.asarray(
            [
                self._sample_crps(samples[:, horizon], realized[horizon])
                for horizon in range(1, self.n_steps + 1)
            ],
            dtype=np.float64,
        )
        terminal_ticks = (
            samples[:, -1] - samples[:, 0]
        ) / self.tick_size
        terminal_centered_square = (
            terminal_ticks - terminal_ticks.mean()
        ) ** 2
        terminal_variance_per_step = float(
            np.mean(terminal_centered_square) / self.n_steps
        )
        terminal_variance_se = float(
            np.std(terminal_centered_square, ddof=1)
            / np.sqrt(self.n_paths)
            / self.n_steps
        )

        fixed_origin_change = samples[:, 1:] - samples[:, [0]]
        probability_up = np.mean(fixed_origin_change > 0.0, axis=0)
        target_up = realized[1:] > realized[0]
        brier = (probability_up - target_up.astype(np.float64)) ** 2
        clipped_probability = np.clip(probability_up, 1e-12, 1.0 - 1e-12)
        direction_log_loss = -np.where(
            target_up,
            np.log(clipped_probability),
            np.log1p(-clipped_probability),
        )
        lower, upper = np.quantile(samples[:, 1:], [0.05, 0.95], axis=0)
        covered = (realized[1:] >= lower) & (realized[1:] <= upper)
        interval_width = (upper - lower) / self.tick_size

        def horizon_standard_error(values: np.ndarray) -> float:
            return float(np.std(values, ddof=1) / np.sqrt(len(values)))

        rows: list[dict[str, float | str]] = [
            {
                "model": model_name,
                "metric": "mean_marginal_crps",
                "value": float(np.mean(crps)),
                "std": horizon_standard_error(crps),
            },
            {
                "model": model_name,
                "metric": "mean_marginal_absolute_error",
                "value": float(np.mean(absolute_errors)),
                "std": horizon_standard_error(absolute_errors),
            },
            {
                "model": model_name,
                "metric": "terminal_variance_per_step",
                "value": terminal_variance_per_step,
                "std": terminal_variance_se,
            },
            {
                "model": model_name,
                "metric": "direction_brier_score",
                "value": float(np.mean(brier)),
                "std": horizon_standard_error(brier),
            },
            {
                "model": model_name,
                "metric": "direction_log_loss",
                "value": float(np.mean(direction_log_loss)),
                "std": horizon_standard_error(direction_log_loss),
            },
            {
                "model": model_name,
                "metric": "coverage_90",
                "value": float(np.mean(covered)),
                "std": horizon_standard_error(covered.astype(np.float64)),
            },
            {
                "model": model_name,
                "metric": "mean_interval_width_90",
                "value": float(np.mean(interval_width)),
                "std": horizon_standard_error(interval_width),
            },
        ]
        terminal_log_return = np.log(samples[:, -1] / samples[:, 0])
        volatility_per_step = float(
            np.std(terminal_log_return, ddof=0) / np.sqrt(self.n_steps)
        )
        return rows, volatility_per_step

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

    def _rolling_one_step_absolute_losses(
        self,
        *,
        qrw_parameters: dict[str, Any],
        crw_models: dict[str, ClassicalRandomWalk],
        garch: GARCHBaseline,
        gbm: GBMBaseline,
    ) -> dict[str, np.ndarray]:
        """Return causally aligned one-step losses on the common holdout."""
        origins = self.test["price"].to_numpy(dtype=np.float64)[:-1]
        targets = self.test["price"].to_numpy(dtype=np.float64)[1:]
        predictor_frame = pd.concat(
            [
                self.train.iloc[-2:],
                self.holdout.iloc[: max(self.n_steps - 1, 0)],
            ],
            ignore_index=True,
        )
        qrw_predictor = MarketQRW(
            predictor_frame,
            {
                "n_positions": 101,
                "gamma_base": qrw_parameters.get("gamma", 0.0),
                "obi_bias": qrw_parameters.get("obi_bias", 0.0),
                "alpha_obi": qrw_parameters.get("alpha_obi", 0.0),
                "alpha_direction": qrw_parameters.get("alpha_direction", 0.0),
                "coin_type": "obi_adaptive",
                "quantum_window": qrw_parameters.get("quantum_window", 5),
            },
        )
        # Carry over the quantum-refinement outcome from the original fit:
        # constructing a fresh MarketQRW here must not silently reset
        # alpha_phase/quantum_improved to their un-refined defaults.
        qrw_predictor.alpha_phase = float(qrw_parameters.get("alpha_phase", 0.0))
        qrw_predictor.quantum_improved = bool(qrw_parameters.get("quantum_improved", False))
        probability_up = np.clip(
            qrw_predictor.predict_right_probabilities(
                predictor_frame["obi"].to_numpy(dtype=np.float64)[:-1],
                predictor_frame["tick_direction"].to_numpy(dtype=np.float64)[:-1]
            ),
            1e-12,
            1.0 - 1e-12,
        )
        qrw_prediction = origins + self.tick_size * self.movement_probability * (
            2.0 * probability_up - 1.0
        )
        losses: dict[str, np.ndarray] = {
            "QRW Adaptive": np.abs(qrw_prediction - targets)
        }

        for name, model in crw_models.items():
            if model.kind == "simple":
                expected_direction = np.zeros(self.n_steps)
            elif model.kind == "biased":
                expected_direction = np.full(
                    self.n_steps, 2.0 * model.p_up - 1.0
                )
            else:
                expected_direction = np.empty(self.n_steps)
                training_direction = self._price_directions(self.train)
                moving = training_direction[training_direction != 0.0]
                last_direction = float(moving[-1])
                realized_direction = np.sign(targets - origins)
                for index in range(self.n_steps):
                    expected_direction[index] = model.rho * last_direction
                    if realized_direction[index] != 0.0:
                        last_direction = float(realized_direction[index])
            prediction = origins + (
                self.tick_size * model.p_move * expected_direction
            )
            losses[name] = np.abs(prediction - targets)

        garch_mean_return = float(garch.parameters["mu"])
        losses["GARCH(1,1)"] = np.abs(
            origins * np.exp(garch_mean_return) - targets
        )
        losses["GBM"] = np.abs(origins * np.exp(gbm.mu) - targets)
        return losses

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
        seeds = self._seed_values(self.random_seed, 6)
        metric_rows: list[dict[str, float | str]] = []
        comparison_rows: list[dict[str, float | int | str]] = []
        crw_models: dict[str, ClassicalRandomWalk] = {}

        qrw, qrw_parameters = self._fit_qrw()
        qrw_marginals = self._simulate_qrw(qrw, seed=seeds[0])
        self.forecast_samples["QRW Adaptive"] = qrw_marginals
        self.sample_semantics["QRW Adaptive"] = "fixed_origin_marginals"
        rows, model_volatility = self._score_marginals(
            "QRW Adaptive", qrw_marginals
        )
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
        train_qrw_predictor = MarketQRW(
            self.train.iloc[:-1].copy(),
            {
                "n_positions": 101,
                "gamma_base": qrw_parameters.get("gamma", 0.0),
                "obi_bias": qrw_parameters.get("obi_bias", 0.0),
                "alpha_obi": qrw_parameters.get("alpha_obi", 0.0),
                "alpha_direction": qrw_parameters.get("alpha_direction", 0.0),
                "coin_type": "obi_adaptive",
                "quantum_window": qrw_parameters.get("quantum_window", 5),
            },
        )
        # Carry over the quantum-refinement outcome from the original fit —
        # see the identical comment in _rolling_one_step_absolute_losses.
        train_qrw_predictor.alpha_phase = float(qrw_parameters.get("alpha_phase", 0.0))
        train_qrw_predictor.quantum_improved = bool(
            qrw_parameters.get("quantum_improved", False)
        )
        train_qrw_probability = np.clip(
            train_qrw_predictor.predict_right_probabilities(
                self.train["obi"].to_numpy(dtype=np.float64)[:-1],
                self.train["tick_direction"].to_numpy(dtype=np.float64)[:-1]
            )[train_valid],
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
        qrw_parameter_count = 5 if train_qrw_predictor.quantum_improved else 3
        comparison_directions = np.where(train_target, 1.0, -1.0)
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

        for index, kind in enumerate(("simple", "biased", "correlated"), 1):
            model = ClassicalRandomWalk(
                kind=kind,
                initial_position=self.initial_price,
                step_size=self.tick_size,
            )
            model.fit(train_directions)
            crw_models[model.model_name] = model
            paths = model.simulate(
                self.n_steps,
                self.n_paths,
                random_state=seeds[index],
            )
            self.forecast_samples[model.model_name] = paths
            self.sample_semantics[model.model_name] = "path_trajectories"
            rows, model_volatility = self._score_marginals(
                model.model_name,
                paths,
            )
            metric_rows.extend(rows)
            log_probability = model.direction_log_probabilities(
                comparison_directions
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
        self.forecast_samples["GARCH(1,1)"] = garch_paths
        self.sample_semantics["GARCH(1,1)"] = "path_trajectories"
        rows, model_volatility = self._score_marginals(
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
        self.forecast_samples["GBM"] = gbm_paths
        self.sample_semantics["GBM"] = "path_trajectories"
        rows, model_volatility = self._score_marginals("GBM", gbm_paths)
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
        self.rolling_one_step_losses = self._rolling_one_step_absolute_losses(
            qrw_parameters=qrw_parameters,
            crw_models=crw_models,
            garch=garch,
            gbm=gbm,
        )
        self.rolling_one_step_timestamps = self.test['timestamp'].to_numpy()[1:]
        self.model_comparison = self.rank_information_criteria(
            pd.DataFrame(comparison_rows)
        )
        simple_ratio = float(
            self.results.loc[
                (self.results["model"] == "CRW Simple")
                & (self.results["metric"] == "terminal_variance_per_step"),
                "value",
            ].iloc[0]
        )
        self.diagnostics = {
            "protocol_version": self.PROTOCOL_VERSION,
            "train_fraction": self.train_fraction,
            "split_strategy": self.split_strategy,
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
                "fixed_origin_density_matrix_train_only_ar1_obi"
            ),
            "uses_holdout_features_for_simulation": False,
            "evaluation_semantics": "fixed_origin_marginals_only",
            "sample_semantics": self.sample_semantics,
            "path_dependent_metrics_include_qrw": False,
            "dm_forecast_alignment": "rolling_origin_one_step",
            "dm_timestamp_alignment": "strictly_increasing_common_timestamp_index",
            "dm_loss_observations": self.n_steps,
            "qrw_probability_at_origin": float(
                qrw.predict_right_probabilities(
                    qrw.tick_data["obi"].to_numpy(dtype=np.float64)[:-1],
                    qrw.tick_data["tick_direction"].to_numpy(dtype=np.float64)[:-1]
                )[-1]
            ),
            "qrw_forecast": self._qrw_forecast_diagnostics,
            "information_criterion_scope": (
                "AIC/BIC ranks are valid only within likelihood_type"
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
    def rank_information_criteria(frame: pd.DataFrame) -> pd.DataFrame:
        """Rank AIC/BIC only within matching likelihood families."""
        required = {"likelihood_type", "aic", "bic"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"information-criterion table is missing columns: {missing}"
            )
        ranked = frame.copy()
        ranked["aic_rank_within_likelihood"] = (
            ranked.groupby("likelihood_type", sort=False)["aic"]
            .rank(method="min", ascending=True)
        )
        ranked["bic_rank_within_likelihood"] = (
            ranked.groupby("likelihood_type", sort=False)["bic"]
            .rank(method="min", ascending=True)
        )
        ranked["information_criterion_scope"] = (
            "within_likelihood_type_only"
        )
        return ranked

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

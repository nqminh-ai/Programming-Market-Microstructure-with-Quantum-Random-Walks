"""QRW market model with causal features and intensity-adaptive decoherence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .coin_operators import obi_adaptive_coin
from .qrw_core import DensityMatrixQRW
from .qrw_market_sim import MarketQRW


class AdaptiveDecoherenceQRW:
    """Use market activity to modulate QRW coherence event by event."""

    REQUIRED_COLUMNS = MarketQRW.REQUIRED_COLUMNS
    FEATURE_NAMES = (
        "obi",
        "tick_direction",
        "obi_change",
        "abs_obi",
        "log_trade_intensity",
    )

    def __init__(self, tick_data: pd.DataFrame, config: dict[str, Any]) -> None:
        missing = sorted(self.REQUIRED_COLUMNS.difference(tick_data.columns))
        if missing:
            raise ValueError(f"tick_data is missing columns: {missing}")
        if tick_data.empty:
            raise ValueError("tick_data cannot be empty")
        data = tick_data.sort_values("timestamp", kind="stable").reset_index(
            drop=True
        ).copy()
        numeric_columns = [
            "price",
            "tick_direction",
            "obi",
            "trade_intensity",
        ]
        numeric = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("market features must be finite")
        if (numeric["trade_intensity"] < 0.0).any():
            raise ValueError("trade_intensity must be non-negative")
        data[numeric_columns] = numeric
        data["obi"] = data["obi"].clip(-1.0, 1.0)

        self.tick_data = data
        self.config = dict(config)
        self.n_positions = int(self.config.get("n_positions", 201))
        if self.n_positions < 3 or self.n_positions % 2 == 0:
            raise ValueError("config['n_positions'] must be an odd integer")
        self.gamma = float(self.config.get("gamma_base", 0.0))
        self.obi_bias = float(self.config.get("obi_bias", 0.0))
        self.coefficients = np.asarray(
            [
                self.config.get("alpha_obi", 0.0),
                self.config.get("alpha_direction", 0.0),
                self.config.get("alpha_obi_change", 0.0),
                self.config.get("alpha_abs_obi", 0.0),
            ],
            dtype=np.float64,
        )
        self.gamma_intensity = float(
            self.config.get("gamma_intensity", 0.0)
        )
        self.movement_probability = float(
            self.config.get("movement_probability", 1.0)
        )
        self.feature_mean = np.asarray(
            self.config.get("feature_mean", np.zeros(5)),
            dtype=np.float64,
        )
        self.feature_scale = np.asarray(
            self.config.get("feature_scale", np.ones(5)),
            dtype=np.float64,
        )
        if self.feature_mean.shape != (5,) or self.feature_scale.shape != (5,):
            raise ValueError("feature_mean and feature_scale must have length 5")
        if (
            not np.isfinite(self.coefficients).all()
            or not np.isfinite(self.feature_mean).all()
            or not np.isfinite(self.feature_scale).all()
            or np.any(self.feature_scale <= 0.0)
            or not np.isfinite(self.gamma_intensity)
            or self.gamma < 0.0
            or not 0.0 <= self.movement_probability <= 1.0
        ):
            raise ValueError("adaptive QRW parameters must be finite and valid")
        self.calibrated = False
        self.calibrated_parameters: dict[str, Any] = {}
        self._last_simulation_steps = 0

    def _raw_features(self) -> np.ndarray:
        obi = self.tick_data["obi"].to_numpy(dtype=np.float64)
        direction = self.tick_data["tick_direction"].to_numpy(dtype=np.float64)
        intensity = self.tick_data["trade_intensity"].to_numpy(dtype=np.float64)
        obi_change = np.zeros(len(obi), dtype=np.float64)
        if len(obi) > 1:
            obi_change[1:] = np.diff(obi)
        if "segment_id" in self.tick_data:
            segment = self.tick_data["segment_id"].to_numpy(copy=False)
            obi_change[1:][segment[:-1] != segment[1:]] = 0.0
        return np.column_stack(
            [
                obi,
                direction,
                obi_change,
                np.abs(obi),
                np.log1p(intensity),
            ]
        )

    def _moving_events(
        self,
        row_limit: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        limit = len(self.tick_data) if row_limit is None else int(row_limit)
        if limit < 2:
            raise ValueError("at least two rows are required")
        price = self.tick_data["price"].to_numpy(dtype=np.float64)[:limit]
        features = self._raw_features()[:limit]
        delta = np.diff(price)
        valid = np.abs(delta) > 1e-12
        if "segment_id" in self.tick_data:
            segment = self.tick_data["segment_id"].to_numpy(copy=False)[:limit]
            valid &= segment[:-1] == segment[1:]
        if "obi_valid" in self.tick_data:
            valid &= (
                self.tick_data["obi_valid"].astype(bool).to_numpy()[:limit][:-1]
            )
        return features[:-1][valid], (delta[valid] > 0.0).astype(np.float64)

    @staticmethod
    def _log_loss(probability: np.ndarray, target: np.ndarray) -> float:
        return MarketQRW._log_loss(probability, target)

    @staticmethod
    def _brier(probability: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((probability - target) ** 2))

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        return (features - self.feature_mean) / self.feature_scale

    @staticmethod
    def _probability(
        normalized: np.ndarray,
        parameters: np.ndarray,
        gamma_base: float,
    ) -> np.ndarray:
        signal = parameters[0] + normalized[:, :4] @ parameters[1:5]
        log_gamma = np.clip(parameters[5] * normalized[:, 4], -5.0, 5.0)
        coherence = np.exp(-gamma_base * np.exp(log_gamma))
        return np.clip(
            0.5 + 0.5 * coherence * np.tanh(signal),
            1e-12,
            1.0 - 1e-12,
        )

    @classmethod
    def _objective(
        cls,
        parameters: np.ndarray,
        normalized: np.ndarray,
        target: np.ndarray,
        gamma_base: float,
        regularization: float,
    ) -> float:
        probability = cls._probability(
            normalized,
            parameters,
            gamma_base,
        )
        return (
            cls._log_loss(probability, target)
            + regularization * float(parameters[1:] @ parameters[1:])
        )

    def _estimate_gamma(self, row_limit: int) -> tuple[float, float]:
        direction = self.tick_data["tick_direction"].to_numpy(
            dtype=np.float64
        )[:row_limit]
        same = np.ones(len(direction) - 1, dtype=bool)
        if "segment_id" in self.tick_data:
            segment = self.tick_data["segment_id"].to_numpy(copy=False)[
                :row_limit
            ]
            same = segment[:-1] == segment[1:]
        previous = direction[:-1][same]
        following = direction[1:][same]
        if (
            len(following) < 3
            or np.var(previous) <= 1e-15
            or np.var(following) <= 1e-15
        ):
            rho_1 = 0.0
        else:
            rho_1 = float(np.corrcoef(previous, following)[0, 1])
        gamma = float(-np.log(np.clip(abs(rho_1), 1e-8, 1.0)))
        return gamma, rho_1

    def calibrate_two_stage(
        self,
        output_path: str | Path | None = "results/calibrated_params.json",
        *,
        warmup_fraction: float = 0.4,
    ) -> dict[str, Any]:
        """Select structure chronologically, freeze it, then update only bias."""
        if not 0.3 <= warmup_fraction <= 0.8:
            raise ValueError("warmup_fraction must be between 0.3 and 0.8")
        warmup_rows = int(len(self.tick_data) * warmup_fraction)
        features, target = self._moving_events(warmup_rows)
        if len(target) < 40:
            raise ValueError(
                "adaptive calibration requires at least 40 warmup moving events"
            )
        validation_count = max(10, int(np.ceil(0.25 * len(target))))
        train_features = features[:-validation_count]
        validation_features = features[-validation_count:]
        train_target = target[:-validation_count]
        validation_target = target[-validation_count:]
        self.feature_mean = train_features.mean(axis=0)
        self.feature_scale = train_features.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-9] = 1.0
        train_normalized = self._normalize(train_features)
        validation_normalized = self._normalize(validation_features)
        gamma_rows = max(3, int(warmup_rows * 0.75))
        self.gamma, rho_1 = self._estimate_gamma(gamma_rows)

        regularization_grid = tuple(
            float(value)
            for value in self.config.get(
                "calibration_regularization_grid",
                (1e-4, 1e-3, 1e-2, 5e-2, 1e-1),
            )
        )
        candidates: list[dict[str, Any]] = []
        for regularization in regularization_grid:
            result = minimize(
                self._objective,
                x0=np.zeros(6, dtype=np.float64),
                args=(
                    train_normalized,
                    train_target,
                    self.gamma,
                    regularization,
                ),
                method="L-BFGS-B",
                bounds=(
                    (-3.0, 3.0),
                    (-5.0, 5.0),
                    (-5.0, 5.0),
                    (-5.0, 5.0),
                    (-5.0, 5.0),
                    (-2.0, 2.0),
                ),
            )
            if not result.success:
                continue
            probability = self._probability(
                validation_normalized,
                result.x,
                self.gamma,
            )
            candidates.append(
                {
                    "regularization": regularization,
                    "parameters": result.x.tolist(),
                    "validation_brier": self._brier(
                        probability,
                        validation_target,
                    ),
                    "validation_log_loss": self._log_loss(
                        probability,
                        validation_target,
                    ),
                }
            )
        if not candidates:
            raise RuntimeError("all adaptive calibration optimizations failed")
        selected = min(candidates, key=lambda item: item["validation_brier"])
        # Validation selects the regularization and parameter candidate but is
        # never included in the structural fit.
        parameters = np.asarray(
            selected["parameters"],
            dtype=np.float64,
        )
        all_features, all_target = self._moving_events()
        price_change = np.diff(
            self.tick_data["price"].to_numpy(dtype=np.float64)
        )
        valid_change = np.ones(len(price_change), dtype=bool)
        if "segment_id" in self.tick_data:
            segment = self.tick_data["segment_id"].to_numpy(copy=False)
            valid_change &= segment[:-1] == segment[1:]
        self.movement_probability = float(
            np.mean(np.abs(price_change[valid_change]) > 1e-12)
        )
        structural = parameters.copy()
        prior_bias = float(parameters[0])
        bias_data = self.tick_data.iloc[warmup_rows:].copy().reset_index(
            drop=True
        )
        bias_model = AdaptiveDecoherenceQRW(
            bias_data,
            {
                **self.config,
                "gamma_base": self.gamma,
                "obi_bias": prior_bias,
                "alpha_obi": float(parameters[1]),
                "alpha_direction": float(parameters[2]),
                "alpha_obi_change": float(parameters[3]),
                "alpha_abs_obi": float(parameters[4]),
                "gamma_intensity": float(parameters[5]),
                "feature_mean": self.feature_mean.tolist(),
                "feature_scale": self.feature_scale.tolist(),
            },
        )
        bias_features, bias_target = bias_model._moving_events()
        bias_normalized = self._normalize(bias_features)

        def bias_objective(value: np.ndarray) -> float:
            candidate = structural.copy()
            candidate[0] = float(value[0])
            probability = self._probability(
                bias_normalized,
                candidate,
                self.gamma,
            )
            return (
                self._log_loss(probability, bias_target)
                + 0.01 * candidate[0] ** 2
                + 0.1 * (candidate[0] - prior_bias) ** 2
            )

        bias_fit = minimize(
            bias_objective,
            x0=np.array([prior_bias]),
            method="L-BFGS-B",
            bounds=((-3.0, 3.0),),
        )
        if not bias_fit.success:
            raise RuntimeError(f"adaptive bias fit failed: {bias_fit.message}")
        parameters[0] = float(bias_fit.x[0])
        self.obi_bias = float(parameters[0])
        self.coefficients = parameters[1:5].copy()
        self.gamma_intensity = float(parameters[5])

        final_probability = self._probability(bias_normalized, parameters, self.gamma)
        final_log_loss = self._log_loss(final_probability, bias_target)
        n_obs = len(bias_target)
        k_params = 6  # bias + 4 features + gamma_intensity
        final_nll = final_log_loss * n_obs
        
        aic = 2 * k_params + 2 * final_nll
        bic = k_params * np.log(n_obs) + 2 * final_nll

        output = {
            "gamma": self.gamma,
            "rho_1": rho_1,
            "obi_bias": self.obi_bias,
            "alpha_obi": float(self.coefficients[0]),
            "alpha_direction": float(self.coefficients[1]),
            "alpha_obi_change": float(self.coefficients[2]),
            "alpha_abs_obi": float(self.coefficients[3]),
            "gamma_intensity": self.gamma_intensity,
            "movement_probability": self.movement_probability,
            "feature_names": list(self.FEATURE_NAMES),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "calibration_rows": int(len(self.tick_data)),
            "structural_calibration_rows": warmup_rows,
            "structural_moving_events": int(len(target)),
            "structural_fit_moving_events": int(len(train_target)),
            "selection_validation_events": int(len(validation_target)),
            "final_refit_includes_validation": False,
            "events_per_structural_parameter": float(len(train_target) / 6.0),
            "low_sample_warning": bool(len(train_target) < 300),
            "moving_events": int(len(all_target)),
            "bias_update_rows": int(len(bias_data)),
            "bias_update_moving_events": int(len(bias_target)),
            "bias_update_start_row": warmup_rows,
            "bias_update_reuses_warmup": False,
            "selected_regularization": selected["regularization"],
            "validation_brier": selected["validation_brier"],
            "validation_log_loss": selected["validation_log_loss"],
            "final_log_loss": final_log_loss,
            "aic": aic,
            "bic": bic,
            "regularization_candidates": candidates,
            "calibration_method": (
                "adaptive_decoherence_disjoint_two_stage_brier_validation"
            ),
            "calibration_status": "fixed_structure_bias_updated",
            "decoherence_channel": (
                "basis_dephasing_gamma_times_exp_intensity"
            ),
        }
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(output, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        self.calibrated_parameters = output
        self.calibrated = True
        return output.copy()

    def _event_kernel(
        self,
        feature: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        normalized = self._normalize(feature[None, :])[0]
        signal = self.obi_bias + normalized[:4] @ self.coefficients
        event_gamma = self.gamma * np.exp(
            np.clip(self.gamma_intensity * normalized[4], -5.0, 5.0)
        )
        coin = obi_adaptive_coin(0.0, 0.0, bias=float(signal))
        return float(event_gamma), coin

    def predict_probability(self) -> np.ndarray:
        features = self._raw_features()
        normalized = self._normalize(features)
        parameters = np.concatenate(
            [
                [self.obi_bias],
                self.coefficients,
                [self.gamma_intensity],
            ]
        )
        return self._probability(normalized, parameters, self.gamma)

    def simulate(self, T: int) -> pd.DataFrame:
        if T < 1 or T > len(self.tick_data):
            raise ValueError("T must be within the available observations")
        if T > self.n_positions // 2:
            raise ValueError("n_positions must be at least 2*T + 1")
        engine = DensityMatrixQRW(self.n_positions)
        features = self._raw_features()
        history = np.empty((T, self.n_positions), dtype=np.float64)
        for index in range(T):
            event_gamma, coin = self._event_kernel(features[index])
            history[index] = engine.step_with_decoherence(
                event_gamma,
                coin_matrix=coin,
            )
        self._last_simulation_steps = T
        return pd.DataFrame(
            {
                "t": np.repeat(np.arange(1, T + 1), self.n_positions),
                "position": np.tile(engine.positions, T),
                "probability": history.reshape(-1),
            }
        )

    def simulate_price_path(
        self,
        n_paths: int,
        T: int | None = None,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        if n_paths < 1:
            raise ValueError("n_paths must be positive")
        steps = (
            self._last_simulation_steps
            if T is None and self._last_simulation_steps
            else (1 if T is None else T)
        )
        if steps < 1 or steps > len(self.tick_data):
            raise ValueError("T must be within the available observations")
        if steps > self.n_positions // 2:
            raise ValueError("n_positions must be at least 2*T + 1")
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        probability = self.predict_probability()[:steps]
        moving = (
            rng.random((n_paths, steps))
            < self.movement_probability
        )
        positions = np.zeros(n_paths, dtype=np.int64)
        paths = np.empty((n_paths, steps), dtype=np.int64)
        for index, probability_right in enumerate(probability):
            direction = np.where(
                rng.random(n_paths) < probability_right,
                1,
                -1,
            )
            positions += np.where(moving[:, index], direction, 0)
            paths[:, index] = positions
        return paths

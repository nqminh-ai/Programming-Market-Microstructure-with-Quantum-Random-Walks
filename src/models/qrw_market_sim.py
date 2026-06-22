"""Market-adapted QRW calibration and simulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .coin_operators import (
    biased_coin,
    grover_coin,
    hadamard_coin,
    obi_adaptive_coin,
)
from .qrw_core import DensityMatrixQRW


class MarketQRW:
    """Drive a density-matrix QRW with observed market features."""

    REQUIRED_COLUMNS = {
        "timestamp",
        "price",
        "tick_direction",
        "obi",
        "trade_intensity",
    }

    def __init__(self, tick_data: pd.DataFrame, config: dict[str, Any]) -> None:
        missing = sorted(self.REQUIRED_COLUMNS.difference(tick_data.columns))
        if missing:
            raise ValueError(f"tick_data is missing columns: {missing}")
        if tick_data.empty:
            raise ValueError("tick_data cannot be empty")

        data = tick_data.copy()
        data = data.sort_values("timestamp", kind="stable").reset_index(drop=True)
        numeric_columns = ["price", "tick_direction", "obi", "trade_intensity"]
        numeric = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("market features must be finite")
        data[numeric_columns] = numeric
        data["obi"] = data["obi"].clip(-1.0, 1.0)

        self.tick_data = data
        self.config = dict(config)
        self.n_positions = int(self.config.get("n_positions", 201))
        if self.n_positions < 3 or self.n_positions % 2 == 0:
            raise ValueError("config['n_positions'] must be an odd integer")

        self.gamma_base = float(self.config.get("gamma_base", 0.0))
        self.alpha_obi = float(self.config.get("alpha_obi", 0.0))
        self.alpha_direction = float(
            self.config.get("alpha_direction", 0.0)
        )
        self.obi_bias = float(self.config.get("obi_bias", 0.0))
        self.movement_probability = float(
            self.config.get("movement_probability", 1.0)
        )
        self.coin_type = str(self.config.get("coin_type", "obi_adaptive")).lower()
        if self.gamma_base < 0.0 or not np.isfinite(self.gamma_base):
            raise ValueError("gamma_base must be finite and non-negative")
        if not np.isfinite(self.alpha_obi) or self.alpha_obi < 0.0:
            raise ValueError("alpha_obi must be finite and non-negative")
        if not np.isfinite(self.alpha_direction):
            raise ValueError("alpha_direction must be finite")
        if not np.isfinite(self.obi_bias):
            raise ValueError("obi_bias must be finite")
        if not 0.0 <= self.movement_probability <= 1.0:
            raise ValueError("movement_probability must be in [0, 1]")

        self.gamma = self.gamma_base
        self.calibrated = False
        self.calibrated_parameters: dict[str, Any] = {}
        self._probability_history: np.ndarray | None = None
        self._last_simulation_steps = 0

    def calibrate(
        self,
        output_path: str | Path | None = "results/calibrated_params.json",
    ) -> dict[str, Any]:
        """Calibrate a regularized directional coin with chronological validation."""
        direction = self.tick_data["tick_direction"].to_numpy(dtype=np.float64)
        same_segment = np.ones(max(len(direction) - 1, 0), dtype=bool)
        if "segment_id" in self.tick_data:
            segments = self.tick_data["segment_id"].to_numpy(copy=False)
            same_segment = segments[:-1] == segments[1:]
        previous_direction = direction[:-1][same_segment]
        next_direction = direction[1:][same_segment]
        if (
            len(next_direction) >= 3
            and np.var(previous_direction) > 1e-15
            and np.var(next_direction) > 1e-15
        ):
            rho_1 = float(np.corrcoef(previous_direction, next_direction)[0, 1])
            if not np.isfinite(rho_1):
                rho_1 = 0.0
        else:
            rho_1 = 0.0

        correlation_floor = float(self.config.get("correlation_floor", 1e-8))
        max_gamma = float(self.config.get("max_gamma", 20.0))
        if not 0.0 < correlation_floor < 1.0:
            raise ValueError("correlation_floor must be in (0, 1)")
        if max_gamma <= 0.0 or not np.isfinite(max_gamma):
            raise ValueError("max_gamma must be finite and positive")
        gamma_estimate = float(
            np.clip(
                -np.log(np.clip(abs(rho_1), correlation_floor, 1.0)),
                0.0,
                max_gamma,
            )
        )
        self.gamma = max(self.gamma_base, gamma_estimate)

        price = self.tick_data["price"].to_numpy(dtype=np.float64)
        obi = self.tick_data["obi"].to_numpy(dtype=np.float64)
        delta_price_all = np.diff(price)
        predictor_all = obi[:-1]
        direction_predictor_all = direction[:-1]
        finite = (
            np.isfinite(delta_price_all)
            & np.isfinite(predictor_all)
            & same_segment
        )
        if "obi_valid" in self.tick_data:
            finite &= self.tick_data["obi_valid"].astype(bool).to_numpy()[:-1]
        tick_size = self._resolve_tick_size(delta_price_all[finite])
        tick_displacement = np.clip(
            delta_price_all[finite] / tick_size,
            -1.0,
            1.0,
        )
        diagnostic_predictor = predictor_all[finite]
        predictor_variance = (
            float(np.var(diagnostic_predictor))
            if len(diagnostic_predictor)
            else 0.0
        )
        if len(diagnostic_predictor) >= 2 and predictor_variance > 1e-15:
            centered_predictor = diagnostic_predictor - diagnostic_predictor.mean()
            centered_delta = tick_displacement - tick_displacement.mean()
            slope_ticks = float(
                centered_predictor @ centered_delta
                / (centered_predictor @ centered_predictor)
            )
            # Around zero, the coherent directional link is bias + alpha*OBI.
            intercept_ticks = float(
                tick_displacement.mean()
                - slope_ticks * diagnostic_predictor.mean()
            )
        else:
            slope_ticks = 0.0
            intercept_ticks = (
                float(tick_displacement.mean()) if len(tick_displacement) else 0.0
            )

        moving = finite & (np.abs(delta_price_all) > 1e-12)
        valid_event_count = int(np.count_nonzero(finite))
        self.movement_probability = (
            float(np.count_nonzero(moving) / valid_event_count)
            if valid_event_count
            else 0.0
        )
        predictor = np.column_stack(
            [
                predictor_all[moving],
                direction_predictor_all[moving],
            ]
        )
        target = (delta_price_all[moving] > 0.0).astype(np.float64)
        minimum_events = int(self.config.get("minimum_calibration_events", 20))
        if minimum_events < 15:
            raise ValueError("minimum_calibration_events must be at least 15")
        if len(target) < minimum_events:
            raise ValueError(
                f"calibration requires at least {minimum_events} moving events"
            )

        validation_fraction = float(
            self.config.get("calibration_validation_fraction", 0.2)
        )
        if not 0.1 <= validation_fraction <= 0.4:
            raise ValueError(
                "calibration_validation_fraction must be between 0.1 and 0.4"
            )
        validation_count = max(5, int(np.ceil(len(target) * validation_fraction)))
        validation_count = min(validation_count, len(target) - 10)
        train_count = len(target) - validation_count
        train_x, validation_x = predictor[:train_count], predictor[train_count:]
        train_y, validation_y = target[:train_count], target[train_count:]

        regularization_grid = tuple(
            float(value)
            for value in self.config.get(
                "calibration_regularization_grid",
                (1e-4, 1e-3, 1e-2, 5e-2, 1e-1),
            )
        )
        if not regularization_grid or any(
            value <= 0.0 or not np.isfinite(value)
            for value in regularization_grid
        ):
            raise ValueError(
                "calibration_regularization_grid must contain positive values"
            )

        coherence = float(np.exp(-self.gamma))
        prior_mean = float(train_y.mean())
        scaled_prior = np.clip(
            (2.0 * prior_mean - 1.0) / max(coherence, 1e-8),
            -0.95,
            0.95,
        )
        initial_bias = float(np.arctanh(scaled_prior))
        train_obi = train_x[:, 0]
        if np.var(train_obi) > 1e-15:
            probability_slope = float(
                (train_obi - train_obi.mean())
                @ (train_y - train_y.mean())
                / (
                    (train_obi - train_obi.mean())
                    @ (train_obi - train_obi.mean())
                )
            )
            initial_alpha = float(
                np.clip(
                    2.0 * max(probability_slope, 0.0) / max(coherence, 1e-8),
                    0.0,
                    5.0,
                )
            )
        else:
            initial_alpha = 0.0
        initial_parameters = np.array(
            [initial_bias, initial_alpha, 0.0],
            dtype=np.float64,
        )

        candidates: list[dict[str, Any]] = []
        for regularization in sorted(set(regularization_grid)):
            result = minimize(
                self._calibration_objective,
                x0=initial_parameters,
                args=(train_x, train_y, coherence, regularization),
                method="L-BFGS-B",
                bounds=((-3.0, 3.0), (0.0, 5.0), (-5.0, 5.0)),
            )
            if not result.success or not np.isfinite(result.fun):
                continue
            bias, alpha, alpha_direction = (
                float(result.x[0]),
                float(result.x[1]),
                float(result.x[2]),
            )
            validation_probability = self._direction_probability(
                validation_x[:, 0],
                bias=bias,
                alpha=alpha,
                tick_direction=validation_x[:, 1],
                alpha_direction=alpha_direction,
                coherence=coherence,
            )
            block_losses = [
                self._log_loss(validation_probability[index], validation_y[index])
                for index in np.array_split(
                    np.arange(len(validation_y)),
                    min(4, len(validation_y)),
                )
                if len(index)
            ]
            candidates.append(
                {
                    "regularization": regularization,
                    "bias": bias,
                    "alpha": alpha,
                    "alpha_direction": alpha_direction,
                    "validation_log_loss": float(np.mean(block_losses)),
                    "validation_log_loss_se": (
                        float(
                            np.std(block_losses, ddof=1)
                            / np.sqrt(len(block_losses))
                        )
                        if len(block_losses) > 1
                        else 0.0
                    ),
                    "validation_brier": float(
                        np.mean((validation_probability - validation_y) ** 2)
                    ),
                }
            )
        if not candidates:
            raise RuntimeError("all calibration optimizations failed")

        best = min(candidates, key=lambda item: item["validation_log_loss"])
        threshold = (
            best["validation_log_loss"] + best["validation_log_loss_se"]
        )
        selected = max(
            (
                candidate
                for candidate in candidates
                if candidate["validation_log_loss"] <= threshold
            ),
            key=lambda item: item["regularization"],
        )
        neutral_validation_probability = np.full(len(validation_y), 0.5)
        neutral_validation_log_loss = self._log_loss(
            neutral_validation_probability,
            validation_y,
        )
        train_prior_validation_probability = np.full(
            len(validation_y),
            prior_mean,
        )
        train_prior_validation_log_loss = self._log_loss(
            train_prior_validation_probability,
            validation_y,
        )

        linear_design = np.column_stack(
            [np.ones(len(train_x), dtype=np.float64), train_x[:, 0]]
        )
        linear_coefficients = np.linalg.lstsq(
            linear_design,
            train_y,
            rcond=None,
        )[0]
        linear_validation_probability = np.clip(
            linear_coefficients[0]
            + linear_coefficients[1] * validation_x[:, 0],
            0.0,
            1.0,
        )
        linear_validation_log_loss = self._log_loss(
            linear_validation_probability,
            validation_y,
        )
        market_design = np.column_stack(
            [np.ones(len(train_x), dtype=np.float64), train_x]
        )
        linear_market_coefficients = np.linalg.lstsq(
            market_design,
            train_y,
            rcond=None,
        )[0]
        linear_market_validation_probability = np.clip(
            np.column_stack(
                [
                    np.ones(len(validation_x), dtype=np.float64),
                    validation_x,
                ]
            )
            @ linear_market_coefficients,
            0.0,
            1.0,
        )
        linear_market_validation_log_loss = self._log_loss(
            linear_market_validation_probability,
            validation_y,
        )

        if selected["validation_log_loss"] >= neutral_validation_log_loss:
            self.obi_bias = 0.0
            self.alpha_obi = 0.0
            self.alpha_direction = 0.0
            calibration_status = "rejected_to_neutral"
        else:
            # Keep validation strictly selection-only. A later test set may
            # therefore score exactly the model that validation selected.
            self.obi_bias = float(selected["bias"])
            self.alpha_obi = float(selected["alpha"])
            self.alpha_direction = float(selected["alpha_direction"])
            calibration_status = (
                "accepted_beats_linear_market"
                if selected["validation_log_loss"]
                < linear_market_validation_log_loss
                else "accepted_below_linear_market"
            )

        final_probability = self._direction_probability(
            predictor[:, 0],
            bias=self.obi_bias,
            alpha=self.alpha_obi,
            tick_direction=predictor[:, 1],
            alpha_direction=self.alpha_direction,
            coherence=coherence,
        )
        final_log_loss = float(self._log_loss(final_probability, target))
        n_obs = len(target)
        k_params = 3  # bias, alpha_obi, alpha_direction
        final_nll = final_log_loss * n_obs
        
        aic = 2 * k_params + 2 * final_nll
        bic = k_params * np.log(n_obs) + 2 * final_nll

        parameters: dict[str, Any] = {
            "gamma": self.gamma,
            "gamma_estimate": gamma_estimate,
            "gamma_base": self.gamma_base,
            "rho_1": rho_1,
            "alpha_obi": self.alpha_obi,
            "alpha_direction": self.alpha_direction,
            "obi_bias": self.obi_bias,
            "obi_slope_ticks": slope_ticks,
            "tick_intercept": intercept_ticks,
            "tick_size": tick_size,
            "obi_direction_supported": self.alpha_obi > 0.0,
            "tick_direction_supported": abs(self.alpha_direction) > 1e-12,
            "obi_variance": predictor_variance,
            "direction_pairs": int(len(next_direction)),
            "moving_events": int(len(target)),
            "movement_probability": self.movement_probability,
            "calibration_rows": int(len(self.tick_data)),
            "calibration_method": (
                "regularized_market_coin_chronological_validation"
            ),
            "calibration_status": calibration_status,
            "calibration_train_events": int(len(train_y)),
            "calibration_validation_events": int(len(validation_y)),
            "structural_fit_events": int(len(train_y)),
            "final_refit_includes_validation": False,
            "events_per_structural_parameter": float(len(train_y) / 3.0),
            "low_sample_warning": bool(len(train_y) < 150),
            "selected_regularization": selected["regularization"],
            "validation_log_loss": selected["validation_log_loss"],
            "validation_brier": selected["validation_brier"],
            "neutral_validation_log_loss": neutral_validation_log_loss,
            "train_prior_validation_log_loss": (
                train_prior_validation_log_loss
            ),
            "linear_obi_validation_log_loss": linear_validation_log_loss,
            "linear_market_validation_log_loss": (
                linear_market_validation_log_loss
            ),
            "validation_beats_neutral": (
                selected["validation_log_loss"]
                < neutral_validation_log_loss
            ),
            "validation_beats_linear_obi": (
                selected["validation_log_loss"]
                < linear_validation_log_loss
            ),
            "validation_beats_linear_market": (
                selected["validation_log_loss"]
                < linear_market_validation_log_loss
            ),
            "final_log_loss": final_log_loss,
            "aic": aic,
            "bic": bic,
            "regularization_candidates": candidates,
            "coin_type": self.coin_type,
            "decoherence_channel": "basis_dephasing_exp_minus_gamma",
        }
        if not all(
            np.isfinite(parameters[key])
            for key in (
                "gamma",
                "gamma_estimate",
                "rho_1",
                "alpha_obi",
                "alpha_direction",
                "obi_bias",
                "obi_slope_ticks",
                "tick_intercept",
                "tick_size",
                "obi_variance",
            )
        ):
            raise RuntimeError("calibration produced non-finite parameters")

        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(parameters, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        self.calibrated_parameters = parameters
        self.calibrated = True
        return parameters.copy()

    def calibrate_two_stage(
        self,
        output_path: str | Path | None = "results/calibrated_params.json",
        *,
        warmup_fraction: float = 0.4,
    ) -> dict[str, Any]:
        """Fit structural parameters on warmup, then update only the bias."""
        if not 0.3 <= warmup_fraction <= 0.8:
            raise ValueError("warmup_fraction must be between 0.3 and 0.8")
        warmup_rows = int(len(self.tick_data) * warmup_fraction)
        if warmup_rows < 20 or len(self.tick_data) - warmup_rows < 10:
            raise ValueError("two-stage calibration requires a larger dataset")

        warmup_model = MarketQRW(
            self.tick_data.iloc[:warmup_rows].copy(),
            self.config,
        )
        structural = warmup_model.calibrate(None)
        self.gamma = float(structural["gamma"])
        self.alpha_obi = float(structural["alpha_obi"])
        self.alpha_direction = float(structural["alpha_direction"])
        self.obi_bias = float(structural["obi_bias"])
        bias_data = self.tick_data.iloc[warmup_rows:].copy().reset_index(
            drop=True
        )
        bias_model = MarketQRW(
            bias_data,
            {
                **self.config,
                "gamma_base": self.gamma,
                "alpha_obi": self.alpha_obi,
                "alpha_direction": self.alpha_direction,
                "obi_bias": self.obi_bias,
            },
        )
        bias_model.gamma = self.gamma
        bias_update = bias_model.calibrate_bias(
            regularization=float(
                self.config.get("bias_regularization", 0.01)
            ),
            prior_bias=self.obi_bias,
            prior_strength=float(
                self.config.get("bias_prior_strength", 0.1)
            ),
        )
        self.obi_bias = float(bias_update["obi_bias"])
        price = self.tick_data["price"].to_numpy(dtype=np.float64)
        valid = np.ones(len(price) - 1, dtype=bool)
        if "segment_id" in self.tick_data:
            segment = self.tick_data["segment_id"].to_numpy(copy=False)
            valid &= segment[:-1] == segment[1:]
        self.movement_probability = float(
            np.mean(np.abs(np.diff(price)[valid]) > 1e-12)
        )

        parameters = structural.copy()
        parameters.update(
            {
                "gamma": self.gamma,
                "alpha_obi": self.alpha_obi,
                "alpha_direction": self.alpha_direction,
                "obi_bias": self.obi_bias,
                "movement_probability": self.movement_probability,
                "moving_events": int(bias_update["moving_events"]),
                "calibration_rows": int(len(self.tick_data)),
                "structural_calibration_rows": warmup_rows,
                "structural_moving_events": int(structural["moving_events"]),
                "bias_update_rows": int(len(bias_data)),
                "bias_update_start_row": warmup_rows,
                "bias_update_reuses_warmup": False,
                "bias_update_objective": float(bias_update["objective"]),
                "calibration_method": (
                    "two_stage_disjoint_fixed_structure_bias_update"
                ),
                "calibration_status": "fixed_structure_bias_updated",
            }
        )
        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(parameters, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        self.calibrated_parameters = parameters
        self.calibrated = True
        return parameters.copy()

    def calibrate_bias(
        self,
        *,
        regularization: float = 0.01,
        prior_bias: float | None = None,
        prior_strength: float = 0.0,
    ) -> dict[str, float | int | str]:
        """Update only the regime intercept while structural parameters stay fixed."""
        if regularization < 0.0 or not np.isfinite(regularization):
            raise ValueError("regularization must be finite and non-negative")
        if prior_strength < 0.0 or not np.isfinite(prior_strength):
            raise ValueError("prior_strength must be finite and non-negative")
        anchor = self.obi_bias if prior_bias is None else float(prior_bias)
        if not np.isfinite(anchor):
            raise ValueError("prior_bias must be finite")

        price = self.tick_data["price"].to_numpy(dtype=np.float64)
        obi = self.tick_data["obi"].to_numpy(dtype=np.float64)
        direction = self.tick_data["tick_direction"].to_numpy(dtype=np.float64)
        delta = np.diff(price)
        valid = np.isfinite(delta) & (np.abs(delta) > 1e-12)
        if "segment_id" in self.tick_data:
            segment = self.tick_data["segment_id"].to_numpy(copy=False)
            valid &= segment[:-1] == segment[1:]
        if "obi_valid" in self.tick_data:
            valid &= self.tick_data["obi_valid"].astype(bool).to_numpy()[:-1]
        predictor_obi = obi[:-1][valid]
        predictor_direction = direction[:-1][valid]
        target = (delta[valid] > 0.0).astype(np.float64)
        if len(target) < 10:
            raise ValueError("bias calibration requires at least 10 moving events")

        coherence = float(np.exp(-self.gamma))

        def objective(value: np.ndarray) -> float:
            bias = float(value[0])
            probability = self._direction_probability(
                predictor_obi,
                bias=bias,
                alpha=self.alpha_obi,
                tick_direction=predictor_direction,
                alpha_direction=self.alpha_direction,
                coherence=coherence,
            )
            penalty = regularization * bias**2
            penalty += prior_strength * (bias - anchor) ** 2
            return self._log_loss(probability, target) + penalty

        result = minimize(
            objective,
            x0=np.array([anchor], dtype=np.float64),
            method="L-BFGS-B",
            bounds=((-3.0, 3.0),),
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"bias calibration failed: {result.message}")
        self.obi_bias = float(result.x[0])
        self.calibrated = True
        return {
            "obi_bias": self.obi_bias,
            "alpha_obi": self.alpha_obi,
            "alpha_direction": self.alpha_direction,
            "gamma": self.gamma,
            "moving_events": int(len(target)),
            "calibration_method": "fixed_structure_bias_update",
            "objective": float(result.fun),
        }

    @staticmethod
    def _direction_probability(
        obi: np.ndarray,
        *,
        bias: float,
        alpha: float,
        tick_direction: np.ndarray | float = 0.0,
        alpha_direction: float = 0.0,
        coherence: float,
    ) -> np.ndarray:
        signal = np.tanh(
            bias
            + alpha * np.clip(obi, -1.0, 1.0)
            + alpha_direction * np.clip(tick_direction, -1.0, 1.0)
        )
        return np.clip(0.5 + 0.5 * coherence * signal, 1e-12, 1.0 - 1e-12)

    @staticmethod
    def _log_loss(probability: np.ndarray, target: np.ndarray) -> float:
        clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
        return float(
            -np.mean(
                target * np.log(clipped)
                + (1.0 - target) * np.log(1.0 - clipped)
            )
        )

    @classmethod
    def _calibration_objective(
        cls,
        parameters: np.ndarray,
        predictor: np.ndarray,
        target: np.ndarray,
        coherence: float,
        regularization: float,
    ) -> float:
        bias, alpha, alpha_direction = (
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
        )
        probability = cls._direction_probability(
            predictor[:, 0],
            bias=bias,
            alpha=alpha,
            tick_direction=predictor[:, 1],
            alpha_direction=alpha_direction,
            coherence=coherence,
        )
        penalty = regularization * (
            bias**2 + alpha**2 + alpha_direction**2
        )
        return cls._log_loss(probability, target) + penalty

    def _resolve_tick_size(self, delta_price: np.ndarray) -> float:
        configured = self.config.get("tick_size")
        if configured is not None:
            tick_size = float(configured)
            if not np.isfinite(tick_size) or tick_size <= 0.0:
                raise ValueError("config['tick_size'] must be finite and positive")
            return tick_size

        positive = np.abs(delta_price[np.abs(delta_price) > 1e-12])
        if len(positive) == 0:
            raise ValueError("cannot infer tick_size from a constant price series")
        rounded = np.round(positive, decimals=12)
        tick_size = float(np.min(rounded[rounded > 0.0]))
        if not np.isfinite(tick_size) or tick_size <= 0.0:
            raise ValueError("could not infer a positive tick_size")
        return tick_size

    def _coin_for_obi(
        self,
        obi: float,
        tick_direction: float = 0.0,
    ) -> np.ndarray:
        if self.coin_type in {"obi", "obi_adaptive", "adaptive"}:
            return obi_adaptive_coin(
                obi,
                self.alpha_obi,
                bias=self.obi_bias,
                tick_direction=tick_direction,
                alpha_direction=self.alpha_direction,
            )
        if self.coin_type == "hadamard":
            return hadamard_coin()
        if self.coin_type in {"grover", "swap"}:
            return grover_coin()
        if self.coin_type == "biased":
            return biased_coin(np.pi / 4.0 + self.alpha_obi * obi)
        raise ValueError(f"unsupported coin_type: {self.coin_type!r}")

    def _one_step_right_probability(
        self,
        obi: float,
        tick_direction: float = 0.0,
    ) -> float:
        """Return the measured right probability for one refreshed market event."""
        initial = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
        rho = np.outer(initial, initial.conj())
        coherence = float(np.exp(-self.gamma))
        rho[0, 1] *= coherence
        rho[1, 0] *= coherence
        coin = self._coin_for_obi(obi, tick_direction)
        evolved = coin @ rho @ coin.conj().T
        return float(np.clip(evolved[0, 0].real, 0.0, 1.0))

    def simulate(self, T: int) -> pd.DataFrame:
        """Simulate ``T`` adaptive steps and return all position marginals."""
        if isinstance(T, bool) or not isinstance(T, int):
            raise TypeError("T must be an integer")
        if T < 1:
            raise ValueError("T must be positive")
        if T > len(self.tick_data):
            raise ValueError("T cannot exceed the number of market observations")
        radius = self.n_positions // 2
        if T > radius:
            raise ValueError(
                "n_positions is too small for T non-wrapping steps; "
                "require n_positions >= 2*T + 1"
            )

        engine = DensityMatrixQRW(self.n_positions)
        history = np.empty((T, self.n_positions), dtype=np.float64)
        obi_values = self.tick_data["obi"].to_numpy(dtype=np.float64, copy=False)
        direction_values = self.tick_data["tick_direction"].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        for time_index in range(T):
            coin = self._coin_for_obi(
                float(obi_values[time_index]),
                float(direction_values[time_index]),
            )
            history[time_index] = engine.step_with_decoherence(
                self.gamma,
                coin_matrix=coin,
            )

        self._probability_history = history
        self._last_simulation_steps = T
        return pd.DataFrame(
            {
                "t": np.repeat(np.arange(1, T + 1, dtype=np.int64), self.n_positions),
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
        """Sample measured local trajectories with shape ``(n_paths, T)``.

        For each market event, the latent coin is refreshed to the symmetric
        initial state, dephased by ``gamma``, transformed by the adaptive coin,
        shifted, and measured. Calibration may add zero increments to reproduce
        the empirical probability that an event leaves the price unchanged.
        This event-kernel protocol is intentionally distinct from the coherent,
        unmeasured density-matrix marginals returned by :meth:`simulate`.
        """
        if isinstance(n_paths, bool) or not isinstance(n_paths, int):
            raise TypeError("n_paths must be an integer")
        if n_paths < 1:
            raise ValueError("n_paths must be positive")
        if T is None:
            steps = (
                self._last_simulation_steps
                if self._last_simulation_steps > 0
                else int(self.config.get("simulation_steps", 1))
            )
        else:
            steps = T
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("T must be an integer")
        if steps < 1:
            raise ValueError("T must be positive")
        if steps > len(self.tick_data):
            raise ValueError("T cannot exceed the number of market observations")
        if steps > self.n_positions // 2:
            raise ValueError(
                "n_positions is too small for T local steps; "
                "require n_positions >= 2*T + 1"
            )
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        paths = np.empty((n_paths, steps), dtype=np.int64)
        positions = np.zeros(n_paths, dtype=np.int64)
        moving = (
            rng.random((n_paths, steps))
            < self.movement_probability
        )
        obi_values = self.tick_data["obi"].to_numpy(dtype=np.float64, copy=False)
        direction_values = self.tick_data["tick_direction"].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        for time_index in range(steps):
            probability_right = self._one_step_right_probability(
                float(obi_values[time_index]),
                float(direction_values[time_index]),
            )
            move_right = rng.random(n_paths) < probability_right
            positions += np.where(
                moving[:, time_index],
                np.where(move_right, 1, -1),
                0,
            )
            paths[:, time_index] = positions
        return paths

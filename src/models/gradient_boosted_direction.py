"""Chronological Histogram Gradient Boosting for directional probabilities."""

from __future__ import annotations

import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from src.evaluation.directional_baselines import _fit_logistic
from src.evaluation.ml_protocol import MLDirectionalProtocol


MODEL_FORMAT_VERSION = "hist_gradient_boosting_directional_v1"
TRAINING_FOLDS = ("train", "selection", "calibration")


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6
    )


def _brier(probability: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((_clip_probability(probability) - target) ** 2))


def _log_loss(probability: np.ndarray, target: np.ndarray) -> float:
    p = _clip_probability(probability)
    y = np.asarray(target, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """Serializable identity, Platt or isotonic probability mapping."""

    kind: str
    parameters: np.ndarray
    thresholds: np.ndarray
    probabilities: np.ndarray

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        raw = _clip_probability(raw_probability)
        if self.kind == "identity":
            return raw
        if self.kind == "platt":
            score = logit(raw)
            return _clip_probability(
                expit(self.parameters[0] + self.parameters[1] * score)
            )
        if self.kind == "isotonic":
            return _clip_probability(
                np.interp(
                    raw,
                    self.thresholds,
                    self.probabilities,
                    left=self.probabilities[0],
                    right=self.probabilities[-1],
                )
            )
        raise RuntimeError(f"unsupported calibrator: {self.kind}")


@dataclass
class HistogramGradientBoostingDirectionalModel:
    """Fitted estimator plus a calibrator frozen before test access."""

    estimator: HistGradientBoostingClassifier
    calibrator: ProbabilityCalibrator
    feature_names: tuple[str, ...]
    selected_parameters: Mapping[str, int | float]
    protocol_version: str
    random_seed: int

    def _matrix(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            missing = sorted(set(self.feature_names).difference(values.columns))
            if missing:
                raise ValueError(f"prediction frame is missing features: {missing}")
            matrix = values.loc[:, self.feature_names].to_numpy(
                dtype=np.float64
            )
        else:
            matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected a matrix with {len(self.feature_names)} features"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("prediction features must be finite")
        return matrix

    def predict_uncalibrated(
        self, values: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        matrix = self._matrix(values)
        return _clip_probability(self.estimator.predict_proba(matrix)[:, 1])

    def predict_proba(
        self, values: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        return self.calibrator.predict(self.predict_uncalibrated(values))


def _parameter_candidates(
    settings: Mapping[str, Any],
) -> list[dict[str, int | float]]:
    grid = settings["grid"]
    fields = (
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
    )
    return [
        dict(zip(fields, values, strict=True))
        for values in itertools.product(*(grid[field] for field in fields))
    ]


def _fit_calibrator(
    kind: str,
    raw_probability: np.ndarray,
    target: np.ndarray,
) -> ProbabilityCalibrator:
    raw = _clip_probability(raw_probability)
    y = np.asarray(target, dtype=np.float64)
    if kind == "identity":
        return ProbabilityCalibrator(
            kind, np.empty(0), np.empty(0), np.empty(0)
        )
    if kind == "platt":
        parameters = _fit_logistic(
            logit(raw).reshape(-1, 1),
            y,
            regularization=1e-8,
        )
        return ProbabilityCalibrator(
            kind,
            np.asarray(parameters, dtype=np.float64),
            np.empty(0),
            np.empty(0),
        )
    if kind == "isotonic":
        fitted = IsotonicRegression(
            y_min=1e-6,
            y_max=1.0 - 1e-6,
            out_of_bounds="clip",
        ).fit(raw, y)
        return ProbabilityCalibrator(
            kind,
            np.empty(0),
            np.asarray(fitted.X_thresholds_, dtype=np.float64),
            np.asarray(fitted.y_thresholds_, dtype=np.float64),
        )
    raise ValueError(f"unsupported calibrator: {kind}")


def fit_probability_calibrator(
    kind: str,
    raw_probability: np.ndarray,
    target: np.ndarray,
) -> ProbabilityCalibrator:
    """Fit one registered calibrator on the dedicated calibration fold."""
    return _fit_calibrator(kind, raw_probability, target)


def _fold(
    frame: pd.DataFrame,
    name: str,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    selected = frame.loc[frame["fold"].eq(name)]
    if selected.empty:
        raise ValueError(f"training dataset has no {name} rows")
    matrix = selected.loc[:, feature_names].to_numpy(dtype=np.float64)
    target = selected["target_up"].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} features must be finite")
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError(f"{name} target must be binary")
    if len(np.unique(target)) != 2:
        raise ValueError(f"{name} target must contain both classes")
    return matrix, target


def train_hist_gradient_boosting(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    protocol: MLDirectionalProtocol,
) -> tuple[HistogramGradientBoostingDirectionalModel, dict[str, Any]]:
    """Fit, select and calibrate without accepting a test row."""
    required = {"fold", "target_up", *feature_names}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"training dataset is missing columns: {missing}")
    observed_folds = set(frame["fold"].astype(str).unique())
    unexpected = sorted(observed_folds.difference(TRAINING_FOLDS))
    if unexpected:
        raise ValueError(
            "Phase 2 cannot access non-training folds: "
            + ", ".join(unexpected)
        )
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")

    train_x, train_y = _fold(frame, "train", names)
    selection_x, selection_y = _fold(frame, "selection", names)
    calibration_x, calibration_y = _fold(frame, "calibration", names)
    settings = protocol.raw["models"]["phase_2_hist_gradient_boosting"]

    candidate_rows: list[dict[str, Any]] = []
    fitted_candidates: list[
        tuple[HistGradientBoostingClassifier, dict[str, int | float]]
    ] = []
    for parameters in _parameter_candidates(settings):
        estimator = HistGradientBoostingClassifier(
            loss=settings["loss"],
            early_stopping=settings["early_stopping"],
            max_bins=int(settings["max_bins"]),
            categorical_features=None,
            random_state=protocol.random_seed,
            **parameters,
        )
        estimator.fit(train_x, train_y)
        probability = _clip_probability(
            estimator.predict_proba(selection_x)[:, 1]
        )
        row = {
            "parameters": dict(parameters),
            "selection_brier": _brier(probability, selection_y),
            "selection_log_loss": _log_loss(probability, selection_y),
        }
        candidate_rows.append(row)
        fitted_candidates.append((estimator, dict(parameters)))

    selected_index = min(
        range(len(candidate_rows)),
        key=lambda index: (
            candidate_rows[index]["selection_brier"],
            candidate_rows[index]["selection_log_loss"],
            tuple(candidate_rows[index]["parameters"].values()),
        ),
    )
    estimator, selected_parameters = fitted_candidates[selected_index]
    raw_calibration = _clip_probability(
        estimator.predict_proba(calibration_x)[:, 1]
    )

    calibration_rows: list[dict[str, Any]] = []
    calibrators: list[ProbabilityCalibrator] = []
    for kind in settings["probability_calibrators"]:
        calibrator = _fit_calibrator(kind, raw_calibration, calibration_y)
        probability = calibrator.predict(raw_calibration)
        calibration_rows.append(
            {
                "kind": kind,
                "calibration_brier": _brier(probability, calibration_y),
                "calibration_log_loss": _log_loss(
                    probability, calibration_y
                ),
            }
        )
        calibrators.append(calibrator)
    calibration_index = min(
        range(len(calibration_rows)),
        key=lambda index: (
            calibration_rows[index]["calibration_brier"],
            calibration_rows[index]["calibration_log_loss"],
            index,
        ),
    )
    selected_calibrator = calibrators[calibration_index]
    model = HistogramGradientBoostingDirectionalModel(
        estimator=estimator,
        calibrator=selected_calibrator,
        feature_names=names,
        selected_parameters=selected_parameters,
        protocol_version=protocol.protocol_version,
        random_seed=protocol.random_seed,
    )
    diagnostics = {
        "kind": "histogram_gradient_boosting_training",
        "protocol_version": protocol.protocol_version,
        "random_seed": protocol.random_seed,
        "feature_names": list(names),
        "fold_rows": {
            "train": len(train_y),
            "selection": len(selection_y),
            "calibration": len(calibration_y),
        },
        "test_fold_read": False,
        "candidates": candidate_rows,
        "selected_parameters": dict(selected_parameters),
        "calibrators": calibration_rows,
        "selected_calibrator": selected_calibrator.kind,
    }
    return model, diagnostics


def save_hist_gradient_boosting_model(
    model: HistogramGradientBoostingDirectionalModel,
    path: str | Path,
) -> Path:
    """Persist a versioned local model artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(
            {"format_version": MODEL_FORMAT_VERSION, "model": model},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return destination


def load_hist_gradient_boosting_model(
    path: str | Path,
) -> HistogramGradientBoostingDirectionalModel:
    """Load a model produced by :func:`save_hist_gradient_boosting_model`."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != MODEL_FORMAT_VERSION
        or not isinstance(
            payload.get("model"),
            HistogramGradientBoostingDirectionalModel,
        )
    ):
        raise ValueError("unsupported Histogram Gradient Boosting artifact")
    return payload["model"]

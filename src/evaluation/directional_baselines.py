"""Strong causal directional baselines with one shared validation fold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


FEATURE_NAMES = (
    "obi",
    "tick_direction",
    "obi_change",
    "abs_obi",
    "log_trade_intensity",
)
MODEL_NAMES = (
    "QRW Directional Link",
    "Logistic L2 (5F)",
    "Logistic L2 + Pairwise",
    "Nonlinear Calibrated",
    "OrderFlow AR(5)",
    "Marked Hawkes Logit",
)


@dataclass(frozen=True)
class DirectionalEvents:
    """Five causal features and the next moving-price direction."""

    features: np.ndarray
    target: np.ndarray
    timestamp: np.ndarray

    def __len__(self) -> int:
        return len(self.target)


def directional_events(frame: pd.DataFrame) -> DirectionalEvents:
    """Create causal next-move events without crossing segment boundaries."""
    required = {"timestamp", "price", "tick_direction", "obi", "trade_intensity"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"directional baseline frame is missing: {missing}")
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if len(ordered) < 3:
        raise ValueError("at least three rows are required")

    obi = ordered["obi"].to_numpy(dtype=np.float64)
    direction = ordered["tick_direction"].to_numpy(dtype=np.float64)
    intensity = ordered["trade_intensity"].to_numpy(dtype=np.float64)
    obi_change = np.zeros(len(ordered), dtype=np.float64)
    obi_change[1:] = np.diff(obi)
    same_segment = np.ones(len(ordered) - 1, dtype=bool)
    if "segment_id" in ordered:
        segment = ordered["segment_id"].to_numpy(copy=False)
        same_segment = segment[:-1] == segment[1:]
        obi_change[1:][~same_segment] = 0.0

    raw = np.column_stack(
        [
            obi,
            direction,
            obi_change,
            np.abs(obi),
            np.log1p(np.maximum(intensity, 0.0)),
        ]
    )
    price_change = np.diff(ordered["price"].to_numpy(dtype=np.float64))
    valid = same_segment & (np.abs(price_change) > 1e-12)
    if "obi_valid" in ordered:
        valid &= ordered["obi_valid"].astype(bool).to_numpy()[:-1]
    features = raw[:-1][valid]
    target = (price_change[valid] > 0.0).astype(np.float64)
    timestamp = ordered["timestamp"].to_numpy()[1:][valid]
    if not np.isfinite(features).all():
        raise ValueError("directional features must be finite")
    return DirectionalEvents(features, target, timestamp)


def chronological_day_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.50,
    validation_fraction: float = 0.25,
) -> tuple[DirectionalEvents, DirectionalEvents, DirectionalEvents]:
    """Split one day by rows, then build events within each disjoint fold."""
    if train_fraction <= 0.0 or validation_fraction <= 0.0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 0.9:
        raise ValueError("at least 10% of rows must remain for test")
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(
        len(ordered) * (train_fraction + validation_fraction)
    )
    folds = (
        directional_events(ordered.iloc[:train_end].copy()),
        directional_events(ordered.iloc[train_end:validation_end].copy()),
        directional_events(ordered.iloc[validation_end:].copy()),
    )
    if min(map(len, folds)) < 20:
        raise ValueError("each daily fold requires at least 20 moving events")
    return folds


def _log_loss(probability: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(target, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def _fit_logistic(
    design: np.ndarray,
    target: np.ndarray,
    regularization: float,
) -> np.ndarray:
    matrix = np.column_stack([np.ones(len(design)), design])
    y = np.asarray(target, dtype=np.float64)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        score = matrix @ beta
        probability = expit(score)
        penalty = regularization * float(beta[1:] @ beta[1:])
        value = float(np.mean(np.logaddexp(0.0, score) - y * score) + penalty)
        gradient = matrix.T @ (probability - y) / len(y)
        gradient[1:] += 2.0 * regularization * beta[1:]
        return value, gradient

    result = minimize(
        objective,
        np.zeros(matrix.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _qrw_probability(
    normalized: np.ndarray,
    parameters: np.ndarray,
    gamma: float,
) -> np.ndarray:
    signal = parameters[0] + normalized[:, :4] @ parameters[1:5]
    intensity_effect = np.exp(
        np.clip(parameters[5] * normalized[:, 4], -5.0, 5.0)
    )
    coherence = np.exp(-gamma * intensity_effect)
    return np.clip(
        0.5 + 0.5 * coherence * np.tanh(signal),
        1e-12,
        1.0 - 1e-12,
    )


def _fit_qrw_link(
    normalized: np.ndarray,
    target: np.ndarray,
    regularization: float,
    gamma: float,
) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)

    def objective(parameters: np.ndarray) -> float:
        probability = _qrw_probability(normalized, parameters, gamma)
        return _log_loss(probability, y) + regularization * float(
            parameters[1:] @ parameters[1:]
        )

    result = minimize(
        objective,
        np.zeros(6, dtype=np.float64),
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
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"QRW directional fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _lagged_direction(direction: np.ndarray, lags: int = 5) -> np.ndarray:
    values = np.asarray(direction, dtype=np.float64)
    result = np.zeros((len(values), lags), dtype=np.float64)
    for lag in range(1, lags + 1):
        result[lag:, lag - 1] = values[:-lag]
    return result


def _normalized_event_time(timestamp: np.ndarray) -> np.ndarray:
    """Return monotone event time in units of the median positive spacing."""
    values = np.asarray(timestamp)
    if np.issubdtype(values.dtype, np.datetime64):
        numeric = values.astype("datetime64[ns]").astype(np.int64) / 1e9
    elif np.issubdtype(values.dtype, np.number):
        numeric = values.astype(np.float64)
    else:
        numeric = (
            pd.to_datetime(values, utc=True).astype("int64").to_numpy()
            / 1e9
        )
    differences = np.diff(numeric)
    positive = differences[differences > 0.0]
    if len(positive) == 0:
        raise ValueError("marked Hawkes events require increasing timestamps")
    scale = float(np.median(positive))
    return (numeric - numeric[0]) / scale


def _marked_hawkes_state(
    features: np.ndarray,
    timestamp: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Build causal positive/negative excitation with an exponential kernel."""
    if decay <= 0.0:
        raise ValueError("marked Hawkes decay must be positive")
    event_time = _normalized_event_time(timestamp)
    state = np.zeros((len(features), 2), dtype=np.float64)
    running = np.zeros(2, dtype=np.float64)
    direction = np.sign(features[:, 1])
    mark = np.sqrt(np.maximum(features[:, 4], 0.0))
    for index in range(len(features)):
        if index > 0:
            elapsed = max(event_time[index] - event_time[index - 1], 0.0)
            running *= np.exp(-decay * elapsed)
        if direction[index] > 0.0:
            running[0] += mark[index]
        elif direction[index] < 0.0:
            running[1] += mark[index]
        state[index] = running
    return state


def _pairwise(values: np.ndarray) -> np.ndarray:
    columns = [values]
    columns.extend(
        (values[:, left] * values[:, right])[:, None]
        for left in range(values.shape[1])
        for right in range(left + 1, values.shape[1])
    )
    return np.column_stack(columns)


def _isotonic_fit(
    score: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic PAVA calibrator on the validation fold."""
    order = np.argsort(score, kind="stable")
    x = np.asarray(score, dtype=np.float64)[order]
    y = np.asarray(target, dtype=np.float64)[order]
    blocks: list[list[float]] = []
    for value, label in zip(x, y, strict=True):
        blocks.append([value, value, label, 1.0])
        while len(blocks) >= 2:
            previous_mean = blocks[-2][2] / blocks[-2][3]
            current_mean = blocks[-1][2] / blocks[-1][3]
            if previous_mean <= current_mean:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                [left[0], right[1], left[2] + right[2], left[3] + right[3]]
            )
    threshold = np.asarray([block[1] for block in blocks], dtype=np.float64)
    calibrated = np.asarray(
        [block[2] / block[3] for block in blocks], dtype=np.float64
    )
    return threshold, np.clip(calibrated, 1e-6, 1.0 - 1e-6)


@dataclass
class DirectionalBaselineModel:
    name: str
    kind: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    beta: np.ndarray
    regularization: float
    validation_log_loss: float
    design_feature_count: int
    decay: float | None = None
    gamma: float | None = None
    calibration_threshold: np.ndarray | None = None
    calibration_probability: np.ndarray | None = None
    history_features: np.ndarray | None = None
    history_timestamp: np.ndarray | None = None

    def _design(
        self,
        features: np.ndarray,
        timestamp: np.ndarray | None = None,
    ) -> np.ndarray:
        normalized = (features - self.feature_mean) / self.feature_scale
        if self.kind == "main":
            return normalized
        if self.kind == "pairwise":
            return _pairwise(normalized)
        if self.kind == "nonlinear":
            return np.column_stack([normalized, normalized**2])
        if self.kind == "autoregressive":
            return np.column_stack(
                [normalized, _lagged_direction(features[:, 1])]
            )
        if self.kind == "marked_hawkes":
            if self.decay is None:
                raise RuntimeError("marked Hawkes decay is unavailable")
            if timestamp is None:
                raise ValueError("marked Hawkes design requires timestamps")
            return np.column_stack(
                [
                    normalized,
                    _marked_hawkes_state(features, timestamp, self.decay),
                ]
            )
        raise RuntimeError(f"unsupported baseline kind: {self.kind}")

    def predict(self, events: DirectionalEvents) -> np.ndarray:
        if self.kind == "qrw_link":
            if self.gamma is None:
                raise RuntimeError("QRW directional gamma is unavailable")
            normalized = (
                events.features - self.feature_mean
            ) / self.feature_scale
            return _qrw_probability(normalized, self.beta, self.gamma)
        if self.kind == "marked_hawkes" and self.history_features is not None:
            if self.history_timestamp is None:
                raise RuntimeError("marked Hawkes history timestamps unavailable")
            combined_features = np.vstack(
                [self.history_features, events.features]
            )
            combined_timestamp = np.concatenate(
                [self.history_timestamp, events.timestamp]
            )
            design = self._design(
                combined_features, combined_timestamp
            )[-len(events) :]
        else:
            design = self._design(events.features, events.timestamp)
        probability = expit(
            np.column_stack([np.ones(len(design)), design]) @ self.beta
        )
        if self.calibration_threshold is not None:
            indices = np.searchsorted(
                self.calibration_threshold, probability, side="left"
            )
            indices = np.clip(indices, 0, len(self.calibration_threshold) - 1)
            probability = self.calibration_probability[indices]
        return np.clip(probability, 1e-12, 1.0 - 1e-12)


def fit_directional_baselines(
    train: DirectionalEvents,
    validation: DirectionalEvents,
    *,
    regularization_grid: Iterable[float] = (1e-4, 1e-3, 1e-2, 1e-1),
    decay_grid: Iterable[float] = (0.50, 0.80, 0.95),
) -> dict[str, DirectionalBaselineModel]:
    """Tune every baseline on the same chronological validation events."""
    regularization_values = sorted({float(value) for value in regularization_grid})
    if not regularization_values or min(regularization_values) <= 0.0:
        raise ValueError("regularization_grid must contain positive values")
    mean = train.features.mean(axis=0)
    scale = train.features.std(axis=0)
    scale[scale < 1e-9] = 1.0

    specifications: tuple[tuple[str, str], ...] = (
        ("QRW Directional Link", "qrw_link"),
        ("Logistic L2 (5F)", "main"),
        ("Logistic L2 + Pairwise", "pairwise"),
        ("Nonlinear Calibrated", "nonlinear"),
        ("OrderFlow AR(5)", "autoregressive"),
        ("Marked Hawkes Logit", "marked_hawkes"),
    )
    fitted: dict[str, DirectionalBaselineModel] = {}
    direction = np.sign(train.features[:, 1])
    if len(direction) > 2 and np.std(direction[:-1]) > 0.0:
        rho = float(np.corrcoef(direction[:-1], direction[1:])[0, 1])
        if not np.isfinite(rho):
            rho = 0.0
    else:
        rho = 0.0
    qrw_gamma = float(-np.log(np.clip(abs(rho), 1e-8, 1.0)))
    for name, kind in specifications:
        if kind == "qrw_link":
            train_normalized = (train.features - mean) / scale
            validation_normalized = (validation.features - mean) / scale
            qrw_candidates: list[DirectionalBaselineModel] = []
            for regularization in regularization_values:
                beta = _fit_qrw_link(
                    train_normalized,
                    train.target,
                    regularization,
                    qrw_gamma,
                )
                probability = _qrw_probability(
                    validation_normalized, beta, qrw_gamma
                )
                qrw_candidates.append(
                    DirectionalBaselineModel(
                        name=name,
                        kind=kind,
                        feature_mean=mean.copy(),
                        feature_scale=scale.copy(),
                        beta=beta,
                        regularization=regularization,
                        validation_log_loss=_log_loss(
                            probability, validation.target
                        ),
                        design_feature_count=5,
                        gamma=qrw_gamma,
                    )
                )
            fitted[name] = min(
                qrw_candidates,
                key=lambda model: (
                    model.validation_log_loss,
                    -model.regularization,
                ),
            )
            continue
        decays = list(decay_grid) if kind == "marked_hawkes" else [None]
        candidates: list[DirectionalBaselineModel] = []
        for decay in decays:
            prototype = DirectionalBaselineModel(
                name=name,
                kind=kind,
                feature_mean=mean,
                feature_scale=scale,
                beta=np.empty(0),
                regularization=0.0,
                validation_log_loss=float("inf"),
                design_feature_count=0,
                decay=None if decay is None else float(decay),
            )
            train_design = prototype._design(
                train.features, train.timestamp
            )
            if kind == "marked_hawkes":
                combined_features = np.vstack(
                    [train.features, validation.features]
                )
                combined_timestamp = np.concatenate(
                    [train.timestamp, validation.timestamp]
                )
                validation_design = prototype._design(
                    combined_features, combined_timestamp
                )[-len(validation) :]
            else:
                validation_design = prototype._design(
                    validation.features, validation.timestamp
                )
            for regularization in regularization_values:
                beta = _fit_logistic(
                    train_design, train.target, regularization
                )
                probability = expit(
                    np.column_stack(
                        [np.ones(len(validation_design)), validation_design]
                    )
                    @ beta
                )
                candidates.append(
                    DirectionalBaselineModel(
                        name=name,
                        kind=kind,
                        feature_mean=mean.copy(),
                        feature_scale=scale.copy(),
                        beta=beta,
                        regularization=regularization,
                        validation_log_loss=_log_loss(
                            probability, validation.target
                        ),
                        design_feature_count=train_design.shape[1],
                        decay=None if decay is None else float(decay),
                    )
                )
        selected = min(
            candidates,
            key=lambda model: (
                model.validation_log_loss,
                -model.regularization,
                -(model.decay or 0.0),
            ),
        )
        if kind == "nonlinear":
            uncalibrated = expit(
                np.column_stack(
                    [
                        np.ones(len(validation)),
                        selected._design(
                            validation.features, validation.timestamp
                        ),
                    ]
                )
                @ selected.beta
            )
            threshold, probability = _isotonic_fit(
                uncalibrated, validation.target
            )
            selected.calibration_threshold = threshold
            selected.calibration_probability = probability
        if kind == "marked_hawkes":
            selected.history_features = np.vstack(
                [train.features, validation.features]
            )
            selected.history_timestamp = np.concatenate(
                [train.timestamp, validation.timestamp]
            )
        fitted[name] = selected
    return fitted


def evaluate_directional_baselines(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit/tune/score all baselines on one shared daily split."""
    train, validation, test = chronological_day_split(frame)
    models = fit_directional_baselines(train, validation)
    return score_directional_baselines(
        models,
        test,
        train_events=len(train),
        validation_events=len(validation),
    )


def score_directional_baselines(
    models: dict[str, DirectionalBaselineModel],
    test: DirectionalEvents,
    *,
    train_events: int,
    validation_events: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score already-selected models on one untouched common test day."""
    if tuple(models) != MODEL_NAMES:
        raise ValueError("directional model set does not match preregistration")
    summary_rows: list[dict[str, float | int | str]] = []
    prediction = pd.DataFrame(
        {"timestamp": test.timestamp, "target_up": test.target.astype(int)}
    )
    for name in MODEL_NAMES:
        model = models[name]
        probability = model.predict(test)
        prediction[name] = probability
        summary_rows.append(
            {
                "model": name,
                "sample_size": len(test),
                "brier": float(np.mean((probability - test.target) ** 2)),
                "log_loss": _log_loss(probability, test.target),
                "accuracy": float(
                    np.mean((probability >= 0.5) == test.target)
                ),
                "selected_regularization": model.regularization,
                "selected_decay": (
                    float("nan") if model.decay is None else model.decay
                ),
                "validation_log_loss": model.validation_log_loss,
                "design_feature_count": model.design_feature_count,
            }
        )
    diagnostics = {
        "feature_names": list(FEATURE_NAMES),
        "train_events": int(train_events),
        "validation_events": int(validation_events),
        "test_events": len(test),
        "same_test_sample_for_all_models": True,
        "selection_fold": "shared_chronological_validation",
        "marked_hawkes_semantics": (
            "conditional_mark_logit_with_exponential_interevent_kernel; "
            "not_full_arrival_process_likelihood"
        ),
    }
    return pd.DataFrame(summary_rows), prediction, diagnostics

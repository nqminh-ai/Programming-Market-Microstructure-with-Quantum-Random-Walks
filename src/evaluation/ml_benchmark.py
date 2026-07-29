"""Common-sample Phase 3 scoring and paired UTC-day uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit

from src.evaluation.directional_baselines import (
    FEATURE_NAMES,
    DirectionalEvents,
    _fit_logistic,
    fit_directional_baselines,
)
from src.evaluation.ml_protocol import MLDirectionalProtocol
from src.models.gradient_boosted_direction import (
    HistogramGradientBoostingDirectionalModel,
    ProbabilityCalibrator,
    fit_probability_calibrator,
)


HGB_NAME = "Histogram Gradient Boosting"
MAJORITY_NAME = "Majority Probability"
WINDOWED_QRW_NAME = "Windowed-QRW (density matrix)"
RAW_ORDERFLOW_NAME = "OrderFlow AR(5)"
ORDERFLOW_LAG_FEATURES = tuple(
    f"tick_direction_lag_{lag}" for lag in range(1, 6)
)


@dataclass(frozen=True)
class MLBenchmarkResult:
    """Phase 3 tables and machine-readable diagnostics."""

    summary: pd.DataFrame
    predictions: pd.DataFrame
    comparisons: Mapping[str, Mapping[str, Any]]
    diagnostics: Mapping[str, Any]


def _clip(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def _brier(probability: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((_clip(probability) - target) ** 2))


def _log_loss(probability: np.ndarray, target: np.ndarray) -> float:
    p = _clip(probability)
    y = np.asarray(target, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def expected_calibration_error(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Equal-width expected calibration error."""
    if bins <= 1:
        raise ValueError("bins must be greater than one")
    p = _clip(probability)
    y = np.asarray(target, dtype=np.float64)
    indices = np.minimum((p * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = indices == index
        if selected.any():
            error += float(selected.mean()) * abs(
                float(p[selected].mean()) - float(y[selected].mean())
            )
    return float(error)


def _events(frame: pd.DataFrame) -> DirectionalEvents:
    return DirectionalEvents(
        features=frame.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64),
        target=frame["target_up"].to_numpy(dtype=np.float64),
        timestamp=frame["timestamp"].to_numpy(),
    )


def _concatenate_events(
    first: DirectionalEvents, second: DirectionalEvents
) -> DirectionalEvents:
    return DirectionalEvents(
        features=np.vstack([first.features, second.features]),
        target=np.concatenate([first.target, second.target]),
        timestamp=np.concatenate([first.timestamp, second.timestamp]),
    )


def _select_calibrator(
    raw_probability: np.ndarray,
    target: np.ndarray,
    kinds: Sequence[str],
) -> tuple[ProbabilityCalibrator, list[dict[str, float | str]]]:
    candidates: list[ProbabilityCalibrator] = []
    rows: list[dict[str, float | str]] = []
    for kind in kinds:
        calibrator = fit_probability_calibrator(
            kind, raw_probability, target
        )
        probability = calibrator.predict(raw_probability)
        candidates.append(calibrator)
        rows.append(
            {
                "kind": kind,
                "brier": _brier(probability, target),
                "log_loss": _log_loss(probability, target),
            }
        )
    index = min(
        range(len(rows)),
        key=lambda position: (
            rows[position]["brier"],
            rows[position]["log_loss"],
            position,
        ),
    )
    return candidates[index], rows


def _fit_raw_orderflow(
    train: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    columns = (*FEATURE_NAMES, *ORDERFLOW_LAG_FEATURES)
    train_x = train.loc[:, columns].to_numpy(dtype=np.float64)
    selection_x = selection.loc[:, columns].to_numpy(dtype=np.float64)
    train_y = train["target_up"].to_numpy(dtype=np.float64)
    selection_y = selection["target_up"].to_numpy(dtype=np.float64)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    train_normalized = (train_x - mean) / scale
    selection_normalized = (selection_x - mean) / scale
    rows: list[dict[str, Any]] = []
    parameters: list[np.ndarray] = []
    for regularization in (1e-4, 1e-3, 1e-2, 1e-1):
        beta = _fit_logistic(
            train_normalized, train_y, regularization
        )
        probability = _clip(
            expit(
                np.column_stack(
                    [
                        np.ones(len(selection_normalized)),
                        selection_normalized,
                    ]
                )
                @ beta
            )
        )
        parameters.append(beta)
        rows.append(
            {
                "regularization": regularization,
                "selection_brier": _brier(probability, selection_y),
                "selection_log_loss": _log_loss(probability, selection_y),
            }
        )
    index = min(
        range(len(rows)),
        key=lambda position: (
            rows[position]["selection_brier"],
            rows[position]["selection_log_loss"],
            -rows[position]["regularization"],
        ),
    )
    return {
        "columns": columns,
        "mean": mean,
        "scale": scale,
        "beta": parameters[index],
        "regularization": rows[index]["regularization"],
    }, rows


def _predict_raw_orderflow(
    model: Mapping[str, Any], frame: pd.DataFrame
) -> np.ndarray:
    matrix = frame.loc[:, model["columns"]].to_numpy(dtype=np.float64)
    normalized = (matrix - model["mean"]) / model["scale"]
    return _clip(
        expit(
            np.column_stack([np.ones(len(normalized)), normalized])
            @ model["beta"]
        )
    )


def paired_utc_day_bootstrap(
    challenger_loss: np.ndarray,
    baseline_loss: np.ndarray,
    utc_day: np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int | bool]:
    """Cluster-resample complete UTC days for a paired loss difference."""
    challenger = np.asarray(challenger_loss, dtype=np.float64)
    baseline = np.asarray(baseline_loss, dtype=np.float64)
    day = np.asarray(utc_day)
    if challenger.shape != baseline.shape or challenger.shape != day.shape:
        raise ValueError("paired bootstrap inputs must share one shape")
    if resamples <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid bootstrap settings")
    unique_days = np.unique(day)
    if len(unique_days) < 2:
        raise ValueError("paired UTC-day bootstrap requires at least two days")
    difference = challenger - baseline
    clusters = [difference[day == value] for value in unique_days]
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        total = sum(float(clusters[position].sum()) for position in chosen)
        count = sum(len(clusters[position]) for position in chosen)
        samples[index] = total / count
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    point = float(difference.mean())
    return {
        "mean_challenger_minus_baseline": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence_level": float(confidence_level),
        "resamples": int(resamples),
        "utc_days": len(unique_days),
        "challenger_significantly_better": bool(high < 0.0),
        "challenger_significantly_worse": bool(low > 0.0),
    }


def run_ml_common_sample_benchmark(
    frame: pd.DataFrame,
    hgb_model: HistogramGradientBoostingDirectionalModel,
    protocol: MLDirectionalProtocol,
    *,
    bootstrap_resamples: int | None = None,
) -> MLBenchmarkResult:
    """Open the supplied test rows and compare every available causal model."""
    required = {
        "timestamp",
        "utc_day",
        "anchor_row",
        "fold",
        "target_up",
        *FEATURE_NAMES,
        *ORDERFLOW_LAG_FEATURES,
        *hgb_model.feature_names,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"benchmark dataset is missing columns: {missing}")
    if hgb_model.protocol_version != protocol.protocol_version:
        raise ValueError("HGB model protocol does not match benchmark protocol")
    if frame["anchor_row"].duplicated().any():
        raise ValueError("benchmark anchor rows must be unique")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("benchmark rows must be chronological")
    folds = {
        name: frame.loc[frame["fold"].eq(name)].reset_index(drop=True)
        for name in ("train", "selection", "calibration", "test")
    }
    if any(value.empty for value in folds.values()):
        raise ValueError("benchmark requires all four chronological folds")
    target = folds["test"]["target_up"].to_numpy(dtype=np.float64)
    if len(np.unique(target)) != 2:
        raise ValueError("test target must contain both classes")

    train_events = _events(folds["train"])
    selection_events = _events(folds["selection"])
    calibration_events = _events(folds["calibration"])
    test_events = _events(folds["test"])
    combined_calibration_test = _concatenate_events(
        calibration_events, test_events
    )
    reference_models = fit_directional_baselines(
        train_events, selection_events
    )
    calibrator_kinds = protocol.raw["models"][
        "phase_2_hist_gradient_boosting"
    ]["probability_calibrators"]

    predictions: dict[str, np.ndarray] = {
        MAJORITY_NAME: np.repeat(
            float(train_events.target.mean()), len(test_events)
        ),
        HGB_NAME: hgb_model.predict_proba(folds["test"]),
    }
    calibration_diagnostics: dict[str, Any] = {}
    baseline_selection: dict[str, Any] = {}
    for name, model in reference_models.items():
        if name == RAW_ORDERFLOW_NAME:
            continue
        raw_calibration = model.predict(calibration_events)
        calibrator, rows = _select_calibrator(
            raw_calibration,
            calibration_events.target,
            calibrator_kinds,
        )
        if model.kind in {"autoregressive", "marked_hawkes"}:
            raw_test = model.predict(combined_calibration_test)[
                -len(test_events) :
            ]
        else:
            raw_test = model.predict(test_events)
        predictions[name] = calibrator.predict(raw_test)
        calibration_diagnostics[name] = {
            "selected": calibrator.kind,
            "candidates": rows,
        }
        baseline_selection[name] = {
            "regularization": model.regularization,
            "validation_log_loss": model.validation_log_loss,
            "design_feature_count": model.design_feature_count,
        }

    orderflow_model, orderflow_selection = _fit_raw_orderflow(
        folds["train"], folds["selection"]
    )
    orderflow_calibration = _predict_raw_orderflow(
        orderflow_model, folds["calibration"]
    )
    orderflow_calibrator, orderflow_calibration_rows = _select_calibrator(
        orderflow_calibration,
        folds["calibration"]["target_up"].to_numpy(dtype=np.float64),
        calibrator_kinds,
    )
    predictions[RAW_ORDERFLOW_NAME] = orderflow_calibrator.predict(
        _predict_raw_orderflow(orderflow_model, folds["test"])
    )
    calibration_diagnostics[RAW_ORDERFLOW_NAME] = {
        "selected": orderflow_calibrator.kind,
        "candidates": orderflow_calibration_rows,
    }
    baseline_selection[RAW_ORDERFLOW_NAME] = {
        "selected_regularization": orderflow_model["regularization"],
        "candidates": orderflow_selection,
        "lag_semantics": "raw_tick_direction_lags_1_through_5",
    }

    test_prediction = pd.DataFrame(
        {
            "timestamp": folds["test"]["timestamp"].to_numpy(),
            "utc_day": folds["test"]["utc_day"].astype(str).to_numpy(),
            "anchor_row": folds["test"]["anchor_row"].to_numpy(),
            "target_up": target.astype(np.int8),
        }
    )
    rows: list[dict[str, float | int | str]] = []
    for name, probability in predictions.items():
        p = _clip(probability)
        if len(p) != len(target) or not np.isfinite(p).all():
            raise RuntimeError(f"invalid common-sample prediction: {name}")
        test_prediction[name] = p
        rows.append(
            {
                "model": name,
                "sample_size": len(target),
                "brier": _brier(p, target),
                "log_loss": _log_loss(p, target),
                "expected_calibration_error": expected_calibration_error(
                    p, target
                ),
                "accuracy": float(
                    np.mean((p >= 0.5) == target.astype(bool))
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["brier", "log_loss", "model"], kind="stable"
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))

    hgb_loss = (predictions[HGB_NAME] - target) ** 2
    settings = protocol.raw["metrics"]["uncertainty"]
    resamples = (
        int(bootstrap_resamples)
        if bootstrap_resamples is not None
        else int(settings["resamples"])
    )
    comparisons = {
        name: paired_utc_day_bootstrap(
            hgb_loss,
            (probability - target) ** 2,
            test_prediction["utc_day"].to_numpy(),
            resamples=resamples,
            confidence_level=float(settings["confidence_level"]),
            seed=protocol.random_seed,
        )
        for name, probability in predictions.items()
        if name != HGB_NAME
    }
    diagnostics = {
        "protocol_version": protocol.protocol_version,
        "test_fold_opened": True,
        "test_rows": len(target),
        "test_days": sorted(test_prediction["utc_day"].unique().tolist()),
        "same_test_sample_for_all_models": True,
        "primary_metric": "brier",
        "challenger": HGB_NAME,
        "calibration": calibration_diagnostics,
        "baseline_selection": baseline_selection,
        "registered_model_not_evaluated": {
            WINDOWED_QRW_NAME: (
                "no preregistered causal t_plus_h adapter; one-tick "
                "probabilities cannot be relabelled as horizon forecasts"
            )
        },
        "complete_registered_model_set": False,
    }
    return MLBenchmarkResult(
        summary=summary,
        predictions=test_prediction,
        comparisons=comparisons,
        diagnostics=diagnostics,
    )

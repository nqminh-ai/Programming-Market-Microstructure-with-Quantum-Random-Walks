"""Test fixed-structure QRW predictive edge on chronological historical days."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qrw_market_sim import MarketQRW


def extract_events(
    path: Path,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Stream causal market events and deterministically subsample them."""
    if stride < 1:
        raise ValueError("stride must be positive")
    parquet = pq.ParquetFile(path)
    columns = [
        "price",
        "obi",
        "tick_direction",
        "segment_id",
        "obi_valid",
    ]
    obi_parts: list[np.ndarray] = []
    direction_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    moving_count = 0
    carry: dict[str, Any] | None = None

    for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
        values = {
            name: batch.column(index).to_numpy(zero_copy_only=False)
            for index, name in enumerate(columns)
        }
        if carry is not None:
            values = {
                name: np.concatenate([[carry[name]], array])
                for name, array in values.items()
            }
        price = values["price"].astype(np.float64, copy=False)
        obi = values["obi"].astype(np.float64, copy=False)
        direction = values["tick_direction"].astype(np.float64, copy=False)
        segment = values["segment_id"]
        obi_valid = values["obi_valid"].astype(bool, copy=False)
        delta = np.diff(price)
        valid = (
            (np.abs(delta) > 1e-12)
            & obi_valid[:-1]
            & (segment[:-1] == segment[1:])
        )
        moving = np.flatnonzero(valid)
        sequence = np.arange(moving_count, moving_count + len(moving))
        selected = moving[sequence % stride == 0]
        moving_count += len(moving)
        if len(selected):
            obi_parts.append(obi[selected])
            direction_parts.append(direction[selected])
            target_parts.append((delta[selected] > 0.0).astype(np.float64))
        carry = {name: array[-1] for name, array in values.items()}

    if not target_parts:
        raise ValueError(f"no moving events found in {path}")
    return (
        np.concatenate(obi_parts),
        np.concatenate(direction_parts),
        np.concatenate(target_parts),
        moving_count,
    )


def direction_autocorrelation(paths: list[Path]) -> float:
    """Compute exact lag-one direction correlation without crossing segments."""
    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    for path in paths:
        parquet = pq.ParquetFile(path)
        carry: tuple[float, Any] | None = None
        for batch in parquet.iter_batches(
            batch_size=500_000,
            columns=["tick_direction", "segment_id"],
        ):
            direction = batch.column(0).to_numpy(
                zero_copy_only=False
            ).astype(np.float64, copy=False)
            segment = batch.column(1).to_numpy(zero_copy_only=False)
            if carry is not None:
                direction = np.concatenate([[carry[0]], direction])
                segment = np.concatenate([[carry[1]], segment])
            same = segment[:-1] == segment[1:]
            x = direction[:-1][same]
            y = direction[1:][same]
            count += len(x)
            sum_x += float(x.sum())
            sum_y += float(y.sum())
            sum_xx += float(x @ x)
            sum_yy += float(y @ y)
            sum_xy += float(x @ y)
            carry = (float(direction[-1]), segment[-1])
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = sum_xx - sum_x**2 / count
    variance_y = sum_yy - sum_y**2 / count
    return float(covariance / np.sqrt(variance_x * variance_y))


def log_loss(probability: np.ndarray, target: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(
            target * np.log(clipped)
            + (1.0 - target) * np.log(1.0 - clipped)
        )
    )


def score(probability: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    return {
        "events": int(len(target)),
        "brier": float(np.mean((probability - target) ** 2)),
        "log_loss": log_loss(probability, target),
        "accuracy": float(np.mean((probability >= 0.5) == target)),
        "mean_probability": float(probability.mean()),
        "up_fraction": float(target.mean()),
    }


def qrw_probability(
    obi: np.ndarray,
    direction: np.ndarray,
    parameters: np.ndarray,
    coherence: float,
) -> np.ndarray:
    return MarketQRW._direction_probability(
        obi,
        bias=float(parameters[0]),
        alpha=float(parameters[1]),
        tick_direction=direction,
        alpha_direction=float(parameters[2]),
        coherence=coherence,
    )


def fit_qrw(
    obi: np.ndarray,
    direction: np.ndarray,
    target: np.ndarray,
    *,
    coherence: float,
    regularization: float,
    initial: np.ndarray,
) -> np.ndarray:
    predictor = np.column_stack([obi, direction])
    result = minimize(
        MarketQRW._calibration_objective,
        x0=initial,
        args=(predictor, target, coherence, regularization),
        method="L-BFGS-B",
        bounds=((-3.0, 3.0), (0.0, 5.0), (-5.0, 5.0)),
    )
    if not result.success:
        raise RuntimeError(f"QRW fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def fit_bias(
    obi: np.ndarray,
    direction: np.ndarray,
    target: np.ndarray,
    *,
    coherence: float,
    structural: np.ndarray,
    prior_bias: float,
) -> float:
    def objective(value: np.ndarray) -> float:
        parameters = structural.copy()
        parameters[0] = float(value[0])
        probability = qrw_probability(
            obi,
            direction,
            parameters,
            coherence,
        )
        return (
            log_loss(probability, target)
            + 0.01 * parameters[0] ** 2
            + 0.1 * (parameters[0] - prior_bias) ** 2
        )

    result = minimize(
        objective,
        x0=np.array([prior_bias]),
        method="L-BFGS-B",
        bounds=((-3.0, 3.0),),
    )
    if not result.success:
        raise RuntimeError(f"bias fit failed: {result.message}")
    return float(result.x[0])


def fit_affine(
    obi: np.ndarray,
    direction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(obi)), obi, direction])
    return np.linalg.lstsq(design, target, rcond=None)[0]


def affine_probability(
    obi: np.ndarray,
    direction: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    return np.clip(
        coefficients[0]
        + coefficients[1] * obi
        + coefficients[2] * direction,
        0.0,
        1.0,
    )


def fit_logistic(
    obi: np.ndarray,
    direction: np.ndarray,
    target: np.ndarray,
    *,
    regularization: float,
    initial: np.ndarray,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(obi)), obi, direction])

    def objective(coefficients: np.ndarray) -> float:
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        return (
            log_loss(probability, target)
            + regularization * float(coefficients[1:] @ coefficients[1:])
        )

    result = minimize(
        objective,
        x0=initial,
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def logistic_probability(
    obi: np.ndarray,
    direction: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(obi)), obi, direction])
    linear = np.clip(design @ coefficients, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-linear))


def block_bootstrap(
    daily_differences: list[np.ndarray],
    *,
    block_size: int,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float | int | list[float]]:
    block_means: list[np.ndarray] = []
    for values in daily_differences:
        blocks = [
            values[start : start + block_size]
            for start in range(0, len(values), block_size)
        ]
        block_means.append(
            np.asarray([float(block.mean()) for block in blocks])
        )
    means = np.concatenate(block_means)
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = means[rng.integers(0, len(means), size=len(means))]
        bootstrap[index] = float(sampled.mean())
    all_values = np.concatenate(daily_differences)
    return {
        "events": int(len(all_values)),
        "blocks": int(len(means)),
        "block_size": int(block_size),
        "model_minus_comparison": float(all_values.mean()),
        "confidence_interval_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "probability_model_improves": float(np.mean(bootstrap < 0.0)),
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    rows = []
    for day in audit["test_days"]:
        rows.append(
            "| "
            + " | ".join(
                [
                    day["date"],
                    str(day["scores"]["qrw"]["events"]),
                    f"{day['scores']['qrw']['brier']:.6f}",
                    f"{day['scores']['affine_market']['brier']:.6f}",
                    f"{day['scores']['logistic_market']['brier']:.6f}",
                    f"{day['alpha_obi']:.6f}",
                    f"{day['alpha_direction']:.6f}",
                    f"{day['obi_bias']:.6f}",
                ]
            )
            + " |"
        )
    affine = audit["bootstrap"]["affine_market"]
    logistic = audit["bootstrap"]["logistic_market"]
    lines = [
        "# Phase 3 multi-day predictive-edge audit",
        "",
        "Structural QRW parameters are selected on June 1-4 and frozen.",
        "June 5-7 are evaluated prequentially; only the regime bias is updated",
        "after each completed day. Baselines use the same OBI and tick-direction",
        "features and are refit only from past data.",
        "",
        "| Day | Events | QRW Brier | Affine Brier | Logistic Brier | alpha_obi | alpha_direction | bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Block bootstrap",
        "",
        (
            "- QRW minus affine Brier: "
            f"`{affine['model_minus_comparison']:.6f}`, 95% CI "
            f"`[{affine['confidence_interval_95'][0]:.6f}, "
            f"{affine['confidence_interval_95'][1]:.6f}]`"
        ),
        (
            "- QRW minus logistic Brier: "
            f"`{logistic['model_minus_comparison']:.6f}`, 95% CI "
            f"`[{logistic['confidence_interval_95'][0]:.6f}, "
            f"{logistic['confidence_interval_95'][1]:.6f}]`"
        ),
        "",
        "## Verdict",
        "",
        audit["verdict"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--fit-stride", type=int, default=20)
    parser.add_argument("--evaluation-stride", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/phase3_multiday_edge_audit.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/phase3_multiday_edge_audit.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [
        args.feature_dir / f"features_BTCUSDT_2026-06-{day:02d}.parquet"
        for day in range(1, 8)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing feature files: {missing}")

    fit_events = [
        extract_events(path, stride=args.fit_stride)
        for path in paths
    ]
    train_obi = np.concatenate([row[0] for row in fit_events[:3]])
    train_direction = np.concatenate([row[1] for row in fit_events[:3]])
    train_target = np.concatenate([row[2] for row in fit_events[:3]])
    validation_obi, validation_direction, validation_target, _ = fit_events[3]
    rho_1 = direction_autocorrelation(paths[:4])
    gamma = float(-np.log(np.clip(abs(rho_1), 1e-8, 1.0)))
    coherence = float(np.exp(-gamma))

    regularization_grid = (1e-4, 1e-3, 1e-2, 5e-2, 1e-1)
    initial = np.array([0.0, 1.0, 0.0])
    qrw_candidates = []
    logistic_candidates = []
    for regularization in regularization_grid:
        qrw = fit_qrw(
            train_obi,
            train_direction,
            train_target,
            coherence=coherence,
            regularization=regularization,
            initial=initial,
        )
        qrw_validation = qrw_probability(
            validation_obi,
            validation_direction,
            qrw,
            coherence,
        )
        qrw_candidates.append(
            (log_loss(qrw_validation, validation_target), regularization, qrw)
        )
        logistic = fit_logistic(
            train_obi,
            train_direction,
            train_target,
            regularization=regularization,
            initial=np.zeros(3),
        )
        logistic_validation = logistic_probability(
            validation_obi,
            validation_direction,
            logistic,
        )
        logistic_candidates.append(
            (
                log_loss(logistic_validation, validation_target),
                regularization,
                logistic,
            )
        )
    _, qrw_regularization, qrw_initial = min(qrw_candidates, key=lambda row: row[0])
    _, logistic_regularization, logistic_initial = min(
        logistic_candidates,
        key=lambda row: row[0],
    )

    development_obi = np.concatenate([row[0] for row in fit_events[:4]])
    development_direction = np.concatenate([row[1] for row in fit_events[:4]])
    development_target = np.concatenate([row[2] for row in fit_events[:4]])
    structural = fit_qrw(
        development_obi,
        development_direction,
        development_target,
        coherence=coherence,
        regularization=qrw_regularization,
        initial=qrw_initial,
    )
    logistic_coefficients = fit_logistic(
        development_obi,
        development_direction,
        development_target,
        regularization=logistic_regularization,
        initial=logistic_initial,
    )
    prior_bias = float(structural[0])
    affine_differences: list[np.ndarray] = []
    logistic_differences: list[np.ndarray] = []
    test_days = []

    for index, path in enumerate(paths[4:], start=4):
        current = structural.copy()
        current[0] = fit_bias(
            development_obi,
            development_direction,
            development_target,
            coherence=coherence,
            structural=structural,
            prior_bias=prior_bias,
        )
        prior_bias = float(current[0])
        affine_coefficients = fit_affine(
            development_obi,
            development_direction,
            development_target,
        )
        logistic_coefficients = fit_logistic(
            development_obi,
            development_direction,
            development_target,
            regularization=logistic_regularization,
            initial=logistic_coefficients,
        )
        evaluation_obi, evaluation_direction, evaluation_target, total = (
            extract_events(path, stride=args.evaluation_stride)
        )
        qrw_prediction = qrw_probability(
            evaluation_obi,
            evaluation_direction,
            current,
            coherence,
        )
        affine_prediction = affine_probability(
            evaluation_obi,
            evaluation_direction,
            affine_coefficients,
        )
        logistic_prediction = logistic_probability(
            evaluation_obi,
            evaluation_direction,
            logistic_coefficients,
        )
        affine_differences.append(
            (qrw_prediction - evaluation_target) ** 2
            - (affine_prediction - evaluation_target) ** 2
        )
        logistic_differences.append(
            (qrw_prediction - evaluation_target) ** 2
            - (logistic_prediction - evaluation_target) ** 2
        )
        test_days.append(
            {
                "date": path.stem.removeprefix("features_BTCUSDT_"),
                "moving_events_total": int(total),
                "alpha_obi": float(current[1]),
                "alpha_direction": float(current[2]),
                "obi_bias": float(current[0]),
                "scores": {
                    "qrw": score(qrw_prediction, evaluation_target),
                    "affine_market": score(
                        affine_prediction,
                        evaluation_target,
                    ),
                    "logistic_market": score(
                        logistic_prediction,
                        evaluation_target,
                    ),
                },
            }
        )
        sampled_obi, sampled_direction, sampled_target, _ = fit_events[index]
        development_obi = np.concatenate([development_obi, sampled_obi])
        development_direction = np.concatenate(
            [development_direction, sampled_direction]
        )
        development_target = np.concatenate(
            [development_target, sampled_target]
        )

    rng = np.random.default_rng(args.random_seed)
    affine_bootstrap = block_bootstrap(
        affine_differences,
        block_size=args.block_size,
        samples=args.bootstrap_samples,
        rng=rng,
    )
    logistic_bootstrap = block_bootstrap(
        logistic_differences,
        block_size=args.block_size,
        samples=args.bootstrap_samples,
        rng=rng,
    )
    affine_edge = affine_bootstrap["confidence_interval_95"][1] < 0.0
    logistic_edge = logistic_bootstrap["confidence_interval_95"][1] < 0.0
    if affine_edge and logistic_edge:
        verdict = (
            "QRW shows a statistically significant multi-day Brier edge over "
            "both fair affine and logistic baselines on the historical proxy."
        )
    elif affine_edge:
        verdict = (
            "QRW shows a statistically significant multi-day Brier edge over "
            "the fair affine baseline, but not over the nonlinear logistic "
            "baseline. This is predictive edge for the QRW link relative to the "
            "registered affine benchmark, not evidence of a uniquely quantum edge."
        )
    else:
        verdict = (
            "QRW does not show a statistically significant multi-day Brier edge "
            "over the fair affine baseline."
        )

    audit = {
        "development_days": [path.stem[-10:] for path in paths[:4]],
        "test_day_names": [path.stem[-10:] for path in paths[4:]],
        "fit_stride": args.fit_stride,
        "evaluation_stride": args.evaluation_stride,
        "rho_1": rho_1,
        "gamma": gamma,
        "coherence": coherence,
        "qrw_regularization": qrw_regularization,
        "logistic_regularization": logistic_regularization,
        "structural_parameters": {
            "obi_bias_initial": float(structural[0]),
            "alpha_obi": float(structural[1]),
            "alpha_direction": float(structural[2]),
        },
        "test_days": test_days,
        "bootstrap": {
            "affine_market": affine_bootstrap,
            "logistic_market": logistic_bootstrap,
        },
        "verdict": verdict,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.output_markdown, audit)
    print(args.output_markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

"""Audit a QRW with causal microstructure signal and adaptive decoherence."""

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


SIGNAL_COLUMNS = ("obi", "tick_direction", "obi_change", "abs_obi")


def extract_events(
    path: Path,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return causal raw features and next-price direction."""
    parquet = pq.ParquetFile(path)
    columns = [
        "price",
        "obi",
        "tick_direction",
        "trade_intensity",
        "segment_id",
        "obi_valid",
    ]
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
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
        intensity = values["trade_intensity"].astype(np.float64, copy=False)
        segment = values["segment_id"]
        valid_obi = values["obi_valid"].astype(bool, copy=False)
        delta = np.diff(price)
        valid = (
            (np.abs(delta) > 1e-12)
            & valid_obi[:-1]
            & (segment[:-1] == segment[1:])
        )
        moving = np.flatnonzero(valid)
        sequence = np.arange(moving_count, moving_count + len(moving))
        selected = moving[sequence % stride == 0]
        moving_count += len(moving)
        if len(selected):
            previous = np.maximum(selected - 1, 0)
            features.append(
                np.column_stack(
                    [
                        obi[selected],
                        direction[selected],
                        obi[selected] - obi[previous],
                        np.abs(obi[selected]),
                        np.log1p(intensity[selected]),
                    ]
                )
            )
            targets.append((delta[selected] > 0.0).astype(np.float64))
        carry = {name: array[-1] for name, array in values.items()}
    if not targets:
        raise ValueError(f"no moving events found in {path}")
    return np.vstack(features), np.concatenate(targets), moving_count


def direction_autocorrelation(paths: list[Path]) -> float:
    count = 0
    sums = np.zeros(5, dtype=np.float64)
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
            sums += [x.sum(), y.sum(), x @ x, y @ y, x @ y]
            carry = (float(direction[-1]), segment[-1])
    sum_x, sum_y, sum_xx, sum_yy, sum_xy = sums
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
    }


def normalize(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (features - mean) / scale


def qrw_probability(
    normalized: np.ndarray,
    parameters: np.ndarray,
    *,
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


def fit_qrw(
    normalized: np.ndarray,
    target: np.ndarray,
    *,
    gamma_base: float,
    regularization: float,
    initial: np.ndarray,
) -> np.ndarray:
    def objective(parameters: np.ndarray) -> float:
        probability = qrw_probability(
            normalized,
            parameters,
            gamma_base=gamma_base,
        )
        return (
            log_loss(probability, target)
            + regularization * float(parameters[1:] @ parameters[1:])
        )

    result = minimize(
        objective,
        x0=initial,
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
        raise RuntimeError(f"adaptive QRW fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def fit_qrw_bias(
    normalized: np.ndarray,
    target: np.ndarray,
    *,
    gamma_base: float,
    structural: np.ndarray,
    prior_bias: float,
) -> float:
    def objective(value: np.ndarray) -> float:
        parameters = structural.copy()
        parameters[0] = float(value[0])
        probability = qrw_probability(
            normalized,
            parameters,
            gamma_base=gamma_base,
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
        raise RuntimeError(f"adaptive QRW bias fit failed: {result.message}")
    return float(result.x[0])


def logistic_design(normalized: np.ndarray, *, pairwise: bool) -> np.ndarray:
    columns = [normalized]
    if pairwise:
        for left in range(normalized.shape[1]):
            for right in range(left, normalized.shape[1]):
                columns.append(
                    (
                        normalized[:, left]
                        * normalized[:, right]
                    )[:, None]
                )
    return np.column_stack([np.ones(len(normalized)), *columns])


def fit_logistic(
    normalized: np.ndarray,
    target: np.ndarray,
    *,
    regularization: float,
    pairwise: bool,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    design = logistic_design(normalized, pairwise=pairwise)
    start = (
        np.zeros(design.shape[1], dtype=np.float64)
        if initial is None
        else initial
    )

    def objective(coefficients: np.ndarray) -> float:
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        return (
            log_loss(probability, target)
            + regularization
            * float(coefficients[1:] @ coefficients[1:])
        )

    result = minimize(
        objective,
        x0=start,
        method="L-BFGS-B",
        options={"maxiter": 400},
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def logistic_probability(
    normalized: np.ndarray,
    coefficients: np.ndarray,
    *,
    pairwise: bool,
) -> np.ndarray:
    design = logistic_design(normalized, pairwise=pairwise)
    linear = np.clip(design @ coefficients, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-linear))


def block_bootstrap(
    differences: list[np.ndarray],
    *,
    block_size: int,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float | int | list[float]]:
    block_means = np.concatenate(
        [
            np.asarray(
                [
                    block.mean()
                    for block in np.array_split(
                        values,
                        max(1, int(np.ceil(len(values) / block_size))),
                    )
                ]
            )
            for values in differences
        ]
    )
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sample = block_means[
            rng.integers(0, len(block_means), size=len(block_means))
        ]
        bootstrap[index] = float(sample.mean())
    all_values = np.concatenate(differences)
    return {
        "events": int(len(all_values)),
        "blocks": int(len(block_means)),
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
                    str(day["scores"]["adaptive_qrw"]["events"]),
                    f"{day['scores']['adaptive_qrw']['brier']:.6f}",
                    f"{day['scores']['logistic_raw']['brier']:.6f}",
                    f"{day['scores']['logistic_pairwise']['brier']:.6f}",
                    f"{day['obi_bias']:.6f}",
                ]
            )
            + " |"
        )
    raw = audit["bootstrap"]["logistic_raw"]
    pairwise = audit["bootstrap"]["logistic_pairwise"]
    lines = [
        "# Adaptive-decoherence QRW edge audit",
        "",
        "Model selection uses June 1-3 for fitting and June 4 for validation.",
        "June 5-7 are prequential tests. All models receive the same five causal",
        "raw features: OBI, current tick direction, OBI change, absolute OBI,",
        "and log trade intensity. QRW uses intensity to modulate decoherence.",
        "",
        "| Day | Events | Adaptive QRW Brier | Logistic Brier | Pairwise logistic Brier | bias |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Block bootstrap",
        "",
        (
            "- QRW minus nonlinear logistic Brier: "
            f"`{raw['model_minus_comparison']:.6f}`, 95% CI "
            f"`[{raw['confidence_interval_95'][0]:.6f}, "
            f"{raw['confidence_interval_95'][1]:.6f}]`"
        ),
        (
            "- QRW minus pairwise logistic Brier: "
            f"`{pairwise['model_minus_comparison']:.6f}`, 95% CI "
            f"`[{pairwise['confidence_interval_95'][0]:.6f}, "
            f"{pairwise['confidence_interval_95'][1]:.6f}]`"
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
        default=Path("reports/phase3_adaptive_decoherence_audit.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/phase3_adaptive_decoherence_audit.md"),
    )
    parser.add_argument(
        "--calibrated-output",
        type=Path,
        default=Path("results/calibrated_params_adaptive_multiday.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [
        args.feature_dir / f"features_BTCUSDT_2026-06-{day:02d}.parquet"
        for day in range(1, 8)
    ]
    fit_events = [
        extract_events(path, stride=args.fit_stride)
        for path in paths
    ]
    train_features = np.vstack([row[0] for row in fit_events[:3]])
    train_target = np.concatenate([row[1] for row in fit_events[:3]])
    validation_features, validation_target, _ = fit_events[3]
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale < 1e-9] = 1.0
    train_normalized = normalize(
        train_features,
        feature_mean,
        feature_scale,
    )
    validation_normalized = normalize(
        validation_features,
        feature_mean,
        feature_scale,
    )

    rho_1 = direction_autocorrelation(paths[:4])
    gamma_base = float(-np.log(np.clip(abs(rho_1), 1e-8, 1.0)))
    regularization_grid = (1e-4, 1e-3, 1e-2, 5e-2, 1e-1)
    qrw_candidates = []
    raw_candidates = []
    pairwise_candidates = []
    for regularization in regularization_grid:
        qrw = fit_qrw(
            train_normalized,
            train_target,
            gamma_base=gamma_base,
            regularization=regularization,
            initial=np.zeros(6),
        )
        qrw_candidates.append(
            (
                score(
                    qrw_probability(
                        validation_normalized,
                        qrw,
                        gamma_base=gamma_base,
                    ),
                    validation_target,
                )["brier"],
                regularization,
                qrw,
            )
        )
        raw = fit_logistic(
            train_normalized,
            train_target,
            regularization=regularization,
            pairwise=False,
        )
        raw_candidates.append(
            (
                score(
                    logistic_probability(
                        validation_normalized,
                        raw,
                        pairwise=False,
                    ),
                    validation_target,
                )["brier"],
                regularization,
                raw,
            )
        )
        pairwise = fit_logistic(
            train_normalized,
            train_target,
            regularization=regularization,
            pairwise=True,
        )
        pairwise_candidates.append(
            (
                score(
                    logistic_probability(
                        validation_normalized,
                        pairwise,
                        pairwise=True,
                    ),
                    validation_target,
                )["brier"],
                regularization,
                pairwise,
            )
        )

    _, qrw_regularization, qrw_initial = min(qrw_candidates)
    _, raw_regularization, raw_initial = min(raw_candidates)
    _, pairwise_regularization, pairwise_initial = min(pairwise_candidates)
    development_features = np.vstack(
        [row[0] for row in fit_events[:4]]
    )
    development_target = np.concatenate(
        [row[1] for row in fit_events[:4]]
    )
    development_normalized = normalize(
        development_features,
        feature_mean,
        feature_scale,
    )
    structural = fit_qrw(
        development_normalized,
        development_target,
        gamma_base=gamma_base,
        regularization=qrw_regularization,
        initial=qrw_initial,
    )
    raw_coefficients = fit_logistic(
        development_normalized,
        development_target,
        regularization=raw_regularization,
        pairwise=False,
        initial=raw_initial,
    )
    pairwise_coefficients = fit_logistic(
        development_normalized,
        development_target,
        regularization=pairwise_regularization,
        pairwise=True,
        initial=pairwise_initial,
    )

    prior_bias = float(structural[0])
    raw_differences: list[np.ndarray] = []
    pairwise_differences: list[np.ndarray] = []
    test_days = []
    for index, path in enumerate(paths[4:], start=4):
        current = structural.copy()
        current[0] = fit_qrw_bias(
            development_normalized,
            development_target,
            gamma_base=gamma_base,
            structural=structural,
            prior_bias=prior_bias,
        )
        prior_bias = float(current[0])
        raw_coefficients = fit_logistic(
            development_normalized,
            development_target,
            regularization=raw_regularization,
            pairwise=False,
            initial=raw_coefficients,
        )
        pairwise_coefficients = fit_logistic(
            development_normalized,
            development_target,
            regularization=pairwise_regularization,
            pairwise=True,
            initial=pairwise_coefficients,
        )
        evaluation_features, evaluation_target, total = extract_events(
            path,
            stride=args.evaluation_stride,
        )
        evaluation_normalized = normalize(
            evaluation_features,
            feature_mean,
            feature_scale,
        )
        qrw_prediction = qrw_probability(
            evaluation_normalized,
            current,
            gamma_base=gamma_base,
        )
        raw_prediction = logistic_probability(
            evaluation_normalized,
            raw_coefficients,
            pairwise=False,
        )
        pairwise_prediction = logistic_probability(
            evaluation_normalized,
            pairwise_coefficients,
            pairwise=True,
        )
        raw_differences.append(
            (qrw_prediction - evaluation_target) ** 2
            - (raw_prediction - evaluation_target) ** 2
        )
        pairwise_differences.append(
            (qrw_prediction - evaluation_target) ** 2
            - (pairwise_prediction - evaluation_target) ** 2
        )
        test_days.append(
            {
                "date": path.stem.removeprefix("features_BTCUSDT_"),
                "moving_events_total": int(total),
                "obi_bias": float(current[0]),
                "scores": {
                    "adaptive_qrw": score(qrw_prediction, evaluation_target),
                    "logistic_raw": score(raw_prediction, evaluation_target),
                    "logistic_pairwise": score(
                        pairwise_prediction,
                        evaluation_target,
                    ),
                },
            }
        )
        sampled_features, sampled_target, _ = fit_events[index]
        development_features = np.vstack(
            [development_features, sampled_features]
        )
        development_target = np.concatenate(
            [development_target, sampled_target]
        )
        development_normalized = normalize(
            development_features,
            feature_mean,
            feature_scale,
        )

    rng = np.random.default_rng(args.random_seed)
    raw_bootstrap = block_bootstrap(
        raw_differences,
        block_size=args.block_size,
        samples=args.bootstrap_samples,
        rng=rng,
    )
    pairwise_bootstrap = block_bootstrap(
        pairwise_differences,
        block_size=args.block_size,
        samples=args.bootstrap_samples,
        rng=rng,
    )
    raw_edge = raw_bootstrap["confidence_interval_95"][1] < 0.0
    pairwise_edge = pairwise_bootstrap["confidence_interval_95"][1] < 0.0
    if raw_edge and pairwise_edge:
        verdict = (
            "Adaptive-decoherence QRW has a statistically significant edge "
            "over both nonlinear logistic baselines."
        )
    elif raw_edge:
        verdict = (
            "Adaptive-decoherence QRW has a statistically significant edge "
            "over the registered nonlinear logistic baseline using the same "
            "raw features. Pairwise logistic remains stronger, so this is not "
            "a universal or uniquely quantum advantage."
        )
    else:
        verdict = (
            "Adaptive-decoherence QRW does not establish edge over nonlinear "
            "logistic regression."
        )

    audit = {
        "development_days": [path.stem[-10:] for path in paths[:4]],
        "test_days_names": [path.stem[-10:] for path in paths[4:]],
        "raw_features": [
            *SIGNAL_COLUMNS,
            "log_trade_intensity",
        ],
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "rho_1": rho_1,
        "gamma_base": gamma_base,
        "qrw_regularization": qrw_regularization,
        "raw_logistic_regularization": raw_regularization,
        "pairwise_logistic_regularization": pairwise_regularization,
        "structural_parameters": {
            "obi_bias_initial": float(structural[0]),
            "alpha_obi": float(structural[1]),
            "alpha_direction": float(structural[2]),
            "alpha_obi_change": float(structural[3]),
            "alpha_abs_obi": float(structural[4]),
            "gamma_intensity": float(structural[5]),
        },
        "test_days": test_days,
        "bootstrap": {
            "logistic_raw": raw_bootstrap,
            "logistic_pairwise": pairwise_bootstrap,
        },
        "verdict": verdict,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    calibrated = {
        "gamma": gamma_base,
        "rho_1": rho_1,
        "obi_bias": float(structural[0]),
        "alpha_obi": float(structural[1]),
        "alpha_direction": float(structural[2]),
        "alpha_obi_change": float(structural[3]),
        "alpha_abs_obi": float(structural[4]),
        "gamma_intensity": float(structural[5]),
        "feature_names": [*SIGNAL_COLUMNS, "log_trade_intensity"],
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "selected_regularization": qrw_regularization,
        "calibration_method": (
            "adaptive_decoherence_multiday_development_validation"
        ),
        "development_days": audit["development_days"],
        "decoherence_channel": (
            "basis_dephasing_gamma_times_exp_intensity"
        ),
    }
    args.calibrated_output.parent.mkdir(parents=True, exist_ok=True)
    args.calibrated_output.write_text(
        json.dumps(calibrated, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(args.output_markdown, audit)
    print(args.output_markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

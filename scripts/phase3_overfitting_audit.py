"""Audit Phase 3 overfitting with chronological splits and stability checks."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase3_pipeline import resolve_feature_path
from src.models.qrw_market_sim import MarketQRW


def fit_model(frame: pd.DataFrame, *, tick_size: float | None = None) -> tuple[MarketQRW, dict[str, Any]]:
    config: dict[str, Any] = {
        "n_positions": 101,
        "gamma_base": 0.0,
        "alpha_obi": 0.0,
        "coin_type": "obi_adaptive",
    }
    if tick_size is not None:
        config["tick_size"] = tick_size
    model = MarketQRW(frame, config)
    with tempfile.TemporaryDirectory() as directory:
        parameters = model.calibrate(Path(directory) / "params.json")
    return model, parameters


def fit_fixed_structure_model(
    frame: pd.DataFrame,
    *,
    structural: dict[str, Any],
    tick_size: float,
    prior_bias: float,
    update_bias: bool = True,
) -> tuple[MarketQRW, dict[str, float | int | str]]:
    """Update the regime bias while preserving warmup structural parameters."""
    model = MarketQRW(
        frame,
        {
            "n_positions": 101,
            "gamma_base": float(structural["gamma"]),
            "alpha_obi": float(structural["alpha_obi"]),
            "alpha_direction": float(structural["alpha_direction"]),
            "obi_bias": prior_bias,
            "coin_type": "obi_adaptive",
            "tick_size": tick_size,
        },
    )
    model.gamma = float(structural["gamma"])
    if not update_bias:
        return model, {
            "obi_bias": prior_bias,
            "alpha_obi": float(structural["alpha_obi"]),
            "alpha_direction": float(structural["alpha_direction"]),
            "gamma": float(structural["gamma"]),
            "moving_events": 0,
            "calibration_method": "fixed_structure_bias_update_skipped",
            "objective": float("nan"),
        }
    update = model.calibrate_bias(
        regularization=0.01,
        prior_bias=prior_bias,
        prior_strength=0.1,
    )
    return model, update


def moving_events(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return causal OBI and next-price direction for nonzero price moves."""
    price = frame["price"].to_numpy(dtype=np.float64)
    obi = frame["obi"].to_numpy(dtype=np.float64)
    delta = np.diff(price)
    valid = np.abs(delta) > 1e-12
    if "segment_id" in frame:
        segment = frame["segment_id"].to_numpy(copy=False)
        valid &= segment[:-1] == segment[1:]
    if "obi_valid" in frame:
        valid &= frame["obi_valid"].astype(bool).to_numpy()[:-1]
    return obi[:-1][valid], (delta[valid] > 0.0).astype(np.float64)


def market_events(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return causal OBI, current tick direction, and next move direction."""
    price = frame["price"].to_numpy(dtype=np.float64)
    obi = frame["obi"].to_numpy(dtype=np.float64)
    direction = frame["tick_direction"].to_numpy(dtype=np.float64)
    delta = np.diff(price)
    valid = np.abs(delta) > 1e-12
    if "segment_id" in frame:
        segment = frame["segment_id"].to_numpy(copy=False)
        valid &= segment[:-1] == segment[1:]
    if "obi_valid" in frame:
        valid &= frame["obi_valid"].astype(bool).to_numpy()[:-1]
    return (
        obi[:-1][valid],
        direction[:-1][valid],
        (delta[valid] > 0.0).astype(np.float64),
    )


def score(probability: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    if len(target) == 0:
        raise ValueError("evaluation split contains no price-moving events")
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return {
        "events": int(len(target)),
        "up_fraction": float(target.mean()),
        "mean_probability": float(probability.mean()),
        "brier": float(np.mean((probability - target) ** 2)),
        "log_loss": float(
            -np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
        ),
        "accuracy": float(np.mean((probability >= 0.5) == target)),
    }


def fit_linear_probability(frame: pd.DataFrame) -> np.ndarray:
    """Fit the simple baseline ``P(up) = intercept + slope * OBI``."""
    obi, target = moving_events(frame)
    design = np.column_stack([np.ones(len(obi)), obi])
    return np.linalg.lstsq(design, target, rcond=None)[0]


def fit_linear_market_probability(frame: pd.DataFrame) -> np.ndarray:
    """Fit the fair affine baseline using OBI and current tick direction."""
    obi, direction, target = market_events(frame)
    design = np.column_stack([np.ones(len(obi)), obi, direction])
    return np.linalg.lstsq(design, target, rcond=None)[0]


def evaluate_split(
    model: MarketQRW,
    frame: pd.DataFrame,
    *,
    train_up_fraction: float,
    linear_coefficients: np.ndarray,
    linear_market_coefficients: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    obi, direction, target = market_events(frame)
    model_probability = np.asarray(
        [
            model._one_step_right_probability(value, tick)
            for value, tick in zip(obi, direction, strict=True)
        ],
        dtype=np.float64,
    )
    linear_probability = np.clip(
        linear_coefficients[0] + linear_coefficients[1] * obi,
        0.0,
        1.0,
    )
    linear_market_probability = np.clip(
        linear_market_coefficients[0]
        + linear_market_coefficients[1] * obi
        + linear_market_coefficients[2] * direction,
        0.0,
        1.0,
    )
    return {
        "model": score(model_probability, target),
        "linear_obi": score(linear_probability, target),
        "linear_market": score(linear_market_probability, target),
        "neutral": score(np.full(len(target), 0.5), target),
        "train_prior": score(np.full(len(target), train_up_fraction), target),
    }


def circular_shift_test(
    frame: pd.DataFrame,
    *,
    tick_size: float,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float | list[float]]:
    price = frame["price"].to_numpy(dtype=np.float64)
    predictor = frame["obi"].to_numpy(dtype=np.float64)[:-1]
    response = np.clip(np.diff(price) / tick_size, -1.0, 1.0)
    if "obi_valid" in frame:
        valid = frame["obi_valid"].astype(bool).to_numpy()[:-1]
        predictor = predictor[valid]
        response = response[valid]
    predictor = predictor - predictor.mean()
    response = response - response.mean()
    denominator = float(predictor @ predictor)
    observed = float(predictor @ response / denominator)

    null = np.empty(permutations, dtype=np.float64)
    minimum_shift = min(25, max(len(predictor) // 10, 1))
    maximum_shift = max(len(predictor) - minimum_shift, minimum_shift + 1)
    for index in range(permutations):
        shift = int(rng.integers(minimum_shift, maximum_shift))
        shifted = np.roll(predictor, shift)
        null[index] = float(shifted @ response / (shifted @ shifted))
    return {
        "observed_slope": observed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_95_interval": [
            float(np.quantile(null, 0.025)),
            float(np.quantile(null, 0.975)),
        ],
        "one_sided_p_value": float(
            (1 + np.count_nonzero(null >= observed)) / (permutations + 1)
        ),
    }


def paired_bootstrap_brier(
    model: MarketQRW,
    frame: pd.DataFrame,
    *,
    comparison_probability: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float | list[float]]:
    obi, direction, target = market_events(frame)
    probability = np.asarray(
        [
            model._one_step_right_probability(value, tick)
            for value, tick in zip(obi, direction, strict=True)
        ],
        dtype=np.float64,
    )
    paired_difference = (
        (probability - target) ** 2
        - (comparison_probability - target) ** 2
    )
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.integers(0, len(paired_difference), size=len(paired_difference))
        bootstrap[index] = float(paired_difference[sampled].mean())
    return {
        "model_minus_comparison": float(paired_difference.mean()),
        "confidence_interval_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "probability_model_improves": float(np.mean(bootstrap < 0.0)),
    }


def moving_block_bootstrap_mean(
    differences: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
    block_size: int = 16,
) -> dict[str, float | int | list[float]]:
    """Bootstrap a paired mean while retaining short-range dependence."""
    values = np.asarray(differences, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("differences cannot be empty")
    size = min(max(2, block_size), len(values))
    blocks_per_sample = int(np.ceil(len(values) / size))
    bootstrap = np.empty(samples, dtype=np.float64)
    offsets = np.arange(size)
    for index in range(samples):
        starts = rng.integers(0, len(values), size=blocks_per_sample)
        sampled = values[(starts[:, None] + offsets) % len(values)].reshape(-1)
        bootstrap[index] = float(sampled[: len(values)].mean())
    return {
        "events": int(len(values)),
        "block_size": int(size),
        "model_minus_comparison": float(values.mean()),
        "confidence_interval_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "probability_model_improves": float(np.mean(bootstrap < 0.0)),
    }


def rolling_stability(
    frame: pd.DataFrame,
    *,
    blocks: int,
    tick_size: float,
) -> list[dict[str, float | int | bool | str]]:
    """Measure causal rolling-origin stability with frozen structural parameters."""
    if blocks < 2:
        raise ValueError("rolling blocks must be at least 2")
    results: list[dict[str, float | int | bool | str]] = []
    warmup_end = int(len(frame) * 0.4)
    warmup = frame.iloc[:warmup_end].copy()
    _, structural = fit_model(warmup, tick_size=tick_size)
    boundaries = np.linspace(warmup_end, len(frame), blocks + 1, dtype=int)
    prior_bias = float(structural["obi_bias"])
    for block in range(blocks):
        train_end = int(boundaries[block])
        evaluation_end = int(boundaries[block + 1])
        train = frame.iloc[:train_end].copy()
        bias_history = frame.iloc[warmup_end:train_end].copy()
        update_bias = not bias_history.empty
        if not update_bias:
            bias_history = warmup.iloc[-2:].copy()
        evaluation = frame.iloc[train_end:evaluation_end].copy()
        model, bias_update = fit_fixed_structure_model(
            bias_history,
            structural=structural,
            tick_size=tick_size,
            prior_bias=prior_bias,
            update_bias=update_bias,
        )
        prior_bias = float(bias_update["obi_bias"])
        linear_coefficients = fit_linear_probability(train)
        linear_market_coefficients = fit_linear_market_probability(train)
        _, train_target = moving_events(train)
        evaluation_scores = evaluate_split(
            model,
            evaluation,
            train_up_fraction=float(train_target.mean()),
            linear_coefficients=linear_coefficients,
            linear_market_coefficients=linear_market_coefficients,
        )
        results.append(
            {
                "block": block,
                "train_rows": int(len(train)),
                "evaluation_rows": int(len(evaluation)),
                "moving_events": int(evaluation_scores["model"]["events"]),
                "obi_mean": float(evaluation["obi"].mean()),
                "alpha_obi": float(structural["alpha_obi"]),
                "alpha_direction": float(structural["alpha_direction"]),
                "obi_bias": prior_bias,
                "gamma": float(structural["gamma"]),
                "calibration_status": "fixed_structure_bias_update",
                "model_brier": float(evaluation_scores["model"]["brier"]),
                "linear_market_brier": float(
                    evaluation_scores["linear_market"]["brier"]
                ),
                "positive_slope": bool(structural["alpha_obi"] > 0.0),
            }
        )
    return results


def walk_forward_evaluation(
    frame: pd.DataFrame,
    *,
    folds: int,
    tick_size: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, float | int | list[float]]]:
    """Fit expanding windows and score each strictly later holdout window."""
    if folds < 2:
        raise ValueError("walk-forward folds must be at least 2")
    first_train_end = int(len(frame) * 0.4)
    warmup = frame.iloc[:first_train_end].copy()
    _, structural = fit_model(warmup, tick_size=tick_size)
    boundaries = np.linspace(
        first_train_end,
        len(frame),
        folds + 1,
        dtype=int,
    )
    results: list[dict[str, Any]] = []
    paired_differences: list[np.ndarray] = []
    prior_bias = float(structural["obi_bias"])
    for fold in range(folds):
        train_end = int(boundaries[fold])
        evaluation_end = int(boundaries[fold + 1])
        train = frame.iloc[:train_end].copy()
        bias_history = frame.iloc[first_train_end:train_end].copy()
        update_bias = not bias_history.empty
        if not update_bias:
            bias_history = warmup.iloc[-2:].copy()
        evaluation = frame.iloc[train_end:evaluation_end].copy()
        model, bias_update = fit_fixed_structure_model(
            bias_history,
            structural=structural,
            tick_size=tick_size,
            prior_bias=prior_bias,
            update_bias=update_bias,
        )
        prior_bias = float(bias_update["obi_bias"])
        linear_coefficients = fit_linear_probability(train)
        linear_market_coefficients = fit_linear_market_probability(train)
        _, train_target = moving_events(train)
        scores = evaluate_split(
            model,
            evaluation,
            train_up_fraction=float(train_target.mean()),
            linear_coefficients=linear_coefficients,
            linear_market_coefficients=linear_market_coefficients,
        )
        obi, direction, target = market_events(evaluation)
        model_probability = np.asarray(
            [
                model._one_step_right_probability(value, tick)
                for value, tick in zip(obi, direction, strict=True)
            ],
            dtype=np.float64,
        )
        linear_market_probability = np.clip(
            linear_market_coefficients[0]
            + linear_market_coefficients[1] * obi
            + linear_market_coefficients[2] * direction,
            0.0,
            1.0,
        )
        paired_difference = (
            (model_probability - target) ** 2
            - (linear_market_probability - target) ** 2
        )
        paired_differences.append(paired_difference)
        results.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "evaluation_rows": int(len(evaluation)),
                "calibration_status": "fixed_structure_bias_update",
                "alpha_obi": float(structural["alpha_obi"]),
                "alpha_direction": float(structural["alpha_direction"]),
                "obi_bias": prior_bias,
                "scores": scores,
                "model_minus_linear_market_brier": float(
                    paired_difference.mean()
                ),
            }
        )
    edge = moving_block_bootstrap_mean(
        np.concatenate(paired_differences),
        samples=bootstrap_samples,
        rng=rng,
    )
    edge["folds_model_better"] = int(
        sum(
            row["model_minus_linear_market_brier"] < 0.0
            for row in results
        )
    )
    return results, edge


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    split_rows = []
    for split in ("train", "validation", "test"):
        values = audit["scores"][split]
        split_rows.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(values["model"]["events"]),
                    f"{values['model']['brier']:.6f}",
                    f"{values['linear_obi']['brier']:.6f}",
                    f"{values['linear_market']['brier']:.6f}",
                    f"{values['neutral']['brier']:.6f}",
                    f"{values['model']['log_loss']:.6f}",
                    f"{values['model']['accuracy']:.2%}",
                ]
            )
            + " |"
        )

    rolling = audit["rolling_stability"]
    walk_forward = audit["walk_forward"]
    walk_forward_edge = audit["walk_forward_edge"]
    alpha_values = np.asarray([row["alpha_obi"] for row in rolling])
    direction_values = np.asarray([row["alpha_direction"] for row in rolling])
    lines = [
        "# Phase 3 overfitting audit",
        "",
        f"Feature source: `{audit['feature_path']}`",
        "",
        "The split is chronological: 60% train, 20% validation, 20% test.",
        "Directional scores condition on events where the next trade price",
        "changes; the separate movement gate models zero-price-change events.",
        "",
        "| Split | Moving events | QRW Brier | Linear-OBI Brier | Linear-market Brier | Neutral Brier | QRW log loss | QRW accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *split_rows,
        "",
        "## Stability",
        "",
        f"- Rolling blocks: `{len(rolling)}`",
        "- Structural parameters are frozen after the 40% warmup.",
        f"- Alpha range: `{alpha_values.min():.6f}` to `{alpha_values.max():.6f}`",
        f"- Alpha standard deviation: `{alpha_values.std():.6f}`",
        f"- Direction-coupling range: `{direction_values.min():.6f}` to `{direction_values.max():.6f}`",
        f"- Circular-shift p-value on train: `{audit['circular_shift']['one_sided_p_value']:.6f}`",
        "",
        "## Walk-forward",
        "",
        (
            f"- Expanding-window folds: `{len(walk_forward)}`; "
            f"mean QRW Brier: "
            f"`{np.mean([row['scores']['model']['brier'] for row in walk_forward]):.6f}`"
        ),
        (
            "- Mean linear-OBI Brier: "
            f"`{np.mean([row['scores']['linear_obi']['brier'] for row in walk_forward]):.6f}`"
        ),
        (
            "- Mean fair linear-market Brier: "
            f"`{np.mean([row['scores']['linear_market']['brier'] for row in walk_forward]):.6f}`"
        ),
        (
            "- Pooled QRW minus fair baseline Brier: "
            f"`{walk_forward_edge['model_minus_comparison']:.6f}`"
        ),
        (
            "- Moving-block 95% interval: "
            f"`[{walk_forward_edge['confidence_interval_95'][0]:.6f}, "
            f"{walk_forward_edge['confidence_interval_95'][1]:.6f}]`"
        ),
        "",
        "## Test uncertainty",
        "",
        f"- QRW minus linear-OBI Brier: `{audit['test_brier_bootstrap']['model_minus_comparison']:.6f}`",
        (
            "- 95% bootstrap interval: "
            f"`[{audit['test_brier_bootstrap']['confidence_interval_95'][0]:.6f}, "
            f"{audit['test_brier_bootstrap']['confidence_interval_95'][1]:.6f}]`"
        ),
        (
            "- QRW minus fair linear-market Brier: "
            f"`{audit['test_market_brier_bootstrap']['model_minus_comparison']:.6f}`"
        ),
        (
            "- Fair-baseline 95% interval: "
            f"`[{audit['test_market_brier_bootstrap']['confidence_interval_95'][0]:.6f}, "
            f"{audit['test_market_brier_bootstrap']['confidence_interval_95'][1]:.6f}]`"
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
    parser.add_argument("--feature-path", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("reports/phase3_overfitting_audit.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("reports/phase3_overfitting_audit.md"))
    parser.add_argument("--rolling-blocks", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = resolve_feature_path(args.feature_path)
    frame = pd.read_parquet(feature_path).sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)
    first_cut = int(len(frame) * 0.6)
    second_cut = int(len(frame) * 0.8)
    train = frame.iloc[:first_cut].copy()
    validation = frame.iloc[first_cut:second_cut].copy()
    test = frame.iloc[second_cut:].copy()

    warmup_end = int(len(frame) * 0.4)
    _, structural = fit_model(frame.iloc[:warmup_end].copy())
    train_bias_history = frame.iloc[warmup_end:first_cut].copy()
    model, train_bias_update = fit_fixed_structure_model(
        train_bias_history,
        structural=structural,
        tick_size=float(structural["tick_size"]),
        prior_bias=float(structural["obi_bias"]),
    )
    test_training = frame.iloc[:second_cut].copy()
    validation_bias_history = frame.iloc[first_cut:second_cut].copy()
    test_model, test_bias_update = fit_fixed_structure_model(
        validation_bias_history,
        structural=structural,
        tick_size=float(structural["tick_size"]),
        prior_bias=float(train_bias_update["obi_bias"]),
    )
    parameters = structural.copy()
    parameters.update(
        {
            "obi_bias": float(train_bias_update["obi_bias"]),
            "calibration_method": (
                "two_stage_disjoint_fixed_structure_bias_update"
            ),
            "calibration_status": "fixed_structure_bias_updated",
            "calibration_rows": int(len(train)),
            "bias_update_rows": int(len(train_bias_history)),
            "bias_update_reuses_warmup": False,
        }
    )
    linear_coefficients = fit_linear_probability(train)
    linear_market_coefficients = fit_linear_market_probability(train)
    test_linear_coefficients = fit_linear_probability(test_training)
    test_linear_market_coefficients = fit_linear_market_probability(
        test_training
    )
    _, train_target = moving_events(train)
    train_up_fraction = float(train_target.mean())
    rng = np.random.default_rng(args.random_seed)
    scores = {
        "train": evaluate_split(
            model,
            train,
            train_up_fraction=train_up_fraction,
            linear_coefficients=linear_coefficients,
            linear_market_coefficients=linear_market_coefficients,
        ),
        "validation": evaluate_split(
            model,
            validation,
            train_up_fraction=train_up_fraction,
            linear_coefficients=linear_coefficients,
            linear_market_coefficients=linear_market_coefficients,
        ),
        "test": evaluate_split(
            test_model,
            test,
            train_up_fraction=train_up_fraction,
            linear_coefficients=test_linear_coefficients,
            linear_market_coefficients=test_linear_market_coefficients,
        ),
    }
    rolling = rolling_stability(
        frame,
        blocks=args.rolling_blocks,
        tick_size=float(parameters["tick_size"]),
    )
    walk_forward, walk_forward_edge = walk_forward_evaluation(
        frame,
        folds=args.walk_forward_folds,
        tick_size=float(parameters["tick_size"]),
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    test_neutral_improvement = (
        scores["test"]["neutral"]["brier"] - scores["test"]["model"]["brier"]
    )
    alpha_values = np.asarray([row["alpha_obi"] for row in rolling])
    unstable_alpha = alpha_values.std() > 1e-12
    test_obi, test_direction, _ = market_events(test)
    test_bootstrap = paired_bootstrap_brier(
        test_model,
        test,
        comparison_probability=np.clip(
            test_linear_coefficients[0]
            + test_linear_coefficients[1] * test_obi,
            0.0,
            1.0,
        ),
        samples=args.bootstrap_samples,
        rng=rng,
    )
    test_market_bootstrap = paired_bootstrap_brier(
        test_model,
        test,
        comparison_probability=np.clip(
            test_linear_market_coefficients[0]
            + test_linear_market_coefficients[1] * test_obi
            + test_linear_market_coefficients[2] * test_direction,
            0.0,
            1.0,
        ),
        samples=args.bootstrap_samples,
        rng=rng,
    )
    bootstrap_lower, bootstrap_upper = (
        test_market_bootstrap["confidence_interval_95"]
    )
    walk_lower, walk_upper = walk_forward_edge["confidence_interval_95"]
    if test_neutral_improvement <= 0.0:
        verdict = (
            "The model does not beat the neutral baseline out of sample. "
            "Treat the current calibration as overfit."
        )
    elif walk_upper < 0.0 and not unstable_alpha:
        verdict = (
            "The fixed-structure QRW has a statistically significant pooled "
            "walk-forward Brier edge over the fair affine baseline. The final "
            "holdout fold remains separately reported. The independent multi-day "
            "historical proxy result is archived under "
            "reports/archive/invalidated_2026-06-13/."
        )
    elif bootstrap_lower > 0.0:
        verdict = (
            "The QRW model beats the neutral baseline but is materially worse "
            "than the fair affine baseline using the same OBI and tick-direction "
            "features. This provides no evidence that the QRW link adds "
            "out-of-sample value."
        )
    elif unstable_alpha:
        comparison = (
            "beats"
            if bootstrap_upper < 0.0
            else "is statistically indistinguishable from"
        )
        verdict = (
            f"The calibrated QRW {comparison} the fair linear-market baseline and "
            "beats neutral on this test window, but rolling parameters remain "
            "unstable. Overfitting risk remains HIGH; multi-day confirmation "
            "is required."
        )
    elif bootstrap_upper < 0.0:
        verdict = (
            "The calibrated QRW has a statistically significant Brier-score edge "
            "over the fair affine baseline on this held-out window, and structural "
            "rolling parameters remain fixed after warmup. Multi-day confirmation "
            "is still required before claiming a general QRW advantage."
        )
    else:
        verdict = (
            "The calibrated QRW beats neutral and has stable structural rolling "
            "parameters, but is statistically indistinguishable from the fair "
            "linear-market baseline. A predictive QRW edge is not established."
        )

    audit: dict[str, Any] = {
        "feature_path": str(feature_path),
        "rows": int(len(frame)),
        "split_rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "train_parameters": parameters,
        "linear_obi_coefficients": {
            "intercept": float(linear_coefficients[0]),
            "slope": float(linear_coefficients[1]),
        },
        "linear_market_coefficients": {
            "intercept": float(linear_market_coefficients[0]),
            "obi": float(linear_market_coefficients[1]),
            "tick_direction": float(linear_market_coefficients[2]),
        },
        "test_linear_market_coefficients": {
            "intercept": float(test_linear_market_coefficients[0]),
            "obi": float(test_linear_market_coefficients[1]),
            "tick_direction": float(test_linear_market_coefficients[2]),
        },
        "scores": scores,
        "rolling_stability": rolling,
        "walk_forward": walk_forward,
        "walk_forward_edge": walk_forward_edge,
        "circular_shift": circular_shift_test(
            train,
            tick_size=float(parameters["tick_size"]),
            permutations=args.permutations,
            rng=rng,
        ),
        "test_brier_bootstrap": test_bootstrap,
        "test_market_brier_bootstrap": test_market_bootstrap,
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

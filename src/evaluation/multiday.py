"""Day-clustered inference for dependent market-event evaluations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _moving_block_indices(
    cluster_count: int,
    *,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if cluster_count < 1:
        raise ValueError("at least one cluster is required")
    if block_size < 1 or block_size > cluster_count:
        raise ValueError("block_size must be within available day clusters")
    block_count = int(np.ceil(cluster_count / block_size))
    starts = rng.integers(0, cluster_count, size=block_count)
    return np.concatenate(
        [
            (start + np.arange(block_size, dtype=int)) % cluster_count
            for start in starts
        ]
    )[:cluster_count]


def day_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    value_columns: Iterable[str],
    iterations: int = 2_000,
    random_seed: int = 2026,
    block_size: int = 1,
) -> pd.DataFrame:
    """Bootstrap model metrics by resampling whole UTC-day clusters."""
    values = tuple(value_columns)
    required = {"day", "model", *values}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"daily metric frame is missing columns: {missing}")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if frame.duplicated(["day", "model"]).any():
        raise ValueError("each model must have at most one row per day")

    rows: list[dict[str, float | int | str]] = []
    children = iter(
        np.random.SeedSequence(random_seed).spawn(frame["model"].nunique())
    )
    for model, group in frame.groupby("model", sort=False):
        ordered = group.sort_values("day", kind="stable").reset_index(drop=True)
        cluster_count = len(ordered)
        if cluster_count < 2:
            raise ValueError(f"{model} requires at least two evaluated days")
        if block_size > cluster_count:
            raise ValueError(f"block_size exceeds days available for {model}")
        matrix = ordered[list(values)].to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError(f"{model} contains non-finite daily metrics")
        rng = np.random.default_rng(next(children))
        bootstrap = np.empty((iterations, len(values)), dtype=np.float64)
        for iteration in range(iterations):
            selected = _moving_block_indices(
                cluster_count, block_size=block_size, rng=rng
            )
            bootstrap[iteration] = matrix[selected].mean(axis=0)
        row: dict[str, float | int | str] = {
            "model": str(model),
            "evaluated_days": cluster_count,
            "bootstrap_iterations": iterations,
            "cluster_unit": "UTC_day",
            "block_size_days": block_size,
            "random_seed": random_seed,
        }
        for column_index, column in enumerate(values):
            row[column] = float(matrix[:, column_index].mean())
            row[f"{column}_ci_low"] = float(
                np.quantile(bootstrap[:, column_index], 0.025)
            )
            row[f"{column}_ci_high"] = float(
                np.quantile(bootstrap[:, column_index], 0.975)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def day_cluster_sensitivity(
    frame: pd.DataFrame,
    *,
    value_columns: Iterable[str],
    seeds: Iterable[int] = (2026, 2027, 2028),
    block_sizes: Iterable[int] = (1, 2, 5),
    iterations: int = 1_000,
) -> pd.DataFrame:
    """Return pre-specified seed and day-block sensitivity results."""
    available_days = int(frame["day"].nunique())
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        for block_size in sorted(
            {int(value) for value in block_sizes if 1 <= int(value) <= available_days}
        ):
            result = day_cluster_bootstrap(
                frame,
                value_columns=value_columns,
                iterations=iterations,
                random_seed=int(seed),
                block_size=block_size,
            )
            rows.append(result)
    if not rows:
        raise ValueError("no valid seed/block-size sensitivity combinations")
    return pd.concat(rows, ignore_index=True)


def _benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(values, dtype=np.float64)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    count = len(ranked)
    adjusted_ranked = np.minimum.accumulate(
        (ranked * count / np.arange(1, count + 1))[::-1]
    )[::-1]
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def paired_day_comparisons(
    frame: pd.DataFrame,
    *,
    metric_directions: dict[str, str],
    iterations: int = 2_000,
    random_seed: int = 2026,
    block_size: int = 1,
) -> pd.DataFrame:
    """Compare every model/metric pair and correct the whole test family."""
    if not metric_directions:
        raise ValueError("at least one selected metric is required")
    invalid = set(metric_directions.values()) - {"lower", "higher"}
    if invalid:
        raise ValueError(f"invalid metric directions: {sorted(invalid)}")
    required = {"day", "model", *metric_directions}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"daily comparison frame is missing: {missing}")
    if frame.duplicated(["day", "model"]).any():
        raise ValueError("daily comparison rows must be unique by model/day")
    models = sorted(frame["model"].unique())
    if len(models) < 2:
        raise ValueError("at least two models are required")

    rows: list[dict[str, float | int | str | bool]] = []
    seed_sequence = np.random.SeedSequence(random_seed)
    children = iter(
        seed_sequence.spawn(
            len(metric_directions) * len(models) * (len(models) - 1) // 2
        )
    )
    for metric, direction in metric_directions.items():
        pivot = frame.pivot(index="day", columns="model", values=metric).sort_index()
        if pivot.isna().any().any():
            raise ValueError(
                f"metric {metric} does not use the same days for every model"
            )
        for left_index, model1 in enumerate(models):
            for model2 in models[left_index + 1 :]:
                difference = (
                    pivot[model1].to_numpy(dtype=np.float64)
                    - pivot[model2].to_numpy(dtype=np.float64)
                )
                if direction == "higher":
                    difference = -difference
                rng = np.random.default_rng(next(children))
                bootstrap = np.empty(iterations, dtype=np.float64)
                null_bootstrap = np.empty(iterations, dtype=np.float64)
                centered = difference - difference.mean()
                for iteration in range(iterations):
                    selected = _moving_block_indices(
                        len(difference), block_size=block_size, rng=rng
                    )
                    bootstrap[iteration] = difference[selected].mean()
                    null_bootstrap[iteration] = centered[selected].mean()
                observed = float(difference.mean())
                pvalue = float(
                    (1 + np.sum(np.abs(null_bootstrap) >= abs(observed)))
                    / (iterations + 1)
                )
                rows.append(
                    {
                        "metric": metric,
                        "model1": model1,
                        "model2": model2,
                        "evaluated_days": len(difference),
                        "model1_minus_model2_oriented": observed,
                        "ci_low": float(np.quantile(bootstrap, 0.025)),
                        "ci_high": float(np.quantile(bootstrap, 0.975)),
                        "pvalue": pvalue,
                        "model1_better": bool(observed < 0.0),
                        "pvalue_method": "centered_moving_block_bootstrap",
                        "block_size_days": block_size,
                        "random_seed": random_seed,
                    }
                )
    result = pd.DataFrame(rows)
    result["pvalue_bh"] = _benjamini_hochberg(
        result["pvalue"].to_numpy(dtype=np.float64)
    )
    result["correction_scope"] = (
        "all_selected_model_metric_comparisons"
    )
    return result

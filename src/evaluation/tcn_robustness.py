"""Pre-holdout robustness diagnostics for the registered causal TCN."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.sequence_dataset import SequenceFeatureSpec
from src.evaluation.ml_protocol import MLDirectionalProtocol
from src.evaluation.provenance import sha256_file
from src.models.temporal_convolutional_direction import (
    TemporalConvolutionalDirectionalModel,
)


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _entries(
    manifest: Mapping[str, Any],
    *,
    horizon: int,
    fold: str,
    allowed_folds: Sequence[str],
) -> list[Mapping[str, Any]]:
    if fold not in allowed_folds and fold != "train":
        raise ValueError(f"Phase 6 cannot access fold: {fold}")
    selected = [
        entry
        for entry in manifest.get("shards", ())
        if int(entry["horizon_ticks"]) == horizon and entry["fold"] == fold
    ]
    if not selected:
        raise ValueError(f"sequence manifest has no {fold} shards at h={horizon}")
    return sorted(selected, key=lambda entry: str(entry["utc_day"]))


def _load_shard(
    entry: Mapping[str, Any],
    *,
    root: Path,
    channels: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = _resolve_path(str(entry["path"]), root)
    registered_hash = entry.get("sha256")
    if registered_hash and sha256_file(path) != registered_hash:
        raise ValueError(f"sequence shard SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32, copy=False)
        target = payload["target"].astype(np.float64, copy=False)
    expected = (int(entry["rows"]), channels, sequence_length)
    if features.shape != expected or target.shape != (expected[0],):
        raise ValueError(f"sequence shard schema mismatch: {path}")
    if not np.isfinite(features).all() or not np.isin(
        target, (0.0, 1.0)
    ).all():
        raise ValueError(f"invalid sequence shard values: {path}")
    return features, target


def _metric_row(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    minimum_rows: int = 1,
) -> dict[str, Any]:
    rows = len(target)
    if rows < minimum_rows:
        return {
            "rows": rows,
            "status": "insufficient_rows",
            "brier": None,
            "log_loss": None,
            "accuracy": None,
        }
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(target, dtype=np.float64)
    return {
        "rows": rows,
        "status": "evaluated",
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(
            -np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p))
        ),
        "accuracy": float(np.mean((p >= 0.5) == y)),
    }


def _regime_values(
    features: np.ndarray,
    *,
    volatility_index: int,
    liquidity_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    volatility = np.sqrt(
        np.mean(
            np.square(
                features[:, volatility_index, :].astype(
                    np.float64, copy=False
                )
            ),
            axis=1,
        )
    )
    liquidity = np.mean(
        features[:, liquidity_index, :].astype(np.float64, copy=False),
        axis=1,
    )
    return volatility, liquidity


def _fit_regime_thresholds(
    entries: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    channels: int,
    sequence_length: int,
    volatility_index: int,
    liquidity_index: int,
) -> dict[str, tuple[float, float]]:
    volatility_blocks: list[np.ndarray] = []
    liquidity_blocks: list[np.ndarray] = []
    for entry in entries:
        features, _ = _load_shard(
            entry,
            root=root,
            channels=channels,
            sequence_length=sequence_length,
        )
        volatility, liquidity = _regime_values(
            features,
            volatility_index=volatility_index,
            liquidity_index=liquidity_index,
        )
        volatility_blocks.append(volatility)
        liquidity_blocks.append(liquidity)
    quantiles = (1.0 / 3.0, 2.0 / 3.0)
    return {
        "volatility": tuple(
            float(value)
            for value in np.quantile(np.concatenate(volatility_blocks), quantiles)
        ),
        "liquidity": tuple(
            float(value)
            for value in np.quantile(np.concatenate(liquidity_blocks), quantiles)
        ),
    }


def _regime_masks(
    values: np.ndarray,
    thresholds: tuple[float, float],
    labels: Sequence[str],
) -> dict[str, np.ndarray]:
    lower, upper = thresholds
    return {
        str(labels[0]): values <= lower,
        str(labels[1]): (values > lower) & (values <= upper),
        str(labels[2]): values > upper,
    }


def _validate_inputs(
    manifest: Mapping[str, Any],
    models: Mapping[int, TemporalConvolutionalDirectionalModel],
    horizon: int,
    protocol: MLDirectionalProtocol,
) -> tuple[SequenceFeatureSpec, tuple[int, ...], tuple[str, ...]]:
    if manifest.get("kind") != "causal_sequence_dataset":
        raise ValueError("unsupported sequence manifest kind")
    if manifest.get("protocol_version") != protocol.protocol_version:
        raise ValueError("sequence manifest protocol does not match config")
    if manifest.get("test_labels_used_for_normalization") is not False:
        raise ValueError("sequence manifest does not prove train-only statistics")
    asset = str(manifest.get("asset"))
    if asset not in protocol.assets:
        raise ValueError("sequence manifest asset is not registered")
    if horizon not in protocol.evaluation_horizons:
        raise ValueError("horizon is not registered by the ML protocol")
    settings = protocol.raw["robustness"]
    seeds = tuple(int(value) for value in settings["seeds"])
    if set(models) != set(seeds):
        raise ValueError("Phase 6 requires exactly the registered seed models")
    folds = tuple(str(value) for value in settings["evaluation_folds"])
    if "test" in folds or settings["test_access"] != "forbidden":
        raise ValueError("Phase 6 robustness must not access test")
    spec = SequenceFeatureSpec.from_protocol(protocol.raw)
    tcn = protocol.raw["models"]["phase_5_temporal_convolutional"]
    expected_architecture = {
        "residual_channels": list(tcn["residual_channels"]),
        "kernel_size": int(tcn["kernel_size"]),
        "dilations": list(tcn["dilations"]),
        "dropout": float(tcn["dropout"]),
        "pooling": str(tcn["pooling"]),
    }
    if (
        tuple(manifest.get("channels", ())) != spec.channels
        or int(manifest.get("sequence_length", 0)) != spec.sequence_length
        or manifest.get("layout") != spec.layout
        or manifest.get("dtype") != spec.dtype
    ):
        raise ValueError("sequence manifest schema does not match protocol")
    for seed, model in models.items():
        if (
            model.random_seed != seed
            or model.protocol_version != protocol.protocol_version
            or model.horizon_ticks != horizon
            or model.channel_names != spec.channels
            or model.sequence_length != spec.sequence_length
            or dict(model.architecture) != expected_architecture
        ):
            raise ValueError(f"TCN model metadata mismatch for seed {seed}")
    return spec, seeds, folds


def evaluate_tcn_robustness(
    manifest: Mapping[str, Any],
    models: Mapping[int, TemporalConvolutionalDirectionalModel],
    horizon_ticks: int,
    protocol: MLDirectionalProtocol,
    *,
    repo_root: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate registered seeds on pretest folds and train-defined regimes."""
    horizon = int(horizon_ticks)
    spec, seeds, folds = _validate_inputs(
        manifest, models, horizon, protocol
    )
    root = Path(repo_root).resolve()
    robustness = protocol.raw["robustness"]
    regime_channels = robustness["regime_channels"]
    volatility_index = spec.channels.index(regime_channels["volatility"])
    liquidity_index = spec.channels.index(regime_channels["liquidity"])
    train_entries = _entries(
        manifest,
        horizon=horizon,
        fold="train",
        allowed_folds=folds,
    )
    thresholds = _fit_regime_thresholds(
        train_entries,
        root=root,
        channels=len(spec.channels),
        sequence_length=spec.sequence_length,
        volatility_index=volatility_index,
        liquidity_index=liquidity_index,
    )
    minimum_rows = int(robustness["minimum_group_rows"])
    seed_results: dict[str, Any] = {
        str(seed): {"folds": {}} for seed in seeds
    }
    fold_probabilities: dict[str, dict[int, list[np.ndarray]]] = {
        fold: {seed: [] for seed in seeds} for fold in folds
    }
    for fold in folds:
        fold_entries = _entries(
            manifest,
            horizon=horizon,
            fold=fold,
            allowed_folds=folds,
        )
        per_seed_targets: dict[int, list[np.ndarray]] = {
            seed: [] for seed in seeds
        }
        per_seed_regimes: dict[int, dict[str, dict[str, list[np.ndarray]]]] = {
            seed: {
                dimension: {
                    label: []
                    for label in (
                        ("low", "medium", "high")
                        if dimension == "volatility"
                        else ("liquid", "normal", "illiquid")
                    )
                }
                for dimension in ("volatility", "liquidity")
            }
            for seed in seeds
        }
        per_seed_regime_targets: dict[
            int, dict[str, dict[str, list[np.ndarray]]]
        ] = {
            seed: {
                dimension: {
                    label: []
                    for label in per_seed_regimes[seed][dimension]
                }
                for dimension in per_seed_regimes[seed]
            }
            for seed in seeds
        }
        for entry in fold_entries:
            features, target = _load_shard(
                entry,
                root=root,
                channels=len(spec.channels),
                sequence_length=spec.sequence_length,
            )
            volatility, liquidity = _regime_values(
                features,
                volatility_index=volatility_index,
                liquidity_index=liquidity_index,
            )
            masks = {
                "volatility": _regime_masks(
                    volatility,
                    thresholds["volatility"],
                    ("low", "medium", "high"),
                ),
                "liquidity": _regime_masks(
                    liquidity,
                    thresholds["liquidity"],
                    ("liquid", "normal", "illiquid"),
                ),
            }
            for seed in seeds:
                probability = models[seed].predict_proba(
                    features, device=device
                )
                fold_probabilities[fold][seed].append(probability)
                per_seed_targets[seed].append(target)
                day_row = {
                    "utc_day": str(entry["utc_day"]),
                    **_metric_row(probability, target),
                }
                seed_fold = seed_results[str(seed)]["folds"].setdefault(
                    fold, {"days": [], "regimes": {}}
                )
                seed_fold["days"].append(day_row)
                for dimension, dimension_masks in masks.items():
                    for label, mask in dimension_masks.items():
                        per_seed_regimes[seed][dimension][label].append(
                            probability[mask]
                        )
                        per_seed_regime_targets[seed][dimension][label].append(
                            target[mask]
                        )
        for seed in seeds:
            probability = np.concatenate(fold_probabilities[fold][seed])
            target = np.concatenate(per_seed_targets[seed])
            seed_fold = seed_results[str(seed)]["folds"][fold]
            seed_fold["overall"] = _metric_row(probability, target)
            seed_fold["regimes"] = {}
            for dimension in ("volatility", "liquidity"):
                seed_fold["regimes"][dimension] = {}
                for label in per_seed_regimes[seed][dimension]:
                    group_probability = np.concatenate(
                        per_seed_regimes[seed][dimension][label]
                    )
                    group_target = np.concatenate(
                        per_seed_regime_targets[seed][dimension][label]
                    )
                    seed_fold["regimes"][dimension][label] = _metric_row(
                        group_probability,
                        group_target,
                        minimum_rows=minimum_rows,
                    )

    multi_seed: dict[str, Any] = {}
    for fold in folds:
        brier = np.asarray(
            [
                seed_results[str(seed)]["folds"][fold]["overall"]["brier"]
                for seed in seeds
            ],
            dtype=np.float64,
        )
        prediction_matrix = np.vstack(
            [
                np.concatenate(fold_probabilities[fold][seed])
                for seed in seeds
            ]
        )
        multi_seed[fold] = {
            "mean_brier": float(np.mean(brier)),
            "std_brier": float(np.std(brier, ddof=0)),
            "min_brier": float(np.min(brier)),
            "max_brier": float(np.max(brier)),
            "mean_prediction_std": float(
                np.mean(np.std(prediction_matrix, axis=0, ddof=0))
            ),
        }
    fold_stability = {
        str(seed): {
            "calibration_minus_selection_brier": float(
                seed_results[str(seed)]["folds"]["calibration"]["overall"][
                    "brier"
                ]
                - seed_results[str(seed)]["folds"]["selection"]["overall"][
                    "brier"
                ]
            ),
            "worst_daily_brier": float(
                max(
                    day["brier"]
                    for fold in folds
                    for day in seed_results[str(seed)]["folds"][fold]["days"]
                )
            ),
        }
        for seed in seeds
    }
    return {
        "kind": "tcn_pretest_robustness",
        "status": robustness["status"],
        "protocol_version": protocol.protocol_version,
        "asset": str(manifest["asset"]),
        "horizon_ticks": horizon,
        "seeds": list(seeds),
        "evaluation_folds": list(folds),
        "regime_thresholds": {
            "source_fold": "train",
            "method": robustness["regime_threshold_source"],
            "volatility": list(thresholds["volatility"]),
            "liquidity": list(thresholds["liquidity"]),
        },
        "seed_results": seed_results,
        "multi_seed": multi_seed,
        "fold_stability": fold_stability,
        "calibration_metric_role": (
            "in_sample_calibrator_diagnostic_not_generalization"
        ),
        "test_fold_read": False,
        "test_metrics": None,
    }


def aggregate_cross_asset_robustness(
    reports: Sequence[Mapping[str, Any]],
    protocol: MLDirectionalProtocol,
) -> dict[str, Any]:
    """Aggregate one independently refitted report per registered asset."""
    if len(reports) != len(protocol.assets):
        raise ValueError("one robustness report is required per asset")
    by_asset = {str(report.get("asset")): report for report in reports}
    if set(by_asset) != set(protocol.assets):
        raise ValueError("robustness reports must cover all registered assets")
    expected_seeds = list(protocol.raw["robustness"]["seeds"])
    expected_folds = list(protocol.raw["robustness"]["evaluation_folds"])
    horizons = {int(report.get("horizon_ticks", -1)) for report in reports}
    if len(horizons) != 1:
        raise ValueError("cross-asset reports must use one common horizon")
    for report in reports:
        if (
            report.get("kind") != "tcn_pretest_robustness"
            or report.get("protocol_version") != protocol.protocol_version
            or report.get("seeds") != expected_seeds
            or report.get("evaluation_folds") != expected_folds
            or report.get("test_fold_read") is not False
            or report.get("test_metrics") is not None
        ):
            raise ValueError("incompatible cross-asset robustness report")
    folds: dict[str, Any] = {}
    for fold in expected_folds:
        asset_brier = {
            asset: float(report["multi_seed"][fold]["mean_brier"])
            for asset, report in by_asset.items()
        }
        values = np.asarray(list(asset_brier.values()), dtype=np.float64)
        folds[fold] = {
            "asset_mean_brier": asset_brier,
            "equal_asset_mean_brier": float(np.mean(values)),
            "asset_brier_std": float(np.std(values, ddof=0)),
            "best_asset_brier": float(np.min(values)),
            "worst_asset_brier": float(np.max(values)),
        }
    return {
        "kind": "tcn_cross_asset_pretest_robustness",
        "status": protocol.raw["robustness"]["status"],
        "protocol_version": protocol.protocol_version,
        "assets": list(protocol.assets),
        "horizon_ticks": horizons.pop(),
        "seeds": expected_seeds,
        "evaluation_folds": expected_folds,
        "cross_asset_policy": protocol.raw["robustness"][
            "cross_asset_policy"
        ],
        "folds": folds,
        "test_fold_read": False,
        "test_metrics": None,
    }

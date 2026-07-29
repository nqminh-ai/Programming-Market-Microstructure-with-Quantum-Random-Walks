"""Frozen configuration contract for exploratory ML directional benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


ML_PROTOCOL_VERSION = "ml_directional_v7"
ML_PROTOCOL_STATUS = "exploratory_only_not_confirmatory"
DEFAULT_ML_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "ml_experiment.yaml"
)


class MLProtocolError(ValueError):
    """Raised when the frozen ML protocol is missing or internally inconsistent."""


@dataclass(frozen=True)
class MLDirectionalProtocol:
    """Validated fields needed by the later dataset and benchmark phases."""

    protocol_version: str
    status: str
    random_seed: int
    assets: tuple[str, ...]
    prediction_skill_horizons: tuple[int, ...]
    economic_horizons: tuple[int, ...]
    evaluation_horizons: tuple[int, ...]
    train_days: int
    selection_days: int
    calibration_days: int
    test_days: int
    purge_ticks: int
    primary_metric: str
    raw: Mapping[str, Any]

    @property
    def total_days(self) -> int:
        return (
            self.train_days
            + self.selection_days
            + self.calibration_days
            + self.test_days
        )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MLProtocolError(f"{field} must be a mapping")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MLProtocolError(f"{field} must be a positive integer")
    return value


def _horizons(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise MLProtocolError(f"{field} must be a non-empty list")
    result = tuple(_positive_int(item, field) for item in value)
    if tuple(sorted(set(result))) != result:
        raise MLProtocolError(f"{field} must be unique and strictly increasing")
    return result


def validate_ml_protocol(config: Mapping[str, Any]) -> MLDirectionalProtocol:
    """Validate invariants that must not drift between ML research phases."""
    root = _mapping(config, "config")
    version = root.get("protocol_version")
    if version != ML_PROTOCOL_VERSION:
        raise MLProtocolError(
            f"protocol_version must be {ML_PROTOCOL_VERSION!r}, found {version!r}"
        )
    status = root.get("status")
    if status != ML_PROTOCOL_STATUS:
        raise MLProtocolError(
            f"status must be {ML_PROTOCOL_STATUS!r}, found {status!r}"
        )
    seed = _positive_int(root.get("random_seed"), "random_seed")

    assets_config = _mapping(root.get("assets"), "assets")
    assets = tuple(assets_config)
    if assets != ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        raise MLProtocolError(
            "assets must be ordered as BTCUSDT, ETHUSDT, BNBUSDT"
        )
    for asset, settings in assets_config.items():
        path = _mapping(settings, f"assets.{asset}").get("feature_path")
        if not isinstance(path, str) or not path.endswith("_69d.parquet"):
            raise MLProtocolError(
                f"assets.{asset}.feature_path must identify a 69-day parquet"
            )

    target = _mapping(root.get("target"), "target")
    if target.get("kind") != "binary_direction":
        raise MLProtocolError("target.kind must be 'binary_direction'")
    if target.get("zero_return_policy") != "exclude":
        raise MLProtocolError("target.zero_return_policy must be 'exclude'")
    prediction = _horizons(
        target.get("prediction_skill_horizons_ticks"),
        "target.prediction_skill_horizons_ticks",
    )
    economic = _horizons(
        target.get("economic_horizons_ticks"),
        "target.economic_horizons_ticks",
    )
    evaluation = _horizons(
        target.get("evaluation_horizons_ticks"),
        "target.evaluation_horizons_ticks",
    )
    if set(evaluation) != set(prediction).union(economic):
        raise MLProtocolError(
            "evaluation horizons must equal the union of prediction and "
            "economic horizons"
        )

    sampling = _mapping(root.get("sampling"), "sampling")
    if sampling.get("anchor_stride") != "horizon":
        raise MLProtocolError("sampling.anchor_stride must be 'horizon'")
    if sampling.get("overlapping_labels") != "forbidden":
        raise MLProtocolError("overlapping labels must be forbidden")
    for guard in ("require_same_segment", "require_same_utc_day"):
        if sampling.get(guard) is not True:
            raise MLProtocolError(f"sampling.{guard} must be true")

    split = _mapping(root.get("split"), "split")
    if split.get("unit") != "utc_day":
        raise MLProtocolError("split.unit must be 'utc_day'")
    day_fields = (
        "train_days",
        "selection_days",
        "calibration_days",
        "test_days",
    )
    days = {
        field: _positive_int(split.get(field), f"split.{field}")
        for field in day_fields
    }
    expected_days = _positive_int(
        split.get("expected_complete_days"), "split.expected_complete_days"
    )
    if sum(days.values()) != expected_days:
        raise MLProtocolError(
            "chronological split day counts must sum to expected_complete_days"
        )
    purge_ticks = _positive_int(split.get("purge_ticks"), "split.purge_ticks")
    if purge_ticks < max(evaluation):
        raise MLProtocolError(
            "split.purge_ticks must cover the largest evaluation horizon"
        )
    if split.get("test_access") != (
        "once_after_model_and_calibrator_are_frozen"
    ):
        raise MLProtocolError("test access policy is not frozen")

    features = _mapping(root.get("features"), "features")
    base_features = features.get("base")
    if not isinstance(base_features, list) or not base_features:
        raise MLProtocolError("features.base must be a non-empty list")
    if len(set(base_features)) != len(base_features):
        raise MLProtocolError("features.base cannot contain duplicates")
    if features.get("causality") != "trailing_only_at_or_before_anchor":
        raise MLProtocolError("feature causality policy is not frozen")
    if features.get("normalization") != "fit_on_train_only":
        raise MLProtocolError("feature normalization must use train only")

    models = _mapping(root.get("models"), "models")
    boosting = _mapping(
        models.get("phase_2_hist_gradient_boosting"),
        "models.phase_2_hist_gradient_boosting",
    )
    if boosting.get("backend") != (
        "sklearn.ensemble.HistGradientBoostingClassifier"
    ):
        raise MLProtocolError("Phase 2 boosting backend is not frozen")
    if boosting.get("loss") != "log_loss":
        raise MLProtocolError("Phase 2 boosting loss must be log_loss")
    if boosting.get("early_stopping") is not False:
        raise MLProtocolError(
            "Phase 2 internal early stopping must be disabled"
        )
    grid = _mapping(
        boosting.get("grid"),
        "models.phase_2_hist_gradient_boosting.grid",
    )
    expected_grid = {
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
    }
    if set(grid) != expected_grid:
        raise MLProtocolError("Phase 2 boosting grid fields are not frozen")
    for field, values in grid.items():
        if not isinstance(values, list) or not values:
            raise MLProtocolError(f"Phase 2 grid {field} cannot be empty")
    if boosting.get("probability_calibrators") != [
        "identity",
        "platt",
        "isotonic",
    ]:
        raise MLProtocolError("Phase 2 calibrator set is not frozen")
    sequence = _mapping(
        models.get("phase_4_sequence_dataset"),
        "models.phase_4_sequence_dataset",
    )
    if (
        sequence.get("sequence_length") != 128
        or sequence.get("layout") != "channels_first"
        or sequence.get("dtype") != "float32"
        or sequence.get("shard_unit") != "utc_day"
        or sequence.get("normalization") != "train_fold_statistics_only"
    ):
        raise MLProtocolError("Phase 4 sequence storage policy is not frozen")
    if sequence.get("channels") != [
        "obi",
        "obi_valid",
        "tick_direction",
        "log_trade_intensity",
        "signed_log_quantity",
        "one_tick_log_return",
        "price_mid_deviation",
        "inter_event_seconds",
    ]:
        raise MLProtocolError("Phase 4 sequence channels are not frozen")
    temporal_lookback = max(
        *features["tick_direction_lags"],
        *features["signed_volume_windows_ticks"],
        *features["return_windows_ticks"],
        *features["realised_volatility_windows_ticks"],
    )
    if int(sequence["sequence_length"]) > temporal_lookback:
        raise MLProtocolError(
            "sequence_length cannot exceed the Phase 1 temporal lookback"
        )
    tcn = _mapping(
        models.get("phase_5_temporal_convolutional"),
        "models.phase_5_temporal_convolutional",
    )
    expected_tcn = {
        "backend": "torch",
        "loss": "binary_cross_entropy_with_logits",
        "residual_channels": [16, 16, 16, 16, 16, 16],
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8, 16, 32],
        "dropout": 0.1,
        "pooling": "last_timestep",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "max_epochs": 30,
        "early_stopping_patience": 6,
        "gradient_clip_norm": 1.0,
        "probability_calibrators": [
            "identity",
            "platt",
            "isotonic",
        ],
        "device_policy": "cpu_default_explicit_override",
    }
    if dict(tcn) != expected_tcn:
        raise MLProtocolError("Phase 5 TCN training contract is not frozen")
    if len(tcn["residual_channels"]) != len(tcn["dilations"]):
        raise MLProtocolError(
            "Phase 5 residual channels and dilations must have equal length"
        )
    hybrid = _mapping(
        models.get("phase_7_neural_adaptive_qrw"),
        "models.phase_7_neural_adaptive_qrw",
    )
    expected_hybrid = {
        "backend": "torch_complex_amplitude",
        "qrw_steps": 8,
        "anchor_signal_channel": "obi",
        "theta_bounds_radians": [0.3926990817, 1.1780972451],
        "gamma_bounds": [0.0, 0.35],
        "fixed_theta_radians": 0.7853981634,
        "fixed_gamma": 0.05,
        "hybrid_neural_weight": 0.5,
        "variants": [
            "neural_only",
            "fixed_qrw_only",
            "adaptive_qrw_only",
            "hybrid_fixed_qrw",
            "hybrid_adaptive_qrw",
        ],
        "training_policy": (
            "inherit_phase_5_optimizer_and_early_stopping"
        ),
        "probability_calibrators": [
            "identity",
            "platt",
            "isotonic",
        ],
        "test_access": "forbidden",
    }
    if dict(hybrid) != expected_hybrid:
        raise MLProtocolError(
            "Phase 7 neural-adaptive QRW contract is not frozen"
        )

    selection = _mapping(root.get("model_selection"), "model_selection")
    primary_metric = selection.get("primary_metric")
    if primary_metric != "brier":
        raise MLProtocolError("model_selection.primary_metric must be 'brier'")
    if selection.get("probability_calibration") != (
        "calibration_fold_selected_identity_platt_or_isotonic"
    ):
        raise MLProtocolError("probability calibration policy is not frozen")
    if selection.get("test_set_may_not_select_models") is not True:
        raise MLProtocolError("the test set may not select models")

    metrics = _mapping(root.get("metrics"), "metrics")
    if metrics.get("common_sample_for_all_models") is not True:
        raise MLProtocolError("all models must use a common evaluation sample")
    uncertainty = _mapping(metrics.get("uncertainty"), "metrics.uncertainty")
    if (
        uncertainty.get("method") != "paired_block_bootstrap"
        or uncertainty.get("unit") != "utc_day"
        or _positive_int(
            uncertainty.get("resamples"), "metrics.uncertainty.resamples"
        )
        != 2000
        or uncertainty.get("confidence_level") != 0.95
    ):
        raise MLProtocolError("Phase 3 uncertainty protocol is not frozen")
    if metrics.get("test_access_gate") != "explicit_open_test_flag":
        raise MLProtocolError("Phase 3 test access gate is not frozen")

    robustness = _mapping(root.get("robustness"), "robustness")
    expected_robustness = {
        "phase": "phase_6_pretest",
        "status": "exploratory_diagnostics_only",
        "seeds": [2026, 2027, 2028],
        "evaluation_folds": ["selection", "calibration"],
        "regime_threshold_source": "train_fold_tertiles",
        "regime_channels": {
            "volatility": "one_tick_log_return",
            "liquidity": "inter_event_seconds",
        },
        "minimum_group_rows": 20,
        "cross_asset_policy": "refit_per_asset_equal_weight_summary",
        "test_access": "forbidden",
    }
    if dict(robustness) != expected_robustness:
        raise MLProtocolError(
            "Phase 6 pretest robustness contract is not frozen"
        )
    release = _mapping(root.get("release"), "release")
    expected_release = {
        "phase": "phase_8",
        "status": "pretest_release_candidate",
        "scope": "single_asset_single_horizon",
        "required_artifact_roles": {
            "phase1_dataset_metadata": "ml_directional_dataset",
            "phase4_sequence_manifest": "causal_sequence_dataset",
            "phase2_hgb_training": "histogram_gradient_boosting_training",
            "phase5_tcn_training": (
                "temporal_convolutional_network_training"
            ),
            "phase6_tcn_robustness": "tcn_pretest_robustness",
            "phase7_hybrid_ablation": "neural_adaptive_qrw_ablation",
        },
        "optional_artifact_roles": {
            "phase6_cross_asset": (
                "tcn_cross_asset_pretest_robustness"
            )
        },
        "test_policy": "closed_no_test_metrics",
        "dashboard_policy": "read_only_manifest_no_recompute",
        "reproduction_shell": "powershell",
        "hash_algorithm": "sha256",
        "official_requires_clean_source_tree": True,
    }
    if dict(release) != expected_release:
        raise MLProtocolError("Phase 8 release contract is not frozen")

    economic_evaluation = _mapping(
        root.get("economic_evaluation"), "economic_evaluation"
    )
    if economic_evaluation.get("live_trading_authorized") is not False:
        raise MLProtocolError("the exploratory protocol cannot authorize trading")

    return MLDirectionalProtocol(
        protocol_version=version,
        status=status,
        random_seed=seed,
        assets=assets,
        prediction_skill_horizons=prediction,
        economic_horizons=economic,
        evaluation_horizons=evaluation,
        train_days=days["train_days"],
        selection_days=days["selection_days"],
        calibration_days=days["calibration_days"],
        test_days=days["test_days"],
        purge_ticks=purge_ticks,
        primary_metric=primary_metric,
        raw=root,
    )


def load_ml_protocol(
    path: str | Path = DEFAULT_ML_CONFIG,
) -> MLDirectionalProtocol:
    """Load and validate the frozen ML experiment configuration."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    return validate_ml_protocol(_mapping(payload, "config"))

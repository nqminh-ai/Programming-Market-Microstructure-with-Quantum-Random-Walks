"""Tests for the frozen exploratory ML protocol."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.evaluation.ml_protocol import (
    DEFAULT_ML_CONFIG,
    ML_PROTOCOL_VERSION,
    MLProtocolError,
    load_ml_protocol,
    validate_ml_protocol,
)


def test_repository_ml_protocol_is_valid_and_frozen() -> None:
    protocol = load_ml_protocol()

    assert DEFAULT_ML_CONFIG.is_file()
    assert protocol.protocol_version == ML_PROTOCOL_VERSION
    assert protocol.assets == ("BTCUSDT", "ETHUSDT", "BNBUSDT")
    assert protocol.prediction_skill_horizons == (1000, 5000, 10000, 50000)
    assert protocol.economic_horizons == (50000, 100000, 200000)
    assert protocol.evaluation_horizons == (
        1000,
        5000,
        10000,
        50000,
        100000,
        200000,
    )
    assert protocol.total_days == 69
    assert protocol.purge_ticks >= max(protocol.evaluation_horizons)
    assert protocol.primary_metric == "brier"
    assert protocol.raw["features"]["tick_direction_lags"] == [
        1,
        2,
        3,
        4,
        5,
        10,
    ]


def test_protocol_rejects_an_underpurged_split() -> None:
    protocol = load_ml_protocol()
    changed = deepcopy(protocol.raw)
    changed["split"]["purge_ticks"] = 10000

    with pytest.raises(MLProtocolError, match="largest evaluation horizon"):
        validate_ml_protocol(changed)


def test_protocol_rejects_overlapping_labels() -> None:
    protocol = load_ml_protocol()
    changed = deepcopy(protocol.raw)
    changed["sampling"]["overlapping_labels"] = "allowed"

    with pytest.raises(MLProtocolError, match="overlapping labels"):
        validate_ml_protocol(changed)


def test_protocol_rejects_using_test_for_model_selection() -> None:
    protocol = load_ml_protocol()
    changed = deepcopy(protocol.raw)
    changed["model_selection"]["test_set_may_not_select_models"] = False

    with pytest.raises(MLProtocolError, match="test set"):
        validate_ml_protocol(changed)


def test_protocol_rejects_tcn_architecture_drift() -> None:
    protocol = load_ml_protocol()
    changed = deepcopy(protocol.raw)
    changed["models"]["phase_5_temporal_convolutional"][
        "dilations"
    ] = [1, 2, 4]

    with pytest.raises(MLProtocolError, match="TCN training contract"):
        validate_ml_protocol(changed)


def test_protocol_rejects_robustness_seed_drift() -> None:
    protocol = load_ml_protocol()
    changed = deepcopy(protocol.raw)
    changed["robustness"]["seeds"] = [2026, 2027, 9999]

    with pytest.raises(MLProtocolError, match="robustness contract"):
        validate_ml_protocol(changed)


def test_protocol_rejects_hybrid_ablation_drift() -> None:
    protocol = load_ml_protocol()
    changed = deepcopy(protocol.raw)
    changed["models"]["phase_7_neural_adaptive_qrw"][
        "variants"
    ].remove("fixed_qrw_only")

    with pytest.raises(MLProtocolError, match="neural-adaptive QRW"):
        validate_ml_protocol(changed)


def test_protocol_rejects_release_role_drift() -> None:
    protocol = load_ml_protocol()
    changed = deepcopy(protocol.raw)
    changed["release"]["required_artifact_roles"].pop(
        "phase7_hybrid_ablation"
    )

    with pytest.raises(MLProtocolError, match="release contract"):
        validate_ml_protocol(changed)

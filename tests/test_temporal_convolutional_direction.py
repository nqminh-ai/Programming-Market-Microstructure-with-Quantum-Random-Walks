"""Tests for the Phase 5 causal Temporal Convolutional Network."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.ml_protocol import load_ml_protocol
from src.models.temporal_convolutional_direction import (
    TemporalConvolutionalNetwork,
    load_temporal_convolutional_model,
    save_temporal_convolutional_model,
    train_temporal_convolutional_network,
)


HORIZON = 1000


def _fast_protocol():
    base = load_ml_protocol()
    raw = deepcopy(base.raw)
    settings = raw["models"]["phase_5_temporal_convolutional"]
    settings.update(
        {
            "residual_channels": [4],
            "dilations": [1],
            "kernel_size": 2,
            "dropout": 0.0,
            "batch_size": 8,
            "max_epochs": 3,
            "early_stopping_patience": 2,
        }
    )
    return replace(base, raw=raw)


def _write_shard(
    path: Path,
    *,
    fold: str,
    seed: int,
    rows: int = 24,
) -> dict:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 8, 128)).astype(np.float32)
    target = np.tile(np.array([0, 1], dtype=np.int8), rows // 2)
    features[:, 0, -1] = np.where(target == 1, 2.0, -2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        target=target,
        timestamp=np.arange(rows, dtype=np.int64),
        target_timestamp=np.arange(rows, dtype=np.int64) + HORIZON,
        anchor_row=np.arange(rows, dtype=np.int64),
        forward_log_return=np.where(target == 1, 0.01, -0.01),
    )
    return {
        "horizon_ticks": HORIZON,
        "utc_day": f"2026-01-{seed:02d}",
        "fold": fold,
        "rows": rows,
        "path": str(path),
        "sha256": None,
    }


def _manifest(tmp_path: Path) -> dict:
    protocol = load_ml_protocol()
    entries = [
        _write_shard(
            tmp_path / "train.npz", fold="train", seed=1
        ),
        _write_shard(
            tmp_path / "selection.npz", fold="selection", seed=2
        ),
        _write_shard(
            tmp_path / "calibration.npz", fold="calibration", seed=3
        ),
        {
            "horizon_ticks": HORIZON,
            "utc_day": "2026-01-04",
            "fold": "test",
            "rows": 999,
            "path": str(tmp_path / "must_not_be_opened.npz"),
            "sha256": None,
        },
    ]
    return {
        "kind": "causal_sequence_dataset",
        "protocol_version": protocol.protocol_version,
        "asset": "BNBUSDT",
        "sequence_length": 128,
        "layout": "channels_first",
        "dtype": "float32",
        "channels": list(
            protocol.raw["models"]["phase_4_sequence_dataset"]["channels"]
        ),
        "normalization": {
            str(HORIZON): {
                "count": 24 * 128,
                "mean": [0.0] * 8,
                "scale": [1.0] * 8,
                "source_fold": "train",
            }
        },
        "shards": entries,
        "test_labels_used_for_normalization": False,
    }


def test_tcn_sequence_logits_are_strictly_causal() -> None:
    torch.manual_seed(2026)
    network = TemporalConvolutionalNetwork(
        2,
        [4, 4],
        kernel_size=3,
        dilations=[1, 2],
        dropout=0.0,
    ).eval()
    original = torch.randn(3, 2, 32)
    mutated = original.clone()
    mutated[..., 17:] += 100.0

    with torch.inference_mode():
        first = network.forward_sequence(original)
        second = network.forward_sequence(mutated)

    torch.testing.assert_close(first[..., :17], second[..., :17])
    assert not torch.equal(first[..., 17:], second[..., 17:])


def test_training_selects_and_calibrates_without_reading_test(
    tmp_path: Path,
) -> None:
    model, diagnostics = train_temporal_convolutional_network(
        _manifest(tmp_path),
        HORIZON,
        _fast_protocol(),
        repo_root=tmp_path,
    )
    with np.load(tmp_path / "selection.npz") as payload:
        probability = model.predict_proba(payload["features"])

    assert probability.shape == (24,)
    assert ((probability > 0.0) & (probability < 1.0)).all()
    assert diagnostics["test_fold_read"] is False
    assert diagnostics["fold_rows"] == {
        "train": 24,
        "selection": 24,
        "calibration": 24,
    }
    assert 1 <= diagnostics["selected_epoch"] <= 3
    assert {row["kind"] for row in diagnostics["calibrators"]} == {
        "identity",
        "platt",
        "isotonic",
    }


def test_training_is_deterministic_and_checkpoint_round_trips(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    protocol = _fast_protocol()
    first, first_diagnostics = train_temporal_convolutional_network(
        manifest, HORIZON, protocol, repo_root=tmp_path
    )
    second, second_diagnostics = train_temporal_convolutional_network(
        manifest, HORIZON, protocol, repo_root=tmp_path
    )
    with np.load(tmp_path / "selection.npz") as payload:
        probe = payload["features"][:8]
    np.testing.assert_array_equal(
        first.predict_proba(probe), second.predict_proba(probe)
    )
    assert (
        first_diagnostics["selected_epoch"]
        == second_diagnostics["selected_epoch"]
    )

    destination = save_temporal_convolutional_model(
        first, tmp_path / "model.pt"
    )
    restored = load_temporal_convolutional_model(destination)
    np.testing.assert_array_equal(
        restored.predict_proba(probe), first.predict_proba(probe)
    )
    assert restored.channel_names == first.channel_names


def test_training_rejects_normalization_that_may_use_test_labels(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["test_labels_used_for_normalization"] = True

    with pytest.raises(ValueError, match="train-only"):
        train_temporal_convolutional_network(
            manifest,
            HORIZON,
            _fast_protocol(),
            repo_root=tmp_path,
        )


def test_training_rejects_an_unregistered_robustness_seed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not registered"):
        train_temporal_convolutional_network(
            _manifest(tmp_path),
            HORIZON,
            _fast_protocol(),
            repo_root=tmp_path,
            random_seed=999,
        )

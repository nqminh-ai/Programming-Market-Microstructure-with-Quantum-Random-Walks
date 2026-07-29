"""Tests for Phase 6 pre-holdout TCN robustness diagnostics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.ml_protocol import load_ml_protocol
from src.evaluation.tcn_robustness import (
    aggregate_cross_asset_robustness,
    evaluate_tcn_robustness,
)
from src.models.temporal_convolutional_direction import (
    train_temporal_convolutional_network,
)
from src.models.neural_adaptive_qrw import (
    NeuralAdaptiveQRWNetwork,
    build_hybrid_ablation_report,
    complex_qrw_right_probability,
    load_neural_adaptive_qrw_model,
    save_neural_adaptive_qrw_model,
    train_neural_adaptive_qrw,
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
            "batch_size": 12,
            "max_epochs": 2,
            "early_stopping_patience": 2,
        }
    )
    raw["robustness"]["minimum_group_rows"] = 2
    return replace(base, raw=raw)


def _write_shard(
    path: Path,
    *,
    fold: str,
    day: str,
    seed: int,
    rows: int = 30,
) -> dict:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 8, 128)).astype(np.float32)
    target = np.tile(np.array([0, 1], dtype=np.int8), rows // 2)
    features[:, 0, -1] = np.where(target == 1, 2.0, -2.0)
    features[:, 5, :] *= np.linspace(0.2, 2.0, rows)[:, None]
    features[:, 7, :] = np.abs(features[:, 7, :])
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
        "utc_day": day,
        "fold": fold,
        "rows": rows,
        "path": str(path),
        "sha256": None,
    }


def _manifest(tmp_path: Path) -> dict:
    protocol = load_ml_protocol()
    channels = protocol.raw["models"]["phase_4_sequence_dataset"]["channels"]
    return {
        "kind": "causal_sequence_dataset",
        "protocol_version": protocol.protocol_version,
        "asset": "BNBUSDT",
        "sequence_length": 128,
        "layout": "channels_first",
        "dtype": "float32",
        "channels": list(channels),
        "normalization": {
            str(HORIZON): {
                "count": 30 * 128,
                "mean": [0.0] * 8,
                "scale": [1.0] * 8,
                "source_fold": "train",
            }
        },
        "shards": [
            _write_shard(
                tmp_path / "train.npz",
                fold="train",
                day="2026-01-01",
                seed=1,
            ),
            _write_shard(
                tmp_path / "selection.npz",
                fold="selection",
                day="2026-01-02",
                seed=2,
            ),
            _write_shard(
                tmp_path / "calibration.npz",
                fold="calibration",
                day="2026-01-03",
                seed=3,
            ),
            {
                "horizon_ticks": HORIZON,
                "utc_day": "2026-01-04",
                "fold": "test",
                "rows": 999,
                "path": str(tmp_path / "must_not_be_opened.npz"),
                "sha256": None,
            },
        ],
        "test_labels_used_for_normalization": False,
    }


def _models(manifest: dict, protocol, tmp_path: Path):
    return {
        seed: train_temporal_convolutional_network(
            manifest,
            HORIZON,
            protocol,
            repo_root=tmp_path,
            random_seed=seed,
        )[0]
        for seed in protocol.raw["robustness"]["seeds"]
    }


def test_registered_seed_fold_and_regime_diagnostics_do_not_read_test(
    tmp_path: Path,
) -> None:
    protocol = _fast_protocol()
    manifest = _manifest(tmp_path)
    report = evaluate_tcn_robustness(
        manifest,
        _models(manifest, protocol, tmp_path),
        HORIZON,
        protocol,
        repo_root=tmp_path,
    )

    assert report["seeds"] == [2026, 2027, 2028]
    assert report["evaluation_folds"] == ["selection", "calibration"]
    assert report["regime_thresholds"]["source_fold"] == "train"
    assert set(report["multi_seed"]) == {"selection", "calibration"}
    assert report["test_fold_read"] is False
    assert report["test_metrics"] is None
    for seed in report["seeds"]:
        folds = report["seed_results"][str(seed)]["folds"]
        assert folds["selection"]["overall"]["rows"] == 30
        assert (
            folds["calibration"]["regimes"]["volatility"]["high"]["status"]
            == "evaluated"
        )


def test_robustness_requires_every_preregistered_seed(
    tmp_path: Path,
) -> None:
    protocol = _fast_protocol()
    manifest = _manifest(tmp_path)
    one_model = train_temporal_convolutional_network(
        manifest,
        HORIZON,
        protocol,
        repo_root=tmp_path,
        random_seed=2026,
    )[0]

    with pytest.raises(ValueError, match="exactly the registered seed"):
        evaluate_tcn_robustness(
            manifest,
            {2026: one_model},
            HORIZON,
            protocol,
            repo_root=tmp_path,
        )


def test_cross_asset_aggregation_requires_all_registered_assets(
    tmp_path: Path,
) -> None:
    protocol = _fast_protocol()
    manifest = _manifest(tmp_path)
    report = evaluate_tcn_robustness(
        manifest,
        _models(manifest, protocol, tmp_path),
        HORIZON,
        protocol,
        repo_root=tmp_path,
    )
    reports = []
    for index, asset in enumerate(protocol.assets):
        asset_report = deepcopy(report)
        asset_report["asset"] = asset
        asset_report["multi_seed"]["selection"]["mean_brier"] += (
            index * 0.01
        )
        reports.append(asset_report)

    aggregate = aggregate_cross_asset_robustness(reports, protocol)

    assert aggregate["assets"] == list(protocol.assets)
    assert aggregate["cross_asset_policy"] == (
        "refit_per_asset_equal_weight_summary"
    )
    assert aggregate["test_fold_read"] is False
    with pytest.raises(ValueError, match="one robustness report"):
        aggregate_cross_asset_robustness(reports[:2], protocol)


def test_complex_qrw_is_bounded_symmetric_and_differentiable() -> None:
    signal = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    theta = torch.full_like(signal, np.pi / 4, requires_grad=True)
    gamma = torch.full_like(signal, 0.05, requires_grad=True)

    probability = complex_qrw_right_probability(
        signal, theta, gamma, steps=8
    )

    assert ((probability > 0.0) & (probability < 1.0)).all()
    assert probability[0] < probability[2]
    assert float(probability[1].detach()) == pytest.approx(0.5, abs=1e-5)
    probability.sum().backward()
    assert torch.isfinite(signal.grad).all()
    assert torch.isfinite(theta.grad).all()
    assert torch.isfinite(gamma.grad).all()


def test_adaptive_coin_and_decoherence_stay_inside_frozen_bounds() -> None:
    network = NeuralAdaptiveQRWNetwork(
        8,
        [4],
        kernel_size=2,
        dilations=[1],
        dropout=0.0,
        qrw_steps=8,
        signal_channel_index=0,
        theta_bounds=[np.pi / 8, 3 * np.pi / 8],
        gamma_bounds=[0.0, 0.35],
        fixed_theta=np.pi / 4,
        fixed_gamma=0.05,
        neural_weight=0.5,
    )
    components = network.components(torch.randn(6, 8, 128))

    assert (components["theta"] >= np.pi / 8).all()
    assert (components["theta"] <= 3 * np.pi / 8).all()
    assert (components["gamma"] >= 0.0).all()
    assert (components["gamma"] <= 0.35).all()


def test_hybrid_suite_requires_and_trains_every_ablation(
    tmp_path: Path,
) -> None:
    protocol = _fast_protocol()
    manifest = _manifest(tmp_path)
    models = {}
    diagnostics = {}
    variants = protocol.raw["models"]["phase_7_neural_adaptive_qrw"][
        "variants"
    ]
    for variant in variants:
        model, values = train_neural_adaptive_qrw(
            manifest,
            HORIZON,
            variant,
            protocol,
            repo_root=tmp_path,
        )
        models[variant] = model
        diagnostics[variant] = values

    report = build_hybrid_ablation_report(diagnostics, protocol)

    assert set(report["ranked_by_selection_brier"]) == set(variants)
    assert diagnostics["fixed_qrw_only"]["selected_epoch"] == 0
    assert report["test_fold_read"] is False
    with np.load(tmp_path / "selection.npz") as payload:
        probe = payload["features"][:5]
    probability = models["hybrid_adaptive_qrw"].predict_proba(probe)
    assert probability.shape == (5,)
    destination = save_neural_adaptive_qrw_model(
        models["hybrid_adaptive_qrw"], tmp_path / "hybrid.pt"
    )
    restored = load_neural_adaptive_qrw_model(destination)
    np.testing.assert_array_equal(
        restored.predict_proba(probe), probability
    )

    incomplete = dict(diagnostics)
    incomplete.pop("neural_only")
    with pytest.raises(ValueError, match="every registered variant"):
        build_hybrid_ablation_report(incomplete, protocol)

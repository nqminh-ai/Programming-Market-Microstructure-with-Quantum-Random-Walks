"""Differentiable causal QRW hybrid with mandatory independently trained ablations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from src.data.sequence_dataset import SequenceFeatureSpec
from src.evaluation.ml_protocol import MLDirectionalProtocol
from src.models.gradient_boosted_direction import (
    ProbabilityCalibrator,
    fit_probability_calibrator,
)
from src.models.temporal_convolutional_direction import (
    TemporalResidualBlock,
    _brier,
    _clip_probability,
    _entries_for_fold,
    _iter_training_batches,
    _load_fold,
    _log_loss,
    _normalization,
    _resolve_device,
    _validate_shard_hashes,
)


MODEL_FORMAT_VERSION = "neural_adaptive_qrw_directional_v1"


def complex_qrw_right_probability(
    anchor_signal: torch.Tensor,
    theta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Evolve a unitary two-state walk and return decohered right-side mass."""
    if steps <= 0:
        raise ValueError("qrw steps must be positive")
    signal, angle, decoherence = torch.broadcast_tensors(
        anchor_signal.flatten(), theta.flatten(), gamma.flatten()
    )
    batch = len(signal)
    positions = 2 * steps + 1
    center = steps
    right_weight = torch.sigmoid(signal)
    state = torch.zeros(
        batch,
        2,
        positions,
        dtype=torch.complex64,
        device=signal.device,
    )
    state[:, 0, center] = torch.sqrt(1.0 - right_weight).to(
        torch.complex64
    )
    state[:, 1, center] = (
        1j * torch.sqrt(right_weight).to(torch.complex64)
    )
    cosine = torch.cos(angle).to(torch.complex64)[:, None]
    sine = torch.sin(angle).to(torch.complex64)[:, None]
    for _ in range(steps):
        left = cosine * state[:, 0] + sine * state[:, 1]
        right = sine * state[:, 0] - cosine * state[:, 1]
        shifted = torch.zeros_like(state)
        shifted[:, 0, :-1] = left[:, 1:]
        shifted[:, 1, 1:] = right[:, :-1]
        state = shifted
    probability = state.abs().square().sum(dim=1)
    unitary_right = (
        probability[:, center + 1 :].sum(dim=1)
        + 0.5 * probability[:, center]
    )
    coherence = torch.exp(-torch.clamp(decoherence, min=0.0) * steps)
    return torch.clamp(
        0.5 + coherence * (unitary_right - 0.5),
        1e-6,
        1.0 - 1e-6,
    )


class NeuralAdaptiveQRWNetwork(nn.Module):
    """TCN backbone with bounded adaptive coin and decoherence heads."""

    def __init__(
        self,
        input_channels: int,
        residual_channels: Sequence[int],
        *,
        kernel_size: int,
        dilations: Sequence[int],
        dropout: float,
        qrw_steps: int,
        signal_channel_index: int,
        theta_bounds: Sequence[float],
        gamma_bounds: Sequence[float],
        fixed_theta: float,
        fixed_gamma: float,
        neural_weight: float,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in residual_channels)
        rates = tuple(int(value) for value in dilations)
        if not widths or len(widths) != len(rates):
            raise ValueError("hybrid residual channels and dilations must align")
        blocks: list[nn.Module] = []
        current = int(input_channels)
        for width, dilation in zip(widths, rates, strict=True):
            blocks.append(
                TemporalResidualBlock(
                    current,
                    width,
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                )
            )
            current = width
        self.blocks = nn.Sequential(*blocks)
        self.neural_head = nn.Linear(current, 1)
        self.theta_head = nn.Linear(current, 1)
        self.gamma_head = nn.Linear(current, 1)
        self.qrw_steps = int(qrw_steps)
        self.signal_channel_index = int(signal_channel_index)
        self.theta_bounds = (float(theta_bounds[0]), float(theta_bounds[1]))
        self.gamma_bounds = (float(gamma_bounds[0]), float(gamma_bounds[1]))
        self.fixed_theta = float(fixed_theta)
        self.fixed_gamma = float(fixed_gamma)
        self.neural_weight = float(neural_weight)
        if (
            self.qrw_steps <= 0
            or not 0 <= self.signal_channel_index < input_channels
            or not self.theta_bounds[0] < self.theta_bounds[1]
            or not 0.0 <= self.gamma_bounds[0] < self.gamma_bounds[1]
            or not 0.0 < self.neural_weight < 1.0
        ):
            raise ValueError("invalid neural-adaptive QRW bounds")

    def components(
        self, values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        hidden = self.blocks(values)[..., -1]
        neural = torch.sigmoid(self.neural_head(hidden).squeeze(1))
        theta_unit = torch.sigmoid(self.theta_head(hidden).squeeze(1))
        gamma_unit = torch.sigmoid(self.gamma_head(hidden).squeeze(1))
        theta = self.theta_bounds[0] + (
            self.theta_bounds[1] - self.theta_bounds[0]
        ) * theta_unit
        gamma = self.gamma_bounds[0] + (
            self.gamma_bounds[1] - self.gamma_bounds[0]
        ) * gamma_unit
        signal = values[:, self.signal_channel_index, -1]
        adaptive_qrw = complex_qrw_right_probability(
            signal, theta, gamma, steps=self.qrw_steps
        )
        fixed_qrw = complex_qrw_right_probability(
            signal,
            torch.full_like(signal, self.fixed_theta),
            torch.full_like(signal, self.fixed_gamma),
            steps=self.qrw_steps,
        )
        return {
            "neural": neural,
            "adaptive_qrw": adaptive_qrw,
            "fixed_qrw": fixed_qrw,
            "theta": theta,
            "gamma": gamma,
        }

    def forward(self, values: torch.Tensor, variant: str) -> torch.Tensor:
        components = self.components(values)
        if variant == "neural_only":
            return components["neural"]
        if variant == "fixed_qrw_only":
            return components["fixed_qrw"]
        if variant == "adaptive_qrw_only":
            return components["adaptive_qrw"]
        if variant == "hybrid_fixed_qrw":
            qrw = components["fixed_qrw"]
        elif variant == "hybrid_adaptive_qrw":
            qrw = components["adaptive_qrw"]
        else:
            raise ValueError(f"unsupported hybrid variant: {variant}")
        return (
            self.neural_weight * components["neural"]
            + (1.0 - self.neural_weight) * qrw
        )


@dataclass
class NeuralAdaptiveQRWDirectionalModel:
    """One independently fitted Phase 7 ablation model."""

    network: NeuralAdaptiveQRWNetwork
    calibrator: ProbabilityCalibrator
    normalization_mean: np.ndarray
    normalization_scale: np.ndarray
    channel_names: tuple[str, ...]
    sequence_length: int
    variant: str
    network_parameters: Mapping[str, Any]
    protocol_version: str
    random_seed: int
    horizon_ticks: int

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        matrix = np.asarray(values, dtype=np.float32)
        expected = (
            len(self.channel_names),
            int(self.sequence_length),
        )
        if matrix.ndim != 3 or matrix.shape[1:] != expected:
            raise ValueError(
                f"expected [rows, {expected[0]}, {expected[1]}] sequences"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("hybrid prediction values must be finite")
        normalized = (
            matrix - self.normalization_mean[None, :, None]
        ) / self.normalization_scale[None, :, None]
        return torch.from_numpy(normalized.astype(np.float32, copy=False))

    def predict_uncalibrated(
        self,
        values: np.ndarray,
        *,
        batch_size: int = 1024,
        device: str = "cpu",
    ) -> np.ndarray:
        selected_device = _resolve_device(device)
        tensor = self._tensor(values)
        self.network.to(selected_device).eval()
        result: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(tensor), batch_size):
                probability = self.network(
                    tensor[start : start + batch_size].to(selected_device),
                    self.variant,
                )
                result.append(probability.cpu().numpy())
        if not result:
            return np.empty(0, dtype=np.float64)
        return _clip_probability(np.concatenate(result))

    def predict_proba(
        self,
        values: np.ndarray,
        *,
        batch_size: int = 1024,
        device: str = "cpu",
    ) -> np.ndarray:
        return self.calibrator.predict(
            self.predict_uncalibrated(
                values, batch_size=batch_size, device=device
            )
        )


def _network_parameters(
    protocol: MLDirectionalProtocol,
    spec: SequenceFeatureSpec,
) -> dict[str, Any]:
    tcn = protocol.raw["models"]["phase_5_temporal_convolutional"]
    hybrid = protocol.raw["models"]["phase_7_neural_adaptive_qrw"]
    return {
        "input_channels": len(spec.channels),
        "residual_channels": list(tcn["residual_channels"]),
        "kernel_size": int(tcn["kernel_size"]),
        "dilations": list(tcn["dilations"]),
        "dropout": float(tcn["dropout"]),
        "qrw_steps": int(hybrid["qrw_steps"]),
        "signal_channel_index": spec.channels.index(
            hybrid["anchor_signal_channel"]
        ),
        "theta_bounds": list(hybrid["theta_bounds_radians"]),
        "gamma_bounds": list(hybrid["gamma_bounds"]),
        "fixed_theta": float(hybrid["fixed_theta_radians"]),
        "fixed_gamma": float(hybrid["fixed_gamma"]),
        "neural_weight": float(hybrid["hybrid_neural_weight"]),
    }


def _build_network(
    parameters: Mapping[str, Any],
) -> NeuralAdaptiveQRWNetwork:
    return NeuralAdaptiveQRWNetwork(**dict(parameters))


def _predict_network(
    network: NeuralAdaptiveQRWNetwork,
    features: np.ndarray,
    variant: str,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    network.eval()
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            values = torch.from_numpy(
                features[start : start + batch_size]
            ).to(device)
            result.append(network(values, variant).cpu().numpy())
    return _clip_probability(np.concatenate(result))


def train_neural_adaptive_qrw(
    manifest: Mapping[str, Any],
    horizon_ticks: int,
    variant: str,
    protocol: MLDirectionalProtocol,
    *,
    repo_root: str | Path,
    device: str = "cpu",
) -> tuple[NeuralAdaptiveQRWDirectionalModel, dict[str, Any]]:
    """Fit one registered ablation without accessing test shards."""
    hybrid = protocol.raw["models"]["phase_7_neural_adaptive_qrw"]
    variants = tuple(str(value) for value in hybrid["variants"])
    if variant not in variants:
        raise ValueError("hybrid variant is not preregistered")
    if hybrid["test_access"] != "forbidden":
        raise ValueError("Phase 7 must forbid test access")
    horizon = int(horizon_ticks)
    if horizon not in protocol.evaluation_horizons:
        raise ValueError("horizon is not registered")
    if (
        manifest.get("kind") != "causal_sequence_dataset"
        or manifest.get("protocol_version") != protocol.protocol_version
        or manifest.get("test_labels_used_for_normalization") is not False
    ):
        raise ValueError("incompatible or leakage-unsafe sequence manifest")
    spec = SequenceFeatureSpec.from_protocol(protocol.raw)
    if (
        tuple(manifest.get("channels", ())) != spec.channels
        or int(manifest.get("sequence_length", 0)) != spec.sequence_length
        or manifest.get("layout") != spec.layout
        or manifest.get("dtype") != spec.dtype
    ):
        raise ValueError("sequence manifest schema does not match protocol")
    root = Path(repo_root).resolve()
    selected_device = _resolve_device(device)
    train_entries = _entries_for_fold(manifest, horizon, "train")
    selection_entries = _entries_for_fold(manifest, horizon, "selection")
    calibration_entries = _entries_for_fold(manifest, horizon, "calibration")
    _validate_shard_hashes(
        [*train_entries, *selection_entries, *calibration_entries], root
    )
    mean, scale = _normalization(manifest, horizon, len(spec.channels))
    selection_x, selection_y = _load_fold(
        selection_entries,
        root=root,
        channels=len(spec.channels),
        sequence_length=spec.sequence_length,
        mean=mean,
        scale=scale,
    )
    calibration_x, calibration_y = _load_fold(
        calibration_entries,
        root=root,
        channels=len(spec.channels),
        sequence_length=spec.sequence_length,
        mean=mean,
        scale=scale,
    )
    settings = protocol.raw["models"]["phase_5_temporal_convolutional"]
    parameters = _network_parameters(protocol, spec)
    torch.manual_seed(protocol.random_seed)
    torch.use_deterministic_algorithms(True)
    network = _build_network(parameters).to(selected_device)
    batch_size = int(settings["batch_size"])
    epoch_rows: list[dict[str, Any]] = []
    best_epoch = 0
    if variant == "fixed_qrw_only":
        selection_probability = _predict_network(
            network,
            selection_x,
            variant,
            batch_size=batch_size,
            device=selected_device,
        )
        best_state = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in network.state_dict().items()
        )
        best_key = (
            _brier(selection_probability, selection_y),
            _log_loss(selection_probability, selection_y),
        )
        epoch_rows.append(
            {
                "epoch": 0,
                "train_log_loss": None,
                "selection_brier": best_key[0],
                "selection_log_loss": best_key[1],
            }
        )
    else:
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        best_key = (float("inf"), float("inf"))
        best_state: OrderedDict[str, torch.Tensor] | None = None
        stale_epochs = 0
        for epoch in range(1, int(settings["max_epochs"]) + 1):
            network.train()
            loss_sum = 0.0
            row_count = 0
            for batch_x, batch_y in _iter_training_batches(
                train_entries,
                root=root,
                channels=len(spec.channels),
                sequence_length=spec.sequence_length,
                mean=mean,
                scale=scale,
                batch_size=batch_size,
                random_seed=protocol.random_seed + epoch,
            ):
                batch_x = batch_x.to(selected_device)
                batch_y = batch_y.to(selected_device)
                optimizer.zero_grad(set_to_none=True)
                probability = torch.clamp(
                    network(batch_x, variant), 1e-6, 1.0 - 1e-6
                )
                loss = functional.binary_cross_entropy(
                    probability, batch_y
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    network.parameters(),
                    float(settings["gradient_clip_norm"]),
                )
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(batch_y)
                row_count += len(batch_y)
            selection_probability = _predict_network(
                network,
                selection_x,
                variant,
                batch_size=batch_size,
                device=selected_device,
            )
            key = (
                _brier(selection_probability, selection_y),
                _log_loss(selection_probability, selection_y),
            )
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "train_log_loss": loss_sum / row_count,
                    "selection_brier": key[0],
                    "selection_log_loss": key[1],
                }
            )
            if key < best_key:
                best_key = key
                best_epoch = epoch
                best_state = OrderedDict(
                    (name, value.detach().cpu().clone())
                    for name, value in network.state_dict().items()
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(
                    settings["early_stopping_patience"]
                ):
                    break
        if best_state is None:
            raise RuntimeError("hybrid training produced no checkpoint")
    network.load_state_dict(best_state)
    network.to(selected_device)
    raw_calibration = _predict_network(
        network,
        calibration_x,
        variant,
        batch_size=batch_size,
        device=selected_device,
    )
    calibration_rows: list[dict[str, Any]] = []
    calibrators: list[ProbabilityCalibrator] = []
    for kind in hybrid["probability_calibrators"]:
        calibrator = fit_probability_calibrator(
            str(kind), raw_calibration, calibration_y
        )
        calibrated = calibrator.predict(raw_calibration)
        calibration_rows.append(
            {
                "kind": str(kind),
                "calibration_brier": _brier(calibrated, calibration_y),
                "calibration_log_loss": _log_loss(
                    calibrated, calibration_y
                ),
            }
        )
        calibrators.append(calibrator)
    selected_index = min(
        range(len(calibration_rows)),
        key=lambda index: (
            calibration_rows[index]["calibration_brier"],
            calibration_rows[index]["calibration_log_loss"],
            index,
        ),
    )
    calibrator = calibrators[selected_index]
    network.to("cpu")
    model = NeuralAdaptiveQRWDirectionalModel(
        network=network,
        calibrator=calibrator,
        normalization_mean=mean,
        normalization_scale=scale,
        channel_names=spec.channels,
        sequence_length=spec.sequence_length,
        variant=variant,
        network_parameters=parameters,
        protocol_version=protocol.protocol_version,
        random_seed=protocol.random_seed,
        horizon_ticks=horizon,
    )
    diagnostics = {
        "kind": "neural_adaptive_qrw_training",
        "status": "trained_calibrated_without_test_access",
        "protocol_version": protocol.protocol_version,
        "random_seed": protocol.random_seed,
        "horizon_ticks": horizon,
        "variant": variant,
        "independently_trained_ablation": True,
        "network_parameters": parameters,
        "fold_rows": {
            "train": sum(int(entry["rows"]) for entry in train_entries),
            "selection": len(selection_y),
            "calibration": len(calibration_y),
        },
        "epochs": epoch_rows,
        "selected_epoch": best_epoch,
        "selection_brier": best_key[0],
        "selection_log_loss": best_key[1],
        "calibrators": calibration_rows,
        "selected_calibrator": calibrator.kind,
        "normalization_source_fold": "train",
        "training_policy": hybrid["training_policy"],
        "test_fold_read": False,
        "test_metrics": None,
    }
    return model, diagnostics


def build_hybrid_ablation_report(
    diagnostics: Mapping[str, Mapping[str, Any]],
    protocol: MLDirectionalProtocol,
) -> dict[str, Any]:
    """Require and rank every preregistered independently trained ablation."""
    variants = tuple(
        protocol.raw["models"]["phase_7_neural_adaptive_qrw"]["variants"]
    )
    if set(diagnostics) != set(variants):
        raise ValueError("ablation report requires every registered variant")
    rows: list[dict[str, Any]] = []
    for variant in variants:
        values = diagnostics[variant]
        if (
            values.get("variant") != variant
            or values.get("protocol_version") != protocol.protocol_version
            or values.get("independently_trained_ablation") is not True
            or values.get("test_fold_read") is not False
            or values.get("test_metrics") is not None
        ):
            raise ValueError(f"invalid ablation diagnostics: {variant}")
        selected_calibrator = values["selected_calibrator"]
        calibration = next(
            row
            for row in values["calibrators"]
            if row["kind"] == selected_calibrator
        )
        rows.append(
            {
                "variant": variant,
                "selection_brier": float(values["selection_brier"]),
                "selection_log_loss": float(values["selection_log_loss"]),
                "calibration_brier": float(
                    calibration["calibration_brier"]
                ),
                "selected_epoch": int(values["selected_epoch"]),
                "selected_calibrator": selected_calibrator,
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            row["selection_brier"],
            row["selection_log_loss"],
            variants.index(row["variant"]),
        ),
    )
    by_variant = {row["variant"]: row for row in rows}
    full = by_variant["hybrid_adaptive_qrw"]["selection_brier"]
    return {
        "kind": "neural_adaptive_qrw_ablation",
        "status": "exploratory_pretest_ablation",
        "protocol_version": protocol.protocol_version,
        "variants": rows,
        "ranked_by_selection_brier": [
            row["variant"] for row in ranked
        ],
        "full_minus_neural_selection_brier": (
            full - by_variant["neural_only"]["selection_brier"]
        ),
        "full_minus_adaptive_qrw_selection_brier": (
            full - by_variant["adaptive_qrw_only"]["selection_brier"]
        ),
        "full_minus_hybrid_fixed_selection_brier": (
            full - by_variant["hybrid_fixed_qrw"]["selection_brier"]
        ),
        "test_fold_read": False,
        "test_metrics": None,
    }


def _calibrator_payload(
    calibrator: ProbabilityCalibrator,
) -> dict[str, Any]:
    return {
        "kind": calibrator.kind,
        "parameters": calibrator.parameters.tolist(),
        "thresholds": calibrator.thresholds.tolist(),
        "probabilities": calibrator.probabilities.tolist(),
    }


def save_neural_adaptive_qrw_model(
    model: NeuralAdaptiveQRWDirectionalModel,
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.network.to("cpu")
    torch.save(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "state_dict": model.network.state_dict(),
            "calibrator": _calibrator_payload(model.calibrator),
            "normalization_mean": model.normalization_mean.tolist(),
            "normalization_scale": model.normalization_scale.tolist(),
            "channel_names": list(model.channel_names),
            "sequence_length": model.sequence_length,
            "variant": model.variant,
            "network_parameters": dict(model.network_parameters),
            "protocol_version": model.protocol_version,
            "random_seed": model.random_seed,
            "horizon_ticks": model.horizon_ticks,
        },
        destination,
    )
    return destination


def load_neural_adaptive_qrw_model(
    path: str | Path,
) -> NeuralAdaptiveQRWDirectionalModel:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("format_version") != MODEL_FORMAT_VERSION
    ):
        raise ValueError("unsupported neural-adaptive QRW artifact")
    parameters = dict(payload["network_parameters"])
    network = _build_network(parameters)
    network.load_state_dict(payload["state_dict"])
    network.eval()
    calibrator_values = payload["calibrator"]
    calibrator = ProbabilityCalibrator(
        kind=str(calibrator_values["kind"]),
        parameters=np.asarray(
            calibrator_values["parameters"], dtype=np.float64
        ),
        thresholds=np.asarray(
            calibrator_values["thresholds"], dtype=np.float64
        ),
        probabilities=np.asarray(
            calibrator_values["probabilities"], dtype=np.float64
        ),
    )
    return NeuralAdaptiveQRWDirectionalModel(
        network=network,
        calibrator=calibrator,
        normalization_mean=np.asarray(
            payload["normalization_mean"], dtype=np.float32
        ),
        normalization_scale=np.asarray(
            payload["normalization_scale"], dtype=np.float32
        ),
        channel_names=tuple(payload["channel_names"]),
        sequence_length=int(payload["sequence_length"]),
        variant=str(payload["variant"]),
        network_parameters=parameters,
        protocol_version=str(payload["protocol_version"]),
        random_seed=int(payload["random_seed"]),
        horizon_ticks=int(payload["horizon_ticks"]),
    )

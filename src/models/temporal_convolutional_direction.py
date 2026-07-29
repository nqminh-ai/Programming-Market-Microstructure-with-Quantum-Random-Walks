"""Causal Temporal Convolutional Network for directional probabilities."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from src.data.sequence_dataset import SequenceFeatureSpec
from src.evaluation.ml_protocol import MLDirectionalProtocol
from src.evaluation.provenance import sha256_file
from src.models.gradient_boosted_direction import (
    ProbabilityCalibrator,
    fit_probability_calibrator,
)


MODEL_FORMAT_VERSION = "temporal_convolutional_directional_v1"
TRAINING_FOLDS = ("train", "selection", "calibration")


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6
    )


def _brier(probability: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((_clip_probability(probability) - target) ** 2))


def _log_loss(probability: np.ndarray, target: np.ndarray) -> float:
    p = _clip_probability(probability)
    y = np.asarray(target, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


class CausalConv1d(nn.Conv1d):
    """Conv1d that preserves length without seeing future timesteps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int,
    ) -> None:
        self.left_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.left_padding,
            dilation=dilation,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        result = super().forward(values)
        if self.left_padding:
            result = result[..., : -self.left_padding]
        return result


class TemporalResidualBlock(nn.Module):
    """Two-layer causal residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.projection(values)
        hidden = self.dropout(functional.relu(self.conv1(values)))
        hidden = self.dropout(functional.relu(self.conv2(hidden)))
        return functional.relu(hidden + residual)


class TemporalConvolutionalNetwork(nn.Module):
    """Registered causal TCN with a logit at every timestep."""

    def __init__(
        self,
        input_channels: int,
        residual_channels: Sequence[int],
        *,
        kernel_size: int,
        dilations: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in residual_channels)
        rates = tuple(int(value) for value in dilations)
        if not widths or len(widths) != len(rates):
            raise ValueError(
                "residual_channels and dilations must be non-empty and aligned"
            )
        if input_channels <= 0 or min(widths) <= 0 or min(rates) <= 0:
            raise ValueError("TCN channel widths and dilations must be positive")
        if kernel_size < 2 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid TCN kernel size or dropout")
        blocks: list[nn.Module] = []
        current = int(input_channels)
        for width, dilation in zip(widths, rates, strict=True):
            blocks.append(
                TemporalResidualBlock(
                    current,
                    width,
                    kernel_size=int(kernel_size),
                    dilation=dilation,
                    dropout=float(dropout),
                )
            )
            current = width
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Conv1d(current, 1, kernel_size=1)

    def forward_sequence(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("TCN input must have shape [batch, channels, time]")
        return self.output(self.blocks(values)).squeeze(1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(values)[..., -1]


@dataclass
class TemporalConvolutionalDirectionalModel:
    """Fitted TCN and calibration mapping frozen before test access."""

    network: TemporalConvolutionalNetwork
    calibrator: ProbabilityCalibrator
    normalization_mean: np.ndarray
    normalization_scale: np.ndarray
    channel_names: tuple[str, ...]
    sequence_length: int
    architecture: Mapping[str, Any]
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
                "expected sequence tensor [rows, "
                f"{expected[0]} channels, {expected[1]} timesteps]"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("prediction sequence values must be finite")
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
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected_device = _resolve_device(device)
        tensor = self._tensor(values)
        self.network.to(selected_device)
        self.network.eval()
        probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(tensor), batch_size):
                logits = self.network(
                    tensor[start : start + batch_size].to(selected_device)
                )
                probabilities.append(
                    torch.sigmoid(logits).cpu().numpy().astype(np.float64)
                )
        if not probabilities:
            return np.empty(0, dtype=np.float64)
        return _clip_probability(np.concatenate(probabilities))

    def predict_proba(
        self,
        values: np.ndarray,
        *,
        batch_size: int = 1024,
        device: str = "cpu",
    ) -> np.ndarray:
        raw = self.predict_uncalibrated(
            values, batch_size=batch_size, device=device
        )
        return self.calibrator.predict(raw)


def _resolve_device(value: str) -> torch.device:
    normalized = str(value).lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda")
    raise ValueError("device must be 'cpu' or 'cuda'")


def _resolve_artifact_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _entries_for_fold(
    manifest: Mapping[str, Any],
    horizon: int,
    fold: str,
) -> list[Mapping[str, Any]]:
    if fold not in TRAINING_FOLDS:
        raise ValueError(f"Phase 5 cannot access fold: {fold}")
    entries = [
        entry
        for entry in manifest.get("shards", ())
        if int(entry["horizon_ticks"]) == horizon and entry["fold"] == fold
    ]
    if not entries:
        raise ValueError(f"sequence manifest has no {fold} shards at h={horizon}")
    return sorted(entries, key=lambda entry: str(entry["utc_day"]))


def _load_shard(
    entry: Mapping[str, Any],
    *,
    root: Path,
    channels: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = _resolve_artifact_path(str(entry["path"]), root)
    with np.load(path, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32, copy=False)
        target = payload["target"].astype(np.float32, copy=False)
    expected = (int(entry["rows"]), channels, sequence_length)
    if features.shape != expected:
        raise ValueError(f"sequence shard shape mismatch: {path}")
    if target.shape != (expected[0],):
        raise ValueError(f"sequence shard target shape mismatch: {path}")
    if not np.isfinite(features).all():
        raise ValueError(f"sequence shard contains nonfinite values: {path}")
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError(f"sequence shard target must be binary: {path}")
    return features, target


def _validate_shard_hashes(
    entries: Sequence[Mapping[str, Any]],
    root: Path,
) -> None:
    for entry in entries:
        registered = entry.get("sha256")
        if not registered:
            continue
        path = _resolve_artifact_path(str(entry["path"]), root)
        if sha256_file(path) != registered:
            raise ValueError(f"sequence shard SHA-256 mismatch: {path}")


def _normalization(
    manifest: Mapping[str, Any],
    horizon: int,
    channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = manifest.get("normalization", {}).get(str(horizon))
    if not isinstance(values, Mapping) or values.get("source_fold") != "train":
        raise ValueError("TCN normalization must come from the train fold")
    mean = np.asarray(values.get("mean"), dtype=np.float32)
    scale = np.asarray(values.get("scale"), dtype=np.float32)
    if (
        mean.shape != (channels,)
        or scale.shape != (channels,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("invalid sequence normalization statistics")
    return mean, scale


def _normalize(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (
        (features - mean[None, :, None]) / scale[None, :, None]
    ).astype(np.float32, copy=False)


def _load_fold(
    entries: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    channels: int,
    sequence_length: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    for entry in entries:
        features, target = _load_shard(
            entry,
            root=root,
            channels=channels,
            sequence_length=sequence_length,
        )
        feature_blocks.append(_normalize(features, mean, scale))
        target_blocks.append(target)
    features = np.concatenate(feature_blocks)
    target = np.concatenate(target_blocks)
    if len(np.unique(target)) != 2:
        raise ValueError("each training fold must contain both target classes")
    return features, target


def _iter_training_batches(
    entries: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    channels: int,
    sequence_length: int,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
    random_seed: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(random_seed)
    shard_order = rng.permutation(len(entries))
    for shard_index in shard_order:
        features, target = _load_shard(
            entries[int(shard_index)],
            root=root,
            channels=channels,
            sequence_length=sequence_length,
        )
        features = _normalize(features, mean, scale)
        row_order = rng.permutation(len(target))
        for start in range(0, len(target), batch_size):
            indices = row_order[start : start + batch_size]
            yield (
                torch.from_numpy(features[indices]),
                torch.from_numpy(target[indices]),
            )


def _predict_network(
    network: TemporalConvolutionalNetwork,
    features: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    network.eval()
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                features[start : start + batch_size]
            ).to(device)
            result.append(torch.sigmoid(network(batch)).cpu().numpy())
    return _clip_probability(np.concatenate(result))


def _architecture(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "residual_channels": [
            int(value) for value in settings["residual_channels"]
        ],
        "kernel_size": int(settings["kernel_size"]),
        "dilations": [int(value) for value in settings["dilations"]],
        "dropout": float(settings["dropout"]),
        "pooling": str(settings["pooling"]),
    }


def _build_network(
    input_channels: int,
    architecture: Mapping[str, Any],
) -> TemporalConvolutionalNetwork:
    return TemporalConvolutionalNetwork(
        input_channels,
        architecture["residual_channels"],
        kernel_size=int(architecture["kernel_size"]),
        dilations=architecture["dilations"],
        dropout=float(architecture["dropout"]),
    )


def train_temporal_convolutional_network(
    manifest: Mapping[str, Any],
    horizon_ticks: int,
    protocol: MLDirectionalProtocol,
    *,
    repo_root: str | Path,
    device: str = "cpu",
    random_seed: int | None = None,
) -> tuple[TemporalConvolutionalDirectionalModel, dict[str, Any]]:
    """Train, select and calibrate a TCN without opening the test fold."""
    horizon = int(horizon_ticks)
    if horizon not in protocol.evaluation_horizons:
        raise ValueError("horizon is not registered by the ML protocol")
    if manifest.get("kind") != "causal_sequence_dataset":
        raise ValueError("unsupported sequence manifest kind")
    if manifest.get("protocol_version") != protocol.protocol_version:
        raise ValueError("sequence manifest protocol does not match config")
    spec = SequenceFeatureSpec.from_protocol(protocol.raw)
    if (
        tuple(manifest.get("channels", ())) != spec.channels
        or int(manifest.get("sequence_length", 0)) != spec.sequence_length
        or manifest.get("layout") != spec.layout
        or manifest.get("dtype") != spec.dtype
    ):
        raise ValueError("sequence manifest schema does not match protocol")
    if manifest.get("test_labels_used_for_normalization") is not False:
        raise ValueError("sequence manifest does not prove train-only statistics")

    root = Path(repo_root).resolve()
    selected_device = _resolve_device(device)
    selected_seed = (
        protocol.random_seed if random_seed is None else int(random_seed)
    )
    allowed_seeds = tuple(
        int(value) for value in protocol.raw["robustness"]["seeds"]
    )
    if selected_seed not in allowed_seeds:
        raise ValueError(
            "random_seed is not registered for Phase 6 robustness"
        )
    settings = protocol.raw["models"]["phase_5_temporal_convolutional"]
    architecture = _architecture(settings)
    mean, scale = _normalization(manifest, horizon, len(spec.channels))
    train_entries = _entries_for_fold(manifest, horizon, "train")
    selection_entries = _entries_for_fold(manifest, horizon, "selection")
    calibration_entries = _entries_for_fold(manifest, horizon, "calibration")
    _validate_shard_hashes(
        [*train_entries, *selection_entries, *calibration_entries], root
    )
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

    torch.manual_seed(selected_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(selected_seed)
    torch.use_deterministic_algorithms(True)
    network = _build_network(len(spec.channels), architecture).to(
        selected_device
    )
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    batch_size = int(settings["batch_size"])
    epoch_rows: list[dict[str, float | int]] = []
    best_key = (float("inf"), float("inf"))
    best_epoch = 0
    best_state: OrderedDict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        network.train()
        loss_sum = 0.0
        row_count = 0
        positive_count = 0.0
        for batch_x, batch_y in _iter_training_batches(
            train_entries,
            root=root,
            channels=len(spec.channels),
            sequence_length=spec.sequence_length,
            mean=mean,
            scale=scale,
            batch_size=batch_size,
            random_seed=selected_seed + epoch,
        ):
            batch_x = batch_x.to(selected_device)
            batch_y = batch_y.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(batch_x)
            loss = functional.binary_cross_entropy_with_logits(
                logits, batch_y
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                network.parameters(),
                float(settings["gradient_clip_norm"]),
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch_y)
            row_count += len(batch_y)
            positive_count += float(batch_y.detach().sum().cpu())
        if row_count == 0:
            raise ValueError("sequence manifest has no train rows")
        if positive_count <= 0.0 or positive_count >= row_count:
            raise ValueError("train target must contain both classes")
        probability = _predict_network(
            network,
            selection_x,
            batch_size=batch_size,
            device=selected_device,
        )
        selection_brier = _brier(probability, selection_y)
        selection_log_loss = _log_loss(probability, selection_y)
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_log_loss": loss_sum / row_count,
                "selection_brier": selection_brier,
                "selection_log_loss": selection_log_loss,
            }
        )
        key = (selection_brier, selection_log_loss)
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
            if stale_epochs >= int(settings["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("TCN training did not produce a checkpoint")
    network.load_state_dict(best_state)
    network.to(selected_device)
    raw_calibration = _predict_network(
        network,
        calibration_x,
        batch_size=batch_size,
        device=selected_device,
    )
    calibration_rows: list[dict[str, float | str]] = []
    calibrators: list[ProbabilityCalibrator] = []
    for kind in settings["probability_calibrators"]:
        calibrator = fit_probability_calibrator(
            str(kind), raw_calibration, calibration_y
        )
        probability = calibrator.predict(raw_calibration)
        calibration_rows.append(
            {
                "kind": str(kind),
                "calibration_brier": _brier(probability, calibration_y),
                "calibration_log_loss": _log_loss(
                    probability, calibration_y
                ),
            }
        )
        calibrators.append(calibrator)
    calibration_index = min(
        range(len(calibration_rows)),
        key=lambda index: (
            calibration_rows[index]["calibration_brier"],
            calibration_rows[index]["calibration_log_loss"],
            index,
        ),
    )
    selected_calibrator = calibrators[calibration_index]
    network.to("cpu")
    model = TemporalConvolutionalDirectionalModel(
        network=network,
        calibrator=selected_calibrator,
        normalization_mean=mean,
        normalization_scale=scale,
        channel_names=spec.channels,
        sequence_length=spec.sequence_length,
        architecture=architecture,
        protocol_version=protocol.protocol_version,
        random_seed=selected_seed,
        horizon_ticks=horizon,
    )
    diagnostics = {
        "kind": "temporal_convolutional_network_training",
        "protocol_version": protocol.protocol_version,
        "random_seed": selected_seed,
        "horizon_ticks": horizon,
        "device": str(selected_device),
        "architecture": architecture,
        "training_parameters": {
            key: settings[key]
            for key in (
                "optimizer",
                "learning_rate",
                "weight_decay",
                "batch_size",
                "max_epochs",
                "early_stopping_patience",
                "gradient_clip_norm",
            )
        },
        "fold_rows": {
            "train": sum(int(entry["rows"]) for entry in train_entries),
            "selection": len(selection_y),
            "calibration": len(calibration_y),
        },
        "epochs": epoch_rows,
        "selected_epoch": best_epoch,
        "calibrators": calibration_rows,
        "selected_calibrator": selected_calibrator.kind,
        "normalization_source_fold": "train",
        "test_fold_read": False,
    }
    return model, diagnostics


def _calibrator_payload(
    calibrator: ProbabilityCalibrator,
) -> dict[str, Any]:
    return {
        "kind": calibrator.kind,
        "parameters": calibrator.parameters.tolist(),
        "thresholds": calibrator.thresholds.tolist(),
        "probabilities": calibrator.probabilities.tolist(),
    }


def save_temporal_convolutional_model(
    model: TemporalConvolutionalDirectionalModel,
    path: str | Path,
) -> Path:
    """Persist a safe, versioned TCN state-dict checkpoint."""
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
            "architecture": dict(model.architecture),
            "protocol_version": model.protocol_version,
            "random_seed": model.random_seed,
            "horizon_ticks": model.horizon_ticks,
        },
        destination,
    )
    return destination


def load_temporal_convolutional_model(
    path: str | Path,
) -> TemporalConvolutionalDirectionalModel:
    """Load a model produced by :func:`save_temporal_convolutional_model`."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("format_version") != MODEL_FORMAT_VERSION
    ):
        raise ValueError("unsupported Temporal Convolutional Network artifact")
    architecture = dict(payload["architecture"])
    channel_names = tuple(str(value) for value in payload["channel_names"])
    network = _build_network(len(channel_names), architecture)
    network.load_state_dict(payload["state_dict"])
    network.eval()
    calibrator_payload = payload["calibrator"]
    calibrator = ProbabilityCalibrator(
        kind=str(calibrator_payload["kind"]),
        parameters=np.asarray(
            calibrator_payload["parameters"], dtype=np.float64
        ),
        thresholds=np.asarray(
            calibrator_payload["thresholds"], dtype=np.float64
        ),
        probabilities=np.asarray(
            calibrator_payload["probabilities"], dtype=np.float64
        ),
    )
    return TemporalConvolutionalDirectionalModel(
        network=network,
        calibrator=calibrator,
        normalization_mean=np.asarray(
            payload["normalization_mean"], dtype=np.float32
        ),
        normalization_scale=np.asarray(
            payload["normalization_scale"], dtype=np.float32
        ),
        channel_names=channel_names,
        sequence_length=int(payload["sequence_length"]),
        architecture=architecture,
        protocol_version=str(payload["protocol_version"]),
        random_seed=int(payload["random_seed"]),
        horizon_ticks=int(payload["horizon_ticks"]),
    )

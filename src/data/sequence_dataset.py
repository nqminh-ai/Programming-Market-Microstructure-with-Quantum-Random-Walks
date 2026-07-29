"""Causal UTC-day sequence shards aligned to the Phase 1 event samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.ml_protocol import (
    DEFAULT_ML_CONFIG,
    MLDirectionalProtocol,
    load_ml_protocol,
)
from src.evaluation.provenance import (
    canonical_repo_path,
    release_dirty_paths,
    sha256_file,
)

from .ml_dataset import (
    discover_parquet_utc_days,
    iter_parquet_utc_days,
    validate_utc_day_sequence,
)


@dataclass(frozen=True)
class SequenceFeatureSpec:
    """Frozen Phase 4 sequence shape and channel order."""

    sequence_length: int
    channels: tuple[str, ...]
    dtype: str
    layout: str

    @classmethod
    def from_protocol(
        cls, config: Mapping[str, Any]
    ) -> "SequenceFeatureSpec":
        settings = config["models"]["phase_4_sequence_dataset"]
        return cls(
            sequence_length=int(settings["sequence_length"]),
            channels=tuple(settings["channels"]),
            dtype=str(settings["dtype"]),
            layout=str(settings["layout"]),
        )


@dataclass(frozen=True)
class SequenceDatasetBuild:
    """Manifest and shard paths produced by a Phase 4 build."""

    manifest_path: Path
    shard_paths: tuple[Path, ...]
    manifest: Mapping[str, Any]


def _trade_sign(frame: pd.DataFrame) -> np.ndarray:
    if "trade_sign" in frame:
        result = frame["trade_sign"].to_numpy(dtype=np.float64)
    elif "side" in frame:
        side = frame["side"].astype("string").str.lower()
        if not side.isin(("buy", "sell")).all():
            raise ValueError("side must contain only buy or sell")
        result = np.where(side.eq("buy"), 1.0, -1.0)
    else:
        raise ValueError("sequence features require trade_sign or side")
    if not np.isin(result, (-1.0, 1.0)).all():
        raise ValueError("trade_sign must contain only -1 or +1")
    return result


def _row_channel_matrix(
    frame: pd.DataFrame, spec: SequenceFeatureSpec
) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "timestamp",
        "price",
        "quantity",
        "tick_direction",
        "trade_intensity",
        "obi",
        "obi_valid",
        "segment_id",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"sequence frame is missing columns: {missing}")
    timestamp = frame["timestamp"].to_numpy(dtype=np.int64)
    if np.any(np.diff(timestamp) < 0):
        raise ValueError("sequence timestamps must be chronological")
    price = frame["price"].to_numpy(dtype=np.float64)
    quantity = frame["quantity"].to_numpy(dtype=np.float64)
    direction = frame["tick_direction"].to_numpy(dtype=np.float64)
    intensity = frame["trade_intensity"].to_numpy(dtype=np.float64)
    obi = frame["obi"].to_numpy(dtype=np.float64)
    obi_valid = frame["obi_valid"].to_numpy(dtype=bool)
    segment = frame["segment_id"].to_numpy(copy=False)
    sign = _trade_sign(frame)
    if "price_mid_deviation" in frame:
        price_mid_deviation = frame["price_mid_deviation"].to_numpy(
            dtype=np.float64
        )
    elif "mid_price" in frame:
        price_mid_deviation = (
            price - frame["mid_price"].to_numpy(dtype=np.float64)
        )
    else:
        raise ValueError(
            "sequence frame requires price_mid_deviation or mid_price"
        )

    one_tick_return = np.zeros(len(frame), dtype=np.float64)
    inter_event_seconds = np.zeros(len(frame), dtype=np.float64)
    if len(frame) > 1:
        adjacent = (
            (segment[1:] == segment[:-1])
            & np.isfinite(price[1:])
            & np.isfinite(price[:-1])
            & (price[1:] > 0.0)
            & (price[:-1] > 0.0)
            & (timestamp[1:] >= timestamp[:-1])
        )
        one_tick_return[1:][adjacent] = np.log(
            price[1:][adjacent] / price[:-1][adjacent]
        )
        inter_event_seconds[1:][adjacent] = (
            timestamp[1:][adjacent] - timestamp[:-1][adjacent]
        ) / 1_000_000_000.0

    channel_values: dict[str, np.ndarray] = {
        "obi": obi,
        "obi_valid": obi_valid.astype(np.float64),
        "tick_direction": direction,
        "log_trade_intensity": np.log1p(np.maximum(intensity, 0.0)),
        "signed_log_quantity": sign * np.log1p(np.maximum(quantity, 0.0)),
        "one_tick_log_return": one_tick_return,
        "price_mid_deviation": price_mid_deviation,
        "inter_event_seconds": inter_event_seconds,
    }
    missing_channels = sorted(set(spec.channels).difference(channel_values))
    if missing_channels:
        raise ValueError(f"unsupported sequence channels: {missing_channels}")
    matrix = np.column_stack(
        [channel_values[name] for name in spec.channels]
    )
    row_valid = (
        np.isfinite(matrix).all(axis=1)
        & np.isfinite(quantity)
        & (quantity > 0.0)
        & np.isin(direction, (-1.0, 1.0))
    )
    return matrix, row_valid


def build_sequence_tensor(
    day_frame: pd.DataFrame,
    anchors: Sequence[int] | np.ndarray,
    spec: SequenceFeatureSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather oldest-to-newest causal sequences ending at each anchor."""
    if spec.sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    if spec.layout != "channels_first" or spec.dtype != "float32":
        raise ValueError("unsupported frozen sequence representation")
    anchor = np.asarray(anchors, dtype=np.int64)
    if anchor.ndim != 1:
        raise ValueError("anchors must be one-dimensional")
    if len(anchor) and (
        anchor.min() < 0
        or anchor.max() >= len(day_frame)
        or np.any(np.diff(anchor) <= 0)
    ):
        raise ValueError("anchors must be unique, increasing and in bounds")
    if not len(anchor):
        return (
            np.empty(
                (0, len(spec.channels), spec.sequence_length),
                dtype=np.float32,
            ),
            np.empty(0, dtype=bool),
        )

    row_matrix, row_valid = _row_channel_matrix(day_frame, spec)
    offsets = np.arange(
        spec.sequence_length - 1, -1, -1, dtype=np.int64
    )
    index = anchor[:, None] - offsets[None, :]
    in_bounds = index[:, 0] >= 0
    safe_index = np.maximum(index, 0)
    segment = day_frame["segment_id"].to_numpy(copy=False)
    same_segment = (
        segment[safe_index] == segment[anchor, None]
    ).all(axis=1)
    valid = (
        in_bounds
        & same_segment
        & row_valid[safe_index].all(axis=1)
    )
    tensor = row_matrix[safe_index].transpose(0, 2, 1).astype(
        np.float32, copy=False
    )
    return tensor, valid


def _resolve_artifact_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_phase1_events(
    metadata: Mapping[str, Any],
    root: Path,
) -> tuple[dict[int, pd.DataFrame], dict[int, Path]]:
    events: dict[int, pd.DataFrame] = {}
    paths: dict[int, Path] = {}
    columns = [
        "timestamp",
        "target_timestamp",
        "utc_day",
        "anchor_row",
        "horizon_ticks",
        "fold",
        "target_up",
        "forward_log_return",
    ]
    for value in metadata["horizons_ticks"]:
        horizon = int(value)
        entry = metadata["datasets"][str(horizon)]
        path = _resolve_artifact_path(entry["path"], root)
        registered_hash = entry.get("sha256")
        if registered_hash and sha256_file(path) != registered_hash:
            raise ValueError(
                f"Phase 1 dataset SHA-256 mismatch at h={horizon}"
            )
        frame = pd.read_parquet(path, columns=columns)
        if not frame["horizon_ticks"].eq(horizon).all():
            raise ValueError(f"Phase 1 horizon column mismatch at h={horizon}")
        if frame["anchor_row"].duplicated().any():
            raise ValueError(f"Phase 1 anchors are duplicated at h={horizon}")
        events[horizon] = frame
        paths[horizon] = path
    return events, paths


def build_sequence_dataset_shards(
    feature_path: str | Path,
    phase1_metadata_path: str | Path,
    output_directory: str | Path,
    *,
    asset: str,
    protocol: MLDirectionalProtocol | None = None,
    config_path: str | Path = DEFAULT_ML_CONFIG,
    batch_size: int = 1_000_000,
    max_days: int = 0,
    official: bool = False,
    repo_root: str | Path | None = None,
) -> SequenceDatasetBuild:
    """Stream raw days and write sequence shards for exact Phase 1 anchors."""
    selected_protocol = protocol or load_ml_protocol(config_path)
    normalized_asset = asset.upper()
    if normalized_asset not in selected_protocol.assets:
        raise ValueError(f"asset is not registered: {normalized_asset}")
    if max_days < 0 or (official and max_days):
        raise ValueError("official builds cannot be capped")
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source = Path(feature_path).resolve()
    metadata_path = Path(phase1_metadata_path).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    if metadata.get("protocol_version") != selected_protocol.protocol_version:
        raise ValueError("Phase 1 metadata protocol does not match config")
    if metadata.get("config_sha256") != config_hash:
        raise ValueError("Phase 1 metadata config SHA-256 does not match")
    if metadata.get("asset") != normalized_asset:
        raise ValueError("Phase 1 metadata asset does not match")
    if official:
        registered_source = (
            root
            / selected_protocol.raw["assets"][normalized_asset][
                "feature_path"
            ]
        ).resolve()
        if source != registered_source:
            raise ValueError(
                "official sequence source does not match registered feature path"
            )
        registered_feature_hash = metadata.get("feature_sha256")
        if (
            not registered_feature_hash
            or sha256_file(source) != registered_feature_hash
        ):
            raise ValueError(
                "official sequence source SHA-256 does not match Phase 1"
            )
        dirty = release_dirty_paths(root)
        if dirty:
            raise RuntimeError(
                "official sequence build requires a clean source tree; "
                f"dirty paths: {', '.join(dirty[:8])}"
            )

    days = discover_parquet_utc_days(source)
    validate_utc_day_sequence(
        days, expected_days=selected_protocol.total_days
    )
    selected_days = days[:max_days] if max_days else days
    phase1_events, phase1_paths = _load_phase1_events(metadata, root)
    expected_horizons = set(selected_protocol.evaluation_horizons)
    if set(phase1_events) != expected_horizons:
        raise ValueError("Phase 1 metadata horizon set does not match protocol")

    spec = SequenceFeatureSpec.from_protocol(selected_protocol.raw)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stats = {
        horizon: {
            "sum": np.zeros(len(spec.channels), dtype=np.float64),
            "sum_sq": np.zeros(len(spec.channels), dtype=np.float64),
            "count": 0,
        }
        for horizon in phase1_events
    }
    shard_rows: list[dict[str, Any]] = []
    shard_paths: list[Path] = []
    global_offset = 0
    processed_days = 0
    for day_index, (day, day_frame) in enumerate(
        iter_parquet_utc_days(source, batch_size=batch_size)
    ):
        if max_days and day_index >= max_days:
            break
        for horizon, event_frame in phase1_events.items():
            selected = event_frame.loc[
                event_frame["utc_day"].astype(str).eq(day)
            ].reset_index(drop=True)
            if selected.empty:
                continue
            local_anchor = (
                selected["anchor_row"].to_numpy(dtype=np.int64)
                - global_offset
            )
            if (
                local_anchor.min() < 0
                or local_anchor.max() >= len(day_frame)
            ):
                raise ValueError(
                    f"Phase 1 anchor offset mismatch on {day}, h={horizon}"
                )
            raw_timestamp = day_frame["timestamp"].to_numpy(dtype=np.int64)
            if not np.array_equal(
                raw_timestamp[local_anchor],
                selected["timestamp"].to_numpy(dtype=np.int64),
            ):
                raise ValueError(
                    f"Phase 1 timestamp alignment mismatch on {day}, h={horizon}"
                )
            tensor, valid = build_sequence_tensor(
                day_frame, local_anchor, spec
            )
            if not valid.all():
                raise ValueError(
                    f"invalid causal sequence on {day}, h={horizon}"
                )
            fold_values = selected["fold"].astype(str).unique()
            if len(fold_values) != 1:
                raise ValueError("one UTC-day shard must belong to one fold")
            fold = str(fold_values[0])
            shard_directory = output / f"h{horizon}" / fold
            shard_directory.mkdir(parents=True, exist_ok=True)
            shard_path = shard_directory / f"{day}.npz"
            np.savez_compressed(
                shard_path,
                features=tensor,
                target=selected["target_up"].to_numpy(dtype=np.int8),
                timestamp=selected["timestamp"].to_numpy(dtype=np.int64),
                target_timestamp=selected["target_timestamp"].to_numpy(
                    dtype=np.int64
                ),
                anchor_row=selected["anchor_row"].to_numpy(dtype=np.int64),
                forward_log_return=selected[
                    "forward_log_return"
                ].to_numpy(dtype=np.float64),
            )
            if fold == "train":
                values = tensor.astype(np.float64, copy=False)
                stats[horizon]["sum"] += values.sum(axis=(0, 2))
                stats[horizon]["sum_sq"] += (values**2).sum(axis=(0, 2))
                stats[horizon]["count"] += (
                    values.shape[0] * values.shape[2]
                )
            shard_paths.append(shard_path)
            shard_rows.append(
                {
                    "horizon_ticks": horizon,
                    "utc_day": day,
                    "fold": fold,
                    "rows": len(selected),
                    "path": canonical_repo_path(shard_path, root),
                    "sha256": sha256_file(shard_path) if official else None,
                }
            )
        global_offset += len(day_frame)
        processed_days += 1
    if processed_days != len(selected_days):
        raise RuntimeError(
            f"streamed {processed_days} days but expected {len(selected_days)}"
        )
    normalization: dict[str, Any] = {}
    for horizon, values in stats.items():
        count = int(values["count"])
        if count <= 0:
            raise RuntimeError(f"no train sequences at h={horizon}")
        mean = values["sum"] / count
        variance = np.maximum(values["sum_sq"] / count - mean**2, 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-8] = 1.0
        normalization[str(horizon)] = {
            "count": count,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "source_fold": "train",
        }

    manifest = {
        "kind": "causal_sequence_dataset",
        "status": (
            selected_protocol.status
            if not max_days
            else "development_smoke_not_for_claims"
        ),
        "protocol_version": selected_protocol.protocol_version,
        "asset": normalized_asset,
        "feature_path": canonical_repo_path(source, root),
        "feature_sha256": sha256_file(source) if official else None,
        "phase1_metadata_path": canonical_repo_path(metadata_path, root),
        "phase1_metadata_sha256": sha256_file(metadata_path),
        "phase1_datasets": {
            str(horizon): canonical_repo_path(path, root)
            for horizon, path in phase1_paths.items()
        },
        "config_path": canonical_repo_path(config_path, root),
        "config_sha256": config_hash,
        "sequence_length": spec.sequence_length,
        "layout": spec.layout,
        "dtype": spec.dtype,
        "channels": list(spec.channels),
        "days": list(selected_days),
        "normalization": normalization,
        "shards": shard_rows,
        "official": bool(official),
        "test_labels_used_for_normalization": False,
    }
    manifest_path = output / f"sequence_{normalized_asset}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return SequenceDatasetBuild(
        manifest_path=manifest_path,
        shard_paths=tuple(shard_paths),
        manifest=manifest,
    )


def iter_sequence_shards(
    manifest: Mapping[str, Any],
    *,
    folds: Sequence[str],
    repo_root: str | Path,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield validated numpy shards for the requested folds."""
    allowed = set(folds)
    root = Path(repo_root).resolve()
    sequence_length = int(manifest["sequence_length"])
    channels = len(manifest["channels"])
    for entry in manifest["shards"]:
        if entry["fold"] not in allowed:
            continue
        path = _resolve_artifact_path(entry["path"], root)
        with np.load(path, allow_pickle=False) as payload:
            shard = {name: payload[name] for name in payload.files}
        if shard["features"].shape != (
            int(entry["rows"]),
            channels,
            sequence_length,
        ):
            raise ValueError(f"sequence shard shape mismatch: {path}")
        yield shard

"""Causal, day-streamed datasets for the frozen ML directional protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

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

from .temporal_features import (
    TemporalFeatureSpec,
    build_temporal_feature_matrix,
)


NANOSECONDS_PER_DAY = 86_400_000_000_000
ML_SOURCE_COLUMNS = (
    "timestamp",
    "price",
    "quantity",
    "side",
    "tick_direction",
    "trade_intensity",
    "obi",
    "obi_valid",
    "segment_id",
    "mid_price",
    "price_mid_deviation",
)
FOLD_ORDER = ("train", "selection", "calibration", "test")


@dataclass(frozen=True)
class MLDatasetBuild:
    """Paths and metadata produced by one streamed dataset build."""

    datasets: Mapping[int, Path]
    metadata_path: Path
    metadata: Mapping[str, Any]


def _utc_day_code(timestamp_ns: np.ndarray) -> np.ndarray:
    return np.floor_divide(timestamp_ns, NANOSECONDS_PER_DAY)


def _day_string(day_code: int) -> str:
    return str(np.datetime64(int(day_code), "D"))


def _batch_frame(batch: pa.RecordBatch) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for name in ML_SOURCE_COLUMNS:
        index = batch.schema.get_field_index(name)
        if index < 0:
            continue
        column = batch.column(index)
        if name == "side":
            is_buy = pc.equal(column, pa.scalar("buy", type=column.type))
            is_sell = pc.equal(column, pa.scalar("sell", type=column.type))
            valid = pc.or_(is_buy, is_sell).to_numpy(zero_copy_only=False)
            if not valid.all():
                raise ValueError("side must contain only buy or sell")
            data["trade_sign"] = np.where(
                is_buy.to_numpy(zero_copy_only=False),
                np.int8(1),
                np.int8(-1),
            )
            continue
        values = column.to_numpy(zero_copy_only=False)
        if name in {"obi", "trade_intensity", "price_mid_deviation"}:
            values = values.astype(np.float32, copy=False)
        data[name] = values
    return pd.DataFrame(data, copy=False)


def iter_parquet_utc_days(
    path: str | Path,
    *,
    batch_size: int = 1_000_000,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield one complete UTC day at a time from an ordered parquet store."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = Path(path)
    handle = pq.ParquetFile(source)
    present = set(handle.schema.names)
    missing = sorted(set(ML_SOURCE_COLUMNS).difference(present))
    if missing:
        raise ValueError(f"ML feature store is missing columns: {missing}")

    current_day: int | None = None
    fragments: list[pd.DataFrame] = []
    last_timestamp: int | None = None
    for batch in handle.iter_batches(
        batch_size=batch_size,
        columns=list(ML_SOURCE_COLUMNS),
        use_threads=True,
    ):
        frame = _batch_frame(batch)
        timestamp = frame["timestamp"].to_numpy(dtype=np.int64)
        if (
            np.any(np.diff(timestamp) < 0)
            or (
                last_timestamp is not None
                and len(timestamp)
                and timestamp[0] < last_timestamp
            )
        ):
            raise ValueError("ML feature store must be chronologically ordered")
        if len(timestamp):
            last_timestamp = int(timestamp[-1])
        day = _utc_day_code(timestamp)
        boundaries = np.r_[0, np.flatnonzero(np.diff(day) != 0) + 1, len(frame)]
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            if stop <= start:
                continue
            piece_day = int(day[start])
            piece = frame.iloc[start:stop].reset_index(drop=True)
            if current_day is None:
                current_day = piece_day
            if piece_day < current_day:
                raise ValueError("UTC days cannot move backwards")
            if piece_day != current_day:
                yield _day_string(current_day), pd.concat(
                    fragments, ignore_index=True
                )
                fragments = []
                current_day = piece_day
            fragments.append(piece)
    if current_day is not None:
        yield _day_string(current_day), pd.concat(
            fragments, ignore_index=True
        )


def discover_parquet_utc_days(
    path: str | Path, *, batch_size: int = 4_000_000
) -> tuple[str, ...]:
    """Read only timestamps and return ordered distinct UTC dates."""
    source = Path(path)
    handle = pq.ParquetFile(source)
    if "timestamp" not in handle.schema.names:
        raise ValueError("ML feature store is missing timestamp")
    result: list[str] = []
    last_code: int | None = None
    last_timestamp: int | None = None
    for batch in handle.iter_batches(
        batch_size=batch_size, columns=["timestamp"], use_threads=True
    ):
        values = batch.column(0).to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        if (
            np.any(np.diff(values) < 0)
            or (
                last_timestamp is not None
                and len(values)
                and values[0] < last_timestamp
            )
        ):
            raise ValueError("ML feature store must be chronologically ordered")
        if not len(values):
            continue
        last_timestamp = int(values[-1])
        codes = _utc_day_code(values)
        starts = np.r_[0, np.flatnonzero(np.diff(codes) != 0) + 1]
        for code in codes[starts]:
            numeric = int(code)
            if numeric != last_code:
                result.append(_day_string(numeric))
                last_code = numeric
    return tuple(result)


def validate_utc_day_sequence(
    days: Sequence[str], *, expected_days: int
) -> None:
    """Require the registered number of consecutive, unique UTC dates."""
    if len(days) != expected_days:
        raise ValueError(
            f"feature store requires {expected_days} UTC days, found {len(days)}"
        )
    values = np.asarray(days, dtype="datetime64[D]")
    if len(np.unique(values)) != len(values):
        raise ValueError("feature store UTC days must be unique")
    if len(values) > 1 and not np.all(np.diff(values).astype(int) == 1):
        raise ValueError("feature store UTC days must be consecutive")


def build_horizon_dataset(
    day_frame: pd.DataFrame,
    horizon: int,
    spec: TemporalFeatureSpec,
    *,
    global_row_offset: int = 0,
) -> pd.DataFrame:
    """Build non-overlapping causal events within one UTC day."""
    return build_horizon_datasets(
        day_frame,
        (horizon,),
        spec,
        global_row_offset=global_row_offset,
    )[horizon]


def build_horizon_datasets(
    day_frame: pd.DataFrame,
    horizons: Sequence[int],
    spec: TemporalFeatureSpec,
    *,
    global_row_offset: int = 0,
) -> dict[int, pd.DataFrame]:
    """Build many horizons while calculating trailing features only once."""
    horizon_values = tuple(int(value) for value in horizons)
    if (
        not horizon_values
        or any(value <= 0 for value in horizon_values)
        or len(set(horizon_values)) != len(horizon_values)
    ):
        raise ValueError("horizons must be unique positive integers")
    if day_frame.empty:
        raise ValueError("day_frame cannot be empty")
    timestamp = day_frame["timestamp"].to_numpy(dtype=np.int64)
    day_codes = np.unique(_utc_day_code(timestamp))
    if len(day_codes) != 1:
        raise ValueError("day_frame must contain exactly one UTC day")
    start = spec.maximum_lookback
    columns = [
        "timestamp",
        "target_timestamp",
        "utc_day",
        "anchor_row",
        "horizon_ticks",
        "target_up",
        "forward_log_return",
        *spec.feature_names,
    ]
    anchors_by_horizon = {
        horizon: np.arange(
            start,
            len(day_frame) - horizon,
            horizon,
            dtype=np.int64,
        )
        for horizon in horizon_values
    }
    nonempty = [value for value in anchors_by_horizon.values() if len(value)]
    if not nonempty:
        return {
            horizon: pd.DataFrame(columns=columns)
            for horizon in horizon_values
        }
    common_anchors = np.unique(np.concatenate(nonempty))
    common_matrix, common_valid = build_temporal_feature_matrix(
        day_frame, common_anchors, spec
    )
    segment = day_frame["segment_id"].to_numpy(copy=False)
    price = day_frame["price"].to_numpy(dtype=np.float64)
    result: dict[int, pd.DataFrame] = {}
    for horizon, anchors in anchors_by_horizon.items():
        if not len(anchors):
            result[horizon] = pd.DataFrame(columns=columns)
            continue
        common_index = np.searchsorted(common_anchors, anchors)
        matrix = common_matrix[common_index]
        feature_valid = common_valid[common_index]
        future = anchors + horizon
        endpoint_valid = (
            (segment[anchors] == segment[future])
            & np.isfinite(price[anchors])
            & np.isfinite(price[future])
            & (price[anchors] > 0.0)
            & (price[future] > 0.0)
        )
        forward_return = np.full(len(anchors), np.nan, dtype=np.float64)
        usable = feature_valid & endpoint_valid
        forward_return[usable] = np.log(
            price[future[usable]] / price[anchors[usable]]
        )
        keep = usable & np.isfinite(forward_return) & (
            np.abs(forward_return) > 1e-12
        )
        kept_anchor = anchors[keep]
        kept_future = future[keep]
        data: dict[str, Any] = {
            "timestamp": timestamp[kept_anchor],
            "target_timestamp": timestamp[kept_future],
            "utc_day": np.repeat(
                _day_string(int(day_codes[0])), keep.sum()
            ),
            "anchor_row": kept_anchor + int(global_row_offset),
            "horizon_ticks": np.repeat(int(horizon), keep.sum()),
            "target_up": (forward_return[keep] > 0.0).astype(np.int8),
            "forward_log_return": forward_return[keep],
        }
        for index, name in enumerate(spec.feature_names):
            data[name] = matrix[keep, index]
        result[horizon] = pd.DataFrame(
            data, columns=columns, copy=False
        )
    return result


def fold_for_day_index(
    day_index: int, protocol: MLDirectionalProtocol
) -> str:
    """Return the frozen chronological fold for a zero-based day index."""
    if day_index < 0 or day_index >= protocol.total_days:
        raise IndexError("day index is outside the frozen chronological split")
    limits = np.cumsum(
        (
            protocol.train_days,
            protocol.selection_days,
            protocol.calibration_days,
            protocol.test_days,
        )
    )
    return FOLD_ORDER[int(np.searchsorted(limits, day_index, side="right"))]


def boundary_purge_mask(
    anchors: np.ndarray,
    *,
    future: np.ndarray,
    day_rows: int,
    day_index: int,
    protocol: MLDirectionalProtocol,
    maximum_lookback: int,
) -> np.ndarray:
    """Embargo both sides of every fold boundary by the frozen tick count."""
    anchor = np.asarray(anchors, dtype=np.int64)
    endpoint = np.asarray(future, dtype=np.int64)
    if anchor.shape != endpoint.shape:
        raise ValueError("anchors and future must share one shape")
    keep = np.ones(len(anchor), dtype=bool)
    boundaries = np.cumsum(
        (
            protocol.train_days,
            protocol.selection_days,
            protocol.calibration_days,
        )
    )
    if day_index in boundaries:
        keep &= anchor - maximum_lookback >= protocol.purge_ticks
    if day_index + 1 in boundaries:
        keep &= day_rows - endpoint >= protocol.purge_ticks
    return keep


def _apply_fold_and_purge(
    events: pd.DataFrame,
    *,
    day_rows: int,
    day_index: int,
    global_row_offset: int,
    protocol: MLDirectionalProtocol,
    spec: TemporalFeatureSpec,
) -> pd.DataFrame:
    if events.empty:
        result = events.copy()
        result.insert(5, "fold", pd.Series(dtype="string"))
        return result
    local_anchor = (
        events["anchor_row"].to_numpy(dtype=np.int64)
        - int(global_row_offset)
    )
    horizon = events["horizon_ticks"].to_numpy(dtype=np.int64)
    mask = boundary_purge_mask(
        local_anchor,
        future=local_anchor + horizon,
        day_rows=day_rows,
        day_index=day_index,
        protocol=protocol,
        maximum_lookback=spec.maximum_lookback,
    )
    result = events.loc[mask].copy()
    result.insert(5, "fold", fold_for_day_index(day_index, protocol))
    return result


def build_ml_dataset_files(
    feature_path: str | Path,
    output_directory: str | Path,
    *,
    asset: str,
    protocol: MLDirectionalProtocol | None = None,
    config_path: str | Path = DEFAULT_ML_CONFIG,
    batch_size: int = 1_000_000,
    max_days: int = 0,
    official: bool = False,
    repo_root: str | Path | None = None,
) -> MLDatasetBuild:
    """Stream one feature store into one parquet dataset per horizon."""
    selected_protocol = protocol or load_ml_protocol(config_path)
    normalized_asset = asset.upper()
    if normalized_asset not in selected_protocol.assets:
        raise ValueError(f"asset is not registered: {normalized_asset}")
    if max_days < 0:
        raise ValueError("max_days cannot be negative")
    if official and max_days:
        raise ValueError("official builds cannot use max_days")
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source = Path(feature_path).resolve()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    days = discover_parquet_utc_days(source)
    validate_utc_day_sequence(
        days, expected_days=selected_protocol.total_days
    )
    if official:
        registered = (
            root
            / selected_protocol.raw["assets"][normalized_asset]["feature_path"]
        ).resolve()
        if source != registered:
            raise ValueError(
                "official feature path does not match the frozen asset path"
            )
        dirty = release_dirty_paths(root)
        if dirty:
            raise RuntimeError(
                "official ML dataset build requires a clean source tree; "
                f"dirty paths: {', '.join(dirty[:8])}"
            )
    selected_days = days[:max_days] if max_days else days
    if len(selected_days) > selected_protocol.total_days:
        raise ValueError("feature store has more days than the frozen protocol")

    spec = TemporalFeatureSpec.from_protocol(selected_protocol.raw)
    destinations = {
        horizon: output
        / f"ml_directional_{normalized_asset}_h{horizon}.parquet"
        for horizon in selected_protocol.evaluation_horizons
    }
    writers: dict[int, pq.ParquetWriter] = {}
    counts = {horizon: {fold: 0 for fold in FOLD_ORDER} for horizon in destinations}
    split_dates = {fold: [] for fold in FOLD_ORDER}
    global_offset = 0
    processed_days = 0
    try:
        for day_index, (day, frame) in enumerate(
            iter_parquet_utc_days(source, batch_size=batch_size)
        ):
            if max_days and day_index >= max_days:
                break
            if day_index >= selected_protocol.total_days:
                raise ValueError("feature store exceeds the frozen day split")
            fold = fold_for_day_index(day_index, selected_protocol)
            split_dates[fold].append(day)
            day_events = build_horizon_datasets(
                frame,
                tuple(destinations),
                spec,
                global_row_offset=global_offset,
            )
            for horizon, destination in destinations.items():
                events = day_events[horizon]
                events = _apply_fold_and_purge(
                    events,
                    day_rows=len(frame),
                    day_index=day_index,
                    global_row_offset=global_offset,
                    protocol=selected_protocol,
                    spec=spec,
                )
                if events.empty:
                    continue
                for name in spec.feature_names:
                    events[name] = events[name].astype(np.float32)
                table = pa.Table.from_pandas(events, preserve_index=False)
                if horizon not in writers:
                    writers[horizon] = pq.ParquetWriter(
                        destination, table.schema, compression="zstd"
                    )
                writers[horizon].write_table(table)
                counts[horizon][fold] += len(events)
            global_offset += len(frame)
            processed_days += 1
    finally:
        for writer in writers.values():
            writer.close()

    if processed_days != len(selected_days):
        raise RuntimeError(
            f"streamed {processed_days} days but discovered {len(selected_days)}"
        )
    missing_outputs = [
        horizon for horizon in destinations if horizon not in writers
    ]
    if missing_outputs:
        raise RuntimeError(f"no usable events for horizons: {missing_outputs}")

    config_source = Path(config_path).resolve()
    metadata: dict[str, Any] = {
        "kind": "ml_directional_dataset",
        "status": (
            selected_protocol.status
            if not max_days
            else "development_smoke_not_for_claims"
        ),
        "protocol_version": selected_protocol.protocol_version,
        "asset": normalized_asset,
        "random_seed": selected_protocol.random_seed,
        "feature_path": canonical_repo_path(source, root),
        "feature_sha256": sha256_file(source) if official else None,
        "config_path": canonical_repo_path(config_source, root),
        "config_sha256": sha256_file(config_source),
        "days": list(selected_days),
        "split_dates": split_dates,
        "feature_names": list(spec.feature_names),
        "horizons_ticks": list(selected_protocol.evaluation_horizons),
        "purge_ticks": selected_protocol.purge_ticks,
        "counts": {str(key): value for key, value in counts.items()},
        "datasets": {
            str(horizon): {
                "path": canonical_repo_path(path, root),
                "sha256": sha256_file(path) if official else None,
            }
            for horizon, path in destinations.items()
        },
        "official": bool(official),
    }
    metadata_path = (
        output / f"ml_directional_{normalized_asset}_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MLDatasetBuild(destinations, metadata_path, metadata)

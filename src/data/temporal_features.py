"""Causal multi-scale features evaluated only at requested event anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFeatureSpec:
    """Frozen trailing windows used by the ML directional protocol."""

    tick_direction_lags: tuple[int, ...]
    signed_volume_windows: tuple[int, ...]
    return_windows: tuple[int, ...]
    realised_volatility_windows: tuple[int, ...]

    @classmethod
    def from_protocol(
        cls, config: Mapping[str, Any]
    ) -> "TemporalFeatureSpec":
        features = config["features"]
        return cls(
            tick_direction_lags=_positive_windows(
                features["tick_direction_lags"], "tick_direction_lags"
            ),
            signed_volume_windows=_positive_windows(
                features["signed_volume_windows_ticks"],
                "signed_volume_windows_ticks",
            ),
            return_windows=_positive_windows(
                features["return_windows_ticks"], "return_windows_ticks"
            ),
            realised_volatility_windows=_positive_windows(
                features["realised_volatility_windows_ticks"],
                "realised_volatility_windows_ticks",
            ),
        )

    @property
    def maximum_lookback(self) -> int:
        return max(
            *self.tick_direction_lags,
            *self.signed_volume_windows,
            *self.return_windows,
            *self.realised_volatility_windows,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "obi",
            "tick_direction",
            "obi_change",
            "abs_obi",
            "log_trade_intensity",
            "price_mid_deviation",
            *(
                f"tick_direction_lag_{lag}"
                for lag in self.tick_direction_lags
            ),
            *(
                f"signed_volume_sum_{window}"
                for window in self.signed_volume_windows
            ),
            *(f"log_return_{window}" for window in self.return_windows),
            *(
                f"realised_volatility_{window}"
                for window in self.realised_volatility_windows
            ),
            "time_since_previous_event_seconds",
        )


def _positive_windows(values: Sequence[int], field: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if (
        not result
        or any(value <= 0 for value in result)
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError(f"{field} must be unique, positive and increasing")
    return result


def _trade_sign(frame: pd.DataFrame) -> np.ndarray:
    if "trade_sign" in frame:
        sign = frame["trade_sign"].to_numpy(dtype=np.float64)
    elif "side" in frame:
        side = frame["side"].astype("string").str.lower()
        if not side.isin(("buy", "sell")).all():
            raise ValueError("side must contain only buy or sell")
        sign = np.where(side.eq("buy"), 1.0, -1.0)
    else:
        raise ValueError("temporal features require trade_sign or side")
    if not np.isin(sign, (-1.0, 1.0)).all():
        raise ValueError("trade_sign must contain only -1 or +1")
    return sign


def _prefix(values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values) + 1, dtype=np.float64)
    result[0] = 0.0
    np.cumsum(values, dtype=np.float64, out=result[1:])
    return result


def _window_values(
    prefix: np.ndarray, anchors: np.ndarray, window: int
) -> np.ndarray:
    end = anchors + 1
    return prefix[end] - prefix[end - window]


def build_temporal_feature_matrix(
    frame: pd.DataFrame,
    anchors: Sequence[int] | np.ndarray,
    spec: TemporalFeatureSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return features and a validity mask for causal anchors.

    Rolling features are gathered with prefix sums. The function therefore
    allocates arrays proportional to one input day, not ``rows x windows``.
    """
    required = {
        "timestamp",
        "price",
        "quantity",
        "tick_direction",
        "trade_intensity",
        "obi",
        "segment_id",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"temporal feature frame is missing: {missing}")

    anchor = np.asarray(anchors, dtype=np.int64)
    if anchor.ndim != 1:
        raise ValueError("anchors must be one-dimensional")
    if len(anchor) and (
        anchor.min() < 0
        or anchor.max() >= len(frame)
        or np.any(np.diff(anchor) <= 0)
    ):
        raise ValueError("anchors must be unique, increasing and in bounds")
    if not len(anchor):
        return (
            np.empty((0, len(spec.feature_names)), dtype=np.float64),
            np.empty(0, dtype=bool),
        )

    timestamp = frame["timestamp"].to_numpy(dtype=np.int64)
    if np.any(np.diff(timestamp) < 0):
        raise ValueError("timestamps must be monotonically increasing")
    price = frame["price"].to_numpy(dtype=np.float64)
    quantity = frame["quantity"].to_numpy(dtype=np.float64)
    direction = frame["tick_direction"].to_numpy(dtype=np.float64)
    intensity = frame["trade_intensity"].to_numpy(dtype=np.float64)
    obi = frame["obi"].to_numpy(dtype=np.float64)
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
            "temporal features require price_mid_deviation or mid_price"
        )

    previous = anchor - 1
    same_previous_segment = (previous >= 0) & (
        segment[np.maximum(previous, 0)] == segment[anchor]
    )
    obi_change = np.zeros(len(anchor), dtype=np.float64)
    obi_change[same_previous_segment] = (
        obi[anchor[same_previous_segment]]
        - obi[previous[same_previous_segment]]
    )
    delta_seconds = np.zeros(len(anchor), dtype=np.float64)
    delta_seconds[same_previous_segment] = (
        timestamp[anchor[same_previous_segment]]
        - timestamp[previous[same_previous_segment]]
    ) / 1_000_000_000.0

    columns: list[np.ndarray] = [
        obi[anchor],
        direction[anchor],
        obi_change,
        np.abs(obi[anchor]),
        np.log1p(np.maximum(intensity[anchor], 0.0)),
        price_mid_deviation[anchor],
    ]
    valid = same_previous_segment.copy()
    if "obi_valid" in frame:
        valid &= frame["obi_valid"].to_numpy(dtype=bool)[anchor]

    for lag in spec.tick_direction_lags:
        source = anchor - lag
        usable = (source >= 0) & (
            segment[np.maximum(source, 0)] == segment[anchor]
        )
        values = np.zeros(len(anchor), dtype=np.float64)
        values[usable] = direction[source[usable]]
        columns.append(values)
        valid &= usable

    signed_volume = quantity * sign
    signed_finite = np.isfinite(signed_volume)
    signed_prefix = _prefix(np.where(signed_finite, signed_volume, 0.0))
    signed_bad_prefix = _prefix((~signed_finite).astype(np.float64))
    for window in spec.signed_volume_windows:
        source = anchor + 1 - window
        usable = (source >= 0) & (
            segment[np.maximum(source, 0)] == segment[anchor]
        )
        values = np.zeros(len(anchor), dtype=np.float64)
        if usable.any():
            selected = anchor[usable]
            values[usable] = _window_values(
                signed_prefix, selected, window
            )
            bad = _window_values(signed_bad_prefix, selected, window)
            usable_indices = np.flatnonzero(usable)
            usable[usable_indices[bad > 0.0]] = False
        columns.append(values)
        valid &= usable

    one_tick_return = np.zeros(len(frame), dtype=np.float64)
    return_valid = np.zeros(len(frame), dtype=bool)
    if len(frame) > 1:
        adjacent = (
            (segment[1:] == segment[:-1])
            & np.isfinite(price[1:])
            & np.isfinite(price[:-1])
            & (price[1:] > 0.0)
            & (price[:-1] > 0.0)
        )
        return_valid[1:] = adjacent
        one_tick_return[1:][adjacent] = np.log(
            price[1:][adjacent] / price[:-1][adjacent]
        )
    return_prefix = _prefix(one_tick_return)
    return_sq_prefix = _prefix(one_tick_return**2)
    return_bad_prefix = _prefix((~return_valid).astype(np.float64))

    return_windows = set(spec.return_windows)
    volatility_windows = set(spec.realised_volatility_windows)
    all_return_windows = sorted(return_windows.union(volatility_windows))
    return_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for window in all_return_windows:
        source = anchor - window
        usable = (source >= 0) & (
            segment[np.maximum(source, 0)] == segment[anchor]
        )
        sums = np.zeros(len(anchor), dtype=np.float64)
        square_sums = np.zeros(len(anchor), dtype=np.float64)
        if usable.any():
            selected = anchor[usable]
            sums[usable] = _window_values(return_prefix, selected, window)
            square_sums[usable] = _window_values(
                return_sq_prefix, selected, window
            )
            bad = _window_values(return_bad_prefix, selected, window)
            usable_indices = np.flatnonzero(usable)
            usable[usable_indices[bad > 0.0]] = False
        return_cache[window] = (sums, square_sums, usable)

    for window in spec.return_windows:
        sums, _, usable = return_cache[window]
        columns.append(sums)
        valid &= usable

    for window in spec.realised_volatility_windows:
        sums, square_sums, usable = return_cache[window]
        mean = sums / window
        variance = np.maximum(square_sums / window - mean**2, 0.0)
        columns.append(np.sqrt(variance))
        valid &= usable

    columns.append(delta_seconds)
    matrix = np.column_stack(columns)
    valid &= np.isfinite(matrix).all(axis=1)
    return matrix, valid

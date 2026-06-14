"""Shared schema and timestamp helpers for the Phase 2 data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TICK_COLUMNS = ("timestamp", "price", "quantity", "side", "trade_id")


def normalize_symbol(symbol: str) -> str:
    """Return an exchange-style symbol such as ``BTCUSDT``."""
    normalized = "".join(character for character in symbol.upper() if character.isalnum())
    if not normalized:
        raise ValueError("symbol must contain at least one alphanumeric character")
    return normalized


def timestamps_to_nanoseconds(values: pd.Series | Iterable[int]) -> pd.Series:
    """Normalize epoch timestamps in seconds, milliseconds, microseconds, or ns."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.isna().any():
        raise ValueError("timestamp contains non-numeric or missing values")

    absolute_max = float(numeric.abs().max()) if len(numeric) else 0.0
    if absolute_max < 1e11:
        multiplier = 1_000_000_000
    elif absolute_max < 1e14:
        multiplier = 1_000_000
    elif absolute_max < 1e17:
        multiplier = 1_000
    else:
        multiplier = 1

    return (numeric.astype("int64") * multiplier).astype("int64")


def normalize_side(values: pd.Series) -> pd.Series:
    """Normalize trade side values to lower-case ``buy`` or ``sell``."""
    if pd.api.types.is_bool_dtype(values):
        # Binance isBuyerMaker=True means the aggressive/taker side sold.
        return values.map({True: "sell", False: "buy"}).astype("string")

    normalized = values.astype("string").str.strip().str.lower()
    aliases = {
        "b": "buy",
        "buyer": "buy",
        "bid": "buy",
        "false": "buy",
        "0": "buy",
        "s": "sell",
        "seller": "sell",
        "ask": "sell",
        "true": "sell",
        "1": "sell",
    }
    normalized = normalized.replace(aliases)
    invalid = ~normalized.isin(["buy", "sell"])
    if invalid.any():
        bad_values = sorted(normalized[invalid].dropna().unique().tolist())
        raise ValueError(f"unsupported side values: {bad_values}")
    return normalized


def coerce_tick_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a DataFrame to the canonical tick schema and dtypes."""
    missing = [column for column in TICK_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"tick data is missing required columns: {missing}")

    result = frame.copy()
    result["timestamp"] = timestamps_to_nanoseconds(result["timestamp"]).to_numpy()
    result["price"] = pd.to_numeric(result["price"], errors="coerce").astype("float64")
    result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce").astype("float64")
    result["trade_id"] = pd.to_numeric(result["trade_id"], errors="coerce")
    if result["trade_id"].isna().any():
        raise ValueError("trade_id contains non-numeric or missing values")
    result["trade_id"] = result["trade_id"].astype("int64")
    result["side"] = normalize_side(result["side"])
    return result


def validate_tick_frame(
    frame: pd.DataFrame,
    *,
    require_unique_ids: bool = True,
    require_monotonic_time: bool = True,
) -> None:
    """Raise ``ValueError`` when a canonical tick frame violates its contract."""
    missing = [column for column in TICK_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"tick data is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("tick data is empty")
    if not np.isfinite(frame["price"]).all() or (frame["price"] <= 0).any():
        raise ValueError("price must be positive and finite")
    if not np.isfinite(frame["quantity"]).all() or (frame["quantity"] <= 0).any():
        raise ValueError("quantity must be positive and finite")
    if require_unique_ids and frame["trade_id"].duplicated().any():
        raise ValueError("trade_id contains duplicates")
    if require_monotonic_time and not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamp must be monotonically increasing")
    if not frame["side"].isin(["buy", "sell"]).all():
        raise ValueError("side must contain only 'buy' or 'sell'")


def ensure_parent(path: str | Path) -> Path:
    """Create and return the parent directory for ``path``."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

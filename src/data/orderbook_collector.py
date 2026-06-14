"""Collect Binance Level-2 snapshots and persist them to HDF5."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import requests
from loguru import logger

from .common import ensure_parent, normalize_symbol


class OrderBookCollector:
    """Fetch, validate, derive, and store Level-2 order-book snapshots."""

    def __init__(
        self,
        *,
        rest_base_url: str = "https://api.binance.com",
        session: requests.Session | None = None,
        timeout: float = 10.0,
        top_levels_for_obi: int = 5,
    ) -> None:
        self.rest_base_url = rest_base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.top_levels_for_obi = top_levels_for_obi
        self.session.headers.setdefault(
            "User-Agent", "qrw-market-microstructure/phase2"
        )

    def get_snapshot(self, symbol: str, depth: int = 20) -> dict[str, Any]:
        """Return a normalized Binance order-book snapshot."""
        if not 1 <= depth <= 5000:
            raise ValueError("depth must be between 1 and 5000")

        response = self.session.get(
            f"{self.rest_base_url}/api/v3/depth",
            params={"symbol": normalize_symbol(symbol), "limit": depth},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        snapshot = {
            "timestamp": time.time_ns(),
            "last_update_id": int(payload["lastUpdateId"]),
            "bids": [(float(price), float(quantity)) for price, quantity in payload["bids"]],
            "asks": [(float(price), float(quantity)) for price, quantity in payload["asks"]],
        }
        self.validate_snapshot(snapshot)
        snapshot.update(self.calculate_metrics(snapshot))
        return snapshot

    def calculate_metrics(
        self,
        snapshot: dict[str, Any],
        *,
        top_levels: int | None = None,
    ) -> dict[str, float]:
        """Calculate spread, mid-price, OBI, and volume-weighted mid price."""
        self.validate_snapshot(snapshot)
        levels = top_levels or self.top_levels_for_obi
        bids = np.asarray(snapshot["bids"], dtype="float64")
        asks = np.asarray(snapshot["asks"], dtype="float64")
        selected_bids = bids[:levels]
        selected_asks = asks[:levels]

        best_bid = float(bids[0, 0])
        best_ask = float(asks[0, 0])
        bid_quantity = float(selected_bids[:, 1].sum())
        ask_quantity = float(selected_asks[:, 1].sum())
        total_quantity = bid_quantity + ask_quantity
        obi = (bid_quantity - ask_quantity) / total_quantity if total_quantity else 0.0
        mid_price = (best_bid + best_ask) / 2.0
        vwmp = (
            (best_ask * bid_quantity + best_bid * ask_quantity) / total_quantity
            if total_quantity
            else mid_price
        )
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "bid_ask_spread": best_ask - best_bid,
            "obi": float(np.clip(obi, -1.0, 1.0)),
            "bid_quantity_top": bid_quantity,
            "ask_quantity_top": ask_quantity,
            "vwmp": vwmp,
        }

    @staticmethod
    def validate_snapshot(snapshot: dict[str, Any]) -> None:
        """Validate ordering, positivity, and non-crossing of a snapshot."""
        if not snapshot.get("bids") or not snapshot.get("asks"):
            raise ValueError("snapshot must contain non-empty bids and asks")
        bids = np.asarray(snapshot["bids"], dtype="float64")
        asks = np.asarray(snapshot["asks"], dtype="float64")
        if bids.ndim != 2 or asks.ndim != 2 or bids.shape[1] != 2 or asks.shape[1] != 2:
            raise ValueError("bids and asks must be sequences of (price, quantity)")
        if not np.isfinite(bids).all() or not np.isfinite(asks).all():
            raise ValueError("snapshot contains non-finite values")
        if (bids <= 0).any() or (asks <= 0).any():
            raise ValueError("prices and quantities must be positive")
        if np.any(np.diff(bids[:, 0]) > 0):
            raise ValueError("bids must be sorted from highest to lowest price")
        if np.any(np.diff(asks[:, 0]) < 0):
            raise ValueError("asks must be sorted from lowest to highest price")
        if bids[0, 0] >= asks[0, 0]:
            raise ValueError("order book is crossed or locked")

    def save_snapshot(
        self,
        save_path: str | Path,
        snapshot: dict[str, Any],
    ) -> str:
        """Append one snapshot under ``/snapshots/{timestamp}`` in HDF5."""
        self.validate_snapshot(snapshot)
        metrics = self.calculate_metrics(snapshot)
        timestamp = int(snapshot.get("timestamp", time.time_ns()))
        output = ensure_parent(save_path)

        with h5py.File(output, "a") as handle:
            snapshots = handle.require_group("snapshots")
            group_name = str(timestamp)
            while group_name in snapshots:
                timestamp += 1
                group_name = str(timestamp)
            group = snapshots.create_group(group_name)
            group.create_dataset(
                "bids", data=np.asarray(snapshot["bids"], dtype="float64")
            )
            group.create_dataset(
                "asks", data=np.asarray(snapshot["asks"], dtype="float64")
            )
            group.attrs["last_update_id"] = int(snapshot.get("last_update_id", -1))
            for name, value in metrics.items():
                group.attrs[name] = value

        logger.info("Saved LOB snapshot {} to {}", group_name, output)
        return group_name

    def collect(
        self,
        symbol: str,
        save_path: str | Path,
        *,
        depth: int = 20,
        interval_seconds: float = 1.0,
        n_snapshots: int = 60,
    ) -> Path:
        """Collect a finite series of REST snapshots at a fixed interval."""
        if n_snapshots < 1:
            raise ValueError("n_snapshots must be positive")
        if interval_seconds < 0:
            raise ValueError("interval_seconds cannot be negative")

        output = Path(save_path)
        next_snapshot_at = time.monotonic()
        for index in range(n_snapshots):
            snapshot = self.get_snapshot(symbol, depth=depth)
            self.save_snapshot(output, snapshot)
            if index + 1 < n_snapshots:
                next_snapshot_at += interval_seconds
                remaining = next_snapshot_at - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
        return output

    @staticmethod
    def snapshots_to_frame(path: str | Path) -> pd.DataFrame:
        """Load HDF5 snapshot attributes into a time-indexed DataFrame."""
        rows: list[dict[str, float | int]] = []
        with h5py.File(path, "r") as handle:
            if "snapshots" not in handle:
                raise ValueError("HDF5 file does not contain /snapshots")
            for timestamp, group in handle["snapshots"].items():
                row: dict[str, float | int] = {"timestamp": int(timestamp)}
                row.update({key: group.attrs[key] for key in group.attrs})
                rows.append(row)
        if not rows:
            raise ValueError("HDF5 file contains no snapshots")
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

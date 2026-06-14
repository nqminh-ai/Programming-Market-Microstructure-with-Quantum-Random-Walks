"""Collect synchronized Binance trades and partial order-book snapshots."""

from __future__ import annotations

import csv
import gzip
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import websocket
from loguru import logger

from .common import ensure_parent, normalize_symbol
from .orderbook_collector import OrderBookCollector


class LiveMarketCollector:
    """Stream trades and top-of-book depth over one combined WebSocket."""

    def __init__(
        self,
        *,
        websocket_base_url: str = "wss://stream.binance.com:9443",
        top_levels_for_obi: int = 5,
        receive_timeout: float = 10.0,
    ) -> None:
        if top_levels_for_obi < 1:
            raise ValueError("top_levels_for_obi must be positive")
        if receive_timeout <= 0:
            raise ValueError("receive_timeout must be positive")
        self.websocket_base_url = websocket_base_url.rstrip("/")
        self.receive_timeout = receive_timeout
        self.order_book = OrderBookCollector(
            top_levels_for_obi=top_levels_for_obi
        )

    def collect(
        self,
        symbol: str,
        tick_path: str | Path,
        lob_path: str | Path,
        *,
        duration_seconds: float,
        snapshot_interval_seconds: float = 1.0,
    ) -> dict[str, Any]:
        """Collect a finite synchronized window and return summary statistics."""
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if snapshot_interval_seconds <= 0:
            raise ValueError("snapshot_interval_seconds must be positive")

        normalized_symbol = normalize_symbol(symbol)
        stream_symbol = normalized_symbol.lower()
        streams = f"{stream_symbol}@trade/{stream_symbol}@depth20@100ms"
        url = (
            f"{self.websocket_base_url}/stream?streams={streams}"
            "&timeUnit=MICROSECOND"
        )

        tick_output = ensure_parent(tick_path)
        lob_output = ensure_parent(lob_path)
        started_ns = time.time_ns()
        deadline = time.monotonic() + duration_seconds
        snapshot_interval_ns = int(snapshot_interval_seconds * 1_000_000_000)
        last_snapshot_ns: int | None = None
        first_lob_ns: int | None = None
        last_lob_ns: int | None = None
        first_trade_ns: int | None = None
        last_trade_ns: int | None = None
        trade_count = 0
        snapshot_count = 0
        reconnect_count = 0

        write_header = not tick_output.exists() or tick_output.stat().st_size == 0
        with gzip.open(tick_output, "at", encoding="utf-8", newline="") as tick_file:
            writer = csv.DictWriter(
                tick_file,
                fieldnames=["timestamp", "price", "quantity", "side", "trade_id"],
            )
            if write_header:
                writer.writeheader()

            with h5py.File(lob_output, "a") as lob_file:
                snapshots = lob_file.require_group("snapshots")
                connection: websocket.WebSocket | None = None
                while time.monotonic() < deadline:
                    try:
                        if connection is None:
                            connection = websocket.create_connection(
                                url,
                                timeout=self.receive_timeout,
                                enable_multithread=False,
                            )
                            logger.info("Connected to {}", url)

                        raw_message = connection.recv()
                        if not raw_message:
                            raise ConnectionError("Binance WebSocket closed")
                        received_ns = time.time_ns()
                        message = json.loads(raw_message)
                        stream = str(message.get("stream", ""))
                        payload = message.get("data", {})

                        if stream.endswith("@depth20@100ms"):
                            if (
                                last_snapshot_ns is None
                                or received_ns - last_snapshot_ns
                                >= snapshot_interval_ns
                            ):
                                snapshot = self._depth_snapshot(payload, received_ns)
                                metrics = self.order_book.calculate_metrics(snapshot)
                                self._append_snapshot(
                                    snapshots,
                                    snapshot,
                                    metrics,
                                )
                                last_snapshot_ns = received_ns
                                first_lob_ns = first_lob_ns or received_ns
                                last_lob_ns = received_ns
                                snapshot_count += 1
                        elif stream.endswith("@trade") and first_lob_ns is not None:
                            trade = self._trade_row(
                                payload,
                                timestamp_ns=received_ns,
                            )
                            writer.writerow(trade)
                            trade_timestamp = int(trade["timestamp"])
                            first_trade_ns = first_trade_ns or trade_timestamp
                            last_trade_ns = trade_timestamp
                            trade_count += 1
                    except (
                        OSError,
                        TimeoutError,
                        ConnectionError,
                        ValueError,
                        json.JSONDecodeError,
                        websocket.WebSocketException,
                    ) as error:
                        if connection is not None:
                            connection.close()
                            connection = None
                        if time.monotonic() >= deadline:
                            break
                        reconnect_count += 1
                        logger.warning(
                            "Live stream interrupted ({}); reconnecting",
                            error,
                        )
                        time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))
                if connection is not None:
                    connection.close()

        if snapshot_count == 0:
            raise RuntimeError("no LOB snapshots were collected")
        if trade_count == 0:
            raise RuntimeError("no trades were collected after the first LOB snapshot")

        summary = {
            "symbol": normalized_symbol,
            "started_ns": started_ns,
            "finished_ns": time.time_ns(),
            "duration_seconds": duration_seconds,
            "snapshot_interval_seconds": snapshot_interval_seconds,
            "timestamp_clock": "local_receive_time_ns",
            "trades": trade_count,
            "lob_snapshots": snapshot_count,
            "reconnects": reconnect_count,
            "first_trade_ns": first_trade_ns,
            "last_trade_ns": last_trade_ns,
            "first_lob_ns": first_lob_ns,
            "last_lob_ns": last_lob_ns,
            "tick_path": str(tick_output),
            "lob_path": str(lob_output),
        }
        logger.info(
            "Collected {:,} trades and {:,} LOB snapshots",
            trade_count,
            snapshot_count,
        )
        return summary

    @staticmethod
    def _trade_row(
        payload: dict[str, Any],
        *,
        timestamp_ns: int | None = None,
    ) -> dict[str, Any]:
        required = {"t", "p", "q", "T", "m"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"trade event is missing fields: {missing}")
        timestamp = (
            int(payload["T"]) * 1_000
            if timestamp_ns is None
            else int(timestamp_ns)
        )
        price = float(payload["p"])
        quantity = float(payload["q"])
        if timestamp <= 0 or price <= 0 or quantity <= 0:
            raise ValueError("trade event contains invalid values")
        return {
            "timestamp": timestamp,
            "price": price,
            "quantity": quantity,
            "side": "sell" if bool(payload["m"]) else "buy",
            "trade_id": int(payload["t"]),
        }

    @staticmethod
    def _depth_snapshot(
        payload: dict[str, Any],
        timestamp_ns: int,
    ) -> dict[str, Any]:
        required = {"lastUpdateId", "bids", "asks"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"depth event is missing fields: {missing}")
        return {
            "timestamp": timestamp_ns,
            "last_update_id": int(payload["lastUpdateId"]),
            "bids": [
                (float(price), float(quantity))
                for price, quantity in payload["bids"]
            ],
            "asks": [
                (float(price), float(quantity))
                for price, quantity in payload["asks"]
            ],
        }

    @staticmethod
    def _append_snapshot(
        snapshots: h5py.Group,
        snapshot: dict[str, Any],
        metrics: dict[str, float],
    ) -> None:
        timestamp = int(snapshot["timestamp"])
        group_name = str(timestamp)
        while group_name in snapshots:
            timestamp += 1
            group_name = str(timestamp)
        group = snapshots.create_group(group_name)
        group.create_dataset(
            "bids",
            data=np.asarray(snapshot["bids"], dtype=np.float64),
        )
        group.create_dataset(
            "asks",
            data=np.asarray(snapshot["asks"], dtype=np.float64),
        )
        group.attrs["last_update_id"] = int(snapshot["last_update_id"])
        for name, value in metrics.items():
            group.attrs[name] = value

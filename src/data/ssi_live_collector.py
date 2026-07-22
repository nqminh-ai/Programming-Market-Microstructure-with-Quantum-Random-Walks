"""Thread-safe SSI FastConnect market-data collector for Streamlit.

The official ``ssi_fc_data`` client uses SignalR rather than a raw WebSocket.
Imports are lazy so the rest of the project remains usable without the
optional SDK. Credentials are read from environment variables or from the
explicit path in ``SSI_CREDENTIALS_FILE``; no repository path is implicit.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time as clock_time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
MARKET_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


def _read_credentials_file(path: Path) -> tuple[str, str]:
    """Read ConsumerID/ConsumerSecret without logging their values."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        return "", ""
    values: dict[str, str] = {}
    for line in lines:
        match = re.match(
            r"^\s*(consumerid|consumersecret)\s*[:=]\s*(.*?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            values[match.group(1).lower()] = match.group(2).strip(" \t\"'")
    return values.get("consumerid", ""), values.get("consumersecret", "")


@dataclass(frozen=True)
class SSIClientConfig:
    """Configuration shape expected by the official SSI Python SDK."""

    consumerID: str
    consumerSecret: str
    auth_type: str = "Bearer"
    url: str = "https://fc-data.ssi.com.vn/"
    stream_url: str = "https://fc-datahub.ssi.com.vn/"

    @classmethod
    def from_environment(cls) -> "SSIClientConfig":
        consumer_id = os.getenv("SSI_CONSUMER_ID", "").strip()
        consumer_secret = os.getenv("SSI_CONSUMER_SECRET", "").strip()
        if not (consumer_id and consumer_secret):
            configured_path = os.getenv("SSI_CREDENTIALS_FILE", "").strip()
            if configured_path:
                credentials_path = Path(configured_path).expanduser()
                file_id, file_secret = _read_credentials_file(credentials_path)
                consumer_id = consumer_id or file_id
                consumer_secret = consumer_secret or file_secret
        return cls(
            consumerID=consumer_id,
            consumerSecret=consumer_secret,
            url=os.getenv("SSI_API_URL", cls.url).strip().rstrip("/") + "/",
            stream_url=(
                os.getenv("SSI_STREAM_URL", cls.stream_url).strip().rstrip("/")
                + "/"
            ),
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.consumerID and self.consumerSecret)


class SSILiveCollector:
    """Collect SSI ticks and depth into a bounded in-memory DataFrame.

    ``VN30F1M`` is treated as a front-month alias. The concrete derivative
    symbol is discovered from SSI when possible, or can be pinned with
    ``SSI_SYMBOL_VN30F1M``.
    """

    STATUS_STOPPED = "STOPPED"
    STATUS_CONNECTING = "CONNECTING"
    STATUS_LIVE = "LIVE"
    STATUS_REST_FALLBACK = "REST FALLBACK"
    STATUS_DISCONNECTED = "DISCONNECTED"
    STATUS_MARKET_CLOSED = "MARKET CLOSED"
    STATUS_MISSING_CREDENTIALS = "MISSING CREDENTIALS"
    STATUS_ERROR = "ERROR"

    INDEX_SYMBOLS = {"VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX"}
    FRAME_COLUMNS = [
        "timestamp",
        "symbol",
        "price",
        "quantity",
        "side",
        "obi",
        "imbalance_source",
        "event_type",
        "source",
    ]

    def __init__(
        self,
        symbol: str = "VN30F1M",
        *,
        max_events: int = 1000,
        depth_levels: int = 5,
        trade_flow_window: int = 100,
        reconnect_initial_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
        rest_poll_seconds: float = 5.0,
        market_check_seconds: float = 30.0,
        config: SSIClientConfig | None = None,
        client_factory: Callable[[Any], Any] | None = None,
        stream_factory: Callable[..., Any] | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must not be empty")
        if int(max_events) < 1:
            raise ValueError("max_events must be positive")
        if int(depth_levels) < 1 or int(depth_levels) > 10:
            raise ValueError("depth_levels must be between 1 and 10")
        if int(trade_flow_window) < 1:
            raise ValueError("trade_flow_window must be positive")
        if reconnect_initial_seconds <= 0 or reconnect_max_seconds <= 0:
            raise ValueError("reconnect delays must be positive")
        if rest_poll_seconds <= 0 or market_check_seconds <= 0:
            raise ValueError("poll intervals must be positive")

        self.symbol = normalized_symbol
        self.stream_symbol = normalized_symbol
        self.max_events = int(max_events)
        self.depth_levels = int(depth_levels)
        self.config = config or SSIClientConfig.from_environment()
        self.reconnect_initial_seconds = float(reconnect_initial_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)
        self.rest_poll_seconds = float(rest_poll_seconds)
        self.market_check_seconds = float(market_check_seconds)
        self._client_factory = client_factory or self._default_client_factory
        self._stream_factory = stream_factory or self._default_stream_factory
        self._now_provider = now_provider or (lambda: datetime.now(MARKET_TIMEZONE))

        self._buffer: deque[dict[str, Any]] = deque(maxlen=self.max_events)
        self._trade_flow: deque[float] = deque(maxlen=int(trade_flow_window))
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._disconnect_event = threading.Event()
        self._connected_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: Any = None
        self._client: Any = None
        self._status = self.STATUS_STOPPED
        self._last_error: str | None = None
        self._last_update: pd.Timestamp | None = None
        self._reconnect_count = 0
        self._last_fingerprint: tuple[Any, ...] | None = None

    @staticmethod
    def _default_client_factory(config: Any) -> Any:
        try:
            from ssi_fc_data.fc_md_client import MarketDataClient
        except ImportError as error:
            raise RuntimeError(
                "ssi-fc-data is not installed; install requirements.txt or the "
                "local SSI tarball"
            ) from error
        return MarketDataClient(config)

    @staticmethod
    def _default_stream_factory(config: Any, client: Any, **callbacks: Any) -> Any:
        try:
            from ssi_fc_data.fc_md_stream import MarketDataStream
        except ImportError as error:
            raise RuntimeError("ssi-fc-data is not installed") from error
        return MarketDataStream(config, client, **callbacks)

    @classmethod
    def is_market_open(cls, at: datetime | None = None) -> bool:
        """Return whether ``at`` is inside the requested weekday session."""
        current = at or datetime.now(MARKET_TIMEZONE)
        if current.tzinfo is None:
            current = current.replace(tzinfo=MARKET_TIMEZONE)
        else:
            current = current.astimezone(MARKET_TIMEZONE)
        current_time = current.time().replace(tzinfo=None)
        return (
            current.weekday() < 5
            and clock_time(9, 0) <= current_time <= clock_time(14, 45)
        )

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start one daemon worker; repeated calls are idempotent."""
        with self._lock:
            if self.is_running:
                return True
            if not self.config.has_credentials:
                self._status = self.STATUS_MISSING_CREDENTIALS
                self._last_error = (
                    "Set SSI_CONSUMER_ID and SSI_CONSUMER_SECRET before enabling live mode"
                )
                return False
            self._stop_event.clear()
            self._disconnect_event.clear()
            self._status = (
                self.STATUS_CONNECTING
                if self.is_market_open(self._now_provider())
                else self.STATUS_MARKET_CLOSED
            )
            self._last_error = None
            self._thread = threading.Thread(
                target=self._run,
                name=f"ssi-live-{self.symbol.lower()}",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the worker and close the active SignalR transport."""
        self._stop_event.set()
        self._disconnect_event.set()
        stream = self._stream
        connection = getattr(stream, "connection", None)
        if connection is not None:
            try:
                connection.close()
            except Exception:
                LOGGER.debug("SSI stream close failed", exc_info=True)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(float(timeout), 0.0))
        with self._lock:
            self._thread = None
            self._stream = None
            self._client = None
            self._status = self.STATUS_STOPPED

    def status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "symbol": self.symbol,
                "stream_symbol": self.stream_symbol,
                "events": len(self._buffer),
                "last_update": self._last_update,
                "last_error": self._last_error,
                "reconnects": self._reconnect_count,
            }

    def snapshot(self) -> pd.DataFrame:
        """Return a detached, chronologically ordered buffer snapshot."""
        with self._lock:
            records = [record.copy() for record in self._buffer]
        if not records:
            return pd.DataFrame(columns=self.FRAME_COLUMNS)
        frame = pd.DataFrame.from_records(records, columns=self.FRAME_COLUMNS)
        return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)

    @staticmethod
    def _normalized_fields(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            re.sub(r"[^a-z0-9]", "", str(key).lower()): value
            for key, value in payload.items()
        }

    @staticmethod
    def _positive_float(*values: Any) -> float | None:
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number) and number > 0:
                return number
        return None

    @staticmethod
    def _timestamp(fields: dict[str, Any]) -> pd.Timestamp:
        trading_date = fields.get("tradingdate", "")
        time_value = fields.get("time", "")
        candidates = []
        if trading_date and time_value:
            candidates.append(f"{trading_date} {time_value}")
        if time_value:
            candidates.append(str(time_value))
        for candidate in candidates:
            parsed = pd.to_datetime(candidate, dayfirst=True, errors="coerce")
            if pd.notna(parsed):
                stamp = pd.Timestamp(parsed)
                return (
                    stamp.tz_localize(MARKET_TIMEZONE)
                    if stamp.tzinfo is None
                    else stamp.tz_convert(MARKET_TIMEZONE)
                )
        return pd.Timestamp.now(tz=MARKET_TIMEZONE)

    def _depth_imbalance(self, fields: dict[str, Any]) -> float | None:
        bid_volume = 0.0
        ask_volume = 0.0
        for level in range(1, self.depth_levels + 1):
            bid_volume += self._positive_float(fields.get(f"bidvol{level}")) or 0.0
            ask_volume += self._positive_float(fields.get(f"askvol{level}")) or 0.0
        total = bid_volume + ask_volume
        if total <= 0:
            return None
        return float(np.clip((bid_volume - ask_volume) / total, -1.0, 1.0))

    def _trade_flow_imbalance(self, side: str, quantity: float) -> float:
        signed_quantity = quantity if side == "buy" else -quantity if side == "sell" else 0.0
        # The feature attached to this trade must contain only information that
        # existed before the trade arrived.  Append the current signed volume
        # only after computing the lagged flow imbalance.
        gross = float(sum(abs(value) for value in self._trade_flow))
        imbalance = float(sum(self._trade_flow) / gross) if gross > 0 else 0.0
        if signed_quantity:
            self._trade_flow.append(float(signed_quantity))
        return imbalance

    def parse_message(self, message: str | dict[str, Any]) -> dict[str, Any] | None:
        """Map one SSI stream message to the platform's canonical tick schema."""
        wrapper = json.loads(message) if isinstance(message, str) else dict(message)
        data_type = str(wrapper.get("DataType", wrapper.get("dataType", "")))
        content = wrapper.get("Content", wrapper.get("content", wrapper))
        if isinstance(content, str):
            content = json.loads(content)
        if isinstance(content, list):
            content = content[-1] if content else {}
        if not isinstance(content, dict):
            return None

        fields = self._normalized_fields(content)
        event_type = str(fields.get("rtype", data_type)).upper().replace("_", "-")
        symbol = str(fields.get("symbol", fields.get("indexid", self.symbol))).upper()

        bid_price = self._positive_float(fields.get("bidprice1"))
        ask_price = self._positive_float(fields.get("askprice1"))
        midpoint = (
            (bid_price + ask_price) / 2.0
            if bid_price is not None and ask_price is not None
            else None
        )
        price = self._positive_float(
            fields.get("lastprice"),
            fields.get("indexvalue"),
            fields.get("close"),
            fields.get("estmatchedprice"),
            fields.get("indexvalest"),
            midpoint,
        )
        if price is None:
            return None

        quantity = self._positive_float(
            fields.get("lastvol"),
            fields.get("volume"),
            fields.get("totalqtty"),
            fields.get("totalvol"),
        ) or 1.0
        raw_side = str(fields.get("side", "")).strip().upper()
        side = "buy" if raw_side in {"BU", "BUY", "B"} else "sell" if raw_side in {"SD", "SELL", "S"} else "neutral"

        with self._lock:
            flow_imbalance = self._trade_flow_imbalance(side, quantity)
            depth_imbalance = self._depth_imbalance(fields)
        imbalance_source = "order_book" if depth_imbalance is not None else "trade_flow"
        obi = depth_imbalance if depth_imbalance is not None else flow_imbalance
        return {
            "timestamp": self._timestamp(fields),
            "symbol": symbol,
            "price": float(price),
            "quantity": float(quantity),
            "side": side,
            "obi": float(obi),
            "imbalance_source": imbalance_source,
            "event_type": event_type or data_type.upper() or "UNKNOWN",
            "source": "ssi_stream",
        }

    def ingest_message(self, message: str | dict[str, Any]) -> dict[str, Any] | None:
        """Parse and append one non-duplicate event."""
        try:
            record = self.parse_message(message)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._record_error(f"Invalid SSI message: {error}", disconnect=False)
            return None
        if record is None:
            return None

        fingerprint = (
            record["symbol"],
            record["timestamp"],
            record["price"],
            record["quantity"],
            record["side"],
            record["obi"],
            record["event_type"],
        )
        with self._lock:
            if fingerprint == self._last_fingerprint:
                return record
            self._buffer.append(record)
            self._last_fingerprint = fingerprint
            self._last_update = record["timestamp"]
            self._status = self.STATUS_LIVE
            self._last_error = None
        return record

    def _record_error(self, error: Any, *, disconnect: bool = True) -> None:
        with self._lock:
            self._last_error = str(error)
            if disconnect:
                self._status = self.STATUS_DISCONNECTED
        if disconnect:
            self._disconnect_event.set()

    def _on_open(self) -> None:
        self._connected_event.set()
        with self._lock:
            self._status = self.STATUS_LIVE
            self._last_error = None

    def _on_close(self) -> None:
        self._record_error("SSI stream closed")

    def _on_error(self, error: Any) -> None:
        self._record_error(error)

    def _stream_channel(self) -> str:
        prefix = "MI" if self.stream_symbol in self.INDEX_SYMBOLS else "X"
        return f"{prefix}:{self.stream_symbol}"

    def _resolve_front_month(self, client: Any) -> str:
        override = os.getenv("SSI_SYMBOL_VN30F1M", "").strip().upper()
        if self.symbol != "VN30F1M":
            return self.symbol
        if override:
            return override
        try:
            from ssi_fc_data import model

            response = client.securities(self.config, model.securities("DER", 1, 1000))
            rows = response.get("data", []) if isinstance(response, dict) else []
            candidates: list[tuple[int, str]] = []
            for row in rows:
                symbol = str(row.get("Symbol", row.get("symbol", ""))).upper()
                match = re.fullmatch(r"VN30F(\d{2})(\d{2})", symbol)
                if match:
                    year, month = int(match.group(1)), int(match.group(2))
                    candidates.append(((2000 + year) * 100 + month, symbol))
            current = self._now_provider().astimezone(MARKET_TIMEZONE)
            current_month = current.year * 100 + current.month
            future = sorted(item for item in candidates if item[0] >= current_month)
            if future:
                return future[0][1]
        except Exception as error:
            LOGGER.warning("Could not resolve VN30F1M alias: %s", error)
        return self.symbol

    def _connect_stream(self) -> None:
        self._connected_event.clear()
        self._disconnect_event.clear()
        with self._lock:
            self._status = self.STATUS_CONNECTING
        client = self._client_factory(self.config)
        self._client = client
        self.stream_symbol = self._resolve_front_month(client)
        stream = self._stream_factory(
            self.config,
            client,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        self._stream = stream
        stream.start(self.ingest_message, self._on_error, self._stream_channel())

        connected = self._connected_event.wait(timeout=10.0)
        if not connected:
            raise ConnectionError("SSI stream connection timed out")
        while not self._stop_event.is_set() and not self._disconnect_event.wait(0.5):
            pass

    def _poll_rest_once(self) -> bool:
        client = self._client or self._client_factory(self.config)
        self._client = client
        try:
            from ssi_fc_data import model
        except ImportError as error:
            raise RuntimeError("ssi-fc-data is not installed") from error
        today = self._now_provider().astimezone(MARKET_TIMEZONE).strftime("%d/%m/%Y")
        request = model.intraday_ohlc(
            self.stream_symbol,
            today,
            today,
            1,
            10,
            True,
            1,
        )
        response = client.intraday_ohlc(self.config, request)
        rows = response.get("data", []) if isinstance(response, dict) else []
        if not rows:
            return False
        record = self.ingest_message({"DataType": "B", "Content": rows[-1]})
        if record is None:
            return False
        with self._lock:
            if self._buffer:
                self._buffer[-1]["source"] = "ssi_rest"
            self._status = self.STATUS_REST_FALLBACK
        return True

    def _close_stream(self) -> None:
        connection = getattr(self._stream, "connection", None)
        if connection is not None:
            try:
                connection.close()
            except Exception:
                LOGGER.debug("SSI connection cleanup failed", exc_info=True)
        self._stream = None

    def _run(self) -> None:
        backoff = self.reconnect_initial_seconds
        try:
            while not self._stop_event.is_set():
                now = self._now_provider()
                if not self.is_market_open(now):
                    with self._lock:
                        self._status = self.STATUS_MARKET_CLOSED
                        self._last_error = None
                    self._stop_event.wait(self.market_check_seconds)
                    continue

                try:
                    self._connect_stream()
                    backoff = self.reconnect_initial_seconds
                except Exception as error:
                    self._record_error(error)
                finally:
                    self._close_stream()

                if self._stop_event.is_set():
                    break
                with self._lock:
                    self._reconnect_count += 1
                try:
                    self._poll_rest_once()
                except Exception as error:
                    self._record_error(f"SSI stream and REST unavailable: {error}")
                self._stop_event.wait(min(backoff, self.rest_poll_seconds))
                backoff = min(backoff * 2.0, self.reconnect_max_seconds)
        finally:
            self._close_stream()
            if not self._stop_event.is_set():
                with self._lock:
                    self._status = self.STATUS_ERROR


def sanitized_config_from_environment() -> SimpleNamespace:
    """Expose non-secret runtime configuration for diagnostics."""
    config = SSIClientConfig.from_environment()
    return SimpleNamespace(
        credentials_configured=config.has_credentials,
        url=config.url,
        stream_url=config.stream_url,
    )

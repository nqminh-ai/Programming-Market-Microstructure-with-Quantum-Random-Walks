"""Tests for the bounded SSI FastConnect live-data adapter."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.data.ssi_live_collector import (
    MARKET_TIMEZONE,
    SSIClientConfig,
    SSILiveCollector,
)


def collector(**kwargs) -> SSILiveCollector:
    return SSILiveCollector(
        config=SSIClientConfig(consumerID="test-id", consumerSecret="test-secret"),
        **kwargs,
    )


def trade_message(index: int = 0, *, side: str = "BU") -> dict:
    return {
        "DataType": "X-Trade",
        "Content": json.dumps(
            {
                "RType": "X-TRADE",
                "TradingDate": "27/06/2026",
                "Time": f"09:00:{index % 60:02d}",
                "Symbol": "VN30F2607",
                "LastPrice": 1345.0 + index,
                "LastVol": 5 + index,
                "Side": side,
            }
        ),
    }


def test_ssi_parser_handles_standard_events() -> None:
    live = collector()
    record = live.ingest_message(trade_message())

    assert record is not None
    assert record["symbol"] == "VN30F2607"
    assert record["price"] == pytest.approx(1345.0)
    assert record["quantity"] == pytest.approx(5.0)
    assert record["side"] == "buy"
    assert record["obi"] == pytest.approx(0.0)
    assert record["imbalance_source"] == "trade_flow"
    assert record["timestamp"].tzinfo is not None


def test_trade_flow_imbalance_is_lagged_before_current_trade() -> None:
    live = collector()

    first = live.ingest_message(trade_message(0, side="BU"))
    second = live.ingest_message(trade_message(1, side="SD"))
    third = live.ingest_message(trade_message(2, side="SD"))

    assert first is not None and first["obi"] == pytest.approx(0.0)
    assert second is not None and second["obi"] == pytest.approx(1.0)
    assert third is not None
    assert third["obi"] == pytest.approx((5.0 - 6.0) / (5.0 + 6.0))


def test_order_book_imbalance_takes_precedence_over_trade_flow() -> None:
    live = collector(depth_levels=2)
    record = live.ingest_message(
        {
            "DataType": "X",
            "Content": {
                "RType": "X",
                "TradingDate": "27/06/2026",
                "Time": "09:01:00",
                "Symbol": "VN30F2607",
                "LastPrice": 1346,
                "LastVol": 10,
                "Side": "SD",
                "BidVol1": 200,
                "BidVol2": 100,
                "AskVol1": 60,
                "AskVol2": 40,
            },
        }
    )

    assert record is not None
    assert record["obi"] == pytest.approx(0.5)
    assert record["imbalance_source"] == "order_book"


def test_buffer_does_not_overflow() -> None:
    live = collector(max_events=1000)
    for index in range(1005):
        live.ingest_message(trade_message(index))

    frame = live.snapshot()
    assert len(frame) == 1000
    assert frame["price"].min() == pytest.approx(1350.0)
    assert frame["price"].max() == pytest.approx(2349.0)


def test_market_hours_are_weekday_only() -> None:
    monday_open = datetime(2026, 6, 29, 10, 0, tzinfo=MARKET_TIMEZONE)
    monday_late = datetime(2026, 6, 29, 15, 0, tzinfo=MARKET_TIMEZONE)
    saturday = datetime(2026, 6, 27, 10, 0, tzinfo=MARKET_TIMEZONE)

    assert SSILiveCollector.is_market_open(monday_open)
    assert not SSILiveCollector.is_market_open(monday_late)
    assert not SSILiveCollector.is_market_open(saturday)


def test_missing_credentials_never_starts_network_thread() -> None:
    live = SSILiveCollector(
        config=SSIClientConfig(consumerID="", consumerSecret=""),
    )

    assert live.start() is False
    assert live.status == live.STATUS_MISSING_CREDENTIALS
    assert not live.is_running


def test_repository_credential_file_is_never_loaded_implicitly(monkeypatch) -> None:
    monkeypatch.delenv("SSI_CONSUMER_ID", raising=False)
    monkeypatch.delenv("SSI_CONSUMER_SECRET", raising=False)
    monkeypatch.delenv("SSI_CREDENTIALS_FILE", raising=False)

    config = SSIClientConfig.from_environment()

    assert config.consumerID == ""
    assert config.consumerSecret == ""


def test_credentials_can_load_from_private_local_file(tmp_path, monkeypatch) -> None:
    credential_file = tmp_path / "ssi-credentials.txt"
    credential_file.write_text(
        "ConsumerID: local-test-id\nConsumerSecret: local-test-secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SSI_CONSUMER_ID", raising=False)
    monkeypatch.delenv("SSI_CONSUMER_SECRET", raising=False)
    monkeypatch.setenv("SSI_CREDENTIALS_FILE", str(credential_file))

    config = SSIClientConfig.from_environment()
    assert config.consumerID == "local-test-id"
    assert config.consumerSecret == "local-test-secret"


def test_environment_credentials_override_private_file(tmp_path, monkeypatch) -> None:
    credential_file = tmp_path / "ssi-credentials.txt"
    credential_file.write_text(
        "ConsumerID: file-id\nConsumerSecret: file-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SSI_CREDENTIALS_FILE", str(credential_file))
    monkeypatch.setenv("SSI_CONSUMER_ID", "environment-id")
    monkeypatch.setenv("SSI_CONSUMER_SECRET", "environment-secret")

    config = SSIClientConfig.from_environment()
    assert config.consumerID == "environment-id"
    assert config.consumerSecret == "environment-secret"


def test_rest_fallback_uses_same_canonical_buffer_schema() -> None:
    class FakeClient:
        def intraday_ohlc(self, _config, _request):
            return {
                "data": [
                    {
                        "TradingDate": "27/06/2026",
                        "Time": "09:10:00",
                        "Symbol": "VN30F2607",
                        "Close": 1350.5,
                        "Volume": 12,
                    }
                ]
            }

    live = collector()
    live.stream_symbol = "VN30F2607"
    live._client = FakeClient()

    assert live._poll_rest_once()
    frame = live.snapshot()
    assert len(frame) == 1
    assert frame.iloc[0]["source"] == "ssi_rest"
    assert frame.iloc[0]["price"] == pytest.approx(1350.5)
    assert live.status == live.STATUS_REST_FALLBACK


def test_trade_flow_imbalance_excludes_current_trade() -> None:
    """OBI returned for trade t must NOT include trade t's own volume.

    This test verifies that _trade_flow_imbalance computes the imbalance
    from the *existing* buffer BEFORE appending the new trade — confirming
    that data leakage (self-prediction) is absent in the live collector.
    """
    from collections import deque

    # Use __new__ to bypass __init__ dependency on SSIClientConfig
    live = SSILiveCollector.__new__(SSILiveCollector)
    # Pre-seed buffer: two buys (+1, +1) and one sell (-1)
    # Net = +1, Gross = 3  →  imbalance = 1/3
    live._trade_flow = deque([1.0, 1.0, -1.0], maxlen=100)

    # Call with a large sell order (should NOT affect returned imbalance)
    imbalance = live._trade_flow_imbalance("sell", 100.0)

    # Returned imbalance must be based on OLD buffer only: +1/3
    assert imbalance == pytest.approx(1.0 / 3.0), (
        f"Expected lagged imbalance 1/3, got {imbalance:.6f}. "
        "Possible data leakage: current trade volume was included."
    )
    # After the call, the sell volume must now be in the buffer
    assert len(live._trade_flow) == 4
    assert live._trade_flow[-1] == pytest.approx(-100.0)

"""Download and normalize Binance historical spot trades."""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger

from .common import (
    coerce_tick_frame,
    ensure_parent,
    normalize_symbol,
    validate_tick_frame,
)


BINANCE_TRADE_COLUMNS = [
    "trade_id",
    "price",
    "quantity",
    "quote_quantity",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]
BINANCE_AGG_TRADE_COLUMNS = [
    "trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]


class TickDownloader:
    """Download daily Binance public-data archives into canonical gzip CSV files."""

    def __init__(
        self,
        *,
        base_url: str = "https://data.binance.vision/data/spot/daily",
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.session.headers.setdefault(
            "User-Agent", "qrw-market-microstructure/phase2"
        )

    def download_historical(
        self,
        symbol: str,
        start_date: str | date,
        end_date: str | date,
        save_path: str | Path,
        *,
        dataset: str = "trades",
        skip_missing: bool = False,
    ) -> list[Path]:
        """Download an inclusive date range of Binance daily trade archives."""
        normalized_symbol = normalize_symbol(symbol)
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        if end < start:
            raise ValueError("end_date must be on or after start_date")
        if dataset not in {"trades", "aggTrades"}:
            raise ValueError("dataset must be 'trades' or 'aggTrades'")

        destination = Path(save_path)
        destination.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []

        current = start
        while current <= end:
            try:
                frame = self.download_day(normalized_symbol, current, dataset=dataset)
            except requests.HTTPError as error:
                if skip_missing and error.response is not None and error.response.status_code == 404:
                    logger.warning("No Binance archive for {} on {}", normalized_symbol, current)
                    current += timedelta(days=1)
                    continue
                raise

            output = destination / f"tick_{normalized_symbol}_{current.isoformat()}.csv.gz"
            frame.to_csv(output, index=False, compression="gzip")
            self._log_quality(frame, output)
            outputs.append(output)
            current += timedelta(days=1)

        return outputs

    def download_day(
        self,
        symbol: str,
        day: str | date,
        *,
        dataset: str = "trades",
    ) -> pd.DataFrame:
        """Download and normalize one Binance daily ZIP archive."""
        normalized_symbol = normalize_symbol(symbol)
        parsed_day = self._parse_date(day)
        filename = f"{normalized_symbol}-{dataset}-{parsed_day.isoformat()}.zip"
        url = f"{self.base_url}/{dataset}/{normalized_symbol}/{filename}"
        logger.info("Downloading {}", url)

        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return self.parse_archive(response.content, dataset=dataset)

    @staticmethod
    def parse_archive(content: bytes, *, dataset: str = "trades") -> pd.DataFrame:
        """Parse a Binance daily ZIP payload into the canonical tick schema."""
        columns = (
            BINANCE_TRADE_COLUMNS
            if dataset == "trades"
            else BINANCE_AGG_TRADE_COLUMNS
        )
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_names = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if len(csv_names) != 1:
                raise ValueError(
                    f"expected exactly one CSV in archive, found {len(csv_names)}"
                )
            with archive.open(csv_names[0]) as csv_file:
                raw = pd.read_csv(csv_file, header=None, names=columns)

        canonical = pd.DataFrame(
            {
                "timestamp": raw["timestamp"],
                "price": raw["price"],
                "quantity": raw["quantity"],
                "side": raw["is_buyer_maker"],
                "trade_id": raw["trade_id"],
            }
        )
        canonical = coerce_tick_frame(canonical)
        canonical = canonical.sort_values(["timestamp", "trade_id"], kind="stable")
        canonical = canonical.reset_index(drop=True)
        validate_tick_frame(canonical)
        return canonical

    def import_csv(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        column_mapping: dict[str, str] | None = None,
    ) -> Path:
        """Normalize a local CSV/Kaggle export and save it as canonical gzip CSV."""
        frame = pd.read_csv(source)
        if column_mapping:
            frame = frame.rename(columns=column_mapping)
        frame = coerce_tick_frame(frame)
        frame = frame.sort_values(["timestamp", "trade_id"], kind="stable")
        frame = frame.drop_duplicates("trade_id", keep="first").reset_index(drop=True)
        validate_tick_frame(frame)

        output = ensure_parent(destination)
        frame.to_csv(output, index=False, compression="gzip")
        self._log_quality(frame, output)
        return output

    @staticmethod
    def _parse_date(value: str | date) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.fromisoformat(value)

    @staticmethod
    def _log_quality(frame: pd.DataFrame, output: Path) -> None:
        gap_threshold_ns = 5 * 60 * 1_000_000_000
        gaps = int((frame["timestamp"].diff() > gap_threshold_ns).sum())
        logger.info(
            "Saved {:,} records to {} (time gaps >5m: {})",
            len(frame),
            output,
            gaps,
        )

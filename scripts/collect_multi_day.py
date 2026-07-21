"""Collect N consecutive days of tick data for a Binance symbol.

Two data sources are supported:
  dump: Binance Vision bulk daily .zip files (fast, no API key needed).
  api:  Binance /api/v3/aggTrades REST endpoint (slower, paginates by ms).
  auto: Try dump first; fall back to api if the dump is unavailable.

Output: one Parquet file per day at <output_dir>/<symbol_lower>/YYYY-MM-DD.parquet
Schema: timestamp (int64, epoch ns), price (float64), quantity (float64),
        side (str "buy"/"sell"), trade_id (int64), day (date string YYYY-MM-DD)

Usage examples
--------------
# Last 3 days via auto-select (dump preferred):
python scripts/collect_multi_day.py --days 3 --source auto

# Explicit date range via dump only:
python scripts/collect_multi_day.py --start-date 2025-01-01 --end-date 2025-01-10 --source dump

# Force REST API for a specific range:
python scripts/collect_multi_day.py --start-date 2025-01-01 --days 5 --source api
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_multi_day")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_VISION_URL = (
    "https://data.binance.vision/data/spot/daily/trades"
    "/{symbol}/{symbol}-trades-{date}.zip"
)
BINANCE_API_URL = "https://api.binance.com/api/v3/aggTrades"

MAX_RETRIES = 3
RETRY_BASE_SECONDS = 2.0
API_PAGE_SIZE = 1000           # max trades per aggTrades call
API_SLEEP_SECONDS = 0.055      # ~18 req/s → well under 1200/min limit
API_BACKOFF_429_SECONDS = 60.0

# Binance Vision daily trade CSV format (no header, 7 or 6 columns):
# trade_id, price, qty, quote_qty, time, is_buyer_maker[, is_best_match]
# 'time' is epoch microseconds (16 digits) in newer files, epoch ms (13 digits) in older.
DUMP_CSV_7_COLUMNS = ["trade_id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]
DUMP_CSV_6_COLUMNS = ["trade_id", "price", "qty", "quote_qty", "time", "is_buyer_maker"]

OUTPUT_SCHEMA_COLUMNS = ["timestamp", "price", "quantity", "side", "trade_id", "day"]


# ---------------------------------------------------------------------------
# Data-fetch helpers
# ---------------------------------------------------------------------------


def _fetch_url_bytes(url: str, *, timeout: int = 60) -> bytes:
    """Fetch ``url`` and return raw bytes; raises ``urllib.error.URLError``."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "QRW-data-collector/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _fetch_with_retry(url: str, *, timeout: int = 60) -> bytes | None:
    """Fetch URL with up to MAX_RETRIES retries; return None on permanent failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _fetch_url_bytes(url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                logger.debug("404 Not Found: %s", url)
                return None   # file genuinely absent — no retry needed
            wait = RETRY_BASE_SECONDS ** attempt
            logger.warning(
                "HTTP %s on attempt %d/%d for %s — retrying in %.1fs",
                exc.code, attempt, MAX_RETRIES, url, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
        except (urllib.error.URLError, OSError) as exc:
            wait = RETRY_BASE_SECONDS ** attempt
            logger.warning(
                "Network error on attempt %d/%d for %s: %s — retrying in %.1fs",
                attempt, MAX_RETRIES, url, exc, wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    logger.error("All %d attempts failed for %s", MAX_RETRIES, url)
    return None


# ---------------------------------------------------------------------------
# Source: Binance Vision dump
# ---------------------------------------------------------------------------


def _normalize_is_buyer_maker(series: pd.Series) -> pd.Series:
    """Map isBuyerMaker True→'sell', False→'buy' (buyer is the passive side)."""
    if pd.api.types.is_bool_dtype(series):
        return series.map({True: "sell", False: "buy"})
    lowered = series.astype(str).str.lower().str.strip()
    return lowered.map({"true": "sell", "false": "buy", "1": "sell", "0": "buy"})


def _parse_dump_csv(raw_bytes: bytes, day_str: str) -> pd.DataFrame | None:
    """Extract and parse a Binance Vision daily trade CSV from a zip archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            names = zf.namelist()
            csv_names = [n for n in names if n.endswith(".csv")]
            if not csv_names:
                logger.error("Zip archive contains no .csv file: %s", names)
                return None
            with zf.open(csv_names[0]) as csv_file:
                raw_content = csv_file.read()
    except zipfile.BadZipFile as exc:
        logger.error("Bad zip file for %s: %s", day_str, exc)
        return None

    try:
        # Detect number of columns from the first data row
        first_line = raw_content[: raw_content.index(b"\n")].decode("ascii", errors="replace").strip()
        n_cols = len(first_line.split(","))
        if n_cols >= 7:
            col_names = DUMP_CSV_7_COLUMNS
        else:
            col_names = DUMP_CSV_6_COLUMNS

        # Binance Vision files do NOT have a header row
        df = pd.read_csv(
            io.BytesIO(raw_content),
            header=None,
            names=col_names,
            dtype=str,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("CSV parse failed for %s: %s", day_str, exc)
        return None

    # Coerce and rename to canonical schema
    out = pd.DataFrame()
    out["trade_id"] = pd.to_numeric(df["trade_id"], errors="coerce").astype("Int64")
    out["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")
    out["quantity"] = pd.to_numeric(df["qty"], errors="coerce").astype("float64")

    # 'time' unit detection: 16 digits = epoch µs, 13 digits = epoch ms
    time_raw = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
    # A 16-digit integer is > 1e15; a 13-digit is < 1e14
    sample_time = int(time_raw.dropna().iloc[0]) if not time_raw.dropna().empty else 0
    if sample_time > 1_000_000_000_000_000:   # epoch µs (16 digits)
        out["timestamp"] = (time_raw.astype("int64") * 1_000).astype("int64")  # µs → ns
    else:                                       # epoch ms (13 digits)
        out["timestamp"] = (time_raw.astype("int64") * 1_000_000).astype("int64")  # ms → ns

    out["side"] = _normalize_is_buyer_maker(df["is_buyer_maker"])
    out["day"] = day_str

    bad = out["trade_id"].isna() | out["price"].isna() | out["quantity"].isna()
    if bad.any():
        logger.warning("Dropped %d rows with missing values for %s", bad.sum(), day_str)
        out = out[~bad].copy()

    out["trade_id"] = out["trade_id"].astype("int64")
    out = out[OUTPUT_SCHEMA_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    return out


def fetch_day_dump(symbol: str, day_str: str) -> pd.DataFrame | None:
    """Fetch one day via Binance Vision dump. Returns None if unavailable."""
    url = BINANCE_VISION_URL.format(symbol=symbol, date=day_str)
    logger.debug("Fetching dump: %s", url)
    raw = _fetch_with_retry(url)
    if raw is None:
        return None
    return _parse_dump_csv(raw, day_str)


# ---------------------------------------------------------------------------
# Source: Binance aggTrades REST API
# ---------------------------------------------------------------------------


def _fetch_agg_trades_page(
    symbol: str,
    *,
    start_time_ms: int,
    end_time_ms: int,
    limit: int = API_PAGE_SIZE,
) -> list[dict] | None:
    """Fetch one page of aggTrades; return list of records or None on failure."""
    params = (
        f"symbol={symbol}"
        f"&startTime={start_time_ms}"
        f"&endTime={end_time_ms}"
        f"&limit={limit}"
    )
    url = f"{BINANCE_API_URL}?{params}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _fetch_url_bytes(url, timeout=30)
            import json
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                logger.warning(
                    "Rate-limited (429) on attempt %d/%d — sleeping %.1fs",
                    attempt, MAX_RETRIES, API_BACKOFF_429_SECONDS,
                )
                time.sleep(API_BACKOFF_429_SECONDS)
            elif exc.code == 418:
                # Binance IP ban
                logger.error("IP banned (418) — stopping API collection")
                return None
            else:
                wait = RETRY_BASE_SECONDS ** attempt
                logger.warning("HTTP %s on attempt %d — retrying in %.1fs", exc.code, attempt, wait)
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
        except (urllib.error.URLError, OSError) as exc:
            wait = RETRY_BASE_SECONDS ** attempt
            logger.warning("Network error attempt %d: %s — retrying in %.1fs", attempt, exc, wait)
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    logger.error("All retries failed for %s", url)
    return None


def fetch_day_api(symbol: str, day_str: str) -> pd.DataFrame | None:
    """Fetch one full day via Binance aggTrades REST endpoint (pagination)."""
    day = date.fromisoformat(day_str)
    day_start_ms = int(pd.Timestamp(day_str, tz="UTC").timestamp() * 1000)
    day_end_ms = int(
        pd.Timestamp(day_str + " 23:59:59.999", tz="UTC").timestamp() * 1000
    )

    records: list[dict] = []
    cursor_ms = day_start_ms
    page_count = 0

    while cursor_ms <= day_end_ms:
        page = _fetch_agg_trades_page(
            symbol,
            start_time_ms=cursor_ms,
            end_time_ms=day_end_ms,
        )
        if page is None:
            logger.error("API fetch failed at cursor %d for %s", cursor_ms, day_str)
            return None
        if not page:
            break  # no more data in this window

        records.extend(page)
        last_trade_time = int(page[-1]["T"])
        if last_trade_time >= day_end_ms:
            break
        cursor_ms = last_trade_time + 1
        page_count += 1
        time.sleep(API_SLEEP_SECONDS)

        if page_count % 100 == 0:
            logger.info("  API page %d — %d trades so far for %s", page_count, len(records), day_str)

    if not records:
        logger.warning("API returned 0 trades for %s", day_str)
        return None

    # aggTrades schema: a=aggId, p=price, q=qty, f=firstId, l=lastId, T=time, m=isBuyerMaker
    df = pd.DataFrame(records)
    out = pd.DataFrame()
    out["trade_id"] = df["a"].astype("int64")
    out["price"] = pd.to_numeric(df["p"], errors="coerce").astype("float64")
    out["quantity"] = pd.to_numeric(df["q"], errors="coerce").astype("float64")
    time_ms = df["T"].astype("int64")
    out["timestamp"] = (time_ms * 1_000_000).astype("int64")  # ms → ns
    # m=True means buyer is maker → aggressive side is seller
    out["side"] = df["m"].map({True: "sell", False: "buy"}).astype(str)
    out["day"] = day_str

    bad = out["price"].isna() | out["quantity"].isna()
    if bad.any():
        logger.warning("Dropped %d invalid API rows for %s", bad.sum(), day_str)
        out = out[~bad].copy()

    out = out[OUTPUT_SCHEMA_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Top-level fetch dispatcher
# ---------------------------------------------------------------------------


class FetchResult(NamedTuple):
    day_str: str
    rows: int
    source_used: str  # "dump", "api", "skip", "failed"
    elapsed_seconds: float


def collect_day(
    symbol: str,
    day_str: str,
    output_path: Path,
    *,
    source: str,  # "dump", "api", "auto"
) -> FetchResult:
    """Fetch one day, save parquet, return result summary."""
    if output_path.exists():
        logger.info("[%s] Already exists — skipping: %s", day_str, output_path)
        return FetchResult(day_str=day_str, rows=-1, source_used="skip", elapsed_seconds=0.0)

    t0 = time.perf_counter()
    df: pd.DataFrame | None = None
    used = "failed"

    if source in ("dump", "auto"):
        df = fetch_day_dump(symbol, day_str)
        if df is not None:
            used = "dump"
        elif source == "auto":
            logger.info("[%s] Dump unavailable — falling back to API", day_str)

    if df is None and source in ("api", "auto"):
        df = fetch_day_api(symbol, day_str)
        if df is not None:
            used = "api"

    elapsed = time.perf_counter() - t0

    if df is None or df.empty:
        logger.error("[%s] FAILED to fetch data (source=%s, elapsed=%.1fs)", day_str, source, elapsed)
        return FetchResult(day_str=day_str, rows=0, source_used="failed", elapsed_seconds=elapsed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    rows = len(df)
    logger.info(
        "[%s] ✓  %s  %d trades  source=%s  %.1fs",
        day_str, output_path.name, rows, used, elapsed,
    )
    return FetchResult(day_str=day_str, rows=rows, source_used=used, elapsed_seconds=elapsed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _date_range(start: date, end: date) -> list[date]:
    """Return [start, ..., end] inclusive."""
    if end < start:
        raise ValueError(f"end-date {end} must be >= start-date {start}")
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol (default: BTCUSDT)")

    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--days",
        type=int,
        metavar="N",
        help="Collect the last N calendar days (counting backwards from yesterday UTC)",
    )
    date_group.add_argument(
        "--start-date",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="First date to collect (inclusive). Requires --end-date.",
    )

    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        default=None,
        help="Last date to collect (inclusive, default: yesterday UTC)",
    )
    parser.add_argument(
        "--source",
        choices=["dump", "api", "auto"],
        default="auto",
        help="Data source (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "assets",
        help="Root output directory (default: data/assets/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    symbol = args.symbol.upper().strip()
    yesterday = date.today() - timedelta(days=1)

    # Resolve date range
    if args.days is not None:
        end = args.end_date or yesterday
        start = end - timedelta(days=args.days - 1)
    elif args.start_date is not None:
        start = args.start_date
        end = args.end_date or yesterday
    else:
        parser = argparse.ArgumentParser()
        parser.error("Specify --days N or --start-date YYYY-MM-DD")
        return  # unreachable but keeps type checker happy

    days = _date_range(start, end)
    logger.info(
        "Collecting %d day(s) for %s: %s → %s  (source=%s)",
        len(days), symbol, start.isoformat(), end.isoformat(), args.source,
    )

    symbol_lower = symbol.lower()
    output_root: Path = args.output_dir / symbol_lower

    results: list[FetchResult] = []
    for day in days:
        day_str = day.isoformat()
        output_path = output_root / f"{day_str}.parquet"
        result = collect_day(symbol, day_str, output_path, source=args.source)
        results.append(result)

    # Summary
    successes = [r for r in results if r.source_used in ("dump", "api")]
    skipped = [r for r in results if r.source_used == "skip"]
    failed = [r for r in results if r.source_used == "failed"]

    total_rows = sum(r.rows for r in successes)
    total_elapsed = sum(r.elapsed_seconds for r in results)

    logger.info("=" * 60)
    logger.info(
        "Done: %d collected, %d skipped (already existed), %d failed",
        len(successes), len(skipped), len(failed),
    )
    logger.info("Total trades collected this run: %d", total_rows)
    logger.info("Total elapsed: %.1fs", total_elapsed)
    if failed:
        logger.warning("Failed days: %s", [r.day_str for r in failed])

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

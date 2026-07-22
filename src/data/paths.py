"""Canonical project paths for market data and per-asset reports."""

from __future__ import annotations

from pathlib import Path

from src.data.common import normalize_symbol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "assets"
REPORTS_ROOT = PROJECT_ROOT / "reports" / "assets"
DATA_KINDS = {"raw", "processed", "features"}
REPORT_KINDS = {"data_quality", "feature_metadata", "feature_stats"}


def asset_key(symbol: str) -> str:
    """Return the lowercase directory key, for example ``btcusdt``."""
    return normalize_symbol(symbol).lower()


def asset_data_dir(symbol: str, kind: str, *, create: bool = False) -> Path:
    """Return ``data/assets/<symbol>/<kind>``."""
    if kind not in DATA_KINDS:
        raise ValueError(f"kind must be one of {sorted(DATA_KINDS)}")
    path = DATA_ROOT / asset_key(symbol) / kind
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def asset_report_dir(symbol: str, kind: str, *, create: bool = False) -> Path:
    """Return ``reports/assets/<symbol>/<kind>``."""
    if kind not in REPORT_KINDS:
        raise ValueError(f"kind must be one of {sorted(REPORT_KINDS)}")
    path = REPORTS_ROOT / asset_key(symbol) / kind
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

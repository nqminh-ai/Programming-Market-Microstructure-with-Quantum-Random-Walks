"""Clean raw tick trades and calculate normalized price changes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .common import coerce_tick_frame, ensure_parent, validate_tick_frame


class TickProcessor:
    """Deduplicate, filter anomalies, segment gaps, and normalize tick data."""

    def __init__(
        self,
        *,
        rolling_window: int = 100,
        outlier_z_score: float = 3.0,
        gap_threshold_seconds: float = 300.0,
    ) -> None:
        if rolling_window < 3:
            raise ValueError("rolling_window must be at least 3")
        if outlier_z_score <= 0:
            raise ValueError("outlier_z_score must be positive")
        if gap_threshold_seconds <= 0:
            raise ValueError("gap_threshold_seconds must be positive")
        self.rolling_window = rolling_window
        self.outlier_z_score = outlier_z_score
        self.gap_threshold_ns = int(gap_threshold_seconds * 1_000_000_000)

    def process(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return cleaned ticks and a serializable data-quality report."""
        original_count = len(frame)
        ticks = coerce_tick_frame(frame)

        invalid_mask = (
            ~np.isfinite(ticks["price"])
            | ~np.isfinite(ticks["quantity"])
            | (ticks["price"] <= 0)
            | (ticks["quantity"] <= 0)
        )
        invalid_count = int(invalid_mask.sum())
        ticks = ticks.loc[~invalid_mask].copy()

        duplicate_count = int(ticks["trade_id"].duplicated(keep="first").sum())
        ticks = ticks.drop_duplicates("trade_id", keep="first")
        ticks = ticks.sort_values(["timestamp", "trade_id"], kind="stable")
        ticks = ticks.reset_index(drop=True)

        minimum_periods = min(self.rolling_window, max(3, self.rolling_window // 5))
        preliminary_segments = (
            ticks["timestamp"].diff().gt(self.gap_threshold_ns).fillna(False).cumsum()
        )
        grouped_prices = ticks["price"].groupby(preliminary_segments, sort=False)
        reference_mean = grouped_prices.transform(
            lambda series: series.rolling(
                self.rolling_window,
                min_periods=minimum_periods,
            )
            .mean()
            .shift(1)
        )
        reference_std = grouped_prices.transform(
            lambda series: series.rolling(
                self.rolling_window,
                min_periods=minimum_periods,
            )
            .std(ddof=0)
            .shift(1)
        )
        outlier_candidate_mask = (
            reference_mean.notna()
            & reference_std.gt(0)
            & ticks["price"].sub(reference_mean).abs().gt(
                self.outlier_z_score * reference_std
            )
        )
        threshold = self.outlier_z_score * reference_std
        previous_price = ticks["price"].shift(1)
        previous_tick_same_segment = preliminary_segments.shift(1).eq(
            preliminary_segments
        )
        current_deviation = ticks["price"] - reference_mean
        previous_deviation = previous_price - reference_mean
        confirmed_regime = (
            outlier_candidate_mask
            & previous_tick_same_segment
            & previous_deviation.abs().gt(threshold)
            & np.sign(current_deviation).eq(np.sign(previous_deviation))
            & ticks["price"].sub(previous_price).abs().le(threshold)
        )
        outlier_mask = outlier_candidate_mask & ~confirmed_regime
        outlier_candidate_count = int(outlier_candidate_mask.sum())
        outlier_count = int(outlier_mask.sum())
        outlier_trade_id_sample = (
            ticks.loc[outlier_mask, "trade_id"].head(100).astype(int).tolist()
        )
        ticks = ticks.loc[~outlier_mask].copy().reset_index(drop=True)

        time_difference = ticks["timestamp"].diff()
        ticks["is_gap_start"] = time_difference.gt(self.gap_threshold_ns).fillna(False)
        ticks["segment_id"] = ticks["is_gap_start"].cumsum().astype("int32")
        gap_timestamps = (
            ticks.loc[ticks["is_gap_start"], "timestamp"].astype(int).tolist()
        )

        grouped = ticks.groupby("segment_id", sort=False)["price"]
        ticks["price_increment"] = grouped.diff().fillna(0.0).astype("float64")
        ticks["log_return"] = (
            np.log(ticks["price"]).groupby(ticks["segment_id"]).diff().fillna(0.0)
        )

        if not np.isfinite(ticks[["price_increment", "log_return"]]).all().all():
            raise ValueError("processing produced non-finite returns")
        validate_tick_frame(ticks)

        removed_count = original_count - len(ticks)
        report: dict[str, Any] = {
            "input_records": original_count,
            "output_records": len(ticks),
            "removed_records": removed_count,
            "removed_fraction": removed_count / original_count if original_count else 0.0,
            "invalid_records": invalid_count,
            "duplicate_trade_ids_removed": duplicate_count,
            "price_outlier_candidates": outlier_candidate_count,
            "price_outliers_removed": outlier_count,
            "outlier_trade_id_sample": outlier_trade_id_sample,
            "gap_count": len(gap_timestamps),
            "gap_start_timestamps_ns": gap_timestamps,
            "rolling_window": self.rolling_window,
            "rolling_window_alignment": (
                "trailing_with_past_only_regime_confirmation"
            ),
            "outlier_z_score": self.outlier_z_score,
            "gap_threshold_ns": self.gap_threshold_ns,
        }
        return ticks, report

    def process_file(
        self,
        source: str | Path,
        output_path: str | Path,
        report_path: str | Path,
    ) -> tuple[Path, Path]:
        """Process a CSV/Parquet source and persist Parquet plus quality report."""
        source_path = Path(source)
        if source_path.suffix == ".parquet":
            raw = pd.read_parquet(source_path)
        else:
            raw = pd.read_csv(source_path)
        processed, report = self.process(raw)

        output = ensure_parent(output_path)
        processed.to_parquet(output, index=False)
        report_output = self.write_report(report, report_path)
        logger.info("Processed {:,} ticks into {}", len(processed), output)
        return output, report_output

    @staticmethod
    def write_report(report: dict[str, Any], path: str | Path) -> Path:
        """Write a human-readable JSON data-quality report."""
        output = ensure_parent(path)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output

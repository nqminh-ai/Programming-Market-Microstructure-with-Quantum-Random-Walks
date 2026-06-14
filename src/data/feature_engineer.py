"""Create QRW-oriented features from processed tick and LOB data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr

from .common import ensure_parent, timestamps_to_nanoseconds
from .orderbook_collector import OrderBookCollector


class FeatureEngineer:
    """Engineer directional, intensity, imbalance, and autocorrelation features."""

    BASE_OUTPUT_COLUMNS = [
        "timestamp",
        "price",
        "quantity",
        "side",
        "trade_id",
        "log_return",
        "price_increment",
        "segment_id",
        "tick_direction",
        "trade_intensity",
        "obi",
        "obi_valid",
        "mid_price",
        "vwmp",
        "price_mid_deviation",
    ]

    def __init__(
        self,
        *,
        autocorrelation_lags: int = 20,
        lob_tolerance_seconds: float = 5.0,
        minimum_lob_match_fraction: float = 0.95,
        trade_imbalance_window: int = 100,
        trade_imbalance_min_periods: int | None = None,
    ) -> None:
        if autocorrelation_lags < 1:
            raise ValueError("autocorrelation_lags must be positive")
        if lob_tolerance_seconds < 0:
            raise ValueError("lob_tolerance_seconds cannot be negative")
        if not 0.0 <= minimum_lob_match_fraction <= 1.0:
            raise ValueError("minimum_lob_match_fraction must be between 0 and 1")
        if trade_imbalance_window < 2:
            raise ValueError("trade_imbalance_window must be at least 2")
        if trade_imbalance_min_periods is None:
            trade_imbalance_min_periods = max(3, trade_imbalance_window // 5)
        if not 1 <= trade_imbalance_min_periods <= trade_imbalance_window:
            raise ValueError(
                "trade_imbalance_min_periods must be between 1 and "
                "trade_imbalance_window"
            )
        self.autocorrelation_lags = autocorrelation_lags
        self.lob_tolerance_ns = int(lob_tolerance_seconds * 1_000_000_000)
        self.minimum_lob_match_fraction = minimum_lob_match_fraction
        self.trade_imbalance_window = trade_imbalance_window
        self.trade_imbalance_min_periods = trade_imbalance_min_periods

    def engineer(
        self,
        ticks: pd.DataFrame,
        lob: pd.DataFrame | str | Path | None = None,
        *,
        require_lob: bool = False,
        obi_source: str = "auto",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return a finite feature matrix and feature metadata."""
        required = {
            "timestamp",
            "price",
            "quantity",
            "side",
            "trade_id",
            "log_return",
            "price_increment",
            "segment_id",
        }
        missing = sorted(required.difference(ticks.columns))
        if missing:
            raise ValueError(f"processed ticks are missing columns: {missing}")

        features = ticks.copy()
        features["timestamp"] = timestamps_to_nanoseconds(features["timestamp"]).to_numpy()
        if not features["timestamp"].is_monotonic_increasing:
            features = features.sort_values("timestamp", kind="stable")
        features = features.reset_index(drop=True)

        raw_direction = np.sign(features["price_increment"]).replace(0, np.nan)
        previous_direction = raw_direction.groupby(
            features["segment_id"],
            sort=False,
        ).ffill()
        side_direction = pd.Series(
            np.where(features["side"].eq("buy"), 1, -1),
            index=features.index,
        )
        features["tick_direction"] = previous_direction.fillna(side_direction).astype(
            "int8"
        )
        event_seconds = features["timestamp"] // 1_000_000_000
        features["trade_intensity"] = (
            features.groupby(event_seconds, sort=False)
            .cumcount()
            .add(1)
            .astype("int32")
        )

        valid_sources = {"auto", "lob", "trade_imbalance"}
        if obi_source not in valid_sources:
            raise ValueError(
                f"obi_source must be one of {sorted(valid_sources)}"
            )
        if require_lob and obi_source == "trade_imbalance":
            raise ValueError("require_lob cannot be combined with synthetic OBI")

        lob_frame = None if obi_source == "trade_imbalance" else self._load_lob(lob)
        if obi_source == "lob" and lob_frame is None:
            raise ValueError("LOB data is required when obi_source='lob'")
        lob_matched_rows = 0
        lob_match_fraction = 0.0
        if lob_frame is not None:
            features = self._merge_lob(features, lob_frame)
            lob_matches = features[["obi", "mid_price"]].notna().all(axis=1)
            features["obi_valid"] = lob_matches
            lob_matched_rows = int(lob_matches.sum())
            lob_match_fraction = (
                lob_matched_rows / len(features) if len(features) else 0.0
            )
            if lob_matched_rows == 0:
                raise ValueError(
                    "LOB data has no timestamp overlap with the processed ticks"
                )
            if require_lob and lob_match_fraction < self.minimum_lob_match_fraction:
                raise ValueError(
                    "LOB match coverage "
                    f"{lob_match_fraction:.2%} is below the required "
                    f"{self.minimum_lob_match_fraction:.2%}"
                )
        else:
            if require_lob:
                raise ValueError("LOB data is required for production feature matrices")
            features = self._compute_synthetic_lob(features)

        features["obi"] = features["obi"].fillna(0.0).clip(-1.0, 1.0)
        features["mid_price"] = features["mid_price"].fillna(features["price"])
        features["vwmp"] = features["vwmp"].fillna(features["mid_price"])
        features["price_mid_deviation"] = features["price"] - features["mid_price"]

        autocorrelations: dict[str, float] = {}
        direction = features["tick_direction"].astype("float64")
        segments = features["segment_id"]
        for lag in range(1, self.autocorrelation_lags + 1):
            value = self._safe_autocorrelation(direction, lag, segments)
            autocorrelations[str(lag)] = value

        rho_1_p_value = self._lag_one_p_value(direction, segments)
        numeric = features.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("feature engineering produced NaN or infinite values")

        metadata: dict[str, Any] = {
            "rows": len(features),
            "columns": features.columns.tolist(),
            "autocorrelation": autocorrelations,
            "rho_1": autocorrelations["1"],
            "rho_1_p_value": rho_1_p_value,
            "lob_attached": lob_frame is not None,
            "lob_source": "real" if lob_frame is not None else "synthetic_trade_imbalance",
            "obi_source": (
                "lob_order_book"
                if lob_frame is not None
                else "trade_volume_imbalance"
            ),
            "obi_is_proxy": lob_frame is None,
            "obi_causal": True,
            "obi_valid_rows": int(features["obi_valid"].sum()),
            "obi_valid_fraction": float(features["obi_valid"].mean()),
            "obi_formula": (
                "(top_bid_qty-top_ask_qty)/(top_bid_qty+top_ask_qty)"
                if lob_frame is not None
                else "(rolling_buy_volume-rolling_sell_volume)/"
                "(rolling_buy_volume+rolling_sell_volume)"
            ),
            "obi_window_trades": (
                None if lob_frame is not None else self.trade_imbalance_window
            ),
            "obi_min_periods": (
                None if lob_frame is not None else self.trade_imbalance_min_periods
            ),
            "obi_lag_trades": 0 if lob_frame is not None else 1,
            "trade_intensity_causal": True,
            "trade_intensity_definition": (
                "number_of_observed_trades_so_far_in_current_epoch_second"
            ),
            "lob_matched_rows": lob_matched_rows,
            "lob_match_fraction": lob_match_fraction,
            "minimum_lob_match_fraction": (
                self.minimum_lob_match_fraction if require_lob else None
            ),
        }
        return features, metadata

    def save(
        self,
        features: pd.DataFrame,
        metadata: dict[str, Any],
        feature_path: str | Path,
        stats_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> tuple[Path, Path, Path | None]:
        """Persist features, descriptive statistics, and optional metadata."""
        feature_output = ensure_parent(feature_path)
        features.to_parquet(feature_output, index=False)

        stats = self.describe(features)
        stats_output = ensure_parent(stats_path)
        stats.to_csv(stats_output, index=False)

        metadata_output: Path | None = None
        if metadata_path is not None:
            metadata_output = ensure_parent(metadata_path)
            metadata_output.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        logger.info("Saved feature matrix with {:,} rows to {}", len(features), feature_output)
        return feature_output, stats_output, metadata_output

    def engineer_files(
        self,
        processed_path: str | Path,
        feature_path: str | Path,
        stats_path: str | Path,
        *,
        lob_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        require_lob: bool = False,
        obi_source: str = "auto",
    ) -> tuple[Path, Path, Path | None]:
        """Load Phase 2 artifacts, engineer features, and persist outputs."""
        ticks = pd.read_parquet(processed_path)
        features, metadata = self.engineer(
            ticks,
            lob=lob_path,
            require_lob=require_lob,
            obi_source=obi_source,
        )
        return self.save(
            features,
            metadata,
            feature_path,
            stats_path,
            metadata_path,
        )

    @staticmethod
    def describe(features: pd.DataFrame) -> pd.DataFrame:
        """Return mean/std/min/max/skewness/kurtosis for numeric features."""
        rows: list[dict[str, float | str]] = []
        for column in features.select_dtypes(include=[np.number]).columns:
            series = features[column].astype("float64")
            rows.append(
                {
                    "feature": column,
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "skewness": float(series.skew()) if series.nunique() > 1 else 0.0,
                    "kurtosis": float(series.kurt()) if series.nunique() > 1 else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def _load_lob(
        self,
        lob: pd.DataFrame | str | Path | None,
    ) -> pd.DataFrame | None:
        if lob is None:
            return None
        if isinstance(lob, pd.DataFrame):
            frame = lob.copy()
        else:
            path = Path(lob)
            frame = (
                OrderBookCollector.snapshots_to_frame(path)
                if path.suffix.lower() in {".h5", ".hdf5"}
                else pd.read_parquet(path)
            )
        if frame.empty:
            raise ValueError("LOB data is empty")
        frame["timestamp"] = timestamps_to_nanoseconds(frame["timestamp"]).to_numpy()
        return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)

    def _compute_synthetic_lob(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute synthetic OBI from rolling trade volume imbalance.

        When real LOB data is unavailable, this constructs a proxy order-book
        imbalance from the directional trade flow.  The formula mirrors the
        real OBI definition:

            synthetic_obi_t = (buy_vol_t - sell_vol_t) / (buy_vol_t + sell_vol_t)

        computed over a **trailing** (causal) rolling window of
        ``trade_imbalance_window``
        trades, shifted by one position so that the current trade is excluded
        from its own feature (preventing trivial self-prediction).

        The synthetic mid-price is a trailing volume-weighted average price
        (VWAP) over the same window.
        """
        obi_window = self.trade_imbalance_window
        min_periods = self.trade_imbalance_min_periods
        is_buy = features["side"].eq("buy")
        qty = features["quantity"].astype("float64")

        signed_volume = qty * np.where(is_buy, 1.0, -1.0)

        segments = features["segment_id"]
        rolling_signed = signed_volume.groupby(segments, sort=False).transform(
            lambda s: s.rolling(obi_window, min_periods=min_periods).sum().shift(1)
        )
        rolling_total = qty.groupby(segments, sort=False).transform(
            lambda s: s.rolling(obi_window, min_periods=min_periods).sum().shift(1)
        )
        valid = rolling_total.notna() & rolling_total.gt(0.0)
        raw_obi = np.where(
            valid,
            rolling_signed / rolling_total,
            0.0,
        )
        features["obi"] = np.clip(
            np.nan_to_num(raw_obi, nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )
        features["obi_valid"] = valid.to_numpy(dtype=bool)

        # Synthetic VWAP as mid-price proxy
        price_x_qty = features["price"] * qty
        grouped_pxq = price_x_qty.groupby(segments, sort=False)

        rolling_pxq = grouped_pxq.transform(
            lambda s: s.rolling(obi_window, min_periods=min_periods).sum().shift(1)
        )

        vwap = np.where(
            rolling_total > 0,
            rolling_pxq / rolling_total,
            features["price"].to_numpy(),
        )
        vwap = np.nan_to_num(vwap, nan=features["price"].to_numpy())

        features["mid_price"] = vwap
        features["vwmp"] = vwap

        logger.info(
            "Computed synthetic OBI from trade imbalance: "
            "window={}, OBI range=[{:.4f}, {:.4f}], OBI std={:.4f}",
            obi_window,
            float(features["obi"].min()),
            float(features["obi"].max()),
            float(features["obi"].std()),
        )
        return features

    def _merge_lob(self, ticks: pd.DataFrame, lob: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "obi", "mid_price"}
        missing = sorted(required.difference(lob.columns))
        if missing:
            raise ValueError(f"LOB data is missing columns: {missing}")

        lob = lob.copy()
        if "vwmp" not in lob:
            if {"best_bid", "best_ask", "bid_quantity_top", "ask_quantity_top"}.issubset(
                lob.columns
            ):
                total = lob["bid_quantity_top"] + lob["ask_quantity_top"]
                lob["vwmp"] = np.where(
                    total > 0,
                    (
                        lob["best_ask"] * lob["bid_quantity_top"]
                        + lob["best_bid"] * lob["ask_quantity_top"]
                    )
                    / total,
                    lob["mid_price"],
                )
            else:
                lob["vwmp"] = lob["mid_price"]

        columns = ["timestamp", "obi", "mid_price", "vwmp"]
        return pd.merge_asof(
            ticks.sort_values("timestamp"),
            lob[columns].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=self.lob_tolerance_ns,
        )

    @staticmethod
    def _safe_autocorrelation(
        series: pd.Series,
        lag: int,
        segments: pd.Series | None = None,
    ) -> float:
        values = series.to_numpy(dtype=np.float64, copy=False)
        if len(values) <= lag + 2:
            return 0.0
        current = values[lag:]
        previous = values[:-lag]
        if segments is not None:
            segment_values = segments.to_numpy(copy=False)
            same_segment = segment_values[lag:] == segment_values[:-lag]
            current = current[same_segment]
            previous = previous[same_segment]
        if len(current) < 3:
            return 0.0
        if np.ptp(current) == 0 or np.ptp(previous) == 0:
            return 0.0
        value = np.corrcoef(current, previous)[0, 1]
        return float(value) if np.isfinite(value) else 0.0

    @staticmethod
    def _lag_one_p_value(
        series: pd.Series,
        segments: pd.Series | None = None,
    ) -> float:
        values = series.to_numpy(dtype=np.float64, copy=False)
        if len(values) < 4 or np.ptp(values) == 0:
            return 1.0
        current = values[1:]
        previous = values[:-1]
        if segments is not None:
            segment_values = segments.to_numpy(copy=False)
            same_segment = segment_values[1:] == segment_values[:-1]
            current = current[same_segment]
            previous = previous[same_segment]
        if len(current) < 3 or np.ptp(current) == 0 or np.ptp(previous) == 0:
            return 1.0
        result = pearsonr(current, previous)
        return float(result.pvalue)

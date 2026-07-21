"""Acceptance tests for Track A Phase 9 (anomaly detector)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly_detector import QRWAnomalyDetector


def tick_frame(size: int = 500, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.0003, size)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-12", periods=size, freq="s"),
            "price": 100 * np.exp(np.cumsum(returns)),
            "obi": np.clip(rng.normal(0.0, 0.18, size), -1, 1),
        }
    )


def fitted_detector() -> QRWAnomalyDetector:
    return QRWAnomalyDetector(window=40).fit_baseline(tick_frame(400))


def test_fit_baseline_runs():
    detector = fitted_detector()
    assert detector.baseline is not None
    assert detector.baseline.sum() == pytest.approx(1.0)


def test_kl_divergence_non_negative():
    detector = fitted_detector()
    assert detector.compute_anomaly_score(detector.baseline.copy()) >= 0.0


def test_normal_data_has_lower_score_than_anomaly():
    detector = fitted_detector()
    normal_score = detector.compute_anomaly_score(detector.baseline.copy())
    anomaly = np.full(detector.n_bins, 1e-6)
    anomaly[-1] = 1.0
    anomaly /= anomaly.sum()
    assert detector.compute_anomaly_score(anomaly) > normal_score


def test_anomalous_data_has_high_sigma_score():
    detector = fitted_detector()
    anomaly = np.full(detector.n_bins, 1e-6)
    anomaly[0] = 1.0
    raw = detector.compute_anomaly_score(anomaly)
    assert detector.compute_sigma_score(raw) > 3.0


def test_classify_type_valid_output():
    detector = fitted_detector()
    # Mass concentrated entirely on the left half (indices 0-14) versus a
    # near-empty right half (indices 16-30) is the directional_pressure case:
    # |p_right - p_left| far exceeds the 0.35 threshold.
    result = detector.classify_anomaly_type(np.r_[np.ones(15), 0.1, np.zeros(15)], 4.0)
    assert result == "directional_pressure"
    assert result in detector.VALID_TYPES


def test_rolling_scan_shape():
    detector = fitted_detector()
    scan = detector.rolling_anomaly_scan(tick_frame(180, seed=4), step=10)
    assert len(scan) == 15
    assert {"timestamp", "raw_score", "sigma_score", "anom_type", "alert"} <= set(scan.columns)


def test_score_above_threshold_alerts():
    baseline = tick_frame(300)
    detector = QRWAnomalyDetector(window=40).fit_baseline(baseline)
    anomalous = tick_frame(140, seed=5)
    anomalous.loc[60:, "obi"] = 0.99
    scan = detector.rolling_anomaly_scan(anomalous, step=5)
    assert scan["alert"].any()


def test_baseline_requires_fit_first():
    detector = QRWAnomalyDetector(window=40)
    with pytest.raises(RuntimeError):
        detector.compute_anomaly_score(np.ones(31) / 31)

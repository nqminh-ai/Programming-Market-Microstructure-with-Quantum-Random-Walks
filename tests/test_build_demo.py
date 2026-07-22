"""Tests for the Track A demo-artifact quality gate (audit finding C7).

build_demo.py used to write "status": "PASS" for every module (A1-A5)
unconditionally, based only on whether the output file existed -- never on
whether the numbers inside it were any good. _module_quality_gate() is the
real numeric check that now gates each module's status.
"""

from __future__ import annotations

import math

from scripts.track_a.build_demo import _module_quality_gate


def test_volatility_gate_passes_on_finite_positive_metrics():
    metrics = {
        "n_observations": 100,
        "current_qrw_vol": 0.15,
        "current_garch_vol": 0.14,
        "current_realized_vol": 0.13,
    }
    assert _module_quality_gate("A1_volatility", metrics) is None


def test_volatility_gate_fails_on_zero_observations():
    metrics = {
        "n_observations": 0,
        "current_qrw_vol": 0.15,
        "current_garch_vol": 0.14,
        "current_realized_vol": 0.13,
    }
    assert _module_quality_gate("A1_volatility", metrics) is not None


def test_volatility_gate_fails_on_non_finite_vol():
    metrics = {
        "n_observations": 100,
        "current_qrw_vol": math.nan,
        "current_garch_vol": 0.14,
        "current_realized_vol": 0.13,
    }
    assert _module_quality_gate("A1_volatility", metrics) is not None


def test_risk_gate_passes_on_valid_backtest():
    metrics = {"path_rows": 500, "n_tests": 100, "violation_rate": 0.05}
    assert _module_quality_gate("A2_risk", metrics) is None


def test_risk_gate_fails_on_out_of_range_violation_rate():
    metrics = {"path_rows": 500, "n_tests": 100, "violation_rate": 1.5}
    assert _module_quality_gate("A2_risk", metrics) is not None


def test_signal_gate_fails_on_zero_trades():
    metrics = {
        "hit_rate": 0.5,
        "profit_factor": 1.2,
        "t_stat": 0.3,
        "net_pnl": 0.01,
        "n_trades": 0,
    }
    assert _module_quality_gate("A3_signal", metrics) is not None


def test_signal_gate_passes_with_trades_and_finite_metrics():
    metrics = {
        "hit_rate": 0.5,
        "profit_factor": 1.2,
        "t_stat": 0.3,
        "net_pnl": 0.01,
        "n_trades": 4,
    }
    assert _module_quality_gate("A3_signal", metrics) is None


def test_optimizer_gate_fails_when_t_stat_and_hit_rate_both_zero():
    metrics = {"t_stat": 0.0, "hit_rate": 0.0, "surface_rows": 100}
    assert _module_quality_gate("A4_optimizer", metrics) is not None


def test_optimizer_gate_passes_on_nonzero_metrics():
    metrics = {"t_stat": 0.4, "hit_rate": 0.55, "surface_rows": 100}
    assert _module_quality_gate("A4_optimizer", metrics) is None


def test_anomaly_gate_fails_on_missing_sigma():
    metrics = {"scan_rows": 50, "latest_sigma": None}
    assert _module_quality_gate("A5_anomaly", metrics) is not None


def test_anomaly_gate_passes_on_valid_metrics():
    metrics = {"scan_rows": 50, "latest_sigma": 1.2}
    assert _module_quality_gate("A5_anomaly", metrics) is None

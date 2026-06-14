"""Numerical validation for the Phase 3 QRW implementation."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.qrw_core import DensityMatrixQRW, QuantumRandomWalk


def test_probability_normalization() -> None:
    walk = QuantumRandomWalk(101)
    for _ in range(50):
        probability = walk.step()
        assert probability.sum() == pytest.approx(1.0, abs=1e-12)


def test_ballistic_spreading() -> None:
    steps = 500
    walk = QuantumRandomWalk(2 * steps + 1)
    walk.run(steps)

    expected_coefficient = 1.0 - 1.0 / np.sqrt(2.0)
    observed_coefficient = walk.variance() / steps**2
    assert observed_coefficient == pytest.approx(expected_coefficient, abs=0.005)


def test_unitarity_preserved() -> None:
    walk = DensityMatrixQRW(41)
    for _ in range(20):
        walk.step_with_decoherence(0.15)
        assert walk.trace() == pytest.approx(1.0, abs=1e-12)
        assert np.max(np.abs(walk.rho - walk.rho.conj().T)) < 1e-12


def test_decoherence_classical_limit() -> None:
    steps = 100
    walk = DensityMatrixQRW(2 * steps + 1)
    walk.run(steps, gamma=np.inf)

    # Full basis dephasing after every Hadamard step gives a symmetric CRW.
    assert walk.variance() / steps == pytest.approx(1.0, abs=0.02)


def test_symmetric_initial_state() -> None:
    walk = QuantumRandomWalk(201)
    walk.run(100)
    probability = walk.get_probability()
    assert np.max(np.abs(probability - probability[::-1])) < 1e-12


def test_grover_coin_localization() -> None:
    walk = QuantumRandomWalk(201, coin="grover")
    variances = []
    for _ in range(100):
        walk.step()
        variances.append(walk.variance())

    assert max(variances) == pytest.approx(1.0, abs=1e-12)
    assert variances[-1] == pytest.approx(0.0, abs=1e-12)

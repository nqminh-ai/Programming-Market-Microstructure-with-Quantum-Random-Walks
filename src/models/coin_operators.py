"""Coin operators and physically valid decoherence channels."""

from __future__ import annotations

import numpy as np


def hadamard_coin() -> np.ndarray:
    """Return the standard two-state Hadamard coin."""
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(
        2.0
    )


def grover_coin() -> np.ndarray:
    """Return the two-state Grover coin, which is the swap operator."""
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def biased_coin(theta: float) -> np.ndarray:
    """Return the real rotation coin from the Phase 3 specification."""
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    cosine = np.cos(theta)
    sine = np.sin(theta)
    return np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.complex128,
    )


def obi_adaptive_coin(
    obi: float,
    alpha: float = 1.0,
    *,
    bias: float = 0.0,
    tick_direction: float = 0.0,
    alpha_direction: float = 0.0,
) -> np.ndarray:
    """Return a phase-adaptive unitary coin with a tanh directional link.

    For the default symmetric initial coin, the coherent one-step expected
    displacement is
    ``tanh(bias + alpha * OBI + alpha_direction * tick_direction)``.
    The phase encoding permits probabilities across the full open interval
    ``(0, 1)`` while preserving unitarity. At zero signal the operator is
    exactly Hadamard.
    """
    if not np.isfinite(obi):
        raise ValueError("obi must be finite")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if not np.isfinite(bias):
        raise ValueError("bias must be finite")
    if not np.isfinite(tick_direction) or not np.isfinite(alpha_direction):
        raise ValueError("tick_direction and alpha_direction must be finite")

    bounded_obi = float(np.clip(obi, -1.0, 1.0))
    bounded_direction = float(np.clip(tick_direction, -1.0, 1.0))
    expected_step = float(
        np.tanh(
            bias
            + alpha * bounded_obi
            + alpha_direction * bounded_direction
        )
    )
    phase = -float(np.arcsin(np.clip(expected_step, -1.0, 1.0)))
    cosine = 1.0 / np.sqrt(2.0)
    sine = cosine
    return np.array(
        [
            [cosine, np.exp(1.0j * phase) * sine],
            [np.exp(-1.0j * phase) * sine, -cosine],
        ],
        dtype=np.complex128,
    )


def obi_expected_step(
    obi: float,
    alpha: float,
    *,
    bias: float = 0.0,
    tick_direction: float = 0.0,
    alpha_direction: float = 0.0,
) -> float:
    """Return the one-step mean displacement for the symmetric initial coin."""
    values = (obi, alpha, bias, tick_direction, alpha_direction)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("coin inputs and sensitivities must be finite")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    return float(
        np.tanh(
            bias
            + alpha * np.clip(obi, -1.0, 1.0)
            + alpha_direction * np.clip(tick_direction, -1.0, 1.0)
        )
    )


def dephasing_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Apply basis dephasing to a density matrix.

    Off-diagonal entries are multiplied by ``exp(-gamma)`` while populations
    are preserved. This is a completely positive trace-preserving channel.
    """
    matrix = np.asarray(rho, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rho must be a square density matrix")
    if gamma < 0.0 or np.isnan(gamma):
        raise ValueError("gamma must be non-negative")

    coherence = 0.0 if np.isposinf(gamma) else float(np.exp(-gamma))
    diagonal = np.diag(matrix).copy()
    dephased = matrix * coherence
    np.fill_diagonal(dephased, diagonal)
    return dephased

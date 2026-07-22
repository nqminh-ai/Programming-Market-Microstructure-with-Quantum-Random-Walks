"""Tests for the genuine heavy-tailed (Lévy) unitary shift.

These guard the exact properties that the retracted classical surrogate could
not provide: the shift must be unitary, must reduce to the ordinary walk at
alpha=1, and must actually produce heavier tails for alpha<1.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.heavy_tail_unitary import (
    LevyUnitaryQRW,
    levy_momentum_phase,
    levy_shift_operator,
)


@pytest.mark.parametrize("alpha", [0.4, 0.7, 1.0, 1.5, 2.0])
def test_levy_shift_is_exactly_unitary(alpha: float) -> None:
    """Unit-modulus momentum eigenvalues make the shift unitary for every alpha."""
    shift = levy_shift_operator(21, alpha)
    assert shift.conj().T @ shift == pytest.approx(np.eye(21), abs=1e-12)
    # Eigenvalues of a circulant are its FFT symbol; all must sit on |z| = 1.
    eigenvalues = np.fft.fft(shift[:, 0])
    assert np.abs(eigenvalues) == pytest.approx(np.ones(21), abs=1e-12)


def test_alpha_one_reproduces_nearest_neighbour_translation() -> None:
    """alpha=1 must recover the ordinary +/-1 shift exactly, not approximately."""
    n = 15
    forward = levy_shift_operator(n, 1.0, direction=1)
    backward = levy_shift_operator(n, 1.0, direction=-1)

    expected_forward = np.zeros((n, n), dtype=np.complex128)
    expected_backward = np.zeros((n, n), dtype=np.complex128)
    for y in range(n):
        expected_forward[(y + 1) % n, y] = 1.0
        expected_backward[(y - 1) % n, y] = 1.0

    assert forward == pytest.approx(expected_forward, abs=1e-12)
    assert backward == pytest.approx(expected_backward, abs=1e-12)
    assert levy_momentum_phase(n, 1.0) == pytest.approx(
        2.0 * np.pi * np.fft.fftfreq(n), abs=1e-12
    )


def test_backward_shift_is_the_inverse_of_forward() -> None:
    forward = levy_shift_operator(19, 0.6, direction=1)
    backward = levy_shift_operator(19, 0.6, direction=-1)
    assert forward @ backward == pytest.approx(np.eye(19), abs=1e-12)


def test_fft_step_matches_explicit_matrix_application() -> None:
    """The O(N log N) path must equal the dense operator it stands in for."""
    n, alpha = 21, 0.5
    walk = LevyUnitaryQRW(n, alpha)
    coin_state = walk.coin @ walk.psi

    forward = levy_shift_operator(n, alpha, direction=1)
    backward = levy_shift_operator(n, alpha, direction=-1)
    expected = np.vstack(
        [forward @ coin_state[0], backward @ coin_state[1]]
    )

    walk.step()
    assert walk.psi == pytest.approx(expected, abs=1e-12)


def test_unitary_evolution_preserves_norm() -> None:
    walk = LevyUnitaryQRW(101, 0.5)
    for _ in range(40):
        walk.step()
        assert walk.norm() == pytest.approx(1.0, abs=1e-12)


def test_smaller_alpha_produces_heavier_tails() -> None:
    """The whole point: alpha<1 must put more mass far from the origin."""
    n, steps, threshold = 401, 30, 60
    heavy = LevyUnitaryQRW(n, 0.4)
    standard = LevyUnitaryQRW(n, 1.0)
    heavy.run(steps)
    standard.run(steps)

    # The ordinary walk cannot exceed its light cone at all.
    assert standard.tail_mass(threshold) == pytest.approx(0.0, abs=1e-12)
    assert heavy.tail_mass(threshold) > 1e-6
    assert heavy.variance() > standard.variance()


def test_hopping_amplitudes_decay_as_power_law_when_alpha_below_one() -> None:
    """A |k|^alpha cusp gives ~|x|^-(1+alpha) hopping, not a delta."""
    n, alpha = 501, 0.5
    column = levy_shift_operator(n, alpha)[:, 0]
    magnitude = np.abs(np.fft.fftshift(column))
    centre = n // 2
    near = magnitude[centre + 5]
    far = magnitude[centre + 50]
    # Power-law decay: still clearly non-zero far away, and decreasing.
    assert far > 0.0
    assert near > far
    ratio = near / far
    # |x|^-(1+alpha) over a 10x distance predicts ~10^1.5 ~= 31; allow a wide
    # band so the test checks the regime, not a brittle constant.
    assert 3.0 < ratio < 300.0


def test_wraparound_mass_flags_an_undersized_lattice() -> None:
    small = LevyUnitaryQRW(41, 0.4)
    small.run(60)
    assert small.wraparound_mass() > 1e-6


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        levy_shift_operator(20, 0.5)  # even lattice
    with pytest.raises(ValueError):
        levy_shift_operator(21, 0.0)  # alpha out of range
    with pytest.raises(ValueError):
        levy_shift_operator(21, 2.5)
    with pytest.raises(ValueError):
        levy_shift_operator(21, 0.5, direction=0)
    with pytest.raises(TypeError):
        LevyUnitaryQRW(21, 0.5).run(1.5)

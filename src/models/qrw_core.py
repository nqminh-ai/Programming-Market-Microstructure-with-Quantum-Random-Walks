"""Statevector and density-matrix simulators for a one-dimensional DTQRW."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .coin_operators import (
    biased_coin,
    dephasing_channel,
    grover_coin,
    hadamard_coin,
)

CoinInput = str | np.ndarray


def _coin_from_input(coin: CoinInput) -> np.ndarray:
    if isinstance(coin, str):
        factories: dict[str, Callable[[], np.ndarray]] = {
            "hadamard": hadamard_coin,
            "grover": grover_coin,
            "swap": grover_coin,
            "biased": lambda: biased_coin(np.pi / 4.0),
        }
        try:
            matrix = factories[coin.lower()]()
        except KeyError as exc:
            supported = ", ".join(sorted(factories))
            raise ValueError(f"unknown coin {coin!r}; choose one of: {supported}") from exc
    else:
        matrix = np.asarray(coin, dtype=np.complex128)

    if matrix.shape != (2, 2):
        raise ValueError("coin matrix must have shape (2, 2)")
    if not np.isfinite(matrix).all():
        raise ValueError("coin matrix must contain only finite values")
    if not np.allclose(
        matrix.conj().T @ matrix,
        np.eye(2),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("coin matrix must be unitary")
    return matrix.copy()


def _initial_coin(initial_coin_state: np.ndarray | None) -> np.ndarray:
    if initial_coin_state is None:
        state = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    else:
        state = np.asarray(initial_coin_state, dtype=np.complex128)
    if state.shape != (2,):
        raise ValueError("initial_coin_state must have shape (2,)")
    if not np.isfinite(state).all():
        raise ValueError("initial_coin_state must contain only finite values")
    norm = float(np.linalg.norm(state))
    if norm == 0.0:
        raise ValueError("initial_coin_state cannot be the zero vector")
    return state / norm


class _BaseQRW:
    """Shared lattice and coin validation for QRW simulators."""

    boundary_tolerance = 1e-14

    def __init__(
        self,
        n_positions: int,
        coin: CoinInput = "hadamard",
        initial_position: int = 0,
        initial_coin_state: np.ndarray | None = None,
    ) -> None:
        if isinstance(n_positions, bool) or not isinstance(n_positions, int):
            raise TypeError("n_positions must be an integer")
        if n_positions < 3 or n_positions % 2 == 0:
            raise ValueError("n_positions must be an odd integer of at least 3")

        self.n_positions = n_positions
        self.positions = np.arange(n_positions, dtype=np.int64) - n_positions // 2
        matches = np.flatnonzero(self.positions == initial_position)
        if len(matches) != 1:
            lower, upper = int(self.positions[0]), int(self.positions[-1])
            raise ValueError(
                f"initial_position must be between {lower} and {upper}"
            )

        self.initial_position = int(initial_position)
        self.initial_index = int(matches[0])
        self.initial_coin_state = _initial_coin(initial_coin_state)
        self.coin_matrix = _coin_from_input(coin)
        self.time = 0

    def set_coin(self, coin: CoinInput) -> None:
        """Replace the coin used by subsequent steps."""
        self.coin_matrix = _coin_from_input(coin)

    def _step_coin(self, coin_matrix: np.ndarray | None) -> np.ndarray:
        return self.coin_matrix if coin_matrix is None else _coin_from_input(coin_matrix)


class QuantumRandomWalk(_BaseQRW):
    """Pure-state coined quantum random walk on a finite, non-wrapping line."""

    def __init__(
        self,
        n_positions: int,
        coin: CoinInput = "hadamard",
        initial_position: int = 0,
        initial_coin_state: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            n_positions,
            coin=coin,
            initial_position=initial_position,
            initial_coin_state=initial_coin_state,
        )
        self.psi = np.zeros((2, n_positions), dtype=np.complex128)
        self.reset()

    def reset(self) -> None:
        """Restore the configured localized initial state."""
        self.psi.fill(0.0)
        self.psi[:, self.initial_index] = self.initial_coin_state
        self.time = 0

    def step(self, coin_matrix: np.ndarray | None = None) -> np.ndarray:
        """Apply one coin-then-shift evolution step."""
        coin = self._step_coin(coin_matrix)
        coin_state = coin @ self.psi
        outgoing_probability = float(
            abs(coin_state[0, -1]) ** 2 + abs(coin_state[1, 0]) ** 2
        )
        if outgoing_probability > self.boundary_tolerance:
            raise RuntimeError(
                "amplitude reached the lattice boundary; increase n_positions"
            )

        shifted = np.zeros_like(coin_state)
        shifted[0, 1:] = coin_state[0, :-1]
        shifted[1, :-1] = coin_state[1, 1:]
        self.psi = shifted
        self.time += 1
        return self.get_probability()

    def get_probability(self) -> np.ndarray:
        """Return the current position marginal."""
        return np.sum(np.abs(self.psi) ** 2, axis=0).real

    def run(self, n_steps: int) -> np.ndarray:
        """Advance by ``n_steps`` and return the final position marginal."""
        if isinstance(n_steps, bool) or not isinstance(n_steps, int):
            raise TypeError("n_steps must be an integer")
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        for _ in range(n_steps):
            self.step()
        return self.get_probability()

    def variance(self) -> float:
        """Return the variance of the current position distribution."""
        probability = self.get_probability()
        mean = float(probability @ self.positions)
        return float(probability @ (self.positions - mean) ** 2)


class DensityMatrixQRW(_BaseQRW):
    """Mixed-state QRW with basis dephasing after each unitary step."""

    def __init__(
        self,
        n_positions: int,
        coin: CoinInput = "hadamard",
        initial_position: int = 0,
        initial_coin_state: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            n_positions,
            coin=coin,
            initial_position=initial_position,
            initial_coin_state=initial_coin_state,
        )
        dimension = 2 * n_positions
        self.rho = np.zeros((dimension, dimension), dtype=np.complex128)
        self.reset()

    def reset(self) -> None:
        """Restore the pure localized initial density matrix."""
        state = np.zeros((2, self.n_positions), dtype=np.complex128)
        state[:, self.initial_index] = self.initial_coin_state
        flat_state = state.reshape(-1)
        self.rho = np.outer(flat_state, flat_state.conj())
        self.time = 0

    def step(self, coin_matrix: np.ndarray | None = None) -> np.ndarray:
        """Apply one unitary evolution step without decoherence."""
        return self.step_with_decoherence(0.0, coin_matrix=coin_matrix)

    def step_with_decoherence(
        self,
        gamma: float,
        coin_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply unitary evolution followed by basis dephasing."""
        coin = self._step_coin(coin_matrix)
        rho4 = self.rho.reshape(2, self.n_positions, 2, self.n_positions)
        coin_rho = np.einsum(
            "ac,cxdy,bd->axby",
            coin,
            rho4,
            coin.conj(),
            optimize=True,
        )

        outgoing_probability = float(
            max(coin_rho[0, -1, 0, -1].real, 0.0)
            + max(coin_rho[1, 0, 1, 0].real, 0.0)
        )
        if outgoing_probability > self.boundary_tolerance:
            raise RuntimeError(
                "probability reached the lattice boundary; increase n_positions"
            )

        shifted = np.zeros_like(coin_rho)
        source_slices = (slice(0, -1), slice(1, None))
        target_slices = (slice(1, None), slice(0, -1))
        for ket_coin in range(2):
            for bra_coin in range(2):
                shifted[
                    ket_coin,
                    target_slices[ket_coin],
                    bra_coin,
                    target_slices[bra_coin],
                ] = coin_rho[
                    ket_coin,
                    source_slices[ket_coin],
                    bra_coin,
                    source_slices[bra_coin],
                ]

        self.rho = dephasing_channel(
            shifted.reshape(2 * self.n_positions, 2 * self.n_positions),
            gamma,
        )
        self.rho = (self.rho + self.rho.conj().T) / 2.0
        self.time += 1
        return self.get_probability()

    def get_probability(self) -> np.ndarray:
        """Return the position marginal from the density-matrix diagonal."""
        populations = np.diag(self.rho).real.reshape(2, self.n_positions)
        probability = populations.sum(axis=0)
        if probability.min(initial=0.0) < -1e-12:
            raise RuntimeError("density matrix produced negative probabilities")
        return np.clip(probability, 0.0, None)

    def run(self, n_steps: int, *, gamma: float = 0.0) -> np.ndarray:
        """Advance by ``n_steps`` with a constant dephasing rate."""
        if isinstance(n_steps, bool) or not isinstance(n_steps, int):
            raise TypeError("n_steps must be an integer")
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        for _ in range(n_steps):
            self.step_with_decoherence(gamma)
        return self.get_probability()

    def trace(self) -> float:
        """Return the real trace, which should remain one."""
        return float(np.trace(self.rho).real)

    def purity(self) -> float:
        """Return ``Tr(rho^2)``."""
        return float(np.einsum("ij,ji->", self.rho, self.rho).real)

    def variance(self) -> float:
        """Return the variance of the current position distribution."""
        probability = self.get_probability()
        mean = float(probability @ self.positions)
        return float(probability @ (self.positions - mean) ** 2)

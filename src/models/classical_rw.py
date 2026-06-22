"""Classical random-walk baselines for market-path comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassicalRandomWalk:
    """Simple, biased, or lag-one correlated random walk."""

    kind: str = "simple"
    p_up: float = 0.5
    p_move: float = 1.0
    rho: float = 0.0
    initial_position: float = 0.0
    step_size: float = 1.0
    random_state: int | np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.kind = self.kind.lower()
        if self.kind not in {"simple", "biased", "correlated"}:
            raise ValueError(
                "kind must be 'simple', 'biased', or 'correlated'"
            )
        if not 0.0 <= self.p_up <= 1.0:
            raise ValueError("p_up must be in [0, 1]")
        if not 0.0 <= self.p_move <= 1.0:
            raise ValueError("p_move must be in [0, 1]")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")
        if not np.isfinite(self.initial_position):
            raise ValueError("initial_position must be finite")
        if not np.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be finite and positive")

    def fit(self, directions: np.ndarray) -> dict[str, float | int | str]:
        """Calibrate direction frequency and persistence from nonzero moves."""
        values = np.asarray(directions, dtype=np.float64).reshape(-1)
        values = np.sign(values[np.isfinite(values)])
        moving = values[values != 0.0]
        if len(moving) == 0:
            raise ValueError("directions must contain at least one nonzero move")

        self.p_move = float(len(moving) / len(values))
        self.p_up = float(np.mean(moving > 0.0))
        if len(moving) > 1:
            persistence = float(np.mean(moving[1:] == moving[:-1]))
            self.rho = float(2.0 * persistence - 1.0)
        else:
            persistence = 0.5
            self.rho = 0.0
        log_probs = self.direction_log_probabilities(directions)
        final_log_loss = float(-np.mean(log_probs)) if len(log_probs) > 0 else float("nan")
        k_params = {"simple": 0, "biased": 1, "correlated": 2}[self.kind.lower()]
        n_obs = len(log_probs)
        
        if n_obs > 0:
            final_nll = final_log_loss * n_obs
            aic = 2 * k_params + 2 * final_nll
            bic = k_params * np.log(n_obs) + 2 * final_nll
        else:
            aic = float("nan")
            bic = float("nan")

        return {
            "model": self.model_name,
            "observations": int(len(values)),
            "moving_observations": int(len(moving)),
            "p_move": self.p_move,
            "p_up": self.p_up,
            "rho": self.rho,
            "p_same": persistence,
            "final_log_loss": final_log_loss,
            "aic": aic,
            "bic": bic,
        }

    @property
    def model_name(self) -> str:
        names = {
            "simple": "CRW Simple",
            "biased": "CRW Biased",
            "correlated": "CRW Correlated",
        }
        return names[self.kind]

    def simulate(
        self,
        n_steps: int,
        n_paths: int,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return paths with shape ``(n_paths, n_steps + 1)``."""
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        if n_paths < 1:
            raise ValueError("n_paths must be positive")
        seed = self.random_state if random_state is None else random_state
        rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )

        moving = rng.random((n_paths, n_steps)) < self.p_move
        if self.kind == "simple":
            directions = np.where(
                rng.random((n_paths, n_steps)) < 0.5,
                1.0,
                -1.0,
            )
            increments = np.where(moving, directions, 0.0)
        elif self.kind == "biased":
            directions = np.where(
                rng.random((n_paths, n_steps)) < self.p_up,
                1.0,
                -1.0,
            )
            increments = np.where(moving, directions, 0.0)
        else:
            increments = np.zeros((n_paths, n_steps), dtype=np.float64)
            last_direction = np.where(
                rng.random(n_paths) < self.p_up,
                1.0,
                -1.0,
            )
            has_moved = np.zeros(n_paths, dtype=bool)
            p_same = 0.5 * (1.0 + self.rho)
            for index in range(n_steps):
                active = moving[:, index]
                continuing = active & has_moved
                keep = rng.random(n_paths) < p_same
                candidate = np.where(
                    keep,
                    last_direction,
                    -last_direction,
                )
                last_direction = np.where(continuing, candidate, last_direction)
                increments[active, index] = last_direction[active]
                has_moved |= active

        paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = self.initial_position
        paths[:, 1:] = (
            self.initial_position
            + self.step_size * np.cumsum(increments, axis=1)
        )
        return paths

    def direction_log_probabilities(
        self,
        directions: np.ndarray,
    ) -> np.ndarray:
        """Return conditional log probabilities for observed nonzero moves."""
        values = np.sign(np.asarray(directions, dtype=np.float64).reshape(-1))
        values = values[np.isfinite(values) & (values != 0.0)]
        if len(values) == 0:
            return np.empty(0, dtype=np.float64)
        epsilon = 1e-12

        if self.kind == "simple":
            probability = np.full(len(values), 0.5)
        elif self.kind == "biased":
            probability = np.where(values > 0.0, self.p_up, 1.0 - self.p_up)
        else:
            probability = np.empty(len(values), dtype=np.float64)
            probability[0] = (
                self.p_up if values[0] > 0.0 else 1.0 - self.p_up
            )
            if len(values) > 1:
                p_same = 0.5 * (1.0 + self.rho)
                probability[1:] = np.where(
                    values[1:] == values[:-1],
                    p_same,
                    1.0 - p_same,
                )
        return np.log(np.clip(probability, epsilon, 1.0))

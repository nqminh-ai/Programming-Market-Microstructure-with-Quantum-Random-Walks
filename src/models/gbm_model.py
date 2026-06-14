"""Geometric Brownian motion baseline."""

from __future__ import annotations

from typing import Any

import numpy as np


class GBMBaseline:
    """Estimate and simulate a constant-drift, constant-volatility GBM."""

    def __init__(
        self,
        *,
        initial_price: float = 1.0,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if not np.isfinite(initial_price) or initial_price <= 0.0:
            raise ValueError("initial_price must be finite and positive")
        self.initial_price = float(initial_price)
        self.random_state = random_state
        self.fitted = False

    def fit(
        self,
        log_returns: np.ndarray,
        *,
        dt: float = 1.0,
    ) -> dict[str, Any]:
        """Estimate drift and volatility by Gaussian MLE."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        values = np.asarray(log_returns, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            raise ValueError("GBM fit requires at least two finite returns")
        mean = float(np.mean(values))
        variance = float(np.mean((values - mean) ** 2))
        self.dt = float(dt)
        self.sigma = float(np.sqrt(max(variance / dt, 1e-24)))
        self.mu = float(mean / dt + 0.5 * self.sigma**2)
        self.fitted = True
        log_probability = self.log_probabilities(values, dt=dt)
        self.log_likelihood_value = float(np.sum(log_probability))
        parameter_count = 2
        self.aic = float(
            2 * parameter_count - 2 * self.log_likelihood_value
        )
        self.bic = float(
            np.log(len(values)) * parameter_count
            - 2 * self.log_likelihood_value
        )
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "log_likelihood": self.log_likelihood_value,
            "aic": self.aic,
            "bic": self.bic,
            "observations": int(len(values)),
            "dt": self.dt,
        }

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("fit must be called before using the model")

    def simulate(
        self,
        n_steps: int,
        n_paths: int,
        *,
        dt: float | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return GBM price paths with the initial value in column zero."""
        self._require_fit()
        if n_steps < 1 or n_paths < 1:
            raise ValueError("n_steps and n_paths must be positive")
        interval = self.dt if dt is None else float(dt)
        if not np.isfinite(interval) or interval <= 0.0:
            raise ValueError("dt must be finite and positive")
        seed = self.random_state if random_state is None else random_state
        rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )
        shocks = rng.standard_normal((n_paths, n_steps))
        log_increment = (
            (self.mu - 0.5 * self.sigma**2) * interval
            + self.sigma * np.sqrt(interval) * shocks
        )
        paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = self.initial_price
        paths[:, 1:] = self.initial_price * np.exp(
            np.cumsum(log_increment, axis=1)
        )
        return paths

    def log_probabilities(
        self,
        log_returns: np.ndarray,
        *,
        dt: float | None = None,
    ) -> np.ndarray:
        """Return Gaussian log densities for observed log returns."""
        self._require_fit()
        interval = self.dt if dt is None else float(dt)
        values = np.asarray(log_returns, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        mean = (self.mu - 0.5 * self.sigma**2) * interval
        variance = max(self.sigma**2 * interval, 1e-24)
        return -0.5 * (
            np.log(2.0 * np.pi * variance)
            + (values - mean) ** 2 / variance
        )

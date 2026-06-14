"""Self-contained Gaussian GARCH(1,1) baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


class GARCHBaseline:
    """Fit and simulate a stationary Gaussian GARCH(1,1) process."""

    def __init__(
        self,
        *,
        initial_price: float = 1.0,
        return_scale: float = 10_000.0,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if not np.isfinite(initial_price) or initial_price <= 0.0:
            raise ValueError("initial_price must be finite and positive")
        if not np.isfinite(return_scale) or return_scale <= 0.0:
            raise ValueError("return_scale must be finite and positive")
        self.initial_price = float(initial_price)
        self.return_scale = float(return_scale)
        self.random_state = random_state
        self.fitted = False
        self.convergence_flag = 1
        self.parameters: dict[str, float | int | str] = {}
        self.log_likelihood_value = float("nan")
        self.aic = float("nan")
        self.bic = float("nan")
        self._last_residual = 0.0
        self._last_variance = 1.0

    @staticmethod
    def _decode(raw: np.ndarray) -> tuple[float, float, float, float]:
        mu = float(raw[0])
        omega = float(np.exp(np.clip(raw[1], -30.0, 30.0)))
        alpha_weight = float(np.exp(np.clip(raw[2], -20.0, 20.0)))
        beta_weight = float(np.exp(np.clip(raw[3], -20.0, 20.0)))
        denominator = 1.0 + alpha_weight + beta_weight
        alpha = alpha_weight / denominator
        beta = beta_weight / denominator
        return mu, omega, alpha, beta

    @classmethod
    def _filter(
        cls,
        returns: np.ndarray,
        raw: np.ndarray,
        *,
        initial_variance: float | None = None,
        initial_residual: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        mu, omega, alpha, beta = cls._decode(raw)
        residual = returns - mu
        variance = np.empty(len(returns), dtype=np.float64)
        unconditional = omega / max(1.0 - alpha - beta, 1e-8)
        if initial_variance is None:
            variance[0] = max(
                float(np.var(returns)),
                unconditional,
                1e-8,
            )
        else:
            variance[0] = (
                omega
                + alpha * float(initial_residual) ** 2
                + beta * max(float(initial_variance), 1e-12)
            )
        for index in range(1, len(returns)):
            variance[index] = (
                omega
                + alpha * residual[index - 1] ** 2
                + beta * variance[index - 1]
            )
        return residual, np.maximum(variance, 1e-12)

    @classmethod
    def _negative_log_likelihood(
        cls,
        raw: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        residual, variance = cls._filter(returns, raw)
        value = 0.5 * np.sum(
            np.log(2.0 * np.pi)
            + np.log(variance)
            + residual**2 / variance
        )
        return float(value) if np.isfinite(value) else 1e100

    def fit(
        self,
        returns: np.ndarray,
        *,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Fit scaled log returns by Gaussian maximum likelihood."""
        values = np.asarray(returns, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)] * self.return_scale
        if len(values) < 20:
            raise ValueError("GARCH fit requires at least 20 finite returns")
        variance = float(np.var(values))
        if variance <= 1e-14:
            raise ValueError("GARCH fit requires non-constant returns")

        alpha0 = 0.08
        beta0 = 0.88
        remainder = 1.0 - alpha0 - beta0
        raw0 = np.array(
            [
                float(np.mean(values)),
                np.log(max(variance * remainder, 1e-8)),
                np.log(alpha0 / remainder),
                np.log(beta0 / remainder),
            ],
            dtype=np.float64,
        )
        result = minimize(
            self._negative_log_likelihood,
            raw0,
            args=(values,),
            method="L-BFGS-B",
            options={"maxiter": 2_000, "ftol": 1e-12},
        )
        if not result.success or not np.isfinite(result.fun):
            retry = minimize(
                self._negative_log_likelihood,
                raw0,
                args=(values,),
                method="Powell",
                options={"maxiter": 4_000, "xtol": 1e-8, "ftol": 1e-10},
            )
            if retry.success and np.isfinite(retry.fun):
                result = retry

        mu, omega, alpha, beta = self._decode(result.x)
        residual, conditional_variance = self._filter(values, result.x)
        self._raw_parameters = np.asarray(result.x, dtype=np.float64)
        self._last_residual = float(residual[-1])
        self._last_variance = float(conditional_variance[-1])
        self.convergence_flag = 0 if result.success else 1
        # The optimizer works on scaled returns; the Jacobian restores the
        # density to the original log-return units.
        self.log_likelihood_value = float(
            -result.fun + len(values) * np.log(self.return_scale)
        )
        parameter_count = 4
        self.aic = float(
            2 * parameter_count - 2 * self.log_likelihood_value
        )
        self.bic = float(
            np.log(len(values)) * parameter_count
            - 2 * self.log_likelihood_value
        )
        self.parameters = {
            "mu": mu / self.return_scale,
            "omega": omega / self.return_scale**2,
            "alpha": alpha,
            "beta": beta,
            "persistence": alpha + beta,
            "log_likelihood": self.log_likelihood_value,
            "aic": self.aic,
            "bic": self.bic,
            "convergence_flag": self.convergence_flag,
            "optimizer_message": str(result.message),
            "observations": int(len(values)),
            "return_scale": self.return_scale,
        }
        self.fitted = True
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(self.parameters, indent=2),
                encoding="utf-8",
            )
        return self.parameters.copy()

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("fit must be called before using the model")

    def simulate(
        self,
        n_steps: int,
        n_paths: int,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate positive price paths from the fitted return process."""
        self._require_fit()
        if n_steps < 1 or n_paths < 1:
            raise ValueError("n_steps and n_paths must be positive")
        seed = self.random_state if random_state is None else random_state
        rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )
        mu, omega, alpha, beta = self._decode(self._raw_parameters)
        residual = np.full(n_paths, self._last_residual, dtype=np.float64)
        variance = np.full(n_paths, self._last_variance, dtype=np.float64)
        scaled_returns = np.empty((n_paths, n_steps), dtype=np.float64)
        for index in range(n_steps):
            variance = omega + alpha * residual**2 + beta * variance
            residual = np.sqrt(np.maximum(variance, 1e-12)) * rng.standard_normal(
                n_paths
            )
            scaled_returns[:, index] = mu + residual
        log_prices = (
            np.log(self.initial_price)
            + np.cumsum(scaled_returns / self.return_scale, axis=1)
        )
        paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = self.initial_price
        paths[:, 1:] = np.exp(log_prices)
        return paths

    def log_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """Score a later return sequence without refitting."""
        self._require_fit()
        values = np.asarray(returns, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)] * self.return_scale
        if len(values) == 0:
            return np.empty(0, dtype=np.float64)
        residual, variance = self._filter(
            values,
            self._raw_parameters,
            initial_variance=self._last_variance,
            initial_residual=self._last_residual,
        )
        return -0.5 * (
            np.log(2.0 * np.pi)
            + np.log(variance)
            + residual**2 / variance
        ) + np.log(self.return_scale)

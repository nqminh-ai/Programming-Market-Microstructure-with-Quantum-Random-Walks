"""QRW-distribution Monte Carlo risk simulation for Track A."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


SCENARIOS: dict[str, dict[str, float | str]] = {
    "normal": {"coin_bias": 0.0, "label": "Normal market", "severity": 1.0},
    "stress": {"coin_bias": 0.25, "label": "Market stress", "severity": 1.5},
    "crisis": {"coin_bias": 0.50, "label": "Flash-crash crisis", "severity": 2.5},
}


class QRWRiskSimulator:
    """Generate cumulative return paths and lower-tail risk measures."""

    def __init__(self, qrw_model=None, n_paths: int = 10_000, seed: int = 42) -> None:
        if int(n_paths) < 1:
            raise ValueError("n_paths must be positive")
        self.qrw = qrw_model
        self.n_paths = int(n_paths)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _probability(state: np.ndarray) -> np.ndarray:
        values = np.asarray(state).reshape(-1)
        if values.size < 3 or not np.all(np.isfinite(values)):
            raise ValueError("state must contain at least three finite values")
        if np.iscomplexobj(values):
            probability = np.abs(values) ** 2
        else:
            probability = values.astype(np.float64, copy=True)
            if np.any(probability < 0):
                raise ValueError("real states must contain non-negative probabilities")
        total = float(probability.sum())
        if total <= 0:
            raise ValueError("state must have positive probability mass")
        return probability / total

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        if not 0.5 < float(confidence) < 1.0:
            raise ValueError("confidence must be between 0.5 and 1")

    def simulate_paths(
        self,
        amplitude_state: np.ndarray,
        horizon: int = 30,
        *,
        return_scale: float = 0.01,
    ) -> np.ndarray:
        """Sample cumulative return paths from a QRW probability state."""
        if int(horizon) < 1 or not np.isfinite(return_scale) or return_scale <= 0:
            raise ValueError("horizon and return_scale must be positive")
        probability = self._probability(amplitude_state)
        lattice_returns = np.linspace(-float(return_scale), float(return_scale), len(probability))
        positions = self.rng.choice(
            len(probability),
            size=(self.n_paths, int(horizon)),
            p=probability,
        )
        increments = lattice_returns[positions]
        return np.cumsum(increments, axis=1)

    def compute_var(self, paths: np.ndarray, confidence: float = 0.95) -> float:
        self._validate_confidence(confidence)
        values = np.asarray(paths, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("paths must have shape (n_paths, horizon)")
        return float(np.quantile(values[:, -1], 1.0 - confidence))

    def compute_es(self, paths: np.ndarray, confidence: float = 0.95) -> float:
        var = self.compute_var(paths, confidence)
        final_returns = np.asarray(paths, dtype=np.float64)[:, -1]
        tail = final_returns[final_returns <= var]
        return float(tail.mean()) if len(tail) else var

    @staticmethod
    def compute_gaussian_var_from_returns(
        returns: np.ndarray,
        confidence: float = 0.95,
        *,
        horizon: int = 1,
    ) -> float:
        QRWRiskSimulator._validate_confidence(confidence)
        values = np.asarray(returns, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) < 2 or int(horizon) < 1:
            raise ValueError("at least two returns and a positive horizon are required")
        mu = float(values.mean()) * horizon
        sigma = float(values.std(ddof=1)) * np.sqrt(horizon)
        return float(mu + norm.ppf(1.0 - confidence) * sigma)

    def compute_gaussian_var(
        self,
        tick_df: pd.DataFrame,
        confidence: float = 0.95,
        *,
        horizon: int = 1,
    ) -> float:
        if "price" not in tick_df.columns:
            raise ValueError("tick_df must contain a price column")
        returns = np.diff(np.log(tick_df["price"].to_numpy(dtype=float)))
        return self.compute_gaussian_var_from_returns(returns, confidence, horizon=horizon)

    def backtest_var(
        self,
        historical_df: pd.DataFrame,
        window: int = 100,
        confidence: float = 0.95,
    ) -> dict:
        """Backtest a causal rolling Gaussian VaR benchmark."""
        self._validate_confidence(confidence)
        if "price" not in historical_df.columns or int(window) < 20:
            raise ValueError("price data and window >= 20 are required")
        prices = historical_df["price"].to_numpy(dtype=float)
        if len(prices) <= window:
            raise ValueError("historical_df must be longer than window")
        returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
        records: list[dict[str, float | bool | int]] = []
        for index in range(window, len(prices)):
            trailing = returns[index - window + 1:index]
            estimate = self.compute_gaussian_var_from_returns(trailing, confidence)
            actual = float(returns[index])
            records.append(
                {
                    "row": index,
                    "var": estimate,
                    "actual": actual,
                    "violation": actual < estimate,
                }
            )
        frame = pd.DataFrame(records)
        frame["rolling_violation_rate"] = frame["violation"].rolling(
            min(100, len(frame)), min_periods=1
        ).mean()
        violations = int(frame["violation"].sum())
        return {
            "violation_rate": float(frame["violation"].mean()),
            "n_violations": violations,
            "n_tests": int(len(frame)),
            "target_rate": float(1.0 - confidence),
            "details": frame,
        }

    def _scenario_probability(self, state: np.ndarray, scenario: str) -> np.ndarray:
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {sorted(SCENARIOS)}")
        base = self._probability(state)
        bias = float(SCENARIOS[scenario]["coin_bias"])
        if bias == 0:
            return base
        positions = np.linspace(-1.0, 1.0, len(base))
        
        # Uncalibrated crisis-tail placement: these constants are hand-picked,
        # not fit from empirical loss data. -1.5 sigma keeps the tail center
        # inside the lattice bounds; 0.15 is an arbitrary fraction of the
        # lattice range for the tail's width.
        empirical_tail_center = -1.5 * float(np.std(positions))
        empirical_tail_width = 0.15  # 15% of lattice range
        
        loss_tail = np.exp(-0.5 * ((positions - empirical_tail_center) / empirical_tail_width) ** 2)
        loss_tail /= loss_tail.sum()
        
        # Mixed distribution: QRW(1-bias) + EmpiricalCrisisTail(bias)
        probability = (1.0 - bias) * base + bias * loss_tail
        return probability / probability.sum()

    def scenario_var(self, amplitude_state: np.ndarray, scenario: str = "normal") -> dict:
        config = SCENARIOS.get(scenario)
        if config is None:
            raise ValueError(f"scenario must be one of {sorted(SCENARIOS)}")
        probability = self._scenario_probability(amplitude_state, scenario)
        severity = float(config["severity"])
        paths = self.simulate_paths(probability, horizon=30, return_scale=0.01 * severity)
        return {
            "scenario_key": scenario,
            "scenario": str(config["label"]),
            "var_95": self.compute_var(paths, 0.95),
            "var_99": self.compute_var(paths, 0.99),
            "es_95": self.compute_es(paths, 0.95),
            "paths": paths,
            "bias": float(config["coin_bias"]),
            "severity": severity,
        }

    def compare_gaussian_vs_qrw(
        self,
        tick_df: pd.DataFrame,
        n_paths_display: int = 1000,
        *,
        horizon: int = 20,
    ) -> dict:
        """Compare matched-scale Gaussian and fat-tail QRW distributions."""
        if "price" not in tick_df.columns:
            raise ValueError("tick_df must contain a price column")
        returns = np.diff(np.log(tick_df["price"].to_numpy(dtype=float)))
        returns = returns[np.isfinite(returns)]
        if len(returns) < 20:
            raise ValueError("at least 20 finite returns are required")

        n_bins = 101
        positions = np.linspace(-1.0, 1.0, n_bins)
        empirical_tail_center = 1.5 * float(np.std(positions))
        empirical_tail_width = 0.10
        center = np.exp(-0.5 * (positions / 0.16) ** 2)
        tails = np.exp(-0.5 * ((np.abs(positions) - empirical_tail_center) / empirical_tail_width) ** 2)
        probability = 0.86 * center / center.sum() + 0.14 * tails / tails.sum()
        probability /= probability.sum()
        normalized_std = float(np.sqrt(np.sum(positions**2 * probability)))
        empirical_std = float(np.std(returns, ddof=1))
        scale = empirical_std / max(normalized_std, 1e-12)
        paths = self.simulate_paths(probability, horizon=horizon, return_scale=scale)

        return {
            "returns": returns[-int(n_paths_display):],
            "mu": float(returns.mean()),
            "sigma": empirical_std,
            "gaussian_var_95": self.compute_gaussian_var_from_returns(returns, 0.95, horizon=horizon),
            "gaussian_var_99": self.compute_gaussian_var_from_returns(returns, 0.99, horizon=horizon),
            "qrw_var_95": self.compute_var(paths, 0.95),
            "qrw_var_99": self.compute_var(paths, 0.99),
            "qrw_es_95": self.compute_es(paths, 0.95),
            "qrw_paths_final": paths[:, -1],
        }

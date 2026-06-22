"""QRW model with heavy-tailed jumps for market microstructure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from scipy.optimize import minimize
from src.models.adaptive_market_qrw import AdaptiveDecoherenceQRW

class HeavyTailAdaptiveQRW(AdaptiveDecoherenceQRW):
    """Adaptive QRW extended with Pareto-distributed heavy-tailed jumps."""

    def __init__(self, tick_data: pd.DataFrame, config: dict[str, Any]) -> None:
        super().__init__(tick_data, config)
        self.tail_index = float(self.config.get("tail_index", 3.0))
        self.jump_scale = float(self.config.get("jump_scale", 1.0))
        
    def _fit_jump_distribution(self, row_limit: int | None = None) -> None:
        """Fit a discrete Pareto distribution to empirical non-zero jump sizes."""
        limit = len(self.tick_data) if row_limit is None else int(row_limit)
        price = self.tick_data["price"].to_numpy(dtype=np.float64)[:limit]
        delta = np.abs(np.diff(price))
        valid = delta > 1e-12
        if "segment_id" in self.tick_data:
            segment = self.tick_data["segment_id"].to_numpy(copy=False)[:limit]
            valid &= segment[:-1] == segment[1:]
            
        jumps = delta[valid]
        if len(jumps) < 20:
            self.tail_index = 3.0
            self.jump_scale = 1.0
            return
            
        # Infer minimum jump size (typically tick size)
        x_min = np.min(jumps)
        
        def neg_log_likelihood(alpha: np.ndarray) -> float:
            a = alpha[0]
            if a <= 0:
                return 1e9
            # Continuous Pareto log likelihood:
            # L = n*log(a) + n*a*log(x_min) - (a+1)*sum(log(x))
            n = len(jumps)
            nll = -n * np.log(a) - n * a * np.log(x_min) + (a + 1) * np.sum(np.log(jumps))
            return float(nll)
            
        res = minimize(
            neg_log_likelihood, 
            x0=np.array([3.0]), 
            bounds=[(1.1, 10.0)]
        )
        if res.success:
            self.tail_index = float(res.x[0])
            self.jump_scale = float(x_min)
        else:
            self.tail_index = 3.0
            self.jump_scale = float(x_min)

    def calibrate_two_stage(
        self,
        output_path: str | Path | None = "results/calibrated_params_heavy_tail.json",
        *,
        warmup_fraction: float = 0.4,
    ) -> dict[str, Any]:
        """Run structural calibration and then fit the jump distribution."""
        out = super().calibrate_two_stage(output_path=None, warmup_fraction=warmup_fraction)
        
        warmup_rows = int(len(self.tick_data) * warmup_fraction)
        self._fit_jump_distribution(warmup_rows)
        
        out["tail_index"] = self.tail_index
        out["jump_scale"] = self.jump_scale
        out["model_type"] = "heavy_tail_adaptive_qrw"
        
        self.calibrated_parameters = out.copy()
        
        if output_path is not None:
            import json
            from pathlib import Path
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(out, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            
        return out

    def simulate_price_path(
        self,
        n_paths: int,
        T: int | None = None,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        if n_paths < 1:
            raise ValueError("n_paths must be positive")
        steps = (
            self._last_simulation_steps
            if T is None and self._last_simulation_steps
            else (1 if T is None else T)
        )
        if steps < 1 or steps > len(self.tick_data):
            raise ValueError("T must be within the available observations")

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        probability = self.predict_probability()[:steps]
        moving = (
            rng.random((n_paths, steps))
            < self.movement_probability
        )
        positions = np.zeros(n_paths, dtype=np.float64)
        paths = np.empty((n_paths, steps), dtype=np.float64)
        
        for index, probability_right in enumerate(probability):
            direction = np.where(
                rng.random(n_paths) < probability_right,
                1.0,
                -1.0,
            )
            # Sample heavy-tailed jumps
            u = rng.random(n_paths)
            # Inverse transform for Pareto: x = x_min / (1 - u)^(1/alpha)
            jumps = self.jump_scale / np.power(1.0 - u, 1.0 / self.tail_index)
            # Round to nearest multiple of jump_scale to stay on grid
            jumps = np.round(jumps / self.jump_scale) * self.jump_scale
            
            movement = np.where(moving[:, index], direction * jumps, 0.0)
            positions += movement
            paths[:, index] = positions
            
        return paths

"""Distribution-drift anomaly detection for Track A QRW states."""

from __future__ import annotations

import numpy as np
import pandas as pd
import collections


class QRWAnomalyDetector:
    """Score current probability states against a normal-market baseline."""

    VALID_TYPES = {
        "normal",
        "suspicious",
        "directional_pressure",
        "market_indecision",
        "quote_stuffing",
    }

    # Uncalibrated heuristic thresholds for classify_anomaly_type -- not fit
    # from data. Named here so their role is discoverable.
    _NORMAL_SIGMA_MAX = 1.5
    _DIRECTIONAL_IMBALANCE_MIN = 0.35
    _BIMODAL_PEAK_MIN = 0.08
    _BIMODAL_CENTER_MASS_MAX = 0.25
    _QUOTE_STUFFING_ENTROPY_MIN = 0.90
    _QUOTE_STUFFING_PEAK_MAX = 0.08

    def __init__(
        self,
        qrw_model=None,
        n_baseline_days: int = 10,
        *,
        n_bins: int = 31,
        window: int = 50,
    ) -> None:
        if int(n_bins) < 7 or int(n_bins) % 2 == 0 or int(window) < 10:
            raise ValueError("n_bins must be odd >= 7 and window must be >= 10")
        self.qrw = qrw_model
        self.n_baseline_days = int(n_baseline_days)
        self.n_bins = int(n_bins)
        self.window = int(window)
        self.baseline: np.ndarray | None = None
        self.baseline_std: np.ndarray | None = None
        self.baseline_score_mean = 0.0
        self.baseline_score_std = 1.0
        self.score_history: collections.deque[float] = collections.deque(maxlen=1000)
        self.current_distribution: np.ndarray | None = None

    @staticmethod
    def _probability(state: np.ndarray) -> np.ndarray:
        values = np.asarray(state).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("state must be a non-empty finite array")
        if np.iscomplexobj(values):
            probability = np.abs(values) ** 2
        else:
            probability = values.astype(np.float64, copy=True)
            if np.any(probability < 0):
                raise ValueError("real states must contain non-negative probabilities")
        probability = np.maximum(probability, 1e-12)
        return probability / probability.sum()

    def _window_distribution(self, frame: pd.DataFrame) -> np.ndarray:
        if "obi" in frame.columns:
            values = frame["obi"].to_numpy(dtype=float)
            values = np.clip(values[np.isfinite(values)], -1.0, 1.0)
        elif "price" in frame.columns:
            prices = frame["price"].to_numpy(dtype=float)
            returns = np.diff(np.log(prices))
            scale = float(np.std(returns, ddof=0)) + 1e-12
            values = np.tanh(returns / (3.0 * scale))
        else:
            raise ValueError("frame must contain obi or price")
        if len(values) < 2:
            raise ValueError("window must contain at least two finite observations")
        counts, _ = np.histogram(values, bins=self.n_bins, range=(-1.0, 1.0))
        probability = counts.astype(float) + 1e-6
        return probability / probability.sum()

    def fit_baseline(self, normal_tick_df: pd.DataFrame) -> "QRWAnomalyDetector":
        if len(normal_tick_df) < self.window * 2:
            raise ValueError("baseline data must contain at least two windows")
        stride = max(1, self.window // 5)
        distributions = [
            self._window_distribution(normal_tick_df.iloc[end - self.window:end])
            for end in range(self.window, len(normal_tick_df) + 1, stride)
        ]
        matrix = np.vstack(distributions)
        baseline = np.maximum(matrix.mean(axis=0), 1e-12)
        self.baseline = baseline / baseline.sum()
        self.baseline_std = matrix.std(axis=0)
        scores = np.array([self._kl(row, self.baseline) for row in matrix])
        self.baseline_score_mean = float(scores.mean())
        self.baseline_score_std = max(float(scores.std(ddof=0)), 1e-6)
        self.score_history = collections.deque(scores.tolist(), maxlen=1000)
        self.current_distribution = matrix[-1]
        return self

    @staticmethod
    def _kl(probability: np.ndarray, baseline: np.ndarray) -> float:
        value = float(np.sum(probability * np.log(probability / baseline)))
        return max(value, 0.0)

    def compute_anomaly_score(self, current_amplitude: np.ndarray) -> float:
        if self.baseline is None:
            raise RuntimeError("fit_baseline() must be called first")
        probability = self._probability(current_amplitude)
        if len(probability) != len(self.baseline):
            source = np.linspace(0.0, 1.0, len(probability))
            target = np.linspace(0.0, 1.0, len(self.baseline))
            probability = np.interp(target, source, probability)
            probability = np.maximum(probability, 1e-12)
            probability /= probability.sum()
        self.current_distribution = probability
        return self._kl(probability, self.baseline)

    def compute_sigma_score(self, raw_score: float) -> float:
        if self.baseline is None:
            raise RuntimeError("fit_baseline() must be called first")
        return max(
            0.0,
            float((raw_score - self.baseline_score_mean) / self.baseline_score_std),
        )

    def classify_anomaly_type(
        self,
        amplitude: np.ndarray,
        sigma_score: float,
    ) -> str:
        if sigma_score < self._NORMAL_SIGMA_MAX:
            return "normal"
        probability = self._probability(amplitude)
        center = len(probability) // 2
        p_left = float(probability[:center].sum())
        p_right = float(probability[center + 1:].sum())
        if abs(p_right - p_left) > self._DIRECTIONAL_IMBALANCE_MIN:
            return "directional_pressure"
        outer = max(1, center // 2)
        left_peak = float(probability[:outer].max(initial=0.0))
        right_peak = float(probability[-outer:].max(initial=0.0))
        center_mass = float(probability[max(0, center - 2):center + 3].sum())
        if (
            left_peak > self._BIMODAL_PEAK_MIN
            and right_peak > self._BIMODAL_PEAK_MIN
            and center_mass < self._BIMODAL_CENTER_MASS_MAX
        ):
            return "market_indecision"
        entropy = float(-np.sum(probability * np.log(probability)) / np.log(len(probability)))
        if entropy > self._QUOTE_STUFFING_ENTROPY_MIN or float(probability.max()) < self._QUOTE_STUFFING_PEAK_MAX:
            return "quote_stuffing"
        return "suspicious"

    def rolling_anomaly_scan(
        self,
        tick_df: pd.DataFrame,
        *,
        step: int = 5,
    ) -> pd.DataFrame:
        if self.baseline is None:
            raise RuntimeError("fit_baseline() must be called first")
        if int(step) < 1 or len(tick_df) < self.window:
            raise ValueError("step must be positive and data must cover one window")
        records: list[dict] = []
        for end in range(self.window, len(tick_df) + 1, int(step)):
            probability = self._window_distribution(tick_df.iloc[end - self.window:end])
            raw_score = self.compute_anomaly_score(probability)
            sigma_score = self.compute_sigma_score(raw_score)
            anomaly_type = self.classify_anomaly_type(probability, sigma_score)
            self.score_history.append(raw_score)
            timestamp = tick_df.iloc[end - 1].get("timestamp", tick_df.index[end - 1])
            records.append(
                {
                    "row": end - 1,
                    "timestamp": timestamp,
                    "raw_score": raw_score,
                    "sigma_score": sigma_score,
                    "anom_type": anomaly_type,
                    "alert": sigma_score > 3.0,
                }
            )
        return pd.DataFrame(records)

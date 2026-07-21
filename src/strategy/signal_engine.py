"""Causal BUY/SELL/HOLD decisions from QRW-style probability states."""

from __future__ import annotations

import numpy as np
import pandas as pd


class QRWSignalEngine:
    """Translate directional probability mass into executable signals."""

    VALID_SIGNALS = {"BUY", "SELL", "HOLD"}

    # Uncalibrated heuristic weights for the momentum/OBI signal blend in
    # _causal_directional_probability -- not fit from data. Named here so
    # their role is discoverable; replace with calibrated values if/when
    # available.
    _MOMENTUM_WEIGHT = 0.65
    _OBI_WEIGHT = 0.35
    _OBI_SENSITIVITY = 2.0
    _FLAT_PROBABILITY_BASE = 0.08
    _FLAT_PROBABILITY_SLOPE = 0.20
    _FLAT_PROBABILITY_MIN = 0.05
    _FLAT_PROBABILITY_MAX = 0.28

    def __init__(self, theta_buy: float = 0.60, theta_sell: float = 0.60, transaction_cost: float = 0.0005) -> None:
        for name, value in (("theta_buy", theta_buy), ("theta_sell", theta_sell)):
            if not 0.5 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0.5, 1.0)")
        self.theta_buy = float(theta_buy)
        self.theta_sell = float(theta_sell)
        self.transaction_cost = float(transaction_cost)

    @staticmethod
    def _probability(amplitude_state: np.ndarray) -> np.ndarray:
        values = np.asarray(amplitude_state).reshape(-1)
        if values.size < 3 or values.size % 2 == 0 or not np.all(np.isfinite(values)):
            raise ValueError("state must have an odd number of at least three finite values")
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

    def amplitude_to_signal(self, amplitude_state: np.ndarray) -> dict[str, float | str]:
        probability = self._probability(amplitude_state)
        center = len(probability) // 2
        p_down = float(probability[:center].sum())
        p_flat = float(probability[center])
        p_up = float(probability[center + 1:].sum())
        if p_up > self.theta_buy:
            signal = "BUY"
        elif p_down > self.theta_sell:
            signal = "SELL"
        else:
            signal = "HOLD"
        confidence = float(np.clip(max(p_up, p_down) - 0.5, 0.0, 0.5))
        return {
            "signal": signal,
            "confidence": confidence,
            "p_up": p_up,
            "p_down": p_down,
            "p_flat": p_flat,
        }

    @staticmethod
    def _causal_directional_probability(history: pd.DataFrame) -> np.ndarray:
        """Build [down, flat, up] mass using only the supplied prefix."""
        prices = history["price"].to_numpy(dtype=float)
        returns = np.diff(np.log(prices[-31:])) if len(prices) > 1 else np.array([0.0])
        scale = float(np.std(returns, ddof=0)) + 1e-12
        momentum = float(np.mean(returns) / scale * np.sqrt(max(len(returns), 1)))
        score = QRWSignalEngine._MOMENTUM_WEIGHT * np.tanh(momentum)
        if "obi" in history.columns:
            obi = history["obi"].tail(30).to_numpy(dtype=float)
            obi = obi[np.isfinite(obi)]
            if len(obi):
                score += QRWSignalEngine._OBI_WEIGHT * np.tanh(
                    QRWSignalEngine._OBI_SENSITIVITY * float(obi.mean())
                )
        score = float(np.clip(score, -1.0, 1.0))
        flat = float(
            np.clip(
                QRWSignalEngine._FLAT_PROBABILITY_BASE
                + QRWSignalEngine._FLAT_PROBABILITY_SLOPE * (1.0 - abs(score)),
                QRWSignalEngine._FLAT_PROBABILITY_MIN,
                QRWSignalEngine._FLAT_PROBABILITY_MAX,
            )
        )
        directional = 1.0 - flat
        p_up = directional * (0.5 + 0.5 * score)
        p_down = directional - p_up
        return np.array([p_down, flat, p_up], dtype=np.float64)

    def build_probability_frame(
        self,
        tick_df: pd.DataFrame,
        qrw_model=None,
        *,
        warmup: int = 30,
    ) -> pd.DataFrame:
        """Create walk-forward probabilities; each row predicts only row i+1."""
        if "price" not in tick_df.columns:
            raise ValueError("tick_df must contain a price column")
        if int(warmup) < 2 or len(tick_df) <= warmup + 1:
            raise ValueError("tick_df must contain more than warmup + 1 rows")
        prices = tick_df["price"].to_numpy(dtype=float)
        if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError("prices must be finite and positive")

        provider = getattr(qrw_model, "probability_from_history", None)
        records: list[dict] = []
        for index in range(int(warmup), len(tick_df) - 1):
            history = tick_df.iloc[:index + 1]
            state = provider(history.copy()) if callable(provider) else self._causal_directional_probability(history)
            probability = self._probability(state)
            center = len(probability) // 2
            p_down = float(probability[:center].sum())
            p_flat = float(probability[center])
            p_up = float(probability[center + 1:].sum())
            future_return = float(prices[index + 1] / prices[index] - 1.0)
            timestamp = tick_df.iloc[index].get("timestamp", tick_df.index[index])
            records.append(
                {
                    "row": index,
                    "timestamp": timestamp,
                    "price": float(prices[index]),
                    "p_up": p_up,
                    "p_down": p_down,
                    "p_flat": p_flat,
                    "ret_1step": future_return,
                }
            )
        return pd.DataFrame(records)

    def backtest_from_probabilities(self, probability_df: pd.DataFrame) -> pd.DataFrame:
        required = {"p_up", "p_down", "p_flat", "ret_1step"}
        missing = required.difference(probability_df.columns)
        if missing:
            raise ValueError(f"probability_df is missing columns: {sorted(missing)}")
        result = probability_df.copy()
        buy = result["p_up"].to_numpy(dtype=float) > self.theta_buy
        sell = (~buy) & (result["p_down"].to_numpy(dtype=float) > self.theta_sell)
        signals = np.where(buy, "BUY", np.where(sell, "SELL", "HOLD"))
        positions = np.where(buy, 1.0, np.where(sell, -1.0, 0.0))
        future_returns = result["ret_1step"].to_numpy(dtype=float)
        result["signal"] = signals
        result["confidence"] = np.clip(
            np.maximum(result["p_up"], result["p_down"]) - 0.5, 0.0, 0.5
        )
        trade_size = np.abs(np.diff(positions, prepend=0.0))
        result["pnl"] = positions * future_returns - trade_size * self.transaction_cost
        correct = np.where(
            positions == 0,
            "HOLD",
            np.where(positions * future_returns > 0, "WIN", "LOSS"),
        )
        result["correct"] = correct
        result["momentum"] = result["p_up"] - result["p_down"]
        return result

    def backtest_signals(
        self,
        tick_df: pd.DataFrame,
        qrw_model=None,
        *,
        warmup: int = 30,
    ) -> pd.DataFrame:
        probabilities = self.build_probability_frame(tick_df, qrw_model, warmup=warmup)
        return self.backtest_from_probabilities(probabilities)

    def latest_signal(
        self,
        tick_df: pd.DataFrame,
        qrw_model=None,
        *,
        min_history: int = 30,
    ) -> dict[str, float | int | str | object]:
        """Compute the forward signal from the latest in-memory buffer state."""
        if "price" not in tick_df.columns:
            raise ValueError("tick_df must contain a price column")
        if int(min_history) < 2 or len(tick_df) < int(min_history):
            raise ValueError(f"at least {int(min_history)} ticks are required")
        prices = tick_df["price"].to_numpy(dtype=float)
        if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError("prices must be finite and positive")

        history = tick_df.copy()
        provider = getattr(qrw_model, "probability_from_history", None)
        state = (
            provider(history)
            if callable(provider)
            else self._causal_directional_probability(history)
        )
        decision = self.amplitude_to_signal(state)
        decision.update(
            {
                "timestamp": history.iloc[-1].get("timestamp", history.index[-1]),
                "price": float(prices[-1]),
                "momentum": float(decision["p_up"] - decision["p_down"]),
                "observations": int(len(history)),
            }
        )
        return decision

    @staticmethod
    def compute_signal_metrics(backtest_df: pd.DataFrame) -> dict[str, float | int]:
        if "signal" not in backtest_df.columns or "pnl" not in backtest_df.columns:
            raise ValueError("backtest_df must contain signal and pnl columns")
        trades = backtest_df[backtest_df["signal"] != "HOLD"]
        pnl = backtest_df["pnl"].to_numpy(dtype=float)
        trade_pnl = trades["pnl"].to_numpy(dtype=float)
        gross_profit = float(trade_pnl[trade_pnl > 0].sum())
        gross_loss = float(-trade_pnl[trade_pnl < 0].sum())
        cumulative = np.cumsum(pnl)
        running_peak = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
        drawdown = cumulative - running_peak
        pnl_std = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
        sharpe = float(np.mean(pnl) / pnl_std * np.sqrt(len(pnl))) if pnl_std > 0 else 0.0
        return {
            "hit_rate": float((trades["correct"] == "WIN").mean()) if len(trades) else 0.0,
            "profit_factor": gross_profit / max(gross_loss, 1e-12),
            "net_pnl": float(pnl.sum()),
            "max_drawdown": float(drawdown.min(initial=0.0)),
            "n_trades": int(len(trades)),
            "avg_confidence": float(trades["confidence"].mean()) if len(trades) else 0.0,
            "sharpe": sharpe,
        }

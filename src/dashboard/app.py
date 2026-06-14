"""Interactive, explicitly exploratory QRW dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.coin_operators import biased_coin, hadamard_coin
from src.models.qrw_core import DensityMatrixQRW


def main() -> None:
    """Launch the Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="QRW Market Microstructure",
        layout="wide",
    )
    st.title("QRW Market Microstructure")
    st.caption(
        "Exploratory interface only. Current results do not establish "
        "QRW superiority."
    )

    with st.sidebar:
        st.header("QRW controls")
        gamma = st.slider("Decoherence gamma", 0.0, 5.0, 0.5, 0.05)
        alpha = st.slider("Coin angle multiplier", 0.1, 2.0, 1.0, 0.05)
        coin_type = st.selectbox("Coin", ("Hadamard", "Biased"))
        n_steps = st.slider("Steps", 10, 120, 60, 5)

    coin = (
        hadamard_coin()
        if coin_type == "Hadamard"
        else biased_coin(np.pi / 4.0 * alpha)
    )
    walk = DensityMatrixQRW(2 * n_steps + 3, coin=coin)
    history = []
    variances = []
    for _ in range(n_steps):
        probability = walk.step_with_decoherence(gamma)
        history.append(probability)
        mean = float(probability @ walk.positions)
        variances.append(
            float(probability @ (walk.positions - mean) ** 2)
        )

    distribution_column, variance_column = st.columns(2)
    with distribution_column:
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(walk.positions, history[-1], color="#0072B2")
        axis.fill_between(
            walk.positions,
            history[-1],
            color="#0072B2",
            alpha=0.25,
        )
        axis.set_title(f"Position probability at t={n_steps}")
        axis.set_xlabel("Position")
        axis.set_ylabel("Probability")
        st.pyplot(fig)
    with variance_column:
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(
            np.arange(1, n_steps + 1),
            variances,
            label="QRW control",
            color="#D55E00",
        )
        axis.plot(
            np.arange(1, n_steps + 1),
            np.arange(1, n_steps + 1),
            label="CRW reference",
            color="#333333",
            linestyle="--",
        )
        axis.set_title("Variance growth")
        axis.set_xlabel("Step")
        axis.set_ylabel("Variance")
        axis.legend()
        st.pyplot(fig)

    scorecard_path = ROOT / "results" / "scorecard.csv"
    if scorecard_path.exists():
        st.subheader("Frozen Phase 5 scorecard")
        st.dataframe(pd.read_csv(scorecard_path), use_container_width=True)
    else:
        st.warning("Run Phase 5 to populate the scorecard.")


if __name__ == "__main__":
    main()

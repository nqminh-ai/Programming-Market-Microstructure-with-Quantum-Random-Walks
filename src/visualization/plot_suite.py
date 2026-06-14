"""Reproducible Phase 6 visualization suite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import binom, gaussian_kde

from src.models.coin_operators import obi_adaptive_coin
from src.models.qrw_core import QuantumRandomWalk

FIGSIZE = (10.0, 6.0)
DPI = 300
COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)
PHASE6_FIGURES = (
    "prob_evolution.gif",
    "variance_scaling.png",
    "return_distributions.png",
    "acf_comparison.png",
    "sample_paths.png",
    "coin_operator_heatmap.png",
    "scorecard.png",
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def _destination(output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    destination = _destination(output_path)
    fig.savefig(
        destination,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return destination


def _crw_probability(positions: np.ndarray, step: int) -> np.ndarray:
    probability = np.zeros(len(positions), dtype=np.float64)
    if step == 0:
        probability[positions == 0] = 1.0
        return probability
    valid = (np.abs(positions) <= step) & ((step + positions) % 2 == 0)
    right_steps = ((step + positions[valid]) // 2).astype(int)
    probability[valid] = binom.pmf(right_steps, step, 0.5)
    return probability


def plot_probability_evolution(
    qrw: QuantumRandomWalk | None = None,
    t_snapshots: Sequence[int] = (0, 5, 10, 20, 30, 40, 50, 60),
    *,
    output_path: str | Path = "figures/prob_evolution.gif",
    fps: float = 1.5,
) -> Path:
    """Animate coherent QRW and symmetric CRW position probabilities."""
    _style()
    snapshots = sorted({int(step) for step in t_snapshots})
    if not snapshots or snapshots[0] < 0:
        raise ValueError("t_snapshots must contain non-negative integers")
    maximum = snapshots[-1]
    engine = qrw or QuantumRandomWalk(2 * maximum + 3)
    if engine.n_positions < 2 * maximum + 1:
        raise ValueError("qrw lattice is too small for the requested snapshots")
    engine.reset()

    qrw_history: dict[int, np.ndarray] = {0: engine.get_probability().copy()}
    for step in range(1, maximum + 1):
        probability = engine.step()
        if step in snapshots:
            qrw_history[step] = probability.copy()

    positions = engine.positions
    crw_history = {
        step: _crw_probability(positions, step) for step in snapshots
    }
    limit = max(
        max(float(values.max()) for values in qrw_history.values()),
        max(float(values.max()) for values in crw_history.values()),
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    for axis, title in zip(
        axes,
        ("Coherent quantum random walk", "Symmetric classical random walk"),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xlabel("Position")
        axis.set_xlim(-maximum - 2, maximum + 2)
        axis.set_ylim(0.0, limit * 1.12)
    axes[0].set_ylabel("Probability")
    qrw_line, = axes[0].plot([], [], color=COLORS[0], linewidth=2.0)
    crw_line, = axes[1].plot([], [], color=COLORS[1], linewidth=2.0)
    qrw_fill = [None]
    crw_fill = [None]
    title = fig.suptitle("")

    def update(frame: int) -> tuple[object, ...]:
        if qrw_fill[0] is not None:
            qrw_fill[0].remove()
        if crw_fill[0] is not None:
            crw_fill[0].remove()
        qrw_probability = qrw_history[frame]
        crw_probability = crw_history[frame]
        qrw_line.set_data(positions, qrw_probability)
        crw_line.set_data(positions, crw_probability)
        qrw_fill[0] = axes[0].fill_between(
            positions,
            qrw_probability,
            color=COLORS[0],
            alpha=0.25,
        )
        crw_fill[0] = axes[1].fill_between(
            positions,
            crw_probability,
            color=COLORS[1],
            alpha=0.25,
        )
        title.set_text(f"Probability evolution at t = {frame}")
        return qrw_line, crw_line, qrw_fill[0], crw_fill[0], title

    animation = FuncAnimation(
        fig,
        update,
        frames=snapshots,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    destination = _destination(output_path)
    animation.save(destination, writer=PillowWriter(fps=fps), dpi=DPI)
    plt.close(fig)
    return destination


def _variance_by_horizon(
    values: np.ndarray,
    horizons: np.ndarray,
    *,
    paths: bool,
) -> np.ndarray:
    variances = []
    for horizon in horizons:
        if paths:
            changes = values[:, horizon:] - values[:, :-horizon]
        else:
            changes = values[horizon:] - values[:-horizon]
        variances.append(
            float(np.var(changes, ddof=1))
            if changes.size > 1
            else float("nan")
        )
    return np.asarray(variances, dtype=np.float64)


def plot_variance_scaling(
    empirical_prices: Sequence[float],
    simulated_paths: Mapping[str, np.ndarray],
    *,
    output_path: str | Path = "figures/variance_scaling.png",
) -> Path:
    """Plot log-log variance growth with fitted scaling exponents."""
    _style()
    empirical = np.asarray(empirical_prices, dtype=np.float64)
    maximum = min(
        len(empirical) - 1,
        min(paths.shape[1] - 1 for paths in simulated_paths.values()),
    )
    horizons = np.unique(
        np.geomspace(1, maximum, num=min(14, maximum)).astype(int)
    )
    series: list[tuple[str, np.ndarray, bool]] = [
        ("Empirical", empirical, False)
    ]
    series.extend(
        (name, np.asarray(paths, dtype=np.float64), True)
        for name, paths in simulated_paths.items()
    )

    fig, axis = plt.subplots(figsize=FIGSIZE)
    for index, (name, values, is_paths) in enumerate(series):
        variance = _variance_by_horizon(values, horizons, paths=is_paths)
        valid = np.isfinite(variance) & (variance > 0.0)
        beta, intercept = np.polyfit(
            np.log(horizons[valid]),
            np.log(variance[valid]),
            deg=1,
        )
        color = COLORS[index % len(COLORS)]
        marker = "o" if name == "Empirical" else None
        axis.loglog(
            horizons,
            variance,
            marker=marker,
            markersize=4,
            linewidth=1.1,
            alpha=0.72,
            color=color,
        )
        fitted = np.exp(intercept) * horizons.astype(float) ** beta
        axis.loglog(
            horizons,
            fitted,
            linewidth=2.0,
            color=color,
            label=f"{name}: beta={beta:.3f}",
        )
    axis.set_title("Variance scaling across empirical and simulated paths")
    axis.set_xlabel("Horizon")
    axis.set_ylabel("Variance of price change")
    axis.legend(ncol=2, fontsize=8)
    return _save(fig, output_path)


def _standardized_returns(values: np.ndarray, *, paths: bool) -> np.ndarray:
    if paths:
        returns = np.diff(np.log(np.maximum(values, 1e-12)), axis=1).ravel()
    else:
        returns = np.diff(np.log(np.maximum(values, 1e-12)))
    returns = returns[np.isfinite(returns)]
    scale = float(np.std(returns, ddof=0))
    if scale <= 1e-15:
        return returns - np.mean(returns)
    return (returns - np.mean(returns)) / scale


def _kde_values(
    values: np.ndarray,
    grid: np.ndarray,
    *,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    if len(values) > 10_000:
        values = rng.choice(values, size=10_000, replace=False)
    if np.unique(np.round(values, decimals=10)).size < 6:
        values = values + rng.normal(0.0, 0.035, size=len(values))
    return gaussian_kde(values, bw_method="scott")(grid)


def plot_return_distribution_comparison(
    empirical_prices: Sequence[float],
    simulated_paths: Mapping[str, np.ndarray],
    *,
    output_path: str | Path = "figures/return_distributions.png",
    random_seed: int = 2026,
) -> Path:
    """Compare standardized one-step return densities."""
    _style()
    selected = (
        "QRW Adaptive",
        "CRW Simple",
        "GARCH(1,1)",
        "GBM",
    )
    samples: list[tuple[str, np.ndarray]] = [
        (
            "Empirical",
            _standardized_returns(
                np.asarray(empirical_prices, dtype=np.float64),
                paths=False,
            ),
        )
    ]
    samples.extend(
        (
            name,
            _standardized_returns(
                np.asarray(simulated_paths[name], dtype=np.float64),
                paths=True,
            ),
        )
        for name in selected
        if name in simulated_paths
    )
    grid = np.linspace(-5.0, 5.0, 600)
    fig, axis = plt.subplots(figsize=FIGSIZE)
    for index, (name, values) in enumerate(samples):
        density = _kde_values(
            values,
            grid,
            random_seed=random_seed + index,
        )
        axis.plot(
            grid,
            density,
            color=COLORS[index],
            linewidth=2.0,
            label=name,
        )
    axis.set_yscale("log")
    axis.set_ylim(1e-4, None)
    axis.set_title("Standardized one-step return distributions")
    axis.set_xlabel("Return z-score")
    axis.set_ylabel("Kernel density, log scale")
    axis.legend()
    axis.text(
        0.01,
        0.02,
        "Each series is standardized separately; shape, not scale, is compared.",
        transform=axis.transAxes,
        fontsize=8,
        color="#444444",
    )
    return _save(fig, output_path)


def _acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = values - np.mean(values)
    denominator = float(centered @ centered)
    if denominator <= 1e-20:
        return np.zeros(max_lag + 1, dtype=np.float64)
    result = np.ones(max_lag + 1, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        result[lag] = float(centered[:-lag] @ centered[lag:] / denominator)
    return result


def _mean_path_acf(paths: np.ndarray, max_lag: int) -> np.ndarray:
    returns = np.diff(np.log(np.maximum(paths, 1e-12)), axis=1)
    count = min(200, len(returns))
    return np.mean([_acf(row, max_lag) for row in returns[:count]], axis=0)


def plot_acf_comparison(
    empirical_prices: Sequence[float],
    simulated_paths: Mapping[str, np.ndarray],
    *,
    output_path: str | Path = "figures/acf_comparison.png",
    max_lag: int = 20,
) -> Path:
    """Plot return autocorrelation bars for empirical data and key models."""
    _style()
    empirical_returns = np.diff(
        np.log(np.maximum(np.asarray(empirical_prices, dtype=float), 1e-12))
    )
    names = [
        name
        for name in (
            "Empirical",
            "QRW Adaptive",
            "CRW Simple",
            "GARCH(1,1)",
            "GBM",
        )
        if name == "Empirical" or name in simulated_paths
    ]
    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    lag_values = np.arange(1, max_lag + 1)
    for index, (axis, name) in enumerate(zip(axes.flat, names, strict=False)):
        values = (
            _acf(empirical_returns, max_lag)
            if name == "Empirical"
            else _mean_path_acf(simulated_paths[name], max_lag)
        )
        axis.bar(
            lag_values,
            values[1:],
            color=COLORS[index],
            width=0.75,
        )
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_title(name, fontsize=10)
    for axis in axes.flat[len(names) :]:
        axis.axis("off")
    fig.suptitle("One-step return autocorrelation")
    fig.supxlabel("Lag")
    fig.supylabel("ACF")
    fig.tight_layout()
    return _save(fig, output_path)


def plot_sample_paths(
    empirical_prices: Sequence[float],
    simulated_paths: Mapping[str, np.ndarray],
    *,
    output_path: str | Path = "figures/sample_paths.png",
    n_paths: int = 10,
) -> Path:
    """Overlay empirical prices with QRW and simple CRW sample paths."""
    _style()
    required = {"QRW Adaptive", "CRW Simple"}
    missing = required.difference(simulated_paths)
    if missing:
        raise ValueError(f"sample paths are missing models: {sorted(missing)}")
    empirical = np.asarray(empirical_prices, dtype=np.float64)
    fig, axis = plt.subplots(figsize=FIGSIZE)
    steps = np.arange(len(empirical))
    for name, color in (
        ("QRW Adaptive", COLORS[0]),
        ("CRW Simple", COLORS[1]),
    ):
        paths = np.asarray(simulated_paths[name], dtype=np.float64)
        for index in range(min(n_paths, len(paths))):
            axis.plot(
                steps,
                paths[index, : len(empirical)],
                color=color,
                alpha=0.24,
                linewidth=0.9,
                label=name if index == 0 else None,
            )
    axis.plot(
        steps,
        empirical,
        color="#111111",
        linewidth=2.4,
        label="Empirical",
        zorder=5,
    )
    axis.set_title("Empirical path and simulated QRW/CRW paths")
    axis.set_xlabel("Event step")
    axis.set_ylabel("BTCUSDT price")
    axis.legend()
    return _save(fig, output_path)


def plot_heatmap_coin_operator(
    *,
    obi: float = 0.75,
    alpha: float = 1.2,
    output_path: str | Path = "figures/coin_operator_heatmap.png",
) -> Path:
    """Show magnitude and phase of a representative adaptive coin."""
    _style()
    coin = obi_adaptive_coin(obi, alpha)
    magnitude = np.abs(coin) ** 2
    phase = np.angle(coin)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    panels = (
        (axes[0], magnitude, r"$|U_{ij}|^2$", 0.0, 1.0, "viridis"),
        (axes[1], phase, r"$\arg(U_{ij})$", -np.pi, np.pi, "coolwarm"),
    )
    for axis, values, title, lower, upper, cmap in panels:
        image = axis.imshow(values, vmin=lower, vmax=upper, cmap=cmap)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if abs(values[row, column]) > 0.7 else "black",
                    fontsize=12,
                )
        axis.set_xticks((0, 1), ("Left", "Right"))
        axis.set_yticks((0, 1), ("Left", "Right"))
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"OBI-adaptive coin at OBI={obi:.2f}, alpha={alpha:.2f}\n"
        "Directional information is phase-encoded; magnitudes remain unitary."
    )
    fig.tight_layout()
    return _save(fig, output_path)


def plot_benchmark_scorecard(
    scorecard: pd.DataFrame | str | Path,
    *,
    output_path: str | Path = "figures/scorecard.png",
) -> Path:
    """Plot model mean ranks, where a lower value is better."""
    _style()
    frame = (
        pd.read_csv(scorecard)
        if isinstance(scorecard, (str, Path))
        else scorecard.copy()
    )
    required = {"model", "mean_rank", "overall_rank"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"scorecard is missing columns: {sorted(missing)}")
    frame = frame.sort_values("mean_rank", ascending=False, kind="stable")
    colors = [
        COLORS[0] if model == "QRW Adaptive" else COLORS[1]
        for model in frame["model"]
    ]
    fig, axis = plt.subplots(figsize=FIGSIZE)
    bars = axis.barh(frame["model"], frame["mean_rank"], color=colors)
    for bar, rank in zip(bars, frame["overall_rank"], strict=True):
        axis.text(
            bar.get_width() + 0.04,
            bar.get_y() + bar.get_height() / 2,
            f"rank {int(rank)}",
            va="center",
            fontsize=9,
        )
    axis.set_title("Phase 5 benchmark scorecard")
    axis.set_xlabel("Mean metric rank, lower is better")
    axis.set_xlim(0.0, float(frame["mean_rank"].max()) + 0.8)
    axis.text(
        0.99,
        0.02,
        "Single-window exploratory ranking",
        transform=axis.transAxes,
        ha="right",
        fontsize=8,
        color="#444444",
    )
    return _save(fig, output_path)


def render_dashboard_preview(
    empirical_prices: Sequence[float],
    scorecard: pd.DataFrame | str | Path,
    *,
    output_path: str | Path = "figures/dashboard_screenshot.png",
) -> Path:
    """Render a static preview matching the optional Streamlit dashboard."""
    _style()
    frame = (
        pd.read_csv(scorecard)
        if isinstance(scorecard, (str, Path))
        else scorecard.copy()
    )
    prices = np.asarray(empirical_prices, dtype=np.float64)
    fig = plt.figure(figsize=(12.0, 7.0), facecolor="#F4F7FB")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(0.8, 2.1, 1.4),
        height_ratios=(1.0, 1.0),
    )
    sidebar = fig.add_subplot(grid[:, 0])
    path_axis = fig.add_subplot(grid[0, 1:])
    rank_axis = fig.add_subplot(grid[1, 1])
    note_axis = fig.add_subplot(grid[1, 2])
    sidebar.set_facecolor("#E8EEF6")
    sidebar.text(0.08, 0.94, "QRW Controls", fontsize=14, weight="bold")
    controls = (
        ("gamma", "2.44"),
        ("alpha OBI", "0.84"),
        ("coin", "adaptive"),
        ("steps", str(len(prices) - 1)),
    )
    for index, (name, value) in enumerate(controls):
        y = 0.78 - index * 0.16
        sidebar.text(0.08, y + 0.06, name, fontsize=9)
        sidebar.add_patch(
            plt.Rectangle(
                (0.08, y - 0.01),
                0.78,
                0.07,
                color="white",
                ec="#AAB7C4",
            )
        )
        sidebar.text(0.47, y + 0.025, value, ha="center", va="center")
    sidebar.set_xticks([])
    sidebar.set_yticks([])
    for spine in sidebar.spines.values():
        spine.set_visible(False)

    path_axis.plot(prices, color=COLORS[0], linewidth=1.8)
    path_axis.set_title("Empirical BTCUSDT event path")
    path_axis.set_xlabel("Event")
    path_axis.set_ylabel("Price")

    ordered = frame.sort_values("mean_rank", ascending=False)
    rank_axis.barh(
        ordered["model"],
        ordered["mean_rank"],
        color=[
            COLORS[0] if name == "QRW Adaptive" else "#9AA9B8"
            for name in ordered["model"]
        ],
    )
    rank_axis.set_title("Mean rank")
    rank_axis.set_xlabel("Lower is better")
    rank_axis.tick_params(axis="y", labelsize=8)

    note_axis.axis("off")
    winner = frame.sort_values("overall_rank").iloc[0]
    qrw = frame.loc[frame["model"] == "QRW Adaptive"].iloc[0]
    note_axis.text(0.02, 0.9, "Current result", fontsize=13, weight="bold")
    note_axis.text(
        0.02,
        0.7,
        f"Top model: {winner['model']}\n"
        f"QRW rank: {int(qrw['overall_rank'])}\n\n"
        "Scope: one June 12, 2026 window.\n"
        "No confirmatory superiority claim.",
        va="top",
        linespacing=1.5,
    )
    fig.suptitle("QRW Market Microstructure Dashboard", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return _save(fig, output_path)

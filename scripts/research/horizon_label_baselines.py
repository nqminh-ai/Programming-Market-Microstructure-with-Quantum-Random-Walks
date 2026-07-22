"""Is there a directional edge at a horizon that can pay its own costs?

Step 1 showed the one-tick horizon the project forecasts cannot be traded at any
accuracy, and that passive execution becomes plausible somewhere between 13 and
41 minutes depending on the asset. This script asks the follow-up question the
roadmap actually turns on: at those horizons, can the strong classical baselines
predict direction well enough to clear the break-even accuracy?

Two methodological points decide whether the answer means anything.

**Overlapping labels.** Predicting the return over the next ``h`` ticks from
every tick makes consecutive labels share almost all of their future, so the
effective sample is far smaller than the row count and ordinary confidence
intervals are badly overoptimistic. Anchors are therefore sampled every ``h``
ticks so that no two labels overlap at all. This costs a great deal of sample
size -- and reporting that honestly is the point, because the alternative is a
tight-looking interval that is simply wrong.

**A majority-class baseline.** Over long horizons a drifting market can be up
in, say, 55% of windows, so a model that always predicts "up" scores 55% without
any skill. Every model is therefore compared against the majority-class rate on
the same test set, not against 50%.

Status: exploratory. Uses the repository's registered causal features, but the
label is new, so nothing here is a confirmatory result.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit

from src.data.common import timestamps_to_nanoseconds
from src.evaluation.directional_baselines import (
    FEATURE_NAMES,
    _fit_logistic,
    _lagged_direction,
    _pairwise,
)
from src.evaluation.provenance import canonical_repo_path, sha256_file
from scripts.research.horizon_feasibility import (
    FEE_SCENARIOS,
    breakeven_accuracy,
    measure_half_spread,
    round_trip_cost,
)

ROOT = Path(__file__).resolve().parents[2]

NEEDED_COLUMNS = [
    "timestamp",
    "price",
    "tick_direction",
    "obi",
    "obi_valid",
    "trade_intensity",
    "segment_id",
    "mid_price",
]
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


# Downcast targets. price stays float64: log returns over long horizons need the
# precision, and float32 on a ~60,000 price loses the tick-scale differences the
# label depends on. The rest are features whose float32 range is ample.
DOWNCAST = {
    "obi": "float32",
    "trade_intensity": "float32",
    "tick_direction": "float32",
    "mid_price": "float32",
}


def _load(path: Path, max_rows: int) -> pd.DataFrame:
    """Load only the needed columns and downcast.

    A 31-day feature store is ~115M rows; read naively at float64 that is over
    7GB and the process is killed on a 16GB machine.
    """
    handle = pq.ParquetFile(path)
    names = set(handle.schema.names)
    columns = [column for column in NEEDED_COLUMNS if column in names]

    # Read one column at a time and release each Arrow buffer as soon as it has
    # been converted. Reading the whole table and calling to_pandas holds the
    # Arrow copy and the pandas copy at once, which on a 113M-row store is
    # ~8GB and gets the process killed. Column-wise, the peak is the finished
    # arrays plus one column.
    data: dict[str, np.ndarray] = {}
    for name in columns:
        column = pq.read_table(path, columns=[name]).column(0)
        dtype = DOWNCAST.get(name)
        if dtype is not None:
            column = column.cast(pa.type_for_alias(dtype))
        values = column.to_numpy(zero_copy_only=False)
        del column
        if max_rows and len(values) > max_rows:
            values = values[:max_rows]
        if name == "segment_id":
            try:
                values = values.astype(np.int32, copy=False)
            except (ValueError, OverflowError, TypeError):
                pass
        elif name == "obi_valid":
            values = values.astype(bool, copy=False)
        data[name] = values
        gc.collect()

    frame = pd.DataFrame(data, copy=False)
    del data
    gc.collect()
    if not frame["timestamp"].is_monotonic_increasing:
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        gc.collect()
    return frame


def build_horizon_events(frame: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    """Causal features at ``t`` against the sign of the return over ``t -> t+h``.

    Anchors are spaced ``horizon`` apart so that no two labels share any future
    return, and both endpoints must lie in the same contiguous segment.
    """
    price = frame["price"].to_numpy(dtype=np.float64)
    obi = frame["obi"].to_numpy(dtype=np.float64)
    direction = frame["tick_direction"].to_numpy(dtype=np.float64)
    intensity = frame["trade_intensity"].to_numpy(dtype=np.float64)

    obi_change = np.zeros(len(frame), dtype=np.float64)
    obi_change[1:] = np.diff(obi)
    if "segment_id" in frame.columns:
        segment = frame["segment_id"].to_numpy(copy=False)
        obi_change[1:][segment[:-1] != segment[1:]] = 0.0
    else:
        segment = np.zeros(len(frame), dtype=np.int64)

    features = np.column_stack(
        [obi, direction, obi_change, np.abs(obi), np.log1p(np.maximum(intensity, 0.0))]
    )
    lagged = _lagged_direction(direction, lags=5)

    anchors = np.arange(0, len(frame) - horizon, horizon, dtype=np.int64)
    if anchors.size == 0:
        raise ValueError(f"horizon {horizon} exceeds the available rows")

    future = anchors + horizon
    usable = (
        (segment[anchors] == segment[future])
        & np.isfinite(price[anchors])
        & np.isfinite(price[future])
        & (price[anchors] > 0)
        & (price[future] > 0)
        & np.isfinite(features[anchors]).all(axis=1)
    )
    if "obi_valid" in frame.columns:
        usable &= frame["obi_valid"].to_numpy()[anchors].astype(bool)

    anchors, future = anchors[usable], future[usable]
    log_return = np.log(price[future] / price[anchors])
    # A window that ends exactly where it began carries no directional
    # information and would otherwise be scored as a "down" label.
    moved = np.abs(log_return) > 1e-12
    anchors, future, log_return = anchors[moved], future[moved], log_return[moved]

    return {
        "features": features[anchors],
        "lagged": lagged[anchors],
        "target": (log_return > 0.0).astype(np.float64),
        "log_return": log_return,
        "timestamp": frame["timestamp"].to_numpy()[anchors],
    }


def _standardise(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (other - mean) / scale


def _fit_and_score(
    design_train: np.ndarray,
    target_train: np.ndarray,
    design_test: np.ndarray,
) -> np.ndarray:
    """Fit with the registered L2 grid, picking the value by training log loss."""
    best_beta, best_loss = None, np.inf
    for regularization in L2_GRID:
        try:
            beta = _fit_logistic(design_train, target_train, regularization)
        except RuntimeError:
            continue
        score = np.column_stack([np.ones(len(design_train)), design_train]) @ beta
        loss = float(
            np.mean(np.logaddexp(0.0, score) - target_train * score)
        )
        if loss < best_loss:
            best_beta, best_loss = beta, loss
    if best_beta is None:
        raise RuntimeError("no logistic fit converged")
    return expit(np.column_stack([np.ones(len(design_test)), design_test]) @ best_beta)


def evaluate_models(events: dict[str, np.ndarray], train_fraction: float) -> dict[str, Any]:
    n = len(events["target"])
    split = int(n * train_fraction)
    if split < 30 or n - split < 30:
        raise ValueError(
            f"need at least 30 non-overlapping windows per side; got {split}/{n - split}"
        )

    target_train = events["target"][:split]
    target_test = events["target"][split:]

    designs = {
        "Logistic L2 (5F)": events["features"],
        "Logistic L2 + Pairwise": _pairwise(events["features"]),
        "OrderFlow AR(5)": np.column_stack([events["features"], events["lagged"]]),
    }

    majority = float(target_train.mean() >= 0.5)
    results: dict[str, dict[str, float]] = {
        "Majority class": {
            "accuracy": float((target_test == majority).mean()),
            "n_parameters": 0,
        }
    }
    for name, design in designs.items():
        train_design, test_design = _standardise(design[:split], design[split:])
        probability = _fit_and_score(train_design, target_train, test_design)
        prediction = (probability > 0.5).astype(np.float64)
        results[name] = {
            "accuracy": float((prediction == target_test).mean()),
            "brier": float(np.mean((probability - target_test) ** 2)),
            "n_parameters": int(design.shape[1] + 1),
        }

    return {
        "n_windows": int(n),
        "n_train": int(split),
        "n_test": int(n - split),
        "majority_class_rate": float(max(target_test.mean(), 1 - target_test.mean())),
        "models": results,
    }


def analyse(
    frame: pd.DataFrame, horizons: tuple[int, ...], train_fraction: float
) -> dict[str, Any]:
    half_spread = measure_half_spread(frame)
    costs = {
        name: round_trip_cost(scenario, half_spread)
        for name, scenario in FEE_SCENARIOS.items()
    }
    try:
        nanoseconds = timestamps_to_nanoseconds(frame["timestamp"]).to_numpy(dtype="int64")
        seconds_per_tick = (
            float(nanoseconds.max() - nanoseconds.min()) / 1e9 / max(len(frame) - 1, 1)
        )
    except (ValueError, TypeError, OverflowError):
        seconds_per_tick = None

    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        try:
            events = build_horizon_events(frame, horizon)
            scored = evaluate_models(events, train_fraction)
        except ValueError as error:
            rows.append({"horizon_ticks": int(horizon), "skipped": str(error)})
            continue

        expected_move = float(np.abs(events["log_return"]).mean())
        thresholds = {
            name: breakeven_accuracy(cost, expected_move) for name, cost in costs.items()
        }
        best_name = max(
            scored["models"], key=lambda key: scored["models"][key]["accuracy"]
        )
        best_accuracy = scored["models"][best_name]["accuracy"]
        # The decisive quantity: a directional bet at hit rate p earns
        # (2p - 1) * E|move| before paying the round trip.
        net_edge = {
            name: (2.0 * best_accuracy - 1.0) * expected_move - cost
            for name, cost in costs.items()
        }
        rows.append(
            {
                "horizon_ticks": int(horizon),
                "seconds": None if seconds_per_tick is None else seconds_per_tick * horizon,
                "expected_abs_move": expected_move,
                "breakeven_accuracy": thresholds,
                "net_edge_per_trade": net_edge,
                "best_model": best_name,
                "best_accuracy": best_accuracy,
                "beats_majority": bool(
                    best_accuracy > scored["majority_class_rate"] + 1e-12
                ),
                "clears_breakeven": {
                    name: bool(best_accuracy > threshold)
                    for name, threshold in thresholds.items()
                },
                **scored,
            }
        )

    return {
        "half_spread": half_spread,
        "seconds_per_tick": seconds_per_tick,
        "round_trip_costs": costs,
        "horizons": rows,
    }


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 90:
        return f"{seconds:.1f} giây"
    if seconds < 5400:
        return f"{seconds / 60:.1f} phút"
    return f"{seconds / 3600:.1f} giờ"


def build_verdict(analysis: dict[str, Any]) -> str:
    scored = [row for row in analysis["horizons"] if "skipped" not in row]
    if not scored:
        return (
            "Không horizon nào có đủ cửa sổ không chồng lấp để đánh giá. Đây tự nó "
            "đã là một kết quả: ở horizon giao dịch được, dữ liệu hiện có **không "
            "đủ** để kết luận."
        )
    tradable = [
        row
        for row in scored
        if row["clears_breakeven"].get("maker_futures_2bps")
    ]
    beat_majority = [row for row in scored if row["beats_majority"]]
    parts = [
        f"Đánh giá {len(scored)} horizon trên các cửa sổ **không chồng lấp**."
    ]
    if not beat_majority:
        parts.append(
            "**Không mô hình nào vượt được baseline đoán theo lớp đa số** ở bất kỳ "
            "horizon nào — tức chưa quan sát được kỹ năng dự báo hướng, chứ chưa "
            "nói tới việc đủ bù chi phí."
        )
    else:
        best = max(beat_majority, key=lambda row: row["best_accuracy"])
        parts.append(
            f"Có {len(beat_majority)}/{len(scored)} horizon mà mô hình tốt nhất vượt "
            f"baseline đa số; cao nhất là h={best['horizon_ticks']:,} "
            f"({best['best_model']}, {best['best_accuracy'] * 100:.1f}% so với "
            f"{best['majority_class_rate'] * 100:.1f}%)."
        )
    if tradable:
        best = max(tradable, key=lambda row: row["best_accuracy"])
        parts.append(
            f"**{len(tradable)} horizon vượt ngưỡng hoà vốn maker 2bps**, tốt nhất là "
            f"h={best['horizon_ticks']:,} ({_format_seconds(best['seconds'])})."
        )
    else:
        parts.append(
            "**Không horizon nào đạt ngưỡng hoà vốn** kể cả ở mức phí maker 2bps."
        )
    # The shape of the result matters more than any single number: skill and
    # payoff sit at opposite ends of the horizon range.
    skilled = [row for row in scored if row["beats_majority"]]
    if skilled:
        sharpest = max(skilled, key=lambda row: row["best_accuracy"])
        payoff = sharpest["net_edge_per_trade"]["maker_futures_2bps"]
        parts.append(
            f"Điểm cốt lõi: **kỹ năng dự báo và khả năng sinh lời nằm ở hai đầu "
            f"đối lập của thang horizon**. Ở h={sharpest['horizon_ticks']:,} độ "
            f"chính xác cao nhất ({sharpest['best_accuracy'] * 100:.1f}%) nhưng "
            f"biên độ giá quá nhỏ nên lãi ròng vẫn là **{payoff * 1e4:+.2f} bps/lệnh**; "
            "ở horizon dài, biên độ đủ lớn thì kỹ năng lại biến mất."
        )
    smallest = min(row["n_test"] for row in scored)
    parts.append(
        f"Cỡ mẫu kiểm định nhỏ nhất chỉ {smallest} cửa sổ — mọi con số ở đây có "
        "khoảng tin cậy rất rộng và không được coi là kết luận."
    )
    return " ".join(parts)


def render_markdown(audit: dict[str, Any]) -> str:
    analysis = audit["analysis"]
    lines = [
        f"# Có edge ở horizon giao dịch được không? — {audit['label']}",
        "",
        "**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — nhãn dự báo mới, "
        "không phải kết quả confirmatory.",
        "",
        f"- Feature file: `{Path(audit['feature_path']).name}` ({audit['rows']:,} dòng)",
        f"- Git commit: `{audit['git_commit']}` · Python {audit['python']}",
        f"- Cửa sổ **không chồng lấp** (mỗi {audit['label']} nhãn cách nhau đúng "
        "`horizon` tick nên không chia sẻ tương lai)",
        "",
        "## Độ chính xác đạt được so với ngưỡng cần có",
        "",
        "| Horizon | Thời gian | Cửa sổ (train/test) | Lớp đa số | Mô hình tốt nhất | "
        "Độ chính xác | Ngưỡng maker 2bps | Lãi ròng/lệnh | Đạt? |",
        "|---:|---:|---:|---:|---|---:|---:|---:|:--:|",
    ]
    for row in analysis["horizons"]:
        if "skipped" in row:
            lines.append(
                f"| {row['horizon_ticks']:,} | — | không đủ mẫu | — | — | — | — | — | — |"
            )
            continue
        threshold = row["breakeven_accuracy"]["maker_futures_2bps"]
        threshold_text = "—" if threshold >= 1.0 else f"{threshold * 100:.1f}%"
        clears = "✅" if row["clears_breakeven"]["maker_futures_2bps"] else "✘"
        net = row["net_edge_per_trade"]["maker_futures_2bps"]
        lines.append(
            f"| {row['horizon_ticks']:,} | {_format_seconds(row['seconds'])} | "
            f"{row['n_train']}/{row['n_test']} | "
            f"{row['majority_class_rate'] * 100:.1f}% | {row['best_model']} | "
            f"{row['best_accuracy'] * 100:.1f}% | {threshold_text} | "
            f"{net * 1e4:+.2f} bps | {clears} |"
        )

    lines += ["", "## Chi tiết từng mô hình", ""]
    for row in analysis["horizons"]:
        if "skipped" in row:
            continue
        lines += [
            f"### Horizon {row['horizon_ticks']:,} ({_format_seconds(row['seconds'])})",
            "",
            "| Mô hình | Độ chính xác | Hơn lớp đa số? |",
            "|---|---:|:--:|",
        ]
        for name, metrics in sorted(
            row["models"].items(), key=lambda item: -item[1]["accuracy"]
        ):
            better = "✅" if metrics["accuracy"] > row["majority_class_rate"] else "✘"
            lines.append(f"| {name} | {metrics['accuracy'] * 100:.1f}% | {better} |")
        lines.append("")

    lines += ["## Kết luận", "", audit["verdict"], ""]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        default="data/assets/btcusdt/features/features_BTCUSDT_multiday.parquet",
    )
    parser.add_argument("--label", default="BTCUSDT")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--horizons", default="1000,5000,10000,50000")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = (ROOT / args.feature_path).resolve()
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())

    frame = _load(feature_path, args.max_rows)
    print(f"[edge] {args.label}: {len(frame):,} dòng từ {feature_path.name}")
    analysis = analyse(frame, horizons, args.train_fraction)

    for row in analysis["horizons"]:
        if "skipped" in row:
            print(f"[edge] h={row['horizon_ticks']:>7,}  BỎ QUA: {row['skipped']}")
            continue
        threshold = row["breakeven_accuracy"]["maker_futures_2bps"]
        print(
            f"[edge] h={row['horizon_ticks']:>7,}  n={row['n_test']:>5}  "
            f"best={row['best_accuracy'] * 100:5.1f}% ({row['best_model']})  "
            f"majority={row['majority_class_rate'] * 100:5.1f}%  "
            f"need={'--' if threshold >= 1 else f'{threshold * 100:.1f}%'}"
        )

    verdict = build_verdict(analysis)
    print(f"[edge] VERDICT: {verdict}")

    audit = {
        "kind": "horizon_label_baselines",
        "status": "EXPLORATORY_ONLY_NOT_CONFIRMATORY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "label": args.label,
        "feature_path": canonical_repo_path(feature_path, ROOT),
        "feature_sha256": sha256_file(feature_path),
        "rows": int(len(frame)),
        "train_fraction": args.train_fraction,
        "analysis": analysis,
        "verdict": verdict,
    }

    json_out = (
        ROOT / (args.json_out or f"reports/research/horizon_edge_{args.label}.json")
    ).resolve()
    md_out = (
        ROOT / (args.md_out or f"reports/research/horizon_edge_{args.label}.md")
    ).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(f"[edge] wrote {json_out}")
    print(f"[edge] wrote {md_out}")


if __name__ == "__main__":
    main()

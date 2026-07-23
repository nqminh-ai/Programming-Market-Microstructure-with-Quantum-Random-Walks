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
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest

from src.data.feature_store import load_feature_columns
from src.evaluation.directional_baselines import (
    FEATURE_NAMES,
    _fit_logistic,
    _pairwise,
)
from src.evaluation.provenance import canonical_repo_path, sha256_file
from scripts.research.horizon_feasibility import (
    FEE_SCENARIOS,
    breakeven_accuracy,
    measure_half_spread,
    round_trip_cost,
    seconds_per_tick as measure_seconds_per_tick,
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
    # Already the stored types; naming them keeps the frame's dtypes fixed even
    # if a future store widens them, and makes an id too large for int32 an
    # error rather than a silent fallback to int64.
    "segment_id": "int32",
    "obi_valid": "bool",
}


def _load(path: Path, max_rows: int) -> pd.DataFrame:
    """Load only the needed columns and downcast."""
    return load_feature_columns(
        path, NEEDED_COLUMNS, downcast=DOWNCAST, max_rows=max_rows
    )


def build_horizon_events(frame: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    """Causal features at ``t`` against the sign of the return over ``t -> t+h``.

    Anchors are spaced ``horizon`` apart so that no two labels share any future
    return, and both endpoints must lie in the same contiguous segment.

    Only the rows the anchors actually touch are materialised. Building the
    five feature columns at full length first costs 5 x 8 bytes per row -- 10GB
    on a 69-day store of ~250M rows for that one array, on top of a float64
    copy of every source column it is built from -- to then keep one row in
    every ``horizon``. At h=50,000 that is 0.002% of what was allocated.
    """
    n_rows = len(frame)
    anchors = np.arange(0, n_rows - horizon, horizon, dtype=np.int64)
    if anchors.size == 0:
        raise ValueError(f"horizon {horizon} exceeds the available rows")
    future = anchors + horizon

    price_column = frame["price"].to_numpy()
    obi_column = frame["obi"].to_numpy()
    direction_column = frame["tick_direction"].to_numpy()

    obi = obi_column[anchors].astype(np.float64)
    direction = direction_column[anchors].astype(np.float64)
    intensity = frame["trade_intensity"].to_numpy()[anchors].astype(np.float64)

    # Anchor 0 has no predecessor; clamping keeps the gather in bounds and the
    # value is discarded by the mask below.
    previous = np.maximum(anchors - 1, 0)
    if "segment_id" in frame.columns:
        segment_column = frame["segment_id"].to_numpy(copy=False)
        segment_at = segment_column[anchors]
        segment_future = segment_column[future]
        segment_previous = segment_column[previous]
    else:
        segment_at = np.zeros(anchors.size, dtype=np.int64)
        segment_future = segment_at
        segment_previous = segment_at

    # obi_change[t] = obi[t] - obi[t-1], zero at t = 0 and across a segment break.
    obi_change = obi - obi_column[previous].astype(np.float64)
    obi_change[anchors == 0] = 0.0
    obi_change[segment_previous != segment_at] = 0.0

    features = np.column_stack(
        [obi, direction, obi_change, np.abs(obi), np.log1p(np.maximum(intensity, 0.0))]
    )

    # lagged[i, k] = direction[anchor_i - (k + 1)], zero before the start.
    lagged = np.zeros((anchors.size, 5), dtype=np.float64)
    for lag in range(1, 6):
        source = anchors - lag
        in_range = source >= 0
        lagged[in_range, lag - 1] = direction_column[source[in_range]].astype(np.float64)

    price_at = price_column[anchors].astype(np.float64)
    price_future = price_column[future].astype(np.float64)

    usable = (
        (segment_at == segment_future)
        & np.isfinite(price_at)
        & np.isfinite(price_future)
        & (price_at > 0)
        & (price_future > 0)
        & np.isfinite(features).all(axis=1)
    )
    if "obi_valid" in frame.columns:
        usable &= frame["obi_valid"].to_numpy()[anchors].astype(bool)

    log_return = np.full(anchors.size, np.nan)
    log_return[usable] = np.log(price_future[usable] / price_at[usable])
    # A window that ends exactly where it began carries no directional
    # information and would otherwise be scored as a "down" label.
    keep = usable & (np.abs(log_return) > 1e-12)

    return {
        "features": features[keep],
        "lagged": lagged[keep],
        "target": (log_return[keep] > 0.0).astype(np.float64),
        "log_return": log_return[keep],
        "timestamp": frame["timestamp"].to_numpy()[anchors[keep]],
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
    # Shared with the feasibility study rather than recomputed: that copy is
    # endpoint-based, so it does not convert the whole timestamp column, and a
    # second implementation of the same thing is a second chance to get the
    # unit inference wrong -- which has already happened once here.
    seconds_per_tick = measure_seconds_per_tick(frame)

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
        # A point estimate above a threshold is not evidence of being above it.
        # De-overlapping leaves a few hundred windows at the long horizons, and
        # at n=323 an accuracy of 53.6% against a 52.3% break-even carries
        # p=0.35 -- its interval also covers a coin flip. Reporting that as
        # "clears break-even" is precisely the overstatement this study exists
        # to avoid, so the comparison is accompanied by a one-sided binomial
        # test and the verdict speaks only of results that survive it.
        n_test = int(scored["n_test"])
        successes = int(round(best_accuracy * n_test))
        interval = binomtest(successes, n_test).proportion_ci(0.95, method="wilson")
        clears = {
            name: bool(best_accuracy > threshold)
            for name, threshold in thresholds.items()
        }
        clears_significantly = {}
        p_values = {}
        for name, threshold in thresholds.items():
            if threshold is None or not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
                clears_significantly[name] = False
                p_values[name] = None
                continue
            p_value = float(
                binomtest(successes, n_test, threshold, alternative="greater").pvalue
            )
            p_values[name] = p_value
            clears_significantly[name] = bool(clears[name] and p_value < 0.05)

        rows.append(
            {
                "horizon_ticks": int(horizon),
                "seconds": None if seconds_per_tick is None else seconds_per_tick * horizon,
                "expected_abs_move": expected_move,
                "breakeven_accuracy": thresholds,
                "net_edge_per_trade": net_edge,
                "best_model": best_name,
                "best_accuracy": best_accuracy,
                "best_accuracy_ci95": [float(interval.low), float(interval.high)],
                "beats_majority": bool(
                    best_accuracy > scored["majority_class_rate"] + 1e-12
                ),
                "clears_breakeven": clears,
                "clears_breakeven_p_value": p_values,
                "clears_breakeven_significant": clears_significantly,
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
        if row["clears_breakeven_significant"].get("maker_futures_2bps")
    ]
    # Above the threshold by eye but not by test: worth naming, because leaving
    # it out of the verdict entirely invites someone to rediscover it in the
    # table and read it as a finding.
    borderline = [
        row
        for row in scored
        if row["clears_breakeven"].get("maker_futures_2bps")
        and not row["clears_breakeven_significant"].get("maker_futures_2bps")
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
        p_value = best["clears_breakeven_p_value"]["maker_futures_2bps"]
        parts.append(
            f"**{len(tradable)} horizon vượt ngưỡng hoà vốn maker 2bps có ý nghĩa "
            f"thống kê**, tốt nhất là h={best['horizon_ticks']:,} "
            f"({_format_seconds(best['seconds'])}, p={p_value:.3f})."
        )
    else:
        parts.append(
            "**Không horizon nào vượt ngưỡng hoà vốn** kể cả ở mức phí maker 2bps."
        )
    if borderline:
        worst = min(
            borderline,
            key=lambda row: row["clears_breakeven_p_value"]["maker_futures_2bps"],
        )
        low, high = worst["best_accuracy_ci95"]
        parts.append(
            f"Có {len(borderline)} horizon *nhìn* như vượt ngưỡng nhưng **không "
            f"qua được kiểm định**: h={worst['horizon_ticks']:,} đạt "
            f"{worst['best_accuracy'] * 100:.1f}% so với ngưỡng "
            f"{worst['breakeven_accuracy']['maker_futures_2bps'] * 100:.1f}% trên "
            f"{worst['n_test']:,} cửa sổ, p="
            f"{worst['clears_breakeven_p_value']['maker_futures_2bps']:.3f}, khoảng "
            f"tin cậy 95% [{low * 100:.1f}%, {high * 100:.1f}%] — vẫn chứa cả mức "
            f"tung đồng xu. Chênh lệch đó không phân biệt được với nhiễu."
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
        "Độ chính xác | KTC 95% | Ngưỡng maker 2bps | Lãi ròng/lệnh | Đạt? |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|:--:|",
    ]
    for row in analysis["horizons"]:
        if "skipped" in row:
            lines.append(
                f"| {row['horizon_ticks']:,} | — | không đủ mẫu | — | — | — | — | — | — | — |"
            )
            continue
        threshold = row["breakeven_accuracy"]["maker_futures_2bps"]
        threshold_text = "—" if threshold >= 1.0 else f"{threshold * 100:.1f}%"
        # A tick here means the point estimate cleared the threshold *and* a
        # one-sided binomial test says the sample can tell the two apart.
        if row["clears_breakeven_significant"]["maker_futures_2bps"]:
            clears = "✅"
        elif row["clears_breakeven"]["maker_futures_2bps"]:
            p_value = row["clears_breakeven_p_value"]["maker_futures_2bps"]
            clears = f"⚠ p={p_value:.2f}"
        else:
            clears = "✘"
        net = row["net_edge_per_trade"]["maker_futures_2bps"]
        low, high = row["best_accuracy_ci95"]
        lines.append(
            f"| {row['horizon_ticks']:,} | {_format_seconds(row['seconds'])} | "
            f"{row['n_train']}/{row['n_test']} | "
            f"{row['majority_class_rate'] * 100:.1f}% | {row['best_model']} | "
            f"{row['best_accuracy'] * 100:.1f}% | "
            f"[{low * 100:.1f}, {high * 100:.1f}]% | {threshold_text} | "
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
    # The progress lines are Vietnamese, and a Windows console defaults to
    # cp1252, which cannot encode them. Without this the run dies on its first
    # print -- after the several minutes it takes to load a 200M-row store, and
    # with a traceback about a codec rather than anything to do with the study.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

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

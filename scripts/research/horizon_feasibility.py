"""Which forecast horizons can pay their own trading costs?

The project forecasts the direction of the next tick. On BTCUSDT the average
absolute one-tick move is about 5e-7 while a taker round trip costs about 1e-3,
so a perfect one-tick forecaster still loses money: the horizon, not the model,
is what makes the strategy unprofitable. This script finds, per asset and per
fee scenario, the shortest horizon at which the typical move is large enough to
cover costs, and the directional accuracy that would then be required.

Method
------
For horizon ``h`` measured in ticks, the expected absolute log return
``E|r_h|`` is computed within contiguous segments only (never across a data
gap). A symmetric directional bet with hit probability ``p`` earns
``(2p - 1) * E|r_h|`` before costs, so break-even requires

    p > 0.5 + cost_round_trip / (2 * E|r_h|)

which exceeds 1 -- unattainable at any accuracy -- whenever the round trip
costs more than twice the typical move.

The effective half-spread is *measured* from ``|price - mid_price| / mid_price``
rather than assumed, and is added to taker costs and subtracted from maker
costs. Note that ``mid_price`` here is a derived reference price, not a true L2
order-book mid, so the spread term is an approximation; it is small relative to
the fees either way.

Status: exploratory. A horizon clearing its costs is necessary, not sufficient:
it says the move is big enough to pay the toll, not that anything in this
project can predict its direction at that horizon.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.common import timestamps_to_nanoseconds
from src.evaluation.provenance import canonical_repo_path, sha256_file

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HORIZONS = (1, 10, 100, 1_000, 5_000, 10_000, 50_000, 100_000, 200_000)

# Round-trip fee in basis points, excluding spread. Binance futures list 4bps
# taker / 2bps maker at base tier; the repository's signal engine uses 5bps per
# side. Both directions are charged, hence the doubling.
FEE_SCENARIOS: dict[str, dict[str, Any]] = {
    "taker_repo_5bps": {
        "fee_bps_per_side": 5.0,
        "crosses_spread": True,
        "short": "Taker 5bps",
        "label": "Taker, 5bps/chiều (mức signal engine đang dùng)",
    },
    "taker_futures_4bps": {
        "fee_bps_per_side": 4.0,
        "crosses_spread": True,
        "short": "Taker 4bps",
        "label": "Taker, 4bps/chiều (Binance futures base)",
    },
    "maker_futures_2bps": {
        "fee_bps_per_side": 2.0,
        "crosses_spread": False,
        "short": "Maker 2bps",
        "label": "Maker, 2bps/chiều (đặt lệnh chờ, ăn spread)",
    },
    "maker_rebate_0bps": {
        "fee_bps_per_side": 0.0,
        "crosses_spread": False,
        "short": "Maker 0bps",
        "label": "Maker, 0bps/chiều (bậc phí ưu đãi nhất)",
    },
}

# Above this, a horizon is treated as not realistically tradable: sustained
# directional accuracy beyond it is not observed in liquid markets.
PLAUSIBLE_ACCURACY_CEILING = 0.60

NEEDED_COLUMNS = ["timestamp", "price", "mid_price", "segment_id"]


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


def _load(path: Path, max_rows: int) -> pd.DataFrame:
    names = set(pq.ParquetFile(path).schema.names)
    columns = [column for column in NEEDED_COLUMNS if column in names]
    frame = pd.read_parquet(path, columns=columns)
    if max_rows and len(frame) > max_rows:
        frame = frame.iloc[:max_rows]
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return frame


def measure_half_spread(frame: pd.DataFrame) -> float | None:
    """Mean ``|price - mid| / mid`` as a fraction, or None when unavailable."""
    if "mid_price" not in frame.columns:
        return None
    price = frame["price"].to_numpy(dtype=float)
    mid = frame["mid_price"].to_numpy(dtype=float)
    valid = np.isfinite(price) & np.isfinite(mid) & (mid > 0)
    if not valid.any():
        return None
    return float(np.abs(price[valid] - mid[valid]).mean() / mid[valid].mean())


def seconds_per_tick(frame: pd.DataFrame) -> float | None:
    """Wall-clock seconds per tick.

    Unit detection is delegated to ``timestamps_to_nanoseconds``, which keys off
    the magnitude of the epoch value. Inferring it from the observed span
    instead is ambiguous -- a nanosecond feed spanning 55 hours reads as a
    plausible 6-year span if misread as microseconds, which silently turned
    41 minutes into 28 days here.
    """
    try:
        nanoseconds = timestamps_to_nanoseconds(frame["timestamp"]).to_numpy(dtype="int64")
    except (ValueError, TypeError, OverflowError):
        return None
    if nanoseconds.size < 2:
        return None
    span_seconds = float(nanoseconds.max() - nanoseconds.min()) / 1e9
    if span_seconds <= 0:
        return None
    # n points span n-1 intervals.
    return span_seconds / (nanoseconds.size - 1)


def expected_absolute_move(frame: pd.DataFrame, horizon: int) -> tuple[float, int]:
    """Mean ``|log(P_{t+h} / P_t)|`` over pairs inside one contiguous segment."""
    price = frame["price"].to_numpy(dtype=float)
    if horizon >= len(price):
        return float("nan"), 0
    start, end = price[:-horizon], price[horizon:]
    usable = np.isfinite(start) & np.isfinite(end) & (start > 0) & (end > 0)
    if "segment_id" in frame.columns:
        segment = frame["segment_id"].to_numpy()
        usable &= segment[:-horizon] == segment[horizon:]
    if not usable.any():
        return float("nan"), 0
    moves = np.abs(np.log(end[usable] / start[usable]))
    return float(moves.mean()), int(usable.sum())


def round_trip_cost(scenario: dict[str, Any], half_spread: float | None) -> float:
    """Total round-trip cost as a fraction of notional."""
    fee = 2.0 * float(scenario["fee_bps_per_side"]) * 1e-4
    if half_spread is None:
        return fee
    # A taker pays the half-spread on entry and exit; a resting maker order is
    # filled at the passive side and earns it instead.
    spread_term = 2.0 * half_spread
    return fee + spread_term if scenario["crosses_spread"] else fee - spread_term


def breakeven_accuracy(cost: float, expected_move: float) -> float:
    """Directional hit rate needed to cover ``cost``; may exceed 1.0."""
    if not np.isfinite(expected_move) or expected_move <= 0:
        return float("inf")
    if cost <= 0:
        return 0.5
    return 0.5 + cost / (2.0 * expected_move)


def analyse(frame: pd.DataFrame, horizons: tuple[int, ...]) -> dict[str, Any]:
    half_spread = measure_half_spread(frame)
    per_tick = seconds_per_tick(frame)
    costs = {
        name: round_trip_cost(scenario, half_spread)
        for name, scenario in FEE_SCENARIOS.items()
    }

    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        move, pairs = expected_absolute_move(frame, horizon)
        if not np.isfinite(move):
            continue
        entry: dict[str, Any] = {
            "horizon_ticks": int(horizon),
            "pairs": pairs,
            "expected_abs_move": move,
            "seconds": None if per_tick is None else per_tick * horizon,
            "scenarios": {},
        }
        for name, cost in costs.items():
            accuracy = breakeven_accuracy(cost, move)
            # A negative round trip means captured spread exceeds fees, and the
            # break-even model then claims profit at any accuracy. That is an
            # artefact: a resting limit order is filled preferentially when the
            # market is moving against it, and this analysis carries no adverse
            # selection term. Such scenarios are flagged, not called tradable.
            spread_dominates = cost <= 0
            entry["scenarios"][name] = {
                "round_trip_cost": cost,
                "move_to_cost_ratio": move / cost if cost > 0 else float("inf"),
                "breakeven_accuracy": accuracy,
                "requires_adverse_selection_model": bool(spread_dominates),
                "tradable": bool(
                    not spread_dominates and accuracy <= PLAUSIBLE_ACCURACY_CEILING
                ),
            }
        rows.append(entry)

    minimum_viable = {}
    for name in costs:
        viable = [r["horizon_ticks"] for r in rows if r["scenarios"][name]["tradable"]]
        minimum_viable[name] = min(viable) if viable else None

    return {
        "half_spread": half_spread,
        "seconds_per_tick": per_tick,
        "round_trip_costs": costs,
        "accuracy_ceiling": PLAUSIBLE_ACCURACY_CEILING,
        "horizons": rows,
        "minimum_viable_horizon": minimum_viable,
    }


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 90:
        return f"{seconds:.1f} giây"
    if seconds < 5400:
        return f"{seconds / 60:.1f} phút"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} giờ"
    return f"{seconds / 86400:.1f} ngày"


def render_markdown(audit: dict[str, Any]) -> str:
    analysis = audit["analysis"]
    lines = [
        f"# Khả thi giao dịch theo horizon — {audit['label']}",
        "",
        "**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — phân tích khả thi chi phí, "
        "không phải bằng chứng có lợi nhuận.",
        "",
        f"- Feature file: `{Path(audit['feature_path']).name}` ({audit['rows']:,} dòng)",
        f"- Git commit: `{audit['git_commit']}` · Python {audit['python']}",
    ]
    half_spread = analysis["half_spread"]
    if half_spread is not None:
        lines.append(
            f"- Half-spread **đo được** từ dữ liệu: {half_spread * 1e4:.3f} bps "
            "(|price − mid| / mid)"
        )
    per_tick = analysis["seconds_per_tick"]
    if per_tick:
        lines.append(f"- Nhịp giao dịch: {1 / per_tick:.1f} tick/giây")
    lines += ["", "## Chi phí một vòng mua-bán", "",
              "| Kịch bản | Chi phí vòng |", "|---|---:|"]
    for name, scenario in FEE_SCENARIOS.items():
        cost = analysis["round_trip_costs"][name]
        lines.append(f"| {scenario['label']} | {cost * 1e4:+.2f} bps |")

    lines += [
        "",
        "## Độ chính xác hướng cần có để hoà vốn",
        "",
        "`p > 0,5 + chi_phí / (2·E|biến động|)`. Ô ✅ nghĩa là ngưỡng hoà vốn nằm "
        f"dưới {analysis['accuracy_ceiling'] * 100:.0f}% — mức còn có thể bàn tới. "
        "`—` nghĩa là không thể hoà vốn dù dự đoán đúng 100%. "
        "`⚠ spread` nghĩa là spread thu được lớn hơn phí, khiến công thức trên "
        "kết luận có lãi ở **mọi** độ chính xác; đó là ảo giác của mô hình vì nó "
        "**chưa tính adverse selection** — lệnh chờ có xu hướng được khớp đúng "
        "lúc thị trường đi ngược lại bạn. Muốn dùng kịch bản đó phải mô hình hoá "
        "hàng đợi lệnh và tỉ lệ khớp bằng dữ liệu L2 thật.",
        "",
        "| Horizon | Thời gian | E\\|biến động\\| | "
        + " | ".join(FEE_SCENARIOS[n]["short"] for n in FEE_SCENARIOS)
        + " |",
        "|---:|---:|---:|" + "---:|" * len(FEE_SCENARIOS),
    ]
    for row in analysis["horizons"]:
        cells = []
        for name in FEE_SCENARIOS:
            entry = row["scenarios"][name]
            accuracy = entry["breakeven_accuracy"]
            if entry["requires_adverse_selection_model"]:
                cells.append("⚠ spread")
            elif not np.isfinite(accuracy) or accuracy >= 1.0:
                cells.append("—")
            else:
                mark = "✅" if entry["tradable"] else ""
                cells.append(f"{accuracy * 100:.1f}% {mark}".strip())
        lines.append(
            f"| {row['horizon_ticks']:,} | {_format_seconds(row['seconds'])} | "
            f"{row['expected_abs_move']:.2e} | " + " | ".join(cells) + " |"
        )

    lines += ["", "## Horizon nhỏ nhất còn giao dịch được", "",
              "| Kịch bản | Horizon | Thời gian |", "|---|---:|---:|"]
    by_horizon = {r["horizon_ticks"]: r for r in analysis["horizons"]}
    for name, scenario in FEE_SCENARIOS.items():
        horizon = analysis["minimum_viable_horizon"][name]
        if horizon is None:
            lines.append(f"| {scenario['label']} | không có | — |")
        else:
            lines.append(
                f"| {scenario['label']} | {horizon:,} tick | "
                f"{_format_seconds(by_horizon[horizon]['seconds'])} |"
            )

    lines += ["", "## Kết luận", "", audit["verdict"], ""]
    return "\n".join(lines)


def build_verdict(analysis: dict[str, Any], label: str) -> str:
    one_tick = next(
        (r for r in analysis["horizons"] if r["horizon_ticks"] == 1), None
    )
    parts: list[str] = []
    if one_tick is not None:
        ratio = one_tick["scenarios"]["taker_repo_5bps"]["move_to_cost_ratio"]
        parts.append(
            f"Ở horizon 1 tick, biến động trung bình chỉ bằng **{ratio:.4f} lần** "
            "chi phí một vòng taker — nghĩa là **một mô hình dự đoán đúng 100% "
            "vẫn lỗ**. Đây là giới hạn của horizon, không phải của mô hình."
        )
    viable = analysis["minimum_viable_horizon"]
    taker = viable.get("taker_repo_5bps")
    maker = viable.get("maker_futures_2bps")
    by_horizon = {r["horizon_ticks"]: r for r in analysis["horizons"]}
    if maker is not None:
        parts.append(
            f"Đặt lệnh chờ (maker, 2bps/chiều) trở nên khả thi từ **{maker:,} tick "
            f"(~{_format_seconds(by_horizon[maker]['seconds'])})**."
        )
    else:
        parts.append(
            "Không horizon nào trong lưới đạt ngưỡng khả thi kể cả ở mức phí maker."
        )
    if taker is not None:
        parts.append(
            f"Giao dịch chủ động (taker, 5bps/chiều) cần tới **{taker:,} tick "
            f"(~{_format_seconds(by_horizon[taker]['seconds'])})**."
        )
    else:
        parts.append(
            "Ở mức phí taker 5bps/chiều, **không horizon nào** trong lưới đạt "
            "ngưỡng khả thi."
        )
    parts.append(
        "Cần nhấn mạnh: vượt ngưỡng chi phí là điều kiện **cần, không đủ**. Bảng "
        "này chỉ nói biến động đủ lớn để trả phí; nó **không** nói dự án có khả "
        "năng dự đoán đúng hướng ở horizon đó — điều chưa từng được chứng minh."
    )
    return " ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        default="data/assets/btcusdt/features/features_BTCUSDT_recent_subset.parquet",
    )
    parser.add_argument("--label", default="BTCUSDT")
    parser.add_argument("--max-rows", type=int, default=4_000_000)
    parser.add_argument(
        "--horizons",
        default=",".join(str(value) for value in DEFAULT_HORIZONS),
    )
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
    print(f"[horizon] {args.label}: {len(frame):,} dòng từ {feature_path.name}")
    analysis = analyse(frame, horizons)

    half_spread = analysis["half_spread"]
    if half_spread is not None:
        print(f"[horizon] half-spread đo được: {half_spread * 1e4:.3f} bps")
    for row in analysis["horizons"]:
        taker = row["scenarios"]["taker_repo_5bps"]["breakeven_accuracy"]
        maker = row["scenarios"]["maker_futures_2bps"]["breakeven_accuracy"]
        print(
            f"[horizon] h={row['horizon_ticks']:>7,}  E|r|={row['expected_abs_move']:.2e}"
            f"  taker={'--' if taker >= 1 else f'{taker * 100:.1f}%'}"
            f"  maker={'--' if maker >= 1 else f'{maker * 100:.1f}%'}"
        )

    verdict = build_verdict(analysis, args.label)
    print(f"[horizon] VERDICT: {verdict}")

    audit = {
        "kind": "horizon_feasibility",
        "status": "EXPLORATORY_ONLY_NOT_CONFIRMATORY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "label": args.label,
        "feature_path": canonical_repo_path(feature_path, ROOT),
        "feature_sha256": sha256_file(feature_path),
        "rows": int(len(frame)),
        "fee_scenarios": FEE_SCENARIOS,
        "analysis": analysis,
        "verdict": verdict,
    }

    json_out = (
        ROOT / (args.json_out or f"reports/research/horizon_feasibility_{args.label}.json")
    ).resolve()
    md_out = (
        ROOT / (args.md_out or f"reports/research/horizon_feasibility_{args.label}.md")
    ).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(f"[horizon] wrote {json_out}")
    print(f"[horizon] wrote {md_out}")


if __name__ == "__main__":
    main()

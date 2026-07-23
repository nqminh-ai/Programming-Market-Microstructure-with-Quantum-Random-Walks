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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.common import timestamps_to_nanoseconds
from src.data.feature_store import load_feature_columns, load_trade_sign
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
    # No downcast here: price and mid_price are differenced against each other
    # at tick scale, and segment_id is only compared for equality.
    frame = load_feature_columns(path, NEEDED_COLUMNS, max_rows=max_rows)
    # The aggressor side, as +1/-1. Loaded apart from the frame because it is
    # stored as a string, and 227M of those cost over 11GB once pandas turns
    # them into objects.
    sign = load_trade_sign(path, max_rows)
    if sign is not None and len(sign) == len(frame):
        frame["trade_sign"] = sign
    return frame


def roll_half_spread(frame: pd.DataFrame, chunk: int = 10_000_000) -> float | None:
    """Roll (1984) effective half-spread as a fraction of price.

    Bid-ask bounce makes consecutive trade-price changes negatively correlated:
    a trade at the ask followed by one at the bid moves the price down by the
    spread and back up again. Roll shows the full spread is
    ``2 * sqrt(-cov(dp_t, dp_{t-1}))``, so on log prices the half-spread is
    simply ``sqrt(-cov)``. It needs only trade prices, which is what this
    project has -- there is no order book here to read a quoted spread from.

    Returns ``None`` when the covariance is non-negative. That happens when
    trending dominates bouncing and the estimator has no real root; reporting
    nothing is correct, and inventing a spread from a positive covariance is
    not.

    Differences are taken within a segment only, and a pair is used only when
    all three prices behind it lie in the same segment, so no gap is crossed.
    """
    price_column = frame["price"].to_numpy()
    rows = len(price_column)
    if rows < 4:
        return None
    segment_column = (
        frame["segment_id"].to_numpy() if "segment_id" in frame.columns else None
    )

    count = 0
    sum_x = sum_y = sum_xy = 0.0
    # Overlap by two rows so pairs straddling a chunk edge are still formed.
    step = max(chunk, 3)
    for begin in range(0, rows - 1, step - 2):
        stop = min(begin + step, rows)
        if stop - begin < 3:
            break
        price = price_column[begin:stop].astype(np.float64, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_price = np.log(price)
        difference = np.diff(log_price)
        current, previous = difference[1:], difference[:-1]
        usable = np.isfinite(current) & np.isfinite(previous)
        if segment_column is not None:
            segment = segment_column[begin:stop]
            usable &= (segment[2:] == segment[1:-1]) & (segment[1:-1] == segment[:-2])
        if not usable.any():
            continue
        x, y = current[usable], previous[usable]
        count += x.size
        sum_x += float(x.sum())
        sum_y += float(y.sum())
        sum_xy += float((x * y).sum())

    if count < 3:
        return None
    covariance = (sum_xy - sum_x * sum_y / count) / (count - 1)
    if covariance >= 0.0:
        return None
    return float(np.sqrt(-covariance))


def measure_half_spread(frame: pd.DataFrame, chunk: int = 10_000_000) -> float | None:
    """Mean ``|price - mid| / mid`` as a fraction, or None when unavailable.

    NOTE: despite the name this is **not** a bid-ask half-spread, and it is no
    longer used as one. ``mid_price`` in these feature stores is a trailing
    100-trade VWAP, not an order-book mid, so this measures how far a trade
    price sits from a lagging average of recent trade prices -- short-horizon
    price dispersion. On BTCUSDT it reads 0.28bps, about 175 ticks, where Roll
    puts the half-spread at 0.0085bps. It is kept and reported because the
    published cost model used it, so the correction has to be auditable.

    Accumulated in chunks. Doing it in one shot casts both columns to float64
    at full length and then takes a boolean-indexed copy of each, which on a
    69-day store of ~250M rows is roughly 8GB of transient allocation to
    produce one scalar. The value is unchanged: the ratio of the two means is
    the ratio of the two sums, since both are over the same rows.
    """
    if "mid_price" not in frame.columns:
        return None
    price_column = frame["price"].to_numpy()
    mid_column = frame["mid_price"].to_numpy()

    deviation_sum = 0.0
    mid_sum = 0.0
    count = 0
    for start in range(0, len(frame), chunk):
        stop = min(start + chunk, len(frame))
        price = price_column[start:stop].astype(np.float64, copy=False)
        mid = mid_column[start:stop].astype(np.float64, copy=False)
        valid = np.isfinite(price) & np.isfinite(mid) & (mid > 0)
        if not valid.any():
            continue
        deviation_sum += float(np.abs(price[valid] - mid[valid]).sum())
        mid_sum += float(mid[valid].sum())
        count += int(valid.sum())

    if count == 0 or mid_sum <= 0.0:
        return None
    return deviation_sum / mid_sum


def seconds_per_tick(frame: pd.DataFrame) -> float | None:
    """Wall-clock seconds per tick.

    Unit detection is delegated to ``timestamps_to_nanoseconds``, which keys off
    the magnitude of the epoch value. Inferring it from the observed span
    instead is ambiguous -- a nanosecond feed spanning 55 hours reads as a
    plausible 6-year span if misread as microseconds, which silently turned
    41 minutes into 28 days here.
    """
    column = frame["timestamp"]
    if len(column) < 2:
        return None
    try:
        # Only the two extremes are needed. Converting the whole column costs
        # several full-length copies inside the helper -- 1.7GB apiece on a
        # 69-day store, which is where this used to die. Unit detection keys off
        # the largest magnitude, and that is always one of the extremes, so the
        # inferred unit is identical either way.
        values = column.to_numpy()
        endpoints = column.iloc[[int(np.argmin(values)), int(np.argmax(values))]]
        nanoseconds = timestamps_to_nanoseconds(endpoints).to_numpy(dtype="int64")
    except (ValueError, TypeError, OverflowError):
        return None
    span_seconds = float(nanoseconds[1] - nanoseconds[0]) / 1e9
    if span_seconds <= 0:
        return None
    # n points span n-1 intervals.
    return span_seconds / (len(column) - 1)


def expected_absolute_move(
    frame: pd.DataFrame, horizon: int, chunk: int = 10_000_000
) -> tuple[float, int]:
    """Mean ``|log(P_{t+h} / P_t)|`` over pairs inside one contiguous segment.

    Accumulated in chunks. Taken whole, the masked division and its logarithm
    are four float64 arrays the length of the frame held at once -- 6.3GB on a
    69-day store, which is where this used to die. The mean is unchanged: it is
    the running sum over the running count, across the same pairs.
    """
    price_column = frame["price"].to_numpy()
    rows = len(price_column)
    if horizon >= rows:
        return float("nan"), 0
    segment_column = (
        frame["segment_id"].to_numpy() if "segment_id" in frame.columns else None
    )

    move_sum = 0.0
    count = 0
    for begin in range(0, rows - horizon, chunk):
        stop = min(begin + chunk, rows - horizon)
        start = price_column[begin:stop].astype(np.float64, copy=False)
        end = price_column[begin + horizon : stop + horizon].astype(np.float64, copy=False)
        usable = np.isfinite(start) & np.isfinite(end) & (start > 0) & (end > 0)
        if segment_column is not None:
            usable &= (
                segment_column[begin:stop] == segment_column[begin + horizon : stop + horizon]
            )
        if not usable.any():
            continue
        move_sum += float(np.abs(np.log(end[usable] / start[usable])).sum())
        count += int(usable.sum())

    if count == 0:
        return float("nan"), 0
    return move_sum / count, count


def realised_half_spread(
    frame: pd.DataFrame, horizon: int, chunk: int = 10_000_000
) -> float | None:
    """What the passive side of a trade still holds ``horizon`` ticks later.

    ``d * (P_t - P_{t+h}) / P_t`` averaged over trades, where ``d`` is +1 when
    the taker bought. The maker took the other side, so a price that keeps
    moving the taker's way is money out of the maker's pocket and the average
    comes out negative.

    This is the quantity a maker actually earns, and it is not the quoted
    spread. The difference between the two is adverse selection: passive orders
    are filled preferentially by traders who are right about the next move. On
    these stores the order flow is 64% directional at h=1,000, so the effect is
    not marginal.

    Uses the trade price ``h`` ticks ahead as the reference. Over short
    horizons that reference carries its own bid-ask bounce, and because trade
    signs are autocorrelated at 0.965 the bounce does not average out -- the
    h=1 estimate is largely that artefact rather than price impact. The long
    horizons, where consecutive signs have decorrelated, are the meaningful
    ones.
    """
    if "trade_sign" not in frame.columns:
        return None
    price_column = frame["price"].to_numpy()
    sign_column = frame["trade_sign"].to_numpy()
    rows = len(price_column)
    if horizon >= rows:
        return None
    segment_column = (
        frame["segment_id"].to_numpy() if "segment_id" in frame.columns else None
    )

    total = 0.0
    count = 0
    for begin in range(0, rows - horizon, chunk):
        stop = min(begin + chunk, rows - horizon)
        entry = price_column[begin:stop].astype(np.float64, copy=False)
        later = price_column[begin + horizon : stop + horizon].astype(
            np.float64, copy=False
        )
        usable = np.isfinite(entry) & np.isfinite(later) & (entry > 0)
        if segment_column is not None:
            usable &= (
                segment_column[begin:stop]
                == segment_column[begin + horizon : stop + horizon]
            )
        if not usable.any():
            continue
        sign = sign_column[begin:stop][usable].astype(np.float64)
        total += float((sign * (entry[usable] - later[usable]) / entry[usable]).sum())
        count += int(usable.sum())

    if count == 0:
        return None
    return total / count


def round_trip_cost(
    scenario: dict[str, Any],
    half_spread: float | None,
    realised: float | None = None,
) -> float:
    """Total round-trip cost as a fraction of notional.

    A taker crosses the spread on both legs and pays it. A maker rests on the
    passive side and earns -- not the quoted half-spread, but ``realised``,
    what is left of it once the price has moved. Passing ``realised=None``
    falls back to crediting the full half-spread, which is the no-adverse-
    selection assumption and is only correct if flow carries no information.
    """
    fee = 2.0 * float(scenario["fee_bps_per_side"]) * 1e-4
    if scenario["crosses_spread"]:
        return fee if half_spread is None else fee + 2.0 * half_spread
    earned = realised if realised is not None else half_spread
    return fee if earned is None else fee - 2.0 * earned


def breakeven_accuracy(cost: float, expected_move: float) -> float:
    """Directional hit rate needed to cover ``cost``; may exceed 1.0."""
    if not np.isfinite(expected_move) or expected_move <= 0:
        return float("inf")
    if cost <= 0:
        return 0.5
    return 0.5 + cost / (2.0 * expected_move)


def analyse(frame: pd.DataFrame, horizons: tuple[int, ...]) -> dict[str, Any]:
    # The spread term comes from Roll, which inverts bid-ask bounce in the
    # trade prices themselves. The earlier |price - mid| statistic is still
    # computed and reported, but only so the correction is auditable: mid_price
    # is a trailing 100-trade VWAP, so that quantity is price dispersion over
    # the averaging window and overstates the spread by 28-109x across these
    # three assets.
    half_spread = roll_half_spread(frame)
    vwap_deviation = measure_half_spread(frame)
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
        # What a passive order keeps at this horizon, and the adverse-selection
        # component it loses. Both are horizon-dependent, so maker costs are
        # no longer a single number for the whole grid.
        realised = realised_half_spread(frame, horizon)
        adverse = None if realised is None or half_spread is None else half_spread - realised
        entry: dict[str, Any] = {
            "horizon_ticks": int(horizon),
            "pairs": pairs,
            "expected_abs_move": move,
            "seconds": None if per_tick is None else per_tick * horizon,
            "realised_half_spread": realised,
            "adverse_selection": adverse,
            "scenarios": {},
        }
        for name, scenario in FEE_SCENARIOS.items():
            cost = round_trip_cost(scenario, half_spread, realised)
            accuracy = breakeven_accuracy(cost, move)
            # A negative round trip means captured spread exceeds fees, and the
            # break-even model then claims profit at any accuracy. With the
            # realised spread in hand this should no longer happen for makers,
            # since what they keep is negative; the flag stays so that a future
            # store where it does happen is reported rather than believed.
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
        "half_spread_estimator": "roll_1984",
        "vwap_deviation_superseded": vwap_deviation,
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
            f"- Half-spread **ước lượng Roll (1984)**: {half_spread * 1e4:.4f} bps "
            "— nghịch đảo bid-ask bounce trong chính chuỗi giá khớp"
        )
    else:
        lines.append(
            "- Half-spread: **không ước lượng được** (hiệp phương sai chuỗi không âm)"
        )
    deviation = analysis.get("vwap_deviation_superseded")
    if deviation is not None and half_spread:
        lines.append(
            f"- *(Đã thay thế)* |price − mid| / mid = {deviation * 1e4:.3f} bps, "
            f"**gấp {deviation / half_spread:.0f}×** — `mid_price` là VWAP trượt "
            "100 lệnh nên đại lượng này đo độ phân tán giá, không phải spread"
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
    # A Windows console defaults to cp1252 and cannot encode the Vietnamese
    # progress lines, which otherwise kills the run on its first print after
    # the load has already been paid for.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    feature_path = (ROOT / args.feature_path).resolve()
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())

    frame = _load(feature_path, args.max_rows)
    print(f"[horizon] {args.label}: {len(frame):,} dòng từ {feature_path.name}")
    analysis = analyse(frame, horizons)

    half_spread = analysis["half_spread"]
    deviation = analysis.get("vwap_deviation_superseded")
    if half_spread is not None:
        print(f"[horizon] half-spread (Roll 1984): {half_spread * 1e4:.4f} bps")
        if deviation is not None:
            print(
                f"[horizon]   (|price-mid| cu: {deviation * 1e4:.3f} bps, "
                f"gap {deviation / half_spread:.0f}x — da thay the)"
            )
    else:
        print("[horizon] half-spread: khong uoc luong duoc (cov khong am)")
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

"""Before you build a signal, can it ever be profitable at your costs?

A pre-trade feasibility screen. Given how far the price moves over your holding
horizon and what you pay to trade, it returns the directional accuracy you would
need just to break even -- and says plainly when that number is above 100%, i.e.
when a *perfect* forecast still loses money to fees. That is the §5e result of
this project turned into a calculator anyone can run on their own numbers.

It is the practical use of a negative result: it stops you spending months
building a signal that no accuracy could ever make pay. This is exactly the trap
the project's own trading analysis fell into and had to correct.

Two ways to run it.

Screen your own scenario:

    python -m scripts.tools.tradeability --move-bps 0.62 --fee-bps 5 --taker
    python -m scripts.tools.tradeability --move-bps 3.0 --fee-bps 2 --maker \
        --realised-bps -1.2

Or read a measured feasibility artifact and screen every horizon in it:

    python -m scripts.tools.tradeability \
        --from-artifact reports/research/horizon_feasibility_BTCUSDT.json

The cost model is imported from the study that produced the report numbers
(``horizon_feasibility``), so this tool cannot drift away from §5e.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.research.horizon_feasibility import (
    FEE_SCENARIOS,
    PLAUSIBLE_ACCURACY_CEILING,
    breakeven_accuracy,
    round_trip_cost,
)

ROOT = Path(__file__).resolve().parents[2]

# What the break-even accuracy means for a decision, from most to least hopeless.
IMPOSSIBLE = "impossible"  # break-even >= 100%: a perfect forecast still loses
IMPLAUSIBLE = "implausible"  # break-even below 100% but above what markets show
POSSIBLE = "possible"  # a real signal could clear it


@dataclass(frozen=True)
class Assessment:
    expected_move: float  # fraction of notional
    round_trip_cost: float  # fraction of notional
    breakeven_accuracy: float  # directional hit rate; may exceed 1.0
    verdict: str
    ceiling: float

    @property
    def cost_to_move(self) -> float:
        """How many times the cost exceeds the move -- the §5e headline number."""
        return self.round_trip_cost / self.expected_move if self.expected_move > 0 else float("inf")

    @property
    def margin(self) -> float:
        """Accuracy headroom below the plausible ceiling. Negative means out of reach."""
        return self.ceiling - self.breakeven_accuracy


def classify(breakeven: float, ceiling: float) -> str:
    if breakeven >= 1.0:
        return IMPOSSIBLE
    if breakeven > ceiling:
        return IMPLAUSIBLE
    return POSSIBLE


def assess(
    expected_move: float,
    fee_bps_per_side: float,
    crosses_spread: bool,
    half_spread: float | None = None,
    realised: float | None = None,
    ceiling: float = PLAUSIBLE_ACCURACY_CEILING,
) -> Assessment:
    """Screen one horizon/cost scenario.

    ``expected_move`` and the spreads are fractions of notional (1 bps = 1e-4).
    A taker crosses the spread; a maker rests and earns ``realised`` -- which the
    project measured to be *negative*, so a maker pays rather than earns.
    """
    scenario = {"fee_bps_per_side": fee_bps_per_side, "crosses_spread": crosses_spread}
    cost = round_trip_cost(scenario, half_spread, realised)
    breakeven = breakeven_accuracy(cost, expected_move)
    return Assessment(
        expected_move=expected_move,
        round_trip_cost=cost,
        breakeven_accuracy=breakeven,
        verdict=classify(breakeven, ceiling),
        ceiling=ceiling,
    )


@dataclass(frozen=True)
class ArtifactRow:
    horizon_ticks: int
    scenario: str
    assessment: Assessment


def from_artifact(path: Path, ceiling: float = PLAUSIBLE_ACCURACY_CEILING) -> list[ArtifactRow]:
    """Screen every horizon x scenario a feasibility artifact measured.

    The artifact already carries break-even numbers; recomputing them here from
    the same expected move and cost model, rather than reading them off, keeps
    this tool a real check on §5e rather than a pretty-printer for it.
    """
    audit = json.loads(path.read_text(encoding="utf-8"))
    analysis = audit["analysis"]
    half_spread = analysis.get("half_spread")
    rows: list[ArtifactRow] = []
    for horizon in analysis["horizons"]:
        move = horizon["expected_abs_move"]
        realised = horizon.get("realised_half_spread")
        for name, scenario in FEE_SCENARIOS.items():
            rows.append(
                ArtifactRow(
                    horizon_ticks=horizon["horizon_ticks"],
                    scenario=name,
                    assessment=assess(
                        expected_move=move,
                        fee_bps_per_side=scenario["fee_bps_per_side"],
                        crosses_spread=scenario["crosses_spread"],
                        half_spread=half_spread,
                        realised=None if scenario["crosses_spread"] else realised,
                        ceiling=ceiling,
                    ),
                )
            )
    return rows


_VERDICT_TEXT = {
    IMPOSSIBLE: "KHONG THE — du bao hoan hao van lo",
    IMPLAUSIBLE: "kho tin — vuot nguong do chinh xac thuc te",
    POSSIBLE: "co the — mot tin hieu that co the vuot",
}


def _line(label: str, a: Assessment) -> str:
    be = a.breakeven_accuracy
    be_text = ">100%" if be >= 1.0 else f"{be * 100:.1f}%"
    return (
        f"{label:<28}"
        f"cost/move={a.cost_to_move:>10.1f}x   "
        f"hoa von={be_text:>7}   "
        f"{_VERDICT_TEXT[a.verdict]}"
    )


def _print_custom(a: Assessment, fee_bps: float, maker: bool) -> None:
    side = "Maker" if maker else "Taker"
    print(f"Kich ban: {side}, {fee_bps:g} bps/chieu")
    print(f"Bien do ky vong (E|r_h|): {a.expected_move * 1e4:.4f} bps")
    print(f"Chi phi mot vong: {a.round_trip_cost * 1e4:.4f} bps")
    print("-" * 72)
    print(_line("", a).strip())
    print()
    if a.verdict == IMPOSSIBLE:
        print(
            "=> O horizon nay, phi lon hon bien do gia. Khong do chinh xac nao — "
            "ke ca 100% — sinh loi. Dung xay tin hieu o day."
        )
    elif a.verdict == IMPLAUSIBLE:
        print(
            "=> Ve ly thuyet co the co lai, nhung nguong hoa von cao hon do chinh "
            f"xac thuong thay o thi truong thanh khoan (~{a.ceiling * 100:.0f}%)."
        )
    else:
        print(
            f"=> Co the giao dich neu tin hieu dat > {a.breakeven_accuracy * 100:.1f}% "
            "do chinh xac huong. Buoc tiep theo: chung minh tin hieu dat muc do."
        )


def _print_artifact(rows: list[ArtifactRow], path: Path) -> None:
    print(f"Nguon: {path.name}")
    print("=" * 84)
    current = None
    for row in rows:
        if row.horizon_ticks != current:
            current = row.horizon_ticks
            print(f"\nHorizon {row.horizon_ticks} tick:")
        label = "  " + FEE_SCENARIOS[row.scenario]["short"]
        print("  " + _line(label, row.assessment))
    reachable = [r for r in rows if r.assessment.verdict == POSSIBLE]
    print("\n" + "=" * 84)
    if reachable:
        print(
            f"{len(reachable)} o co nguong hoa von DUOI tran do chinh xac — tuc MOT "
            "tin hieu du tot CO THE vuot. Day la dieu kien CAN, chua phai du:"
        )
        print(
            "  §5e cho thay cac model hien tai KHONG dat muc do chinh xac do o "
            "nhung o nay. Cong cu nay loc ra o dang theo duoi, khong khang dinh co lai."
        )
        print(
            "  Cac o kha thi deu o horizon rat dai — dung motivation cho Tang 2 "
            "cua roadmap (ban do horizon x chi phi)."
        )
    else:
        print(
            "Khong o nao co nguong hoa von duoi tran: khong horizon/phi nao khien "
            "mot du bao that sinh loi. Day la ket luan §5e."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-artifact", type=str, default=None)
    parser.add_argument("--move-bps", type=float, default=None,
                        help="Expected absolute move over the horizon, in bps.")
    parser.add_argument("--fee-bps", type=float, default=None,
                        help="Fee per side, in bps.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--taker", action="store_true", help="Crosses the spread.")
    group.add_argument("--maker", action="store_true", help="Rests on the book.")
    parser.add_argument("--spread-bps", type=float, default=None,
                        help="Quoted half-spread, bps (taker pays it).")
    parser.add_argument("--realised-bps", type=float, default=None,
                        help="Realised half-spread, bps (maker earns it; measured negative).")
    parser.add_argument("--ceiling", type=float, default=PLAUSIBLE_ACCURACY_CEILING,
                        help="Plausible directional-accuracy ceiling.")
    return parser.parse_args()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    if args.from_artifact:
        path = (ROOT / args.from_artifact) if not Path(args.from_artifact).is_absolute() else Path(args.from_artifact)
        if not path.is_file():
            print(f"Khong thay artifact: {args.from_artifact}")
            return 1
        _print_artifact(from_artifact(path, args.ceiling), path)
        return 0

    if args.move_bps is None or args.fee_bps is None:
        print("Can --move-bps va --fee-bps (hoac --from-artifact). Xem --help.")
        return 1

    maker = args.maker or not args.taker  # default to the resting side if unspecified
    a = assess(
        expected_move=args.move_bps * 1e-4,
        fee_bps_per_side=args.fee_bps,
        crosses_spread=not maker,
        half_spread=None if args.spread_bps is None else args.spread_bps * 1e-4,
        realised=None if args.realised_bps is None else args.realised_bps * 1e-4,
        ceiling=args.ceiling,
    )
    _print_custom(a, args.fee_bps, maker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

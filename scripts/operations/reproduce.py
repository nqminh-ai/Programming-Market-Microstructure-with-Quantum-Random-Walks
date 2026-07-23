"""Turn "everything reproduces" from a claim into something you can run.

The Makefile regenerates Phases 2-6, but the numbers the report's conclusions
now rest on -- the corrected edge (§5b), the strong-baseline defeat (§5c), the
CRPS ranking (§5d), the volatility ruling (§5d'), and the trading feasibility
(§5e) -- come from scripts under ``scripts/research/`` that the Makefile never
mentioned. A jury had the commands for the pipeline but not for the findings.

This closes that gap with a manifest. Every headline artifact is listed here
next to the exact command that produces it and the report section it feeds.
Two modes:

    python -m scripts.operations.reproduce            # verify shipped artifacts
    python -m scripts.operations.reproduce --commands  # print regeneration cmds

Verify does not re-run the studies -- some take hours over hundreds of millions
of rows. It checks that every shipped artifact exists, is valid JSON, carries
its provenance (git commit, Python, and where applicable the feature SHA-256),
and is honestly labelled exploratory rather than confirmatory. Combined with
``tests/test_report_numbers.py`` (which checks the docs quote these artifacts),
the chain is complete: documented command -> provenanced artifact -> prose.

Exit code is non-zero if any artifact is missing or malformed, so this can gate
a release.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = "reports/research"
EXPLORATORY = "EXPLORATORY_ONLY_NOT_CONFIRMATORY"

# The full 69-day stores, used by everything except the pre-registered §5d
# table, which predates them.
FEATURES = {
    "BTCUSDT": "data/assets/btcusdt/features/features_BTCUSDT_69d.parquet",
    "ETHUSDT": "data/assets/ethusdt/features/features_ETHUSDT_69d.parquet",
    "BNBUSDT": "data/assets/bnbusdt/features/features_BNBUSDT_69d.parquet",
}
# The §5d marginal-CRPS table was run before the 69-day stores existed, each
# asset off its own file. Two of those runs predate the feature_sha256 field,
# so their artifacts carry weaker provenance than the rest; the report's §5d box
# documents this, and the day-cluster runs (limitation #4) are the full-data
# replacement. The manifest records the files actually used, not the 69-day
# ones, so the printed command reproduces what shipped.
CRPS_5WINDOW = {
    "BTCUSDT": "data/assets/btcusdt/features/features_BTCUSDT_recent_subset.parquet",
    "ETHUSDT": "data/assets/ethusdt/features/features_ETHUSDT_2026-06-12.parquet",
    "BNBUSDT": "data/assets/bnbusdt/features/features_BNBUSDT_multiday.parquet",
}
CRPS_5WINDOW_HAS_SHA = {"BNBUSDT"}  # only the reran one carries a SHA
ASSETS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")


@dataclass(frozen=True)
class Artifact:
    """One headline JSON, the command that makes it, and what it feeds."""

    path: str
    section: str
    command: tuple[str, ...]
    note: str = ""
    # Some studies do not read a feature file (they aggregate other artifacts),
    # so a feature SHA-256 is only required where one is actually consumed.
    expect_feature_sha: bool = True


def _manifest() -> tuple[Artifact, ...]:
    items: list[Artifact] = [
        Artifact(
            f"{RESEARCH}/full_dataset_confirmation.json",
            "§5b — corrected edge on the full 32.4M-tick BTC dataset (-0.013091)",
            (
                "python", "-m", "scripts.research.full_dataset_confirmation",
                "--feature-path", FEATURES["BTCUSDT"],
                "--label", "BTCUSDT_full",
            ),
            note="Full 32.4M rows; needs column-wise load, ~4 GB.",
            expect_feature_sha=False,
        ),
        Artifact(
            f"{RESEARCH}/confirmation_btcusdt_69d_100M.json",
            "§5b — replication on 100M ticks from a different period",
            (
                "python", "-m", "scripts.research.full_dataset_confirmation",
                "--feature-path", FEATURES["BTCUSDT"],
                "--label", "BTCUSDT_69d_100M",
                "--max-rows", "100000000",
                "--json-out", f"{RESEARCH}/confirmation_btcusdt_69d_100M.json",
                "--md-out", f"{RESEARCH}/confirmation_btcusdt_69d_100M.md",
            ),
            note="100M rows; the heaviest run in the project.",
        ),
        Artifact(
            f"{RESEARCH}/volatility_claim.json",
            "§5d' — cross-asset ruling on the volatility claim (not established)",
            ("python", "-m", "scripts.research.volatility_claim_summary"),
            note="Aggregates the three marginal_crps_vol_* artifacts; seconds.",
            expect_feature_sha=False,
        ),
    ]
    for asset in ASSETS:
        lower = asset.lower()
        items.append(
            Artifact(
                f"{RESEARCH}/strong_baseline_{asset}.json",
                f"§5c — {asset}: QRW vs strong causal baselines",
                (
                    "python", "-m", "scripts.research.strong_baseline_comparison",
                    "--feature-path", FEATURES[asset],
                    "--label", asset,
                ),
                expect_feature_sha=False,
            )
        )
        items.append(
            Artifact(
                f"{RESEARCH}/marginal_crps_{asset}.json",
                f"§5d — {asset}: marginal CRPS, 5 pre-registered windows",
                (
                    "python", "-m", "scripts.research.marginal_crps_comparison",
                    "--feature-path", CRPS_5WINDOW[asset],
                    "--label", asset,
                    "--windows", "5",
                    "--json-out", f"{RESEARCH}/marginal_crps_{asset}.json",
                ),
                note="Pre-registered §5d run; predates the 69-day stores."
                + ("" if asset in CRPS_5WINDOW_HAS_SHA else " No feature SHA (predates the field)."),
                expect_feature_sha=asset in CRPS_5WINDOW_HAS_SHA,
            )
        )
        items.append(
            Artifact(
                f"{RESEARCH}/marginal_crps_vol_{asset}.json",
                f"§5d' — {asset}: marginal CRPS, 40 windows, volatility retest",
                (
                    "python", "-m", "scripts.research.marginal_crps_comparison",
                    "--feature-path", FEATURES[asset],
                    "--label", f"{asset}_69d",
                    "--max-rows", "20000000",
                    "--windows", "40",
                    "--window-unit", "rows",
                    "--json-out", f"{RESEARCH}/marginal_crps_vol_{asset}.json",
                ),
                note="20M rows; ~1 hour per asset.",
            )
        )
        items.append(
            Artifact(
                f"{RESEARCH}/horizon_feasibility_{asset}.json",
                f"§5e — {asset}: Roll spread, adverse selection, break-even cost",
                (
                    "python", "-m", "scripts.research.horizon_feasibility",
                    "--feature-path", FEATURES[asset],
                    "--label", asset,
                ),
            )
        )
        items.append(
            Artifact(
                f"{RESEARCH}/horizon_edge_{asset}.json",
                f"§5e — {asset}: directional accuracy vs break-even by horizon",
                (
                    "python", "-m", "scripts.research.horizon_label_baselines",
                    "--feature-path", FEATURES[asset],
                    "--label", asset,
                ),
            )
        )
    return tuple(items)


MANIFEST = _manifest()


@dataclass
class Check:
    artifact: Artifact
    ok: bool
    problems: list[str] = field(default_factory=list)


def verify_one(artifact: Artifact, root: Path = ROOT) -> Check:
    path = root / artifact.path
    problems: list[str] = []
    if not path.is_file():
        return Check(artifact, False, ["missing"])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return Check(artifact, False, [f"unreadable: {error}"])

    prov = data.get("provenance", {}) if isinstance(data.get("provenance"), dict) else {}

    def _get(key: str):
        return data.get(key, prov.get(key))

    status = _get("status")
    if status != EXPLORATORY:
        # A confirmatory label would be the one mislabelling this project must
        # never ship, so it is a hard failure, not a warning.
        problems.append(f"status is {status!r}, not the exploratory marker")
    if not _get("git_commit") and not _get("code_commit"):
        problems.append("no git commit recorded")
    if not _get("python"):
        problems.append("no Python version recorded")
    if artifact.expect_feature_sha and not _get("feature_sha256"):
        problems.append("no feature SHA-256 recorded")

    return Check(artifact, not problems, problems)


def verify_all(root: Path = ROOT) -> list[Check]:
    return [verify_one(a, root) for a in MANIFEST]


def _print_report(checks: list[Check]) -> None:
    width = max(len(Path(c.artifact.path).name) for c in checks)
    print(f"{'artifact':<{width}}  status  section")
    print("-" * (width + 40))
    for check in checks:
        name = Path(check.artifact.path).name
        mark = "OK  " if check.ok else "FAIL"
        print(f"{name:<{width}}  {mark}    {check.artifact.section}")
        for problem in check.problems:
            print(f"{'':<{width}}          -> {problem}")
    failed = [c for c in checks if not c.ok]
    print("-" * (width + 40))
    if failed:
        print(f"{len(failed)} of {len(checks)} artifact(s) failed verification.")
    else:
        print(f"All {len(checks)} headline artifacts present and provenanced.")


def _print_commands() -> None:
    print("# Regenerate the headline research artifacts.")
    print("# Run from the repo root, in the project virtualenv.")
    print("# Order is independent; heavy runs are noted.\n")
    for artifact in MANIFEST:
        print(f"# {artifact.section}")
        if artifact.note:
            print(f"#   ({artifact.note})")
        print(" ".join(_quote(part) for part in artifact.command))
        print()


def _quote(part: str) -> str:
    return f'"{part}"' if " " in part else part


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commands",
        action="store_true",
        help="Print the regeneration commands instead of verifying.",
    )
    args = parser.parse_args()

    if args.commands:
        _print_commands()
        return 0

    checks = verify_all()
    _print_report(checks)
    return 1 if any(not c.ok for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())

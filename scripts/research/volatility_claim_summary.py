"""Does the report's volatility claim survive being measured on all three assets?

The report has carried, since Phase 5, an interpretation read off five windows
per asset: the QRW wins quiet windows and loses badly on volatile ones, so it
does not model volatility dynamics. Limitation #5 rests on it. Five points
cannot support that either way, so it was re-measured at forty windows per
asset (``marginal_crps_comparison.py --windows 40``).

This reads those three runs back and answers the claim once, across all of
them. Two things it does not take on faith:

* The per-window gap is recomputed from the stored per-model scores rather than
  read off the stored ``qrw_crps_gap``, which was the absolute difference. CRPS
  carries the scale of its window, so the absolute figure is not comparable
  across windows.
* One asset agreeing is not the claim. The three per-asset tests are combined
  one-sided *in the direction the report asserts*, so a consistent-but-weak
  signal gets its fair hearing rather than being dismissed three times over.

    python -m scripts.research.volatility_claim_summary

EXPLORATORY ONLY. Not a confirmatory run; do not relabel as confirmatory.
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
from scipy.stats import combine_pvalues

from scripts.research.marginal_crps_comparison import (
    QRW_MODEL,
    PRIMARY,
    volatility_relationship,
)
from src.evaluation.provenance import canonical_repo_path, sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = (
    "reports/research/marginal_crps_vol_BNBUSDT.json",
    "reports/research/marginal_crps_vol_ETHUSDT.json",
    "reports/research/marginal_crps_vol_BTCUSDT.json",
)
ALPHA = 0.05


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def relative_gaps(per_window: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild each window's gap as a fraction of what the best rival scored.

    Recomputed from ``scores`` rather than trusted from the stored gap: the
    runs that produced these files wrote the absolute difference, and a
    difference of 0.02 is a rout in a window scoring 0.05 and noise in one
    scoring 5.0.
    """
    rebuilt = []
    for row in per_window:
        scores = row["scores"]
        rivals = [
            s[PRIMARY] for name, s in scores.items() if name != QRW_MODEL and PRIMARY in s
        ]
        qrw = scores.get(QRW_MODEL, {}).get(PRIMARY)
        if qrw is None or not rivals:
            continue
        best_rival = min(rivals)
        if not best_rival > 0:
            continue
        rebuilt.append(
            {
                "window": row["window"],
                "realised_volatility": row["realised_volatility"],
                "qrw_crps_gap": float(qrw - best_rival),
                "qrw_crps_gap_relative": float((qrw - best_rival) / best_rival),
            }
        )
    return rebuilt


def one_sided_p(spearman: float, two_sided_p: float) -> float:
    """The report asserts a direction, so it is tested in that direction.

    A two-sided p-value spends half its budget on the possibility that the QRW
    does *better* when volatility rises, which the report does not claim.
    """
    return two_sided_p / 2.0 if spearman > 0 else 1.0 - two_sided_p / 2.0


def combine_assets(per_asset: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool the per-asset tests, so three weak agreements are not read as three
    refusals.

    Stouffer alongside Fisher because they can disagree here: Fisher is driven
    by the smallest p-value, Stouffer weighs the whole set, and a claim that
    holds mildly on all three is exactly the case where that matters.
    """
    usable = [a for a in per_asset if a["spearman"] is not None]
    if len(usable) < 2:
        return {"assets_used": len(usable), "note": "too few assets to combine"}
    p_values = [one_sided_p(a["spearman"], a["p_value"]) for a in usable]
    fisher = combine_pvalues(p_values, method="fisher")
    stouffer = combine_pvalues(p_values, method="stouffer")
    all_positive = all(a["spearman"] > 0 for a in usable)
    return {
        "assets_used": len(usable),
        "one_sided_p_values": [round(p, 6) for p in p_values],
        "all_in_claimed_direction": bool(all_positive),
        "fisher_statistic": float(fisher.statistic),
        "fisher_p": float(fisher.pvalue),
        "stouffer_z": float(stouffer.statistic),
        "stouffer_p": float(stouffer.pvalue),
        # The claim needs the same direction everywhere and a pooled result
        # that a coin could not have produced -- under *both* poolings. Letting
        # either one alone decide would mean the answer is chosen by which test
        # is quoted, which is how a borderline result becomes a finding.
        "supports_claim": bool(
            all_positive and fisher.pvalue < ALPHA and stouffer.pvalue < ALPHA
        ),
    }


def summarise(source_paths: list[Path]) -> dict[str, Any]:
    per_asset: list[dict[str, Any]] = []
    for path in source_paths:
        audit = json.loads(path.read_text(encoding="utf-8"))
        rebuilt = relative_gaps(audit["per_window"])
        measured = volatility_relationship(rebuilt)
        qrw = audit["aggregate"].get(QRW_MODEL, {})
        per_asset.append(
            {
                "label": audit["label"],
                "source": canonical_repo_path(path, ROOT),
                "source_sha256": sha256_file(path),
                "source_git_commit": audit.get("git_commit"),
                "windows": audit["windows"],
                "rows": audit["rows"],
                "qrw_rank": audit["ranked_by_mean_crps"].index(QRW_MODEL) + 1,
                "qrw_windows_best": qrw.get("windows_best"),
                "ranked_by_mean_crps": audit["ranked_by_mean_crps"],
                **measured,
            }
        )
    combined = combine_assets(per_asset)
    return {
        "kind": "volatility_claim_summary",
        "status": "EXPLORATORY_ONLY_NOT_CONFIRMATORY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "claim_under_test": (
            "The QRW falls further behind the best alternative as realised "
            "volatility rises, i.e. it does not model volatility dynamics."
        ),
        "gap_definition": "(QRW CRPS - best rival CRPS) / best rival CRPS, per window",
        "alpha": ALPHA,
        "per_asset": per_asset,
        "combined": combined,
        "verdict": _verdict(per_asset, combined),
    }


def _verdict(per_asset: list[dict[str, Any]], combined: dict[str, Any]) -> str:
    if combined.get("supports_claim"):
        return (
            f"Supported: the relationship runs in the asserted direction on all "
            f"{combined['assets_used']} assets and survives pooling "
            f"(Fisher p={combined['fisher_p']:.4f})."
        )
    singles = ", ".join(
        f"{a['label'].split('_')[0]} rho={a['spearman']:+.2f} (p={a['p_value']:.3f})"
        for a in per_asset
        if a["spearman"] is not None
    )
    direction = (
        "runs in the asserted direction on every asset"
        if combined.get("all_in_claimed_direction")
        else "does not even run the same way on every asset"
    )
    if "fisher_p" not in combined:
        pooled = "the assets could not be pooled"
    elif (combined["fisher_p"] < ALPHA) != (combined["stouffer_p"] < ALPHA):
        # Straddling alpha is not a detail to round off. Fisher is driven by the
        # single smallest p-value, Stouffer by the whole set, and a claim whose
        # fate depends on which one is quoted has not been established.
        pooled = (
            f"the two pooling methods straddle alpha -- Fisher p="
            f"{combined['fisher_p']:.4f}, Stouffer p={combined['stouffer_p']:.4f} -- "
            f"so the pooled answer depends on which test is quoted"
        )
    else:
        pooled = (
            f"the pooled test does not reach it either (Fisher p="
            f"{combined['fisher_p']:.4f}, Stouffer p={combined['stouffer_p']:.4f})"
        )
    return (
        f"NOT established at alpha={ALPHA}. The relationship {direction} but no "
        f"asset reaches significance on its own ({singles}), and {pooled}. The "
        f"claim should be stated as a direction the data leans towards, not a "
        f"finding."
    )


def _render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Does the volatility claim survive measurement?",
        "",
        f"**Status:** `{audit['status']}` — exploratory.",
        "",
        f"**Claim under test:** {audit['claim_under_test']}",
        "",
        f"**Gap definition:** {audit['gap_definition']}",
        "",
        f"- Git commit: `{audit['git_commit']}` · Python {audit['python']}",
        "",
        "## Per asset (40 non-overlapping windows each)",
        "",
        "| Asset | Windows | Spearman | p (two-sided) | Supports claim | QRW CRPS rank | QRW wins |",
        "|---|---:|---:|---:|:--:|:--:|:--:|",
    ]
    for a in audit["per_asset"]:
        rho = "—" if a["spearman"] is None else f"{a['spearman']:+.3f}"
        p = "—" if a["p_value"] is None else f"{a['p_value']:.3f}"
        lines.append(
            f"| {a['label']} | {a['windows_used']} | {rho} | {p} | "
            f"{'yes' if a['supports_claim'] else 'no'} | "
            f"{a['qrw_rank']}/6 | {a['qrw_windows_best']}/{a['windows']} |"
        )
    c = audit["combined"]
    lines += ["", "## Pooled across assets (one-sided, in the asserted direction)", ""]
    if "fisher_p" in c:
        lines += [
            f"- One-sided p-values: {c['one_sided_p_values']}",
            f"- All in the claimed direction: **{c['all_in_claimed_direction']}**",
            f"- Fisher: chi2={c['fisher_statistic']:.2f}, p=**{c['fisher_p']:.4f}**",
            f"- Stouffer: z={c['stouffer_z']:.2f}, p=**{c['stouffer_p']:.4f}**",
        ]
    else:
        lines.append(f"- {c.get('note')}")
    lines += ["", "## Verdict", "", audit["verdict"], ""]
    lines.append("### Sources")
    lines.append("")
    for a in audit["per_asset"]:
        lines.append(f"- `{a['source']}` — sha256 `{a['source_sha256'][:16]}…`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="*", default=list(DEFAULT_SOURCES))
    parser.add_argument("--json-out", default="reports/research/volatility_claim.json")
    parser.add_argument("--md-out", default="reports/research/volatility_claim.md")
    return parser.parse_args()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    paths = [ROOT / s for s in args.sources]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(
            "missing source run(s): " + ", ".join(str(p) for p in missing)
        )

    audit = summarise(paths)
    json_out = ROOT / args.json_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (ROOT / args.md_out).write_text(_render_markdown(audit), encoding="utf-8")

    print(f"[vol] VERDICT: {audit['verdict']}")
    print(f"[vol] wrote {args.json_out} and {args.md_out}")


if __name__ == "__main__":
    main()

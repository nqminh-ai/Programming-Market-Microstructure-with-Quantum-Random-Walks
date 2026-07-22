"""Does the windowed-QRW edge survive against STRONG classical baselines?

Phase 1-2 established that (a) the quantum phase contributes nothing and (b)
after the Phase 2 bias fix the windowed-QRW beats the *affine* baseline (OBI +
tick direction) stably on BTC and BNB. But affine is a weak reference. The
decisive question for the project's thesis is whether that windowing/decoherence
edge survives against the genuinely competitive classical models named in the
pre-registration (``docs/data_collection_todo.md``):

    Logistic L2 (5 features), Logistic L2 + pairwise interactions,
    Nonlinear calibrated, OrderFlow AR(5), Marked Hawkes logit,
    plus the QRW directional-link logistic approximation.

Those baselines are already implemented in ``src/evaluation/directional_baselines``.
This script fits them on a chronological train/validation split, fits the real
``MarketQRW`` windowed density-matrix model on the same train frame, and scores
everyone on the identical disjoint test events. ``directional_events`` filters
events exactly like the QRW audit's ``market_events`` (same valid mask, same
next-move target), so the per-event targets align and Brier differences are
genuinely paired and block-bootstrapped.

If the windowed-QRW does not beat the best strong baseline, the windowing edge
over affine is not a real predictive advantage -- a stronger classical model
captures it.

EXPLORATORY ONLY. Not a confirmatory run; do not relabel as confirmatory.
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

from scripts.audits.phase3_overfitting_audit import moving_block_bootstrap_mean
from src.evaluation.directional_baselines import (
    directional_events,
    fit_directional_baselines,
)
from src.evaluation.provenance import canonical_repo_path, sha256_file
from src.models.qrw_market_sim import MarketQRW

ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _score(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return {
        "brier": float(np.mean((probability - target) ** 2)),
        "log_loss": float(
            -np.mean(
                target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)
            )
        ),
        "accuracy": float(np.mean((probability >= 0.5) == target)),
    }


def _fit_windowed_qrw(train_frame: pd.DataFrame) -> MarketQRW:
    """Calibrate the real MarketQRW windowed model on the train frame."""
    model = MarketQRW(
        train_frame,
        {
            "n_positions": 101,
            "gamma_base": 0.0,
            "alpha_obi": 0.0,
            "coin_type": "obi_adaptive",
        },
    )
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        model.calibrate(Path(directory) / "params.json")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        default="data/assets/btcusdt/features/features_BTCUSDT_recent_subset.parquet",
    )
    parser.add_argument("--label", default="BTCUSDT")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = (ROOT / args.feature_path).resolve()
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    frame = (
        pd.read_parquet(feature_path)
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )
    if args.max_rows and len(frame) > args.max_rows:
        frame = frame.iloc[: args.max_rows].reset_index(drop=True)
    print(f"[strong] {args.label}: {len(frame):,} rows from {feature_path.name}")

    train_end = int(len(frame) * args.train_fraction)
    val_end = int(len(frame) * (args.train_fraction + args.validation_fraction))
    train_frame = frame.iloc[:train_end].copy()
    val_frame = frame.iloc[train_end:val_end].copy()
    test_frame = frame.iloc[val_end:].copy()

    train_events = directional_events(train_frame)
    val_events = directional_events(val_frame)
    test_events = directional_events(test_frame)
    target = test_events.target
    print(
        f"[strong] events train/val/test = "
        f"{len(train_events)}/{len(val_events)}/{len(test_events)}"
    )

    # Strong classical baselines (tuned on the shared validation fold).
    print("[strong] fitting strong classical baselines...")
    baselines = fit_directional_baselines(train_events, val_events)
    predictions: dict[str, np.ndarray] = {}
    for name, model in baselines.items():
        predictions[name] = model.predict(test_events)

    # The real windowed-QRW density-matrix model, evaluated on the identical
    # test events (features[:,0]=OBI, features[:,1]=tick direction).
    print("[strong] calibrating windowed-QRW (MarketQRW)...")
    qrw = _fit_windowed_qrw(train_frame)
    predictions["Windowed-QRW (density matrix)"] = qrw.predict_right_probabilities(
        test_events.features[:, 0],
        test_events.features[:, 1],
    )

    scores = {name: _score(prob, target) for name, prob in predictions.items()}

    # Rank by test Brier; identify the strongest non-QRW baseline.
    qrw_name = "Windowed-QRW (density matrix)"
    baseline_names = [n for n in predictions if n != qrw_name]
    best_baseline = min(baseline_names, key=lambda n: scores[n]["brier"])
    print(
        f"[strong] QRW Brier={scores[qrw_name]['brier']:.6f} | "
        f"best baseline '{best_baseline}' Brier={scores[best_baseline]['brier']:.6f}"
    )

    # Paired block-bootstrap: QRW minus each baseline (negative => QRW better).
    seed_seq = np.random.SeedSequence(args.seed)
    children = dict(zip(baseline_names, seed_seq.spawn(len(baseline_names))))
    comparisons: dict[str, Any] = {}
    for name in baseline_names:
        diff = (predictions[qrw_name] - target) ** 2 - (
            predictions[name] - target
        ) ** 2
        edge = moving_block_bootstrap_mean(
            diff,
            samples=args.bootstrap_samples,
            rng=np.random.default_rng(children[name]),
            block_size=args.block_size,
        )
        lo, hi = edge["confidence_interval_95"]
        edge["qrw_significantly_better"] = bool(hi < 0.0)
        edge["qrw_significantly_worse"] = bool(lo > 0.0)
        comparisons[name] = edge

    vs_best = comparisons[best_baseline]
    if vs_best["qrw_significantly_better"]:
        verdict = (
            f"Windowed-QRW significantly beats even the strongest baseline "
            f"('{best_baseline}') on {args.label}."
        )
    elif vs_best["qrw_significantly_worse"]:
        verdict = (
            f"Windowed-QRW is significantly WORSE than the strongest baseline "
            f"('{best_baseline}') on {args.label}: the edge over affine does not "
            f"survive a competitive classical model."
        )
    else:
        verdict = (
            f"Windowed-QRW is statistically tied with the strongest baseline "
            f"('{best_baseline}') on {args.label}: no advantage over a "
            f"competitive classical model."
        )
    print(f"[strong] VERDICT: {verdict}")

    audit = {
        "kind": "strong_baseline_comparison",
        "status": "EXPLORATORY_ONLY_NOT_CONFIRMATORY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "label": args.label,
        "feature_path": canonical_repo_path(feature_path, ROOT),
        "feature_sha256": sha256_file(feature_path),
        "rows": int(len(frame)),
        "events": {
            "train": int(len(train_events)),
            "validation": int(len(val_events)),
            "test": int(len(test_events)),
        },
        "split": {
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
        },
        "bootstrap_samples": int(args.bootstrap_samples),
        "block_size": int(args.block_size),
        "seed": int(args.seed),
        "scores": scores,
        "best_baseline": best_baseline,
        "comparisons_qrw_minus_baseline": comparisons,
        "verdict": verdict,
    }

    label = args.label
    json_out = (
        ROOT / (args.json_out or f"reports/research/strong_baseline_{label}.json")
    ).resolve()
    md_out = (
        ROOT / (args.md_out or f"reports/research/strong_baseline_{label}.md")
    ).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md_out.write_text(_render_markdown(audit), encoding="utf-8")
    print(f"[strong] wrote {json_out}")
    print(f"[strong] wrote {md_out}")


def _render_markdown(audit: dict[str, Any]) -> str:
    scores = audit["scores"]
    cmp = audit["comparisons_qrw_minus_baseline"]
    qrw_name = "Windowed-QRW (density matrix)"
    ordered = sorted(scores, key=lambda n: scores[n]["brier"])
    rows = [
        f"# Strong-baseline comparison — {audit['label']}",
        "",
        f"**Status:** `{audit['status']}` — exploratory, not confirmatory.",
        "",
        f"- Feature file: `{Path(audit['feature_path']).name}` "
        f"({audit['rows']:,} rows)",
        f"- Events train/val/test: {audit['events']['train']:,} / "
        f"{audit['events']['validation']:,} / {audit['events']['test']:,}",
        f"- Git commit: `{audit['git_commit']}` · Python {audit['python']} · "
        f"seed {audit['seed']}",
        "",
        "## Test-set scores (ranked by Brier, lower is better)",
        "",
        "| Rank | Model | Brier | log loss | accuracy |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, name in enumerate(ordered, 1):
        s = scores[name]
        mark = " **(QRW)**" if name == qrw_name else ""
        rows.append(
            f"| {i} | {name}{mark} | {s['brier']:.6f} | "
            f"{s['log_loss']:.6f} | {s['accuracy']:.4f} |"
        )
    rows += [
        "",
        "## Windowed-QRW vs each baseline (paired Brier diff, 95% block-bootstrap)",
        "",
        "Negative mean = QRW has the lower (better) Brier.",
        "",
        "| Baseline | mean (QRW−base) | 95% CI | QRW better? |",
        "|---|---:|---|:--:|",
    ]
    for name in sorted(cmp, key=lambda n: cmp[n]["model_minus_comparison"]):
        e = cmp[name]
        lo, hi = e["confidence_interval_95"]
        if e["qrw_significantly_better"]:
            flag = "✔ better"
        elif e["qrw_significantly_worse"]:
            flag = "✘ worse"
        else:
            flag = "— tie"
        rows.append(
            f"| {name} | {e['model_minus_comparison']:+.6f} | "
            f"[{lo:+.6f}, {hi:+.6f}] | {flag} |"
        )
    rows += ["", "## Verdict", "", audit["verdict"], ""]
    return "\n".join(rows)


if __name__ == "__main__":
    main()

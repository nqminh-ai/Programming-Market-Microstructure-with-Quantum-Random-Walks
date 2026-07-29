"""Read-only Streamlit dashboard for one Phase 8 ML release manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.provenance import sha256_file


DEFAULT_MANIFEST = (
    ROOT
    / "reports"
    / "research"
    / "ml_release"
    / "BNBUSDT"
    / "h50000"
    / "ml_release_manifest.json"
)


def load_release_manifest(path: str | Path) -> Mapping[str, Any]:
    """Load and validate display-safe release metadata without recomputation."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("release manifest must be a JSON object")
    if payload.get("kind") != "ml_directional_release":
        raise ValueError("unsupported release manifest kind")
    if (
        payload.get("holdout_state") != "closed"
        or payload.get("test_fold_read") is not False
        or payload.get("test_metrics") is not None
        or payload.get("live_trading_authorized") is not False
    ):
        raise ValueError("dashboard refuses a release with unsafe test state")
    if payload.get("dashboard", {}).get("policy") != (
        "read_only_manifest_no_recompute"
    ):
        raise ValueError("release dashboard policy is not read-only")
    return payload


def artifact_table(manifest: Mapping[str, Any]) -> pd.DataFrame:
    """Return manifest rows only; never load predictions or refit models."""
    rows = list(manifest.get("artifacts", ()))
    required = {"role", "kind", "path", "sha256", "bytes"}
    if not rows or any(required.difference(row) for row in rows):
        raise ValueError("release manifest artifact table is incomplete")
    return pd.DataFrame(rows).loc[
        :, ["role", "kind", "path", "bytes", "sha256"]
    ]


def verify_release_files(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path = ROOT,
) -> list[dict[str, Any]]:
    """Read files and compare SHA-256; this does not execute any artifact."""
    root = Path(repo_root).resolve()
    results: list[dict[str, Any]] = []
    for entry in manifest.get("artifacts", ()):
        path = Path(entry["path"])
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        exists = resolved.is_file()
        matches = exists and sha256_file(resolved) == entry["sha256"]
        results.append(
            {
                "role": entry["role"],
                "exists": exists,
                "sha256_matches": matches,
            }
        )
    return results


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="ML/QRW Release",
        page_icon="🔒",
        layout="wide",
    )
    st.title("ML/QRW pretest release")
    st.caption(
        "Read-only artifact inventory. This page never trains models or opens "
        "the test fold."
    )
    selected = st.sidebar.text_input(
        "Release manifest",
        value=str(DEFAULT_MANIFEST),
    )
    source = Path(selected)
    if not source.is_file():
        st.warning("Release manifest not found. Build Phase 8 first.")
        return
    try:
        manifest = load_release_manifest(source)
        table = artifact_table(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(str(error))
        return
    left, middle, right, fourth = st.columns(4)
    left.metric("Protocol", manifest["protocol_version"])
    middle.metric("Asset", manifest["asset"])
    right.metric("Horizon", f"{manifest['horizon_ticks']:,} ticks")
    fourth.metric("Artifacts", len(table))
    if manifest["official"]:
        st.success("Official pretest release; holdout remains closed.")
    else:
        st.warning(
            "Development bundle: dirty source paths may be present and this "
            "must not support research claims."
        )
    st.subheader("Artifact inventory")
    st.dataframe(table, width="stretch", hide_index=True)
    st.subheader("Safety state")
    st.json(
        {
            "holdout_state": manifest["holdout_state"],
            "test_fold_read": manifest["test_fold_read"],
            "test_metrics": manifest["test_metrics"],
            "live_trading_authorized": manifest[
                "live_trading_authorized"
            ],
        }
    )
    if st.button("Verify artifact hashes (read-only)"):
        st.dataframe(
            pd.DataFrame(verify_release_files(manifest)),
            width="stretch",
            hide_index=True,
        )
    st.code(
        manifest["reproduction"]["path"],
        language="powershell",
    )


if __name__ == "__main__":
    main()

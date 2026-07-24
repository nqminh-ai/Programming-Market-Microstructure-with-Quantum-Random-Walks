"""Tests for the reproducibility verifier.

The verifier's whole purpose is to fail when a shipped artifact is missing,
malformed, or -- the one that would embarrass this project -- accidentally
labelled confirmatory. These cover that it actually does fail in those cases,
rather than printing OK regardless.
"""

from __future__ import annotations

import json

import pytest

from scripts.operations.reproduce import (
    ASSETS,
    CRPS_5WINDOW,
    EXPLORATORY,
    MANIFEST,
    Artifact,
    verify_one,
)


def _valid_artifact(path):
    path.write_text(
        json.dumps(
            {
                "status": EXPLORATORY,
                "git_commit": "abc123",
                "python": "3.14.5",
                "feature_sha256": "deadbeef",
            }
        ),
        encoding="utf-8",
    )


def _entry(rel: str, expect_sha: bool = True) -> Artifact:
    return Artifact(path=rel, section="test", command=("noop",), expect_feature_sha=expect_sha)


def test_a_well_formed_exploratory_artifact_passes(tmp_path) -> None:
    (tmp_path / "reports").mkdir()
    _valid_artifact(tmp_path / "reports" / "x.json")

    check = verify_one(_entry("reports/x.json"), root=tmp_path)

    assert check.ok
    assert check.problems == []


def test_a_missing_artifact_fails(tmp_path) -> None:
    check = verify_one(_entry("reports/gone.json"), root=tmp_path)

    assert not check.ok
    assert "missing" in check.problems


def test_a_confirmatory_label_is_a_hard_failure(tmp_path) -> None:
    """The one mislabel this project must never ship."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "x.json").write_text(
        json.dumps({"status": "CONFIRMATORY", "git_commit": "a", "python": "3.14"}),
        encoding="utf-8",
    )

    check = verify_one(_entry("reports/x.json", expect_sha=False), root=tmp_path)

    assert not check.ok
    assert any("confirmatory" in p.lower() or "exploratory" in p.lower() for p in check.problems)


def test_missing_provenance_fails(tmp_path) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "x.json").write_text(
        json.dumps({"status": EXPLORATORY}), encoding="utf-8"
    )

    check = verify_one(_entry("reports/x.json"), root=tmp_path)

    assert not check.ok
    assert any("git commit" in p for p in check.problems)
    assert any("Python" in p for p in check.problems)


def test_a_missing_feature_sha_only_fails_when_expected(tmp_path) -> None:
    """Aggregators and the pre-registered originals read no feature file."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "x.json").write_text(
        json.dumps({"status": EXPLORATORY, "git_commit": "a", "python": "3.14"}),
        encoding="utf-8",
    )

    assert verify_one(_entry("reports/x.json", expect_sha=False), root=tmp_path).ok
    assert not verify_one(_entry("reports/x.json", expect_sha=True), root=tmp_path).ok


def test_provenance_may_live_in_a_nested_block(tmp_path) -> None:
    """Some artifacts nest provenance under a 'provenance' key."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "x.json").write_text(
        json.dumps(
            {
                "status": EXPLORATORY,
                "provenance": {"code_commit": "a", "python": "3.14", "feature_sha256": "d"},
            }
        ),
        encoding="utf-8",
    )

    assert verify_one(_entry("reports/x.json"), root=tmp_path).ok


def test_unreadable_json_fails_rather_than_raising(tmp_path) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "x.json").write_text("{not json", encoding="utf-8")

    check = verify_one(_entry("reports/x.json"), root=tmp_path)

    assert not check.ok
    assert any("unreadable" in p for p in check.problems)


# ---------------------------------------------------------------------------
# The manifest itself
# ---------------------------------------------------------------------------


def test_the_manifest_command_points_at_the_file_the_artifact_records() -> None:
    """A command that reproduces a different file than shipped is worse than
    none -- it looks reproducible and is not.

    The §5d artifacts were each run off their own file, so the manifest must
    name those, not the 69-day store.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for asset in ASSETS:
        entry = next(
            a for a in MANIFEST if a.path.endswith(f"marginal_crps_{asset}.json")
        )
        recorded = json.loads(
            (root / entry.path).read_text(encoding="utf-8")
        )["feature_path"]
        # The command's feature file must be the basename the artifact records.
        assert Path(CRPS_5WINDOW[asset]).name == Path(recorded).name, asset
        assert CRPS_5WINDOW[asset] in entry.command


def test_every_manifest_entry_is_a_headline_section() -> None:
    for artifact in MANIFEST:
        assert artifact.section.startswith("§5"), artifact.path


def test_the_manifest_covers_all_three_assets_for_the_per_asset_studies() -> None:
    for stem in (
        "strong_baseline",
        "horizon_feasibility",
        "horizon_edge",
        "marginal_crps_vol",
        "marginal_crps_daycluster",
    ):
        covered = {a.path for a in MANIFEST if stem in a.path}
        assert len(covered) == 3, f"{stem}: {covered}"

"""Do the reports still quote the numbers their artifacts actually hold?

Every headline figure in ``docs/`` was copied by hand out of a JSON artifact,
and the artifacts get regenerated as the data grows. Nothing has been checking
that the two still agree -- the executive summary carried "more than two
thousand times" for a ratio that the artifact put at 1,610 until a dashboard
test happened to cross-check it.

So this reads the numbers out of the artifacts and asserts the prose quotes
them. A test failing here does not mean the docs are wrong: it means a rerun
moved a number and the prose has not caught up. Fix the prose, or explain in
the doc why the old figure is being kept.

Vietnamese decimal comma throughout, because that is how the reports are
written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESEARCH = ROOT / "reports" / "research"


def _artifact(name: str) -> dict:
    path = RESEARCH / name
    if not path.is_file():
        pytest.skip(f"artifact not present: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _doc(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def _vn(value: float, places: int) -> str:
    """Format the way the reports do: decimal comma, fixed places."""
    return f"{value:.{places}f}".replace(".", ",")


@pytest.fixture(scope="module")
def report() -> str:
    return _doc("final_report.md")


@pytest.fixture(scope="module")
def summary() -> str:
    return _doc("executive_summary.md")


@pytest.fixture(scope="module")
def paper() -> str:
    return _doc("paper_draft.md")


# ---------------------------------------------------------------------------
# The central corrected result
# ---------------------------------------------------------------------------


def test_the_headline_edge_is_the_one_the_confirmation_run_produced(
    report: str, summary: str
) -> None:
    audit = _artifact("full_dataset_confirmation.json")
    three_fold = next(f for f in audit["fold_results"] if f["folds"] == 3)
    quoted = _vn(three_fold["edge_qrw_minus_affine"], 6)

    assert quoted == "-0,013091"  # guards the formatting, not just the match
    assert quoted.lstrip("-") in report
    assert quoted.lstrip("-") in summary


def test_the_five_fold_edge_is_quoted_as_measured(report: str) -> None:
    audit = _artifact("full_dataset_confirmation.json")
    five_fold = next(f for f in audit["fold_results"] if f["folds"] == 5)

    assert _vn(five_fold["edge_qrw_minus_affine"], 6).lstrip("-") in report


def test_the_withdrawn_figure_is_quoted_exactly_as_the_artifact_records_it(
    report: str, summary: str
) -> None:
    """The bug's number matters as much as the correction; misquoting it would
    make the correction unauditable."""
    stale = _artifact("full_dataset_confirmation.json")["stale_prefix_edge_reference"]

    assert _vn(stale, 6) in report
    assert _vn(stale, 6) in summary


def test_the_row_count_behind_the_headline_edge_is_stated_correctly(
    summary: str,
) -> None:
    rows = _artifact("full_dataset_confirmation.json")["rows"]
    millions = rows / 1e6

    assert f"{millions:.1f}".replace(".", ",") + " triệu" in summary


# ---------------------------------------------------------------------------
# The replication on 3.1x the data
# ---------------------------------------------------------------------------


def test_the_replication_run_is_described_at_its_real_size(report: str) -> None:
    audit = _artifact("confirmation_btcusdt_69d_100M.json")

    assert audit["rows"] == 100_000_000
    assert "100 triệu" in report or "100M" in report


def test_every_fold_of_the_replication_is_quoted(report: str) -> None:
    """Quoting only the friendliest fold would be the kind of selection this
    project exists to avoid."""
    audit = _artifact("confirmation_btcusdt_69d_100M.json")

    for fold in audit["fold_results"]:
        quoted = _vn(fold["edge_qrw_minus_affine"], 6).lstrip("-")
        assert quoted in report, f"folds={fold['folds']} edge {quoted} not in report"


# ---------------------------------------------------------------------------
# Trading feasibility
# ---------------------------------------------------------------------------


def test_the_cost_to_move_ratio_headline_matches_the_feasibility_run(
    summary: str,
) -> None:
    """The summary's most quotable number: cost exceeds the move by N times."""
    audit = _artifact("horizon_feasibility_BTCUSDT.json")
    one_tick = next(h for h in audit["analysis"]["horizons"] if h["horizon_ticks"] == 1)
    ratio = one_tick["scenarios"]["taker_repo_5bps"]["move_to_cost_ratio"]
    times = round(1.0 / ratio)

    assert f"{times:,}".replace(",", ".") in summary


def test_the_roll_estimator_is_the_one_named_in_the_reports(report: str) -> None:
    """The superseded VWAP-dispersion figure must not be what the prose cites."""
    audit = _artifact("horizon_feasibility_BTCUSDT.json")

    assert audit["analysis"]["half_spread_estimator"] == "roll_1984"
    assert "Roll" in report


def test_the_maker_side_is_reported_as_paying_not_earning(
    report: str, summary: str
) -> None:
    """The sign of the realised half-spread is the whole self-refutation.

    Both documents have to carry it: the summary is where a reader who never
    opens the full report forms their view of the trading result.
    """
    for asset in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        audit = _artifact(f"horizon_feasibility_{asset}.json")
        one_tick = next(
            h for h in audit["analysis"]["horizons"] if h["horizon_ticks"] == 1
        )
        assert one_tick["realised_half_spread"] < 0, asset

    assert "realised half-spread âm" in report
    assert "sai dấu" in summary


# ---------------------------------------------------------------------------
# The volatility claim
# ---------------------------------------------------------------------------


def test_the_volatility_verdict_in_the_docs_matches_the_measurement(
    report: str, summary: str
) -> None:
    audit = _artifact("volatility_claim.json")
    combined = audit["combined"]

    assert combined["supports_claim"] is False
    for text in (report, summary):
        assert _vn(combined["fisher_p"], 3) in text
        assert _vn(combined["stouffer_p"], 3) in text


def test_the_docs_do_not_claim_the_volatility_story_was_confirmed(
    report: str, summary: str
) -> None:
    """It runs the asserted way on all three assets but is not established.

    The failure mode is prose drifting back to stating it as a finding once the
    direction looks agreeable.
    """
    audit = _artifact("volatility_claim.json")
    assert audit["combined"]["all_in_claimed_direction"] is True

    for text in (report, summary):
        assert "nghiêng về" in text


def test_each_asset_spearman_is_quoted_as_measured(report: str) -> None:
    audit = _artifact("volatility_claim.json")

    for asset in audit["per_asset"]:
        assert _vn(asset["spearman"], 3) in report, asset["label"]


def test_the_window_count_the_claim_was_retested_on_is_stated(
    report: str, summary: str
) -> None:
    """Five windows was the problem; the reader has to be told it is now forty."""
    audit = _artifact("volatility_claim.json")
    windows = {a["windows_used"] for a in audit["per_asset"]}

    assert windows == {40}
    for text in (report, summary):
        assert "40 window" in text


# ---------------------------------------------------------------------------
# Self-refutation count -- quoted in three places, easy to leave stale
# ---------------------------------------------------------------------------


def test_the_self_refutation_count_agrees_across_the_summary_and_dashboard(
    summary: str,
) -> None:
    from src.dashboard.plain_language import WHY_NEGATIVE_MATTERS

    spelled = "sáu lần tự bác bỏ"
    assert spelled in summary
    assert any(spelled in entry for entry in WHY_NEGATIVE_MATTERS)


def test_the_summary_lists_as_many_refutations_as_it_counts(summary: str) -> None:
    """The count and the numbered list drift apart when one gets appended to."""
    body = summary.split("lần tự bác bỏ chính mình")[1]
    listed = [n for n in range(1, 10) if f"\n{n}. " in body[:2500]]

    assert listed == list(range(1, 7))


# ---------------------------------------------------------------------------
# The paper draft -- the document that went stale, so it gets its own guard
# ---------------------------------------------------------------------------

# It described the project at 1,908 ticks / 118.5s with the full dataset unrun
# for ten-plus iterations after that stopped being true. These pin it to the
# artifacts so it cannot silently fall behind them again.


def test_the_paper_carries_the_current_dataset_size_not_the_old_one(
    paper: str,
) -> None:
    assert "493,7 triệu" in paper
    assert "69 ngày" in paper
    # The tiny early dataset must not be described as the current evidence.
    assert "1.908" not in paper
    assert "118,5" not in paper


def test_the_paper_headline_edge_is_the_corrected_full_dataset_one(
    paper: str,
) -> None:
    audit = _artifact("full_dataset_confirmation.json")
    three_fold = next(f for f in audit["fold_results"] if f["folds"] == 3)

    assert _vn(three_fold["edge_qrw_minus_affine"], 6).lstrip("-") in paper
    # The superseded subset figure and the "could not rerun" excuse are gone.
    assert "0,007383" not in paper
    assert "chưa chạy lại được trên toàn bộ dataset" not in paper


def test_the_paper_does_not_relabel_itself_confirmatory(paper: str) -> None:
    assert "exploratory" in paper.lower()
    assert "confirmatory" in paper.lower()  # only ever as the thing not yet done


def test_the_paper_agrees_with_the_summary_on_the_refutation_count(
    paper: str,
) -> None:
    assert "sáu lần" in paper

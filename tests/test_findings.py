"""Tests for the research-findings tab.

The chart makes a claim in its headline -- that no horizon closes the gap. The
data it draws has to still say that, and the figures have to still come from
the report artifacts rather than from prose someone typed once.
"""

from __future__ import annotations

import json

import pytest

from src.dashboard import findings as fd


def _write_report(directory, asset: str, horizons: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"horizon_edge_{asset}.json").write_text(
        json.dumps({"analysis": {"horizons": horizons}}), encoding="utf-8"
    )


def _horizon(ticks: int, achieved: float, required: float, **extra) -> dict:
    entry = {
        "horizon_ticks": ticks,
        "best_accuracy": achieved,
        "breakeven_accuracy": {fd.FEE_SCENARIO: required},
        "net_edge_per_trade": {fd.FEE_SCENARIO: -0.0005},
        "seconds": 26.2,
        "n_test": 1234,
    }
    entry.update(extra)
    return entry


def test_rows_are_read_from_the_report_artifacts(tmp_path) -> None:
    _write_report(tmp_path, "BTCUSDT", [_horizon(1_000, 0.644, 1.639)])

    rows = fd.load_horizon_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0].achieved == pytest.approx(64.4)
    assert rows[0].required == pytest.approx(163.9)
    assert rows[0].shortfall == pytest.approx(99.5)


def test_a_skipped_horizon_is_not_plotted(tmp_path) -> None:
    """A horizon without enough windows has no accuracy to draw."""
    _write_report(
        tmp_path,
        "BTCUSDT",
        [_horizon(1_000, 0.644, 1.639), {"horizon_ticks": 50_000, "skipped": "too few"}],
    )

    assert [row.horizon for row in fd.load_horizon_rows(tmp_path)] == [1_000]


def test_missing_reports_yield_no_rows_rather_than_an_error(tmp_path) -> None:
    assert fd.load_horizon_rows(tmp_path) == []


def test_an_unreadable_report_is_skipped_not_raised(tmp_path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "horizon_edge_BTCUSDT.json").write_text("{not json", encoding="utf-8")
    assert fd.load_horizon_rows(tmp_path) == []


def test_the_published_reports_still_support_the_headline() -> None:
    """The tab says no horizon closes the gap. The data must still agree.

    If a rerun ever produced a viable cell, this fails rather than letting the
    dashboard keep asserting something its own chart contradicts.
    """
    rows = fd.load_horizon_rows()
    assert rows, "the edge reports should be generated in this repository"
    viable = [row.label for row in rows if row.shortfall <= 0]
    assert not viable, f"headline says none are viable, but {viable} are"


def test_the_figure_reads_as_two_labelled_series() -> None:
    rows = fd.load_horizon_rows()
    figure = fd.build_gap_figure(rows)

    assert len(figure.data) == 2
    assert [trace.name for trace in figure.data] == ["Đạt được", "Cần có để hoà vốn"]
    # One connector per row, plus the two reference rules.
    assert len(figure.layout.shapes) == len(rows) + 2
    # Reading order: the first row must sit at the top.
    assert figure.layout.yaxis.autorange == "reversed"


def test_the_series_colours_are_the_validated_pair() -> None:
    """Swapping in the accent red would reuse a status token for identity.

    These two hexes passed the dataviz checker against this dashboard's plot
    surface; the accent red/green mean good/bad here and are not free to use.
    """
    figure = fd.build_gap_figure(fd.load_horizon_rows())
    used = {trace.marker.color for trace in figure.data}

    assert used == {fd.ACHIEVED_COLOR, fd.REQUIRED_COLOR}
    from src.dashboard.design_system import COLORS

    assert COLORS["accent_red"] not in used
    assert COLORS["accent_green"] not in used


def test_the_axis_leaves_the_coin_flip_rule_inside_the_plot() -> None:
    """At the axis edge the 50% rule reads as the axis, not as a reference."""
    figure = fd.build_gap_figure(fd.load_horizon_rows())
    low, high = figure.layout.xaxis.range

    assert low < 50.0
    assert high > max(row.required for row in fd.load_horizon_rows())

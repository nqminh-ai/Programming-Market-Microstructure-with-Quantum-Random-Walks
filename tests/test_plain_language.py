"""Tests for the dashboard's plain-language layer.

The point of this layer is that a non-technical visitor can read the dashboard
without being misled. Two things must therefore hold and stay holding: every
panel carries its own caveat, and the demo disclaimer keeps naming the specific
numbers a lay reader would otherwise take as evidence of profitability.
"""

from __future__ import annotations

import re

from src.dashboard import plain_language as pl
from src.dashboard.platform import tab_volatility  # noqa: F401  (import smoke)


def test_every_tab_guide_is_complete() -> None:
    for key, guide in pl.TAB_GUIDES.items():
        assert guide["title"], key
        assert guide["what"], key
        assert guide["how"], key
        # A caveat is not optional: an easier-to-read number is also an
        # easier-to-misread number.
        assert guide["caveat"], key
        assert isinstance(guide["how"], list) and len(guide["how"]) >= 2, key


def test_tab_guides_cover_exactly_the_rendered_tabs() -> None:
    assert set(pl.TAB_GUIDES) == {
        "volatility",
        "risk",
        "signal",
        "optimizer",
        "anomaly",
    }


def test_trading_tab_warns_it_is_not_investment_advice() -> None:
    caveat = str(pl.TAB_GUIDES["signal"]["caveat"]).upper()
    assert "KHÔNG PHẢI KHUYẾN NGHỊ ĐẦU TƯ" in caveat


def test_demo_disclaimer_states_the_strategy_loses_money() -> None:
    """The corrected demo has profit factor 0.095 and an annualised Sharpe of -48.6.

    A lay reader must be told the outcome directly, not left to infer it from a
    ratio, so the disclaimer states the loss in words and by value.
    """
    body = " ".join(pl.DEMO_WARNING_BODY)
    assert "LỖ" in body
    assert "0,095" in body
    assert "−4,2%" in body


def test_demo_disclaimer_records_the_metric_bug_that_inflated_the_old_figures() -> None:
    """The 4.9 / 265 figures were a measurement bug, and saying so is the point.

    Silently swapping in corrected numbers would hide the most instructive part:
    a broken backtest flatters itself, because errors in the favourable
    direction rarely get audited.
    """
    body = " ".join(pl.DEMO_WARNING_BODY)
    assert "4,9" in body and "265" in body
    assert "651" in body and "19" in body
    assert "phí giao dịch" in body.lower()


def test_glossary_defines_the_terms_the_dashboard_actually_shows() -> None:
    defined = {term.split(" (")[0].strip().lower() for term, _ in pl.GLOSSARY}
    for required in ("qrw", "sharpe", "profit factor", "drawdown", "overfitting"):
        assert required in defined, required


def test_glossary_does_not_explain_jargon_with_more_jargon() -> None:
    """Definitions must not lean on undefined acronyms."""
    defined_terms = " ".join(term for term, _ in pl.GLOSSARY).lower()
    for term, meaning in pl.GLOSSARY:
        for acronym in re.findall(r"\b[A-Z]{3,}\b", meaning):
            if acronym in {"KHÔNG", "QRW"}:
                continue
            assert acronym.lower() in defined_terms, (term, acronym)


def test_project_answer_states_the_negative_result() -> None:
    """The landing copy must not soften the finding into a positive one."""
    assert "không" in pl.PROJECT_ANSWER.lower()
    combined = " ".join(answer for _, answer in pl.WHAT_IS_QRW).lower()
    assert "không đóng góp gì" in combined
    assert "thua" in combined


def _lay_copy() -> str:
    """Everything a visitor reads on the landing tab, as one string.

    Questions and glossary headwords included, not just the bodies -- a visitor
    reads those too, and some of what matters is stated in them.
    """
    parts: list[str] = []
    for question, answer in pl.WHAT_IS_QRW:
        parts += [question, answer]
    parts += list(pl.WHY_NEGATIVE_MATTERS)
    for term, meaning in pl.GLOSSARY:
        parts += [term, meaning]
    return "\n".join(parts)


def test_the_fee_multiple_quoted_to_visitors_matches_the_report() -> None:
    """The copy states a specific multiple; it has to be the one on file.

    This number has already moved once -- correcting the spread estimator
    changed it from about 2,000x to 1,610x -- and prose does not get rerun the
    way a report artifact does.
    """
    import json
    from pathlib import Path

    report = json.loads(
        Path("reports/research/horizon_feasibility_BTCUSDT.json").read_text(
            encoding="utf-8"
        )
    )
    one_tick = next(
        row
        for row in report["analysis"]["horizons"]
        if row["horizon_ticks"] == 1
    )
    multiple = 1.0 / one_tick["scenarios"]["taker_repo_5bps"]["move_to_cost_ratio"]
    # Vietnamese thousands separator.
    expected = f"{round(multiple / 10) * 10:,.0f}".replace(",", ".")

    assert expected in _lay_copy(), (
        f"copy should quote {expected}x, the report says {multiple:,.0f}x"
    )


def test_visitors_are_told_a_perfect_forecast_still_loses() -> None:
    """The most counter-intuitive finding, and the easiest one to omit."""
    copy = _lay_copy()
    assert "ĐÚNG 100%" in copy or "đúng 100%" in copy
    assert "vẫn **lỗ**" in copy or "vẫn lỗ" in copy


def test_the_passive_order_explanation_has_the_sign_the_data_shows() -> None:
    """Textbooks say a resting order earns the spread; here it pays.

    Getting this backwards would restate the assumption the project disproved.
    """
    copy = _lay_copy()
    assert "adverse selection" in copy.lower()
    assert "mất" in copy, "a resting order loses; saying it earns inverts the finding"


def test_visitors_are_told_nothing_is_confirmed() -> None:
    """The exploratory label is the honest frame, not a footnote."""
    joined = "\n".join(pl.WHY_NEGATIVE_MATTERS)
    assert "thăm dò" in joined
    assert "20 ngày" in joined

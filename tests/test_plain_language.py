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


def test_demo_disclaimer_names_the_misleading_metrics() -> None:
    """Sharpe ~4.9 and profit factor ~265 come from a 1,000-row demo.

    Those are the two figures a lay reader is most likely to read as proof the
    model makes money, so the disclaimer must call them out by value rather
    than warning vaguely about "demo data".
    """
    body = " ".join(pl.DEMO_WARNING_BODY)
    assert "1.000" in body
    assert "4,9" in body
    assert "265" in body


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

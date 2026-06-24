"""Render docs/research_paper.md to a paginated, IMRAD-formatted academic PDF.

markdown -> HTML -> xhtml2pdf (pure Python, no system deps). Body text uses
Times New Roman 13pt, 1.5 line spacing, A4 with academic margins (top 2.7cm,
bottom 3cm, left 3.2cm, right 1.8cm) and a 1.25cm first-line indent, per the
target thesis-formatting specification. Fenced code blocks use DejaVu Sans Mono
(full Unicode coverage) so formal equations render exactly; a small set of math
glyphs that Times New Roman lacks (tensor, bra-ket, proportional, element-of)
are substituted with ASCII-standard equivalents in non-code text only.

Usage:
    python scripts/render_paper_pdf.py [--md docs/research_paper.md] [--out docs/research_paper.pdf]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import markdown as md
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
WIN_FONTS = Path("C:/Windows/Fonts")


def _dejavu_mono() -> Path:
    import matplotlib
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf"


CSS_TEMPLATE = """
@font-face {{ font-family: "Times"; src: url("{times}"); }}
@font-face {{ font-family: "Times"; src: url("{timesbd}"); font-weight: bold; }}
@font-face {{ font-family: "Times"; src: url("{timesi}"); font-style: italic; }}
@font-face {{ font-family: "Mono"; src: url("{mono}"); }}
@page {{ size: a4; margin: 2.7cm 1.8cm 3.0cm 3.2cm; }}
body {{ font-family: "Times"; font-size: 13pt; line-height: 1.5; color: #000; }}
h1 {{ font-size: 16pt; page-break-before: always; margin: 6pt 0 10pt 0; }}
h1:first-of-type {{ page-break-before: avoid; }}
h2 {{ font-size: 14pt; margin: 14pt 0 6pt 0; }}
h3 {{ font-size: 13pt; margin: 10pt 0 6pt 0; }}
p {{ text-align: justify; text-indent: 1.25cm; margin: 6pt 0; }}
li {{ margin: 3pt 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 9.5pt; }}
th, td {{ border: 0.5px solid #777; padding: 3pt 5pt; text-align: left; }}
th {{ background: #e8e8e8; font-weight: bold; }}
code, pre {{ font-family: "Mono"; font-size: 9pt; }}
pre {{ background: #f5f5f5; padding: 7pt; border: 0.5px solid #ddd; white-space: pre-wrap; margin: 8pt 0; }}
blockquote {{ background: #f3f7fb; border-left: 3px solid #5588bb; padding: 5pt 11pt; margin: 8pt 0; text-indent: 0; }}
blockquote p {{ text-indent: 0; }}
img {{ max-width: 540px; }}
h1, h2, h3, th, td, li {{ text-indent: 0; }}
"""


def build_css() -> str:
    return CSS_TEMPLATE.format(
        times=(WIN_FONTS / "times.ttf").as_uri(),
        timesbd=(WIN_FONTS / "timesbd.ttf").as_uri(),
        timesi=(WIN_FONTS / "timesi.ttf").as_uri(),
        mono=_dejavu_mono().as_uri(),
    )


# Glyphs absent from Times New Roman -> ASCII-standard equivalents (applied to
# non-code text only; code blocks render in DejaVu Mono and keep the originals).
GLYPH_FIX = {
    "⊗": "(x)",   # tensor product
    "⟨": "<",     # left angle bracket (bra)
    "⟩": ">",     # right angle bracket (ket)
    "∝": "~",     # proportional to
    "∈": " in ",  # element of
}


def _fix_glyphs_outside_pre(html: str) -> str:
    parts = re.split(r"(<pre.*?</pre>)", html, flags=re.DOTALL)
    for i, seg in enumerate(parts):
        if seg.startswith("<pre"):
            continue
        for bad, good in GLYPH_FIX.items():
            seg = seg.replace(bad, good)
        parts[i] = seg
    return "".join(parts)


def link_callback(uri: str, rel: str) -> str:
    if uri.startswith(("http://", "https://", "file:")):
        return uri
    candidate = (ROOT / "docs" / uri).resolve()
    if candidate.exists():
        return str(candidate)
    return str((ROOT / uri.lstrip("./")).resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=Path, default=ROOT / "docs" / "research_paper.md")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "research_paper.pdf")
    args = parser.parse_args()

    text = args.md.read_text(encoding="utf-8")
    html_body = md.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "toc"])
    html_body = _fix_glyphs_outside_pre(html_body)
    html = f"<html><head><style>{build_css()}</style></head><body>{html_body}</body></html>"

    with args.out.open("wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, link_callback=link_callback, encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF generation reported {result.err} error(s)")

    from pypdf import PdfReader
    pages = len(PdfReader(str(args.out)).pages)
    print(f"Rendered {args.out} ({pages} pages)")


if __name__ == "__main__":
    main()

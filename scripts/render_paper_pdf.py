"""Render docs/research_paper.md to a paginated PDF.

Uses markdown -> HTML -> xhtml2pdf (pure Python, no system deps). A full-Unicode
TrueType font (DejaVu Sans, shipped with matplotlib) is registered so Vietnamese
diacritics and mathematical symbols render correctly. Section headings (h1) start
on a new page. Figures referenced as ../figures/*.png are resolved to absolute
paths via a link callback.

Usage:
    python scripts/render_paper_pdf.py [--md docs/research_paper.md] [--out docs/research_paper.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import markdown as md
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]


def _dejavu_dir() -> Path:
    import matplotlib
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


CSS_TEMPLATE = """
@font-face {{ font-family: "DejaVu"; src: url("{regular}"); }}
@font-face {{ font-family: "DejaVu"; src: url("{bold}"); font-weight: bold; }}
@font-face {{ font-family: "DejaVuMono"; src: url("{mono}"); }}
@page {{ size: a4; margin: 2.2cm 2.0cm 2.2cm 2.0cm; }}
body {{ font-family: "DejaVu"; font-size: 11.5px; line-height: 1.62; color: #111; }}
h1 {{ font-size: 19px; page-break-before: always; border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 6px; }}
h1:first-of-type {{ page-break-before: avoid; }}
h2 {{ font-size: 14.5px; margin-top: 16px; }}
h3 {{ font-size: 12.5px; margin-top: 10px; }}
p {{ text-align: justify; margin: 7px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.6px; }}
th, td {{ border: 0.5px solid #999; padding: 4px 6px; text-align: left; }}
th {{ background: #ececec; font-weight: bold; }}
code, pre {{ font-family: "DejaVuMono"; font-size: 9.6px; background: #f4f4f4; }}
pre {{ padding: 7px; border: 0.5px solid #ddd; white-space: pre-wrap; margin: 8px 0; }}
blockquote {{ background: #f3f7fb; border-left: 3px solid #5588bb; padding: 6px 12px; color: #223; margin: 8px 0; }}
img {{ max-width: 580px; }}
.figcaption {{ font-size: 9px; color: #555; font-style: italic; }}
"""


def build_css() -> str:
    d = _dejavu_dir()
    return CSS_TEMPLATE.format(
        regular=(d / "DejaVuSans.ttf").as_uri(),
        bold=(d / "DejaVuSans-Bold.ttf").as_uri(),
        mono=(d / "DejaVuSansMono.ttf").as_uri(),
    )


def link_callback(uri: str, rel: str) -> str:
    """Resolve relative figure paths (../figures/..) to absolute filesystem paths."""
    if uri.startswith(("http://", "https://", "file:")):
        return uri
    candidate = (ROOT / "docs" / uri).resolve()
    if candidate.exists():
        return str(candidate)
    alt = (ROOT / uri.lstrip("./")).resolve()
    return str(alt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=Path, default=ROOT / "docs" / "research_paper.md")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "research_paper.pdf")
    args = parser.parse_args()

    text = args.md.read_text(encoding="utf-8")
    html_body = md.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
    )
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

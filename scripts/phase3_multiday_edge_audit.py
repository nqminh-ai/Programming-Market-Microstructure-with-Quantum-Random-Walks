"""Backward-compatible entry point for the multi-day edge audit."""

from __future__ import annotations

from scripts.audits.phase3_multiday_edge_audit import *  # noqa: F401,F403
from scripts.audits.phase3_multiday_edge_audit import main


if __name__ == "__main__":
    main()

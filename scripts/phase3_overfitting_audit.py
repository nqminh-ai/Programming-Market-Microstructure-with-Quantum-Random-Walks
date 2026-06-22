"""Backward-compatible entry point for the Phase 3 overfitting audit."""

from __future__ import annotations

from scripts.audits.phase3_overfitting_audit import *  # noqa: F401,F403
from scripts.audits.phase3_overfitting_audit import main


if __name__ == "__main__":
    main()

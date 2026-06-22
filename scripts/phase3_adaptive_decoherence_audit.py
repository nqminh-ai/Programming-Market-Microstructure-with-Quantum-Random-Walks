"""Backward-compatible entry point for the adaptive decoherence audit."""

from __future__ import annotations

from scripts.audits.phase3_adaptive_decoherence_audit import *  # noqa: F401,F403
from scripts.audits.phase3_adaptive_decoherence_audit import main


if __name__ == "__main__":
    main()

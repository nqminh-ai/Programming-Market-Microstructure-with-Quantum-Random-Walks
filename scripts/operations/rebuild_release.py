"""Rebuild Phase 3-6 artifacts and freeze one auditable manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipelines.phase3_pipeline import resolve_feature_path


MODULES = (
    "scripts.pipelines.phase3_pipeline",
    "scripts.audits.phase3_overfitting_audit",
    "scripts.pipelines.phase4_pipeline",
    "scripts.pipelines.phase5_pipeline",
    "scripts.pipelines.phase6_pipeline",
    "scripts.operations.freeze_release",
)


def rebuild(feature_path: Path) -> None:
    """Run every phase sequentially; stop immediately on any failed gate."""
    for module in MODULES:
        command = [
            sys.executable,
            "-m",
            module,
            "--feature-path",
            str(feature_path),
        ]
        print(f"Running: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rebuild(resolve_feature_path(args.feature_path).resolve())


if __name__ == "__main__":
    main()

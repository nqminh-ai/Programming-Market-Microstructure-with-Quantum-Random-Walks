"""Release provenance and SHA-256 guards for research artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


GENERATED_PREFIXES = (
    ".pytest_cache/",
    ".test-cache/",
    ".test-tmp/",
    "data/",
    "figures/",
    "reports/",
    "results/",
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: str | Path) -> str:
    """Return the full commit id for ``repo_root``."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"unexpected Git commit id: {commit!r}")
    return commit


def _status_path(line: str) -> str:
    value = line[3:].strip().replace("\\", "/")
    if " -> " in value:
        value = value.split(" -> ", 1)[1]
    return value.strip('"')


def release_dirty_paths(repo_root: str | Path) -> list[str]:
    """Return dirty source/release paths, excluding generated data artifacts."""
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=Path(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    dirty: list[str] = []
    for line in completed.stdout.splitlines():
        path = _status_path(line)
        if not path or path.startswith(GENERATED_PREFIXES):
            continue
        dirty.append(path)
    return sorted(dirty)


def canonical_repo_path(path: str | Path, repo_root: str | Path) -> str:
    """Return a stable POSIX-style path, relative to the repository if possible."""
    source = Path(path).resolve()
    root = Path(repo_root).resolve()
    try:
        return source.relative_to(root).as_posix()
    except ValueError:
        return source.as_posix()


def collect_provenance(
    feature_path: str | Path,
    *,
    protocol_version: str,
    repo_root: str | Path,
    require_clean: bool = True,
) -> dict[str, str]:
    """Collect immutable identifiers for one official research run."""
    root = Path(repo_root).resolve()
    source = Path(feature_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if require_clean:
        dirty = release_dirty_paths(root)
        if dirty:
            preview = ", ".join(dirty[:8])
            suffix = " ..." if len(dirty) > 8 else ""
            raise RuntimeError(
                "official artifact generation requires a clean source tree; "
                f"dirty paths: {preview}{suffix}"
            )
    return {
        "protocol_version": str(protocol_version),
        "code_commit": git_commit(root),
        "feature_path": canonical_repo_path(source, root),
        "feature_sha256": sha256_file(source),
    }


def validate_provenance(
    provenance: Mapping[str, Any],
    feature_path: str | Path,
    *,
    protocol_version: str,
    repo_root: str | Path,
    expected_commit: str | None = None,
) -> dict[str, str]:
    """Hard-fail when an artifact does not match code, protocol, path, or data."""
    root = Path(repo_root).resolve()
    source = Path(feature_path).resolve()
    expected = {
        "protocol_version": str(protocol_version),
        "code_commit": expected_commit or git_commit(root),
        "feature_path": canonical_repo_path(source, root),
        "feature_sha256": sha256_file(source),
    }
    mismatches = [
        f"{key}: expected {value!r}, found {provenance.get(key)!r}"
        for key, value in expected.items()
        if provenance.get(key) != value
    ]
    if mismatches:
        raise ValueError("artifact provenance mismatch:\n- " + "\n- ".join(mismatches))
    return expected


def build_sha256_manifest(
    *,
    repo_root: str | Path,
    provenance: Mapping[str, str],
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
) -> dict[str, Any]:
    """Build a deterministic manifest for every declared input and output."""
    root = Path(repo_root).resolve()

    def entries(paths: Iterable[str | Path]) -> list[dict[str, str | int]]:
        rows: list[dict[str, str | int]] = []
        for value in sorted({Path(path).resolve() for path in paths}, key=str):
            if not value.is_file():
                raise FileNotFoundError(value)
            rows.append(
                {
                    "path": canonical_repo_path(value, root),
                    "sha256": sha256_file(value),
                    "bytes": value.stat().st_size,
                }
            )
        return rows

    return {
        "provenance": dict(provenance),
        "inputs": entries(inputs),
        "outputs": entries(outputs),
    }


def write_sha256_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Persist a canonical JSON release manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination

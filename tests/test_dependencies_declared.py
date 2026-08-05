"""Every third-party module the code imports must be declared in the lock.

CI installs exactly ``requirements.lock`` and nothing else. A developer machine
has whatever has accumulated on it, so an import that is missing from the lock
passes locally and fails only on the runner -- which is how two of them survived
here: ``statsmodels``, imported by a cross-check test, and ``ssi_fc_data``,
reached through a REST fallback path that one test exercises. Both were absent
from requirements.txt and therefore from the lock resolved out of it. Six CI runs
failed on the same step before anyone read the step name.

A guarded import does not exempt a package. ``ssi_live_collector`` wraps its
imports in try/except and re-raises a helpful RuntimeError, which converts a
missing dependency from one kind of failure into another rather than removing it.

The check is deliberately static. Resolving a module to its distribution by
asking the installed metadata sounds more precise and is worse: it reports every
package that merely failed to install as undeclared. An early draft of this file
did exactly that and accused torch, streamlit and scikit-learn -- all three
pinned -- because the environment it ran in was incomplete. Reading the lock as
text cannot make that mistake.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("src", "tests", "scripts")
FIRST_PARTY = {"src", "tests", "scripts", "config", "data"}

# Import name on the left, distribution name on the right, for the cases where
# they differ. A new module whose names differ fails the test until it is added
# here, which is the intended forcing function rather than a maintenance burden.
DISTRIBUTION_OF = {
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "websocket": "websocket-client",
    "yaml": "pyyaml",
}


def normalise(name: str) -> str:
    """PEP 503 normalisation, so ssi_fc_data and ssi-fc-data are one name."""
    return name.lower().replace("_", "-")


def pins(filename: str) -> set[str]:
    """Distribution names pinned in a requirements file, PEP 503 normalised."""
    text = (ROOT / filename).read_text(encoding="utf-8")
    return {
        normalise(line.split("==")[0].strip())
        for line in text.splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    }


def imported() -> dict[str, str]:
    """Top-level third-party modules, mapped to where each was first seen."""
    found: dict[str, str] = {}
    for name in SOURCE_ROOTS:
        root = ROOT / name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module] if node.module else []
                else:
                    continue
                for module in modules:
                    top = module.split(".")[0]
                    if top in sys.stdlib_module_names or top in FIRST_PARTY:
                        continue
                    found.setdefault(
                        top, f"{path.relative_to(ROOT).as_posix()}:{node.lineno}"
                    )
    return found


def undeclared_against(pinned: set[str]) -> list[str]:
    return [
        f"{module} ({where}) -> needs {normalise(DISTRIBUTION_OF.get(module, module))}"
        for module, where in sorted(imported().items())
        if normalise(DISTRIBUTION_OF.get(module, module)) not in pinned
    ]


def test_every_imported_package_is_pinned_in_the_lock() -> None:
    missing = undeclared_against(pins("requirements.lock"))
    assert not missing, (
        "imported but not pinned, so CI installs an environment the code cannot "
        "run in:\n  " + "\n  ".join(missing)
    )


def test_the_lock_is_a_superset_of_the_direct_requirements() -> None:
    """The lock is resolved from requirements.txt; a direct pin cannot be lost."""
    missing = sorted(pins("requirements.txt") - pins("requirements.lock"))
    assert not missing, f"in requirements.txt but not in the lock: {missing}"


def test_the_check_would_have_caught_the_failure_it_was_written_for() -> None:
    """A guard that cannot fail is decoration.

    Remove the two pins that actually broke CI and the check must name both, and
    only those two -- a check that flags everything is as useless as one that
    flags nothing.
    """
    broken = pins("requirements.lock") - {"statsmodels", "ssi-fc-data"}
    caught = {report.split(" ", 1)[0] for report in undeclared_against(broken)}
    assert caught == {"statsmodels", "ssi_fc_data"}

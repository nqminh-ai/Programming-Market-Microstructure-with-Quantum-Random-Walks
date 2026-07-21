"""Shared RNG coercion helper for the baseline/QRW simulators."""

from __future__ import annotations

import numpy as np


def as_generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Return `random_state` unchanged if already a Generator, else seed a new one."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)

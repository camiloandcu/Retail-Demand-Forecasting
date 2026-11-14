"""Random seed helpers used by every executable entry point."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy; interpreter hash randomization is launch-only."""

    random.seed(seed)
    np.random.seed(seed)

"""
engines/feature_access.py
──────────────────────────
Safe feature access for Layer 1 engines (Phase 2, Step 4)

Replaces the dangerous pattern:
    value = features.get("x", 0)   # silent default — missing = neutral!

With:
    value, real = self._feat(features, "x")
    if not real:
        self._n_missing += 1  # tracked, visible, impacts confidence

DESIGN
------
Each engine tracks how many features were missing during compute().
The combiner uses this to scale engine confidence down when data is sparse.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class FeatureAccessMixin:
    """
    Mixin for Layer 1 engines providing safe feature access.

    Usage:
        class MyEngine(FeatureAccessMixin):
            def compute(self, features):
                self._reset_access()
                val, ok = self._feat(features, "my_feature")
                ...
                return {... "access_completeness": self._access_completeness()}
    """

    def _reset_access(self) -> None:
        """Call at start of compute()."""
        self._n_accessed: int = 0
        self._n_missing: int = 0
        self._missing_names: list = []

    def _feat(
        self,
        features: pd.Series,
        name: str,
        neutral: float = 0.0,
    ) -> Tuple[float, bool]:
        """
        Safe feature access. Returns (value, is_real).

        If the feature is missing, NaN, or inf:
          - Returns (neutral, False)
          - Increments missing counter
          - Logs the miss

        NEVER silently returns a default that looks like real data.
        """
        self._n_accessed += 1
        val = features.get(name, None)

        # Missing entirely
        if val is None:
            self._n_missing += 1
            self._missing_names.append(name)
            return neutral, False

        # Type check
        if not isinstance(val, (int, float, np.integer, np.floating)):
            self._n_missing += 1
            self._missing_names.append(name)
            return neutral, False

        fval = float(val)

        # NaN or Inf
        if np.isnan(fval) or np.isinf(fval):
            self._n_missing += 1
            self._missing_names.append(name)
            return neutral, False

        return fval, True

    def _access_completeness(self) -> float:
        """Fraction of accessed features that were available."""
        if self._n_accessed == 0:
            return 1.0
        return (self._n_accessed - self._n_missing) / self._n_accessed

    def _log_missing(self, engine_name: str) -> None:
        """Log missing features if any."""
        if self._n_missing > 0:
            logger.debug(
                "[{}] {}/{} features missing: {}",
                engine_name,
                self._n_missing,
                self._n_accessed,
                ", ".join(self._missing_names[:5])
                + ("..." if self._n_missing > 5 else ""),
            )

"""
ml.research — Offline ML research tools for HYDRA.

CANONICAL ENTRY POINTS:
  python -m ml.research.run_v2       # Walk-forward evaluation
  python -m ml.research.evidence     # Full evidence package

v1 files have been removed. Only v2 pipeline is supported.
"""
from ml.research.walk_forward_v2 import (
    WalkForwardEngineV2,
    WalkForwardResultV2,
    FoldResult,
    FeatureStabilityReport,
    CostModel,
    COST,
)

__all__ = [
    "WalkForwardEngineV2",
    "WalkForwardResultV2",
    "FoldResult",
    "FeatureStabilityReport",
    "CostModel",
    "COST",
]

"""Inference engine: Fitter, Posterior, HierarchicalFitter, and backends."""

from tengri.inference.fitter import Fitter
from tengri.inference.hierarchical import HierarchicalFitter, HierarchicalResult
from tengri.inference.posterior import Posterior
from tengri.inference.raytrace import sample_raytrace
from tengri.inference.vi_config import VIConfig

__all__ = [
    "Fitter",
    "HierarchicalFitter",
    "HierarchicalResult",
    "Posterior",
    "VIConfig",
    "sample_raytrace",
]

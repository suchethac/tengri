"""Inference engine: Fitter, Posterior, PopulationFitter, and backends."""

from tengri.inference.backends.mcmc.raytrace import sample_raytrace
from tengri.inference.fitter import Fitter, resolve_method
from tengri.inference.hierarchical import HierarchicalFitter, HierarchicalResult
from tengri.inference.posterior import Posterior
from tengri.inference.vi_config import VIConfig

__all__ = [
    "Fitter",
    "HierarchicalFitter",
    "HierarchicalResult",
    "Posterior",
    "VIConfig",
    "resolve_method",
    "sample_raytrace",
]

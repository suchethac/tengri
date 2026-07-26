# SPDX-License-Identifier: BSD-3-Clause
"""Inference engine: Fitter, Posterior, PopulationFitter, and backends."""

# Populate the backend registry. Imported for side effects only —
# every ``@register_backend(...)`` call in this module installs a
# ``BackendEntry`` into ``_BACKENDS`` so that ``Fitter.run(method=...)``
# can dispatch by name. See ADR-0010.
from tengri.inference import _registration as _registration
from tengri.inference.backends.mcmc.raytrace import sample_raytrace
from tengri.inference.catalog_fitter import (
    CatalogPosterior,
    CatalogFitter,
)
from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter, resolve_method
from tengri.inference.hierarchical import PopulationFitter, PopulationPosterior
from tengri.inference.posterior import Posterior
from tengri.inference.vi_config import VIConfig

__all__ = [
    "CatalogFitter",
    "CatalogPosterior",
    "Fitter",
    "InferenceContext",
    "PopulationFitter",
    "PopulationPosterior",
    "Posterior",
    "VIConfig",
    "resolve_method",
    "sample_raytrace",
]

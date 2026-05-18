"""Fused SED kernels.

Kernel builders organized by fusion strategy:

- ``hybrid.py``: Precomputed stellar + exact non-stellar (fastest for photometry)
- ``exact.py``: Full-resolution dust and stellar CSP
- ``compositional.py``: Compositional pipeline with all components

Higher-level selection logic lives in :mod:`._protocol`, :mod:`._adapters`,
and :mod:`.strategy`. The strategy module is the seam that ``sed_model.py``
uses to choose among the seven concrete builders without scattering the
policy across hundreds of lines.
"""

from tengri.forward._kernels._adapters import (
    ALL_ADAPTERS,
    CompositionalPhotometryKernel,
    CompositionalRestSEDKernel,
    CompositionalSpectrumKernel,
    ExactRestSEDKernel,
    HybridPhotometryKernel,
    HybridPhotometryZTableKernel,
    HybridSpectrumKernel,
    adapters_by_name,
)
from tengri.forward._kernels._protocol import (
    Kernel,
    KernelBuildFailure,
    NoCompatibleKernelError,
    Product,
)
from tengri.forward._kernels.compositional import (
    build_fused_tier2_photometry,
    build_fused_tier2_spectrum,
    build_hybrid_spectrum,
    observe_photometry_from_rest_sed,
    observe_spectrum_from_rest_sed,
)
from tengri.forward._kernels.exact import build_exact_sed, build_fused_rest_sed
from tengri.forward._kernels.hybrid import (
    build_hybrid_photometry,
    build_hybrid_photometry_ztable,
)
from tengri.forward._kernels.strategy import (
    COMPOSITIONAL_ONLY,
    DEFAULT,
    EXACT_ONLY,
    LOW_MEMORY,
    KernelStrategy,
)

__all__ = [
    "ALL_ADAPTERS",
    "COMPOSITIONAL_ONLY",
    "DEFAULT",
    "EXACT_ONLY",
    "LOW_MEMORY",
    "CompositionalPhotometryKernel",
    "CompositionalRestSEDKernel",
    "CompositionalSpectrumKernel",
    "ExactRestSEDKernel",
    "HybridPhotometryKernel",
    "HybridPhotometryZTableKernel",
    "HybridSpectrumKernel",
    "Kernel",
    "KernelBuildFailure",
    "KernelStrategy",
    "NoCompatibleKernelError",
    "Product",
    "adapters_by_name",
    "build_exact_sed",
    "build_fused_rest_sed",
    "build_fused_tier2_photometry",
    "build_fused_tier2_spectrum",
    "build_hybrid_photometry",
    "build_hybrid_photometry_ztable",
    "build_hybrid_spectrum",
    "observe_photometry_from_rest_sed",
    "observe_spectrum_from_rest_sed",
]

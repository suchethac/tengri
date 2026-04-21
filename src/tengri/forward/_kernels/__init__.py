"""Fused SED kernels.

Kernel builders organized by fusion strategy:
- ``hybrid.py``: Precomputed stellar + exact non-stellar (fastest for photometry)
- ``exact.py``: Full-resolution dust and stellar CSP
- ``compositional.py``: Compositional pipeline with all components


"""

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

__all__ = [
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

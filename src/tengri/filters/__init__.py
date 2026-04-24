"""DEPRECATED: Filter discovery helpers — use tengri.observation.filters instead.

This module is a backward-compatibility shim. All symbols have been consolidated
into tengri.observation.filters. Importing from this module emits a DeprecationWarning.

Direct imports from tengri.observation.filters do NOT trigger the warning.
"""

import warnings

# Emit deprecation warning on module import (happens once per Python session)
warnings.warn(
    "The tengri.filters module is deprecated and will be removed in a future version. "
    "Please use tengri.observation.filters instead. "
    "Direct imports from tengri.observation.filters do not trigger this warning.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all public symbols from the canonical location
from tengri.observation.filters import (
    FILTER_REGISTRY,
    compute_effective_wavelength,
    compute_fwhm,
    describe,
    download_filter,
    filter_info,
    list_available_filters,
    list_filters,
    load,
    load_alma_band,
    load_custom_filter,
    load_filter,
    load_filter_set,
    load_tophat_filter,
    suggest,
)

__all__ = [
    "FILTER_REGISTRY",
    "compute_effective_wavelength",
    "compute_fwhm",
    "describe",
    "download_filter",
    "filter_info",
    "list_available_filters",
    "list_filters",
    "load",
    "load_alma_band",
    "load_custom_filter",
    "load_filter",
    "load_filter_set",
    "load_tophat_filter",
    "suggest",
]

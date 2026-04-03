"""Emission line catalog for tengri spectral fitting.

Single source of truth for emission line rest-frame wavelengths
and groupings. Imported by both ``eline_marginalization.py`` and
``eline_priors.py`` to eliminate duplicate/inconsistent line lists.

All wavelengths are rest-frame vacuum values in Angstrom.
"""

from __future__ import annotations

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Primary line catalog
# ---------------------------------------------------------------------------

# name: (rest_wavelength_aa, line_type, default_prior_width_dex)
EMISSION_LINES: dict[str, tuple[float, str, float]] = {
    "Lya": (1215.67, "recombination", 0.3),
    "OII3726": (3727.09, "forbidden", 0.3),
    "OII3729": (3729.88, "forbidden", 0.3),
    "Hdelta": (4102.89, "recombination", 0.3),
    "Hgamma": (4341.68, "recombination", 0.3),
    "Hbeta": (4862.68, "recombination", 0.15),
    "OIII4959": (4960.30, "forbidden", 0.15),
    "OIII5007": (5008.24, "forbidden", 0.15),
    "OI6300": (6300.30, "forbidden", 0.4),
    "NII6548": (6549.86, "forbidden", 0.3),
    "Halpha": (6564.61, "recombination", 0.15),
    "NII6583": (6585.28, "forbidden", 0.3),
    "SII6716": (6718.29, "forbidden", 0.35),
    "SII6731": (6732.67, "forbidden", 0.35),
}

# ---------------------------------------------------------------------------
# Named line groups
# ---------------------------------------------------------------------------

LINE_GROUPS: dict[str, list[str]] = {
    "optical_narrow": [
        "OII3726",
        "OII3729",
        "Hdelta",
        "Hgamma",
        "Hbeta",
        "OIII4959",
        "OIII5007",
        "NII6548",
        "Halpha",
        "NII6583",
        "SII6716",
        "SII6731",
    ],
    "bpt": ["Hbeta", "OIII5007", "Halpha", "NII6583"],
    "balmer": ["Lya", "Hbeta", "Halpha"],
    "blr_broad": ["Lya", "Hbeta", "Halpha"],
    "cloudy_default": [
        "OII3726",
        "OII3729",
        "Hdelta",
        "Hgamma",
        "Hbeta",
        "OIII4959",
        "OIII5007",
        "NII6548",
        "Halpha",
        "NII6583",
        "SII6716",
        "SII6731",
    ],
}

# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_line_wavelengths(group: str | list[str]) -> jnp.ndarray:
    """Return JAX array of wavelengths for a named group or list of line names.

    Parameters
    ----------
    group : str or list of str
        Named group (e.g. ``"bpt"``) or list of line names.

    Returns
    -------
    jnp.ndarray
        Rest-frame wavelengths [Angstrom] in catalog order.
    """
    if isinstance(group, str):
        names = LINE_GROUPS[group]
    else:
        names = list(group)
    return jnp.array([EMISSION_LINES[n][0] for n in names])


def get_line_names(group: str) -> tuple[str, ...]:
    """Return line names for a named group."""
    return tuple(LINE_GROUPS[group])


# ---------------------------------------------------------------------------
# Backward-compatibility arrays
# (Match the old DEFAULT_LINE_* and CLOUDY_LINE_* arrays so existing code
#  importing from eline_marginalization/eline_priors still works unchanged.)
# ---------------------------------------------------------------------------

# 13-line set (was DEFAULT_LINE_NAMES/DEFAULT_LINE_WAVELENGTHS in eline_marginalization.py)
DEFAULT_LINE_NAMES: tuple[str, ...] = (
    "Ly-alpha",
    "H-delta",
    "H-gamma",
    "H-beta",
    "[OIII]4959",
    "[OIII]5007",
    "H-alpha",
    "[NII]6548",
    "[NII]6583",
    "[OII]3726",
    "[OII]3729",
    "[SII]6717",
    "[SII]6731",
)
DEFAULT_LINE_WAVELENGTHS: jnp.ndarray = jnp.array(
    [
        1215.67,  # Ly-alpha (vacuum)
        4102.89,  # H-delta (vacuum)
        4341.68,  # H-gamma (vacuum)
        4862.68,  # H-beta (vacuum)
        4960.30,  # [OIII]4959 (vacuum)
        5008.24,  # [OIII]5007 (vacuum)
        6564.61,  # H-alpha (vacuum)
        6549.86,  # [NII]6548 (vacuum)
        6585.28,  # [NII]6583 (vacuum)
        3727.09,  # [OII]3726 (vacuum)
        3729.88,  # [OII]3729 (vacuum)
        6718.29,  # [SII]6717 (vacuum)
        6732.67,  # [SII]6731 (vacuum)
    ]
)

# 12-line CLOUDY set (was CLOUDY_LINE_NAMES/CLOUDY_LINE_WAVELENGTHS in eline_priors.py)
CLOUDY_LINE_NAMES: tuple[str, ...] = (
    "[OII]3726",
    "[OII]3729",
    "H-delta",
    "H-gamma",
    "H-beta",
    "[OIII]4959",
    "[OIII]5007",
    "[NII]6548",
    "H-alpha",
    "[NII]6583",
    "[SII]6716",
    "[SII]6731",
)
CLOUDY_LINE_WAVELENGTHS: jnp.ndarray = jnp.array(
    [
        3727.09,  # [OII] 3726 (vacuum)
        3729.88,  # [OII] 3729 (vacuum)
        4102.89,  # H-delta (vacuum)
        4341.68,  # H-gamma (vacuum)
        4862.68,  # H-beta (vacuum)
        4960.30,  # [OIII] 4959 (vacuum)
        5008.24,  # [OIII] 5007 (vacuum)
        6549.86,  # [NII] 6548 (vacuum)
        6564.61,  # H-alpha (vacuum)
        6585.28,  # [NII] 6583 (vacuum)
        6718.29,  # [SII] 6716 (vacuum)
        6732.67,  # [SII] 6731 (vacuum)
    ]
)

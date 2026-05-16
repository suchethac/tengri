# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Curated model recipes for common observational scenarios.

Provides ready-to-use model configurations for typical galaxy fitting workflows.
Each recipe returns a nested-dict suitable for splat ting into
:func:`~tengri.parameters.Parameters.from_groups()` or
:func:`~tengri.forward.SEDModel.from_groups()`.

Examples
--------
Fit a star-forming galaxy at low-intermediate redshift with photometry::

    from tengri import SEDModel, recipes
    import sps_data

    model = SEDModel.from_groups(
        ssp_data=sps_data, filters=my_filter_list, **recipes.star_forming_photometry()
    )

Fit a quiescent galaxy at z~0.05::

    model = SEDModel.from_groups(
        ssp_data=sps_data, filters=my_filter_list, **recipes.quiescent_z0()
    )
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform
from tengri.parameters.sentinels import FIXED, FREE

__all__ = [
    "agn_panchromatic",
    "mock_recovery_minimal",
    "quiescent_z0",
    "star_forming_photometry",
    "stochastic_sfh_jwst",
]


def star_forming_photometry() -> dict:
    """Recipe for star-forming galaxies with broadband photometry (0 < z < 6).

    Suitable for optical+NIR+MIR photometry fits of star-forming galaxies
    at low-to-intermediate redshift.

    **Configuration:**
    - **SFH**: Dual power-law (DPL) with all parameters free
    - **Dust**: Two-component Calzetti attenuation (both optical depths free)
    - **Dust IR emission**: Dale2014 templates (fixed)
    - **Nebular**: Cue neural emulator (fixed)
    - **Redshift**: Free with bounds (0.01, 6.0)
    - **IGM**: Applied (Inoue+2014)
    - **Metallicity**: Free

    Returns
    -------
    dict
        Nested-dict ready for Parameters.from_groups() or SEDModel.from_groups().

    Notes
    -----
    Designed for typical optical-through-MIR photometry. IGM absorption is
    included to handle Lyman-alpha forest opacity at high redshift. Dust
    attenuation spans realistic ranges for star-forming galaxies.

    Examples
    --------
    >>> from tengri import SEDModel, recipes
    >>> model = SEDModel.from_groups(ssp_data=ssp, **recipes.star_forming_photometry())
    >>> assert "sfh_dpl_alpha" in model.spec.free_params
    >>> assert "redshift" in model.spec.free_params
    """
    return dict(
        sfh={
            "type": "dpl",
            "*": FREE,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FREE,
            "emission": {
                "type": "dale2014",
                "*": FIXED,
            },
        },
        neb={
            "type": "cue",
            "*": FIXED,
        },
        redshift=Uniform(0.01, 6.0),
        apply_igm=True,
    )


def quiescent_z0() -> dict:
    """Recipe for quiescent galaxies at low redshift (z ~ 0.05).

    Suitable for local quiescent galaxy samples (e.g., SDSS passive galaxies).

    **Configuration:**
    - **SFH**: Delayed-exponential (dexp) with all parameters free
    - **Dust**: Two-component Calzetti attenuation (both optical depths free,
      lower bounds than star-forming recipe)
    - **Dust IR emission**: Disabled
    - **Nebular**: Cue neural emulator (fixed)
    - **Redshift**: Fixed at z=0.05
    - **Metallicity**: Free

    Returns
    -------
    dict
        Nested-dict ready for Parameters.from_groups() or SEDModel.from_groups().

    Notes
    -----
    Optimized for low-redshift quiescent galaxies where dust attenuation is
    minimal and SFH has largely ceased. The delayed-exponential model captures
    older stellar populations better than DPL for completely quenched systems.

    Examples
    --------
    >>> from tengri import recipes
    >>> params = recipes.quiescent_z0()
    >>> assert params["sfh"]["type"] == "dexp"
    >>> assert params["redshift"] == Fixed(0.05)
    """
    return dict(
        sfh={
            "type": "dexp",
            "*": FREE,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FREE,
            "tau_bc": Uniform(0, 0.5),
            "tau_diff": Uniform(0, 0.3),
        },
        neb={
            "type": "cue",
            "*": FIXED,
        },
        redshift=Fixed(0.05),
    )


def agn_panchromatic() -> dict:
    """Recipe for AGN-dominated galaxies with multi-wavelength data.

    Suitable for AGN host galaxy fitting using UV through radio data
    (panchromatic coverage).

    **Configuration:**
    - **SFH**: DPL (free)
    - **Dust**: Two-component Calzetti attenuation (free)
    - **Dust IR emission**: Dale2014 (free)
    - **Nebular**: Cue (fixed)
    - **AGN**:
        - **Disc**: Multicolor disc (free)
        - **Torus**: SKIRTOR clumpy torus model (free)
        - **Lines**: NLR emission (free)
    - **Radio**: Enabled (free)
    - **X-ray**: Enabled (free)
    - **Redshift**: Free

    Returns
    -------
    dict
        Nested-dict ready for Parameters.from_groups() or SEDModel.from_groups().

    Notes
    -----
    Includes AGN disc accretion, dust reprocessing, and emission line physics.
    Suitable for quasars, Seyferts, and LINERs with rich multi-wavelength coverage.
    Radio and X-ray enable synchrotron and Compton cooling components.

    Examples
    --------
    >>> from tengri import recipes
    >>> params = recipes.agn_panchromatic()
    >>> assert "agn" in params
    >>> assert "disc" in params["agn"]
    """
    return dict(
        sfh={
            "type": "dpl",
            "*": FREE,
        },
        dust={
            "type": "two_component",
            "*": FREE,
            "emission": {
                "type": "dale2014",
                "*": FREE,
            },
        },
        neb={
            "type": "cue",
            "*": FIXED,
        },
        agn={
            "disc": {
                "type": "multicolor",
                "*": FREE,
            },
            "torus": {
                "type": "skirtor",
                "*": FREE,
            },
            "lines": {
                "type": "nlr",
                "*": FREE,
            },
        },
        radio=True,
        xray=True,
        redshift=Uniform(0.01, 6.0),
    )


def stochastic_sfh_jwst() -> dict:
    """Recipe for JWST high-redshift galaxies with stochastic SFH.

    Suitable for JWST spectrophotometry of 0.5 < z < 12 galaxies
    where burstiness and temporal structure in star formation matter.

    **Configuration:**
    - **SFH**: DPL + stochastic field composition (both free)
    - **Dust**: Two-component Calzetti attenuation (free)
    - **Dust IR emission**: Dale2014 (fixed)
    - **Nebular**: Cue (fixed)
    - **Redshift**: Free with JWST-appropriate bounds (0.5, 12.0)
    - **IGM**: Applied (for Lyman opacity)
    - **Metallicity**: Free

    Returns
    -------
    dict
        Nested-dict ready for Parameters.from_groups() or SEDModel.from_groups().

    Notes
    -----
    Includes a stochastic field component to capture burstiness and SFH
    fluctuations common in high-redshift systems. IGM absorption is critical
    for accurate SED fitting above z ~ 0.5.

    Examples
    --------
    >>> from tengri import recipes
    >>> params = recipes.stochastic_sfh_jwst()
    >>> assert params["sfh"]["type"] == ["dpl", "field"]
    >>> assert params["apply_igm"] is True
    """
    return dict(
        sfh={
            "type": ["dpl", "field"],
            "*": FREE,
        },
        dust={
            "type": "two_component",
            "*": FREE,
            "emission": {
                "type": "dale2014",
                "*": FIXED,
            },
        },
        neb={
            "type": "cue",
            "*": FIXED,
        },
        redshift=Uniform(0.5, 12.0),
        apply_igm=True,
    )


def mock_recovery_minimal() -> dict:
    """Recipe for mock data recovery and benchmarking (minimal model).

    Suitable for fast mock data fits, parameter recovery tests, and
    forward-model benchmarking.

    **Configuration:**
    - **SFH**: Truncated skew-normal (tsnorm) with ~4 free params
    - **Dust**: Calzetti attenuation (tau_bc free)
    - **Dust IR emission**: Disabled (no PAH/continuum)
    - **Nebular**: Disabled
    - **Redshift**: Fixed at z=0.05 (local universe)
    - **Metallicity**: Free

    Returns
    -------
    dict
        Nested-dict ready for Parameters.from_groups() or SEDModel.from_groups().

    Notes
    -----
    Intentionally minimal to keep forward-model runtime short and memory
    footprint small. Useful for debugging, unit tests, and quick validation
    of inference algorithms. All non-essential physics disabled.

    Examples
    --------
    >>> from tengri import recipes, Parameters
    >>> params_dict = recipes.mock_recovery_minimal()
    >>> spec = Parameters.from_groups(**params_dict)
    >>> assert 4 <= spec.n_free <= 8  # ~5 SFH + dust + met
    >>> assert "redshift" in spec.fixed_params
    """
    return dict(
        sfh={
            "type": "tsnorm",
            "*": FREE,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "tau_bc": Uniform(0, 1),
        },
        neb={
            "type": "none",
        },
        redshift=Fixed(0.05),
    )

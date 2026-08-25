# SPDX-License-Identifier: BSD-3-Clause
"""Vetted factory functions for common galaxy types (legacy expert surface).

This module provides preset Parameter + SEDModelConfig tuples for typical galaxy
populations, avoiding the 10-line setup ritual. Each preset is a callable that
returns a fully-configured (Parameters, SEDModelConfig) pair ready for forward
modeling or inference. It is the surface consumed by :class:`tengri.Galaxy`
(``Galaxy.from_arrays(preset=...)``) via :func:`resolve_preset`.

For new code prefer the nested-dict grammar recipes in :mod:`tengri.recipes`
(``SEDModel.build(**recipes.high_z())``): the ``high_z`` and ``photoz``
recipes are grammar implementations of the presets here.

.. note::
    Relocated from the former top-level ``tengri/presets.py`` module (2026-07),
    which had been silently shadowed (and made unreachable) by this
    ``tengri.presets`` package since the package landed.

Usage
-----
>>> from tengri import presets
>>> params, config = presets.starforming(redshift=0.5)
>>> # Use with SEDModel: model = SEDModel(params, ssp_data, config, observation)

Preset Gallery
--------------

- ``starforming()``: Main sequence galaxies (z ~ 0–3): moderate dust, solar metallicity
- ``quiescent()``: Red/dead galaxies (z ~ 0–2): minimal dust, low metallicity spread
- ``high_z()``: Young galaxies (z > 4): strong nebular, younger ages, SMC-like dust
- ``photoz()``: Photometric-redshift surveys: wide z prior, uninformative SFH/dust
- ``jwst_spec()``: JWST NIRSpec spectroscopy: known/constrained z, moderate-high dust
- ``agn_host()``: AGN host galaxies: high dust for Type 2 AGN, no torus component

References
----------
.. [1] Calzetti, D., et al. 2000, ApJ 533, 682
       Attenuation and dust extinction in starburst galaxies.
.. [2] Charlot, S., & Fall, S. M. 2000, ApJ 539, 718
       A simple model for the absorption of radiation by dust in galaxies.
"""

from __future__ import annotations

from tengri.config.settings import (
    DustConfig,
    NebularConfig,
    SEDModelConfig,
    SFHConfig,
)
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform
from tengri.presets._registry import register_preset

__all__ = [
    "agn_host",
    "describe",
    "high_z",
    "jwst_spec",
    "photoz",
    "quiescent",
    "resolve_preset",
    "starforming",
]


@register_preset(
    "starforming",
    short_doc="Main-sequence star-forming galaxies (z ~ 0-3), Calzetti+CF00 dust",
    citations=["Calzetti_2000", "Charlot_2000"],
)
def starforming(redshift: float | None = None) -> tuple[Parameters, SEDModelConfig]:
    """Star-forming galaxy preset (main sequence, z ~ 0–3).

    Star-forming galaxies typically have:

    - Delayed-exponential or double power-law SFH
    - Moderate dust attenuation (Calzetti 2000 [1]_)
    - Solar or near-solar metallicity
    - Possible weak nebular emission from H II regions

    If redshift is not specified, it is left as a free parameter to support
    multi-redshift surveys.

    Parameters
    ----------
    redshift : float or None, optional
        Redshift of the galaxy. If None (default), redshift is a free parameter
        with Uniform(0.01, 6.0). If specified, redshift is fixed.

    Returns
    -------
    tuple[Parameters, SEDModelConfig]
        (params, config); fully configured Parameter spec and model settings.

    Notes
    -----
    **JIT-compatible**: no, returns configuration objects, not arrays.

    **Physics:**

    - SFH model: double power-law (``dpl``)
    - Dust: Calzetti (birth cloud) + power-law diffuse ISM (two-component,
      Charlot & Fall 2000 [2]_)
    - Attenuation: Av ~ Uniform(0, 2) mag
    - Metallicity: Uniform(-0.5, 0.3) [Z/Zsun]
    - Nebular emission: off (use ``high_z`` preset for stronger nebular)

    Examples
    --------
    >>> params, config = presets.starforming(redshift=0.5)
    >>> model = SEDModel(params, ssp_data, config, observation)

    Free-redshift fit:

    >>> params, config = presets.starforming()  # redshift free
    >>> assert "redshift" in params.free_params
    """
    if redshift is None:
        z_prior = Uniform(0.01, 6.0)
    else:
        z_prior = Fixed(redshift)

    params = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(8.0, 12.0),
        met_logzsol=Uniform(-0.5, 0.3),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=z_prior,
    )

    config = SEDModelConfig(
        sfh=SFHConfig(mean_type=("dpl",)),
        dust=DustConfig(
            model="two_component",
            law_bc="calzetti",
            law_diff="power_law",
            emission=None,
        ),
        nebular=NebularConfig(backend="off"),
    )

    return params, config


@register_preset(
    "quiescent",
    short_doc="Quiescent/red galaxies (z ~ 0-2), minimal power-law dust",
    citations=["Charlot_2000"],
)
def quiescent(redshift: float | None = None) -> tuple[Parameters, SEDModelConfig]:
    """Quiescent / early-type galaxy preset (z ~ 0–2).

    Quiescent (red) galaxies typically have:

    - Very old stellar populations (age > 1–10 Gyr)
    - Minimal dust attenuation
    - Broad metallicity distribution (0.2–2 Zsun common)
    - Negligible star formation

    If redshift is not specified, it is left as a free parameter.

    Parameters
    ----------
    redshift : float or None, optional
        Redshift of the galaxy. If None (default), redshift is a free parameter
        with Uniform(0.01, 3.0). If specified, redshift is fixed.

    Returns
    -------
    tuple[Parameters, SEDModelConfig]
        (params, config); fully configured Parameter spec and model settings.

    Notes
    -----
    **JIT-compatible**: no, returns configuration objects, not arrays.

    **Physics:**

    - SFH model: delayed-exponential with short timescale (old ages)
    - Dust: power-law attenuation (two-component, minimal)
    - Attenuation: Av ~ Uniform(0, 0.5) mag (low dust)
    - Metallicity: Uniform(0.0, 0.5) [Z/Zsun] (solar to super-solar)
    - Nebular emission: off (minimal ongoing star formation)

    References
    ----------
    .. [2] Charlot, S., & Fall, S. M. 2000, ApJ 539, 718
           A simple model for the absorption of radiation by dust in galaxies.

    Examples
    --------
    >>> params, config = presets.quiescent(redshift=0.3)
    >>> model = SEDModel(params, ssp_data, config, observation)
    """
    if redshift is None:
        z_prior = Uniform(0.01, 3.0)
    else:
        z_prior = Fixed(redshift)

    params = Parameters(
        mean_sfh_type="dexp",
        sfh_dexp_log_total_mass=Uniform(8.0, 11.0),
        sfh_dexp_tau_gyr=Uniform(0.1, 1.0),
        met_logzsol=Uniform(0.0, 0.5),
        dust_tau_bc=Uniform(0.0, 0.5),
        dust_tau_diff=Uniform(0.0, 0.3),
        dust_slope=Fixed(-0.7),
        redshift=z_prior,
    )

    config = SEDModelConfig(
        sfh=SFHConfig(mean_type=("dexp",)),
        dust=DustConfig(
            model="two_component",
            law_bc="power_law",
            law_diff="power_law",
            emission=None,
        ),
        nebular=NebularConfig(backend="off"),
    )

    return params, config


@register_preset(
    "high_z",
    short_doc="Young bursty galaxies at z > 3.5, tsnorm SFH + strong nebular",
    citations=["Calzetti_2000", "Charlot_2000"],
)
def high_z(redshift: float | None = None) -> tuple[Parameters, SEDModelConfig]:
    """High-redshift galaxy preset (z > 4, young starburst).

    High-redshift galaxies typically have:

    - Young stellar populations (age < 0.5–1 Gyr)
    - Strong nebular emission (young massive stars → H II regions)
    - Dust-obscured (SMC-like or Calzetti attenuation)
    - Elevated ionization parameter (logU ~ −2 to −1)

    If redshift is not specified, it defaults to a realistic range (z > 3.5).

    Parameters
    ----------
    redshift : float or None, optional
        Redshift of the galaxy. If None (default), redshift is a free parameter
        with Uniform(3.5, 10.0). If specified, redshift is fixed.

    Returns
    -------
    tuple[Parameters, SEDModelConfig]
        (params, config); fully configured Parameter spec and model settings.

    Notes
    -----
    **JIT-compatible**: no, returns configuration objects, not arrays.

    **Physics:**

    - SFH model: truncated skew-normal (bursty, short timescales)
    - Dust: Calzetti attenuation (intermediate opacity; SMC alternative available)
    - Attenuation: Av ~ Uniform(0.1, 1.5) mag (dust-affected)
    - Metallicity: Uniform(-1.0, 0.2) [Z/Zsun] (young, metal-poor → enriched)
    - Nebular emission: baked-in from SSP (no free nebular parameters)

    Examples
    --------
    >>> params, config = presets.high_z(redshift=5.0)
    >>> model = SEDModel(params, ssp_data, config, observation)

    Free-redshift fit (z > 3.5):

    >>> params, config = presets.high_z()
    >>> assert params.get("redshift") is None or params["redshift"].bounds[0] >= 3.5
    """
    if redshift is None:
        z_prior = Uniform(3.5, 10.0)
    else:
        z_prior = Fixed(redshift)

    params = Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_total_mass=Uniform(8.0, 12.0),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.1, 1.5),
        sfh_tsnorm_width_gyr=Uniform(0.05, 1.0),
        sfh_tsnorm_skew=Uniform(-1.0, 1.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-1.0, 0.2),
        dust_tau_bc=Uniform(0.1, 1.5),
        dust_tau_diff=Uniform(0.0, 0.8),
        dust_slope=Fixed(-0.7),
        redshift=z_prior,
    )

    config = SEDModelConfig(
        sfh=SFHConfig(mean_type=("tsnorm",)),
        dust=DustConfig(
            model="two_component",
            law_bc="calzetti",
            law_diff="power_law",
            emission=None,
        ),
        nebular=NebularConfig(backend="baked_in"),
    )

    return params, config


@register_preset(
    "photoz",
    short_doc="Photometric-redshift surveys: wide free z, uninformative SFH/dust",
    citations=["Calzetti_2000", "Charlot_2000"],
)
def photoz(redshift: float | None = None) -> tuple[Parameters, SEDModelConfig]:
    """Photometric-redshift galaxy preset (redshift-unconstrained survey).

    Photometric-redshift (photo-z) surveys estimate galaxy redshifts from
    broad-band photometry. This preset prioritizes redshift as the primary
    parameter of interest, with broad uninformative priors on other galaxy
    properties to avoid prior-driven biases.

    If redshift is not specified, it remains a free parameter with very wide
    bounds (Uniform(0.01, 12)) to support self-consistent photo-z inference
    across the full observable universe. This preset is almost always invoked
    with redshift=None.

    Parameters
    ----------
    redshift : float or None, optional
        Redshift of the galaxy. If None (default), redshift is a free parameter
        with Uniform(0.01, 12). If specified, redshift is fixed.

    Returns
    -------
    tuple[Parameters, SEDModelConfig]
        (params, config); fully configured Parameter spec and model settings.

    Notes
    -----
    **JIT-compatible**: no, returns configuration objects, not arrays.

    **Physics:**

    - SFH model: double power-law (flexible timescales for mixed populations)
    - Dust: Calzetti (birth cloud) + power-law (diffuse ISM, two-component)
    - Attenuation: Av ~ Uniform(0, 3) mag (wide range for photo-z degeneracies)
    - Age range: extended to 13.8 Gyr (tau_gyr up to 13.0 Gyr)
    - Metallicity: Uniform(-1.0, 0.5) [Z/Zsun] (wide to cover all populations)
    - Nebular emission: off (not constrained by broad photometry)

    Examples
    --------
    >>> params, config = presets.photoz()  # redshift free
    >>> assert "redshift" in params.free_params
    >>> assert params.get("redshift").bounds[0] < 0.1
    """
    if redshift is None:
        z_prior = Uniform(0.01, 12.0)
    else:
        z_prior = Fixed(redshift)

    params = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_total_mass=Uniform(8.0, 12.5),
        met_logzsol=Uniform(-1.0, 0.5),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=z_prior,
    )

    config = SEDModelConfig(
        sfh=SFHConfig(mean_type=("dpl",)),
        dust=DustConfig(
            model="two_component",
            law_bc="calzetti",
            law_diff="power_law",
            emission=None,
        ),
        nebular=NebularConfig(backend="off"),
    )

    return params, config


@register_preset(
    "jwst_spec",
    short_doc="JWST NIRSpec spectroscopy: known/constrained z, moderate-high dust",
    citations=["Calzetti_2000", "Charlot_2000"],
)
def jwst_spec(redshift: float | None = None) -> tuple[Parameters, SEDModelConfig]:
    """JWST NIRSpec spectroscopic fitting preset (known or constrained redshift).

    JWST NIRSpec provides high signal-to-noise spectroscopy with excellent
    wavelength resolution and sensitivity from 0.6 to 5.3 μm. This preset
    assumes redshift is either known from previous measurements or constrained
    via emission lines (H-alpha, [OIII], etc.) visible in the spectrum.

    Dust attenuation is allowed to be moderate-to-high (Av up to 3 mag) to
    accommodate dust-obscured star-forming galaxies. Metallicity priors are
    broad to allow inference from emission-line diagnostics when available.

    Parameters
    ----------
    redshift : float or None, optional
        Redshift of the galaxy. If None (default), redshift is a free parameter
        with Uniform(0.01, 15) to cover ground-based and space-based spectroscopy.
        If specified, redshift is fixed.

    Returns
    -------
    tuple[Parameters, SEDModelConfig]
        (params, config); fully configured Parameter spec and model settings.

    Notes
    -----
    **JIT-compatible**: no, returns configuration objects, not arrays.

    **Physics:**

    - SFH model: double power-law (same as starforming for consistency)
    - Dust: Calzetti (birth cloud) + power-law (diffuse ISM, two-component)
    - Attenuation: Av ~ Uniform(0, 3) mag (moderate-to-high for obscured sources)
    - Metallicity: Uniform(-0.5, 0.3) [Z/Zsun]
    - Nebular emission: off (will be handled separately by emission-line fitting)
    - Redshift: free Uniform(0.01, 15) unless specified

    Examples
    --------
    >>> params, config = presets.jwst_spec(redshift=2.5)
    >>> model = SEDModel(params, ssp_data, config, observation)

    Free-redshift fit (e.g., redshift from emission lines):

    >>> params, config = presets.jwst_spec()
    >>> assert "redshift" in params.free_params
    """
    if redshift is None:
        z_prior = Uniform(0.01, 15.0)
    else:
        z_prior = Fixed(redshift)

    params = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(8.0, 12.0),
        met_logzsol=Uniform(-0.5, 0.3),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=z_prior,
    )

    config = SEDModelConfig(
        sfh=SFHConfig(mean_type=("dpl",)),
        dust=DustConfig(
            model="two_component",
            law_bc="calzetti",
            law_diff="power_law",
            emission=None,
        ),
        nebular=NebularConfig(backend="off"),
    )

    return params, config


@register_preset(
    "agn_host",
    short_doc="AGN host galaxies (Type 1/2): high dust, torus/NLR not included",
    citations=["Calzetti_2000", "Charlot_2000"],
)
def agn_host(redshift: float | None = None) -> tuple[Parameters, SEDModelConfig]:
    """AGN host galaxy preset (Type 1/2 AGN).

    This preset models the stellar population and dust properties of galaxies
    hosting active galactic nuclei (AGN). It is optimized for decomposing the
    host galaxy contribution to the SED while the AGN itself is modeled
    separately (e.g., via torus or accretion disk templates).

    Dust attenuation is widened (Av up to 4 mag) to accommodate Type 2 (obscured)
    AGN hosts, which often exhibit significant dust columns from both the ISM
    and circumnuclear material. The SFH and metallicity priors are similar to
    starforming galaxies but allow broader flexibility.

    **Future enhancement:** A dedicated ``agn_unified`` preset will add explicit
    torus emission (K&D disc, SKIRTOR) and NLR modeling once SEDModelConfig supports
    AGN-specific knobs. This version focuses purely on the host stellar population.

    Parameters
    ----------
    redshift : float or None, optional
        Redshift of the galaxy. If None (default), redshift is a free parameter
        with Uniform(0.01, 6.0). If specified, redshift is fixed.

    Returns
    -------
    tuple[Parameters, SEDModelConfig]
        (params, config); fully configured Parameter spec and model settings.

    Notes
    -----
    **JIT-compatible**: no, returns configuration objects, not arrays.

    **Physics:**

    - SFH model: double power-law (inherits from starforming)
    - Dust: Calzetti (birth cloud) + power-law (diffuse ISM, two-component)
    - Attenuation: Av ~ Uniform(0, 4) mag (high dust for Type 2 AGN hosts)
    - Metallicity: Uniform(-0.5, 0.3) [Z/Zsun]
    - Nebular emission: off (AGN dominates ionization, modeled separately)
    - Redshift: free Uniform(0.01, 6.0) unless specified

    **Limitations:**
    This preset does NOT include AGN torus or narrow-line region (NLR) components.
    Host decomposition should be performed post-inference by subtracting the
    best-fit SED from the observed photometry before fitting the AGN SED.

    Examples
    --------
    >>> params, config = presets.agn_host(redshift=0.5)
    >>> model = SEDModel(params, ssp_data, config, observation)

    Free-redshift fit:

    >>> params, config = presets.agn_host()
    >>> assert "redshift" in params.free_params
    """
    if redshift is None:
        z_prior = Uniform(0.01, 6.0)
    else:
        z_prior = Fixed(redshift)

    params = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(8.0, 12.0),
        met_logzsol=Uniform(-0.5, 0.3),
        dust_tau_bc=Uniform(0.0, 4.0),
        dust_tau_diff=Uniform(0.0, 2.5),
        dust_slope=Fixed(-0.7),
        redshift=z_prior,
    )

    config = SEDModelConfig(
        sfh=SFHConfig(mean_type=("dpl",)),
        dust=DustConfig(
            model="two_component",
            law_bc="calzetti",
            law_diff="power_law",
            emission=None,
        ),
        nebular=NebularConfig(backend="off"),
    )

    return params, config


PRESETS: dict[str, callable] = {
    "agn_host": agn_host,
    "high_z": high_z,
    "jwst_spec": jwst_spec,
    "photoz": photoz,
    "quiescent": quiescent,
    "starforming": starforming,
}


def resolve_preset(
    name: str,
    redshift: float | None = None,
    model_config: SEDModelConfig | None = None,
) -> tuple[Parameters, SEDModelConfig]:
    """Look up a preset by name and invoke its factory.

    Parameters
    ----------
    name : str
        Preset name, one of :func:`list_presets`.
    redshift : float or None
        Redshift to pass through to the preset factory.
    model_config : SEDModelConfig or None
        Optional override for the returned SEDModelConfig. If provided, it is
        returned verbatim; the preset's own SEDModelConfig is discarded.

    Returns
    -------
    tuple[Parameters, SEDModelConfig]

    Raises
    ------
    ValueError
        If ``name`` is not a known preset.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {sorted(PRESETS.keys())}")
    params, config = PRESETS[name](redshift=redshift)
    if model_config is not None:
        config = model_config
    return params, config


def describe(name: str) -> str:
    """Return a multi-line human description of a preset.

    Parameters
    ----------
    name : str
        Preset name, one of the keys of the package preset registry
        (:func:`tengri.presets.list_presets`).

    Returns
    -------
    str
        Multi-line description of the preset's physics, prior ranges, and use cases.

    Raises
    ------
    ValueError
        If ``name`` is not a valid preset.

    Notes
    -----
    **JIT-compatible**: no; returns formatted string.

    Lookup resolution uses the same path as :func:`tengri.describe` (issue #1611),
    so a name cannot resolve in one surface and not the other.

    Examples
    --------
    >>> print(presets.describe("starforming"))
    Starforming galaxies (main sequence, z ~ 0–3):
    ...
    """
    descriptions = {
        "starforming": """\
Starforming galaxies (main sequence, z ~ 0–3):
- SFH: double power-law (α, β, τ, log(SFR_peak))
- Dust: Calzetti (birth cloud) + power-law (diffuse ISM), two-component
- Av: Uniform(0, 2) mag
- Metallicity: Uniform(−0.5, +0.3) [Z/Zsun]
- Nebular: off (baked-in SSP default)
- Redshift: free Uniform(0.01, 6.0) unless specified

Use for: optical/NIR surveys, moderate-redshift galaxies with ongoing star formation.
""",
        "quiescent": """\
Quiescent / red galaxies (z ~ 0–2):
- SFH: delayed-exponential (fast quenching, old ages)
- Dust: power-law (minimal), two-component
- Av: Uniform(0, 0.5) mag (low dust)
- Metallicity: Uniform(0, +0.5) [Z/Zsun] (solar to super-solar)
- Nebular: off (negligible star formation)
- Redshift: free Uniform(0.01, 3.0) unless specified

Use for: high-mass ellipticals, massive quenched systems, red-sequence studies.
""",
        "high_z": """\
High-redshift galaxies (z > 4, young starburst):
- SFH: truncated skew-normal (young, bursty, short timescales)
- Dust: Calzetti attenuation (dust-affected), two-component
- Av: Uniform(0.1, 1.5) mag
- Metallicity: Uniform(−1.0, +0.2) [Z/Zsun] (young, evolving)
- Nebular: baked-in SSP (strong, no free parameters)
- Redshift: free Uniform(3.5, 10.0) unless specified

Use for: z > 4 Lyman-break galaxies, JWST/HST high-z samples, young starbursts.
""",
        "photoz": """\
photoz preset: photometric-redshift surveys (z unconstrained):
- SFH: double power-law (flexible timescales)
- Dust: Calzetti (birth cloud) + power-law (diffuse ISM), two-component
- Av: Uniform(0, 3) mag (wide range for photo-z degeneracies)
- Age range: τ up to 13 Gyr (includes all populations)
- Metallicity: Uniform(−1.0, +0.5) [Z/Zsun] (wide, uninformative)
- Nebular: off (broad photometry insensitive)
- Redshift: free Uniform(0.01, 12.0) unless specified

Use for: photo-z surveys, self-consistent redshift inference, low-resolution photometry.
""",
        "jwst_spec": """\
jwst_spec preset: JWST NIRSpec spectroscopy (known/constrained redshift):
- SFH: double power-law (same as starforming)
- Dust: Calzetti (birth cloud) + power-law (diffuse ISM), two-component
- Av: Uniform(0, 3) mag (moderate-to-high for obscured sources)
- Metallicity: Uniform(−0.5, +0.3) [Z/Zsun]
- Nebular: off (handled by separate emission-line fitting)
- Redshift: free Uniform(0.01, 15.0) unless specified

Use for: JWST NIRSpec spectroscopy, known-z SED fitting, dust-obscured starbursts.
""",
        "agn_host": """\
agn_host preset: AGN host galaxies (Type 1 & 2):
- SFH: double power-law (inherits from starforming)
- Dust: Calzetti (birth cloud) + power-law (diffuse ISM), two-component
- Av: Uniform(0, 4) mag (high dust for Type 2 AGN)
- Metallicity: Uniform(−0.5, +0.3) [Z/Zsun]
- Nebular: off (AGN dominates ionization)
- Redshift: free Uniform(0.01, 6.0) unless specified

Use for: AGN host decomposition, Type 2 AGN (Compton-thick), ignoring torus/NLR.
Note: Torus/NLR modeled separately; future agn_unified preset will integrate them.
""",
    }

    if name not in descriptions:
        raise ValueError(f"Unknown preset {name!r}. Choose from: {', '.join(sorted(PRESETS))}")
    return descriptions[name]

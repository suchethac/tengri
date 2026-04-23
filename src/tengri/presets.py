"""Vetted factory functions for common galaxy types.

This module provides preset Parameter + ModelConfig tuples for typical galaxy
populations, avoiding the 10-line setup ritual. Each preset is a callable that
returns a fully-configured (Parameters, ModelConfig) pair ready for forward
modeling or inference.

Usage
-----
>>> from tengri import presets
>>> params, config = presets.starforming(redshift=0.5)
>>> # Use with SEDModel: model = SEDModel(params, ssp_data, config, observation)

Preset Gallery
--------------
- ``starforming()`` — Main sequence galaxies (z ~ 0–3): moderate dust, solar metallicity
- ``quiescent()`` — Red/dead galaxies (z ~ 0–2): minimal dust, low metallicity spread
- ``high_z()`` — Young galaxies (z > 4): strong nebular, younger ages, SMC-like dust

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
    ModelConfig,
    NebularConfig,
    SFHConfig,
)
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

__all__ = [
    "describe",
    "high_z",
    "list_presets",
    "quiescent",
    "starforming",
]


def starforming(redshift: float | None = None) -> tuple[Parameters, ModelConfig]:
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
    tuple[Parameters, ModelConfig]
        (params, config) — fully configured Parameter spec and model settings.

    Notes
    -----
    **JIT-compatible**: no — returns configuration objects, not arrays.

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
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
        met_logzsol=Uniform(-0.5, 0.3),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=z_prior,
    )

    config = ModelConfig(
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


def quiescent(redshift: float | None = None) -> tuple[Parameters, ModelConfig]:
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
    tuple[Parameters, ModelConfig]
        (params, config) — fully configured Parameter spec and model settings.

    Notes
    -----
    **JIT-compatible**: no — returns configuration objects, not arrays.

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
        sfh_dexp_log_peak_sfr=Uniform(-2.0, 0.5),
        sfh_dexp_tau_gyr=Uniform(0.1, 1.0),
        met_logzsol=Uniform(0.0, 0.5),
        dust_tau_bc=Uniform(0.0, 0.5),
        dust_tau_diff=Uniform(0.0, 0.3),
        dust_slope=Fixed(-0.7),
        redshift=z_prior,
    )

    config = ModelConfig(
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


def high_z(redshift: float | None = None) -> tuple[Parameters, ModelConfig]:
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
    tuple[Parameters, ModelConfig]
        (params, config) — fully configured Parameter spec and model settings.

    Notes
    -----
    **JIT-compatible**: no — returns configuration objects, not arrays.

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
        sfh_tsnorm_log_peak_sfr=Uniform(0.0, 2.5),
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

    config = ModelConfig(
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


PRESETS: dict[str, callable] = {
    "starforming": starforming,
    "quiescent": quiescent,
    "high_z": high_z,
}


def resolve_preset(
    name: str,
    redshift: float | None = None,
    model_config: ModelConfig | None = None,
) -> tuple[Parameters, ModelConfig]:
    """Look up a preset by name and invoke its factory.

    Parameters
    ----------
    name : str
        Preset name, one of :func:`list_presets`.
    redshift : float or None
        Redshift to pass through to the preset factory.
    model_config : ModelConfig or None
        Optional override for the returned ModelConfig. If provided, it is
        returned verbatim; the preset's own ModelConfig is discarded.

    Returns
    -------
    tuple[Parameters, ModelConfig]

    Raises
    ------
    ValueError
        If ``name`` is not a known preset.
    """
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {sorted(PRESETS.keys())}"
        )
    params, config = PRESETS[name](redshift=redshift)
    if model_config is not None:
        config = model_config
    return params, config


def list_presets() -> list[str]:
    """Return a list of available preset names.

    Returns
    -------
    list[str]
        Sorted list of preset names: ``["high_z", "quiescent", "starforming"]``.

    Notes
    -----
    **JIT-compatible**: no — returns Python list of strings.

    Examples
    --------
    >>> presets.list_presets()
    ['high_z', 'quiescent', 'starforming']
    """
    return sorted(PRESETS.keys())


def describe(name: str) -> str:
    """Return a multi-line human description of a preset.

    Parameters
    ----------
    name : str
        Preset name, one of ``list_presets()``.

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
    **JIT-compatible**: no — returns formatted string.

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
    }

    if name not in descriptions:
        raise ValueError(f"Unknown preset {name!r}. Choose from: {', '.join(list_presets())}")
    return descriptions[name]

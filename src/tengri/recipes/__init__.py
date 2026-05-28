# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Curated model recipes for common observational scenarios.

Provides ready-to-use model configurations for typical galaxy fitting workflows.
Each recipe returns a nested-dict suitable for splat ting into
:func:`~tengri.parameters.parse_groups()` or
:func:`~tengri.forward.SEDModel.build()`.

Examples
--------
Fit a star-forming galaxy at low-intermediate redshift with photometry::

    from tengri import SEDModel, recipes
    import sps_data

    model = SEDModel.build(
        ssp_data=sps_data, filters=my_filter_list, **recipes.star_forming_photometry()
    )

Fit a quiescent galaxy at z~0.05::

    model = SEDModel.build(ssp_data=sps_data, filters=my_filter_list, **recipes.quiescent_z0())
"""

from __future__ import annotations

import tengri.builders as builders
from tengri.forward.sed_model import WavePrecomp
from tengri.parameters.priors import Fixed, Uniform
from tengri.parameters.sentinels import FIXED, FREE

__all__ = [
    "agn_panchromatic",
    "dust_demo",
    "mock_recovery_minimal",
    "quiescent_z0",
    "star_forming_photometry",
    "stochastic_sfh_jwst",
    "vw07_attenuation",
]


def star_forming_photometry() -> dict:
    """Recipe for star-forming galaxies with broadband photometry (0 < z < 6).

    Suitable for optical+NIR+MIR photometry fits of star-forming galaxies
    at low-to-intermediate redshift.

    **SSP requirement:** bare-stellar (e.g., ``fsps_prsc_miles_chabrier.h5``,
    ``fsps_mist_c3k_a_chabrier.h5``). The Cue nebular backend cannot be paired
    with wNE (with-nebular-emission) SSP files; doing so raises
    ``CueWNESSPError``.

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
        Nested-dict ready for parse_groups() or SEDModel.build().

    Notes
    -----
    Designed for typical optical-through-MIR photometry. IGM absorption is
    included to handle Lyman-alpha forest opacity at high redshift. Dust
    attenuation spans realistic ranges for star-forming galaxies.

    Examples
    --------
    >>> from tengri import SEDModel, recipes
    >>> model = SEDModel.build(ssp_data=ssp, **recipes.star_forming_photometry())
    >>> assert "sfh_dpl_alpha" in model.spec.free_params
    >>> assert "redshift" in model.spec.free_params
    """
    return dict(
        sfh=builders.sfh.dpl(defaults=FREE),
        dust=builders.dust.two_component(
            defaults=FREE,
            law_bc="calzetti",
            emission=builders.dust.emission.dale2014(defaults=FIXED),
        ),
        neb=builders.neb.cue(defaults=FIXED),
        redshift=Uniform(0.01, 6.0),
        apply_igm=True,
        approx=WavePrecomp(),
    )


def quiescent_z0() -> dict:
    """Recipe for quiescent galaxies at low redshift (z ~ 0.05).

    Suitable for local quiescent galaxy samples (e.g., SDSS passive galaxies).

    **SSP requirement:** bare-stellar (Cue nebular backend; see
    :func:`star_forming_photometry` for details).

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
        Nested-dict ready for parse_groups() or SEDModel.build().

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
        sfh=builders.sfh.dexp(defaults=FREE),
        dust=builders.dust.two_component(
            defaults=FREE,
            law_bc="calzetti",
            tau_bc=Uniform(0, 0.5),
            tau_diff=Uniform(0, 0.3),
        ),
        neb=builders.neb.cue(defaults=FIXED),
        redshift=Fixed(0.05),
        approx=WavePrecomp(),
    )


def agn_panchromatic() -> dict:
    """Recipe for AGN-dominated galaxies with multi-wavelength data.

    Suitable for AGN host galaxy fitting using UV through radio data
    (panchromatic coverage).

    **SSP requirement:** bare-stellar (Cue nebular backend; see
    :func:`star_forming_photometry` for details).

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
        Nested-dict ready for parse_groups() or SEDModel.build().

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
        sfh=builders.sfh.dpl(defaults=FREE),
        dust=builders.dust.two_component(
            defaults=FREE,
            emission=builders.dust.emission.dale2014(defaults=FREE),
        ),
        neb=builders.neb.cue(defaults=FIXED),
        agn=builders.agn.composable(
            defaults=FREE,
            disc=builders.agn.disc.multicolor(defaults=FREE),
            torus=builders.agn.torus.skirtor(defaults=FREE),
            lines=builders.agn.lines.nlr(defaults=FREE),
        ),
        radio=True,
        xray=True,
        redshift=Uniform(0.01, 6.0),
        approx=WavePrecomp(),
    )


def stochastic_sfh_jwst() -> dict:
    """Recipe for JWST high-redshift galaxies with stochastic SFH.

    Suitable for JWST spectrophotometry of 0.5 < z < 12 galaxies
    where burstiness and temporal structure in star formation matter.

    **SSP requirement:** bare-stellar (Cue nebular backend; see
    :func:`star_forming_photometry` for details).

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
        Nested-dict ready for parse_groups() or SEDModel.build().

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
    # Composed SFH ("dpl" + "field" stochastic component) is expressed as a
    # type-list dict. The builder factories cover individual variants; the
    # composed form remains the canonical grammar for now.
    return dict(
        sfh={"type": ["dpl", "field"], "*": FREE},
        dust=builders.dust.two_component(
            defaults=FREE,
            emission=builders.dust.emission.dale2014(defaults=FIXED),
        ),
        neb=builders.neb.cue(defaults=FIXED),
        redshift=Uniform(0.5, 12.0),
        apply_igm=True,
        approx=WavePrecomp(),
    )


def mock_recovery_minimal() -> dict:
    """Recipe for mock data recovery and benchmarking (minimal model).

    Suitable for fast mock data fits, parameter recovery tests, and
    forward-model benchmarking.

    **SSP requirement:** any. Nebular emission is disabled, so this recipe
    works with both bare-stellar and wNE SSP files.

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
        Nested-dict ready for parse_groups() or SEDModel.build().

    Notes
    -----
    Intentionally minimal to keep forward-model runtime short and memory
    footprint small. Useful for debugging, unit tests, and quick validation
    of inference algorithms. All non-essential physics disabled.

    Examples
    --------
    >>> from tengri import recipes, Parameters
    >>> params_dict = recipes.mock_recovery_minimal()
    >>> spec = parse_groups(**params_dict)
    >>> assert 4 <= spec.n_free <= 8  # ~5 SFH + dust + met
    >>> assert "redshift" in spec.fixed_params
    """
    return dict(
        sfh=builders.sfh.tsnorm(defaults=FREE),
        dust=builders.dust.two_component(
            defaults=FIXED,
            law_bc="calzetti",
            tau_bc=Uniform(0, 1),
        ),
        neb=builders.neb.none(),
        redshift=Fixed(0.05),
        approx=WavePrecomp(),
    )


def dust_demo() -> dict:
    """Recipe for forward-only dust attenuation gallery sweeps.

    Young star-forming galaxy at z = 0.1 with every parameter ``FIXED`` so
    that :func:`~tengri.analysis.plotting.sweep_parameter` can override
    one knob at a time without touching the rest.

    **SSP requirement:** wNE (with-nebular-emission), e.g.
    ``load_ssp()`` default. Uses the BakedIn nebular path bundled with
    the SSP, so optical emission lines render in the SED plots.

    **Configuration:**
    - **SFH**: Truncated skew-normal peaked at ~0.5 Gyr (young SF)
    - **Dust**: Two-component Calzetti attenuation, τ_BC = 1, τ_diff = 0.3, δ = -0.7
    - **Dust IR emission**: Disabled (gallery wavelength range is UV-optical)
    - **Nebular**: BakedIn (whatever the wNE SSP carries)
    - **Redshift**: Fixed at z = 0.1
    - **Metallicity**: Fixed at log(Z/Z_⊙) = -0.3

    Returns
    -------
    dict
        Nested-dict ready for ``SEDModel.build(**recipes.dust_demo())``.

    Examples
    --------
    Sweep birth-cloud optical depth on the canonical demo galaxy::

        from tengri import SEDModel, load_ssp, recipes
        from tengri.analysis.plotting import sweep_parameter, SWEEP_CMAPS

        model = SEDModel.build(ssp_data=load_ssp(), **recipes.dust_demo())
        sweep_parameter(
            model,
            "dust_tau_bc",
            [0.0, 0.5, 1.0, 2.0],
            cmap=SWEEP_CMAPS["dust"],
            wave_range=(1000, 10000),
        )
    """
    # Metallicity is not a parse_groups key; its default Gaussian(-0.3, 0.2)
    # prior centres at the value we want, so we leave it FREE — sweep_parameter
    # uses the prior median (= -0.3) for every iteration anyway.
    return dict(
        sfh=builders.sfh.tsnorm(
            defaults=FIXED,
            log_total_mass=10.0,
            peak_lbt_gyr=2.0,
            width_gyr=1.5,
            skew=0.2,
            trunc=3.0,
        ),
        dust=builders.dust.two_component(
            defaults=FIXED,
            law_bc="calzetti",
            tau_bc=0.5,
            tau_diff=0.3,
            slope=-0.7,
        ),
        redshift=Fixed(0.1),
    )


def vw07_attenuation() -> dict:
    """Recipe for Wild+2007 two-component attenuation (BAGPIPES VW07 counterpart).

    Wild, Charlot & Disney (2007, MNRAS 381, 543) define a Charlot-and-Fall-like
    two-component model in which the birth-cloud (n_bc = -1.3) and diffuse-ISM
    (n_diff = -0.7) attenuation curves have **independent** power-law slopes,
    unlike the single-slope CF00 form tengri's other recipes default to. This
    recipe leaves both slopes free with priors centred on the Wild+07 values
    so a fit can recover the slope ratio that fits the data. Closes #500.

    **SSP requirement:** any.

    **Configuration:**
    - **SFH**: Delayed double power-law (free)
    - **Dust**: Two-component power law with FREE per-leaf slopes
      (``dust_slope_bc`` ~ Uniform(-2.0, -0.5),
       ``dust_slope_diff`` ~ Uniform(-1.2, -0.3))
    - **Nebular**: Cue (off by default; turn on by overriding)

    Returns
    -------
    dict
        Nested-dict ready for ``SEDModel.build(**recipes.vw07_attenuation())``.
    """
    return dict(
        sfh=builders.sfh.dpl(defaults=FREE),
        dust=builders.dust.two_component(
            defaults=FIXED,
            law_bc="power_law",
            law_diff="power_law",
            tau_bc=Uniform(0.0, 4.0),
            tau_diff=Uniform(0.0, 3.0),
            slope_bc=Uniform(-2.0, -0.5),
            slope_diff=Uniform(-1.2, -0.3),
        ),
        redshift=FREE,
    )

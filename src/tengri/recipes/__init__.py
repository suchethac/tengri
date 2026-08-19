# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Curated model recipes for common observational scenarios.

Provides ready-to-use model configurations for typical galaxy fitting workflows.
Each recipe returns a nested-dict suitable for splat ting into
:func:`~tengri.parameters.parse_groups()` or
:func:`~tengri.SEDModel.build()`.

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
from tengri._completion import curated_dir
from tengri.forward.sed_model import WavePrecomp
from tengri.parameters.priors import Fixed, Uniform
from tengri.parameters.sentinels import FIXED, FREE, WILDCARD_ALIAS

__all__ = [
    "agn_panchromatic",
    "composable_agn",
    "dust_demo",
    "high_z",
    "mock_recovery_minimal",
    "photoz",
    "quiescent_z0",
    "star_forming_photometry",
    "stochastic_sfh_jwst",
    "unified_agn",
]


def star_forming_photometry() -> dict:
    """Recipe for star-forming galaxies with broadband photometry (0 < z < 6).

    Suitable for optical+NIR+MIR photometry fits of star-forming galaxies
    at low-to-intermediate redshift.

    **SSP requirement:** bare-stellar (e.g., ``fsps_prsc_miles_chabrier.h5``,
    ``fsps_mist_c3k_a_chabrier.h5``). The Cue nebular backend cannot be paired
    with wNE (with-nebular-emission) SSP files; doing so raises
    ``CueWNESSPError``.

    **To download the bare-stellar SSP:**

    .. code-block:: python

        import tengri

        tengri.download_ssp("fsps_prsc_miles_chabrier")

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
        met={"logzsol": FREE},
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

    **To download the bare-stellar SSP:**

    .. code-block:: python

        import tengri

        tengri.download_ssp("fsps_prsc_miles_chabrier")

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
        # ``defaults=FIXED``: only tau_bc / tau_diff are fitted here. The
        # remaining attenuation params (slope, Rv, delta, bump_strength,
        # f_obscuration) carry Fixed registry defaults, so the FREE this
        # previously requested never freed any of them — the recipe now says
        # what it has always actually done.
        dust=builders.dust.two_component(
            defaults=FIXED,
            law_bc="calzetti",
            tau_bc=Uniform(0, 0.5),
            tau_diff=Uniform(0, 0.3),
        ),
        met={"logzsol": FREE},
        neb=builders.neb.cue(defaults=FIXED),
        redshift=Fixed(0.05),
        approx=WavePrecomp(),
    )


def high_z() -> dict:
    """Recipe for high-redshift galaxies (z > 3.5, young starburst).

    Suitable for rest-UV/optical photometry of young, bursty systems at
    z > 3.5: strong nebular emission, metal-poor stellar populations, and
    SMC/Calzetti-like dust.

    **SSP requirement:** with-nebular-emission (wNE) SSP grids — the ``ssp``
    nebular backend reads line and continuum emission baked into the SSP
    file itself, so a bare-stellar grid would silently drop the nebular
    contribution.

    **Configuration:**

    - **SFH**: Truncated skew-normal (tsnorm) — bursty, short timescales
    - **Dust**: Two-component, Calzetti birth cloud + power-law diffuse
      (slope fixed at -0.7), no IR emission block
    - **Nebular**: baked into the SSP grid (``ssp`` backend)
    - **Redshift**: Free with bounds (3.5, 10.0)
    - **IGM**: Applied (mandatory across the Lyman forest)
    - **Metallicity**: Free, Uniform(-1.0, 0.2) [log10(Z/Zsun)]

    Returns
    -------
    dict
        Nested-dict ready for parse_groups() or SEDModel.build().

    Notes
    -----
    Rescued from the pre-grammar ``presets.high_z()`` factory (2026-07);
    prior ranges are unchanged. Deliberately **no** ``approx=WavePrecomp()``:
    the LUT's first-order dust projection biases rest-UV bands (#617, #731),
    which is exactly the regime this recipe samples — the exact wave-grid
    path is the correct default here.

    Examples
    --------
    >>> from tengri import SEDModel, recipes
    >>> model = SEDModel.build(ssp_data=ssp_wne, **recipes.high_z())
    >>> assert "sfh_tsnorm_peak_lbt_gyr" in model.spec.free_params
    """
    return dict(
        sfh={
            "type": "tsnorm",
            "all_params": FIXED,
            "log_total_mass": Uniform(8.0, 12.0),
            "peak_lbt_gyr": Uniform(0.1, 1.5),
            "width_gyr": Uniform(0.05, 1.0),
            "skew": Uniform(-1.0, 1.0),
            "trunc": Uniform(1.0, 10.0),
            "met_logzsol": Uniform(-1.0, 0.2),
        },
        dust={
            "type": "two_component",
            "all_params": FIXED,
            "law_bc": "calzetti",
            "law_diff": "power_law",
            "tau_bc": Uniform(0.1, 1.5),
            "tau_diff": Uniform(0.0, 0.8),
            "slope": Fixed(-0.7),
        },
        neb={"type": "ssp"},
        redshift=Uniform(3.5, 10.0),
        apply_igm=True,
    )


def photoz() -> dict:
    """Recipe for photometric-redshift fits (redshift-unconstrained surveys).

    Prioritizes redshift as the parameter of interest, with broad
    uninformative priors on SFH, dust, and metallicity to avoid
    prior-driven photo-z biases.

    **SSP requirement:** any (nebular emission is off).

    **Configuration:**

    - **SFH**: Double power-law with extended timescales (tau up to 13 Gyr)
    - **Dust**: Two-component, Calzetti birth cloud + power-law diffuse
      (slope fixed at -0.7), wide optical-depth ranges for photo-z
      degeneracies
    - **Nebular**: Off (unconstrained by broadband photometry)
    - **Redshift**: Free with bounds (0.01, 12.0)
    - **IGM**: Applied (the wide z range crosses the Lyman forest)
    - **Metallicity**: Free, Uniform(-1.0, 0.5) [log10(Z/Zsun)]

    Returns
    -------
    dict
        Nested-dict ready for parse_groups() or SEDModel.build().

    Notes
    -----
    Rescued from the pre-grammar ``presets.photoz()`` factory (2026-07);
    prior ranges are unchanged. Deliberately **no** ``approx=WavePrecomp()``:
    the default ztable does not cover z up to 12 and the LUT's dust
    projection biases blue bands at high redshift (#617, #731).

    Examples
    --------
    >>> from tengri import SEDModel, recipes
    >>> model = SEDModel.build(ssp_data=ssp, **recipes.photoz())
    >>> assert "redshift" in model.spec.free_params
    """
    return dict(
        sfh={
            "type": "dpl",
            "all_params": FIXED,
            "alpha": Uniform(0.5, 3.0),
            "beta": Uniform(0.3, 2.0),
            "tau_gyr": Uniform(0.5, 13.0),
            "log_total_mass": Uniform(8.0, 12.5),
            "met_logzsol": Uniform(-1.0, 0.5),
        },
        dust={
            "type": "two_component",
            "all_params": FIXED,
            "law_bc": "calzetti",
            "law_diff": "power_law",
            "tau_bc": Uniform(0.0, 3.0),
            "tau_diff": Uniform(0.0, 2.0),
            "slope": Fixed(-0.7),
        },
        neb={"type": "none"},
        redshift=Uniform(0.01, 12.0),
        apply_igm=True,
    )


def agn_panchromatic() -> dict:
    """Recipe for AGN-dominated galaxies with multi-wavelength data.

    Suitable for AGN host galaxy fitting using UV through radio data
    (panchromatic coverage).

    **SSP requirement:** bare-stellar (Cue nebular backend; see
    :func:`star_forming_photometry` for details).

    **To download the bare-stellar SSP:**

    .. code-block:: python

        import tengri

        tengri.download_ssp("fsps_prsc_miles_chabrier")

    **Configuration:**

    - **SFH**: DPL (free)
    - **Dust**: Two-component Calzetti attenuation (free)
    - **Dust IR emission**: Dale2014 (free)
    - **Nebular**: Cue (fixed)
    - **AGN**:
        - **Disc**: Multicolor disc (free)
        - **Torus**: SKIRTOR clumpy torus model (free)
        - **NLR**: Analytic narrow-line region (free)
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
            # ``defaults=FIXED``: the Dale+2014 knobs are a template-family
            # choice, not something a wildcard should open by default.
            # ``dale2014_cigale``: this recipe enables the radio component, and
            # plain dale2014 embeds its own SF radio continuum — the pair
            # double-counts the synchrotron and is refused at build (#1970).
            emission=builders.dust.emission.dale2014_cigale(defaults=FIXED),
        ),
        met={"logzsol": FREE},
        neb=builders.neb.cue(defaults=FIXED),
        agn=builders.agn.composable(
            defaults=FREE,
            disc=builders.agn.disc.multicolor(defaults=FREE),
            torus=builders.agn.torus.skirtor(defaults=FREE),
            nlr=builders.agn.nlr.analytic(defaults=FREE),
        ),
        radio={"type": "condon92"},
        xray={"type": "simple"},
        redshift=Uniform(0.01, 6.0),
        approx=WavePrecomp(),
    )


def composable_agn() -> dict:
    """Fully composable AGN — all slots switchable on committed data.

    Provides a fully wired AGN recipe where every slot (disc, NLR, BLR, FeII,
    torus, attenuation) uses committed data only. All six blocks are present
    and their parameters are free, making this recipe ideal for exploratory
    fitting and model swapping without grid dependencies.

    **SSP requirement:** bare-stellar (Cue nebular backend; see
    :func:`star_forming_photometry` for details).

    **To download the bare-stellar SSP:**

    .. code-block:: python

        import tengri

        tengri.download_ssp("fsps_prsc_miles_chabrier")

    **Data requirement:** All components use committed templates bundled with
    tengri. No external Synthesizer Cloudy grids or other downloads needed.

    **Configuration:**

    - **Disc**: Multicolor accretion disc (free)
    - **NLR**: Analytic narrow-line region (free)
    - **BLR**: Analytic broad-line region (free)
    - **FeII**: Boroson & Green FeII pseudo-continuum (free)
    - **Torus**: SKIRTOR clumpy torus model (free)
    - **Attenuation**: Polar dust Type-1/2 screen (free)
    - **SFH**: DPL (free)
    - **Dust**: Two-component Calzetti attenuation (free)
    - **Dust IR emission**: Dale2014 (free)
    - **Nebular**: Cue (fixed)
    - **Radio**: Enabled (free)
    - **X-ray**: Enabled (free)
    - **Redshift**: Free
    - **AGN normalization**: CIGALE joint (disc + torus coupled energy balance)

    Returns
    -------
    dict
        Nested-dict ready for parse_groups() or SEDModel.build().

    Notes
    -----
    The CIGALE joint normalization couples the disc and torus luminosities via
    the energy reprocessing fraction, ensuring energy conservation across AGN
    components. All slots are independently switchable by changing the ``type``
    key (e.g., ``disc={'type': 'powerlaw'}`` or ``nlr={'type': 'none'}``).

    Examples
    --------
    >>> from tengri import recipes
    >>> params = recipes.composable_agn()
    >>> assert "agn" in params
    >>> assert "disc" in params["agn"]
    >>> assert "feii" in params["agn"]
    """
    return dict(
        sfh=builders.sfh.dpl(defaults=FREE),
        dust=builders.dust.two_component(
            defaults=FREE,
            # ``defaults=FIXED``: the Dale+2014 knobs are a template-family
            # choice, not something a wildcard should open by default.
            # ``dale2014_cigale``: this recipe enables the radio component, and
            # plain dale2014 embeds its own SF radio continuum — the pair
            # double-counts the synchrotron and is refused at build (#1970).
            emission=builders.dust.emission.dale2014_cigale(defaults=FIXED),
        ),
        met={"logzsol": FREE},
        neb=builders.neb.cue(defaults=FIXED),
        agn={
            "type": "composable",
            "disc": {"type": "multicolor"},
            "nlr": {"type": "analytic"},
            "blr": {"type": "analytic"},
            "feii": {"type": "boroson_green"},
            "torus": {"type": "skirtor"},
            "atten": {"type": "polar_dust"},
            "norm": "cigale_joint",
            # agn_ir_frac constraint is [0, 1) — keep the upper bound strictly < 1.
            "ir_frac": Uniform(0.01, 0.99),
            WILDCARD_ALIAS: FREE,
        },
        radio={"type": "condon92"},
        xray={"type": "simple"},
        redshift=Uniform(0.01, 6.0),
        approx=WavePrecomp(),
    )


def stochastic_sfh_jwst() -> dict:
    """Recipe for JWST high-redshift galaxies with stochastic SFH.

    Suitable for JWST spectrophotometry of 0.5 < z < 12 galaxies
    where burstiness and temporal structure in star formation matter.

    **SSP requirement:** bare-stellar (Cue nebular backend; see
    :func:`star_forming_photometry` for details).

    **To download the bare-stellar SSP:**

    .. code-block:: python

        import tengri

        tengri.download_ssp("fsps_prsc_miles_chabrier")

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
        sfh={"type": ["dpl", "field"], WILDCARD_ALIAS: FREE},
        dust=builders.dust.two_component(
            defaults=FREE,
            emission=builders.dust.emission.dale2014(defaults=FIXED),
        ),
        met={"logzsol": FREE},
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
        met={"all_params": FIXED, "logzsol": FREE},
        neb=builders.neb.none(),
        redshift=Fixed(0.05),
        approx=WavePrecomp(),
    )


def dust_demo() -> dict:
    """Recipe for forward-only dust attenuation gallery sweeps.

    Young star-forming galaxy at z = 0.1 with every parameter ``FIXED`` so
    that :func:`~tengri.analysis.plotting.sweep_parameter` can override
    one knob at a time without touching the rest.

    **SSP requirement:** wNE (with-nebular-emission), i.e.
    ``load_ssp("prsc_miles_chabrier_wNE")``. Uses the BakedIn nebular path
    bundled with the SSP, so optical emission lines render in the SED plots.
    (The bare-stellar ``load_ssp()`` default carries no baked nebular lines.)

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

        model = SEDModel.build(ssp_data=load_ssp("prsc_miles_chabrier_wNE"), **recipes.dust_demo())
        sweep_parameter(
            model,
            "dust_tau_bc",
            [0.0, 0.5, 1.0, 2.0],
            cmap=SWEEP_CMAPS["dust"],
            wave_range=(1000, 10000),
        )
    """
    # Metallicity is not a parse_groups key; its default Gaussian(-0.3, 0.2)
    # prior centers at the value we want, so we leave it FREE — sweep_parameter
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


def unified_agn() -> dict:
    """Synthesizer UnifiedAGN reproduction (disc + NLR/BLR grids).

    Reproduces the Synthesizer UnifiedAGN model: a Kubota & Done disc,
    SKIRTOR simple torus, and narrow + broad line region photoionization grids
    from the Cloudy+AGNfitter framework. The runner applies the Type-1/2
    visibility mask to the anisotropic central engine (disc + BLR), driven by
    ``agn_cos_inc`` / ``agn_theta_torus``, while the isotropic NLR stays
    unmasked.

    **SSP requirement:** bare-stellar (Cue nebular backend; see
    :func:`star_forming_photometry` for details). Synthesizer grids cannot be
    paired with wNE SSP files.

    **Grid requirement:** The Synthesizer AGN Cloudy grids must be present at
    ``data/synthesizer_grids/``. Fetch via ``synthesizer-download
    --agn-test-grids``. Tests will be skipped if grids are absent.

    **Configuration:**

    - **Disc**: Kubota & Done accretion disc (free)
    - **NLR**: Synthesizer Cloudy photoionization grid (free)
    - **BLR**: Synthesizer Cloudy photoionization grid (free)
    - **Torus**: Simple graybody (fixed)
    - **SFH**: Delayed exponential (fixed)
    - **Dust**: Two-component Calzetti (both optical depths fixed to 0)
    - **Redshift**: Fixed at z=0.0

    Returns
    -------
    dict
        Nested-dict ready for parse_groups() or SEDModel.build().

    Examples
    --------
    >>> from tengri import recipes
    >>> params = recipes.unified_agn()
    >>> assert params["agn"]["nlr"]["type"] == "synthesizer_spectra"
    >>> assert params["agn"]["blr"]["type"] == "synthesizer_spectra"
    """
    return dict(
        sfh={"type": "delayed", WILDCARD_ALIAS: FIXED},
        dust={"type": "two_component", WILDCARD_ALIAS: FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
        agn={
            "type": "composable",
            "disc": {"type": "kubota_done"},
            "torus": {"type": "simple"},
            "nlr": {"type": "synthesizer_spectra"},
            "blr": {"type": "synthesizer_spectra"},
            # Parametric luminosity mode: the AGN strength is set by the free
            # agn_log_lbol, so the two alternative scaling knobs are held fixed
            # (freeing them would add no-op nuisance dimensions in this mode).
            "lum_ratio": Fixed(1.0),
            "ir_frac": Fixed(0.0),
            WILDCARD_ALIAS: FREE,
        },
        redshift=Fixed(0.0),
    )


__dir__ = curated_dir(__all__)

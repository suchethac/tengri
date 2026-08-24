# SPDX-License-Identifier: BSD-3-Clause
"""Synthesizer-parity model preset.

Reproduces synthesizer-project/synthesizer's default UnifiedAGN +
two-component Calzetti dust + DL14 dust emission + Inoue14 IGM +
Cue nebular configuration, expressed as a tengri SEDModel.

See ``docs/dev/synthesizer_parity.md`` for the full mapping table.
"""

from __future__ import annotations

from tengri.config.settings import (
    AGNConfig,
    DustConfig,
    MultiwavelengthConfig,
    NebularConfig,
    SEDModelConfig,
    SFHConfig,
)
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform
from tengri.presets._registry import register_preset


@register_preset(
    "synthesizer_default",
    short_doc="Synthesizer's default UnifiedAGN + Calzetti + DL14 + Inoue14 + Cue nebular.",
    citations=[
        "Bruzual_2003",  # SPS backend BC03
        "Carnall_2017",  # DPL parametric SFH
        "Calzetti_2000",  # Dust attenuation BC + ISM
        "Charlot_2000",  # Two-component dust geometry
        "Draine_2014",  # DL14 dust emission
        "Inoue_2014",  # IGM transmission
        "Li_2024a",  # Cue nebular emulator
        "Shakura_1973",  # AGN disc baseline
        "Stalevski_2016",  # SKIRTOR torus
        "Groves_2004",  # NLR templates
        "Lovell_2025",  # Synthesizer (cite together with Roper_2026)
        "Roper_2026",  # Synthesizer JOSS paper: both required by citation policy
    ],
    status="experimental",
)
def synthesizer_default(
    *,
    redshift: float = 1.0,
) -> tuple[SEDModelConfig, Parameters]:
    """Build synthesizer-parity config + Parameters template.

    Returns the configuration and default parameters for synthesizer-parity
    fitting. The user supplies SSP data and constructs the SEDModel::

        config, params = tengri.presets.synthesizer_default(redshift=2.5)
        ssp = tengri.load_ssp_data("data/ssp.h5")
        model = tengri.SEDModel(params, ssp_data=ssp)

    Reproduces the default assumptions of synthesizer-project/synthesizer:

    - SPS: BC03 via DSPS
    - SFH: parametric DPL (delayed-tau power law)
    - Dust geometry: Charlot & Fall 2000 two-component (BC + ISM)
    - Dust attenuation: Calzetti 2000 (starburst law)
    - Dust emission: Draine & Li 2014
    - IGM transmission: Inoue et al. 2014 (Lyman series + metal absorption)
    - Nebular: Cue neural emulator (validated equivalent to Byler grids)
    - AGN: Unified model with multicolor disc + SKIRTOR torus + Groves analytic NLR

    Parameters
    ----------
    redshift : float, default 1.0
        Source redshift (fixed, not fitted).

    Returns
    -------
    config : SEDModelConfig
        Frozen configuration encoding which physics modules are active.
    params : Parameters
        Default Parameters with priors matching synthesizer's fitting ranges
        (per docs/dev/synthesizer_parity.md Parameter Prefix Mapping).

    Notes
    -----
    This preset does NOT include:

    - Radio (disabled)
    - X-ray (disabled)
    - Shock (disabled)

    The AGN model is wired to the unified Shakura-Sunyaev disc model
    with SKIRTOR torus, a simplification of synthesizer's more flexible AGN
    handling. Validation against synthesizer runs is in ``tests/regression/synthesizer_parity/``.

    **For detailed component choices and their justifications, read**
    ``docs/dev/synthesizer_parity.md``. It lists every physics layer,
    the synthesizer default, the tengri equivalent, and pitfall IDs.

    Examples
    --------
    >>> import tengri
    >>> config, params = tengri.presets.synthesizer_default(redshift=2.0)
    >>> print(config)
    >>> print(params)
    """
    # ────────────────────────────────────────────────────────────────
    # Configuration: structural choices (not fittable)
    # ────────────────────────────────────────────────────────────────

    sfh_config = SFHConfig(
        mean_type=("dpl",),  # Delayed-tau power law
        n_grid=64,  # Standard; not used for DPL
        evolving_metallicity=False,  # Uniform metallicity
        alpha_fe_evolving=False,
        chem_evol=False,
        met_interp="smooth",  # Triweight interp (synthesizer equiv)
    )

    dust_config = DustConfig(
        model="two_component",  # Charlot & Fall 2000
        law_bc="calzetti",  # Birth cloud
        law_diff="calzetti",  # Same law for diffuse ISM
        emission="draine_li2014",  # DL14 dust emission
    )

    nebular_config = NebularConfig(
        backend="cue",  # Cue neural emulator (validated ≈ Byler grids)
        grid_path=None,  # Cue uses bundled weights
        weights_path=None,  # Use defaults
        ionization="ssp",  # Ionizing continuum from SSP
        eline_mode="off",  # No free emission line params
        eline_broad=False,
    )

    multiwavelength_config = MultiwavelengthConfig(
        radio=False,
        xray=False,
        shock=False,
        apply_igm=True,
        igm_model="inoue",  # Inoue et al. 2014
    )

    agn_config = AGNConfig(
        disc="multicolor",  # Multi-color blackbody disc
        torus="skirtor",  # SKIRTOR clumpy torus
        nlr="analytic",  # Gaussian line profiles
        blr=True,  # Include broad line region for Type 1
        polar_dust=False,  # No SMC polar reddening
        fe2=False,
    )

    config = SEDModelConfig(
        sfh=sfh_config,
        dust=dust_config,
        nebular=nebular_config,
        multiwavelength=multiwavelength_config,
        agn_model="unified_nlr_blr",  # The unified AGN factory
        agn_config=agn_config,
    )

    # ────────────────────────────────────────────────────────────────
    # Parameters: fittable priors + fixed values
    # ────────────────────────────────────────────────────────────────

    # Parameter defaults from _param_defs.py (registry defaults).
    # Synthesizer's typical fitting ranges per the parity table.
    # Note: nebular and AGN parameters are only available when those
    # components are enabled; they are auto-discovered by Parameters.__init__.
    param_kwargs = {
        # Stellar mass history
        "sfh_dpl_alpha": Uniform(0.01, 10.0),  # e-folding timescale inverse [Gyr^-1]
        "sfh_dpl_beta": Uniform(0.01, 3.0),  # Early power-law index
        "sfh_dpl_tau_gyr": Uniform(0.1, 10.0),  # e-folding timescale [Gyr]
        # Metallicity
        "met_logzsol": Uniform(-2.0, 0.2),  # log10(Z/Zsun)
        # Dust attenuation
        "dust_tau_bc": Uniform(0.0, 4.0),  # Birth cloud optical depth
        "dust_tau_diff": Uniform(0.0, 3.0),  # Diffuse ISM optical depth
        "dust_slope": Fixed(-0.7),  # Power-law index (used if law="power_law")
        # Redshift (passed as kwarg, not a fitted parameter)
        "redshift": Fixed(redshift),
        # Noise (photometric)
        "noise_frac_cal": Fixed(0.0),  # Calibration floor (fraction)
    }

    params = Parameters(**param_kwargs)

    return config, params

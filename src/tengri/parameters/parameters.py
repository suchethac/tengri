# SPDX-License-Identifier: BSD-3-Clause
"""Parameter specification for tengri models.

Parameters defines all model parameters: their names, distributions (or fixed
values), and physical bounds. A single Parameters is used for both mock
generation (sampling from priors) and inference (defining the prior).

The parameter set is dynamically determined by ``mean_sfh_type``, which
selects SFH model(s) from the registry. Non-SFH parameters (metallicity,
dust, redshift) are always present.

Usage
-----
Default (dense_basis + GP field)::

    spec = Parameters(
        sfh_db_log_total_mass=Uniform(8, 12),
        sfh_db_log_sfr_inst=Uniform(-2, 3),
        sfh_db_tx_frac_0=Uniform(0.05, 0.95),
        sfh_db_tx_frac_1=Uniform(0.05, 0.95),
        sfh_db_tx_frac_2=Uniform(0.05, 0.95),
        sfh_field_psd_sigma=Uniform(0.01, 1.0),
        sfh_field_psd_tau_myr=Uniform(10, 500),
        met_logzsol=Gaussian(-0.3, 0.2),
        dust_tau_bc=Uniform(0, 4),
        redshift=0.1,
    )

Shorthand tsnorm equivalent::

    spec = Parameters(
        mean_sfh_type = "tsnorm",
        sfh_tsnorm_log_total_mass = Uniform(8, 12),
        sfh_tsnorm_peak_lbt_gyr = Uniform(1, 12),
        sfh_tsnorm_width_gyr = Uniform(0.5, 5),
        sfh_tsnorm_skew = Uniform(-1, 1),
        sfh_tsnorm_trunc = Uniform(1, 10),
        ...
    )

Shorthand DPL equivalent::

    spec = Parameters(
        mean_sfh_type = "dpl",
        sfh_dpl_alpha    = Uniform(0.5, 3.0),
        sfh_dpl_beta     = Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr  = Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass = Uniform(8, 12),
        ...
    )
"""

from __future__ import annotations

import copy
import zlib

import jax
import jax.numpy as jnp

from tengri._display import _display
from tengri.parameters._aliases import (
    resolve_param_name,
    resolve_sfh_type,
)
from tengri.parameters._builders import (
    SETTINGS_KEYS,
    _build_param_registry,
    _resolve_lazy_bucket,
)
from tengri.parameters.priors import (
    Distribution,
    Fixed,
    resolve_shorthand,
)
from tengri.parameters.sentinels import WILDCARD_ALIAS

__all__ = ["SETTINGS_KEYS", "Parameters"]


def _stable_param_seed(name: str) -> int:
    """Deterministic 32-bit seed for a parameter name's sampling substream.

    Uses ``zlib.crc32`` rather than the built-in ``hash``: ``hash`` is salted
    per process (``PYTHONHASHSEED``) and would make ``Parameters.sample``
    irreproducible across runs. crc32 of the UTF-8 bytes is stable everywhere
    and fits the int32 domain ``jax.random.fold_in`` expects.
    """
    return int(zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF)


# ── Parameters class ───────────────────────────────────────────────────


class Parameters:
    """Parameter specification defining all model parameters and their priors.

    A Parameters object defines the complete parameter set for a SEDModel,
    including both the mean SFH model(s) and all optional components (dust,
    nebular emission, AGN, etc.). Parameters can be sampled (for mock data
    generation) or used as priors for inference.

    Parameters are specified as keyword arguments, each of which can be:

    - **Scalar** (int/float) → ``Fixed(value)`` — parameter is constant
    - **Tuple** (lo, hi) → ``Uniform(lo, hi)`` — shorthand for uniform prior
    - **Distribution object**: ``Uniform``, ``Gaussian``, ``LogUniform``,
      ``LogNormal``, ``StudentT``, or ``Fixed``

    A Parameters object also stores model configuration settings
    (mean_sfh_type, dust_law, nebular mode, etc.) that control which
    components are enabled. These are not fittable parameters.

    Parameters
    ----------
    **kwargs : keyword arguments
        Model parameters (distribution objects or shorthands) and settings
        (see "Settings" section below).

    Attributes
    ----------
    mean_sfh_type : list[str]
        Active SFH model(s). Read-only.
    n_grid : int
        Grid size for stochastic SFH. Read-only.
    stochastic : bool
        True if mean_sfh_type includes 'field'. Read-only.
    all_params : list[str]
        All valid parameter names (free + fixed).
    free_params : list[str]
        Non-fixed parameter names (to be inferred).
    fixed_params : list[str]
        Fixed parameter names (constants).
    n_free : int
        Number of free parameters.
    nebular_mode : str
        Nebular emission backend: 'off', 'ssp', 'cue', 'cloudy', or 'cb19'.
    dust_model : str
        Dust model: 'two_component' or 'single_component'.
    dust_emission : str or None
        Dust emission template: 'modified_blackbody', 'casey2012', 'dale2014', etc.
    agn_model : str or None
        AGN SED model, e.g. 'kubota_done', 'skirtor', 'qsogen'.
    apply_igm : bool
        If True, apply IGM absorption (Inoue+2014).
    radio : bool
        If True, include radio synchrotron + AGN jet emission.
    xray : bool
        If True, include X-ray (XRB + AGN) emission.

    Raises
    ------
    ValueError
        If parameter names are invalid for the selected mean_sfh_type.
    ValueError
        If nebular/dust/AGN settings are mutually incompatible.

    Notes
    -----
    **Not JAX-traced**: Parameters is the central user-facing object for
    configuring model parameters and their priors. Parameters objects cannot be
    created or modified inside a JAX gradient tape (jax.grad, jax.vmap, jax.jit).
    Create all Parameters objects at the Python level before tracing. Once created,
    a Parameters object is immutable — use the `with_params()` method to create
    modified copies.

    **Parameter auto-detection**: If mean_sfh_type is not explicit, it is
    inferred from the parameter name prefixes (e.g., 'sfh_dpl_alpha' implies
    'dpl' is active). The inferred type is normalized to a sorted list.

    **Mirror parameters**: A parameter can be tied to another by passing the
    target name as a string instead of a distribution. Example:
    ``neb_logZ_gas="met_logzsol"`` ties gas metallicity to stellar.

    Settings (model configuration, not fittable parameters)
    ========================================================
    mean_sfh_type : str or list[str]
        SFH model(s). Composable: ``["dpl", "field"]``.
        Options: ``dpl``, ``tsnorm``, ``snorm``, ``norm``, ``lnorm``, ``const``,
        ``exp``, ``dexp``, ``burst``, ``field``.
        Default: ``["dpl", "field"]``.
    n_grid : int
        Grid size for stochastic SFH (latent dimensions).
        Default: 64.
    stochastic : bool
        **DEPRECATED**. Use mean_sfh_type with/without 'field' instead.

    **Dust Attenuation Settings**

    dust_law_bc : str
        Attenuation curve for birth cloud.  Default: ``"power_law"``.
        Options: ``power_law``, ``calzetti``, ``kriek_conroy``, ``smc``,
        ``cardelli``, ``salim``, ``li08``.
    dust_law_diff : str
        Attenuation curve for diffuse ISM.  Default: same as ``dust_law_bc``.
        Can be different for per-component control.
    dust_law_neb : str or None
        Attenuation curve for the nebular birth cloud.  Default ``None`` —
        inherit ``dust_law_bc`` so the nebular continuum is reddened exactly
        like the youngest stars (bagpipes/FSPS/CIGALE).  Set it to give
        HII-region emission its own birth-cloud curve while still sharing the
        diffuse ISM screen (``dust_law_diff``) with the stars.

    **Dust Emission Settings**

    dust_emission : str or None
        IR emission model.  Default: ``None`` (disabled).
        Options: ``"modified_blackbody"``, ``"casey2012"``, ``"dale2014"``,
        ``"draine_li2007"``, ``"draine_li2014"``, ``"dl07_tabulated"``,
        ``"astrodust"``, ``"bosa"``, ``"themis"``, ``"draine2021_pah"``.
    dl07_grid_path : str
        Path to DL07 HDF5 template grid (for ``"dl07_tabulated"``).

    **Nebular Emission Settings**

    nebular_ssp : bool
        Use SSP files with pre-included nebular emission (wNE files).
        No free nebular parameters.  Default: ``False``.
    nebular : bool
        Enable CLOUDY grid nebular emission.  Requires ``cloudy_grid_path``.
        Default: ``False``.
    nebular_cue : bool
        Enable Cue neural emulator.  Default weights loaded automatically.
        Default: ``False``.
    cloudy_grid_path : str
        Path to CLOUDY HDF5 grid.  Required when ``nebular=True``.
    cue_weights_path : str
        Override default Cue weights path.
    neb_ionization : str
        Ionization source for Cue: ``"ssp"`` (default), ``"agn"`` (future),
        ``"ssp+agn"`` (future).

    **AGN Settings**

    agn_model : str or None
        AGN SED model.  Default: ``None`` (disabled).
        Options: ``"simple"`` (3 params), ``"standard"`` (SS73 disc + 2T torus),
        ``"kubota_done"`` (physical disc), ``"unified_nlr_blr"`` (NLR/BLR with
        geometric masking), ``"qsogen"`` (empirical quasar, Temple+2021),
        ``"skirtor"`` (clumpy torus RT templates, Stalevski+2016).

    **Multi-wavelength Settings**

    radio : bool
        Enable radio synchrotron + AGN jet emission.  Default: ``False``.
    xray : bool
        Enable X-ray (XRB + AGN corona) emission.  Default: ``False``.

    **IGM Settings**

    apply_igm : bool
        Apply Inoue+2014 IGM absorption.  Default: ``True``.

    **Metallicity Settings**

    evolving_metallicity : bool
        Replace ``met_logzsol`` with ``met_logzsol_0`` (old stars) and
        ``met_logzsol_final`` (young stars) for a linear-in-log Z(t) ramp.
        Default: ``False``.
    met_interp : str
        Metallicity interpolation method.  Default: ``"smooth"``.

        - ``"smooth"``: Triweight kernel (same as DSPS, Hearin+2023).
          8.5x smoother gradients at <1% speed overhead. Recommended.
        - ``"linear"``: 2-point linear in log(Z) (same as FSPS/Prospector).

    lgmet_scatter : float
        Triweight kernel bandwidth in dex for ``met_interp="smooth"``.
        Default: 0.1 (DSPS default). Physically: intrinsic Z scatter.

    Fittable Parameters (always available)
    ---------------------------------------
    ========================== ================= =======================================
    Parameter                  Default           Description
    ========================== ================= =======================================
    met_logzsol                Uniform(-2, 0.2)  Stellar metallicity log10(Z/Zsun)
    met_alpha_fe               Fixed(0.0)        [alpha/Fe] enhancement (dex)
    dust_tau_bc                Uniform(0, 4)     Birth cloud V-band optical depth
    dust_tau_diff              Uniform(0, 3)     Diffuse ISM V-band optical depth
    dust_slope                 Fixed(-0.7)       Power-law index (for power_law curve)
    dust_f_obscuration         Fixed(0.0)        Unobscured fraction (Lower+2022)
    dust_bump_strength         Fixed(0.0)        UV 2175A bump (Kriek&Conroy/Salim)
    dust_delta                 Fixed(0.0)        Attenuation slope modification
    dust_Rv                    Fixed(3.1)        R_V (Cardelli curve)
    redshift                   Fixed(0.1)        Source redshift
    noise_frac_cal             Fixed(0.0)        Fractional calibration noise floor
    noise_dof                  Fixed(0.0)        Student-t degrees of freedom
    ========================== ================= =======================================

    Conditional Parameters (added when modules enabled)
    ----------------------------------------------------
    **Nebular** (``nebular=True``):

    ========================== ================= =======================================
    neb_logU                   Fixed(-3.0)       Ionization parameter log10(U)
    neb_logZ_gas               Fixed(-0.3)       Gas metallicity (None = tie to stellar)
    neb_fesc                   Fixed(0.0)        Ionizing photon escape fraction
    neb_fesc_lya               Fixed(0.0)        Ly-alpha escape fraction
    ========================== ================= =======================================

    **Dust emission** (``dust_emission != None``):

    ========================== ================= =======================================
    dust_T                     Fixed(35)         Dust temperature (K) for graybody
    dust_beta_ir               Fixed(1.6)        Emissivity index
    dust_alpha_mir             Fixed(2.0)        MIR slope (Casey 2012)
    dust_alpha_dale            Fixed(2.0)        Dale+2014 alpha
    dust_umin                  Fixed(1.0)        DL07/DL14 minimum radiation field
    dust_gamma_dl              Fixed(0.01)       DL07/DL14 PDR fraction
    dust_qpah                  Fixed(2.5)        DL07/DL14 PAH mass fraction (%)
    dust_alpha_dl14            Fixed(2.0)        DL14 radiation field slope (1-3)
    dust_eta_balance           Fixed(1.0)        Energy balance deviation factor
    ========================== ================= =======================================

    **AGN** (``agn_model != None``):

    ========================== ================= =======================================
    agn_lum_ratio              Fixed(1.0)        L_AGN / L_stellar_bol (1.0 = full AGN)
    agn_log_lbol               Fixed(10.0)       AGN log L_bol [log10(L_sun)] (parametric)
    agn_alpha                  Fixed(-1.0)       Disc power-law slope
    agn_T_torus                Fixed(1000)       Torus temperature (K)
    agn_tau_torus              Fixed(5.0)        Torus optical depth at 9.7 um
    agn_torus_frac             Fixed(0.5)        Torus covering fraction
    agn_log_mbh                Fixed(7.0)        Black hole mass log10(M/Msun)
    agn_log_ledd               Fixed(-1.0)       Eddington ratio log10(L/L_Edd)
    agn_tau_skirtor            Fixed(7.0)        SKIRTOR 9.7 um optical depth
    agn_p_skirtor              Fixed(1.0)        SKIRTOR radial density gradient
    agn_q_skirtor              Fixed(1.0)        SKIRTOR polar density gradient
    agn_oa_skirtor             Fixed(40)         SKIRTOR opening angle (degrees)
    agn_cos_inc                Fixed(0.5)        Cosine of inclination (0=edge-on)
    ========================== ================= =======================================

    **Radio** (``radio=True``):

    ========================== ================= =======================================
    radio_q_ir                 Fixed(2.64)       FIR-radio correlation (Bell 2003)
    radio_alpha_sf             Fixed(0.8)        SF synchrotron spectral index
    radio_loudness             Fixed(0.0)        AGN radio-loudness log10(L_5GHz/L_B)
    radio_alpha_agn            Fixed(0.7)        AGN radio spectral index
    ========================== ================= =======================================

    **X-ray** (``xray=True``):

    ========================== ================= =======================================
    xray_gamma_agn             Fixed(1.8)        AGN X-ray photon index
    xray_delta_alpha_ox              Fixed(0.0)        Offset to L_2500-derived alpha_ox [dex]
    ========================== ================= =======================================

    **Evolving metallicity** (``evolving_metallicity=True``):

    ========================== ================= =======================================
    met_logzsol_0              Uniform(-2, 0.2)  Initial metallicity (oldest stars)
    met_logzsol_final          Uniform(-2, 0.2)  Final metallicity (present-day)
    ========================== ================= =======================================

    Examples
    --------
    Minimal parametric model::

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.5, 3.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
            sfh_dpl_log_total_mass=Uniform(8.0, 12.5),
            met_logzsol=Uniform(-2.0, 0.5),
            dust_tau_bc=Uniform(0.0, 2.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            redshift=Fixed(0.1),
        )

    Full model with all physics::

        spec = Parameters(
            mean_sfh_type=["dpl", "field"],
            n_grid=256,
            # Dust attenuation
            dust_law_bc="kriek_conroy",
            dust_f_obscuration=Uniform(0.0, 0.5),
            dust_bump_strength=Uniform(0.0, 5.0),
            # Dust emission (DL07 tabulated templates)
            dust_emission="dl07_tabulated",
            dl07_grid_path="data/dl07_templates.h5",
            dust_umin=Uniform(0.1, 25.0),
            # Nebular (Cue neural emulator)
            nebular_cue=True,
            neb_logU=Uniform(-4.0, -1.0),
            neb_fesc_lya=Uniform(0.0, 1.0),
            # AGN (qsogen empirical quasar)
            agn_model="qsogen",
            agn_log_lbol=Uniform(6.42, 12.42),
            # IGM
            apply_igm=True,
            # Radio + X-ray
            radio=True,
            xray=True,
            # Evolving metallicity
            evolving_metallicity=True,
            met_logzsol_0=Uniform(-2.0, 0.2),
            met_logzsol_final=Uniform(-2.0, 0.2),
            met_alpha_fe=Uniform(-0.2, 0.6),
        )
    """

    # ── Construction ──────────────────────────────────────────────────

    def __init__(self, **kwargs):
        # ── Settings ──────────────────────────────────────────────
        raw_sfh_type = kwargs.pop("mean_sfh_type", None)
        explicit_stochastic = kwargs.pop("stochastic", None)
        n_grid = int(kwargs.pop("n_grid", 256))
        # Non-parametric SFH bin edges (``prospector_beta`` and other
        # ``_NONPARAM_NAMES`` entries). Stored as a structural setting
        # so ``_build_legacy`` can forward to ``resolve_sfh(...,
        # bin_edges_gyr=...)``. Default ``None`` falls back to the
        # registry's own canonical edges. See #337.
        self.bin_edges_gyr = kwargs.pop("bin_edges_gyr", None)
        # SFH→SSP age-weight kernel ("cic" / "dsps"), or None to auto-select.
        # A structural setting, not a free parameter; forwarded to
        # ``build_components(age_kernel=...)``. See #964.
        self.age_kernel = kwargs.pop("age_kernel", None)
        # GP-field parameterization: which coordinates the field latent is
        # sampled in. 1.0 = the shipped non-centered map; a < 1 moves amplitude
        # dependence out of it. A structural setting, not a free parameter, and
        # it must travel with the matching latent prior. See #1355.
        self.field_centering = float(kwargs.pop("field_centering", 1.0))
        # MW foreground extinction screen — applied at the
        # observed-frame SED boundary, independent of host-galaxy dust
        # (#297). ``foreground_ebmv_mw=0.0`` is the no-op default.
        self.foreground_ebmv_mw = float(kwargs.pop("foreground_ebmv_mw", 0.0))
        self.foreground_law = kwargs.pop("foreground_law", "cardelli")
        self.foreground_rv = float(kwargs.pop("foreground_rv", 3.1))
        self.apply_igm = kwargs.pop("apply_igm", True)
        # IGM transmission model: 'inoue' (default), 'madau', or 'meiksin06'.
        # Stored as a structural setting so the grammar-layer choice
        # propagates through to :meth:`SEDModel._init_igm` (#344, #440).
        self.igm_model = kwargs.pop("igm_model", "inoue")

        # ── Nebular emission ──────────────────────────────────────
        self._init_nebular_config(kwargs)

        # ── Dust ──────────────────────────────────────────────────
        self._init_dust_config(kwargs)

        # ── Component flags ───────────────────────────────────────
        self.igm_patchy = kwargs.pop("igm_patchy", False)
        self.dla = kwargs.pop("dla", False)
        self.agn_model = kwargs.pop("agn_model", None)
        # AGN block-recipe selectors (consumed when agn_model="composable").
        # Validation of the (category, name) pair is deferred to
        # validate_block_recipe in the runner; we only extract the strings
        # here and let typo-detection happen at composition time so the
        # error message points at the right registry.
        self.agn_disc_block = kwargs.pop("agn_disc_block", "none")
        # Handle legacy agn_lines_block → (agn_nlr_block, agn_blr_block) split.
        # Accept both new independent selectors and deprecated combined form.
        agn_nlr_block_explicit = kwargs.pop("agn_nlr_block", None)
        agn_blr_block_explicit = kwargs.pop("agn_blr_block", None)
        agn_lines_block_legacy = kwargs.pop("agn_lines_block", None)
        if agn_lines_block_legacy is not None and agn_lines_block_legacy != "none":
            # Expand the legacy name to (nlr, blr) pair
            import warnings

            from tengri.components.agn.blocks._aliases import expand_lines_alias

            nlr_expanded, blr_expanded = expand_lines_alias(agn_lines_block_legacy)
            warnings.warn(
                f"The AGN 'lines' selector is deprecated; use agn_nlr_block / "
                f"agn_blr_block. '{agn_lines_block_legacy}' maps to "
                f"nlr='{nlr_expanded}', blr='{blr_expanded}'.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Set from expansion if user didn't provide explicit nlr/blr
            if agn_nlr_block_explicit is None:
                agn_nlr_block_explicit = nlr_expanded
            if agn_blr_block_explicit is None:
                agn_blr_block_explicit = blr_expanded
        self.agn_nlr_block = agn_nlr_block_explicit or "none"
        self.agn_blr_block = agn_blr_block_explicit or "none"
        self.agn_feii_block = kwargs.pop("agn_feii_block", "none")
        self.agn_torus_block = kwargs.pop("agn_torus_block", "none")
        self.agn_attenuation_block = kwargs.pop("agn_attenuation_block", "none")
        # Cross-block normalization policy (#556): static selector, not a
        # fittable param. "cigale_joint" (default) ties disc/torus/polar to
        # the single agn_power reference; "independent" keeps legacy scaling.
        self.agn_norm = kwargs.pop("agn_norm", "cigale_joint")
        # Block-recipe validation (typo hard-error + suspicious-combo warnings)
        # is deferred until after the parameter distributions are built, so the
        # concrete agn_polar_ebv can be passed to Rule 7 (#890). Block selectors
        # do not feed the parameter registry, so deferring does not change which
        # error a malformed recipe raises first.
        # Composable AGN precompute axes. ``dict[param_name → ndarray]``;
        # SEDModel consumes this to build a triweight lookup at construction
        # time. Defaults to None → precompute disabled, runtime path used.
        self.agn_axis_grids = kwargs.pop("agn_axis_grids", None)
        if self.agn_axis_grids is not None:
            if self.agn_model != "composable":
                raise ValueError(
                    f"agn_axis_grids requires agn_model='composable'; got {self.agn_model!r}."
                )
            if not isinstance(self.agn_axis_grids, dict):
                raise TypeError(
                    "agn_axis_grids must be a dict[param_name → ndarray]; "
                    f"got {type(self.agn_axis_grids).__name__}."
                )
            if not self.agn_axis_grids:
                raise ValueError(
                    "agn_axis_grids must contain at least one axis when set; "
                    "use agn_axis_grids=None (default) to disable precompute."
                )
        self.radio = kwargs.pop("radio", False)
        self.radio_sfr_mode = kwargs.pop("radio_sfr_mode", "bell2003")
        self.radio_agn_model = kwargs.pop("radio_agn_model", "powerlaw")
        self.xray = kwargs.pop("xray", False)
        self.xray_model = kwargs.pop("xray_model", "yang20")
        self.shock = kwargs.pop("shock", False)
        # Composable MAPPINGS V shock config (#851). ``shock_norm`` selects the
        # relative (``"frac"``) vs absolute (``"lhalpha"``) Halpha
        # normalization; ``shock_abundance`` / ``shock_component`` are the
        # categorical MAPPINGS knobs. These are static structural settings
        # (like ``radio_sfr_mode`` / ``xray_model``), not traced free params.
        self.shock_norm = kwargs.pop("shock_norm", "frac")
        self.shock_abundance = kwargs.pop("shock_abundance", "solar")
        self.shock_component = kwargs.pop("shock_component", "combined")

        # ── Metallicity ───────────────────────────────────────────
        self._init_metallicity_config(kwargs)

        # ── Emission lines ────────────────────────────────────────
        self.eline_mode = kwargs.pop("eline_mode", "off")
        if self.eline_mode not in ("off", "fixed", "marginalized", "fitted"):
            raise ValueError(
                f"eline_mode must be 'off', 'fixed', 'marginalized', or 'fitted', "
                f"got '{self.eline_mode}'"
            )
        self.eline_broad = bool(kwargs.pop("eline_broad", False))

        # --- Resolve short-form parameter aliases ---
        resolved_kwargs = {}
        detected_models = set()
        for name, val in kwargs.items():
            new_name = resolve_param_name(name)
            resolved_kwargs[new_name] = val
            # Auto-detect model from param name prefixes
            if new_name.startswith("sfh_dpl_"):
                detected_models.add("dpl")
            elif new_name.startswith("sfh_tsnorm_"):
                detected_models.add("tsnorm")
            elif new_name.startswith("sfh_snorm_"):
                detected_models.add("snorm")
            elif new_name.startswith("sfh_norm_"):
                detected_models.add("norm")
            elif new_name.startswith("sfh_lnorm_"):
                detected_models.add("lnorm")
            elif new_name.startswith("sfh_const_"):
                detected_models.add("const")
            elif new_name.startswith("sfh_exp_"):
                detected_models.add("exp")
            elif new_name.startswith("sfh_dexp_"):
                detected_models.add("dexp")
            elif new_name.startswith("sfh_burst_"):
                detected_models.add("burst")
            elif new_name.startswith("sfh_field_"):
                detected_models.add("field")

        # --- Resolve mean_sfh_type ---
        # Auto-detect model from parameter name prefixes if no explicit type given
        if raw_sfh_type is None and detected_models:
            raw_sfh_type = sorted(detected_models)

        mean_sfh_type = self._resolve_sfh_type(raw_sfh_type, explicit_stochastic, detected_models)

        # Normalize to list
        if isinstance(mean_sfh_type, str):
            mean_sfh_type = [mean_sfh_type]

        self._mean_sfh_type = mean_sfh_type
        self._n_grid = n_grid

        # --- Build dynamic parameter registry ---
        self._param_registry, self._defaults = _build_param_registry(
            mean_sfh_type,
            nebular=self.nebular_mode,
            dust_model=self.dust_model,
            dust_law_bc=self.dust_law_bc,
            dust_law_diff=self.dust_law_diff,
            dust_emission=self.dust_emission,
            agn_model=self.agn_model,
            radio=self.radio,
            xray=self.xray,
            shock=self.shock,
            igm_patchy=self.igm_patchy,
            dla=self.dla,
            met_mode=self.met_mode,
            alpha_fe_evolving=self.alpha_fe_evolving,
            eline_mode=self.eline_mode,
            eline_broad=self.eline_broad,
        )
        # --- Cue optional params (ionspec / gas extras) ---
        _cue_ionspec = _resolve_lazy_bucket("_CUE_IONSPEC_PARAMS")
        _cue_gas_extra = _resolve_lazy_bucket("_CUE_GAS_EXTRA_PARAMS")
        _ALL_CUE_OPTIONAL = {**_cue_ionspec, **_cue_gas_extra}
        if self.nebular_mode == "cue":
            # Register any optional Cue params the user explicitly provided
            for pname, (desc, check, err, default) in _ALL_CUE_OPTIONAL.items():
                if pname in resolved_kwargs:
                    self._param_registry[pname] = (desc, check, err)
                    self._defaults[pname] = default
        else:
            # Raise if user tried to set ionspec params in non-Cue mode
            ionspec_in_kwargs = [p for p in _cue_ionspec if p in resolved_kwargs]
            if ionspec_in_kwargs:
                raise ValueError(
                    f"ionspec params {ionspec_in_kwargs} require nebular_cue=True "
                    f"(current mode: '{self.nebular_mode}')."
                )

        self._valid_param_names = frozenset(self._param_registry.keys())

        # --- Extract mirror specifications (string values → param tying) ---
        # Must run before validation so mirrored params (which may reference
        # params from other modules, e.g. neb_logZ_gas → met_logzsol) are
        # converted to Fixed(0.0) before the unknown-param check.
        self._mirrors: dict[str, str] = {}
        for name, val in list(resolved_kwargs.items()):
            if (
                isinstance(val, str)
                and name in self._valid_param_names
                and val in self._valid_param_names
            ):
                self._mirrors[name] = val
                resolved_kwargs[name] = Fixed(0.0)

        for target, source in self._mirrors.items():
            if source in self._mirrors:
                raise ValueError(
                    f"Chained mirror: '{target}' -> '{source}' -> "
                    f"'{self._mirrors[source]}'. Only direct mirrors are allowed."
                )

        # --- Validate parameter names ---
        # Drop field params if field was removed (e.g., stochastic=False
        # with short-form psd_sigma/psd_tau_myr that are Fixed)
        resolved_kwargs = {
            name: val
            for name, val in resolved_kwargs.items()
            if name in self._valid_param_names or not name.startswith("sfh_field_")
        }
        for name in resolved_kwargs:
            if name not in self._valid_param_names:
                valid_sorted = sorted(self._valid_param_names)
                raise ValueError(
                    f"Unknown parameter '{name}' for mean_sfh_type={mean_sfh_type}. "
                    f"Valid parameters: {valid_sorted}"
                )

        # --- Resolve shorthands and store distributions ---
        self._distributions: dict[str, Distribution] = {}
        self._user_provided: frozenset[str] = frozenset()
        user_names = set()
        for name in sorted(self._valid_param_names):
            if name in resolved_kwargs:
                self._distributions[name] = resolve_shorthand(resolved_kwargs[name])
                user_names.add(name)
            else:
                self._distributions[name] = self._defaults[name]
        self._user_provided = frozenset(user_names)

        # Eagerly validate the composable block recipe now that distributions
        # exist: a typo raises and suspicious combos warn *before* the forward
        # model is built. Passing the concrete agn_polar_ebv (a Fixed value, not
        # a free prior) lets Rule 7 surface the polar-dust E(B-V)=0 no-op at
        # construction (#890); a free/fitted E(B-V) stays a tracer at runtime and
        # is intentionally left out so no no-op warning fires.
        # The grid-support checks need the range each parameter can actually
        # take, not its value: a prior's bounds, or (v, v) for a fixed one.
        # Every Distribution defines .bounds, so read it directly rather than
        # behind a blanket except — a guard against a silent failure must not
        # itself fail silently. A string-valued Fixed (a categorical choice)
        # reports (None, None) and is skipped.
        param_support: dict[str, tuple[float, float]] = {}
        for _name, _dist in self._distributions.items():
            _lo, _hi = _dist.bounds
            if _lo is None or _hi is None:
                continue
            param_support[_name] = (float(_lo), float(_hi))

        self._warn_on_grid_overhang(param_support)

        if self.agn_model == "composable":
            from tengri.components.agn.blocks import validate_block_recipe

            recipe_params: dict | None = None
            if self.agn_attenuation_block == "polar_dust":
                _ebv_dist = self._distributions.get("agn_polar_ebv")
                if _ebv_dist is not None and _ebv_dist.is_fixed:
                    recipe_params = {"agn_polar_ebv": float(_ebv_dist.value)}
            validate_block_recipe(
                agn_disc_block=self.agn_disc_block,
                agn_nlr_block=self.agn_nlr_block,
                agn_blr_block=self.agn_blr_block,
                agn_feii_block=self.agn_feii_block,
                agn_torus_block=self.agn_torus_block,
                agn_attenuation_block=self.agn_attenuation_block,
                params=recipe_params,
                param_support=param_support,
            )

        # E fix (#846): the physical disc blocks (multicolor, kubota_done) now
        # DERIVE the Eddington ratio from agn_log_lbol + agn_log_mbh, so
        # agn_log_ledd has no effect on them. Warn at construction (Python-side,
        # jit-safe) if a user explicitly sets or frees it, so it is never a
        # silent no-op. Gated on _user_provided; agn_log_ledd is no longer in
        # those blocks' AGN_BLOCK_CONSUMES, so a scoped '*': FREE will not free
        # it — only an explicit override triggers this.
        if (
            self.agn_model == "composable"
            and getattr(self, "agn_disc_block", "none") in ("multicolor", "kubota_done")
            and "agn_log_ledd" in self._user_provided
        ):
            _ledd_dist = self._distributions.get("agn_log_ledd")
            _ledd_active = _ledd_dist is not None and (
                not _ledd_dist.is_fixed or abs(float(_ledd_dist.value) - (-1.0)) > 1e-9
            )
            if _ledd_active:
                import warnings

                warnings.warn(
                    f"agn_log_ledd has no effect on the "
                    f"'{self.agn_disc_block}' disc: the Eddington ratio is now "
                    "derived from agn_log_lbol and agn_log_mbh "
                    "(lambda_Edd = L_bol / L_Edd, #846). Set the AGN luminosity "
                    "via agn_log_lbol and remove agn_log_ledd.",
                    UserWarning,
                    stacklevel=2,
                )

        # --- Validate physical bounds ---
        self._validate_bounds()

    def _init_nebular_config(self, kwargs):
        """Resolve nebular emission backend from kwargs."""
        nebular_ssp = kwargs.pop("nebular_ssp", False)
        nebular = kwargs.pop("nebular", False)
        nebular_cue = kwargs.pop("nebular_cue", False)
        self.cloudy_grid_path = kwargs.pop("cloudy_grid_path", None)
        self.cue_weights_path = kwargs.pop("cue_weights_path", None)
        # When True, the Cue orchestrator path publishes the full
        # ~271-species line catalog instead of the default 128
        # CLOUDY/FSPS subset, so HeII 1640, HeI 10830, etc. can be
        # read via ``pred.lines.get(wavelength)``. See #303.
        self.cue_full_catalog = kwargs.pop("cue_full_catalog", False)
        self.neb_ionization = kwargs.pop("neb_ionization", "ssp")

        self._nebular_cb19 = False
        if nebular == "cue":
            nebular_cue = True
            nebular = False
        elif nebular == "cb19":
            self._nebular_cb19 = True
            nebular = False
        elif nebular == "cloudy":
            nebular = True

        # Path implies backend
        no_explicit = not nebular_cue and not nebular and not nebular_ssp
        if self.cue_weights_path is not None and no_explicit:
            nebular_cue = True
        no_explicit_cloudy = not nebular and not nebular_cue and not nebular_ssp
        if self.cloudy_grid_path is not None and no_explicit_cloudy:
            nebular = True

        # Mutual exclusion
        n_set = sum([bool(nebular_ssp), bool(nebular), bool(nebular_cue)])
        if n_set > 1:
            raise ValueError(
                "nebular_ssp, nebular (CLOUDY), and nebular_cue are "
                "mutually exclusive — choose one."
            )

        # Resolve mode
        if nebular_cue:
            self.nebular_mode = "cue"
            if self.cue_weights_path is None:
                from tengri.components.nebular import _DEFAULT_CUE_WEIGHTS_PATH

                self.cue_weights_path = str(_DEFAULT_CUE_WEIGHTS_PATH)
        elif self._nebular_cb19:
            self.nebular_mode = "cb19"
        elif nebular:
            self.nebular_mode = "cloudy"
            if self.cloudy_grid_path is None:
                default_grid = self._default_cloudy_grid()
                if default_grid is None:
                    self._raise_missing_grid_path()
                self.cloudy_grid_path = default_grid
        elif nebular_ssp:
            self.nebular_mode = "ssp"
        else:
            self.nebular_mode = "off"

        self.nebular = self.nebular_mode != "off"

        if self.neb_ionization in ("agn", "ssp+agn"):
            raise NotImplementedError(
                "AGN ionization not yet implemented — use neb_ionization='ssp'"
            )

        # Warn if nebular_ssp user sets nebular params
        if self.nebular_mode == "ssp":
            _neb_params = _resolve_lazy_bucket("_NEBULAR_PARAMS")
            _cue_ionspec = _resolve_lazy_bucket("_CUE_IONSPEC_PARAMS")
            _cue_gas_extra = _resolve_lazy_bucket("_CUE_GAS_EXTRA_PARAMS")
            _NEB_PARAM_NAMES = set(_neb_params) | set(_cue_ionspec) | set(_cue_gas_extra)
            for name in list(kwargs):
                if name in _NEB_PARAM_NAMES:
                    import warnings

                    warnings.warn(
                        f"'{name}' is ignored with nebular_ssp=True "
                        f"(emission is baked into SSP at fixed logU/logZ).",
                        UserWarning,
                        stacklevel=2,
                    )
                    kwargs.pop(name)

    def _init_dust_config(self, kwargs):
        """Resolve dust model, attenuation law, and emission from kwargs."""
        self.dust_model = kwargs.pop("dust_model", "two_component")
        # 'none' is the user-facing spelling; 'off' is the internal sentinel the
        # forward model reads as use_dust=False (a dust-free model).
        if self.dust_model == "none":
            self.dust_model = "off"
        if self.dust_model not in ("two_component", "single_component", "wg00", "off"):
            raise ValueError(
                f"dust_model must be 'two_component', 'single_component', 'wg00', "
                f"or 'off'/'none' (no dust), got '{self.dust_model}'"
            )

        # Witt & Gordon (2000) screen (dust_model='wg00', FSPS dust_type=3):
        # static structural selectors. Always stored so the forward model and
        # compile_signature can read them via getattr.
        self.dust_wg00_curve = kwargs.pop("dust_wg00_curve", "mw")
        self.dust_wg00_geometry = kwargs.pop("dust_wg00_geometry", "shell")
        self.dust_wg00_structure = kwargs.pop("dust_wg00_structure", "homogeneous")

        self.dust_approx = kwargs.pop("dust_approx", "fast")
        if self.dust_approx not in ("fast", "exact"):
            raise ValueError(f"dust_approx must be 'fast' or 'exact', got '{self.dust_approx}'")

        # For single-component, accept `dust_law` as cleaner alias for `dust_law_bc`.
        # Pop both laws WITHOUT defaults first so inheritance can tell which the
        # user actually set.
        dust_law_alias = kwargs.pop("dust_law", None)
        law_bc_explicit = kwargs.pop("dust_law_bc", dust_law_alias)
        law_diff_explicit = kwargs.pop("dust_law_diff", None)

        if self.dust_model == "single_component":
            if law_diff_explicit is not None:
                import warnings

                warnings.warn(
                    "dust_law_diff is ignored with dust_model='single_component' "
                    "(only one attenuation curve is used).",
                    UserWarning,
                    stacklevel=2,
                )
            self.dust_law_bc = law_bc_explicit or "power_law"
            self.dust_law_diff = self.dust_law_bc
        else:
            # Symmetric inheritance: setting either law alone applies it to BOTH
            # components (a single shared attenuation curve); setting neither
            # defaults both to "power_law". This makes the birth cloud follow
            # the diffuse ISM law when only ``dust_law_diff`` is given, instead
            # of silently mixing power_law (BC) with the user's diffuse curve.
            if law_bc_explicit is None and law_diff_explicit is None:
                self.dust_law_bc = "power_law"
                self.dust_law_diff = "power_law"
            elif law_bc_explicit is None:
                self.dust_law_bc = law_diff_explicit
                self.dust_law_diff = law_diff_explicit
            elif law_diff_explicit is None:
                self.dust_law_bc = law_bc_explicit
                self.dust_law_diff = law_bc_explicit
            else:
                self.dust_law_bc = law_bc_explicit
                self.dust_law_diff = law_diff_explicit

        # Nebular birth-cloud law. None -> inherit the stellar birth cloud
        # (``dust_law_bc``), so the nebular continuum is reddened exactly like
        # the youngest stars (default). Set it to give HII-region emission its
        # own birth-cloud curve while still sharing the diffuse ISM screen.
        self.dust_law_neb = kwargs.pop("dust_law_neb", None)

        # Per-component law-parameter overrides: {'bc': {law_kwarg: value}, ...,
        # 'neb': {...}}. Empty -> both stellar components share the global
        # dust_slope / dust_bump_strength / dust_delta / dust_Rv, and the
        # nebular birth cloud inherits the stellar birth-cloud params. Set by
        # the builder from slope_bc / delta_diff / slope_neb /…
        self.dust_law_overrides = kwargs.pop("dust_law_overrides", {}) or {}
        # Lyman-limit clip [Å]: zero the attenuation curve below this wavelength
        # (0.0 -> off). Static config, set by the builder from ``lyman_cutoff``.
        self.dust_lyman_cutoff_aa = float(kwargs.pop("dust_lyman_cutoff_aa", 0.0) or 0.0)
        # Absorb ALL stellar LyC by neb_fesc (FSPS/CIGALE) vs young/birth-cloud
        # only (default; bagpipes). See DustSEDComponent.lyc_absorb_all.
        self.dust_lyc_absorb_all = bool(kwargs.pop("dust_lyc_absorb_all", False))
        # Include the LyC (λ < 912 Å) in the dust energy-balance integral
        # (FSPS/Prospector parity, ~10% higher L_IR for star-forming galaxies,
        # #961) vs the canonical LyC-masked L_absorbed (default; #922, CIGALE).
        self.dust_eb_include_lyc = bool(kwargs.pop("dust_eb_include_lyc", False))

        self.dust_emission = kwargs.pop("dust_emission", None)
        self.dl07_grid_path = kwargs.pop("dl07_grid_path", None)

        # Astrodust+PAH (HD23) optional configuration: spinning dust (AME) enable flag
        # and cold-neutral-medium filling fraction. Structural settings, not free
        # parameters; forwarded to component_factory. See #1093.
        self.astrodust_spinning_dust = bool(kwargs.pop("astrodust_spinning_dust", False))
        self.astrodust_f_cnm = float(kwargs.pop("astrodust_f_cnm", 0.28))

    def _init_metallicity_config(self, kwargs):
        """Resolve metallicity evolution mode from kwargs.

        Priority:
        1. Explicit ``met_mode="..."`` always wins.
        2. Legacy ``evolving_metallicity=True`` / ``chem_evol=True``
           flags map onto the corresponding mode.
        3. Auto-infer from the metallicity-related parameter keys the
           user passed (e.g. ``met_logzsol_0`` + ``met_logzsol_final``
           implies ``"ramp"``). Inference is data-driven from the
           registry — adding a new mode + its discriminator keys to
           ``met_registry._MET_MODE_DISCRIMINATORS`` is enough.
        4. Default ``"delta"`` if nothing matches.

        Validation: explicit ``met_mode`` that conflicts with the
        keys (e.g. ``met_mode="delta"`` while passing ``met_logzsol_0``
        and ``met_logzsol_final``) raises with a helpful message.
        """
        from tengri.components.stellar.sfh.met_registry import infer_met_mode

        _met_mode_explicit = kwargs.pop("met_mode", None)
        _evolving_met = kwargs.pop("evolving_metallicity", False)
        _chem_evol = kwargs.pop("chem_evol", False)

        # Inference only sees met_/chem_ keys; everything else is irrelevant
        # noise. Some modes share parameter keys (e.g. massmap_box's discriminator
        # is a superset of massmap_lin's), so inference can be genuinely
        # ambiguous — only run it where it is needed and let an explicit
        # ``met_mode`` win over an ambiguous inference.
        if _met_mode_explicit is not None:
            if _evolving_met or _chem_evol:
                raise ValueError(
                    "Cannot use met_mode with evolving_metallicity or chem_evol. "
                    "Use met_mode='ramp' instead of evolving_metallicity=True, "
                    "or met_mode='chem_evol' instead of chem_evol=True."
                )
            # Consistency guard: only flag a conflict when inference resolves
            # *unambiguously* to a different mode. An ambiguous inference (the
            # user already disambiguated by setting met_mode) is not a conflict.
            try:
                _inferred_mode = infer_met_mode(set(kwargs.keys()))
            except ValueError:
                _inferred_mode = _met_mode_explicit
            if _inferred_mode != "delta" and _inferred_mode != _met_mode_explicit:
                raise ValueError(
                    f"met_mode={_met_mode_explicit!r} conflicts with parameter "
                    f"keys that imply met_mode={_inferred_mode!r}. Either remove "
                    f"the conflicting parameters or set met_mode={_inferred_mode!r} "
                    f"explicitly."
                )
            self.met_mode = _met_mode_explicit
        elif _evolving_met and _chem_evol:
            raise ValueError(
                "chem_evol and evolving_metallicity are mutually exclusive. "
                "chem_evol derives Z(t) from SFH; evolving_metallicity uses "
                "a linear Z(t) ramp with met_logzsol_0/met_logzsol_final."
            )
        elif _evolving_met:
            self.met_mode = "ramp"
        elif _chem_evol:
            self.met_mode = "chem_evol"
        else:
            self.met_mode = infer_met_mode(set(kwargs.keys()))

        # Backward-compat properties for sed_model / pipeline
        self.evolving_metallicity = self.met_mode == "ramp"
        self.chem_evol = self.met_mode == "chem_evol"

        self.alpha_fe_evolving = kwargs.pop("alpha_fe_evolving", False)
        self.met_interp = kwargs.pop("met_interp", "smooth")
        self.lgmet_scatter = float(kwargs.pop("lgmet_scatter", 0.1))
        # Redshift-table interpolation mode (used when a precomputed z-table
        # is enabled via ``approx=WavePrecomp(...)`` AND redshift is free).
        # "linear" → piecewise-linear (C^0, default).
        # "smooth" → triweight kernel (C^2) — recommended for HMC/NUTS with free z.
        self.z_interp = kwargs.pop("z_interp", "linear")

    @staticmethod
    def _default_cloudy_grid():
        """Auto-resolve the default CLOUDY grid, mirroring the Cue-weights default.

        Prefers ``data/cloudy_grid_mist.h5`` at the repo root — the grid
        matching the default MIST/FSPS SSP family. Returns None when absent
        (wheel installs, grid not generated) so the caller can raise the
        listing error instead.
        """
        from tengri._data_setup import find_data

        # Honors $TENGRI_DATA_DIR as well as the repo root (#1431).
        candidate = find_data("cloudy_grid_mist.h5")
        return str(candidate) if candidate is not None else None

    @staticmethod
    def _raise_missing_grid_path():
        """Raise ValueError listing available CLOUDY grids."""
        from tengri._data_setup import data_dirs

        # Repo-level data/ — where convert_fsps_cloudy_grid.py writes grids.
        # (An earlier revision listed the packaged src/tengri/data/ directory,
        # which never contains CLOUDY grids, so the listing was always empty.)
        # Listing every searched directory keeps the message honest for users
        # whose grids live under $TENGRI_DATA_DIR (#1431).
        searched = data_dirs()
        seen: set[str] = set()
        grids = [
            g
            for d in searched
            for g in sorted(d.glob("cloudy_grid_*.h5"))
            if not (g.name in seen or seen.add(g.name))
        ]
        # Full paths, not bare names: with several search directories, a bare
        # name no longer says which one the grid came from.
        grid_list = "\n".join(f"  {g}" for g in grids) if grids else "  (none found)"
        where = ", ".join(str(d) for d in searched[:3])
        raise ValueError(
            f"The CLOUDY nebular backend needs a grid file. Pass one via "
            f"neb={{'type': 'cloudy', 'grid': 'data/cloudy_grid_<iso>.h5'}} "
            f"(or the flat kwarg cloudy_grid_path=...). "
            f"Searched {where} (and further ancestors); set $TENGRI_DATA_DIR to "
            f"point elsewhere. Available grids:\n{grid_list}\n"
            f"Generate them with scripts/convert_fsps_cloudy_grid.py, and "
            f"match the grid isochrone to your SSP for consistency."
        )

    @staticmethod
    def _resolve_sfh_type(raw_sfh_type, explicit_stochastic, detected_models=None):
        """Determine mean_sfh_type from user inputs.

        Priority:
        1. Explicit ``mean_sfh_type`` kwarg (highest)
        2. Auto-detected from parameter name prefixes
        3. ``stochastic`` kwarg (adds/removes "field")
        4. Default: ``["dpl", "field"]``
        """
        if detected_models is None:
            detected_models = set()

        if raw_sfh_type is not None:
            result = resolve_sfh_type(raw_sfh_type)

            # Honor stochastic kwarg
            if explicit_stochastic is True and "field" not in result:
                result.append("field")
            elif explicit_stochastic is False and "field" in result:
                result = [s for s in result if s != "field"]

            return result

        # No explicit mean_sfh_type and no auto-detected models
        # Use stochastic flag or default
        if explicit_stochastic is True:
            return ["dpl", "field"]
        elif explicit_stochastic is False:
            return ["dpl"]
        else:
            # Default: dpl + field
            return ["dpl", "field"]

    def _selected_grid_components(self) -> list[tuple[str, str]]:
        """``(selector, name)`` pairs for the template-backed components in play.

        Returns
        -------
        list of (str, str)
            Selectors spelled as in the build grammar, e.g.
            ``[("dust.emission", "themis")]``. AGN blocks are excluded: they are
            checked by ``validate_block_recipe``, which carries the extra
            AGN-only rules and its own message framing.

        Notes
        -----
        **JIT-compatible**: not applicable -- composition-time only.
        """
        selected: list[tuple[str, str]] = []
        dust_em = getattr(self, "dust_emission", None)
        if isinstance(dust_em, str):
            # Pass the selected name through verbatim. The registry carries
            # every selectable spelling, including aliases, so no normalization
            # happens here — an earlier draft stripped a "_tabulated" suffix
            # and thereby missed "draine_li2007" entirely.
            selected.append(("dust.emission", dust_em))
        return selected

    def _warn_on_grid_overhang(self, param_support: dict[str, tuple[float, float]]) -> None:
        """Warn when a reachable range overhangs the grid that consumes it.

        Parameters
        ----------
        param_support : dict of str to (float, float)
            ``{param_name: (lo, hi)}`` each parameter can actually take.

        Notes
        -----
        **JIT-compatible**: not applicable -- composition-time only.

        Covers every template-backed component except the AGN blocks, whose
        equivalent check lives in ``validate_block_recipe`` (#1586).
        """
        import warnings

        from tengri.components.grid_support import check_grid_support
        from tengri.config.exceptions import GridSupportWarning

        findings = check_grid_support(self._selected_grid_components(), param_support)
        for selector, name, pname, detail, (g_lo, g_hi) in findings:
            warnings.warn(
                f"{selector}={name!r}: {pname} — {detail}. The SED there is "
                "bit-identical to the edge node and the gradient is exactly "
                f"zero, so a fit cannot move it. Narrow {pname} to "
                f"[{g_lo:g}, {g_hi:g}], or select a {selector} component with "
                "no template grid.",
                GridSupportWarning,
                stacklevel=3,
            )

    def _validate_bounds(self):
        """Check that distribution bounds respect physical constraints."""
        for name, dist in self._distributions.items():
            if dist.is_fixed:
                lo = hi = dist.bounds[0]
            else:
                lo, hi = dist.bounds

            desc, check_fn, err_msg = self._param_registry[name]
            if not check_fn(lo, hi):
                raise ValueError(
                    f"Parameter '{name}' ({desc}): bounds ({lo}, {hi}) "
                    f"violate physical constraint: {err_msg}"
                )
        self._validate_orderings()

    #: Pairs that must stay ordered ``greater > lesser`` in LOOKBACK time, with the
    #: physical reason. ``bound_check`` above is strictly per-parameter and cannot
    #: express this, which is why an inverted pair passed validation and produced a
    #: silent zero-mass galaxy (#1277).
    _ORDERED_PAIRS: tuple[tuple[str, str, str], ...] = (
        (
            "sfh_const_start_gyr",
            "sfh_const_end_gyr",
            "star formation cannot stop before it starts: 'start_gyr' is the lookback "
            "to SF ONSET and 'end_gyr' the lookback to SF CESSATION, so start_gyr must "
            "be the LARGER number. The names read backwards on purpose (they are "
            "chronological, the axis is lookback), which is exactly why this is easy "
            "to invert by accident",
        ),
    )

    def _validate_orderings(self):
        """Reject parameter pairs whose ordering is physically contradictory.

        An inverted ``const`` window makes the top-hat empty, so the requested
        ``log_total_mass`` is silently discarded and the galaxy has zero flux — and
        because the shape is identically zero, the gradient w.r.t. both bounds is
        zero too, giving a gradient sampler an absorbing basin with no way out
        (#1277). Free priors are rejected when their supports *overlap*, since an
        overlap means the sampler can reach the dead region: measured at 21.8 % of
        prior draws for ``start_gyr=Uniform(0.5, 10)``, ``end_gyr=Uniform(0, 5)``.
        """
        for greater, lesser, reason in self._ORDERED_PAIRS:
            if greater not in self._distributions or lesser not in self._distributions:
                continue
            # Lowest value ``greater`` can reach vs highest ``lesser`` can reach:
            # the only pair that can violate the ordering. ``bounds`` is a 1-tuple
            # for Fixed and (lo, hi) for a prior, so [0]/[-1] covers both.
            g_lo = self._distributions[greater].bounds[0]
            l_hi = self._distributions[lesser].bounds[-1]
            if g_lo > l_hi:
                continue  # every reachable pair is correctly ordered
            both_fixed = (
                self._distributions[greater].is_fixed and self._distributions[lesser].is_fixed
            )
            if both_fixed:
                raise ValueError(
                    f"'{greater}' ({g_lo}) must be greater than '{lesser}' ({l_hi}): {reason}."
                )
            raise ValueError(
                f"The priors on '{greater}' (reaching down to {g_lo}) and '{lesser}' "
                f"(reaching up to {l_hi}) overlap, so the sampler can reach "
                f"{greater} <= {lesser}, where {reason}. That region returns a "
                "zero-mass galaxy with an exactly-zero gradient, which a gradient-based "
                f"sampler cannot escape. Give them non-overlapping supports (raise the "
                f"lower bound of '{greater}' above the upper bound of '{lesser}', or "
                "lower the latter), or fix one of them."
            )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def stochastic(self) -> bool:
        """Whether the model includes a GP field component.

        Returns
        -------
        bool
            True if 'field' is in mean_sfh_type, False otherwise.
        """
        return "field" in self._mean_sfh_type

    @property
    def n_grid(self) -> int:
        """GP grid size (only relevant when stochastic=True).

        Returns
        -------
        int
            Number of latent dimensions for stochastic SFH field.
        """
        return self._n_grid

    @property
    def mean_sfh_type(self) -> list[str]:
        """SFH model type(s) as a list of strings.

        Returns
        -------
        list[str]
            Normalized, sorted list of active SFH model names
            (e.g., ['dpl', 'field']).
        """
        return list(self._mean_sfh_type)

    @property
    def all_params(self) -> list[str]:
        """All parameter names (sorted, excludes settings).

        Returns
        -------
        list[str]
            Sorted list of all fittable and fixed parameters
            (free_params + fixed_params).
        """
        return sorted(self._distributions.keys())

    @property
    def free_params(self) -> list[str]:
        """Names of free (non-fixed) parameters.

        Returns
        -------
        list[str]
            Sorted list of parameter names that are not fixed
            (vary during sampling and inference).
        """
        return sorted(k for k, d in self._distributions.items() if not d.is_fixed)

    @property
    def fixed_params(self) -> list[str]:
        """Names of fixed parameters.

        Returns
        -------
        list[str]
            Sorted list of parameter names with constant values
            (not varied during sampling or inference).
        """
        return sorted(k for k, d in self._distributions.items() if d.is_fixed)

    @property
    def n_free(self) -> int:
        """Number of free parameters (excludes sfh_field_xi).

        Returns
        -------
        int
            Count of all non-fixed parameters available for inference.
        """
        return len(self.free_params)

    @property
    def n_latent(self) -> int:
        """Flattened dimensionality of all free parameters (#1408).

        Sum of each free parameter's flattened size, including vector latents
        (e.g., sfh_field_xi). This is the true sampled dimension for MCMC/VI —
        what samplers actually see — as opposed to n_free which counts named
        parameters only.

        Returns
        -------
        int
            Flattened size of the free-parameter pytree. Equals n_free for
            scalar-only specs; exceeds n_free when vector latents are present
            (e.g., stochastic SFH fields).

        Notes
        -----
        Used by auto-pick methods to choose NUTS vs raytrace (#1408): a field
        model with few named parameters but hundreds of field latents must
        route to the high-D sampler, not NUTS (dense mass matrix OOM risk).

        Mirrors the accounting of ``Fitter._initialize_unbounded`` — the two
        must stay in agreement (asserted against the engine's ``d_total`` in
        ``tests/inference/test_noise_broadcast_fix_1303.py``). ``sample()``
        is NOT the right source: its tree carries fixed/default parameters
        too, and the field latent under a non-free name.
        """
        import numpy as np

        get_shape = getattr(self, "param_init_shape", lambda _n: ())
        total = 0
        for name in self.free_params:
            shape = get_shape(name)
            total += int(np.prod(shape)) if shape else 1
        if self.stochastic:
            psd_shape = getattr(self, "psd_xi_init_shape", None) or (self.n_grid,)
            if callable(psd_shape):
                psd_shape = psd_shape()
            total += int(np.prod(psd_shape))
        return total

    @property
    def valid_param_names(self) -> frozenset:
        """Set of valid parameter names for this model configuration.

        Returns
        -------
        frozenset
            Immutable set of all parameter names allowed for this configuration,
            excluding settings and model configuration keys.
        """
        return self._valid_param_names

    @property
    def mirrors(self) -> dict[str, str]:
        """Parameter mirrors: {target_name: source_name}.

        Returns
        -------
        dict[str, str]
            Mapping of tied parameter names to their source parameters.
            When resolved, the target takes the value of the source.
        """
        return dict(self._mirrors)

    # ── Public API ────────────────────────────────────────────────────

    def to_groups(self) -> dict:
        """Convert this Parameters to nested-dict form.

        Inverts :func:`tengri.parse_groups` by reconstructing the nested-dict
        structure that would reproduce this Parameters when re-parsed. Uses
        provenance metadata to collapse wildcard-expanded parameters and
        preserve explicit overrides.

        Returns
        -------
        dict
            Nested-dict suitable for re-passing to
            ``tengri.parse_groups(**result)`` or ``SEDModel.build(ssp, **result)``.

        See Also
        --------
        tengri.parse_groups : The inverse operation.
        tengri.SEDModel.build : End-to-end model construction from nested dicts.

        Notes
        -----
        **Provenance-aware collapsing**: If this Parameters was built via
        ``parse_groups``, provenance tags are used to collapse parameters that
        shared the same wildcard marker (``'all_params': FREE`` or
        ``'all_params': FIXED``) back into that wildcard, with explicit
        overrides listed separately.

        **Flat-built fallback**: If this Parameters was built via flat-kwarg
        ``Parameters(...)``, all parameters are listed explicitly (no wildcard).

        **Roundtrip guarantee**: The output dict, when passed to
        ``parse_groups(**output)``, produces a Parameters with identical
        free/fixed partitions and distributions, *and* identical structural
        settings — including every group's ``type``.

        The wording here used to stop at "distributions", and that narrowness
        is why two rounds of structural loss went unnoticed: the guarantee was
        true as written while ``sfh['age_kernel']`` (#964) and then the whole
        AGN block and ``neb={'type': 'ssp'}`` (#1777) were silently reverting
        to their defaults. Keep this sentence and
        ``parameters_to_groups``'s copy of it in step.

        Examples
        --------
        >>> from tengri import parse_groups, FREE, FIXED, Uniform, Fixed
        >>> spec = parse_groups(
        ...     sfh={"type": "dpl", "all_params": FREE, "beta": Uniform(1, 3)},
        ...     redshift=Fixed(0.05),
        ... )
        >>> groups = spec.to_groups()
        >>> assert "all_params" in groups["sfh"]  # preferred spelling on output
        >>> roundtripped = parse_groups(**groups)
        >>> spec.free_params == roundtripped.free_params
        True
        """
        from tengri.parameters.groups import parameters_to_groups

        return parameters_to_groups(self)

    def with_params(self, **kwargs) -> Parameters:
        """Return a new Parameters with additional parameters merged in.

        Creates an independent copy of this Parameters with extra parameters
        added (usually observation-level parameters like calibration or noise).
        User-defined parameters take precedence — if a name already exists in
        this spec, the new value is silently skipped (user intent is preserved).

        Typically used internally by ``SEDModel`` to auto-inject noise and
        calibration parameters into the specification.

        Parameters
        ----------
        **kwargs
            Parameter name → Distribution (or scalar/tuple shorthand).
            Only params not already explicitly provided by the user are added.

        Returns
        -------
        Parameters
            New instance with merged parameters. The original is not modified
            (immutable pattern).

        Notes
        -----
        **Immutability**: The original Parameters object is never modified.
        A new Parameters instance is created via ``copy.copy()``, with internal
        mutable structures (dicts) replaced with copies.

        **Parameter priority**: User-provided parameters (via __init__) take
        absolute precedence. Auto-merged parameters are added only if their
        name is not in the user-provided set.

        Examples
        --------
        >>> from tengri import Parameters, Uniform
        >>> spec = Parameters(redshift=0.1)
        >>> # Merge in observation-level calibration parameters
        >>> spec_aug = spec.with_params(
        ...     cal_offset_aper=Uniform(-0.1, 0.1),
        ...     noise_frac=Uniform(0, 0.05),
        ... )
        >>> print(set(spec_aug.all_params) - set(spec.all_params))
        {'cal_offset_aper', 'noise_frac'}
        """
        if not kwargs:
            return self

        new_spec = copy.copy(self)
        # Deep-copy mutable internals so the original is untouched
        new_distributions = dict(self._distributions)
        new_registry = dict(self._param_registry)
        new_defaults = dict(self._defaults)

        for name, val in kwargs.items():
            if name in self._user_provided:
                # User explicitly set this param — their definition wins
                continue
            dist = resolve_shorthand(val)
            new_distributions[name] = dist
            new_registry[name] = (
                f"Auto-merged from Observation ({name})",
                lambda lo, hi: True,
                "",
            )
            new_defaults[name] = dist

        object.__setattr__(new_spec, "_distributions", new_distributions)
        object.__setattr__(new_spec, "_param_registry", new_registry)
        object.__setattr__(new_spec, "_defaults", new_defaults)
        object.__setattr__(
            new_spec,
            "_valid_param_names",
            frozenset(new_registry.keys()),
        )
        # Preserve user_provided set — auto-merged params are NOT user-provided
        object.__setattr__(new_spec, "_user_provided", self._user_provided)
        return new_spec

    def resolve_mirrors(self, params: dict) -> dict:
        """Copy mirrored parameter values from source to target.

        For each mirror ``target → source``, copies the sampled source value
        to the target parameter. Used after sampling to ensure tied parameters
        have identical values. Returns a new dict (immutable pattern).

        Parameters
        ----------
        params : dict[str, ndarray]
            Parameter name → sampled value. Must include all source parameters.

        Returns
        -------
        dict[str, ndarray]
            New dict with mirrored values filled in. For each target, the
            sampled value of the source parameter is assigned. Non-mirrored
            parameters are unchanged.

        Notes
        -----
        **Parameter tying**: Mirrors are specified in __init__ by passing
        a source parameter name as a string instead of a distribution::

            Parameters(
                neb_logZ_gas="met_logzsol",  # Gas Z tied to stellar Z
                ...
            )

        This is more elegant than using Fixed(0) + manual post-hoc copying.

        Examples
        --------
        >>> from tengri import Parameters
        >>> spec = Parameters(
        ...     met_logzsol=(-2, 0.5),
        ...     neb_logZ_gas="met_logzsol",  # Mirror: neb → met
        ... )
        >>> params = {"met_logzsol": -0.3, "neb_logZ_gas": 0.0}
        >>> resolved = spec.resolve_mirrors(params)
        >>> print(resolved["neb_logZ_gas"])
        -0.3
        """
        if not self._mirrors:
            return params
        out = dict(params)
        for target, source in self._mirrors.items():
            out[target] = out[source]
        return out

    def get_distribution(self, name: str) -> Distribution:
        """Get the prior distribution object for a parameter.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        Distribution
            The prior distribution object (Uniform, Gaussian, LogUniform, etc.)
            or Fixed for non-free parameters.

        Raises
        ------
        KeyError
            If parameter name is not valid for this model configuration.

        Notes
        -----
        For fixed parameters, the returned Distribution has ``is_fixed=True``.
        For free parameters, the returned Distribution is one of Uniform,
        Gaussian, LogUniform, LogNormal, StudentT, or other prior types.

        Examples
        --------
        >>> from tengri import Parameters, Uniform
        >>> spec = Parameters(
        ...     dust_tau_bc=Uniform(0, 4),
        ...     redshift=0.1,
        ... )
        >>> prior = spec.get_distribution("dust_tau_bc")
        >>> print(prior)
        Uniform(0, 4)
        """
        if name not in self._distributions:
            raise KeyError(f"Unknown parameter '{name}'")
        return self._distributions[name]

    def get_fixed_values(self) -> dict[str, float]:
        """Extract all numeric fixed parameter values as a dict.

        Fixed (non-free) parameters are constants that do not vary during
        inference. This method returns only the numeric ones; categorical
        Fixed parameters (strings) are excluded.

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, float]
            Mapping of numeric fixed parameter names to their constant values.
            String-valued Fixed parameters are not included because they cannot
            be represented as float.

        Notes
        -----
        This is useful for freezing parameters before optimization, or for
        passing to upstream code that requires a flat parameter vector.

        Examples
        --------
        >>> from tengri import Parameters
        >>> spec = Parameters(
        ...     redshift=0.1,
        ...     dust_tau_bc=(0, 4),
        ...     eline_broad="broad",  # String-valued
        ... )
        >>> fixed = spec.get_fixed_values()
        >>> print(fixed)
        {'redshift': 0.1}
        """
        result: dict[str, float] = {}
        for name, dist in self._distributions.items():
            if dist.is_fixed:
                v = dist.bounds[0]
                if v is not None:
                    result[name] = float(v)
        return result

    def is_fixed(self, name: str) -> bool:
        """Check whether a parameter is fixed (non-free).

        Parameters
        ----------
        name : str
            Parameter name (e.g., ``"redshift"``, ``"met_logzsol"``).

        Returns
        -------
        bool
            ``True`` if the parameter is fixed; ``False`` if free.

        Raises
        ------
        KeyError
            If the parameter name is not in the specification.

        Examples
        --------
        >>> from tengri import Parameters, Uniform, Fixed
        >>> spec = Parameters(
        ...     redshift=Fixed(0.1),
        ...     dust_tau_bc=Uniform(0, 4),
        ... )
        >>> spec.is_fixed("redshift")
        True
        >>> spec.is_fixed("dust_tau_bc")
        False
        """
        if name not in self._distributions:
            raise KeyError(f"Unknown parameter '{name}'")
        return self._distributions[name].is_fixed

    def fixed_value(self, name: str) -> float | str | None:
        """Get the fixed value of a parameter.

        Parameters
        ----------
        name : str
            Parameter name (e.g., ``"redshift"``).

        Returns
        -------
        float | str | None
            The fixed value. Numeric values are returned as float;
            string-valued enums as str; None if the parameter is not fixed
            or if the fixed value is None.

        Raises
        ------
        KeyError
            If the parameter name is not in the specification.
        ValueError
            If the parameter is not fixed.

        Examples
        --------
        >>> from tengri import Parameters, Fixed
        >>> spec = Parameters(redshift=Fixed(0.1))
        >>> spec.fixed_value("redshift")
        0.1
        """
        if name not in self._distributions:
            raise KeyError(f"Unknown parameter '{name}'")
        dist = self._distributions[name]
        if not dist.is_fixed:
            raise ValueError(f"Parameter '{name}' is not fixed")
        v = dist.bounds[0]
        # Try to convert to float if numeric; return as-is if string or None
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    def merge_observation_params(self, **extra_params: Distribution) -> Parameters:
        """Return a copy augmented with extra observation-level parameters.

        Used by inference to inject emission-line amplitude parameters so they
        flow through bounds, prior penalty loops, and summary output without
        requiring special-casing in downstream code.

        Parameters
        ----------
        **extra_params : Distribution
            Mapping of parameter name → Distribution to add (e.g.,
            ``eline_EW_Halpha=Uniform(0, 1000)``).

        Returns
        -------
        Parameters
            New Parameters instance with ``extra_params`` included in
            ``free_params``. The original instance is not modified
            (immutable pattern).

        Notes
        -----
        This is distinct from ``with_params()`` in that all extra parameters
        are unconditionally added, whereas with_params() respects user-provided
        settings.

        Examples
        --------
        >>> from tengri import Parameters, Uniform
        >>> spec = Parameters(redshift=0.1)
        >>> spec_aug = spec.merge_observation_params(
        ...     eline_EW_Halpha=Uniform(0, 1000),
        ...     eline_EW_OIII=Uniform(0, 500),
        ... )
        >>> print(spec_aug.n_free - spec.n_free)
        2
        """
        new_spec = copy.copy(self)
        new_spec._distributions = {**self._distributions, **extra_params}
        new_spec._valid_param_names = self._valid_param_names | frozenset(extra_params.keys())
        return new_spec

    def sample(self, key: jax.Array) -> dict[str, jnp.ndarray]:
        """Draw one random sample from all parameter prior distributions.

        Samples all free parameters from their priors, returns fixed parameters
        at their fixed values, and (if stochastic) generates the latent field
        ξ ~ N(0,I). Mirrors are resolved (target ← source value).

        Parameters
        ----------
        key : jax.Array (PRNGKey)
            Random key for sampling.

        Returns
        -------
        dict[str, ndarray]
            Parameter name → sampled value. Free parameters are sampled from
            their prior distributions. Fixed parameters return their constant
            value (as float or string). If stochastic, ``sfh_field_xi`` is an
            array of shape ``(n_grid,)``. Dictionary is immutable-ready (no
            direct mutation of values).

        Notes
        -----
        **JIT-compatible**: yes — safe to call inside :func:`jax.jit` on the
        key and parameters only (not on branching logic).

        **Stochastic SFH**: When the model includes a GP field, ``sfh_field_xi``
        is an independent N(0,1) vector of length n_grid. The SED model uses
        this to generate the stochastic log-SFR perturbations.

        Examples
        --------
        >>> import jax.random
        >>> from tengri import Parameters, Uniform
        >>> spec = Parameters(
        ...     sfh_dpl_alpha=Uniform(0.5, 3.0),
        ...     sfh_dpl_beta=Uniform(0.5, 3.0),
        ...     redshift=0.1,
        ... )
        >>> key = jax.random.PRNGKey(42)
        >>> samples = spec.sample(key)
        >>> print(sorted(samples.keys()))
        ['redshift', 'sfh_dpl_alpha', 'sfh_dpl_beta']

        Per-parameter substreams
        ------------------------
        Each parameter is drawn from a substream derived from its **name**
        (``fold_in(key, crc32(name))``), not from its position in a
        ``jax.random.split``. This guarantees that a given parameter samples
        to the same value for a given ``key`` **regardless of which other
        parameters are free in the spec**.

        Without this, two specs sharing a free parameter but differing in
        their free-parameter set (e.g. one adds ``dust_emission`` →
        free ``dust_T``/``dust_beta_ir``) would split the key differently and
        draw the *shared* parameter to different values — a silent footgun
        that surfaced in #548 (an eta=0 "energy-balance" inequivalence that
        was really two galaxies with different sampled ``sfh_dpl_age_gyr``)
        and #563. crc32 (not the salted built-in ``hash``) keeps the mapping
        reproducible across processes.
        """
        params = {}
        for name in sorted(self._distributions.keys()):
            subkey = jax.random.fold_in(key, _stable_param_seed(name))
            params[name] = self._distributions[name].sample(subkey)

        if self.stochastic:
            xi_key = jax.random.fold_in(key, _stable_param_seed("sfh_field_xi"))
            params["sfh_field_xi"] = jax.random.normal(xi_key, shape=(self._n_grid,))

        return self.resolve_mirrors(params)

    def sample_batch(self, key: jax.Array, n: int) -> dict[str, jnp.ndarray]:
        """Draw n random samples from all parameter prior distributions.

        Vectorized sampling via :func:`jax.vmap`. Each parameter becomes
        a batch of n independent samples.

        Parameters
        ----------
        key : jax.Array (PRNGKey)
            Random key for sampling.
        n : int
            Number of independent samples to draw.

        Returns
        -------
        dict[str, ndarray]
            Parameter name → array of samples. Each entry has shape:

            - ``(n,)`` for scalar parameters
            - ``(n, n_grid)`` for ``sfh_field_xi`` (stochastic SFH only)

        Notes
        -----
        **JIT-compatible**: yes. The function is implemented via
        :func:`jax.vmap` applied to the single-sample method.

        **Memory**: For n=1000, a 20-parameter model, and n_grid=64 (stochastic),
        the output dict occupies roughly 100 KB.

        Examples
        --------
        >>> import jax.random
        >>> from tengri import Parameters, Uniform
        >>> spec = Parameters(
        ...     sfh_dpl_alpha=Uniform(0.5, 3.0),
        ...     dust_tau_bc=Uniform(0, 4),
        ... )
        >>> key = jax.random.PRNGKey(0)
        >>> batch = spec.sample_batch(key, n=100)
        >>> print(batch["sfh_dpl_alpha"].shape)
        (100,)
        """
        keys = jax.random.split(key, n)
        return jax.vmap(self.sample)(keys)

    def validate(self, params: dict[str, jnp.ndarray]) -> None:
        """Check that all parameter values respect their distribution bounds.

        Useful before inference or after optimization to ensure no parameter
        has drifted outside its valid range.

        Parameters
        ----------
        params : dict[str, ndarray or float or str]
            Parameter name → value (sampled or optimized).

        Returns
        -------
        None
            Returns nothing. Raises an exception if validation fails.

        Raises
        ------
        ValueError
            If any parameter is outside its bounds. Fixed parameters are
            always valid.

        Notes
        -----
        Missing parameters (not in dict) are silently ignored — this allows
        checking partial parameter sets.

        Examples
        --------
        >>> from tengri import Parameters, Uniform
        >>> spec = Parameters(dust_tau_bc=Uniform(0, 4))
        >>> params_valid = {"dust_tau_bc": 2.0}
        >>> spec.validate(params_valid)  # OK
        >>> params_bad = {"dust_tau_bc": 5.0}
        >>> spec.validate(params_bad)  # Raises ValueError
        """
        for name, dist in self._distributions.items():
            if name not in params:
                continue
            val = float(params[name])
            lo, hi = dist.bounds
            if not dist.is_fixed and (val < lo or val > hi):
                raise ValueError(f"Parameter '{name}' = {val} is outside bounds [{lo}, {hi}]")

    def summary_str(self) -> str:
        """Return the summary as a string (e.g. for logging or tests)."""
        return self._build_summary_str()

    def summary(self) -> None:
        """Print a human-readable summary of the model configuration.

        Displays SFH type, enabled components (nebular, dust, AGN, etc.),
        dimensionality, and a table of all parameters grouped by category
        (free first, then fixed). Useful for printing model status before fitting.

        Use :meth:`summary_str` if you need the underlying string (e.g. for
        logging) — :meth:`summary` itself prints and returns ``None``,
        matching the rest of the discovery API
        (:func:`tengri.summary`, :func:`tengri.help`, etc.).

        Parameters
        ----------
        None

        Returns
        -------
        None
            Output is printed to stdout.

        Notes
        -----
        Output includes:

        - SFH type and composition
        - Dimensions (n free, latent ξ, mirrored, fixed)
        - Enabled optional modules (nebular, dust_emission, AGN, etc.)
        - Tabular list of parameters with their distributions/values

        Examples
        --------
        >>> from tengri import Parameters, Uniform
        >>> spec = Parameters(
        ...     mean_sfh_type="dpl",
        ...     sfh_dpl_alpha=Uniform(0.5, 3),
        ...     dust_tau_bc=Uniform(0, 4),
        ... )
        >>> print(spec.summary())
        Parameters  SFH: dpl
        ────────────────────────────────────────────────────────────
          Dimensions:  3 free + 6 fixed
          Modules:     none
        ────────────────────────────────────────────────────────────
        Free parameters:
          sfh_dpl_alpha            Uniform(0.5, 3)
          sfh_dpl_beta             Uniform(0.5, 3)
          ...
        """
        lines: list[str] = []
        sep = "─" * 66

        # Header
        sfh_label = "+".join(self._mean_sfh_type)
        lines.append(f"Parameters  SFH: {sfh_label}")
        lines.append(sep)

        # Dimensionality
        n_free = self.n_free
        n_mirror = len(self._mirrors)
        n_fixed = len(self.fixed_params) - n_mirror
        dim_parts = [f"{n_free} free"]
        if self.stochastic:
            dim_parts.append(f"+ {self._n_grid} latent (ξ)")
        if n_mirror:
            dim_parts.append(f"+ {n_mirror} mirrored")
        dim_parts.append(f"+ {n_fixed} fixed")
        lines.append(f"  Dimensions:  {', '.join(dim_parts)}")

        # Enabled modules
        modules: list[str] = []
        if self.nebular_mode != "off":
            modules.append(f"nebular={self.nebular_mode}")
        dust_em = getattr(self, "dust_emission", None)
        if dust_em:
            modules.append(f"dust_emission={dust_em}")
        agn = getattr(self, "agn_model", None)
        if agn:
            if agn == "composable":
                # Surface the composable block selectors + cross-block
                # normalization policy so the energy-balance choice is visible
                # in the model description (not buried in source).
                _blocks = [
                    f"disc={getattr(self, 'agn_disc_block', 'none')}",
                    f"torus={getattr(self, 'agn_torus_block', 'none')}",
                ]
                _norm = getattr(self, "agn_norm", "cigale_joint")
                modules.append(f"agn=composable[{', '.join(_blocks)}, norm={_norm}]")
            else:
                modules.append(f"agn={agn}")
        if getattr(self, "apply_igm", False):
            modules.append("igm")
        if getattr(self, "dla", False):
            modules.append("dla")
        if getattr(self, "radio", False):
            modules.append("radio")
        if getattr(self, "xray", False):
            modules.append("xray")
        if getattr(self, "shock", False):
            modules.append("shock")
        dust_mdl = getattr(self, "dust_model", "two_component")
        if dust_mdl == "single_component":
            modules.append(f"dust=single({getattr(self, 'dust_law_bc', 'power_law')})")
        else:
            dust_bc = getattr(self, "dust_law_bc", "power_law")
            dust_diff = getattr(self, "dust_law_diff", None) or dust_bc
            if dust_bc != "power_law" or dust_diff != "power_law":
                modules.append(f"dust_law={dust_bc}/{dust_diff}")
        met_mode = getattr(self, "met_mode", "delta")
        if met_mode != "delta":
            modules.append(f"met={met_mode}")
        if modules:
            lines.append(f"  Modules:     {', '.join(modules)}")
        lines.append("")

        # Provenance tagging (only present when built via parse_groups)
        provenance = getattr(self, "_group_provenance", None)
        _TAGS = {
            "user_prior": "[user]",
            "user_fixed": "[user]",
            "user_free": "[user FREE]",
            "wildcard_free": f"[{WILDCARD_ALIAS} FREE]",
            "wildcard_fixed": f"[{WILDCARD_ALIAS} FIXED]",
            "registry_default": "[default]",
            # The "_grid" suffix marks a declared free prior that was
            # intersected with the selected component's template grid, whose
            # axes it overhung. Shown, never silent: the printed range is then
            # not the one the declaration carries (#1586).
            "user_free_grid": "[user FREE -> grid]",
            "wildcard_free_grid": f"[{WILDCARD_ALIAS} FREE -> grid]",
            # A wildcard-FREE that found no declared prior. The parameter stays
            # Fixed, so reporting the request would put a row reading FREE
            # inside the Fixed block (#1726). The remedy is in the tag: give it
            # a prior explicitly.
            "wildcard_free_pinned": f"[{WILDCARD_ALIAS} FREE -> pinned, no prior]",
        }

        # Parameter table
        if provenance:
            hdr = f"  {'Parameter':<32s} {'Prior':<26s} {'Bounds':<22s} {'Source'}"
        else:
            hdr = f"  {'Parameter':<32s} {'Prior':<26s} {'Bounds'}"
        lines.append(hdr)
        lines.append("  " + "─" * (90 if provenance else 64))

        # Group: free parameters first, then fixed
        for name in self.free_params:
            dist = self._distributions[name]
            lo, hi = dist.bounds
            prior_str = repr(dist)
            bounds_str = f"[{lo:.4g}, {hi:.4g}]"
            if provenance:
                tag = _TAGS.get(provenance.get(name, "registry_default"), "")
                lines.append(f"  {name:<32s} {prior_str:<26s} {bounds_str:<22s} {tag}")
            else:
                lines.append(f"  {name:<32s} {prior_str:<26s} {bounds_str}")

        if self.fixed_params:
            lines.append("  " + "─" * (90 if provenance else 64))
            for name in self.fixed_params:
                if name in self._mirrors:
                    continue
                dist = self._distributions[name]
                val = dist.bounds[0]
                if provenance:
                    tag = _TAGS.get(provenance.get(name, "registry_default"), "")
                    val_str = f"{val:.4g}"
                    lines.append(f"  {name:<32s} {'Fixed':<26s} {val_str:<22s} {tag}")
                else:
                    lines.append(f"  {name:<32s} {'Fixed':<26s} {val:.4g}")

        if self._mirrors:
            lines.append("  " + "─" * 64)
            for target, source in self._mirrors.items():
                mirror_str = f"Mirror({source})"
                lines.append(f"  {target:<32s} {mirror_str:<26s} ──►")

        lines.append(sep)
        _display("\n".join(lines))

    def _build_summary_str(self) -> str:
        """Build the same multi-line summary, returned as a string."""
        # Reuse summary() logic by capturing stdout — simpler than refactor
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.summary()
        return buf.getvalue().rstrip("\n")

    def __repr__(self) -> str:
        lines = [f"Parameters(mean_sfh_type={self._mean_sfh_type},"]
        for name in sorted(self._distributions.keys()):
            dist = self._distributions[name]
            lines.append(f"    {name:30s} = {dist!r},")
        if self.stochastic:
            lines.append(f"    {'n_grid':30s} = {self._n_grid},")
        lines.append(")")
        return "\n".join(lines)

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
        sfh_tsnorm_log_peak_sfr = Uniform(-1, 2),
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
        sfh_dpl_log_peak_sfr = Uniform(-1, 2),
        ...
    )
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp

from tengri.parameters._aliases import (
    resolve_param_name,
    resolve_sfh_type,
)
from tengri.parameters._param_defs import (
    _CUE_GAS_EXTRA_PARAMS,
    _CUE_IONSPEC_PARAMS,
    _DUST_EMISSION_PARAMS,
    _NEBULAR_PARAMS,
    SETTINGS_KEYS,
    _build_param_registry,
)
from tengri.parameters.priors import (
    Distribution,
    Fixed,
    resolve_shorthand,
)

__all__ = ["SETTINGS_KEYS", "_DUST_EMISSION_PARAMS", "Parameters"]


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
    - **Distribution object** — ``Uniform``, ``Gaussian``, ``LogUniform``,
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

    Dust Attenuation Settings
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    dust_law_bc : str
        Attenuation curve for birth cloud.  Default: ``"power_law"``.
        Options: ``power_law``, ``calzetti``, ``kriek_conroy``, ``smc``,
        ``cardelli``, ``salim``, ``li08``.
    dust_law_diff : str
        Attenuation curve for diffuse ISM.  Default: same as ``dust_law_bc``.
        Can be different for per-component control.

    Dust Emission Settings
    ~~~~~~~~~~~~~~~~~~~~~~
    dust_emission : str or None
        IR emission model.  Default: ``None`` (disabled).
        Options: ``"modified_blackbody"``, ``"casey2012"``, ``"dale2014"``,
        ``"draine_li2007"``, ``"draine_li2014"``, ``"dl07_tabulated"``,
        ``"astrodust"``, ``"bosa"``, ``"themis"``.
    dl07_grid_path : str
        Path to DL07 HDF5 template grid (for ``"dl07_tabulated"``).

    Nebular Emission Settings
    ~~~~~~~~~~~~~~~~~~~~~~~~~
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

    AGN Settings
    ~~~~~~~~~~~~
    agn_model : str or None
        AGN SED model.  Default: ``None`` (disabled).
        Options: ``"simple"`` (3 params), ``"standard"`` (SS73 disc + 2T torus),
        ``"kubota_done"`` (physical disc), ``"unified_nlr_blr"`` (NLR/BLR with
        geometric masking), ``"qsogen"`` (empirical quasar, Temple+2021),
        ``"skirtor"`` (clumpy torus RT templates, Stalevski+2016).

    Multi-wavelength Settings
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    radio : bool
        Enable radio synchrotron + AGN jet emission.  Default: ``False``.
    xray : bool
        Enable X-ray (XRB + AGN corona) emission.  Default: ``False``.

    IGM Settings
    ~~~~~~~~~~~~
    apply_igm : bool
        Apply Inoue+2014 IGM absorption.  Default: ``True``.

    Metallicity Settings
    ~~~~~~~~~~~~~~~~~~~~
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
    dust_T                     Fixed(35)         Dust temperature (K) for greybody
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
    agn_frac                   Fixed(0.0)        AGN fraction of stellar L_bol
    agn_log_lbol               Fixed(10.0)       AGN log L_bol [erg/s] (parametric)
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
    xray_alpha_ox              Fixed(-1.4)       UV-to-X-ray slope
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
            sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
            met_logzsol=Uniform(-2.0, 0.5),
            dust_tau_bc=Uniform(0.0, 2.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            redshift=Fixed(0.1),
        )

    Full model with all physics::

        spec = Parameters(
            mean_sfh_type=["dpl", "field"],
            n_grid=64,
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
            agn_log_lbol=Uniform(40.0, 46.0),
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
        n_grid = int(kwargs.pop("n_grid", 64))
        self.apply_igm = kwargs.pop("apply_igm", True)

        # ── Nebular emission ──────────────────────────────────────
        self._init_nebular_config(kwargs)

        # ── Dust ──────────────────────────────────────────────────
        self._init_dust_config(kwargs)

        # ── Component flags ───────────────────────────────────────
        self.igm_patchy = kwargs.pop("igm_patchy", False)
        self.dla = kwargs.pop("dla", False)
        self.agn_model = kwargs.pop("agn_model", None)
        self.radio = kwargs.pop("radio", False)
        self.xray = kwargs.pop("xray", False)
        self.shock = kwargs.pop("shock", False)

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
        _ALL_CUE_OPTIONAL = {**_CUE_IONSPEC_PARAMS, **_CUE_GAS_EXTRA_PARAMS}
        if self.nebular_mode == "cue":
            # Register any optional Cue params the user explicitly provided
            for pname, (desc, check, err, default) in _ALL_CUE_OPTIONAL.items():
                if pname in resolved_kwargs:
                    self._param_registry[pname] = (desc, check, err)
                    self._defaults[pname] = default
        else:
            # Raise if user tried to set ionspec params in non-Cue mode
            ionspec_in_kwargs = [p for p in _CUE_IONSPEC_PARAMS if p in resolved_kwargs]
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

        # --- Validate physical bounds ---
        self._validate_bounds()

    def _init_nebular_config(self, kwargs):
        """Resolve nebular emission backend from kwargs."""
        nebular_ssp = kwargs.pop("nebular_ssp", False)
        nebular = kwargs.pop("nebular", False)
        nebular_cue = kwargs.pop("nebular_cue", False)
        self.cloudy_grid_path = kwargs.pop("cloudy_grid_path", None)
        self.cue_weights_path = kwargs.pop("cue_weights_path", None)
        self.neb_ionization = kwargs.pop("neb_ionization", "ssp")

        # Handle string-style flags ("cue", "cb19") passed as nebular=
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
                self._raise_missing_grid_path()
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
            _NEB_PARAM_NAMES = (
                set(_NEBULAR_PARAMS) | set(_CUE_IONSPEC_PARAMS) | set(_CUE_GAS_EXTRA_PARAMS)
            )
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
        if self.dust_model not in ("two_component", "single_component"):
            raise ValueError(
                f"dust_model must be 'two_component' or 'single_component', "
                f"got '{self.dust_model}'"
            )

        self.dust_approx = kwargs.pop("dust_approx", "fast")
        if self.dust_approx not in ("fast", "exact"):
            raise ValueError(f"dust_approx must be 'fast' or 'exact', got '{self.dust_approx}'")

        # For single-component, accept `dust_law` as cleaner alias for `dust_law_bc`
        dust_law_alias = kwargs.pop("dust_law", None)
        if dust_law_alias is not None:
            self.dust_law_bc = kwargs.pop("dust_law_bc", dust_law_alias)
        else:
            self.dust_law_bc = kwargs.pop("dust_law_bc", "power_law")

        if self.dust_model == "single_component":
            dust_law_diff_explicit = kwargs.pop("dust_law_diff", None)
            if dust_law_diff_explicit is not None:
                import warnings

                warnings.warn(
                    "dust_law_diff is ignored with dust_model='single_component' "
                    "(only one attenuation curve is used).",
                    UserWarning,
                    stacklevel=2,
                )
            self.dust_law_diff = self.dust_law_bc
        else:
            self.dust_law_diff = kwargs.pop("dust_law_diff", self.dust_law_bc)

        self.dust_emission = kwargs.pop("dust_emission", None)
        self.dl07_grid_path = kwargs.pop("dl07_grid_path", None)

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

        # Snapshot the user's keys before any further popping. Inference
        # only sees met_/chem_ keys; everything else is irrelevant noise.
        _inferred_mode = infer_met_mode(set(kwargs.keys()))

        if _met_mode_explicit is not None:
            if _evolving_met or _chem_evol:
                raise ValueError(
                    "Cannot use met_mode with evolving_metallicity or chem_evol. "
                    "Use met_mode='ramp' instead of evolving_metallicity=True, "
                    "or met_mode='chem_evol' instead of chem_evol=True."
                )
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
            self.met_mode = _inferred_mode

        # Backward-compat properties for sed_model / pipeline
        self.evolving_metallicity = self.met_mode == "ramp"
        self.chem_evol = self.met_mode == "chem_evol"

        self.alpha_fe_evolving = kwargs.pop("alpha_fe_evolving", False)
        self.met_interp = kwargs.pop("met_interp", "smooth")
        self.lgmet_scatter = float(kwargs.pop("lgmet_scatter", 0.1))
        # Redshift-table interpolation mode (used when a precomputed z-table
        # is enabled via ``model.precompute_ztable()`` AND redshift is free).
        # "linear" → piecewise-linear (C^0, default).
        # "smooth" → triweight kernel (C^2) — recommended for HMC/NUTS with free z.
        self.z_interp = kwargs.pop("z_interp", "linear")

    @staticmethod
    def _raise_missing_grid_path():
        """Raise ValueError listing available CLOUDY grids."""
        from pathlib import Path

        data_dir = Path(__file__).resolve().parents[1] / "data"
        grids = sorted(data_dir.glob("cloudy_grid_*.h5"))
        grid_list = "\n".join(f"  {g.name}" for g in grids) if grids else "  (none found)"
        raise ValueError(
            f"nebular=True requires cloudy_grid_path. "
            f"Available grids in {data_dir}/:\n{grid_list}\n"
            f"Match the grid isochrone to your SSP for consistency."
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
        """
        keys = jax.random.split(key, len(self._distributions) + 1)
        params = {}
        for i, name in enumerate(sorted(self._distributions.keys())):
            params[name] = self._distributions[name].sample(keys[i])

        if self.stochastic:
            params["sfh_field_xi"] = jax.random.normal(keys[-1], shape=(self._n_grid,))

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

        # Parameter table
        hdr = f"  {'Parameter':<32s} {'Prior':<26s} {'Bounds'}"
        lines.append(hdr)
        lines.append("  " + "─" * 64)

        # Group: free parameters first, then fixed
        for name in self.free_params:
            dist = self._distributions[name]
            lo, hi = dist.bounds
            prior_str = repr(dist)
            bounds_str = f"[{lo:.4g}, {hi:.4g}]"
            lines.append(f"  {name:<32s} {prior_str:<26s} {bounds_str}")

        if self.fixed_params:
            lines.append("  " + "─" * 64)
            for name in self.fixed_params:
                if name in self._mirrors:
                    continue
                dist = self._distributions[name]
                val = dist.bounds[0]
                lines.append(f"  {name:<32s} {'Fixed':<26s} {val:.4g}")

        if self._mirrors:
            lines.append("  " + "─" * 64)
            for target, source in self._mirrors.items():
                mirror_str = f"Mirror({source})"
                lines.append(f"  {target:<32s} {mirror_str:<26s} ──►")

        lines.append(sep)
        print("\n".join(lines))

    def _build_summary_str(self) -> str:
        """Build the same multi-line summary, returned as a string."""
        # Reuse summary() logic by capturing stdout — simpler than refactor
        import io
        import contextlib

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

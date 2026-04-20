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

Legacy tsnorm (backward compatible)::

    spec = Parameters(
        mean_sfh_type = "tsnorm",
        sfh_tsnorm_log_peak_sfr = Uniform(-1, 2),
        sfh_tsnorm_peak_lbt_gyr = Uniform(1, 12),
        sfh_tsnorm_width_gyr = Uniform(0.5, 5),
        sfh_tsnorm_skew = Uniform(-1, 1),
        sfh_tsnorm_trunc = Uniform(1, 10),
        ...
    )

Legacy DPL (backward compatible)::

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

from tengri.parameters._param_defs import (
    _CUE_GAS_EXTRA_PARAMS,
    _CUE_IONSPEC_PARAMS,
    _DUST_EMISSION_PARAMS,
    _LEGACY_PARAM_ALIASES,
    _LEGACY_SFH_TYPE_ALIASES,
    _NEBULAR_PARAMS,
    SETTINGS_KEYS,
    _build_param_registry,
)
from tengri.parameters.priors import (
    Distribution,
    Fixed,
    resolve_shorthand,
)

__all__ = ["SETTINGS_KEYS", "_DUST_EMISSION_PARAMS", "ParamSpec", "Parameters"]


# ── Parameters class ───────────────────────────────────────────────────


class Parameters:
    """Parameter specification defining model parameters and their priors.

    Parameters are specified as keyword arguments.  Each can be:

    - A scalar (int/float) → ``Fixed`` value
    - A tuple (lo, hi)     → ``Uniform`` prior
    - A ``Distribution`` object (``Uniform``, ``Gaussian``, ``LogUniform``,
      ``LogNormal``, ``StudentT``, ``Fixed``)

    Settings (model configuration, not fittable parameters)
    --------------------------------------------------------
    mean_sfh_type : str or list[str]
        SFH model(s).  Composable: ``["dpl", "field"]``.
        Options: dpl, tsnorm, snorm, norm, lnorm, const, exp, dexp, burst, field.
        Default: ``["dpl", "field"]``.
    n_grid : int
        GP grid size (latent dimensions for stochastic SFH).  Default: 64.
    stochastic : bool
        DEPRECATED.  Use ``mean_sfh_type`` with/without ``"field"`` instead.

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
        ``"astrodust"``, ``"bosa"``, ``"themis"``, ``"magphys"``.
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
    agn_frac                   Fixed(0.0)        AGN fraction of stellar L_bol (legacy)
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

        # --- Resolve legacy parameter aliases ---
        resolved_kwargs = {}
        detected_models = set()
        for name, val in kwargs.items():
            new_name = _LEGACY_PARAM_ALIASES.get(name, name)
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
        # with legacy psd_sigma/psd_tau_myr that are Fixed)
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

        # Backward compat: old string-style flags
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
        """Resolve metallicity evolution mode from kwargs."""
        _met_mode_explicit = kwargs.pop("met_mode", None)
        _evolving_met = kwargs.pop("evolving_metallicity", False)
        _chem_evol = kwargs.pop("chem_evol", False)

        if _met_mode_explicit is not None:
            if _evolving_met or _chem_evol:
                raise ValueError(
                    "Cannot use met_mode with evolving_metallicity or chem_evol. "
                    "Use met_mode='ramp' instead of evolving_metallicity=True, "
                    "or met_mode='chem_evol' instead of chem_evol=True."
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
            self.met_mode = "delta"

        # Backward-compat properties for sed_model / pipeline
        self.evolving_metallicity = self.met_mode == "ramp"
        self.chem_evol = self.met_mode == "chem_evol"

        self.alpha_fe_evolving = kwargs.pop("alpha_fe_evolving", False)
        self.met_interp = kwargs.pop("met_interp", "smooth")
        self.lgmet_scatter = float(kwargs.pop("lgmet_scatter", 0.1))

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
            if isinstance(raw_sfh_type, str):
                raw_sfh_type = _LEGACY_SFH_TYPE_ALIASES.get(raw_sfh_type, raw_sfh_type)
                result = [raw_sfh_type]
            else:
                result = [_LEGACY_SFH_TYPE_ALIASES.get(s, s) for s in raw_sfh_type]

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
        """Whether the model includes a GP field (backward-compat property)."""
        return "field" in self._mean_sfh_type

    @property
    def n_grid(self) -> int:
        """GP grid size (only relevant when stochastic=True)."""
        return self._n_grid

    @property
    def mean_sfh_type(self) -> list[str]:
        """SFH model type(s) as a list of strings."""
        return list(self._mean_sfh_type)

    @property
    def all_params(self) -> list[str]:
        """All parameter names (sorted, excludes settings)."""
        return sorted(self._distributions.keys())

    @property
    def free_params(self) -> list[str]:
        """Names of free (non-fixed) parameters."""
        return sorted(k for k, d in self._distributions.items() if not d.is_fixed)

    @property
    def fixed_params(self) -> list[str]:
        """Names of fixed parameters."""
        return sorted(k for k, d in self._distributions.items() if d.is_fixed)

    @property
    def n_free(self) -> int:
        """Number of free parameters (excludes sfh_field_xi)."""
        return len(self.free_params)

    @property
    def valid_param_names(self) -> frozenset:
        """Set of valid parameter names for this model configuration."""
        return self._valid_param_names

    @property
    def mirrors(self) -> dict[str, str]:
        """Parameter mirrors: {target_name: source_name}."""
        return dict(self._mirrors)

    # ── Public API ────────────────────────────────────────────────────

    def with_params(self, **kwargs) -> Parameters:
        """Return a new Parameters with additional parameters merged in.

        Creates a copy of this Parameters with extra parameters added.
        Existing user-defined parameters take precedence — if a param
        name already exists, the new value is silently ignored.

        This is used by Model to auto-merge observation-driven parameters
        (calibration coefficients, noise model params) into the spec.

        Parameters
        ----------
        **kwargs
            Parameter name → Distribution (or scalar/tuple shorthand).
            Only params not already present are added.

        Returns
        -------
        Parameters
            New instance with merged parameters.
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

        For each mirror ``target -> source``, sets ``params[target] =
        params[source]``.  Returns a new dict (immutable pattern).

        Parameters
        ----------
        params : dict
            Parameter name -> value.

        Returns
        -------
        dict
            New dict with mirrored values filled in.
        """
        if not self._mirrors:
            return params
        out = dict(params)
        for target, source in self._mirrors.items():
            out[target] = out[source]
        return out

    def get_distribution(self, name: str) -> Distribution:
        """Get the distribution object for a parameter."""
        if name not in self._distributions:
            raise KeyError(f"Unknown parameter '{name}'")
        return self._distributions[name]

    def get_fixed_values(self) -> dict[str, float]:
        """Get a dict of {name: value} for all numeric fixed parameters.

        String-valued Fixed parameters (categorical config, e.g. shock_abundance)
        are excluded because they cannot be represented as float.
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

        Used by ``Fitter`` to inject emission-line amplitude parameters so they
        flow through bounds, prior penalty loops, and summary output without
        requiring special-casing in downstream code.

        Parameters
        ----------
        **extra_params : Distribution
            Mapping of parameter name → Distribution to add.

        Returns
        -------
        Parameters
            New ``Parameters`` instance with ``extra_params`` included in
            ``free_params``. The original instance is not modified.
        """
        new_spec = copy.copy(self)
        new_spec._distributions = {**self._distributions, **extra_params}
        new_spec._valid_param_names = self._valid_param_names | frozenset(extra_params.keys())
        return new_spec

    def sample(self, key: jax.Array) -> dict[str, jnp.ndarray]:
        """Draw one sample from all parameter distributions.

        Fixed parameters return their fixed value.
        If "field" in mean_sfh_type, also generates sfh_field_xi ~ N(0,I).

        Parameters
        ----------
        key : PRNGKey
            Random key.

        Returns
        -------
        dict
            Parameter name → sampled value.
        """
        keys = jax.random.split(key, len(self._distributions) + 1)
        params = {}
        for i, name in enumerate(sorted(self._distributions.keys())):
            params[name] = self._distributions[name].sample(keys[i])

        if self.stochastic:
            params["sfh_field_xi"] = jax.random.normal(keys[-1], shape=(self._n_grid,))

        return self.resolve_mirrors(params)

    def sample_batch(self, key: jax.Array, n: int) -> dict[str, jnp.ndarray]:
        """Draw n samples from all parameter distributions.

        Parameters
        ----------
        key : PRNGKey
            Random key.
        n : int
            Number of samples.

        Returns
        -------
        dict
            Parameter name → array of shape (n,) or (n, n_grid) for xi.
        """
        keys = jax.random.split(key, n)
        return jax.vmap(self.sample)(keys)

    def validate(self, params: dict[str, jnp.ndarray]) -> None:
        """Check that parameter values are within bounds.

        Parameters
        ----------
        params : dict
            Parameter name → value.

        Raises
        ------
        ValueError
            If any parameter is out of bounds.
        """
        for name, dist in self._distributions.items():
            if name not in params:
                continue
            val = float(params[name])
            lo, hi = dist.bounds
            if not dist.is_fixed and (val < lo or val > hi):
                raise ValueError(f"Parameter '{name}' = {val} is outside bounds [{lo}, {hi}]")

    def summary(self) -> str:
        """Return a human-readable summary of the model configuration.

        Displays SFH type, enabled modules, dimensionality, and a table
        of all parameters grouped by component (free first, then fixed).

        Returns
        -------
        str
            Formatted summary string.
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
        return "\n".join(lines)

    def __repr__(self) -> str:
        lines = [f"Parameters(mean_sfh_type={self._mean_sfh_type},"]
        for name in sorted(self._distributions.keys()):
            dist = self._distributions[name]
            lines.append(f"    {name:30s} = {dist!r},")
        if self.stochastic:
            lines.append(f"    {'n_grid':30s} = {self._n_grid},")
        lines.append(")")
        return "\n".join(lines)


# ── Deprecated alias (removed in v1.0) ─────────────────────────────────


def _make_deprecated_paramspec():
    import warnings

    class ParamSpec(Parameters):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "ParamSpec is deprecated. Use Parameters instead. Will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    ParamSpec.__name__ = "ParamSpec"
    ParamSpec.__qualname__ = "ParamSpec"
    return ParamSpec


ParamSpec = _make_deprecated_paramspec()

"""SFH model registry and composition engine.

Provides a registry of parametric SFH models that can be composed
via ``mean_sfh_type`` lists. Three composition types are supported:

- **additive**: smooth models summed together (tsnorm, snorm, norm, ...).
- **mixture**: mass-fraction mixing with smooth component (burst).
- **modulator**: multiplicative GP field modulation (field).

Usage
-----
Single model::

    fn, params, param_map, settings = resolve_sfh("tsnorm")

Composed models::

    fn, params, param_map, settings = resolve_sfh(["tsnorm", "burst", "field"])

The returned ``fn`` is a pure JAX closure that can be JIT-compiled.

References
----------
- Bellstedt+2020 (arXiv:2005.11917): snorm, tsnorm.
- Carnall+2018: DPL.
- Zacharegkas+2025 (arXiv:2506.19919): triweight burst.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from tengri.distributions import Distribution, Fixed, Uniform
from tengri.models.sfh.gp_sfh import compute_sqrt_power_drw, gp_from_xi
from tengri.models.sfh.mean_sfh import (
    AGEMAX_YR,
    constant_sfh,
    delayed_exponential_sfh,
    dpl,
    exponential_sfh,
    lnorm,
    norm,
    snorm,
    triweight_burst,
    tsnorm,
)
from tengri.models.sfh.nonparametric import continuity_sfh, dirichlet_sfh
from tengri.models.sfh.psd_models import drw_variance

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ParamDef(NamedTuple):
    """Definition of a single fittable SFH parameter.

    Attributes
    ----------
    description : str
        Human-readable description.
    bound_check : callable
        Function(lo, hi) -> bool for physical bound validation.
    bound_error : str
        Error message when bound check fails.
    default : Distribution
        Default prior distribution.
    """

    description: str
    bound_check: object  # Callable[[float, float], bool]
    bound_error: str
    default: Distribution


class SFHModelSpec(NamedTuple):
    """Specification of a registered SFH model.

    Attributes
    ----------
    name : str
        Model name (e.g., "tsnorm", "dpl", "burst", "field").
    fn : callable
        Pure JAX function: fn(t_lookback, **internal_params) -> SFR.
    params : dict[str, ParamDef]
        Fittable parameters: public_name -> ParamDef.
    settings : dict[str, Any]
        Non-fittable settings with defaults (e.g., ngrid for field).
    internal_param_map : dict[str, tuple[str, float, float]]
        public_name -> (internal_name, scale, offset).
        Conversion: internal = public * scale + offset.
    composition_type : str
        "additive", "mixture", or "modulator".
    """

    name: str
    fn: object  # Callable
    params: dict[str, ParamDef]
    settings: dict[str, Any]
    internal_param_map: dict[str, tuple[str, float, float]]
    composition_type: str


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

SFH_REGISTRY: dict[str, SFHModelSpec] = {}

# Field sub-model registry: PSD model name -> sqrt_power function
FIELD_MODEL_REGISTRY: dict[str, object] = {
    "drw": compute_sqrt_power_drw,
}

_always_true = lambda lo, hi: True  # noqa: E731
_lo_positive = lambda lo, hi: lo > 0  # noqa: E731
_lo_nonneg = lambda lo, hi: lo >= 0  # noqa: E731


def _register(spec: SFHModelSpec) -> None:
    """Register an SFH model spec in the global registry."""
    SFH_REGISTRY[spec.name] = spec


# ---------------------------------------------------------------------------
# Register smooth (additive) models
# ---------------------------------------------------------------------------

# --- tsnorm (truncated skew-normal) ---
_register(
    SFHModelSpec(
        name="tsnorm",
        fn=tsnorm,
        params={
            "sfh_tsnorm_log_peak_sfr": ParamDef(
                "log10 peak SFR", _always_true, "", Uniform(-1.0, 3.0)
            ),
            "sfh_tsnorm_peak_lbt_gyr": ParamDef(
                "Peak lookback time (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.5, 12.0)
            ),
            "sfh_tsnorm_width_gyr": ParamDef(
                "Gaussian width (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.2, 5.0)
            ),
            "sfh_tsnorm_skew": ParamDef("Skewness", _always_true, "", Uniform(-1.0, 1.0)),
            "sfh_tsnorm_trunc": ParamDef(
                "Truncation sharpness", _lo_positive, "must have lo > 0", Uniform(1.0, 10.0)
            ),
        },
        settings={},
        internal_param_map={
            "sfh_tsnorm_log_peak_sfr": ("log_peak_sfr", 1.0, 0.0),
            "sfh_tsnorm_peak_lbt_gyr": ("peak_lbt", 1e9, 0.0),
            "sfh_tsnorm_width_gyr": ("width", 1e9, 0.0),
            "sfh_tsnorm_skew": ("skew", 1.0, 0.0),
            "sfh_tsnorm_trunc": ("trunc", 1.0, 0.0),
        },
        composition_type="additive",
    )
)

# --- snorm (skew-normal) ---
_register(
    SFHModelSpec(
        name="snorm",
        fn=snorm,
        params={
            "sfh_snorm_log_peak_sfr": ParamDef(
                "log10 peak SFR", _always_true, "", Uniform(-1.0, 3.0)
            ),
            "sfh_snorm_peak_lbt_gyr": ParamDef(
                "Peak lookback time (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.5, 12.0)
            ),
            "sfh_snorm_width_gyr": ParamDef(
                "Gaussian width (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.2, 5.0)
            ),
            "sfh_snorm_skew": ParamDef("Skewness", _always_true, "", Uniform(-1.0, 1.0)),
        },
        settings={},
        internal_param_map={
            "sfh_snorm_log_peak_sfr": ("log_peak_sfr", 1.0, 0.0),
            "sfh_snorm_peak_lbt_gyr": ("peak_lbt", 1e9, 0.0),
            "sfh_snorm_width_gyr": ("width", 1e9, 0.0),
            "sfh_snorm_skew": ("skew", 1.0, 0.0),
        },
        composition_type="additive",
    )
)

# --- norm (Gaussian) ---
_register(
    SFHModelSpec(
        name="norm",
        fn=norm,
        params={
            "sfh_norm_log_peak_sfr": ParamDef(
                "log10 peak SFR", _always_true, "", Uniform(-1.0, 3.0)
            ),
            "sfh_norm_peak_lbt_gyr": ParamDef(
                "Peak lookback time (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.5, 12.0)
            ),
            "sfh_norm_width_gyr": ParamDef(
                "Gaussian width (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.2, 5.0)
            ),
        },
        settings={},
        internal_param_map={
            "sfh_norm_log_peak_sfr": ("log_peak_sfr", 1.0, 0.0),
            "sfh_norm_peak_lbt_gyr": ("peak_lbt", 1e9, 0.0),
            "sfh_norm_width_gyr": ("width", 1e9, 0.0),
        },
        composition_type="additive",
    )
)

# --- lnorm (log-normal) ---
_register(
    SFHModelSpec(
        name="lnorm",
        fn=lnorm,
        params={
            "sfh_lnorm_log_peak_sfr": ParamDef(
                "log10 peak SFR", _always_true, "", Uniform(-1.0, 3.0)
            ),
            "sfh_lnorm_peak_lbt_gyr": ParamDef(
                "Peak lookback time (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.5, 12.0)
            ),
            "sfh_lnorm_width_gyr": ParamDef(
                "Log-space width (dex)", _lo_positive, "must have lo > 0", Uniform(0.1, 2.0)
            ),
        },
        settings={},
        internal_param_map={
            "sfh_lnorm_log_peak_sfr": ("log_peak_sfr", 1.0, 0.0),
            "sfh_lnorm_peak_lbt_gyr": ("peak_lbt", 1e9, 0.0),
            "sfh_lnorm_width_gyr": ("width", 1.0, 0.0),  # already in dex
        },
        composition_type="additive",
    )
)

# --- dpl (double power law) ---
_register(
    SFHModelSpec(
        name="dpl",
        fn=dpl,
        params={
            "sfh_dpl_alpha": ParamDef(
                "DPL falling slope", _lo_positive, "must have lo > 0", Uniform(0.1, 5.0)
            ),
            "sfh_dpl_beta": ParamDef(
                "DPL rising slope", _lo_positive, "must have lo > 0", Uniform(0.1, 3.0)
            ),
            "sfh_dpl_tau_gyr": ParamDef(
                "DPL turnover time (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.1, 12.0)
            ),
            "sfh_dpl_log_peak_sfr": ParamDef(
                "log10 peak SFR", _always_true, "", Uniform(-1.0, 3.0)
            ),
        },
        settings={},
        internal_param_map={
            "sfh_dpl_alpha": ("alpha", 1.0, 0.0),
            "sfh_dpl_beta": ("beta", 1.0, 0.0),
            "sfh_dpl_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_dpl_log_peak_sfr": ("log_peak_sfr", 1.0, 0.0),
        },
        composition_type="additive",
    )
)

# --- const (constant) ---
_register(
    SFHModelSpec(
        name="const",
        fn=constant_sfh,
        params={
            "sfh_const_log_sfr": ParamDef("log10 SFR", _always_true, "", Uniform(-1.0, 3.0)),
            "sfh_const_start_gyr": ParamDef(
                "Start lookback (Gyr)", _lo_nonneg, "must have lo >= 0", Fixed(0.0)
            ),
            "sfh_const_end_gyr": ParamDef(
                "End lookback (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Fixed(AGEMAX_YR / 1e9),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_const_log_sfr": ("log_sfr", 1.0, 0.0),
            "sfh_const_start_gyr": ("start", 1e9, 0.0),
            "sfh_const_end_gyr": ("end", 1e9, 0.0),
        },
        composition_type="additive",
    )
)

# --- exp (exponential) ---
_register(
    SFHModelSpec(
        name="exp",
        fn=exponential_sfh,
        params={
            "sfh_exp_log_peak_sfr": ParamDef(
                "log10 peak SFR", _always_true, "", Uniform(-1.0, 3.0)
            ),
            "sfh_exp_tau_gyr": ParamDef(
                "e-folding timescale (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.1, 10.0)
            ),
            "sfh_exp_start_gyr": ParamDef(
                "Start lookback (Gyr)", _lo_nonneg, "must have lo >= 0", Fixed(0.0)
            ),
        },
        settings={},
        internal_param_map={
            "sfh_exp_log_peak_sfr": ("log_peak_sfr", 1.0, 0.0),
            "sfh_exp_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_exp_start_gyr": ("start", 1e9, 0.0),
        },
        composition_type="additive",
    )
)

# --- dexp (delayed exponential) ---
_register(
    SFHModelSpec(
        name="dexp",
        fn=delayed_exponential_sfh,
        params={
            "sfh_dexp_log_peak_sfr": ParamDef(
                "log10 peak SFR", _always_true, "", Uniform(-1.0, 3.0)
            ),
            "sfh_dexp_tau_gyr": ParamDef(
                "Timescale (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.1, 10.0)
            ),
            "sfh_dexp_start_gyr": ParamDef(
                "Start lookback (Gyr)", _lo_nonneg, "must have lo >= 0", Fixed(0.0)
            ),
        },
        settings={},
        internal_param_map={
            "sfh_dexp_log_peak_sfr": ("log_peak_sfr", 1.0, 0.0),
            "sfh_dexp_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_dexp_start_gyr": ("start", 1e9, 0.0),
        },
        composition_type="additive",
    )
)


# ---------------------------------------------------------------------------
# Register tabulated SFH model (for simulations)
# ---------------------------------------------------------------------------


def _table_sfh_placeholder(t_lookback, **kwargs):
    """Placeholder — tabulated SFH is handled directly in Model._compute_sed_components."""
    return jnp.zeros_like(t_lookback)


_register(
    SFHModelSpec(
        name="table",
        fn=_table_sfh_placeholder,
        params={},  # no fittable params — the table IS the SFH
        settings={},
        internal_param_map={},
        composition_type="additive",
    )
)


# ---------------------------------------------------------------------------
# Register non-parametric SFH models (Leja+2017, Leja+2019)
# ---------------------------------------------------------------------------

# --- continuity (Leja+2019): piecewise-constant with Student-t smoothness prior ---
_register(
    SFHModelSpec(
        name="continuity",
        fn=continuity_sfh,
        params={
            "sfh_cont_log_total_mass": ParamDef(
                "log10 total stellar mass formed (Msun)",
                _always_true,
                "",
                Uniform(8.0, 12.0),
            ),
            **{
                f"sfh_cont_ratio_{i}": ParamDef(
                    f"log10 SFR ratio bin {i}/{i + 1}",
                    _always_true,
                    "",
                    Uniform(-1.0, 1.0),
                )
                for i in range(6)  # 7 bins -> 6 ratios
            },
        },
        settings={},
        internal_param_map={
            "sfh_cont_log_total_mass": ("log_total_mass", 1.0, 0.0),
            **{
                f"sfh_cont_ratio_{i}": (f"ratio_{i}", 1.0, 0.0)
                for i in range(6)
            },
        },
        composition_type="additive",
    )
)

# --- dirichlet (Leja+2017): piecewise-constant with Dirichlet mass fraction prior ---
_register(
    SFHModelSpec(
        name="dirichlet",
        fn=dirichlet_sfh,
        params={
            "sfh_dir_log_total_mass": ParamDef(
                "log10 total stellar mass formed (Msun)",
                _always_true,
                "",
                Uniform(8.0, 12.0),
            ),
            **{
                f"sfh_dir_z_{i}": ParamDef(
                    f"Dirichlet stick-breaking variable {i}",
                    lambda lo, hi: lo >= 0 and hi <= 1,
                    "must be in [0, 1]",
                    Uniform(0.01, 0.99),
                )
                for i in range(6)  # 7 bins -> 6 auxiliary variables
            },
        },
        settings={},
        internal_param_map={
            "sfh_dir_log_total_mass": ("log_total_mass", 1.0, 0.0),
            **{
                f"sfh_dir_z_{i}": (f"z_frac_{i}", 1.0, 0.0)
                for i in range(6)
            },
        },
        composition_type="additive",
    )
)


# ---------------------------------------------------------------------------
# Register burst (mixture) model
# ---------------------------------------------------------------------------

_register(
    SFHModelSpec(
        name="burst",
        fn=triweight_burst,
        params={
            "sfh_burst_log_fburst": ParamDef(
                "log10 burst mass fraction",
                lambda lo, hi: hi < 0,
                "must have hi < 0 (fraction < 1)",
                Uniform(-3.0, -0.1),
            ),
            "sfh_burst_log_tpeak_myr": ParamDef(
                "log10 burst peak time (Myr)", _always_true, "", Uniform(0.0, 3.0)
            ),
            "sfh_burst_log_tmax_myr": ParamDef(
                "log10 burst duration (Myr)", _always_true, "", Uniform(1.0, 4.0)
            ),
        },
        settings={},
        internal_param_map={
            "sfh_burst_log_fburst": ("log_fburst", 1.0, 0.0),
            "sfh_burst_log_tpeak_myr": ("log_tpeak_myr", 1.0, 0.0),
            "sfh_burst_log_tmax_myr": ("log_tmax_myr", 1.0, 0.0),
        },
        composition_type="mixture",
    )
)


# ---------------------------------------------------------------------------
# Register field (modulator) model
# ---------------------------------------------------------------------------


def _field_fn_placeholder(t_lookback, **kwargs):
    """Placeholder — field modulation is applied in the composed closure."""
    raise RuntimeError("field fn should not be called directly; use resolve_sfh()")


_register(
    SFHModelSpec(
        name="field",
        fn=_field_fn_placeholder,
        params={
            "sfh_field_psd_sigma": ParamDef(
                "PSD amplitude (dex)", _lo_nonneg, "must have lo >= 0", Uniform(0.01, 1.0)
            ),
            "sfh_field_psd_tau_myr": ParamDef(
                "PSD timescale (Myr)", _lo_positive, "must have lo > 0", Uniform(10.0, 500.0)
            ),
        },
        settings={
            "sfh_field_ngrid": 256,
            "sfh_field_model": "drw",
        },
        internal_param_map={
            "sfh_field_psd_sigma": ("psd_sigma", 1.0, 0.0),
            "sfh_field_psd_tau_myr": ("psd_tau_yr", 1e6, 0.0),
        },
        composition_type="modulator",
    )
)


# ---------------------------------------------------------------------------
# Composition: resolve_sfh()
# ---------------------------------------------------------------------------


def resolve_sfh(
    mean_sfh_type: str | list[str],
) -> tuple[object, dict[str, ParamDef], dict[str, tuple[str, float, float]], dict[str, Any]]:
    """Resolve SFH specification to a composed function + params.

    Parameters
    ----------
    mean_sfh_type : str or list[str]
        Model name(s). E.g., ``"tsnorm"`` or ``["tsnorm", "burst", "field"]``.

    Returns
    -------
    composed_fn : callable
        Pure JAX function: fn(t_lookback, **all_internal_kwargs) -> SFR.
    merged_params : dict[str, ParamDef]
        All fittable parameters across selected models.
    merged_param_map : dict[str, tuple[str, float, float]]
        Public name -> (internal, scale, offset) for all params.
    merged_settings : dict[str, Any]
        Non-fittable settings (e.g., sfh_field_ngrid).

    Raises
    ------
    KeyError
        If a model name is not in the registry.
    ValueError
        If composition constraints are violated.
    """
    if isinstance(mean_sfh_type, str):
        mean_sfh_type = [mean_sfh_type]

    # Look up models
    specs = []
    for name in mean_sfh_type:
        if name not in SFH_REGISTRY:
            valid = sorted(SFH_REGISTRY.keys())
            raise KeyError(f"Unknown SFH model '{name}'. Valid models: {valid}")
        specs.append(SFH_REGISTRY[name])

    additive = [s for s in specs if s.composition_type == "additive"]
    mixtures = [s for s in specs if s.composition_type == "mixture"]
    modulators = [s for s in specs if s.composition_type == "modulator"]

    if len(mixtures) > 1:
        raise ValueError("At most one mixture component (burst) allowed")
    if len(modulators) > 1:
        raise ValueError("At most one modulator component (field) allowed")
    if len(additive) == 0:
        raise ValueError("At least one additive (smooth) SFH component required")

    # Merge params, param_map, settings — check for collisions
    merged_params: dict[str, ParamDef] = {}
    merged_param_map: dict[str, tuple[str, float, float]] = {}
    merged_settings: dict[str, Any] = {}

    for s in specs:
        for pname, pdef in s.params.items():
            if pname in merged_params:
                raise ValueError(f"Parameter name collision: '{pname}' appears in multiple models")
            merged_params[pname] = pdef
        merged_param_map.update(s.internal_param_map)
        merged_settings.update(s.settings)

    # Build lists of (fn, set_of_internal_names) for each additive component
    additive_info = []
    for s in additive:
        internal_names = {v[0] for v in s.internal_param_map.values()}
        additive_info.append((s.fn, internal_names))

    has_burst = len(mixtures) > 0
    burst_info = None
    if has_burst:
        bs = mixtures[0]
        burst_internal = {v[0] for v in bs.internal_param_map.values()}
        burst_info = (bs.fn, burst_internal)

    has_field = len(modulators) > 0

    # Build the composed closure
    def composed_fn(t_lookback, **kw):
        # 1. Sum additive components
        smooth = jnp.zeros_like(t_lookback)
        for fn_i, int_names_i in additive_info:
            kw_i = {k: kw[k] for k in int_names_i if k in kw}
            smooth = smooth + fn_i(t_lookback, **kw_i)

        # 2. Apply burst mixture
        if has_burst:
            burst_fn, burst_int = burst_info
            log_fburst = kw["log_fburst"]
            f = 10.0**log_fburst
            burst_kw = {k: kw[k] for k in burst_int if k != "log_fburst" and k in kw}
            burst_shape = burst_fn(t_lookback, **burst_kw)
            # Normalize burst shape to match smooth integral scale
            smooth = (1.0 - f) * smooth + f * burst_shape * jnp.max(smooth)

        # 3. Apply field modulation
        if has_field and "gp_x" in kw and "k0_half" in kw:
            smooth = smooth * jnp.exp(kw["gp_x"] - kw["k0_half"])

        return smooth

    return composed_fn, merged_params, merged_param_map, merged_settings


def compute_field_gp(
    xi: jnp.ndarray,
    psd_sigma: float,
    psd_tau_yr: float,
    n_grid: int,
    d_log_age: float,
    field_model: str = "drw",
) -> tuple[jnp.ndarray, float]:
    """Compute GP realization and lognormal correction for the field component.

    Parameters
    ----------
    xi : array, shape (n_grid,)
        Latent vector (xi ~ N(0, I)).
    psd_sigma : float
        PSD amplitude (dex).
    psd_tau_yr : float
        PSD timescale (yr).
    n_grid : int
        Grid size.
    d_log_age : float
        Grid spacing in dex.
    field_model : str
        PSD model name. Default "drw".

    Returns
    -------
    gp_x : array, shape (n_grid,)
        GP realization on the log-age grid.
    k0_half : float
        Lognormal correction: K(0)/2 = sigma_PS^2 / 4.
    """
    sqrt_power_fn = FIELD_MODEL_REGISTRY[field_model]
    sqrt_power = sqrt_power_fn(n_grid, d_log_age, psd_sigma, psd_tau_yr)
    gp_x = gp_from_xi(xi, sqrt_power, n_grid)
    k0_half = drw_variance(psd_sigma) / 2.0
    return gp_x, k0_half

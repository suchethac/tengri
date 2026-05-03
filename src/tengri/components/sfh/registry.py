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
- Robotham+2020 (arXiv:2002.06980): snorm_burst, tsnorm_burst (ProSpect).
- Carnall+2018: DPL.
- Zacharegkas+2025 (arXiv:2506.19919): triweight burst.

"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax.numpy as jnp

from tengri.components.sfh.dense_basis import dense_basis, dense_basis_pure
from tengri.components.sfh.gp_sfh import compute_sqrt_power_drw, gp_from_xi
from tengri.components.sfh.mean_sfh import (
    AGEMAX_YR,
    buat08,
    constant,
    constant_then_exponential_sfh,
    declining_exponential_sfh,
    delayed_bq,
    delayed_exponential,
    dpl,
    exponential,
    lnorm,
    norm,
    periodic,
    psb_wild2020,
    snorm,
    snorm_burst,
    snorm_trunc_burst,
    triweight_burst,
    tsnorm,
)
from tengri.components.sfh.nonparametric import (
    CFLEX_DEFAULT_ANCHOR_GYR,
    continuity,
    continuity_flex,
    dirichlet,
)
from tengri.components.sfh.psd_models import drw_variance
from tengri.parameters.priors import Distribution, Fixed, Uniform

# ── Data structures ───────────────────────────────────────────────


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

    Notes
    -----
    **JIT-compatible**: no — Python dataclass for registry initialization.

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

    Notes
    -----
    **JIT-compatible**: no — Python dataclass for registry initialization.

    """

    name: str
    fn: object  # Callable
    params: dict[str, ParamDef]
    settings: dict[str, Any]
    internal_param_map: dict[str, tuple[str, float, float]]
    composition_type: str


# ── Registries ────────────────────────────────────────────────────

SFH_REGISTRY: dict[str, SFHModelSpec] = {}

# Field sub-model registry: PSD model name -> sqrt_power function
FIELD_MODEL_REGISTRY: dict[str, object] = {
    "drw": compute_sqrt_power_drw,
}

_always_true = lambda lo, hi: True  # noqa: E731
_lo_positive = lambda lo, hi: lo > 0  # noqa: E731
_lo_nonneg = lambda lo, hi: lo >= 0  # noqa: E731


def _register(spec: SFHModelSpec) -> None:
    """Register an SFH model spec in the global registry.

    Parameters
    ----------
    spec : SFHModelSpec
        Model specification to register.

    Returns
    -------
    None

    Notes
    -----
    **JIT-compatible**: no — mutates global registry dictionary during initialization.

    """
    SFH_REGISTRY[spec.name] = spec


# ── Register smooth (additive) models ─────────────────────────────

# --- tsnorm (truncated skew-normal) — canonical: truncated_skewnormal_sfh ---
_tsnorm_spec = SFHModelSpec(
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
_register(_tsnorm_spec)
# Register canonical name (same spec object, just different key)
SFH_REGISTRY["truncated_skewnormal_sfh"] = _tsnorm_spec

# --- snorm (skew-normal) — canonical: skewnormal_sfh ---
_snorm_spec = SFHModelSpec(
    name="snorm",
    fn=snorm,
    params={
        "sfh_snorm_log_peak_sfr": ParamDef("log10 peak SFR", _always_true, "", Uniform(-1.0, 3.0)),
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
_register(_snorm_spec)
# Register canonical name (same spec object, just different key)
SFH_REGISTRY["skewnormal_sfh"] = _snorm_spec

# --- snorm_burst (skew-normal + flat burst) — canonical: snorm_burst_sfh ---
_snorm_burst_spec = SFHModelSpec(
    name="snorm_burst",
    fn=snorm_burst,
    params={
        "sfh_snorm_burst_log_peak_sfr": ParamDef(
            "log10 peak SFR of skew-normal component", _always_true, "", Uniform(-1.0, 3.0)
        ),
        "sfh_snorm_burst_peak_lbt_gyr": ParamDef(
            "Peak lookback time (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.5, 12.0)
        ),
        "sfh_snorm_burst_width_gyr": ParamDef(
            "Gaussian width (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.2, 5.0)
        ),
        "sfh_snorm_burst_skew": ParamDef("Skewness", _always_true, "", Uniform(-1.0, 1.0)),
        "sfh_snorm_burst_burst_sfr": ParamDef(
            "Constant burst SFR amplitude (Msun/yr)", _lo_nonneg, "must have lo >= 0", Fixed(0.0)
        ),
        "sfh_snorm_burst_burst_age_gyr": ParamDef(
            "Burst lookback duration (Gyr)", _lo_positive, "must have lo > 0", Fixed(0.1)
        ),
    },
    settings={},
    internal_param_map={
        "sfh_snorm_burst_log_peak_sfr": ("log_peak_sfr", 1.0, 0.0),
        "sfh_snorm_burst_peak_lbt_gyr": ("peak_lbt", 1e9, 0.0),
        "sfh_snorm_burst_width_gyr": ("width", 1e9, 0.0),
        "sfh_snorm_burst_skew": ("skew", 1.0, 0.0),
        "sfh_snorm_burst_burst_sfr": ("burst_sfr", 1.0, 0.0),
        "sfh_snorm_burst_burst_age_gyr": ("burst_age", 1e9, 0.0),
    },
    composition_type="additive",
)
_register(_snorm_burst_spec)
SFH_REGISTRY["snorm_burst_sfh"] = _snorm_burst_spec

# --- tsnorm_burst (truncated skew-normal + flat burst) — canonical: snorm_trunc_burst_sfh ---
_tsnorm_burst_spec = SFHModelSpec(
    name="tsnorm_burst",
    fn=snorm_trunc_burst,
    params={
        "sfh_tsnorm_burst_log_peak_sfr": ParamDef(
            "log10 peak SFR of tsnorm component", _always_true, "", Uniform(-1.0, 3.0)
        ),
        "sfh_tsnorm_burst_peak_lbt_gyr": ParamDef(
            "Peak lookback time (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.5, 12.0)
        ),
        "sfh_tsnorm_burst_width_gyr": ParamDef(
            "Gaussian width (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.2, 5.0)
        ),
        "sfh_tsnorm_burst_skew": ParamDef("Skewness", _always_true, "", Uniform(-1.0, 1.0)),
        "sfh_tsnorm_burst_trunc": ParamDef(
            "Truncation sharpness", _lo_positive, "must have lo > 0", Uniform(1.0, 10.0)
        ),
        "sfh_tsnorm_burst_burst_sfr": ParamDef(
            "Constant burst SFR amplitude (Msun/yr)", _lo_nonneg, "must have lo >= 0", Fixed(0.0)
        ),
        "sfh_tsnorm_burst_burst_age_gyr": ParamDef(
            "Burst lookback duration (Gyr)", _lo_positive, "must have lo > 0", Fixed(0.1)
        ),
    },
    settings={},
    internal_param_map={
        "sfh_tsnorm_burst_log_peak_sfr": ("log_peak_sfr", 1.0, 0.0),
        "sfh_tsnorm_burst_peak_lbt_gyr": ("peak_lbt", 1e9, 0.0),
        "sfh_tsnorm_burst_width_gyr": ("width", 1e9, 0.0),
        "sfh_tsnorm_burst_skew": ("skew", 1.0, 0.0),
        "sfh_tsnorm_burst_trunc": ("trunc", 1.0, 0.0),
        "sfh_tsnorm_burst_burst_sfr": ("burst_sfr", 1.0, 0.0),
        "sfh_tsnorm_burst_burst_age_gyr": ("burst_age", 1e9, 0.0),
    },
    composition_type="additive",
)
_register(_tsnorm_burst_spec)
SFH_REGISTRY["snorm_trunc_burst_sfh"] = _tsnorm_burst_spec

# --- norm (Gaussian) — canonical: gaussian_sfh ---
_norm_spec = SFHModelSpec(
    name="norm",
    fn=norm,
    params={
        "sfh_norm_log_peak_sfr": ParamDef("log10 peak SFR", _always_true, "", Uniform(-1.0, 3.0)),
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
_register(_norm_spec)
# Register canonical name (same spec object, just different key)
SFH_REGISTRY["gaussian_sfh"] = _norm_spec

# --- lnorm (log-normal) — canonical: lognormal_sfh ---
_lnorm_spec = SFHModelSpec(
    name="lnorm",
    fn=lnorm,
    params={
        "sfh_lnorm_log_peak_sfr": ParamDef("log10 peak SFR", _always_true, "", Uniform(-1.0, 3.0)),
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
_register(_lnorm_spec)
# Register canonical name (same spec object, just different key)
SFH_REGISTRY["lognormal_sfh"] = _lnorm_spec

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
        fn=constant,
        params={
            "sfh_const_log_sfr": ParamDef("log10 SFR", _always_true, "", Uniform(-1.0, 3.0)),
            "sfh_const_start_gyr": ParamDef(
                "Lookback to SF onset (Gyr): when did SF start?",
                _lo_positive,
                "must have lo > 0",
                Fixed(AGEMAX_YR / 1e9),
            ),
            "sfh_const_end_gyr": ParamDef(
                "Lookback to SF cessation (Gyr): when did SF stop? (0 = ongoing)",
                _lo_nonneg,
                "must have lo >= 0",
                Fixed(0.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_const_log_sfr": ("log_sfr", 1.0, 0.0),
            # User's "start" (when SF began) = older lookback = internal "end"
            # User's "end" (when SF stopped) = younger lookback = internal "start"
            "sfh_const_start_gyr": ("end", 1e9, 0.0),
            "sfh_const_end_gyr": ("start", 1e9, 0.0),
        },
        composition_type="additive",
    )
)

# --- exp (exponential) ---
_register(
    SFHModelSpec(
        name="exp",
        fn=exponential,
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
        fn=delayed_exponential,
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

# --- tau (declining exponential, matches FSPS sfh=1 / bagpipes 'exponential') ---
# SFR(t_lb) = peak * exp(-(age - t_lb)/tau): highest at galaxy formation (t_lb=age),
# declining to present (t_lb=0).  See declining_exponential_sfh for full derivation.
_register(
    SFHModelSpec(
        name="tau",
        fn=declining_exponential_sfh,
        params={
            "sfh_tau_log_peak_sfr": ParamDef(
                "log10 peak SFR at formation (Msun/yr)", _always_true, "", Uniform(-1.0, 3.0)
            ),
            "sfh_tau_tau_gyr": ParamDef(
                "e-folding timescale (Gyr)", _lo_positive, "must have lo > 0", Uniform(0.1, 10.0)
            ),
            "sfh_tau_age_gyr": ParamDef(
                "Galaxy age / lookback time of formation (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 13.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_tau_log_peak_sfr": ("log_peak_sfr", 1.0, 0.0),
            "sfh_tau_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_tau_age_gyr": ("age", 1e9, 0.0),
        },
        composition_type="additive",
    )
)

# --- const_exp (constant + exponential decline — "quenching at time T") ---
_register(
    SFHModelSpec(
        name="const_exp",
        fn=constant_then_exponential_sfh,
        params={
            "sfh_cexp_log_sfr": ParamDef(
                "log10 constant SFR before quenching (Msun/yr)",
                _always_true,
                "",
                Uniform(-1.0, 3.0),
            ),
            "sfh_cexp_tau_gyr": ParamDef(
                "Post-quench e-folding timescale (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 10.0),
            ),
            "sfh_cexp_quench_gyr": ParamDef(
                "Lookback time when quenching began (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 10.0),
            ),
            "sfh_cexp_age_gyr": ParamDef(
                "Galaxy age / lookback to formation (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 13.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_cexp_log_sfr": ("log_sfr", 1.0, 0.0),
            "sfh_cexp_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_cexp_quench_gyr": ("quench_age", 1e9, 0.0),
            "sfh_cexp_age_gyr": ("age", 1e9, 0.0),
        },
        composition_type="additive",
    )
)
SFH_REGISTRY["constant_then_exponential"] = SFH_REGISTRY["const_exp"]


# --- delayed_bq (delayed-tau with burst/quench, Ciesla+2017) ---
_register(
    SFHModelSpec(
        name="delayed_bq",
        fn=delayed_bq,
        params={
            "sfh_delayed_bq_tau_main_gyr": ParamDef(
                "e-folding timescale of main component (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 10.0),
            ),
            "sfh_delayed_bq_age_main_gyr": ParamDef(
                "Galaxy age / lookback to formation (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 13.0),
            ),
            "sfh_delayed_bq_age_bq_gyr": ParamDef(
                "Lookback time of burst/quench onset (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 5.0),
            ),
            "sfh_delayed_bq_r_sfr": ParamDef(
                "SFR ratio after/before burst/quench",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 10.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_delayed_bq_tau_main_gyr": ("tau_main_yr", 1e9, 0.0),
            "sfh_delayed_bq_age_main_gyr": ("age_main_yr", 1e9, 0.0),
            "sfh_delayed_bq_age_bq_gyr": ("age_bq_yr", 1e9, 0.0),
            "sfh_delayed_bq_r_sfr": ("r_sfr", 1.0, 0.0),
        },
        composition_type="additive",
    )
)


# --- periodic (periodic SF events, Ciesla+2017) ---
_register(
    SFHModelSpec(
        name="periodic",
        fn=periodic,
        params={
            "sfh_periodic_delta_bursts_gyr": ParamDef(
                "Spacing between burst onsets (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 1.0),
            ),
            "sfh_periodic_tau_bursts_gyr": ParamDef(
                "Duration/e-folding timescale of each burst (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.001, 0.5),
            ),
            "sfh_periodic_burst_type": ParamDef(
                "Burst type: 0=exponential, 1=delayed, 2=rectangular",
                lambda lo, hi: lo >= 0 and hi <= 2 and int(lo) == lo,
                "must be 0, 1, or 2",
                Fixed(0),
            ),
            "sfh_periodic_age_gyr": ParamDef(
                "Galaxy age / lookback to formation (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 13.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_periodic_delta_bursts_gyr": ("delta_bursts_yr", 1e9, 0.0),
            "sfh_periodic_tau_bursts_gyr": ("tau_bursts_yr", 1e9, 0.0),
            "sfh_periodic_burst_type": ("burst_type", 1.0, 0.0),
            "sfh_periodic_age_gyr": ("age_yr", 1e9, 0.0),
        },
        composition_type="additive",
    )
)


# --- buat08 (velocity-parameterized SFH, Buat+2008) ---
_register(
    SFHModelSpec(
        name="buat08",
        fn=buat08,
        params={
            "sfh_buat08_velocity_km_s": ParamDef(
                "Rotational velocity (km/s), range [40, 360]",
                lambda lo, hi: lo >= 40 and hi <= 360,
                "must have 40 <= lo and hi <= 360",
                Uniform(80.0, 360.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_buat08_velocity_km_s": ("velocity_km_s", 1.0, 0.0),
        },
        composition_type="additive",
    )
)


# --- psb (post-starburst, Wild+2020) ---
_register(
    SFHModelSpec(
        name="psb",
        fn=psb_wild2020,
        params={
            "sfh_psb_log_peak_sfr": ParamDef(
                "log10 overall SFR normalization (Msun/yr)",
                _always_true,
                "",
                Uniform(-1.0, 3.0),
            ),
            "sfh_psb_age_gyr": ParamDef(
                "Galaxy age / lookback to formation (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 13.0),
            ),
            "sfh_psb_tau_gyr": ParamDef(
                "Old-component e-folding timescale (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 10.0),
            ),
            "sfh_psb_burstage_gyr": ParamDef(
                "Lookback time of burst onset (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 5.0),
            ),
            "sfh_psb_alpha": ParamDef(
                "DPL burst falling slope", _lo_positive, "must have lo > 0", Uniform(0.5, 5.0)
            ),
            "sfh_psb_beta": ParamDef(
                "DPL burst rising slope", _lo_positive, "must have lo > 0", Uniform(0.5, 5.0)
            ),
            "sfh_psb_fburst": ParamDef(
                "Burst mass fraction",
                lambda lo, hi: lo >= 0 and hi <= 1,
                "must be in [0, 1]",
                Uniform(0.01, 0.99),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_psb_log_peak_sfr": ("log_peak_sfr", 1.0, 0.0),
            "sfh_psb_age_gyr": ("age", 1e9, 0.0),
            "sfh_psb_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_psb_burstage_gyr": ("burstage", 1e9, 0.0),
            "sfh_psb_alpha": ("alpha", 1.0, 0.0),
            "sfh_psb_beta": ("beta", 1.0, 0.0),
            "sfh_psb_fburst": ("fburst", 1.0, 0.0),
        },
        composition_type="additive",
    )
)
SFH_REGISTRY["psb_wild2020"] = SFH_REGISTRY["psb"]


# ── Register tabulated SFH model (for simulations) ────────────────


def _table_sfh_placeholder(t_lookback, **kwargs):
    """Placeholder — tabulated SFH is handled directly in Model._compute_sed_components.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    **kwargs
        Unused; for registry compatibility.

    Returns
    -------
    ndarray, shape (n_age,)
        Zero array (actual tabulated SFH handled separately).

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp.zeros_like``.

    """
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


# ── Register non-parametric SFH models (Leja+2017, Leja+2019) ─────

# --- continuity (Leja+2019): piecewise-constant with Student-t smoothness prior ---
_register(
    SFHModelSpec(
        name="continuity",
        fn=continuity,
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
            **{f"sfh_cont_ratio_{i}": (f"ratio_{i}", 1.0, 0.0) for i in range(6)},
        },
        composition_type="additive",
    )
)

# --- continuity_flex (Leja+2019): piecewise-constant with flexible bin edges ---
# Default: 3 flex ratios (flex_0..flex_2) + ratio_young + ratio_old = 5 free params.
# n_flex ratios → n_flex+1 internal flex time bins.
_N_CFLEX = 3
_register(
    SFHModelSpec(
        name="continuity_flex",
        fn=continuity_flex,
        params={
            "sfh_cflex_log_total_mass": ParamDef(
                "log10 total stellar mass formed (Msun)",
                _always_true,
                "",
                Uniform(8.0, 12.0),
            ),
            "sfh_cflex_ratio_young": ParamDef(
                "log10(SFR_young / SFR_flex[0])",
                _always_true,
                "",
                Uniform(-1.0, 1.0),
            ),
            **{
                f"sfh_cflex_flex_{i}": ParamDef(
                    f"log10 flex bin SFR ratio {i} (controls bin width)",
                    _always_true,
                    "",
                    Uniform(-1.0, 1.0),
                )
                for i in range(_N_CFLEX)
            },
            "sfh_cflex_ratio_old": ParamDef(
                "log10(SFR_old / SFR_flex[N])",
                _always_true,
                "",
                Uniform(-1.0, 1.0),
            ),
        },
        settings={
            "sfh_cflex_n_flex": _N_CFLEX,
            "sfh_cflex_anchor_gyr": CFLEX_DEFAULT_ANCHOR_GYR.tolist(),
        },
        internal_param_map={
            "sfh_cflex_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_cflex_ratio_young": ("ratio_young", 1.0, 0.0),
            **{f"sfh_cflex_flex_{i}": (f"flex_{i}", 1.0, 0.0) for i in range(_N_CFLEX)},
            "sfh_cflex_ratio_old": ("ratio_old", 1.0, 0.0),
        },
        composition_type="additive",
    )
)

# --- dirichlet (Leja+2017): piecewise-constant with Dirichlet mass fraction prior ---
_register(
    SFHModelSpec(
        name="dirichlet",
        fn=dirichlet,
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
            **{f"sfh_dir_z_{i}": (f"z_frac_{i}", 1.0, 0.0) for i in range(6)},
        },
        composition_type="additive",
    )
)


# --- dense_basis (Iyer+2017, 2019): GP-SFH via mass-time quantiles ---
_register(
    SFHModelSpec(
        name="dense_basis",
        fn=dense_basis,
        params={
            "sfh_db_log_total_mass": ParamDef(
                "log10 total stellar mass formed (Msun)",
                _always_true,
                "",
                Uniform(8.0, 12.0),
            ),
            "sfh_db_log_sfr_inst": ParamDef(
                "log10 instantaneous SFR at observation (Msun/yr)",
                _always_true,
                "",
                Uniform(-2.0, 3.0),
            ),
            **{
                f"sfh_db_tx_frac_{i}": ParamDef(
                    f"Cosmic time fraction at {(i + 1) * 25}% mass",
                    lambda lo, hi: lo >= 0 and hi <= 1,
                    "must be in [0, 1]",
                    Uniform(0.05, 0.95),
                )
                for i in range(3)  # default Nparam=3 → 3 quantile parameters
            },
        },
        settings={
            "sfh_db_nparam": 3,
            "sfh_db_age_universe_gyr": 13.47,
        },
        internal_param_map={
            "sfh_db_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_db_log_sfr_inst": ("log_sfr_inst", 1.0, 0.0),
            **{f"sfh_db_tx_frac_{i}": (f"tx_frac_{i}", 1.0, 0.0) for i in range(3)},
        },
        composition_type="additive",
    )
)
SFH_REGISTRY["db"] = SFH_REGISTRY["dense_basis"]


# --- dense_basis_pure: quantile-only GP-SFH (no SFR constraint, for field) ---
_register(
    SFHModelSpec(
        name="dense_basis_pure",
        fn=dense_basis_pure,
        params={
            "sfh_dbp_log_total_mass": ParamDef(
                "log10 total stellar mass formed (Msun)",
                _always_true,
                "",
                Uniform(8.0, 12.0),
            ),
            **{
                f"sfh_dbp_tx_frac_{i}": ParamDef(
                    f"Cosmic time fraction at {(i + 1) * 25}% mass",
                    lambda lo, hi: lo >= 0 and hi <= 1,
                    "must be in [0, 1]",
                    Uniform(0.05, 0.95),
                )
                for i in range(3)
            },
        },
        settings={
            "sfh_dbp_nparam": 3,
            "sfh_dbp_age_universe_gyr": 13.47,
        },
        internal_param_map={
            "sfh_dbp_log_total_mass": ("log_total_mass", 1.0, 0.0),
            **{f"sfh_dbp_tx_frac_{i}": (f"tx_frac_{i}", 1.0, 0.0) for i in range(3)},
        },
        composition_type="additive",
    )
)
SFH_REGISTRY["dbp"] = SFH_REGISTRY["dense_basis_pure"]


# ── Register burst (mixture) model ────────────────────────────────

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


# ── Register field (modulator) model ──────────────────────────────


def _field_fn_placeholder(t_lookback, **kwargs):
    """Placeholder — field modulation is applied in the composed closure.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    **kwargs
        Unused.

    Returns
    -------
    NotImplemented
        This function should not be called directly.

    Raises
    ------
    RuntimeError
        Always — field modulation happens in :func:`resolve_sfh`, not here.

    Notes
    -----
    **JIT-compatible**: no — raises at runtime.

    """
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


# ── Composition: resolve_sfh() ────────────────────────────────────


def resolve_sfh(
    mean_sfh_type: str | list[str],
    bin_edges_gyr: object = None,
) -> tuple[object, dict[str, ParamDef], dict[str, tuple[str, float, float]], dict[str, Any]]:
    """Resolve SFH specification to a composed function + params.

    Parameters
    ----------
    mean_sfh_type : str or list[str]
        Model name(s). E.g., ``"tsnorm"`` or ``["tsnorm", "burst", "field"]``.
    bin_edges_gyr : array-like, shape (n_bins+1,), optional
        Custom age bin edges [Gyr] for ``continuity`` and ``dirichlet`` models.
        When provided, overrides the default ``DEFAULT_BIN_EDGES_GYR``. Use
        ``make_agebins_from_zred`` to generate redshift-appropriate edges.
        Ignored for non-nonparametric models. Default None (use model default).

    Returns
    -------
    composed_fn : callable
        Pure JAX function: fn(t_lookback, **all_internal_kwargs) -> SFR [Msun/yr].
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
        If composition constraints are violated (e.g., >1 burst, no smooth component).

    Notes
    -----
    **JIT-compatible**: yes — returns a JIT-compatible closure (composed_fn).

    Composition rules:

    - **Additive**: smooth models summed. E.g., ``["tsnorm", "dpl"]`` yields
      ``SFR_total = SFR_tsnorm + SFR_dpl``.
    - **Mixture** (burst): mass-fraction weighted, replaces smooth. E.g.,
      ``["tsnorm", "burst"]`` yields ``SFR = (1-f)*SFR_tsnorm + f*burst_shape``.
    - **Modulator** (field): multiplicative GP modulation. E.g.,
      ``["tsnorm", "field"]`` yields ``SFR = SFR_tsnorm * exp(gp_x - K_0/2)``.

    Auto-swap: ``dense_basis`` → ``dense_basis_pure`` if burst or field is
    present (to avoid SFR constraint interference with composition).

    Examples
    --------
    >>> from tengri import resolve_sfh
    >>> fn, params, param_map, settings = resolve_sfh("dpl")
    >>> "sfh_dpl_alpha" in params
    True
    >>> fn  # doctest: +ELLIPSIS
    <function ...>

    """
    if isinstance(mean_sfh_type, str):
        mean_sfh_type = [mean_sfh_type]

    # Auto-swap: dense_basis → dense_basis_pure when field or burst is present.
    # The SFR constraint points in dense_basis pin recent SFH shape, which
    # interferes with both the GP field modulator and the triweight burst
    # kernel (Zacharegkas+2025) that also control recent SFR variability.
    _DB_TO_PURE = {"dense_basis": "dense_basis_pure", "db": "dbp"}
    has_compositor = any(n in ("field", "burst") for n in mean_sfh_type)
    if has_compositor:
        mean_sfh_type = [_DB_TO_PURE.get(n, n) for n in mean_sfh_type]

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
    _NONPARAM_NAMES = {"continuity", "dirichlet", "continuity_flex"}
    additive_info = []
    for s in additive:
        internal_names = {v[0] for v in s.internal_param_map.values()}
        fn_i = s.fn
        if bin_edges_gyr is not None and s.name in _NONPARAM_NAMES:
            fn_i = functools.partial(fn_i, bin_edges_gyr=bin_edges_gyr)
        additive_info.append((fn_i, internal_names))

    has_burst = len(mixtures) > 0
    burst_info = None
    if has_burst:
        bs = mixtures[0]
        burst_internal = {v[0] for v in bs.internal_param_map.values()}
        burst_info = (bs.fn, burst_internal)

    has_field = len(modulators) > 0

    # Build the composed closure
    def composed_fn(t_lookback, **kw):
        """Evaluate the composed SFH: sum additive components, then apply burst and field."""
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

    Notes
    -----
    **JIT-compatible**: yes — uses ``gp_from_xi`` and PSD model functions
    from the field model registry.

    The Gaussian process realization models burstiness via a correlated
    random field in log-space. The PSD model (e.g., "drw" for Damped
    Random Walk) controls temporal correlations. The lognormal correction
    ``k0_half`` accounts for the bias introduced when exponentiation is
    applied to the Gaussian latents.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import compute_field_gp, make_log_age_grid
    >>> n = 64
    >>> grid = make_log_age_grid(n)
    >>> d = float(grid[1] - grid[0])
    >>> xi = jnp.zeros(n)
    >>> gp_x, k0_half = compute_field_gp(xi, psd_sigma=1.0, psd_tau_yr=1e8, n_grid=n, d_log_age=d)
    >>> gp_x.shape
    (64,)

    """
    sqrt_power_fn = FIELD_MODEL_REGISTRY[field_model]
    sqrt_power = sqrt_power_fn(n_grid, d_log_age, psd_sigma, psd_tau_yr)
    gp_x = gp_from_xi(xi, sqrt_power, n_grid)
    k0_half = drw_variance(psd_sigma) / 2.0
    return gp_x, k0_half

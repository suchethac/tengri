"""Metallicity mode registry (mirrors SFH registry pattern).

Provides a registry of metallicity evolution modes, each with its own
parameter definitions and internal param map.  The user selects a mode
via ``met_mode="two_step"`` (etc.) in :class:`Parameters`, and
:func:`resolve_met` returns the function + params + param_map.

Modes
-----
- ``delta``:  Single metallicity for all ages (default).
- ``ramp``:  Linear ramp from initial to final Z.
- ``two_step``:  Step function at a lookback time.
- ``psb_two_step``:  Step at PSB burst age.
- ``bins``:  Per-bin metallicities (pairs with continuity SFH).
- ``bins_continuity``:  Cumulative delta-log-Z steps from base.
- ``chem_evol``:  Gas-regulator model (Z derived from SFH).
- ``table``:  User-provided Z(t) table.

All metallicity inputs are in **log10(Z/Zsun)** at the user-facing level.
The ``internal_param_map`` applies the ``LOG10_ZSUN`` offset to convert
to absolute log10(Z) where needed.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from tengri.parameters.priors import Distribution, Fixed, Uniform
from tengri.parameters.translate import LOG10_ZSUN

# ── Data structures ───────────────────────────────────────────────


class MetParamDef(NamedTuple):
    """Definition of a single fittable metallicity parameter.

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
    bound_check: object
    bound_error: str
    default: Distribution


class MetModelSpec(NamedTuple):
    """Specification of a registered metallicity mode.

    Attributes
    ----------
    name : str
        Mode name (e.g., "delta", "two_step", "bins").
    fn : callable or None
        Pure JAX function that returns log10(Z) absolute per SSP age.
        None for modes handled specially (delta, chem_evol, table).
    params : dict[str, MetParamDef]
        Fittable parameters: public_name -> MetParamDef.
    settings : dict[str, Any]
        Non-fittable settings with defaults.
    internal_param_map : dict[str, tuple[str, float, float]]
        public_name -> (internal_name, scale, offset).
        Conversion: internal = public * scale + offset.
    """

    name: str
    fn: object
    params: dict[str, MetParamDef]
    settings: dict[str, Any]
    internal_param_map: dict[str, tuple[str, float, float]]


# ── Registry ──────────────────────────────────────────────────────

MET_REGISTRY: dict[str, MetModelSpec] = {}

_always_true = lambda lo, hi: True  # noqa: E731


def _register(spec: MetModelSpec) -> None:
    """Register a metallicity mode spec in the global registry.

    Parameters
    ----------
    spec : MetModelSpec
        Metallicity mode specification to register.

    Returns
    -------
    None

    Notes
    -----
    **JIT-compatible**: no — mutates global registry dictionary.
    """
    MET_REGISTRY[spec.name] = spec


# ── delta: single metallicity (the default) ───────────────────────

_register(
    MetModelSpec(
        name="delta",
        fn=None,
        params={
            "met_logzsol": MetParamDef(
                "log10(Z/Zsun)",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
        },
        settings={},
        internal_param_map={
            "met_logzsol": ("log_z_abs", 1.0, LOG10_ZSUN),
        },
    )
)

# ── ramp: linear evolving metallicity ─────────────────────────────

_register(
    MetModelSpec(
        name="ramp",
        fn=None,  # handled via compute_log_z_evolving in dsps_wrapper
        params={
            "met_logzsol_0": MetParamDef(
                "log10(Z/Zsun) at earliest time",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
            "met_logzsol_final": MetParamDef(
                "log10(Z/Zsun) at present day",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
        },
        settings={},
        internal_param_map={
            "met_logzsol_0": ("log_z_abs_initial", 1.0, LOG10_ZSUN),
            "met_logzsol_final": ("log_z_abs_final", 1.0, LOG10_ZSUN),
        },
    )
)

# ── two_step: step function at a lookback time ────────────────────

_register(
    MetModelSpec(
        name="two_step",
        fn=None,  # set below after imports are safe
        params={
            "met_logzsol_old": MetParamDef(
                "log10(Z/Zsun) for old stars (before step)",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
            "met_logzsol_young": MetParamDef(
                "log10(Z/Zsun) for young stars (after step)",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
            "met_step_age_gyr": MetParamDef(
                "Lookback time of metallicity step (Gyr)",
                lambda lo, hi: lo > 0,
                "must have lo > 0",
                Uniform(0.1, 10.0),
            ),
        },
        settings={},
        internal_param_map={
            "met_logzsol_old": ("log_z_abs_old", 1.0, LOG10_ZSUN),
            "met_logzsol_young": ("log_z_abs_young", 1.0, LOG10_ZSUN),
            "met_step_age_gyr": ("step_age_gyr", 1.0, 0.0),
        },
    )
)

# ── psb_two_step: step at PSB burst age ───────────────────────────

_register(
    MetModelSpec(
        name="psb_two_step",
        fn=None,
        params={
            "met_logzsol_old": MetParamDef(
                "log10(Z/Zsun) for pre-burst stars",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
            "met_logzsol_burst": MetParamDef(
                "log10(Z/Zsun) for burst stars",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
        },
        settings={},
        internal_param_map={
            "met_logzsol_old": ("log_z_abs_old", 1.0, LOG10_ZSUN),
            "met_logzsol_burst": ("log_z_abs_burst", 1.0, LOG10_ZSUN),
            # burstage_gyr comes from the SFH params, not declared here
        },
    )
)

# ── bins: per-bin metallicities (pairs with continuity SFH) ───────

# Default: 6 bins with 7 edges matching typical continuity SFH
_N_MET_BINS_DEFAULT = 6

_register(
    MetModelSpec(
        name="bins",
        fn=None,
        params={
            **{
                f"met_bin_{i}": MetParamDef(
                    f"log10(Z/Zsun) for time bin {i} (youngest=0)",
                    _always_true,
                    "",
                    Uniform(-2.0, 0.2),
                )
                for i in range(_N_MET_BINS_DEFAULT)
            },
        },
        settings={
            "met_n_bins": _N_MET_BINS_DEFAULT,
        },
        internal_param_map={
            **{
                f"met_bin_{i}": (f"met_bin_abs_{i}", 1.0, LOG10_ZSUN)
                for i in range(_N_MET_BINS_DEFAULT)
            },
        },
    )
)

# ── bins_continuity: cumulative delta-log-Z steps ─────────────────

_register(
    MetModelSpec(
        name="bins_continuity",
        fn=None,
        params={
            "met_logzsol_base": MetParamDef(
                "log10(Z/Zsun) of the oldest bin",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
            **{
                f"met_d_log_z_{i}": MetParamDef(
                    f"delta-log-Z step {i} (old→young)",
                    _always_true,
                    "",
                    Uniform(-0.5, 0.5),
                )
                for i in range(_N_MET_BINS_DEFAULT - 1)
            },
        },
        settings={
            "met_n_bins": _N_MET_BINS_DEFAULT,
        },
        internal_param_map={
            "met_logzsol_base": ("log_z_abs_base", 1.0, LOG10_ZSUN),
            **{
                f"met_d_log_z_{i}": (f"d_log_z_{i}", 1.0, 0.0)
                for i in range(_N_MET_BINS_DEFAULT - 1)
            },
        },
    )
)

# ── chem_evol: gas-regulator model (Z derived from SFH) ───────────

_register(
    MetModelSpec(
        name="chem_evol",
        fn=None,  # handled in pipeline via chemical_evolution module
        params={
            "chem_yield": MetParamDef(
                "Nucleosynthetic yield",
                lambda lo, hi: lo > 0,
                "must have lo > 0",
                Fixed(0.03),
            ),
            "chem_eta_outflow": MetParamDef(
                "Mass loading factor (outflow rate / SFR)",
                lambda lo, hi: lo >= 0,
                "must have lo >= 0",
                Fixed(0.0),
            ),
            "chem_f_gas_init": MetParamDef(
                "Initial gas fraction",
                lambda lo, hi: lo > 0 and hi <= 1,
                "must be in (0, 1]",
                Fixed(0.9),
            ),
            "chem_return_frac": MetParamDef(
                "Stellar return fraction",
                lambda lo, hi: lo >= 0 and hi < 1,
                "must be in [0, 1)",
                Fixed(0.4),
            ),
        },
        settings={},
        internal_param_map={
            "chem_yield": ("chem_yield", 1.0, 0.0),
            "chem_eta_outflow": ("chem_eta_outflow", 1.0, 0.0),
            "chem_f_gas_init": ("chem_f_gas_init", 1.0, 0.0),
            "chem_return_frac": ("chem_return_frac", 1.0, 0.0),
        },
    )
)

# ── table: user-provided Z(t) ─────────────────────────────────────

_register(
    MetModelSpec(
        name="table",
        fn=None,  # handled via tabulated_metallicity_on_ssp_grid
        params={},  # no fittable params — the table IS the Z(t)
        settings={},
        internal_param_map={},
    )
)


# ── resolve_met() ─────────────────────────────────────────────────


def resolve_met(
    met_mode: str,
) -> tuple[
    MetModelSpec,
    dict[str, MetParamDef],
    dict[str, tuple[str, float, float]],
    dict[str, Any],
]:
    """Resolve metallicity mode to spec + params + param_map + settings.

    Parameters
    ----------
    met_mode : str
        Metallicity mode name (e.g., "delta", "two_step", "bins").

    Returns
    -------
    spec : MetModelSpec
        Full model specification.
    params : dict[str, MetParamDef]
        Fittable parameters: public_name -> MetParamDef.
    param_map : dict[str, tuple[str, float, float]]
        public_name -> (internal_name, scale, offset) for unit/offset conversion.
    settings : dict[str, Any]
        Non-fittable settings (e.g., met_n_bins for binned modes).

    Raises
    ------
    KeyError
        If met_mode is not registered.

    Notes
    -----
    **JIT-compatible**: no — performs dictionary lookup at initialization time.

    All metallicity inputs are in **log10(Z/Zsun)** (relative to solar).
    The param_map applies LOG10_ZSUN offset to convert to absolute log10(Z) internally.
    """
    if met_mode not in MET_REGISTRY:
        valid = sorted(MET_REGISTRY.keys())
        raise KeyError(f"Unknown met_mode '{met_mode}'. Valid modes: {valid}")
    spec = MET_REGISTRY[met_mode]
    return spec, dict(spec.params), dict(spec.internal_param_map), dict(spec.settings)

# SPDX-License-Identifier: BSD-3-Clause
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
        Default prior distribution: what the parameter resolves to when
        nothing asks for it to be free. Usually ``Fixed``.
    free_prior : Distribution or None, optional
        The admissible range ``all_params: FREE`` expands to. ``None`` means
        the parameter is not freeable by the wildcard.

        Mirrors the field of the same name on
        :class:`~tengri.components.stellar.sfh.registry.ParamDef` and on
        :class:`~tengri.protocols.component.ParamDeclaration`. All three
        declaration mechanisms must carry it or the parameters they own are
        invisible to ``FREE`` no matter what they declare (#887).

    Notes
    -----
    **JIT-compatible**: no; Python dataclass for registry initialization.

    """

    description: str
    bound_check: object
    bound_error: str
    default: Distribution
    free_prior: Distribution | None = None


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

    Notes
    -----
    **JIT-compatible**: no; Python dataclass for registry initialization.

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
    **JIT-compatible**: no, mutates global registry dictionary.

    """
    MET_REGISTRY[spec.name] = spec


# ── delta: single metallicity (the default) ───────────────────────

_register(
    MetModelSpec(
        name="delta",
        fn=None,
        params={
            "met_logzsol": MetParamDef(
                "log10(Z/Zsun); Zsun = 0.0142 (Asplund 2009, MIST)",
                _always_true,
                "",
                # Flat-in-log prior across the SSP template range. The
                # wildcard-Fixed(DEFAULT) resolver special-cases ``met_logzsol`` to
                # pin the default at **solar** (0.0) rather than the prior
                # midpoint: matching FSPS (``logzsol=0.0``) and Bagpipes
                # (``metallicity=1.0 Z⊙``). The previous default (midpoint
                # -0.9) silently introduced a ~0.85 dex offset in CIGALE
                # comparisons: see #412 for the trace.
                #
                # "Solar" is Asplund 2009 Zsun = 0.0142 (= MIST). For SSP
                # libraries built against a different Zsun reference (BC03 /
                # Padova: 0.0190; PARSEC: 0.0152; BASTI: 0.0200) reason in
                # absolute ``log_z_abs`` for bit-exact cross-code matches;
                # see tengri.parameters.translate.LOG10_ZSUN_BY_LIBRARY.
                Uniform(-2.0, 0.2),
            ),
            "met_logzsol_scatter": MetParamDef(
                "Lognormal metallicity scatter sigma [dex]; Gaussian-in-log10(Z) "
                "MDF width about the mean (Carnall+2018 §3.2). sigma -> 0 recovers "
                "a single-Z (delta) population.",
                _always_true,
                "",
                # Default pinned to the historical fixed ``config.lgmet_scatter``
                # (0.1) so delta models that do not free it are byte-unchanged;
                # the wildcard-Fixed(DEFAULT) resolver reads this default from
                # ``_CANONICAL_FIXED_DEFAULTS`` in ``parameters/groups.py``. Free
                # it (e.g. ``Uniform(0.0, 0.5)``) to fit the MDF width like
                # Bagpipes' ``lognorm`` chemical-enrichment mode.
                Fixed(0.1),
                # Deliberately NO free_prior (#887), and note what the comment
                # above is actually saying: "*Free it* ... like Bagpipes'
                # lognorm mode" is an instruction to the caller, mirroring a
                # mode Bagpipes gates behind an explicit opt-in. It is not a
                # request that the wildcard reach it.
                #
                # Measured: declaring one added this parameter to 6 of the 10
                # shipped recipes, because ``met_*`` routes into the ``sfh``
                # group and every recipe frees that block. It is the second
                # moment of the MDF -- broadband photometry constrains the mean
                # metallicity weakly and its width barely at all -- so that is a
                # near-unconstrained dimension in every default fit, returning
                # the prior and costing warmup. The mean (``met_logzsol``) is
                # already free; the width stays explicit.
            ),
        },
        settings={},
        internal_param_map={
            "met_logzsol": ("log_z_abs", 1.0, LOG10_ZSUN),
            # Identity map: the stellar component reads the public
            # ``met_logzsol_scatter`` name directly (a width, no Zsun offset).
            "met_logzsol_scatter": ("met_logzsol_scatter", 1.0, 0.0),
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
                # The mass of metals returned per unit mass locked in stars.
                # For a standard IMF this is of order the solar metallicity and
                # a few times it; the range brackets the values used across
                # closed-box and leaky-box chemical evolution treatments.
                Uniform(0.005, 0.1, "Nucleosynthetic yield", default=0.03),
            ),
            "chem_eta_outflow": MetParamDef(
                "Mass loading factor (outflow rate / SFR)",
                lambda lo, hi: lo >= 0,
                "must have lo >= 0",
                Fixed(0.0),
                # 0 (the default) is a closed box. Mass loading rises steeply
                # toward low mass, reaching order 10 in dwarfs, which sets the
                # ceiling; the quantity is a ratio to the SFR so it is
                # galaxy-mass dependent but not galaxy-scale dependent.
                Uniform(0.0, 10.0, "Outflow mass loading factor", default=0.0),
            ),
            "chem_f_gas_init": MetParamDef(
                "Initial gas fraction",
                lambda lo, hi: lo > 0 and hi <= 1,
                "must be in (0, 1]",
                Fixed(0.9),
                # The validator's own (0, 1] interval; the floor is nudged off
                # zero because a zero initial gas fraction has no gas to form
                # any stars from.
                Uniform(0.01, 1.0, "Initial gas fraction", default=0.9),
            ),
            "chem_return_frac": MetParamDef(
                "Stellar return fraction",
                lambda lo, hi: lo >= 0 and hi < 1,
                "must be in [0, 1)",
                Fixed(0.4),
                # The fraction of formed stellar mass returned to the ISM. It
                # is set by the IMF rather than by the galaxy -- roughly 0.3 for
                # a Salpeter-like slope and 0.45 for Chabrier/Kroupa -- so the
                # range covers the IMFs in use rather than the validator's full
                # [0, 1).
                Uniform(0.2, 0.6, "Stellar return fraction", default=0.4),
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
        params={},  # no fittable params: the table IS the Z(t)
        settings={},
        internal_param_map={},
    )
)

# ── massmap_lin: Linear metallicity tied to cumulative mass formed ────

_register(
    MetModelSpec(
        name="massmap_lin",
        fn=None,  # handled via massmap_lin_metallicity
        params={
            "met_logzsol_start": MetParamDef(
                "log10(Z/Zsun) at the oldest age (Zstart)",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
            "met_logzsol_final": MetParamDef(
                "log10(Z/Zsun) at present day (Zfinal)",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
        },
        settings={},
        internal_param_map={
            "met_logzsol_start": ("log_z_abs_start", 1.0, LOG10_ZSUN),
            "met_logzsol_final": ("log_z_abs_final", 1.0, LOG10_ZSUN),
        },
    )
)

# ── massmap_box: Closed-box metallicity tied to cumulative mass formed ────

_register(
    MetModelSpec(
        name="massmap_box",
        fn=None,  # handled via massmap_box_metallicity
        params={
            "met_logzsol_start": MetParamDef(
                "log10(Z/Zsun) at the oldest age (Zstart)",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
            "met_logzsol_final": MetParamDef(
                "log10(Z/Zsun) at present day (Zfinal)",
                _always_true,
                "",
                Uniform(-2.0, 0.2),
            ),
            # ProSpect names this ``yield`` (the nucleosynthetic yield ρ), but
            # ``yield`` is a Python keyword and cannot be used as a builder
            # group key, so the user-facing name is ``met_yield`` (consistent
            # with the ``met_*`` prefix of the other metallicity params).
            "met_yield": MetParamDef(
                "Fixed nucleosynthetic yield parameter ρ (ProSpect ``yield``)",
                lambda lo, hi: lo > 0,
                "must have lo > 0",
                Fixed(0.03),
                # Same quantity and same range as ``chem_yield`` above, under
                # ProSpect's spelling; kept identical so the two cannot drift.
                Uniform(0.005, 0.1, "Nucleosynthetic yield", default=0.03),
            ),
        },
        settings={},
        internal_param_map={
            "met_logzsol_start": ("log_z_abs_start", 1.0, LOG10_ZSUN),
            "met_logzsol_final": ("log_z_abs_final", 1.0, LOG10_ZSUN),
            "met_yield": ("yield_rho", 1.0, 0.0),
        },
    )
)


# ── Auto-inference of met_mode from prior keys ────────────────────

# Discriminator keys per mode: keys whose presence in the user's
# Parameters kwargs *strongly implies* that mode. Modes are tested
# in order of specificity (most specific first), so adding a new
# mode means inserting a (name, discriminator_keys) tuple here.
#
# Excluded from this table:
# - "delta": no positive discriminator (it's the default fallback).
# - "chem_evol": detected separately via any `chem_*` key, since the
#   four chem_ params can independently appear as Fixed or Free.
# - "table": cannot be inferred; it has no fittable params, so users
#   must set ``met_mode="table"`` explicitly.
_MET_MODE_DISCRIMINATORS: tuple[tuple[str, frozenset[str]], ...] = (
    ("bins_continuity", frozenset({"met_logzsol_base"})),
    ("bins", frozenset({"met_bin_0"})),
    ("massmap_box", frozenset({"met_logzsol_start", "met_yield"})),
    ("massmap_lin", frozenset({"met_logzsol_start", "met_logzsol_final"})),
    ("two_step", frozenset({"met_step_age_gyr"})),
    ("psb_two_step", frozenset({"met_logzsol_burst"})),
    ("ramp", frozenset({"met_logzsol_0", "met_logzsol_final"})),
)


def infer_met_mode(provided_keys: set[str] | frozenset[str]) -> str:
    """Infer the metallicity mode from the parameter keys a user provided.

    Used by :class:`tengri.Parameters` when ``met_mode`` is
    not set explicitly: presence of mode-specific keys (e.g.
    ``met_logzsol_0`` and ``met_logzsol_final``) implies the
    corresponding mode (``"ramp"``).

    Parameters
    ----------
    provided_keys : set or frozenset of str
        The set of parameter / kwarg names the user passed to
        :class:`Parameters`. Typically ``set(kwargs.keys())`` after
        the constructor pops its non-prior settings.

    Returns
    -------
    str
        One of: ``"delta"`` (default fallback), ``"ramp"``,
        ``"two_step"``, ``"psb_two_step"``, ``"bins"``,
        ``"bins_continuity"``, ``"chem_evol"``.

    Raises
    ------
    ValueError
        If the keys imply more than one mode unambiguously (e.g. both
        ``met_logzsol_0``+``met_logzsol_final`` and
        ``met_step_age_gyr`` are present).

    Notes
    -----
    **JIT-compatible**: no, pure-Python set membership at construction time.

    Cannot infer ``"table"`` (no characteristic params); set explicitly.
    """
    keys = set(provided_keys)
    if any(k.startswith("chem_") for k in keys):
        return "chem_evol"
    matches = [name for name, disc in _MET_MODE_DISCRIMINATORS if disc.issubset(keys)]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous metallicity-mode inference: parameter keys match multiple "
            f"modes {matches}. Set met_mode=... explicitly to disambiguate."
        )
    return matches[0] if matches else "delta"


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
    **JIT-compatible**: no, performs dictionary lookup at initialization time.

    All metallicity inputs are in **log10(Z/Zsun)** (relative to solar).
    The param_map applies LOG10_ZSUN offset to convert to absolute log10(Z) internally.

    """
    if met_mode not in MET_REGISTRY:
        valid = sorted(MET_REGISTRY.keys())
        raise KeyError(f"Unknown met_mode '{met_mode}'. Valid modes: {valid}")
    spec = MET_REGISTRY[met_mode]
    return spec, dict(spec.params), dict(spec.internal_param_map), dict(spec.settings)

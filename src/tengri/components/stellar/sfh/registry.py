# SPDX-License-Identifier: BSD-3-Clause
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp

from tengri.components.stellar.sfh.dense_basis import dense_basis, dense_basis_pure
from tengri.components.stellar.sfh.gp_sfh import (
    compute_sqrt_power_drw,
    drw_innovations_gp_from_xi,
    gp_from_xi,
    make_log_age_grid,
)
from tengri.components.stellar.sfh.mean_sfh import (
    AGEMAX_YR,
    buat08,
    constant,
    constant_then_exponential,
    declining_exponential,
    delayed_bq,
    delayed_exponential,
    dpl,
    exponential,
    gaussian_burst,
    lnorm,
    norm,
    periodic,
    psb_wild2020,
    sfh2exp,
    sfhdelayed,
    snorm,
    snorm_burst,
    snorm_trunc_burst,
    top_hat,
    triweight_burst,
    tsnorm,
)
from tengri.components.stellar.sfh.nonparametric import (
    CFLEX_DEFAULT_ANCHOR_GYR,
    DEFAULT_BIN_EDGES_GYR,
    continuity,
    continuity_flex,
    dirichlet,
    psb_continuity,
)
from tengri.components.stellar.sfh.psd_models import drw_variance
from tengri.parameters.priors import Distribution, Fixed, StudentT, Uniform
from tengri.utils.cosmology import age_at_z0

# Age of the universe today [Gyr], from the default cosmology — never a
# literal. Used as the prior upper bound and default for the dpl/lnorm
# formation anchors ``sfh_*_age_gyr`` (cosmic time available for star
# formation = lookback of formation at the Big Bang). Per-fit, users
# override this with ``cosmology.age_at_z(z)`` at the source redshift.
_AGE_UNIV_GYR = round(float(age_at_z0()), 3)

# ── Data structures ───────────────────────────────────────────────


@dataclass(frozen=True)
class SFHRegistryEntry:
    """Registry entry for an SFH model with optional metadata.

    Attributes
    ----------
    callable : Callable
        The pure JAX SFH function.
    citation : str
        Optional academic citation. Default empty string.
    status : str
        Model status: "production", "experimental", "demo", or "deprecated".
        Default "production".
    short_doc : str
        Optional one-line description. Default empty string.

    Notes
    -----
    **JIT-compatible**: no — dataclass for registry initialization.

    """

    callable: Callable
    citation: str = ""
    status: str = "production"
    short_doc: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward calls to the wrapped callable."""
        return object.__getattribute__(self, "callable")(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped callable (SFHModelSpec)."""
        callable_obj = object.__getattribute__(self, "callable")
        return getattr(callable_obj, name)


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
        Default prior distribution — what the parameter resolves to when
        nothing asks for it to be free. Usually ``Fixed``.
    free_prior : Distribution or None, optional
        The admissible range ``all_params: FREE`` expands to. ``None`` means
        the parameter is not freeable by the wildcard and ``FREE`` falls back
        to ``default``.

        This mirrors :class:`~tengri.protocols.component.ParamDeclaration`'s
        field of the same name. Without it an SFH parameter could not be
        declared freeable at all: the component ``_params.py`` path carries a
        ``free_prior`` and this registry did not, so every ``sfh_*`` entry
        reached :func:`~tengri.parameters.registry.registry` with ``None`` no
        matter what it declared (#887).

    Notes
    -----
    **JIT-compatible**: no — Python dataclass for registry initialization.

    """

    description: str
    bound_check: object  # Callable[[float, float], bool]
    bound_error: str
    default: Distribution
    free_prior: Distribution | None = None


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

#: Every registered mean-SFH family, keyed by the ``sfh={'type': ...}`` name
#: (``'dpl'``, ``'tsnorm'``, ``'delayed_tau'``, ``'field'``, …). Values are the
#: :class:`SFHSpec` entries that :func:`resolve_sfh` dispatches on and that
#: ``tengri.builders.sfh.*`` is generated from. Populated at import time by
#: :func:`register_sfh`; treat it as read-only.
SFH_REGISTRY: dict[str, Any] = {}

#: SFH names present in :data:`SFH_REGISTRY` but NOT yet validated against the
#: DSPS forward path (the ``_SUPPORTED_SFH`` allowlist in
#: ``components/stellar/component.py``). The public grammar
#: (``parameters.groups._valid_sfh_types``) and the auto-generated
#: ``tengri.builders.sfh.*`` factories exclude these, so the advertised set
#: matches what actually forward-models — otherwise ``SEDModel.build`` succeeds
#: and ``predict`` then raises ``NotImplementedError``. Promote a name out of
#: this set once it is added to ``_SUPPORTED_SFH`` and crossvalidated.
UNVALIDATED_SFH_TYPES: frozenset[str] = frozenset(
    {
        "bursty_continuity",
        "gaussian_burst",
        "prospector_beta",
        "psb_wild2020",
        "top_hat",
        # Registered but absent from the stellar component's runtime
        # _SUPPORTED_SFH allowlist (components/stellar/component.py) —
        # without an entry here they build fine and then die at the first
        # predict with NotImplementedError. Keep the two gates in sync
        # until they share one source of truth.
        "constant_then_exponential",
        "db",
        "dbp",
    }
)

#: Stochastic-field PSD models, keyed by the ``sfh={'psd': ...}`` name. Values
#: are the ``sqrt_power(omega, ...)`` callables the Gaussian-process field
#: draws its amplitude operator from. ``'drw'`` (damped random walk) is the
#: only entry today.
FIELD_MODEL_REGISTRY: dict[str, object] = {
    "drw": compute_sqrt_power_drw,
}

_always_true = lambda lo, hi: True  # noqa: E731
_lo_positive = lambda lo, hi: lo > 0  # noqa: E731
_lo_nonneg = lambda lo, hi: lo >= 0  # noqa: E731


def _register(
    spec: SFHModelSpec,
    citation: str = "",
    status: str = "production",
    short_doc: str = "",
) -> None:
    """Register an SFH model spec in the global registry.

    Parameters
    ----------
    spec : SFHModelSpec
        Model specification to register.
    citation : str, optional
        Academic citation for the model. Default empty string.
    status : str, optional
        Model status ("production", "experimental", "demo", "deprecated").
        Default "production".
    short_doc : str, optional
        One-line description. Default empty string.

    Returns
    -------
    None

    Notes
    -----
    **JIT-compatible**: no — mutates global registry dictionary during initialization.

    """
    entry = SFHRegistryEntry(
        callable=spec,
        citation=citation,
        status=status,
        short_doc=short_doc,
    )
    SFH_REGISTRY[spec.name] = entry


# ── Register smooth (additive) models ─────────────────────────────

# --- tsnorm (truncated skew-normal) — canonical: truncated_skewnormal_sfh ---
_tsnorm_spec = SFHModelSpec(
    name="tsnorm",
    fn=tsnorm,
    params={
        "sfh_tsnorm_log_total_mass": ParamDef(
            "log10 total stellar mass formed [Msun]",
            _always_true,
            "",
            Uniform(7.0, 12.5, default=10.0),
        ),
        "sfh_tsnorm_peak_lbt_gyr": ParamDef(
            "Peak lookback time (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.5, 12.0, default=5.0),
        ),
        "sfh_tsnorm_width_gyr": ParamDef(
            "Gaussian width (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.2, 5.0, default=1.0),
        ),
        "sfh_tsnorm_skew": ParamDef("Skewness", _always_true, "", Uniform(-1.0, 1.0, default=0.0)),
        "sfh_tsnorm_trunc": ParamDef(
            "Truncation sharpness",
            _lo_positive,
            "must have lo > 0",
            Uniform(1.0, 10.0, default=2.0),
        ),
    },
    settings={},
    internal_param_map={
        "sfh_tsnorm_log_total_mass": ("log_total_mass", 1.0, 0.0),
        "sfh_tsnorm_peak_lbt_gyr": ("peak_lbt", 1e9, 0.0),
        "sfh_tsnorm_width_gyr": ("width", 1e9, 0.0),
        "sfh_tsnorm_skew": ("skew", 1.0, 0.0),
        "sfh_tsnorm_trunc": ("trunc", 1.0, 0.0),
    },
    composition_type="additive",
)
_register(
    _tsnorm_spec,
    citation="Bellstedt et al. 2020 (arXiv:2005.11917)",
    short_doc="Truncated skew-normal SFH (Bellstedt+2020)",
)

# --- snorm (skew-normal) — canonical: skewnormal_sfh ---
_snorm_spec = SFHModelSpec(
    name="snorm",
    fn=snorm,
    params={
        "sfh_snorm_log_total_mass": ParamDef(
            "log10 total stellar mass formed [Msun]",
            _always_true,
            "",
            Uniform(7.0, 12.5, default=10.0),
        ),
        "sfh_snorm_peak_lbt_gyr": ParamDef(
            "Peak lookback time (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.5, 12.0, default=5.0),
        ),
        "sfh_snorm_width_gyr": ParamDef(
            "Gaussian width (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.2, 5.0, default=1.0),
        ),
        "sfh_snorm_skew": ParamDef("Skewness", _always_true, "", Uniform(-1.0, 1.0, default=0.0)),
    },
    settings={},
    internal_param_map={
        "sfh_snorm_log_total_mass": ("log_total_mass", 1.0, 0.0),
        "sfh_snorm_peak_lbt_gyr": ("peak_lbt", 1e9, 0.0),
        "sfh_snorm_width_gyr": ("width", 1e9, 0.0),
        "sfh_snorm_skew": ("skew", 1.0, 0.0),
    },
    composition_type="additive",
)
_register(
    _snorm_spec,
    citation="Bellstedt et al. 2020 (arXiv:2005.11917)",
    short_doc="Skew-normal SFH (Bellstedt+2020)",
)
# --- snorm_burst (skew-normal + flat burst) ---
_snorm_burst_spec = SFHModelSpec(
    name="snorm_burst",
    fn=snorm_burst,
    params={
        "sfh_snorm_burst_log_total_mass": ParamDef(
            "log10 total stellar mass formed [Msun]",
            _always_true,
            "",
            Uniform(7.0, 12.5, default=10.0),
        ),
        "sfh_snorm_burst_peak_lbt_gyr": ParamDef(
            "Peak lookback time (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.5, 12.0, default=5.0),
        ),
        "sfh_snorm_burst_width_gyr": ParamDef(
            "Gaussian width (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.2, 5.0, default=1.0),
        ),
        "sfh_snorm_burst_skew": ParamDef(
            "Skewness", _always_true, "", Uniform(-1.0, 1.0, default=0.0)
        ),
        # burst_sfr deliberately gets no free_prior: it is an absolute rate in
        # Msun/yr, so its plausible range is set by the galaxy being fitted and
        # no galaxy-independent interval exists (same reasoning as
        # ``dust_L_agn_ir``). Free it explicitly against your own SFR scale.
        "sfh_snorm_burst_burst_sfr": ParamDef(
            "Constant burst SFR amplitude (Msun/yr)", _lo_nonneg, "must have lo >= 0", Fixed(0.0)
        ),
        "sfh_snorm_burst_burst_age_gyr": ParamDef(
            "Burst lookback duration (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Fixed(0.1),
            # A burst is short by definition -- beyond ~2 Gyr it is no longer a
            # burst but the underlying SFH -- and the lower end stays above zero
            # because the validator requires it and a zero-duration burst
            # carries no mass.
            Uniform(0.01, 2.0, "Burst duration", units="Gyr", default=0.1),
        ),
    },
    settings={},
    internal_param_map={
        "sfh_snorm_burst_log_total_mass": ("log_total_mass", 1.0, 0.0),
        "sfh_snorm_burst_peak_lbt_gyr": ("peak_lbt", 1e9, 0.0),
        "sfh_snorm_burst_width_gyr": ("width", 1e9, 0.0),
        "sfh_snorm_burst_skew": ("skew", 1.0, 0.0),
        "sfh_snorm_burst_burst_sfr": ("burst_sfr", 1.0, 0.0),
        "sfh_snorm_burst_burst_age_gyr": ("burst_age", 1e9, 0.0),
    },
    composition_type="additive",
)
_register(
    _snorm_burst_spec,
    citation="Robotham et al. 2020 (arXiv:2002.06980) ProSpect",
    short_doc="Skew-normal + flat recent burst (ProSpect)",
)
# --- tsnorm_burst (truncated skew-normal + flat burst) ---
_tsnorm_burst_spec = SFHModelSpec(
    name="tsnorm_burst",
    fn=snorm_trunc_burst,
    params={
        "sfh_tsnorm_burst_log_total_mass": ParamDef(
            "log10 total stellar mass formed [Msun]",
            _always_true,
            "",
            Uniform(7.0, 12.5, default=10.0),
        ),
        "sfh_tsnorm_burst_peak_lbt_gyr": ParamDef(
            "Peak lookback time (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.5, 12.0, default=5.0),
        ),
        "sfh_tsnorm_burst_width_gyr": ParamDef(
            "Gaussian width (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.2, 5.0, default=1.0),
        ),
        "sfh_tsnorm_burst_skew": ParamDef(
            "Skewness", _always_true, "", Uniform(-1.0, 1.0, default=0.0)
        ),
        "sfh_tsnorm_burst_trunc": ParamDef(
            "Truncation sharpness",
            _lo_positive,
            "must have lo > 0",
            Uniform(1.0, 10.0, default=2.0),
        ),
        # burst_sfr: no free_prior, for the same reason as the snorm variant.
        "sfh_tsnorm_burst_burst_sfr": ParamDef(
            "Constant burst SFR amplitude (Msun/yr)", _lo_nonneg, "must have lo >= 0", Fixed(0.0)
        ),
        "sfh_tsnorm_burst_burst_age_gyr": ParamDef(
            "Burst lookback duration (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Fixed(0.1),
            Uniform(0.01, 2.0, "Burst duration", units="Gyr", default=0.1),
        ),
    },
    settings={},
    internal_param_map={
        "sfh_tsnorm_burst_log_total_mass": ("log_total_mass", 1.0, 0.0),
        "sfh_tsnorm_burst_peak_lbt_gyr": ("peak_lbt", 1e9, 0.0),
        "sfh_tsnorm_burst_width_gyr": ("width", 1e9, 0.0),
        "sfh_tsnorm_burst_skew": ("skew", 1.0, 0.0),
        "sfh_tsnorm_burst_trunc": ("trunc", 1.0, 0.0),
        "sfh_tsnorm_burst_burst_sfr": ("burst_sfr", 1.0, 0.0),
        "sfh_tsnorm_burst_burst_age_gyr": ("burst_age", 1e9, 0.0),
    },
    composition_type="additive",
)
_register(
    _tsnorm_burst_spec,
    citation="Robotham et al. 2020 (arXiv:2002.06980) ProSpect",
    short_doc="Truncated skew-normal + flat burst (ProSpect)",
)
# --- norm (Gaussian) ---
_norm_spec = SFHModelSpec(
    name="norm",
    fn=norm,
    params={
        "sfh_norm_log_total_mass": ParamDef(
            "log10 total stellar mass formed [Msun]",
            _always_true,
            "",
            Uniform(7.0, 12.5, default=10.0),
        ),
        "sfh_norm_peak_lbt_gyr": ParamDef(
            "Peak lookback time (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.5, 12.0, default=5.0),
        ),
        "sfh_norm_width_gyr": ParamDef(
            "Gaussian width (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.2, 5.0, default=1.0),
        ),
    },
    settings={},
    internal_param_map={
        "sfh_norm_log_total_mass": ("log_total_mass", 1.0, 0.0),
        "sfh_norm_peak_lbt_gyr": ("peak_lbt", 1e9, 0.0),
        "sfh_norm_width_gyr": ("width", 1e9, 0.0),
    },
    composition_type="additive",
)
_register(_norm_spec, short_doc="Gaussian SFH")

# --- lnorm (log-normal) ---
_lnorm_spec = SFHModelSpec(
    name="lnorm",
    fn=lnorm,
    params={
        "sfh_lnorm_log_total_mass": ParamDef(
            "log10 total stellar mass formed [Msun]",
            _always_true,
            "",
            Uniform(7.0, 12.5, default=10.0),
        ),
        "sfh_lnorm_peak_gyr": ParamDef(
            "Peak in cosmic time since formation (Gyr)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.5, 12.0, default=5.0),
        ),
        "sfh_lnorm_width_gyr": ParamDef(
            "Log-space width (dex)",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.1, 2.0, default=0.5),
        ),
        "sfh_lnorm_age_gyr": ParamDef(
            "Cosmic time for SF (Gyr) = lookback of formation; "
            "set to age_of_universe(z) for BAGPIPES direction",
            _lo_positive,
            "must have lo > 0",
            Uniform(0.5, _AGE_UNIV_GYR, default=_AGE_UNIV_GYR),
        ),
    },
    settings={},
    internal_param_map={
        "sfh_lnorm_log_total_mass": ("log_total_mass", 1.0, 0.0),
        "sfh_lnorm_peak_gyr": ("peak", 1e9, 0.0),
        "sfh_lnorm_width_gyr": ("width", 1.0, 0.0),  # already in dex
        "sfh_lnorm_age_gyr": ("age", 1e9, 0.0),
    },
    composition_type="additive",
)
_register(_lnorm_spec, short_doc="Log-normal SFH")

# --- dpl (double power law) ---
_register(
    SFHModelSpec(
        name="dpl",
        fn=dpl,
        params={
            # DPL defaults from Carnall+2018 (MNRAS 480, 4379) fiducial galaxy:
            # α = falling slope ~ 1.5; β = rising ~ 1.0; τ = 3 Gyr turnover.
            "sfh_dpl_alpha": ParamDef(
                "DPL falling slope",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 5.0, default=1.5),
            ),
            "sfh_dpl_beta": ParamDef(
                "DPL rising slope",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 3.0, default=1.0),
            ),
            "sfh_dpl_tau_gyr": ParamDef(
                "DPL turnover time (Gyr), cosmic time since formation",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 12.0, default=3.0),
            ),
            "sfh_dpl_age_gyr": ParamDef(
                "Cosmic time for SF (Gyr) = lookback of formation; "
                "set to age_of_universe(z) for BAGPIPES parity",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, _AGE_UNIV_GYR, default=_AGE_UNIV_GYR),
            ),
            "sfh_dpl_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun]",
                _always_true,
                "",
                # Typical galaxy: M_⋆ ~ 10^10 M_⊙.
                Uniform(7.0, 12.5, default=10.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_dpl_alpha": ("alpha", 1.0, 0.0),
            "sfh_dpl_beta": ("beta", 1.0, 0.0),
            "sfh_dpl_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_dpl_age_gyr": ("age", 1e9, 0.0),
            "sfh_dpl_log_total_mass": ("log_total_mass", 1.0, 0.0),
        },
        composition_type="additive",
    ),
    citation="Carnall et al. 2018 (MNRAS 480, 4379)",
    short_doc="Double power-law SFH",
)

# --- const (constant) ---
_register(
    SFHModelSpec(
        name="const",
        fn=constant,
        params={
            "sfh_const_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun]",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_const_start_gyr": ParamDef(
                "Lookback to SF onset (Gyr): when did SF start?",
                _lo_positive,
                "must have lo > 0",
                Fixed(AGEMAX_YR / 1e9),
                # No free_prior, for the redshift-dependence reason given on
                # ``sfh_exp_start_gyr`` below, which applies to every SF-onset
                # lookback: the ceiling is the age of the universe at the source
                # redshift and the declaration cannot know it.
            ),
            "sfh_const_end_gyr": ParamDef(
                "Lookback to SF cessation (Gyr): when did SF stop? (0 = ongoing)",
                _lo_nonneg,
                "must have lo >= 0",
                Fixed(0.0),
                # Deliberately NO free_prior, because of the ordering constraint
                # with ``sfh_const_start_gyr`` above. These two are lookback
                # times bracketing the SF episode, so start_gyr (onset) must
                # exceed end_gyr (cessation), and ``Parameters._validate_orderings``
                # rejects any pair whose supports overlap -- the overlap region
                # is a zero-mass galaxy with an exactly-zero gradient that a
                # gradient sampler cannot escape. Freeing both from a wildcard
                # would need an arbitrary split of the age axis between them.
                # So the wildcard frees the onset and leaves cessation at 0
                # ("still forming stars"), which is the common case; free it
                # explicitly with a prior that stays below your onset's floor,
                # e.g. sfh={'const_start_gyr': Uniform(8, 14),
                #           'const_end_gyr': Uniform(0, 6)}.
            ),
        },
        settings={},
        internal_param_map={
            "sfh_const_log_total_mass": ("log_total_mass", 1.0, 0.0),
            # User's "start" (when SF began) = older lookback = internal "end"
            # User's "end" (when SF stopped) = younger lookback = internal "start"
            "sfh_const_start_gyr": ("end", 1e9, 0.0),
            "sfh_const_end_gyr": ("start", 1e9, 0.0),
        },
        composition_type="additive",
    ),
    short_doc="Constant SFR",
)

# --- exp (exponential) ---
_register(
    SFHModelSpec(
        name="exp",
        fn=exponential,
        params={
            "sfh_exp_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun]",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_exp_tau_gyr": ParamDef(
                "e-folding timescale (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 10.0, default=2.0),
            ),
            # Deliberately NO free_prior (#887), and this covers the ``dexp`` and
            # ``const`` onsets too. ``start`` is a lookback: these SFHs form
            # stars only at ``t_lookback >= start``, so the parameter's ceiling
            # is the age of the universe at the SOURCE redshift -- 8.6 Gyr at
            # z=0.5, 3.3 at z=2, 0.9 at z=6. A declaration cannot know that, and
            # no static interval is right for all of them: any bound generous
            # enough for z~0 admits draws at z=2 where star formation never
            # happens, giving a zero-mass galaxy and zero flux.
            #
            # Measured, not argued: declaring Uniform(0, 14) made
            # test_bug_1031_dense_basis_composite::
            # test_working_sfh_topologies_still_predict[dexp] draw such a value
            # at z=0.5 and fail `assert jnp.all(flux > 0)`.
            #
            # Free it explicitly against your own redshift, e.g.
            # sfh={'start_gyr': Uniform(0, 6)} for a z=1 target.
            "sfh_exp_start_gyr": ParamDef(
                "Start lookback (Gyr)", _lo_nonneg, "must have lo >= 0", Fixed(0.0)
            ),
        },
        settings={},
        internal_param_map={
            "sfh_exp_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_exp_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_exp_start_gyr": ("start", 1e9, 0.0),
        },
        composition_type="additive",
    ),
    short_doc="Exponential rise SFH",
)

# --- dexp (delayed exponential) ---
_register(
    SFHModelSpec(
        name="dexp",
        fn=delayed_exponential,
        params={
            "sfh_dexp_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun]",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_dexp_tau_gyr": ParamDef(
                "Timescale (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 10.0, default=2.0),
            ),
            # No free_prior -- see the shared note on ``sfh_exp_start_gyr``.
            "sfh_dexp_start_gyr": ParamDef(
                "Start lookback (Gyr)", _lo_nonneg, "must have lo >= 0", Fixed(0.0)
            ),
        },
        settings={},
        internal_param_map={
            "sfh_dexp_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_dexp_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_dexp_start_gyr": ("start", 1e9, 0.0),
        },
        composition_type="additive",
    ),
    short_doc="Delayed exponential SFH",
)

# --- declining_exp (FSPS sfh=1 / Bagpipes 'exponential') ---
# SFR(T) ∝ exp(-T/τ) with T = age - t_lb: maximal at formation, declining to
# the present. The classic declining-tau model, and the parametric SFH most
# often quoted in the literature.
#
# Registered as ``declining_exp``, NOT as ``tau`` (#1750). It was previously
# registered as ``tau``, and #406 removed it because users read that name as
# CIGALE's ``sfhdelayed`` — which is τ-*delayed* and rises from zero — giving a
# silent wavelength-dependent residual. Both models have a τ, so ``tau`` cannot
# distinguish them and putting that name back would reinstate the defect. What
# #406 actually established is that the *name* was wrong, not that the model
# should be unreachable: leaving it importable-but-unselectable made
# ``SEDModel.build`` unable to express FSPS ``sfh=1`` at all, and the
# FSPS/Bagpipes tau-model parity comparison — the most directly meaningful
# external check available for a parametric SFH — had no model to run against.
#
# ``declining_exp`` states the shape, matches the ``const_exp`` / ``sfh2exp``
# naming already in this file, and gives ``sfh_declining_exp_tau_gyr`` rather
# than the ``sfh_tau_tau_gyr`` that spelling ``tau`` would have forced.
_register(
    SFHModelSpec(
        name="declining_exp",
        fn=declining_exponential,
        params={
            "sfh_declining_exp_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun]",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_declining_exp_tau_gyr": ParamDef(
                "e-folding decline timescale (Gyr) — larger τ declines slower",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 10.0, default=2.0),
            ),
            "sfh_declining_exp_age_gyr": ParamDef(
                "Galaxy age / lookback time of formation (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 13.0, default=5.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_declining_exp_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_declining_exp_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_declining_exp_age_gyr": ("age", 1e9, 0.0),
        },
        composition_type="additive",
    ),
    short_doc="Declining-τ SFH (FSPS sfh=1 / Bagpipes 'exponential')",
    citation="Conroy et al. 2009 (FSPS); Carnall et al. 2018 (Bagpipes)",
)

# --- delayed (τ-delayed, matches CIGALE sfh_delayed / Bagpipes 'delayed') ---
# SFR(T) ∝ T · exp(-T/τ) with T = age - t_lb. Rises from 0 at formation,
# peaks at cosmic-time τ-after-formation (lookback age − τ), declines to
# present. Distinct from ``tau`` above — see #406 for the audit that
# surfaced the convention mismatch.
_register(
    SFHModelSpec(
        name="delayed",
        fn=sfhdelayed,
        params={
            "sfh_delayed_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun]",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_delayed_tau_gyr": ParamDef(
                "Timescale (Gyr) — cosmic-time location of SFR peak",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 10.0, default=2.0),
            ),
            "sfh_delayed_age_gyr": ParamDef(
                "Galaxy age / lookback time of formation (Gyr); must be > τ",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 13.0, default=5.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_delayed_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_delayed_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_delayed_age_gyr": ("age", 1e9, 0.0),
        },
        composition_type="additive",
    ),
    short_doc="τ-delayed SFH (CIGALE sfh_delayed / Bagpipes 'delayed')",
    citation="Boquien et al. 2019 (CIGALE); Carnall et al. 2018 (Bagpipes)",
)

# --- const_exp (constant + exponential decline — "quenching at time T") ---
_register(
    SFHModelSpec(
        name="const_exp",
        fn=constant_then_exponential,
        params={
            "sfh_cexp_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun] (constant + decline phases)",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_cexp_tau_gyr": ParamDef(
                "Post-quench e-folding timescale (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 10.0, default=2.0),
            ),
            "sfh_cexp_quench_gyr": ParamDef(
                "Lookback time when quenching began (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 10.0, default=1.0),
            ),
            "sfh_cexp_age_gyr": ParamDef(
                "Galaxy age / lookback to formation (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 13.0, default=5.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_cexp_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_cexp_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_cexp_quench_gyr": ("quench_age", 1e9, 0.0),
            "sfh_cexp_age_gyr": ("age", 1e9, 0.0),
        },
        composition_type="additive",
    ),
    short_doc="Constant SFR then exponential quenching",
)
SFH_REGISTRY["constant_then_exponential"] = SFH_REGISTRY["const_exp"]


# --- sfh2exp (double declining exponential: main + recent burst, CIGALE) ---
_register(
    SFHModelSpec(
        name="sfh2exp",
        fn=sfh2exp,
        params={
            "sfh_sfh2exp_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun]",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_sfh2exp_tau_main_gyr": ParamDef(
                "e-folding timescale of main population (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 10.0, default=2.0),
            ),
            "sfh_sfh2exp_tau_burst_gyr": ParamDef(
                "e-folding timescale of the burst (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 1.0, default=0.05),
            ),
            "sfh_sfh2exp_f_burst": ParamDef(
                "Fraction of stellar mass formed in the burst",
                lambda lo, hi: lo >= 0.0 and hi < 1.0,
                "must have 0 <= f_burst < 1",
                Uniform(0.0, 0.5, default=0.0),
            ),
            "sfh_sfh2exp_age_gyr": ParamDef(
                "Age of main population / lookback to formation (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 13.0, default=5.0),
            ),
            "sfh_sfh2exp_burst_age_gyr": ParamDef(
                "Lookback time of burst onset (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 2.0, default=0.02),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_sfh2exp_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_sfh2exp_tau_main_gyr": ("tau_main_yr", 1e9, 0.0),
            "sfh_sfh2exp_tau_burst_gyr": ("tau_burst_yr", 1e9, 0.0),
            "sfh_sfh2exp_f_burst": ("f_burst", 1.0, 0.0),
            "sfh_sfh2exp_age_gyr": ("age_yr", 1e9, 0.0),
            "sfh_sfh2exp_burst_age_gyr": ("burst_age_yr", 1e9, 0.0),
        },
        composition_type="additive",
    ),
    short_doc="Double declining exponential: main + recent burst (CIGALE sfh2exp)",
)


# --- delayed_bq (delayed-tau with burst/quench, Ciesla+2017) ---
_register(
    SFHModelSpec(
        name="delayed_bq",
        fn=delayed_bq,
        params={
            "sfh_delayed_bq_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun]",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_delayed_bq_tau_main_gyr": ParamDef(
                "e-folding timescale of main component (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 10.0, default=2.0),
            ),
            "sfh_delayed_bq_age_main_gyr": ParamDef(
                "Galaxy age / lookback to formation (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 13.0, default=5.0),
            ),
            "sfh_delayed_bq_age_bq_gyr": ParamDef(
                "Lookback time of burst/quench onset (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 5.0, default=0.5),
            ),
            "sfh_delayed_bq_r_sfr": ParamDef(
                "SFR ratio after/before burst/quench",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 10.0, default=1.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_delayed_bq_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_delayed_bq_tau_main_gyr": ("tau_main_yr", 1e9, 0.0),
            "sfh_delayed_bq_age_main_gyr": ("age_main_yr", 1e9, 0.0),
            "sfh_delayed_bq_age_bq_gyr": ("age_bq_yr", 1e9, 0.0),
            "sfh_delayed_bq_r_sfr": ("r_sfr", 1.0, 0.0),
        },
        composition_type="additive",
    ),
    short_doc="Delayed tau with burst/quench (Ciesla et al.)",
)


# --- periodic (periodic SF events, Ciesla+2017) ---
_register(
    SFHModelSpec(
        name="periodic",
        fn=periodic,
        params={
            "sfh_periodic_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun]",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_periodic_delta_bursts_gyr": ParamDef(
                "Spacing between burst onsets (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 1.0, default=0.1),
            ),
            "sfh_periodic_tau_bursts_gyr": ParamDef(
                "Duration/e-folding timescale of each burst (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.001, 0.5, default=0.02),
            ),
            # Deliberately NO free_prior, and this one is not a judgment call:
            # the validator itself demands ``int(lo) == lo``. The parameter
            # selects a burst *shape* from three discrete alternatives, so a
            # continuous prior over [0, 2] would spend almost all its mass on
            # values that name no model at all. Structural choices belong in the
            # grammar (``sfh={'burst_type': ...}``), not in the sampler.
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
                Uniform(0.5, 13.0, default=5.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_periodic_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_periodic_delta_bursts_gyr": ("delta_bursts_yr", 1e9, 0.0),
            "sfh_periodic_tau_bursts_gyr": ("tau_bursts_yr", 1e9, 0.0),
            "sfh_periodic_burst_type": ("burst_type", 1.0, 0.0),
            "sfh_periodic_age_gyr": ("age_yr", 1e9, 0.0),
        },
        composition_type="additive",
    ),
    short_doc="Periodic SF events (Ciesla et al.)",
)


# --- buat08 (velocity-parameterized SFH, Buat+2008) ---
_register(
    SFHModelSpec(
        name="buat08",
        fn=buat08,
        params={
            "sfh_buat08_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun]",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_buat08_velocity_km_s": ParamDef(
                "Rotational velocity (km/s), range [40, 360]",
                lambda lo, hi: lo >= 40 and hi <= 360,
                "must have 40 <= lo and hi <= 360",
                Uniform(80.0, 360.0, default=200.0),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_buat08_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_buat08_velocity_km_s": ("velocity_km_s", 1.0, 0.0),
        },
        composition_type="additive",
    ),
    citation="Buat et al. 2008",
    short_doc="Velocity-parameterized SFH (Buat et al.)",
)


# --- psb (post-starburst, Wild+2020) ---
_register(
    SFHModelSpec(
        name="psb",
        fn=psb_wild2020,
        params={
            "sfh_psb_log_total_mass": ParamDef(
                "log10 total stellar mass formed [Msun]",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_psb_age_gyr": ParamDef(
                "Galaxy age / lookback to formation (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 13.0, default=5.0),
            ),
            "sfh_psb_tau_gyr": ParamDef(
                "Old-component e-folding timescale (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 10.0, default=2.0),
            ),
            "sfh_psb_burstage_gyr": ParamDef(
                "Lookback time of burst onset (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 5.0, default=0.5),
            ),
            "sfh_psb_alpha": ParamDef(
                "DPL burst falling slope",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 5.0, default=2.0),
            ),
            "sfh_psb_beta": ParamDef(
                "DPL burst rising slope",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 5.0, default=2.0),
            ),
            "sfh_psb_fburst": ParamDef(
                "Burst mass fraction",
                lambda lo, hi: lo >= 0 and hi <= 1,
                "must be in [0, 1]",
                Uniform(0.01, 0.99, default=0.3),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_psb_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_psb_age_gyr": ("age", 1e9, 0.0),
            "sfh_psb_tau_gyr": ("tau", 1e9, 0.0),
            "sfh_psb_burstage_gyr": ("burstage", 1e9, 0.0),
            "sfh_psb_alpha": ("alpha", 1.0, 0.0),
            "sfh_psb_beta": ("beta", 1.0, 0.0),
            "sfh_psb_fburst": ("fburst", 1.0, 0.0),
        },
        composition_type="additive",
    ),
    citation="Wild et al. 2020 (MNRAS 494, 529)",
    short_doc="Post-starburst SFH (Wild et al.)",
)
SFH_REGISTRY["psb_wild2020"] = SFH_REGISTRY["psb"]


# --- top_hat (constant-window with smooth sigmoid edges) ---
_register(
    SFHModelSpec(
        name="top_hat",
        fn=top_hat,
        params={
            "sfh_top_hat_log_total_mass": ParamDef(
                "log10 total stellar mass formed in window [Msun]",
                _always_true,
                "",
                Uniform(7.0, 12.5, default=10.0),
            ),
            "sfh_top_hat_t_start_gyr": ParamDef(
                "Older lookback boundary / SF onset (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.1, 13.0, default=5.0),
            ),
            "sfh_top_hat_t_end_gyr": ParamDef(
                "Younger lookback boundary / SF cessation (Gyr)",
                _lo_nonneg,
                "must have lo >= 0",
                Uniform(0.0, 12.0, default=4.0),
            ),
            "sfh_top_hat_smooth_width_gyr": ParamDef(
                "Sigmoid transition width (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Fixed(0.1),
                # How sharply the top hat turns on and off. Below ~0.01 Gyr the
                # sigmoid is a step at any realistic SSP age resolution; above
                # ~2 Gyr the "top hat" has smoothed into the underlying SFH and
                # the shape stops being a top hat at all.
                Uniform(0.01, 2.0, "Top-hat transition width", units="Gyr", default=0.1),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_top_hat_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_top_hat_t_start_gyr": ("t_start", 1e9, 0.0),
            "sfh_top_hat_t_end_gyr": ("t_end", 1e9, 0.0),
            "sfh_top_hat_smooth_width_gyr": ("smooth_width", 1e9, 0.0),
        },
        composition_type="additive",
    ),
    short_doc="Constant-SFR window with smooth edges (top-hat)",
)

# --- gaussian_burst (Gaussian-in-age burst, Robotham+2020) ---
_register(
    SFHModelSpec(
        name="gaussian_burst",
        fn=gaussian_burst,
        params={
            "sfh_gaussian_burst_log_total_mass": ParamDef(
                "log10 total stellar mass formed in burst [Msun]",
                _always_true,
                "",
                Uniform(6.0, 11.0, default=10.0),
            ),
            "sfh_gaussian_burst_t_peak_gyr": ParamDef(
                "Burst peak age / lookback time (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 13.0, default=0.5),
            ),
            "sfh_gaussian_burst_sigma_gyr": ParamDef(
                "Gaussian width / standard deviation (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 5.0, default=0.1),
            ),
        },
        settings={},
        internal_param_map={
            "sfh_gaussian_burst_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_gaussian_burst_t_peak_gyr": ("t_peak", 1e9, 0.0),
            "sfh_gaussian_burst_sigma_gyr": ("sigma", 1e9, 0.0),
        },
        composition_type="additive",
    ),
    citation="Robotham et al. 2020 (arXiv:2002.06980) ProSpect",
    short_doc="Gaussian-in-age burst (Robotham+2020)",
)


# ── Register tabulated SFH model (for simulations) ────────────────


def _table_sfh_placeholder(t_lookback, **kwargs):
    """Placeholder — tabulated SFH is handled directly in the orchestrator path.

    Parameters
    ----------
    t_lookback : array_like, shape (n_age,)
        Lookback time [yr].
    **kwargs
        Unused; for registry compatibility.

    Returns
    -------
    ndarray, shape (n_age,)
        Zero array (the runtime table is wired in StellarSEDComponent, #996).

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
    ),
    short_doc="User-supplied tabulated SFH(t)",
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
                Uniform(8.0, 12.0, default=10.0),
            ),
            **{
                f"sfh_cont_ratio_{i}": ParamDef(
                    f"log10 SFR ratio bin {i}/{i + 1}",
                    _always_true,
                    "",
                    StudentT(mu=0.0, sigma=0.3, df=2.0, default=0.0),
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
    ),
    citation="Leja et al. 2019 (ApJ 876, 39)",
    short_doc=(
        "Non-parametric piecewise continuity SFH (Leja+19); "
        "StudentT(0, 0.3, df=2) on log-SFR ratios"
    ),
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
                Uniform(8.0, 12.0, default=10.0),
            ),
            "sfh_cflex_ratio_young": ParamDef(
                "log10(SFR_young / SFR_flex[0])",
                _always_true,
                "",
                StudentT(mu=0.0, sigma=0.3, df=2.0, default=0.0),
            ),
            **{
                f"sfh_cflex_flex_{i}": ParamDef(
                    f"log10 flex bin SFR ratio {i} (controls bin width)",
                    _always_true,
                    "",
                    StudentT(mu=0.0, sigma=0.3, df=2.0, default=0.0),
                )
                for i in range(_N_CFLEX)
            },
            "sfh_cflex_ratio_old": ParamDef(
                "log10(SFR_old / SFR_flex[N])",
                _always_true,
                "",
                StudentT(mu=0.0, sigma=0.3, df=2.0, default=0.0),
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
    ),
    citation="Leja et al. 2019 (ApJ 876, 39)",
    short_doc=(
        "Non-parametric continuity + flexible bin edges (Leja+19); "
        "StudentT(0, 0.3, df=2) on log-SFR ratios"
    ),
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
                Uniform(8.0, 12.0, default=10.0),
            ),
            **{
                f"sfh_dir_z_{i}": ParamDef(
                    f"Dirichlet stick-breaking variable {i}",
                    lambda lo, hi: lo >= 0 and hi <= 1,
                    "must be in [0, 1]",
                    # Beta(1, 1) is exactly Uniform(0, 1); faithful Leja+2017
                    # symmetric Dirichlet(1,...,1) marginal on mass fractions.
                    Uniform(0.0, 1.0, default=0.5),
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
    ),
    citation="Leja et al. 2017 (ApJ 837, 170)",
    short_doc=(
        "Non-parametric Dirichlet SFH (Leja+17); "
        "Beta(1,1) = Uniform(0,1) stick-breaking aux variables"
    ),
)


# --- bursty_continuity (Tacchella+2022): bin-edge-dependent Student-t scale ---
# Same shape function as `continuity`, but per-ratio sigma toggles between
# scale_young = 1.0 dex (when the younger edge bin_edges_gyr[i+1] < t_split_gyr)
# and scale_old = 0.3 dex otherwise. For the default 7-bin grid
# DEFAULT_BIN_EDGES_GYR = [0, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0, 13.7] Gyr and
# t_split = 1.0 Gyr, this gives sigma = [1.0, 1.0, 1.0, 0.3, 0.3, 0.3] for
# ratios 0..5. See bursty_continuity_prior_logp in nonparametric.py.
_BURSTY_T_SPLIT_GYR = 1.0
_BURSTY_SCALE_YOUNG = 1.0
_BURSTY_SCALE_OLD = 0.3
_BURSTY_DEFAULT_SIGMAS = [
    _BURSTY_SCALE_YOUNG
    if float(DEFAULT_BIN_EDGES_GYR[i + 1]) < _BURSTY_T_SPLIT_GYR
    else _BURSTY_SCALE_OLD
    for i in range(6)
]
_register(
    SFHModelSpec(
        name="bursty_continuity",
        fn=continuity,
        params={
            "sfh_burstcont_log_total_mass": ParamDef(
                "log10 total stellar mass formed (Msun)",
                _always_true,
                "",
                Uniform(8.0, 12.0, default=10.0),
            ),
            **{
                f"sfh_burstcont_ratio_{i}": ParamDef(
                    f"log10 SFR ratio bin {i}/{i + 1} (Tacchella+22 piecewise scale)",
                    _always_true,
                    "",
                    StudentT(mu=0.0, sigma=_BURSTY_DEFAULT_SIGMAS[i], df=2.0, default=0.0),
                )
                for i in range(6)
            },
        },
        settings={
            "sfh_burstcont_t_split_gyr": _BURSTY_T_SPLIT_GYR,
            "sfh_burstcont_scale_young": _BURSTY_SCALE_YOUNG,
            "sfh_burstcont_scale_old": _BURSTY_SCALE_OLD,
        },
        internal_param_map={
            "sfh_burstcont_log_total_mass": ("log_total_mass", 1.0, 0.0),
            **{f"sfh_burstcont_ratio_{i}": (f"ratio_{i}", 1.0, 0.0) for i in range(6)},
        },
        composition_type="additive",
    ),
    citation="Tacchella et al. 2022 (ApJ 926, 134); arXiv:2102.11954",
    short_doc=(
        "Bursty continuity SFH (Tacchella+22); StudentT df=2 with sigma=1.0 dex on "
        "ratios whose younger edge < 1 Gyr and sigma=0.3 dex otherwise"
    ),
)


# --- psb_suess2022 (Suess+2022): post-starburst nonparametric SFH ---
# Distinct from the existing `psb_wild2020` (Wilkinson+2020 parametric).
# Free params: log_total_mass, tlast_gyr (quenching epoch), tflex_gyr (upper
# bound of the flex zone), ratio_young (youngest-vs-flex SFR ratio), and
# ratio_old_0..N-2 for the fixed old bins. Defaults match Suess+2022: tlast
# in [0.01, 1.0] Gyr, tflex in [0.5, 5.0] Gyr, StudentT(0, 0.3, 2) on ratios.
# Default fixed old bins = DEFAULT_BIN_EDGES_GYR[2:] = [0.3, 1.0, 3.0, 6.0, 13.7]
# -> 4 old bins -> 3 ratio_old_* parameters.
_N_PSB_OLD_RATIOS = 3
_register(
    SFHModelSpec(
        name="psb_suess2022",
        fn=psb_continuity,
        params={
            "sfh_psb2022_log_total_mass": ParamDef(
                "log10 total stellar mass formed (Msun)",
                _always_true,
                "",
                Uniform(8.0, 12.0, default=10.0),
            ),
            "sfh_psb2022_tlast_gyr": ParamDef(
                "Quenching-onset lookback time (Gyr); width of the youngest bin",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.01, 1.0, default=0.1),
            ),
            "sfh_psb2022_tflex_gyr": ParamDef(
                "Upper boundary of the flexible quenching zone (Gyr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(0.5, 5.0, default=2.0),
            ),
            "sfh_psb2022_ratio_young": ParamDef(
                "log10(SFR_young / SFR_flex); large positive = recent burst",
                _always_true,
                "",
                StudentT(mu=0.0, sigma=0.3, df=2.0, default=0.0),
            ),
            **{
                f"sfh_psb2022_ratio_old_{i}": ParamDef(
                    f"log10 SFR ratio old bin {i}/{i + 1}",
                    _always_true,
                    "",
                    StudentT(mu=0.0, sigma=0.3, df=2.0, default=0.0),
                )
                for i in range(_N_PSB_OLD_RATIOS)
            },
        },
        settings={},
        internal_param_map={
            "sfh_psb2022_log_total_mass": ("log_total_mass", 1.0, 0.0),
            "sfh_psb2022_tlast_gyr": ("tlast_gyr", 1.0, 0.0),
            "sfh_psb2022_tflex_gyr": ("tflex_gyr", 1.0, 0.0),
            "sfh_psb2022_ratio_young": ("ratio_young", 1.0, 0.0),
            **{
                f"sfh_psb2022_ratio_old_{i}": (f"ratio_old_{i}", 1.0, 0.0)
                for i in range(_N_PSB_OLD_RATIOS)
            },
        },
        composition_type="additive",
    ),
    citation="Suess et al. 2022 (ApJ 935, 146); arXiv:2207.05895",
    short_doc=(
        "Post-starburst non-parametric SFH (Suess+22): youngest bin [0, tlast] + "
        "flex zone [tlast, tflex] + fixed old bins, with StudentT(0, 0.3, df=2) ratios"
    ),
)


# --- prospector_beta (Wang+2024): redshift-aware continuity SFH ---
# Same shape and ratio prior as `continuity`; differs by using
# redshift-dependent bin edges produced by `make_agebins_from_zred(zred)`.
# Edges are passed via the composer's `bin_edges_gyr` argument at build
# time (call `make_agebins_from_zred` in your recipe). The joint mass-z-SFR
# hierarchical prior from Wang+2023 is out of scope here.
_register(
    SFHModelSpec(
        name="prospector_beta",
        fn=continuity,
        params={
            "sfh_pbeta_log_total_mass": ParamDef(
                "log10 total stellar mass formed (Msun)",
                _always_true,
                "",
                Uniform(8.0, 12.0, default=10.0),
            ),
            **{
                f"sfh_pbeta_ratio_{i}": ParamDef(
                    f"log10 SFR ratio bin {i}/{i + 1}",
                    _always_true,
                    "",
                    StudentT(mu=0.0, sigma=0.3, df=2.0, default=0.0),
                )
                for i in range(6)
            },
        },
        settings={},
        internal_param_map={
            "sfh_pbeta_log_total_mass": ("log_total_mass", 1.0, 0.0),
            **{f"sfh_pbeta_ratio_{i}": (f"ratio_{i}", 1.0, 0.0) for i in range(6)},
        },
        composition_type="additive",
    ),
    citation="Wang et al. 2024 (arXiv:2401.12198); SFH shape from Leja+2019",
    short_doc=(
        "Prospector-beta: continuity SFH with redshift-aware bin edges from "
        "make_agebins_from_zred(zred); StudentT(0, 0.3, df=2) on log-SFR ratios"
    ),
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
                Uniform(8.0, 12.0, default=10.0),
            ),
            "sfh_db_log_sfr_inst": ParamDef(
                "log10 instantaneous SFR at observation (Msun/yr)",
                _always_true,
                "",
                Uniform(-2.0, 3.0, default=0.0),
            ),
            **{
                f"sfh_db_tx_frac_{i}": ParamDef(
                    f"Cosmic time fraction at {(i + 1) * 25}% mass",
                    lambda lo, hi: lo >= 0 and hi <= 1,
                    "must be in [0, 1]",
                    Uniform(0.05, 0.95, default=0.5),
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
    ),
    short_doc="Dense-basis non-parametric SFH (Iyer+19)",
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
                Uniform(8.0, 12.0, default=10.0),
            ),
            **{
                f"sfh_dbp_tx_frac_{i}": ParamDef(
                    f"Cosmic time fraction at {(i + 1) * 25}% mass",
                    lambda lo, hi: lo >= 0 and hi <= 1,
                    "must be in [0, 1]",
                    Uniform(0.05, 0.95, default=0.5),
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
    ),
    short_doc="Dense-basis SFH without GP constraints",
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
                Uniform(-3.0, -0.1, default=-1.0),
            ),
            "sfh_burst_log_tpeak_myr": ParamDef(
                "log10 burst peak time (Myr)", _always_true, "", Uniform(0.0, 3.0, default=2.0)
            ),
            "sfh_burst_log_tmax_myr": ParamDef(
                "log10 burst duration (Myr)", _always_true, "", Uniform(1.0, 4.0, default=2.5)
            ),
        },
        settings={},
        internal_param_map={
            "sfh_burst_log_fburst": ("log_fburst", 1.0, 0.0),
            "sfh_burst_log_tpeak_myr": ("log_tpeak_myr", 1.0, 0.0),
            "sfh_burst_log_tmax_myr": ("log_tmax_myr", 1.0, 0.0),
        },
        composition_type="mixture",
    ),
    short_doc="Triweight burst kernel (Zacharegkas+2025)",
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
            # Stochastic SFH (DRW correlated field).
            #
            # THE PRIORS BELOW ARE DELIBERATELY UNIFORM AND UNINFORMATIVE. The
            # literature supports a narrower, measured prior on tau (see below);
            # uniform is kept so the data, not the prior, drives the posterior,
            # and so a recovered value can be compared against the measurement
            # rather than assuming it. Both options are documented here so the
            # choice stays visible.
            #
            # What a DRW is, in this literature's terms: its PSD is
            # 1/(1 + (2 pi f tau)^2) — slope 2 above the break, flat below. That
            # is exactly the alpha = 2 single-break case of Caplar & Tacchella
            # [1]_, so the model family matches their assumed form rather than
            # approximating it. It CANNOT represent the three-break PSD of
            # Tacchella, Forbes & Caplar [2]_ (inflow correlation time, gas
            # equilibrium timescale, molecular cloud lifetime); that needs a
            # flexible per-timescale PSD, not a single (sigma, tau).
            #
            # tau — MEASURED VALUE, not what is set here. Caplar & Tacchella [1]_
            # infer tau_break = 178 (+104, -66) Myr assuming alpha = 2, i.e.
            # galaxies lose memory of prior activity on ~200 Myr. The default of
            # 100 Myr below is LOW against that, and Uniform(10, 500) places most
            # of its mass above it. A measured-centered alternative would be a
            # lognormal about 178 Myr carrying the asymmetric +104/-66 spread.
            #
            # sigma — no PSD amplitude in dex is quoted by either reference; the
            # 0.3 dex default is consistent with the observed star-forming
            # main-sequence scatter (~0.2-0.4 dex) that [1]_ models.
            #
            # Recent context, NOT verified to this level of detail: Kravtsov &
            # Belokurov, "Stochastic star formation and the abundance of z>10
            # UV-bright galaxies", arXiv:2405.04578 (2024), require SFR
            # stochasticity rising with redshift (scatter in M_UV of ~0.75-2.0
            # mag) to reproduce the z>10 UV luminosity function. Their PSD form
            # and timescale were not confirmed against the text and must be read
            # from the paper before being cited for a number.
            #
            # References
            # ----------
            # .. [1] N. Caplar & S. Tacchella, "Stochastic modeling of
            #    star-formation histories I: the scatter of the star-forming main
            #    sequence", MNRAS, 487, 3845 (2019). arXiv:1901.07556.
            # .. [2] S. Tacchella, J. C. Forbes & N. Caplar, "Stochastic
            #    modelling of star-formation histories II: star-formation
            #    variability from molecular clouds and gas inflow", MNRAS (2020).
            #    DOI: 10.1093/mnras/staa1838. arXiv:2006.09382.
            # .. [3] K. G. Iyer et al., "The star formation history and
            #    variability of galaxies", MNRAS, 498, 430 (2020). [EAGLE
            #    decorrelation timescale; the source of the 100 Myr default]
            "sfh_field_psd_sigma": ParamDef(
                "PSD amplitude (dex)",
                _lo_nonneg,
                "must have lo >= 0",
                Uniform(0.01, 1.0, default=0.3),
            ),
            "sfh_field_psd_tau_myr": ParamDef(
                "PSD timescale (Myr)",
                _lo_positive,
                "must have lo > 0",
                Uniform(10.0, 500.0, default=100.0),
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
    ),
    short_doc=(
        "GP field modulator with DRW power spectrum; sampling coordinates "
        "selectable via sfh={'field_centering': a}"
    ),
)


# ── Composition: resolve_sfh() ────────────────────────────────────

#: ``dense_basis`` carries SFR-constraint points that pin the recent SFH, which
#: fights both the GP field modulator and the triweight burst kernel
#: (Zacharegkas+2025). Composing with either swaps in the quantile-only variant.
#: Its public parameters are prefixed ``sfh_dbp_*``, not ``sfh_db_*``.
_DB_TO_PURE: dict[str, str] = {"dense_basis": "dense_basis_pure", "db": "dbp"}

#: SFH names that compose with (rather than replace) the smooth model.
_COMPOSITORS: frozenset[str] = frozenset({"field", "burst"})


def apply_compositor_swap(names: list[str]) -> list[str]:
    """Apply the ``dense_basis`` → ``dense_basis_pure`` auto-swap (#1074).

    Every consumer that resolves an SFH name to a spec must apply this — the
    swap renames the public parameters (``sfh_db_*`` → ``sfh_dbp_*``), so a
    consumer that skips it looks up the wrong spec, fails to find the user's
    parameters, and silently substitutes registry defaults. That is exactly how
    ``tx_frac_*`` became a no-op in the composite forward model (#1074).

    Parameters
    ----------
    names : list of str
        Requested SFH model names, e.g. ``["dense_basis", "field"]``.

    Returns
    -------
    list of str
        The names with ``dense_basis`` swapped for its pure variant when a
        compositor (``field`` or ``burst``) is present. Unchanged otherwise.
    """
    if not any(n in _COMPOSITORS for n in names):
        return list(names)
    return [_DB_TO_PURE.get(n, n) for n in names]


# The SFH models whose bin layout is user-settable via ``bin_edges_gyr``.
# Module scope, not a local inside ``resolve_sfh``: the live stellar component
# needs the same set to decide whether to forward the edges, and a second copy
# would drift (#1975).
_NONPARAM_NAMES = frozenset(
    {
        "continuity",
        "dirichlet",
        "continuity_flex",
        "bursty_continuity",
        "psb_suess2022",
        "prospector_beta",
    }
)


def validate_bin_edges_gyr(sfh_type, edges) -> None:
    """Reject a ``bin_edges_gyr`` array that the named SFH cannot use.

    Parameters
    ----------
    sfh_type : str
        Registry name of the SFH the edges are for.
    edges : array_like, shape (n_edges,)
        Candidate bin edges [Gyr], expected strictly ascending.

    Raises
    ------
    ValueError
        If the edges are not 1-D ascending, if the SFH does not take custom
        edges at all, or if a ``continuity``-shaped SFH is given a count its
        declared ratio parameters cannot fill.

    Notes
    -----
    The ratio-count rule holds only for the models whose shape function *is*
    :func:`continuity` (``continuity``, ``bursty_continuity``,
    ``prospector_beta``): they declare ``n_bins - 1`` ratios, so an array of
    ``n`` edges needs exactly ``n - 2`` declared ratios. ``continuity_flex``
    spends some of its parameters on bin *widths* and ``dirichlet`` declares no
    ratios at all, so the rule is not applied to them.

    A mismatched count is not cosmetic: the surplus ratios are swallowed by the
    SFH's ``**ratio_kwargs``, sample a prior that reaches no bin, and change no
    output. That is the silent-config failure of #1975 one step later, so it is
    refused rather than warned about.
    """
    import numpy as _np

    arr = _np.asarray(edges, dtype=float)
    if arr.ndim != 1 or arr.shape[0] < 2:
        raise ValueError(
            f"bin_edges_gyr must be a 1-D array of at least 2 edges, got shape {arr.shape}."
        )
    if not _np.all(_np.diff(arr) > 0):
        raise ValueError(f"bin_edges_gyr must be strictly ascending, got {arr}.")

    if not isinstance(sfh_type, str):
        return
    if sfh_type not in _NONPARAM_NAMES:
        raise ValueError(
            f"bin_edges_gyr applies only to the non-parametric SFHs {sorted(_NONPARAM_NAMES)}, "
            f"not {sfh_type!r}. Passing it here would have no effect."
        )

    spec = SFH_REGISTRY.get(sfh_type)
    if spec is None or spec.fn is not continuity:
        return
    n_declared = sum(1 for name in spec.params if "ratio" in name)
    n_needed = arr.shape[0] - 2  # n_bins - 1 ratios, with n_bins = len(edges) - 1
    if n_declared and n_needed != n_declared:
        raise ValueError(
            f"sfh type={sfh_type!r} declares {n_declared} ratio parameters, which needs "
            f"{n_declared + 2} bin edges, but bin_edges_gyr has {arr.shape[0]}. Supply "
            f"{n_declared + 2} edges (for example tengri.make_agebins_from_zred(zred=...))."
        )


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
        Pure JAX function ``fn(t_lookback, **all_internal_kwargs) -> SFR``
        [Msun/yr].
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

    mean_sfh_type = apply_compositor_swap(mean_sfh_type)

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

    # Build per-spec dispatch info for each additive component.
    #
    # Each entry holds: (callable, public->internal map, set of internal names).
    # The composer dispatches PER-COMPONENT using the public-name map so that
    # two additive SFHs sharing the same internal kwarg (e.g. ``log_total_mass``
    # for any two parametric SFHs after the 2026-05-25 normalization refactor)
    # do not collide. See #372 for the bug this fixes.
    additive_info = []
    for s in additive:
        pub_to_internal = dict(
            s.internal_param_map
        )  # public_name -> (internal_name, scale, offset)
        internal_names = {v[0] for v in pub_to_internal.values()}
        fn_i = s.fn
        if bin_edges_gyr is not None and s.name in _NONPARAM_NAMES:
            fn_i = functools.partial(fn_i, bin_edges_gyr=bin_edges_gyr)
        additive_info.append((fn_i, pub_to_internal, internal_names))

    has_burst = len(mixtures) > 0
    burst_info = None
    if has_burst:
        bs = mixtures[0]
        burst_pub_to_internal = dict(bs.internal_param_map)
        burst_internal = {v[0] for v in burst_pub_to_internal.values()}
        burst_info = (bs.fn, burst_pub_to_internal, burst_internal)

    has_field = len(modulators) > 0

    def _build_component_kw(kw, pub_to_internal, internal_names, *, skip=()):
        """Slice ``kw`` to the internal kwargs this component expects.

        Prefers per-spec public-name entries (``sfh_X_log_total_mass``) so two
        additive components sharing an internal name (``log_total_mass``) each
        get their own value. Falls back to the internal-name entry for
        backward compatibility with callers that pre-translated.

        ``kw[public]`` is assumed already scaled/offset by the upstream
        translator (``parameters/translate.py::get_internal_params``), so the
        composer copies the value verbatim under its internal kwarg name.
        """
        kw_i = {}
        for pub_name, (intl_name, _scale, _offset) in pub_to_internal.items():
            if intl_name in skip:
                continue
            if pub_name in kw:
                kw_i[intl_name] = kw[pub_name]
            elif intl_name in kw:
                kw_i[intl_name] = kw[intl_name]
        return kw_i

    # Build the composed closure
    def composed_fn(t_lookback, **kw):
        """Evaluate the composed SFH: sum additive components, then apply burst and field."""
        # 1. Sum additive components (per-spec public-name dispatch — no collision)
        smooth = jnp.zeros_like(t_lookback)
        for fn_i, pub_to_internal, internal_names in additive_info:
            kw_i = _build_component_kw(kw, pub_to_internal, internal_names)
            smooth = smooth + fn_i(t_lookback, **kw_i)

        # 2. Apply burst mixture
        if has_burst:
            burst_fn, burst_pub_to_internal, burst_internal = burst_info
            log_fburst = kw["log_fburst"]
            f = 10.0**log_fburst
            burst_kw = _build_component_kw(
                kw, burst_pub_to_internal, burst_internal, skip=("log_fburst",)
            )
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
    log_age_grid: jnp.ndarray | None = None,
    centering: float = 1.0,
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

    log_age_grid : array, shape (n_grid,), optional
        ``log10(age/yr)`` grid the SFH is sampled on. Required by the ``drw``
        (linear-time) path to place the covariance in physical time; if omitted
        it is reconstructed with :func:`make_log_age_grid`.

    Returns
    -------
    gp_x : array, shape (n_grid,)
        GP realization sampled on the log-age grid.
    k0_half : float
        Lognormal bias correction K(0)/2 so ``exp(gp_x - k0_half)`` is
        mean-preserving. For ``drw`` this is ``(psd_sigma * ln10)^2 / 2``.

    Notes
    -----
    **JIT-compatible**: yes.

    ``field_model="drw"`` builds a damped random walk stationary in **linear
    (physical) time** — the covariance ``(sigma ln10)^2 exp(-|t_i-t_j|/tau)`` at
    physical times ``t_i = 10**u_i`` (#865). ``psd_sigma`` is then the modulation
    std in dex and ``psd_tau_yr`` the physical decorrelation timescale. It is
    realized via the exact OU state-space (innovations) recursion
    (:func:`~tengri.components.stellar.sfh.gp_sfh.drw_innovations_gp_from_xi`),
    which for a Markov covariance *is* the Cholesky factor — the same ``xi -> SFH``
    map as a dense Cholesky, computed in ``O(n)`` instead of ``O(n^3)`` and without
    the positive-definiteness jitter, so the realized prior is the exact ``K``. The
    posterior geometry is unchanged by this (#1301 is not addressed by it). Other
    PSD models keep the Fourier/log-age construction (:func:`gp_from_xi`).

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
    if field_model == "drw":
        # Damped random walk stationary in LINEAR (physical) time: the DRW
        # covariance is built directly in cosmic time and sampled on the
        # log-age grid (#865). ``psd_sigma`` is the modulation std in dex and
        # ``psd_tau_yr`` the physical decorrelation timescale — both mean exactly
        # what the ``field`` priors document (Caplar & Tacchella 2019 amplitude;
        # Iyer+2020 timescale). Replaces the former Fourier/log-age construction,
        # whose correlation length was fixed in dex (scale-free) and only matched
        # the physical timescale near a single reference age.
        #
        # Realized via the exact OU state-space (innovations) recursion rather
        # than a dense Cholesky. For a Markov covariance the recursion *is* the
        # Cholesky factor (lower-triangular, positive diagonal, M M^T = K — and
        # that factor is unique), so the ``xi -> SFH`` map is numerically
        # identical: O(n) instead of O(n^3), and jitter-free, so the prior is the
        # exact K rather than K + 1e-6 var I. It does NOT change the posterior
        # geometry and is therefore not a fix for the #1301 divergences; the
        # zero-rotation Fourier basis (a different prior) is tracked in #1333.
        # The dense Cholesky (``drw_linear_gp_from_xi``) is retained as the oracle.
        if log_age_grid is None:
            log_age_grid = make_log_age_grid(n_grid)
        if float(centering) == 1.0:
            # Bit-identical default: the production O(n) recursion, untouched.
            return drw_innovations_gp_from_xi(xi, psd_sigma, psd_tau_yr, log_age_grid)
        # Partial centering (#1355). ``a < 1`` moves amplitude dependence out of
        # the xi -> SFH map, which is where the funnel lives: s = L(sigma,tau) xi
        # is BILINEAR, and no fixed metric linearizes a multiplicative coupling,
        # so preconditioning cannot reach it.
        #
        # The caller MUST pair this with the matching latent log-prior
        # (drw_latent_log_prior). At a < 1 the prior on zeta is
        # N(0, sigma_s^(2-2a) I), not N(0, I); omitting the -n(1-a) log sigma_s
        # normalizer leaves a sampler that runs cleanly, reports nothing, and
        # targets a DIFFERENT posterior at every a.
        from tengri.components.stellar.sfh.gp_sfh import drw_partial_gp_from_zeta

        return drw_partial_gp_from_zeta(
            xi, psd_sigma, psd_tau_yr, log_age_grid, centering=centering
        )

    # Other PSD models (e.g. flex-PSD) keep the Fourier/log-age construction.
    sqrt_power_fn = FIELD_MODEL_REGISTRY[field_model]
    sqrt_power = sqrt_power_fn(n_grid, d_log_age, psd_sigma, psd_tau_yr)
    gp_x = gp_from_xi(xi, sqrt_power, n_grid)
    k0_half = drw_variance(psd_sigma) / 2.0
    return gp_x, k0_half

# SPDX-License-Identifier: BSD-3-Clause
"""Analytic marginalization of the total-stellar-mass amplitude (``profile_mass``).

Every channel tengri fits is linear in the total stellar mass
:math:`M = 10^{\\ell}`, where :math:`\\ell` is a model's ``*_log_total_mass``
free parameter: photometric fluxes, spectral pixels, and (were the channel
wired in, see Guards) a spectrum's line fluxes all satisfy
:math:`f_i(\\theta, M) = M\\,f_i(\\theta)` for a data point :math:`i`, with
:math:`f_i(\\theta)` the model prediction per unit mass at the other
``D - 1`` free parameters :math:`\\theta`. Under a Gaussian likelihood this
makes :math:`\\chi^2(M)`, summed over the FULL data vector
:math:`d = (\\text{photometry}, \\text{spectrum})` (photometry-only,
spectroscopy-only, or their concatenation for ``data_type="joint"``, in
that order -- see ``loss_functions._build_prediction`` and
``tests/regression/bug/test_bug_1366_joint_data_record.py``) with noise
:math:`\\sigma`, an exact quadratic,

.. math::

    \\chi^2(M) = \\chi^2_{\\min} + A\\,(M - M^*)^2, \\qquad
    A = \\sum_i \\frac{f_i(\\theta)^2}{\\sigma_i^2}, \\qquad
    M^* = \\frac{1}{A}\\sum_i \\frac{d_i\\, f_i(\\theta)}{\\sigma_i^2}, \\qquad
    \\chi^2_{\\min} = \\sum_i \\frac{d_i^2}{\\sigma_i^2} - A\\,(M^*)^2,

with :math:`d_i` the observed flux and :math:`\\sigma_i` its 1-sigma
uncertainty. Marginalizing :math:`M` out of the likelihood under its own
prior :math:`p(\\ell)` (the mass parameter's declared prior density, evaluated
directly on :math:`\\ell`, exactly as every other free parameter's prior is)
is the standard closed-form result for a linear amplitude in a Gaussian
likelihood [1]_:

.. math::

    \\log p(d \\mid \\theta) = -\\tfrac12 \\chi^2_{\\min}
        + \\log \\int \\exp\\!\\left(-\\tfrac12 A\\,(M(\\ell) - M^*)^2\\right)
          p(\\ell)\\, d\\ell.

The integral is evaluated by a fixed-size (48-node) trapezoid quadrature in
:math:`\\ell` over a window :math:`\\pm 8\\sigma_\\ell` around
:math:`\\ell^* = \\log_{10} M^*` (clamped to the prior's own support), with
:math:`\\sigma_\\ell = 1 / (\\sqrt{A}\\, M^* \\ln 10)` the local Gaussian width
of :math:`\\chi^2(M)` mapped into :math:`\\log_{10} M` — no forward-model
evaluation happens inside the integral, only evaluations of the already-known
exact quadratic. Sampling then runs on the remaining ``D - 1`` parameters:
the returned log-posterior is ``log p(d | theta) + log p(theta)`` with the
usual standardized :math:`\\mathcal{N}(0, I)` prior on the surviving
:math:`\\xi`'s, and the mass's own :math:`\\mathcal{N}(0, 1)` prior term does
not appear, because :math:`\\int p(\\ell)\\, d\\ell` (the term this module
replaces it with) already accounts for it.

**Approach.** The mass parameter is fixed in the ``Fitter``'s *working* spec
at a placeholder value (the mass prior's own bounds midpoint, so
``predict_photometry`` always runs at a realistic mass scale regardless of
dtype, see the comment above :func:`configure_profile_mass`'s placeholder
line) before ``Fitter._free_names`` is read off it, so every existing
free-parameter
mechanism (``_free_names``, ``_to_physical``, ``_engine_cache_key``,
``InferenceContext.free_names``) sees ``D - 1`` parameters with no special
casing. :func:`build_profiled_loss_fn` then supplies the marginal
log-likelihood above in place of the ordinary chi-squared term, and
:func:`finalize_profile_mass` reinserts the mass into the ``Posterior`` once
inference is done: an exact conditional draw ``p(log10(M) | theta, d)`` per
posterior sample (inverse-CDF on the same quadrature grid, mapped to the
sampler's standardized coordinate via the prior's own
``standardize``/``unstandardize`` pushforward) for sample-based backends, or
the conditional mode :math:`\\ell^* = \\log_{10} M^*` for ``method="map"``.

**Guards.** The exact quadratic in :math:`M` (and hence this whole module)
requires: ``data_type`` one of ``"photometry"``, ``"spectroscopy"``, or
``"joint"`` (any concatenation these fitter data types assemble, per
``loss_functions._build_prediction``) with a plain
Gaussian likelihood; no emission-line channel (marginalized, fitted, or
measured line fluxes -- a channel with its own likelihood plumbing, not yet
wired into this module even though line fluxes are linear in :math:`M` too);
no line-ratio or spectral-index channel; no calibration marginalization
(another block of linear parameters marginalized separately, not this
module's math); no variable-noise model; no censored upper/lower limits;
exactly one free ``*log_total_mass`` parameter with a bounded-support prior;
and the full data vector numerically linear in that parameter (a
mass-independent additive component, e.g. an unmasked AGN continuum, breaks
this). See :func:`configure_profile_mass`.

Measured on a D=8, 14-band mock (ctl-dpl): Hessian condition number
3.7e4 -> 1.2e3, worst-of-six-seeds NUTS wall 227 s -> 40 s (see
``docs/dev/inference_methods.md``, "Profiling the Mass").

References
----------
.. [1] D. S. Sivia and J. Skilling, *Data Analysis: A Bayesian Tutorial*,
   2nd ed., Oxford University Press (2006), Sec. 3.2 ("The best-fit
   amplitude of a model" / marginalizing a linear amplitude out of a
   Gaussian likelihood).
"""

from __future__ import annotations

import contextlib
import copy
import logging
import math
import re
import warnings
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from tengri.parameters.priors import Fixed

if TYPE_CHECKING:
    from tengri.inference.fitter import Fitter
    from tengri.inference.posterior import Posterior
    from tengri.parameters.priors import Distribution

__all__ = [
    "build_profiled_loglikelihood_fn",
    "build_profiled_loglikelihood_unbounded_fn",
    "build_profiled_loss_fn",
    "configure_profile_mass",
    "finalize_profile_mass",
    "suppress_placeholder_dead_fit_warning",
]

logger = logging.getLogger(__name__)

#: Trapezoid quadrature nodes spanning the marginalization window (measured;
#: see the module docstring's references to ``docs/dev/inference_methods.md``).
_QUAD_NODES = 48
_QUAD_HALF_WIDTH_SIGMAS = 8.0
#: Mathematically, any log10(mass) placeholder works for the value the mass
#: parameter is pinned to in the working spec once it is profiled out: the
#: unit-mass flux computed from it (``predict_photometry(...) /
#: 10**placeholder``) divides it back out exactly, by the linearity guard's
#: own construction (see :func:`_profile_stats`). *Numerically* it does not:
#: a fixed placeholder far from the mass prior's own scale (0.0, i.e. M=1
#: solar mass, against a typical galaxy prior of 1e8-1e12) evaluates
#: ``predict_photometry`` at a flux many orders of magnitude below the real
#: one, and under float32 that unit-mass flux can underflow toward the
#: subnormal floor -- ``_profile_stats``'s ``f**2`` then collapses ``A`` to
#: (numerically) zero, and ``mstar = B / A`` and everything downstream of it
#: goes inf/nan (measured: a NaN posterior gradient on
#: ``tests/regression/precision/test_float32_fitting_path_seams.py``'s
#: ``stellar_dust/exact_fixedz`` seam). :func:`configure_profile_mass`
#: instead pins the mass at the prior's own bounds midpoint, so
#: ``predict_photometry`` always runs at a realistic mass scale regardless of
#: dtype.
#: Floor for the two-mass linearity probe's tolerance (physical flux ratio one
#: dex apart), tuned for float64 roundoff. :func:`_linearity_max_deviation`
#: takes ``max(_LINEARITY_TOL, 1e4 * eps(dtype))`` so the same exactly-linear
#: model is not refused under float32 roundoff alone -- see that function's
#: docstring.
_LINEARITY_TOL = 1e-8

#: ``fitter.data_type`` values the exact chi^2(M) quadratic covers: photometry
#: alone, spectroscopy alone, or their concatenation (photometry-then-spectrum,
#: see ``tests/regression/bug/test_bug_1366_joint_data_record.py``). Excludes
#: any channel with its own likelihood plumbing (emission lines, line ratios,
#: spectral indices) or its own marginalized linear block (calibration) --
#: those are refused by the guards below regardless of ``data_type``.
_SUPPORTED_DATA_TYPES = frozenset({"photometry", "spectroscopy", "joint"})


# ── Guards ──────────────────────────────────────────────────────────────────


def _mass_prior_bounds(dist: Distribution) -> tuple[float, float]:
    """Finite (lo, hi) support of a mass prior, or raise ``ValueError``."""
    lo, hi = dist.bounds
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(
            "profile_mass requires the mass parameter's prior to have a bounded "
            f"support (finite lower and upper bounds) for the marginalization "
            f"quadrature; got bounds={dist.bounds} for a {type(dist).__name__}. "
            "Use Uniform(lo, hi) or another prior with finite support."
        )
    return float(lo), float(hi)


def _linearity_max_deviation(fitter: Fitter, mass_name: str) -> tuple[float, float]:
    """``(max|ratio(theta, +1 dex mass) - 10|, tolerance)`` at a representative theta.

    Probes the model's own physics, not the declared prior's support: two
    mass values one dex apart, held fixed, with every other free parameter at
    its prior median (``unstandardize(0.0)``, matching
    ``forward.forward_model._central_params``) and every fixed parameter at
    its declared value. The prediction probed is the fitter's own data-vector
    assembly for its ``data_type`` (:func:`_predict_full_vector`), photometry,
    spectroscopy, or their concatenation, so a spectroscopy channel with a
    mass-independent additive term is caught exactly as a photometric one is.
    Catches a mass-independent additive component (e.g. an AGN continuum)
    that the exact chi^2(M) quadratic cannot marginalize.

    The tolerance scales with the *actual* dtype the prediction returned
    (``jax.enable_x64`` may be off process-wide, silently downcasting every
    ``float64`` request to ``float32``, see the warning in
    https://docs.jax.dev/en/latest/notes/gotchas.html -- ``fitter.model`` and
    its inputs carry whatever precision the ambient JAX config gives them,
    not necessarily what a caller asked for): a fixed ``_LINEARITY_TOL``
    tuned for float64 roundoff (~1e-13 measured) is unreachable under float32
    roundoff on the exact same, exactly-linear model (~4.3e-05 measured on a
    4-band stellar+dust seam, tests/regression/precision), which used to make
    "auto" engage under float64 and silently refuse under float32 for
    identical configurations -- a defect this module exists to be free of,
    caught by ``test_float32_fitting_path_seams.py`` reporting the resulting
    posteriors as different *dimensionality* (mass profiled out on one arm,
    not the other) rather than merely different values. The scaled floor
    stays far below a genuine non-linearity (a mass-independent component
    like an unmasked AGN continuum), which perturbs the ratio at the
    O(1) scale, not the O(1e3 * eps) one.
    """
    spec = fitter.spec
    phys: dict[str, Any] = {}
    for name in spec.free_params:
        if name != mass_name:
            phys[name] = spec.get_distribution(name).unstandardize(0.0)
    for name, val in spec.get_fixed_values().items():
        if name != mass_name:
            phys[name] = jnp.asarray(val)
    if getattr(spec, "stochastic", False):
        phys["sfh_field_xi"] = jnp.zeros(spec.n_grid)

    use_components = bool(getattr(fitter, "use_components", False))
    pred_a = _predict_full_vector(
        fitter.model,
        fitter.data_type,
        {**phys, mass_name: jnp.asarray(9.0)},
        use_components=use_components,
    )
    pred_b = _predict_full_vector(
        fitter.model,
        fitter.data_type,
        {**phys, mass_name: jnp.asarray(10.0)},
        use_components=use_components,
    )
    max_dev = float(jnp.max(jnp.abs(pred_b / pred_a - 10.0)))
    tol = max(_LINEARITY_TOL, 1e4 * float(jnp.finfo(pred_a.dtype).eps))
    return max_dev, tol


def _check_guards(fitter: Fitter, params_override: dict | None) -> tuple[str | None, dict]:
    """Every ``profile_mass`` precondition, evaluated without raising.

    Returns
    -------
    reason : str or None
        ``None`` if every guard passes; otherwise a human-readable reason
        naming the first one that failed.
    context : dict
        ``{"mass_name", "mass_prior", "bounds"}`` when ``reason is None``,
        else ``{}``.
    """
    from tengri.inference.fitter import _has_line_adjacent_channel
    from tengri.observation.noise import has_noise_model, uses_student_t

    spec = fitter.spec
    if not hasattr(spec, "_distributions"):
        return f"parameter spec {type(spec).__name__} is not a plain Parameters (unsupported)", {}

    # The marginal's curvature in flux units (A ~ 1e-13 at fluxes ~1e-30) and the
    # per-band cotangent scales its reverse-mode gradient needs are outside
    # float32's range: measured on 2026-09-11 (bench/reports/2026-09-11_profile_mass_20s.md,
    # Finding 8), the float64 gradient is finite where the float32 one is NaN on
    # the same fixture. Profiling is a float64 feature until that seam is closed.
    if not jax.config.jax_enable_x64:
        return "float32 mode (jax_enable_x64 is off); profiling needs float64", {}

    candidates = [n for n in spec.free_params if n.endswith("log_total_mass")]
    if len(candidates) != 1:
        return (
            "expected exactly one free parameter named '*log_total_mass', found "
            f"{candidates or 'none'}"
        ), {}
    mass_name = candidates[0]

    if len(spec.free_params) < 2:
        return (
            "profiling the mass would leave zero other free parameters to sample "
            f"(free_params={list(spec.free_params)})"
        ), {}

    if params_override is not None and mass_name in params_override:
        return f"'{mass_name}' cannot be both profiled and pinned via params_override", {}

    mass_prior = spec.get_distribution(mass_name)
    try:
        bounds = _mass_prior_bounds(mass_prior)
    except ValueError as exc:
        return str(exc), {}

    if fitter.data_type not in _SUPPORTED_DATA_TYPES:
        return (
            f"data_type={fitter.data_type!r} (profile_mass requires one of "
            f"{sorted(_SUPPORTED_DATA_TYPES)})"
        ), {}
    if fitter._fits_lines(fitter.model):
        return "an emission-line channel is configured (marginalized, fitted, or measured)", {}
    if _has_line_adjacent_channel(fitter.model):
        return "a line-ratio or spectral-index channel is configured", {}
    if fitter._calibration_marginalize:
        return "calibration_marginalize=True", {}
    if uses_student_t(spec):
        return "the likelihood is Student-t (noise_dof != 0), not Gaussian", {}
    if has_noise_model(spec):
        return "a variable-noise model (noise_frac_cal) is configured", {}
    if fitter.data_mask is not None and bool(jnp.any(jnp.asarray(fitter.data_mask) != 0)):
        return "censored data (upper/lower limits) is present", {}

    max_dev, tol = _linearity_max_deviation(fitter, mass_name)
    if not (max_dev < tol):
        return (
            f"photometry is not exactly linear in '{mass_name}' "
            f"(max|flux_ratio(+1 dex) - 10| = {max_dev:.3e}, need < {tol:.0e}); "
            "likely a mass-independent component (e.g. an AGN continuum)"
        ), {}

    return None, {"mass_name": mass_name, "mass_prior": mass_prior, "bounds": bounds}


def configure_profile_mass(fitter: Fitter, profile_mass: bool | str, params_override) -> None:
    """Resolve ``profile_mass`` and, if engaged, fix the mass in the working spec.

    Called once from ``Fitter.__init__``, immediately before
    ``self._free_names = self.spec.free_params`` is read, so every
    downstream free-parameter accounting sees ``D - 1`` parameters with the
    mass excluded, no special-casing required anywhere else.

    Parameters
    ----------
    fitter : Fitter
        The fitter under construction. Mutated in place: ``fitter.spec`` is
        replaced when profiling engages, and ``_profile_mass``,
        ``_profile_mass_resolved``, ``_profile_mass_reason`` are always set
        (plus ``_profile_mass_name``/``_profile_mass_prior``/
        ``_profile_mass_bounds`` when profiling engages).
    profile_mass : bool or "auto"
        ``True`` engages profiling, raising ``ValueError`` naming the first
        failed guard (see :func:`_check_guards`). ``False`` disables it
        unconditionally. ``"auto"`` (the ``Fitter``/``ForwardModel.fit``
        default) engages it only when every guard passes, logging one
        ``INFO`` line naming the reason when it does not.
    params_override : dict or None
        The fitter's per-fit parameter override (raw constructor argument,
        not yet merged into ``fitter._fixed_values``): profiling and pinning
        the same mass parameter are mutually exclusive.

    Raises
    ------
    ValueError
        If ``profile_mass`` is not one of ``True``, ``False``, ``"auto"``, or
        if ``profile_mass=True`` and any guard fails.
    """
    if profile_mass not in (True, False, "auto"):
        raise ValueError(f"profile_mass must be True, False, or 'auto'; got {profile_mass!r}")

    # Every profiling attribute exists on every Fitter, engaged or not, so the
    # cache-key ledgers (``_engine_policy``) see one attribute set per Fitter.
    fitter._profile_mass_name = None
    fitter._profile_mass_prior = None
    fitter._profile_mass_bounds = None
    fitter._profile_mass_original_spec = None

    if profile_mass is False:
        fitter._profile_mass = False
        fitter._profile_mass_resolved = False
        fitter._profile_mass_reason = "profile_mass=False"
        return

    if profile_mass is True:
        reason, ctx = _check_guards(fitter, params_override)
        if reason is not None:
            raise ValueError(f"profile_mass=True but {reason}.")
        engage, resolved_reason = True, "profile_mass=True"
    else:  # "auto"
        try:
            reason, ctx = _check_guards(fitter, params_override)
        except Exception as exc:
            reason, ctx = f"guard check raised {exc!r}", {}
        engage = reason is None
        if engage:
            resolved_reason = "auto-enabled: every guard passed"
        else:
            resolved_reason = f"auto-disabled: {reason}"
            logger.info("profile_mass='auto': disabled (%s).", reason)

    fitter._profile_mass = engage
    fitter._profile_mass_resolved = engage
    fitter._profile_mass_reason = resolved_reason
    if not engage:
        return

    mass_name = ctx["mass_name"]
    ell_lo, ell_hi = ctx["bounds"]
    # The bounds midpoint, not a fixed 0.0: see the module-level comment
    # above (where the old ``_MASS_PLACEHOLDER_LOG10`` constant lived) for
    # why the numerics -- not the math -- need this to track the prior's
    # own scale.
    placeholder = 0.5 * (ell_lo + ell_hi)
    new_spec = copy.copy(fitter.spec)
    new_spec._distributions = {
        **fitter.spec._distributions,
        mass_name: Fixed(placeholder),
    }
    # Kept so :func:`resolve_profile_mass_for_method` can undo the rewrite at
    # ``run()`` time for a backend that does not consume the profiled loss.
    fitter._profile_mass_original_spec = fitter.spec
    fitter.spec = new_spec
    fitter._profile_mass_name = mass_name
    fitter._profile_mass_prior = ctx["mass_prior"]
    fitter._profile_mass_bounds = ctx["bounds"]


#: Backends that consume the Fitter's own loss functions
#: (``_get_or_build_loss_fn`` / ``_get_or_build_logdensity_fn`` / the
#: log-likelihood pair) and therefore see the profiled marginal. Every
#: BlackJAX sampler goes through ``mcmc/_shared._get_flat_logdensity``; MAP,
#: Laplace, HMC-IS and SMC read the same seam. NIFTy geoVI/MGVI and the native
#: Gaussian VI build their objective from the model and its spec directly, so
#: under profiling they would fit with the mass frozen at the placeholder --
#: measured on 2026-09-12 (ctl-dpl seed 7, geoVI: mass 10.24 against the NUTS
#: reference 11.96, age 0.5 Gyr against 5.2). Anything not listed here runs
#: unprofiled; add a backend only after checking it reads the Fitter's loss.
PROFILE_MASS_BACKENDS = frozenset(
    {
        "map",
        "laplace",
        "mcmc",
        "mcmc_nuts",
        "mcmc_nuts_fast",
        "mcmc_hmc",
        "mcmc_dynamic_hmc",
        "mcmc_ghmc",
        "mcmc_chees",
        "mcmc_mclmc",
        "mcmc_adjusted_mclmc",
        "mcmc_barker",
        "mcmc_mala",
        "mcmc_hmc_lowrank",
        "mcmc_smc",
        "hmc_is",
    }
)


def disable_profile_mass(fitter: Fitter, reason: str) -> None:
    """Undo the working-spec rewrite so the fitter samples the mass again.

    Restores the original spec and the free-parameter bookkeeping derived
    from it. The cached loss functions are keyed on the free-name tuple, so
    the profiled ones become unreachable and the unprofiled ones rebuild on
    first use; nothing is evicted by hand.
    """
    if not getattr(fitter, "_profile_mass", False):
        return
    original = getattr(fitter, "_profile_mass_original_spec", None)
    if original is not None:
        fitter.spec = original
        fitter._free_names = fitter.spec.free_params
        fitter._fixed_values = fitter.spec.get_fixed_values()
        fitter._bounds = {n: fitter.spec.get_distribution(n).bounds for n in fitter._free_names}
    fitter._profile_mass = False
    fitter._profile_mass_resolved = False
    fitter._profile_mass_reason = reason
    logger.info("profile_mass: disabled for this fit (%s).", reason)


def resolve_profile_mass_for_method(fitter: Fitter, method: str, requested) -> None:
    """Keep profiling only for a backend that consumes the profiled loss.

    Called from ``Fitter.run()`` once the method name is known. ``requested``
    is the constructor's ``profile_mass`` argument: an explicit ``True`` on an
    unsupported backend is an error, ``"auto"`` steps aside with a logged
    reason, ``False`` was never engaged.
    """
    if not getattr(fitter, "_profile_mass", False) or method in PROFILE_MASS_BACKENDS:
        return
    reason = (
        f"method={method!r} builds its objective from the model rather than the "
        "Fitter's loss, so it cannot see the profiled marginal"
    )
    if requested is True:
        raise ValueError(f"profile_mass=True but {reason}.")
    disable_profile_mass(fitter, f"auto-disabled: {reason}")


@contextlib.contextmanager
def suppress_placeholder_dead_fit_warning(fitter: Fitter):
    """Silence the one false-positive ``DeadFitWarning`` the placeholder causes.

    ``Fitter._to_physical`` publishes the profiled-out mass at its fixed
    placeholder for every draw (exactly as it does for any genuinely fixed
    parameter) before :func:`finalize_profile_mass` overwrites it with the
    real per-draw value. ``Posterior.__post_init__``'s dead-fit check reads
    ``fitter.model.spec`` -- the ORIGINAL, un-profiled spec, never mutated,
    where the mass is still declared free -- to decide which zero-variance
    columns are suspicious, so at construction time (before the reinsertion
    above runs) it cannot yet tell "profiled out" apart from "genuinely
    frozen" and raises a false alarm naming the mass column alone.

    Filters out only that one message signature (this exact parameter name,
    alone) for the duration of the wrapped call; a ``DeadFitWarning`` naming
    any other parameter, or the mass alongside a genuinely frozen one, is
    unaffected and still surfaces normally.
    """
    if not getattr(fitter, "_profile_mass", False):
        yield
        return

    from tengri.config.exceptions import DeadFitWarning

    mass_name = re.escape(fitter._profile_mass_name)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=rf"^dead fit: parameter\(s\) '{mass_name}' have \d+ unique draw",
            category=DeadFitWarning,
        )
        yield


# ── The exact chi^2(M) quadratic and its marginal integral ──────────────────


def _predict_full_vector(
    model,
    data_type: str,
    phys: dict,
    *,
    use_components: bool = False,
    jit_inputs: dict | None = None,
    threaded_impl=None,
) -> jnp.ndarray:
    """The model's prediction for the fitter's FULL data vector, phot/spec/joint.

    Delegates to ``loss_functions._build_prediction``,
    the single assembly the ordinary (unprofiled) Gaussian likelihood scores
    against for this ``data_type``: ``predict_photometry`` alone,
    ``predict_spectrum`` alone, or their concatenation photometry-then-
    spectrum for ``"joint"`` (the order
    ``tests/regression/bug/test_bug_1366_joint_data_record.py`` pins). Reusing
    it rather than re-deriving the dispatch means the profiled statistics see
    exactly the vector (data mask, ``SpectrumPrecomp`` path, JIT-threaded SSP
    grid included) the unprofiled likelihood does. Every emission-line /
    line-ratio / spectral-index channel is excluded by construction: this
    module's own guards (:func:`_check_guards`) refuse any fit carrying one,
    so ``_build_prediction`` is always called with every feature-channel flag
    off.
    """
    from tengri.inference.loss_functions import _build_prediction

    _, predicted, _, _ = _build_prediction(
        model,
        phys,
        data_type,
        has_line_fluxes=False,
        has_indices=False,
        index_defs=None,
        data_args={},
        use_components=use_components,
        has_line_ratios=False,
        measured_line_defs=None,
        jit_inputs=jit_inputs,
        threaded_impl=threaded_impl,
    )
    return predicted


def _profile_stats(
    model,
    mass_name: str,
    phys: dict,
    data: jnp.ndarray,
    noise: jnp.ndarray,
    presence: jnp.ndarray | None = None,
    *,
    data_type: str,
    use_components: bool = False,
    jit_inputs: dict | None = None,
    threaded_impl=None,
):
    """``(A, M*, chi2_min)`` of ``chi2(M) = chi2_min + A * (M - M*)**2``.

    Exact whenever the ``data_type`` data vector (:func:`_predict_full_vector`
    -- photometry, spectroscopy, or their photometry-then-spectrum
    concatenation for ``"joint"``) is linear in ``M = 10**phys[mass_name]``,
    i.e. the linearity guard's own condition. No assumption on what
    ``phys[mass_name]`` actually is: the unit-mass prediction ``pred /
    10**phys[mass_name]`` divides out whatever mass the prediction was made
    at, so this is safe to call with the placeholder-fixed value used during
    inference or a real posterior-sample mass used post hoc.
    """
    pred = _predict_full_vector(
        model,
        data_type,
        phys,
        use_components=use_components,
        jit_inputs=jit_inputs,
        threaded_impl=threaded_impl,
    )
    placeholder_mass = 10.0 ** phys[mass_name]
    # Divide by noise *before* squaring/multiplying (every chi^2 in tengri
    # does, see ``likelihoods.gaussian.standardized_residual``), AND divide
    # ``pred`` by the mass placeholder only *after* that -- not ``f =
    # pred / placeholder_mass`` up front, then ``f / noise``.
    #
    # The two orderings are exact in real arithmetic but not in float32.
    # ``pred`` at the placeholder mass and its per-band noise are each
    # individually tiny for a faint band (measured: ~3e-30 and ~1e-31 on a
    # far-IR band ~1e3x fainter than the others, tests/regression/precision's
    # stellar_dust/exact_fixedz seam) but ``snr_pred = pred / noise`` is an
    # ordinary S/N-scale, order-1-to-100 number regardless. Forming ``f =
    # pred / placeholder_mass`` first, before dividing by noise, computes an
    # intermediate "unit-mass flux" that can land in float32's subnormal
    # range (~3e-40 here) -- and CPU XLA flushes subnormals in that
    # computation to exactly 0.0 -- which is fine for the forward value (that
    # band's contribution to A/B is negligible) but starves the reverse-mode
    # cotangent reaching the prediction (``predict_photometry`` /
    # ``predict_spectrum``): dividing by
    # ``placeholder_mass`` *before* the noise normalization multiplies the
    # local Jacobian by an extra ``1/placeholder_mass`` on the way back,
    # weakening the cotangent by that same large factor (measured: enough to
    # zero out d(pred)/d(theta) entirely in float32, though the same
    # derivative is finite -- if tiny -- in float64). Deferring the
    # ``/ placeholder_mass`` rescaling to the aggregate sums below (themselves
    # ordinary-magnitude numbers) never revisits either extreme and restores
    # the same reverse-mode scale the rest of tengri's chi^2 gradients get.
    snr_pred = pred / noise
    snr_d = data / noise
    if presence is not None:
        # The same rule as ``likelihoods.gaussian``: an absent band (presence 0)
        # contributes exactly zero to every sum and to its gradient, so the
        # mass marginal never sees a band the ordinary likelihood does not.
        snr_pred = presence * snr_pred
        snr_d = presence * snr_d
    sum_pred2 = jnp.sum(snr_pred**2)
    sum_data_pred = jnp.sum(snr_d * snr_pred)
    A = sum_pred2 / placeholder_mass**2
    B = sum_data_pred / placeholder_mass
    mstar = B / A
    d2_sum = jnp.sum(snr_d**2)
    chi2_min = d2_sum - B**2 / A
    return A, mstar, chi2_min


def _quad_nodes(A, mstar, ell_lo: float, ell_hi: float):
    """48-node trapezoid grid in log10(M), +/- 8 sigma around log10(M*).

    The window is centered on ``log10(M*)`` *clamped* to ``[ell_lo, ell_hi]``
    first, then widened by ``+/- 8 sigma`` and clipped to the same bounds.
    Clamping the center (rather than clamping each edge of the unclamped
    window independently) guarantees ``lo <= center <= hi`` however far
    outside the prior's support the unconstrained best-fit mass falls, e.g.
    at an optimizer's first, far-from-converged evaluation: clipping the
    edges alone can otherwise invert them (``lo > hi``), turning the
    trapezoid weights negative and ``log(weight)`` into ``nan``.
    """
    center = jnp.clip(jnp.log10(mstar), ell_lo, ell_hi)
    sigma_ell = 1.0 / (jnp.sqrt(A) * mstar * jnp.log(10.0))
    lo = jnp.maximum(ell_lo, center - _QUAD_HALF_WIDTH_SIGMAS * sigma_ell)
    hi = jnp.minimum(ell_hi, center + _QUAD_HALF_WIDTH_SIGMAS * sigma_ell)
    ell_nodes = jnp.linspace(lo, hi, _QUAD_NODES)
    d_ell = (hi - lo) / (_QUAD_NODES - 1)
    weights = jnp.ones(_QUAD_NODES).at[0].set(0.5).at[-1].set(0.5) * d_ell
    return ell_nodes, weights


def _log_quadrature_terms(A, mstar, ell_lo: float, ell_hi: float, mass_prior: Distribution):
    """Node grid and its ``-0.5*A*(M-M*)^2 + log p(ell) + log(weight)`` terms."""
    ell_nodes, weights = _quad_nodes(A, mstar, ell_lo, ell_hi)
    m_values = 10.0**ell_nodes
    log_terms = (
        -0.5 * A * (m_values - mstar) ** 2 + mass_prior.log_prob(ell_nodes) + jnp.log(weights)
    )
    return ell_nodes, log_terms


def _log_mass_integral(A, mstar, ell_lo: float, ell_hi: float, mass_prior: Distribution):
    """``log integral(exp(-0.5*A*(M(ell)-M*)**2) * p(ell) dell)``, quadrature."""
    _, log_terms = _log_quadrature_terms(A, mstar, ell_lo, ell_hi, mass_prior)
    return logsumexp(log_terms)


def _sample_log_mass(key, A, mstar, ell_lo: float, ell_hi: float, mass_prior: Distribution):
    """Inverse-CDF draw of log10(M) from the exact conditional, on the same grid."""
    ell_nodes, log_terms = _log_quadrature_terms(A, mstar, ell_lo, ell_hi, mass_prior)
    probs = jax.nn.softmax(log_terms)
    cdf = jnp.cumsum(probs)
    u = jax.random.uniform(key)
    return jnp.interp(u, cdf / cdf[-1], ell_nodes)


# ── The profiled loss (drop-in for Fitter._build_loss_fn) ───────────────────


def _threaded_predict_impl(model, use_components: bool):
    """The threaded observables builder for ``model``, or ``None`` when unavailable.

    Mirrors the threading ``loss_functions._build_data_neg_log_likelihood_fn``
    sets up for the unprofiled loss, so the profiled one keeps the SSP grid
    and template arrays as outer-level JAX ``Parameter`` ops rather than
    baking them into the HLO as closure-captured constants (see that
    function's Notes on cold-compile cost).
    """
    if use_components or not hasattr(model, "_get_or_build_predict_observables_jit"):
        return None
    return model._get_or_build_predict_observables_jit()


def build_profiled_loss_fn(fitter: Fitter):
    """Build ``loss_fn(params_unbounded, data_args) -> scalar`` with mass profiled out.

    Mirrors ``tengri.inference.loss_functions.build_loss_fn`` exactly, except
    the photometric chi-squared data term is replaced by the marginal
    log-likelihood of the module docstring: the mass has already been fixed
    out of ``fitter.spec``/``fitter._free_names`` by
    :func:`configure_profile_mass`, so ``standardized_neg_log_prior`` below
    naturally excludes its standardized N(0,1) term, exactly as the
    replacement requires.
    """
    from tengri.inference.loss_functions import (
        _unstandardize_parameters,
        standardized_neg_log_prior,
    )

    spec = fitter.spec
    free_names = fitter._free_names
    fixed_values = fitter._fixed_values
    stochastic = spec.stochastic
    field_centering = float(getattr(spec, "field_centering", 1.0))
    model = fitter.model
    mass_name = fitter._profile_mass_name
    mass_prior = fitter._profile_mass_prior
    ell_lo, ell_hi = fitter._profile_mass_bounds
    data_type = fitter.data_type
    use_components = bool(getattr(fitter, "use_components", False))
    threaded_impl = _threaded_predict_impl(model, use_components)

    def loss_fn(params_unbounded, data_args):
        """-log p(d | xi) with the mass profiled out, plus 1/2 xi^T xi on the rest."""
        params = _unstandardize_parameters(
            params_unbounded, spec, free_names, fixed_values, stochastic
        )
        if "redshift" in data_args:
            params = {**params, "redshift": data_args["redshift"]}

        A, mstar, chi2_min = _profile_stats(
            model,
            mass_name,
            params,
            data_args["data"],
            data_args["noise"],
            presence=data_args.get("presence"),
            data_type=data_type,
            use_components=use_components,
            jit_inputs=data_args.get("_jit_inputs"),
            threaded_impl=threaded_impl,
        )
        loglik = -0.5 * chi2_min + _log_mass_integral(A, mstar, ell_lo, ell_hi, mass_prior)

        return -loglik + standardized_neg_log_prior(
            params_unbounded,
            free_names,
            stochastic=stochastic,
            centering=field_centering,
            psd_sigma_dex=(params.get("sfh_field_psd_sigma") if field_centering != 1.0 else None),
        )

    return loss_fn


def build_profiled_loglikelihood_unbounded_fn(fitter: Fitter):
    """Build ``loglik(params_unbounded, data_args) -> scalar`` with mass profiled out.

    Companion to :func:`build_profiled_loss_fn`, mirroring
    ``loss_functions.build_loglikelihood_unbounded_fn``: the same marginal
    log-likelihood, without the standardized prior term. Backends that read
    the prior and likelihood *separately* rather than through
    ``neg_log_posterior_fn`` -- tempered SMC's annealing path and
    importance-weighted HMC both reach ``InferenceContext.log_likelihood_fn``
    directly -- would otherwise still see the ordinary (mass-fixed-at-a-
    placeholder) likelihood while every other backend sees the profiled one,
    silently targeting a different posterior.
    """
    from tengri.inference.loss_functions import _unstandardize_parameters

    spec = fitter.spec
    free_names = fitter._free_names
    fixed_values = fitter._fixed_values
    stochastic = spec.stochastic
    model = fitter.model
    mass_name = fitter._profile_mass_name
    mass_prior = fitter._profile_mass_prior
    ell_lo, ell_hi = fitter._profile_mass_bounds
    data_type = fitter.data_type
    use_components = bool(getattr(fitter, "use_components", False))
    threaded_impl = _threaded_predict_impl(model, use_components)

    def loglik_unbounded(params_unbounded, data_args):
        """log p(d | xi) with the mass profiled out; no prior term."""
        params = _unstandardize_parameters(
            params_unbounded, spec, free_names, fixed_values, stochastic
        )
        if "redshift" in data_args:
            params = {**params, "redshift": data_args["redshift"]}
        A, mstar, chi2_min = _profile_stats(
            model,
            mass_name,
            params,
            data_args["data"],
            data_args["noise"],
            presence=data_args.get("presence"),
            data_type=data_type,
            use_components=use_components,
            jit_inputs=data_args.get("_jit_inputs"),
            threaded_impl=threaded_impl,
        )
        return -0.5 * chi2_min + _log_mass_integral(A, mstar, ell_lo, ell_hi, mass_prior)

    return loglik_unbounded


def build_profiled_loglikelihood_fn(fitter: Fitter):
    """Build ``loglikelihood_fn(free_params, data_args) -> scalar``, physical space.

    Companion to :func:`build_profiled_loss_fn` for the physical-parameter
    likelihood (``loss_functions.build_loglikelihood_fn``'s profiled
    counterpart), consumed by nested sampling (``backends/evidence.py``
    calls ``Fitter._get_or_build_loglikelihood_fn()`` directly). Without
    this override, an NSS evidence run under ``profile_mass`` would score
    every live point at the mass's fixed placeholder instead of its
    marginal likelihood.
    """
    spec = fitter.spec
    fixed_values = fitter._fixed_values
    model = fitter.model
    mass_name = fitter._profile_mass_name
    mass_prior = fitter._profile_mass_prior
    ell_lo, ell_hi = fitter._profile_mass_bounds
    data_type = fitter.data_type
    use_components = bool(getattr(fitter, "use_components", False))
    threaded_impl = _threaded_predict_impl(model, use_components)

    def loglikelihood_fn(free_params, data_args):
        """log p(d | params), physical params, mass profiled out, no prior."""
        params = dict(free_params)
        for name, val in fixed_values.items():
            params[name] = val
        params = spec.resolve_mirrors(params)
        if "redshift" in data_args:
            params = {**params, "redshift": data_args["redshift"]}
        A, mstar, chi2_min = _profile_stats(
            model,
            mass_name,
            params,
            data_args["data"],
            data_args["noise"],
            presence=data_args.get("presence"),
            data_type=data_type,
            use_components=use_components,
            jit_inputs=data_args.get("_jit_inputs"),
            threaded_impl=threaded_impl,
        )
        return -0.5 * chi2_min + _log_mass_integral(A, mstar, ell_lo, ell_hi, mass_prior)

    return loglikelihood_fn


# ── Reinserting the mass once inference is done ──────────────────────────────


def _reinsert_mass_fn(fitter: Fitter):
    """The compiled draws -> conditional-mass-draws map, cached on the model.

    One ``jax.vmap`` over 1200 draws of a forward prediction is ~1.2 s when
    dispatched eagerly (op-by-op under batching) and ~5 s the first time in a
    process; under ``jax.jit`` it is one program. The jitted function closes
    over the model, the fixed values and the mass prior -- everything the
    fitter's engine cache key already covers -- and takes the draws, the
    keys, the data, the noise and the presence mask as traced arguments, so
    it is keyed by ``_engine_cache_key`` alone and reused across fits on the
    same model, whatever the galaxy.
    """
    from tengri.inference._model_cache import _default_owner as _model_cache_owner

    cache = _model_cache_owner.get_or_compile_model(fitter.model).setdefault(
        "profile_mass_reinsert", {}
    )
    cache_key = fitter._engine_cache_key()
    fn = cache.get(cache_key)
    if fn is not None:
        return fn

    model = fitter.model
    mass_name = fitter._profile_mass_name
    mass_prior = fitter._profile_mass_prior
    ell_lo, ell_hi = fitter._profile_mass_bounds
    fixed_values = fitter._fixed_values
    data_type = fitter.data_type
    use_components = bool(getattr(fitter, "use_components", False))

    def _reinsert(samples_no_mass, draw_keys, data, noise, presence):
        def stats_one(sample_dict):
            phys = {**fixed_values, **sample_dict}
            return _profile_stats(
                model,
                mass_name,
                phys,
                data,
                noise,
                presence=presence,
                data_type=data_type,
                use_components=use_components,
            )

        A_all, mstar_all, _ = jax.vmap(stats_one)(samples_no_mass)
        return jax.vmap(lambda k, a, m: _sample_log_mass(k, a, m, ell_lo, ell_hi, mass_prior))(
            draw_keys, A_all, mstar_all
        )

    fn = jax.jit(_reinsert)
    cache[cache_key] = fn
    return fn


def finalize_profile_mass(fitter: Fitter, posterior: Posterior, *, key) -> Posterior:
    """Record the resolved ``profile_mass`` choice, and reinsert the mass if engaged.

    Called once from ``Fitter.run()``, immediately after the backend runner
    returns, for every fit regardless of whether profiling engaged: the
    resolved choice and its reason are always recorded so ``"auto"`` is
    inspectable after the fact.

    When profiling engaged, draws (or, for a point estimate, sets) the
    marginalized mass and merges it into ``posterior.samples``/``params`` so
    they carry the mass parameter exactly as they would without profiling
    (``posterior.properties`` and other derived quantities work unchanged).

    Parameters
    ----------
    fitter : Fitter
        The fitter that produced ``posterior``.
    posterior : Posterior
        The backend's result, not yet returned to the caller.
    key : jax.Array
        The fit's PRNG key; the mass draw uses a key folded in from it, so it
        is reproducible alongside the rest of the fit.

    Returns
    -------
    Posterior
        ``posterior``, mutated in place and returned for convenience.
    """
    posterior.diagnostics = {
        **posterior.diagnostics,
        "profile_mass": fitter._profile_mass_resolved,
        "profile_mass_resolved": fitter._profile_mass_resolved,
        "profile_mass_reason": fitter._profile_mass_reason,
    }
    if not fitter._profile_mass:
        return posterior

    mass_name = fitter._profile_mass_name
    model = fitter.model
    data, noise = fitter.data, fitter.noise
    presence = None if fitter.presence is None else jnp.asarray(fitter.presence)
    fixed_values = fitter._fixed_values
    data_type = fitter.data_type
    use_components = bool(getattr(fitter, "use_components", False))

    if posterior.samples is not None:
        samples_no_mass = {k: v for k, v in posterior.samples.items() if k != mass_name}
        n_draws = next(iter(samples_no_mass.values())).shape[0]
        mass_key = jax.random.fold_in(key, abs(hash("tengri.profile_mass")) % (2**31))
        draw_keys = jax.random.split(mass_key, n_draws)
        ell_samples = _reinsert_mass_fn(fitter)(samples_no_mass, draw_keys, data, noise, presence)

        posterior.samples = {**posterior.samples, mass_name: ell_samples}
        posterior.params = {**posterior.params, mass_name: jnp.mean(ell_samples)}
    else:
        phys = {**fixed_values, **{k: v for k, v in posterior.params.items() if k != mass_name}}
        _, mstar, _ = _profile_stats(
            model,
            mass_name,
            phys,
            data,
            noise,
            presence=presence,
            data_type=data_type,
            use_components=use_components,
        )
        posterior.params = {**posterior.params, mass_name: jnp.log10(mstar)}

    return posterior

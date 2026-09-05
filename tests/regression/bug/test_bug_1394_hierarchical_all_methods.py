# SPDX-License-Identifier: BSD-3-Clause
"""Every registered backend is reachable hierarchically — and still guarded.

``PopulationFitter`` accepted 8 of 20 registered backends. The other 12 raised
``ValueError``, not because hierarchical inference is incompatible with them but
because the only flat-vector formulation lived *inside* ``_run_raytrace`` as
closures that exactly one sampler could reach.

``_hierarchical_flat`` lifts that out. These tests pin the three properties that
make the result safe rather than merely wide:

1. every registered backend resolves to *some* runner (no silent gaps),
2. ``tier="broken"`` backends stay gated — reachable is not unguarded,
3. no method is silently substituted for another.

Cheap by construction: they assert on dispatch structure, not on fits. A real
hierarchical fit costs 1.5-9.4 GB (measured), which does not belong in the
regression shard (#1346).
"""

from __future__ import annotations

import inspect
import re

import pytest

from tengri.inference._backend_registry import _BACKENDS
from tengri.inference._hierarchical_flat import (
    FLAT_SAMPLERS,
    FLAT_UNSUPPORTED,
    build_flat_problem,
)
from tengri.inference.hierarchical import PopulationFitter

pytestmark = pytest.mark.regression_bug


def _method_map_keys():
    """The hand-written table inside ``PopulationFitter.run``."""
    src = inspect.getsource(PopulationFitter.run)
    body = src.split("_method_map = {")[1].split("\n        }")[0]
    return set(re.findall(r'"([a-z_0-9]+)":', body))


def _tier(name):
    e = _BACKENDS[name]
    return getattr(e, "tier", None) or (e.get("tier") if isinstance(e, dict) else None)


def test_every_registered_backend_is_accounted_for():
    """No backend is silently absent: it either has a runner or a stated reason.

    Every registered backend is either driven (``_method_map`` or
    ``FLAT_SAMPLERS``) or refused with an explanation (``FLAT_UNSUPPORTED``) —
    that is the honest state; see the module docstring. The failure this
    guards is a backend that is missing from BOTH tables, i.e. refused with a
    generic error and no reason anyone can find later.
    """
    accounted = _method_map_keys() | set(FLAT_SAMPLERS) | set(FLAT_UNSUPPORTED)
    missing = sorted(set(_BACKENDS) - accounted)
    assert not missing, f"backends neither driven nor explained: {missing}"


def test_nss_is_driven_by_the_real_nested_slice_sampler():
    """The founding refusal is resolved by wiring, not by lowering the bar.

    The blind-rejection nested sampler first written here exhausted its attempt
    budget and terminated at iteration 147 of a requested 200 on a 2-galaxy
    D=18 problem, returning silently truncated -- therefore biased -- samples.
    Removed rather than shipped, and the refusal recorded (#1429). The
    resolution is the in-tree Nested Slice Sampler (Yallup+2026 -- constrained
    HRSS exploration WITHIN the likelihood contour, the same implementation
    the single-galaxy ``nss`` backend runs), driven on the seam's standardized
    problem with the exact probit prior transform it was always waiting for.
    """
    assert "nss" in FLAT_SAMPLERS, "the refusal is resolved; nss has a real driver (#1429)"
    assert FLAT_SAMPLERS["nss"] == "nss"
    assert "nss" not in FLAT_UNSUPPORTED
    # The wiring must call the real sampler, not a rejection stand-in: the
    # driver builds the NSS algorithm and hands it the LIKELIHOOD alone (the
    # prior lives in the live-point draws and the probit transform).
    src = inspect.getsource(
        __import__("tengri.inference._hierarchical_flat", fromlist=["x"]).run_flat_sampler
    )
    assert "as_top_level_api" in src, "the driver must run the in-tree NSS, not a stand-in"
    assert "log_likelihood_with_data" in src


def test_raytrace_and_the_seam_share_ONE_posterior_definition():
    """The seam's central claim, made structural rather than asserted.

    ``_run_raytrace`` used to build its own ``init``, its own ``ravel_pytree``
    and its own ``log_prob`` inline -- ~135 lines textually equivalent to
    ``build_flat_problem`` but structurally independent, so nothing stopped the
    two from drifting into sampling different distributions while every
    docstring claimed they agreed. It now calls the shared builder.

    Verified bit-for-bit at the time of the change: raytrace on a fixed key
    returned sigma=1.667, tau=154.89 both before and after.
    """
    src = inspect.getsource(PopulationFitter._run_raytrace)
    assert "build_flat_problem(" in src, (
        "raytrace must use the shared posterior, not a private copy"
    )
    assert "prob.extract_shared" in src, "the latent->physical map must be shared too"
    assert "def log_prob(" not in src, "a second log_prob has reappeared in raytrace"


def test_broken_tier_backends_stay_gated():
    """Reachable is not unguarded.

    ``pathfinder`` on this path was measured to OOM-kill the process outright
    (SIGKILL, exit 137) on a 2-galaxy D=18 problem. That measurement stands and
    is why the gate matters here — but it is no longer "exactly what its tier
    records": #231 moved ``pathfinder`` to ``experimental`` after establishing
    that its ``tier="broken"`` label came from a harness that never read a child
    return code, and its short_doc now carries the D=18 OOM as a caveat instead.
    The gate this test pins is unaffected, since it is about ``check_usable``
    being applied at all, and the subject is derived from the registry by
    :func:`_a_broken_tier_flat_method` rather than named.
    """
    src = inspect.getsource(
        __import__("tengri.inference._hierarchical_flat", fromlist=["x"]).run_flat_sampler
    )
    assert "check_usable(" in src, "the flat path must apply the same gate as Fitter.run"
    assert "allow_unvalidated" in src, "the opt-in must be threaded, not hardcoded"


def test_no_method_is_silently_substituted_for_another():
    """A ``FLAT_SAMPLERS`` entry must run the algorithm its name promises.

    ``mcmc_ess`` used to be rewritten to ``native_vi_linear`` with no warning.
    That silently handed back MGVI — and after #231 a tier="broken" backend — to
    a caller who asked for elliptical slice sampling, with nothing in the result
    to reveal it. Silent substitution is never the right repair for an
    unsupported method: support it, or raise.

    The seam's first draft repeated the pattern one layer down: five
    distinct-algorithm names (ESS, dynamic HMC, GHMC, MCLMC, adjusted MCLMC)
    all mapped onto the plain static-leapfrog ``"hmc"`` driver, and ``laplace``
    onto the bare ``"map"`` point estimate — the result's diagnostics recorded
    the requested name while a different algorithm ran. Until a name's real
    driver is wired at the seam, the honest state is refusal with a stated
    reason and a working alternative.
    """
    src = inspect.getsource(PopulationFitter.run)
    assert "_HIERARCHICAL_OVERRIDES" not in src, (
        "a silent method-substitution table has come back; support the method "
        "through the flat seam or raise, but do not swap it out"
    )

    # Every surviving entry names the algorithm its driver actually runs. An
    # addition here is welcome exactly when its real driver is wired — edit
    # this set in the same commit as the wiring, never before.
    # mcmc_dynamic_hmc and mcmc_ghmc joined when the seam gained their real
    # _shared.py full-scan drivers (they spent one commit refused, wired next);
    # mcmc_ess followed once _ess_full_scan existed — the flat prior is exactly
    # the iid N(0,1) its ellipse assumes; the MCLMC pair joined with their
    # blackjax (adjusted_)mclmc_find_L_and_step_size tuning, the piece whose
    # absence had kept them refused.
    assert set(FLAT_SAMPLERS) == {
        "mcmc",
        "mcmc_nuts",
        "mcmc_hmc",
        "mcmc_dynamic_hmc",
        "mcmc_ghmc",
        "mcmc_ess",
        "mcmc_mclmc",
        "mcmc_adjusted_mclmc",
        "map",
        "laplace",
        "pathfinder",
        "nss",
    }, (
        "FLAT_SAMPLERS gained or lost a name; if the new name's driver truly "
        "implements that algorithm, update this set in the same commit"
    )
    # laplace joined once its driver computed what distinguishes it from map:
    # a Gaussian covariance from the negative Hessian at a GRADIENT-VERIFIED
    # mode (#1537: curvature off a mode is a plausible wrong answer). nss —
    # the founding refusal — joined last, when the in-tree Nested Slice
    # Sampler was driven on the standardized problem (#1429), emptying
    # FLAT_UNSUPPORTED for the first time.


class _SentinelReached(Exception):
    """Raised by the stubbed builder: reaching it proves every gate passed."""


def _a_broken_tier_flat_method() -> str:
    """A ``tier="broken"`` backend this seam actually drives, read from the registry.

    Derived rather than named. Both tests below need "a method the tier gate must
    refuse", and both used to spell that ``"pathfinder"`` -- which stopped being
    true when #231 promoted it to ``experimental`` on the evidence that its
    quarantine label was a harness artifact. The forwarding test then still
    passed while proving nothing (an experimental method dispatches with or
    without the opt-in), and only its control went red. A hard-coded example of a
    category is a second census of the registry, and this file already exists
    because of what those cost.
    """
    for name in _BACKENDS:
        if _tier(name) == "broken" and name in FLAT_SAMPLERS:
            return name
    pytest.skip("no tier='broken' backend is driven by this seam; the gate has no subject")


def test_the_allow_unvalidated_opt_in_reaches_the_inner_gate(monkeypatch):
    """``run()`` must forward ``allow_unvalidated`` into ``run_flat_sampler``.

    The seam documents the opt-in as required for its tier="broken" names, but
    ``PopulationFitter.run`` declares ``allow_unvalidated`` as a named kwarg —
    so it is CONSUMED from ``**kwargs``, and without explicit forwarding the
    inner ``check_usable`` always sees the default False. The documented
    opt-in then refuses every flat-seam broken-tier method even when the
    caller said yes.

    The fit itself is irrelevant here, so ``build_flat_problem`` is replaced
    with a sentinel; with the builder stubbed out, nothing touches the fitter
    before the sentinel fires, so a bare uninitialized instance suffices.
    """
    import tengri.inference._hierarchical_flat as hf

    def _sentinel(*args, **kwargs):
        raise _SentinelReached

    monkeypatch.setattr(hf, "build_flat_problem", _sentinel)
    stub = object.__new__(PopulationFitter)

    with pytest.raises(_SentinelReached):
        PopulationFitter.run(stub, _a_broken_tier_flat_method(), allow_unvalidated=True)


def test_the_gate_still_refuses_a_broken_tier_method_without_the_opt_in(monkeypatch):
    """Control for the forwarding test: no opt-in, no dispatch.

    If this ever reaches the sentinel, the outer gate is gone and the
    forwarding test above is proving nothing.
    """
    import tengri.inference._hierarchical_flat as hf

    def _sentinel(*args, **kwargs):
        raise _SentinelReached

    monkeypatch.setattr(hf, "build_flat_problem", _sentinel)
    stub = object.__new__(PopulationFitter)

    method = _a_broken_tier_flat_method()
    with pytest.raises(Exception) as exc:
        PopulationFitter.run(stub, method)
    assert not isinstance(exc.value, _SentinelReached), (
        "the outer tier gate is gone: a broken-tier method dispatched with no opt-in"
    )
    assert method in str(exc.value) or "unvalidated" in str(exc.value).lower()


def test_the_unknown_method_error_derives_its_list():
    """The advertised list must come from the tables, never a prose literal.

    The literal it replaced named ``vi_nonlinear_fast`` "(default)" for months
    after b7c4fa1e2 moved the default off it, and separately advertised two
    ``tier="broken"`` backends that ``refuse_if_broken`` had already rejected
    three lines earlier — advice that raises when taken (#1576).

    Asserted against the **produced message**. An earlier version of this test
    grepped ``run`` for the expression ``sorted(set(_method_map) |
    set(FLAT_SAMPLERS)``, which pinned the shape of one particular fix rather
    than the property that matters: it went red when the derivation moved into
    ``_unknown_method_message`` even though the behavior was unchanged, and it
    could never have caught a message built correctly but never raised.

    ``run`` dispatches from two tables. Both must be represented, so the
    derivation cannot narrow to either one alone.
    """
    message = PopulationFitter._unknown_method_message("__nope__", {"vi_nonlinear_fast": None})

    assert "'vi_nonlinear_fast'" in message, "the NIFTy _method_map is missing from the advice"
    assert "'mcmc_nuts'" in message, "the flat seam (FLAT_SAMPLERS) is missing from the advice"

    # And nothing the caller would be refused for taking.
    advertised_broken = sorted(
        name
        for name, entry in _BACKENDS.items()
        if getattr(entry, "tier", None) == "broken" and repr(name) in message
    )
    assert not advertised_broken, (
        f"the advice recommends tier='broken' backends that run() refuses: "
        f"{advertised_broken}; got: {message}"
    )


@pytest.mark.parametrize("name", sorted(FLAT_SAMPLERS))
def test_every_flat_sampler_names_a_real_backend_and_driver(name):
    """No entry may name a backend that does not exist or a driver that is not implemented."""
    assert name in _BACKENDS, f"{name!r} is in FLAT_SAMPLERS but not registered"
    driver = FLAT_SAMPLERS[name]
    assert driver in {
        "nuts",
        "hmc",
        "dynamic_hmc",
        "ghmc",
        "ess",
        "mclmc",
        "adjusted_mclmc",
        "nuts_pathfinder",
        "map",
        "laplace",
        "nss",
    }, f"{name!r} declares unknown driver {driver!r}"


def test_declared_priors_map_exactly_through_the_seam():
    """Any declared prior is realized EXACTLY via its own pushforward (#1651).

    The seam used to route every free parameter through the Uniform box map
    (``to_bounded``), silently replacing a declared ``Gaussian`` with
    Uniform-over-truncation-bounds — refused since the hardening PR. The
    distributions have carried the exact N(0,1) pushforward all along
    (``unstandardize``, the classes' declared single source of truth, used by
    ``sample`` and by the single-galaxy unbounded machinery); the seam now
    builds its physical map from it, so the standardized space realizes the
    DECLARED prior for every distribution — the #1651 quantile map under the
    name it already had.
    """
    import jax.numpy as jnp

    from tengri.inference._hierarchical_flat import _physical_map
    from tengri.parameters.priors import Gaussian, Uniform

    class _Spec:
        def __init__(self, dists):
            self._dists = dists

        def get_distribution(self, name):
            return self._dists[name]

    g = Gaussian(2.0, 0.5, lo=1.0, hi=3.0)
    spec = _Spec({"a": Uniform(0.0, 1.0), "b": g})
    phys = _physical_map(spec, ["a", "b"])
    u = jnp.array(0.7)
    assert jnp.allclose(phys["b"](u), g.unstandardize(u)), (
        "the seam's map must BE the distribution's own pushforward"
    )
    assert jnp.allclose(phys["a"](u), spec.get_distribution("a").unstandardize(u))

    class _NoPushforward:
        """A distribution-like object without the standardization contract."""

    with pytest.raises(NotImplementedError) as exc:
        _physical_map(_Spec({"a": _NoPushforward()}), ["a"])
    msg = str(exc.value)
    assert "a" in msg and "unstandardize" in msg


def test_every_prior_pushforward_is_the_declared_density():
    """unstandardize really is the exact quantile map, class by class.

    Change of variables: pushing u ~ N(0,1) through ``unstandardize`` implies
    the physical density phi(u) / |d unstandardize/du|, which must equal
    ``exp(log_prob(theta))`` — both are normalized, so agreement is exact,
    not up to a constant. This is the load-bearing claim behind fitting any
    declared prior hierarchically (#1651); a class whose ``unstandardize``
    drifted from its ``log_prob`` would silently fit a wrong prior.
    """
    import jax
    import jax.numpy as jnp

    from tengri.parameters.priors import (
        Gaussian,
        Laplace,
        LogNormal,
        LogUniform,
        StudentT,
        Uniform,
    )

    dists = {
        "uniform": Uniform(0.5, 3.5),
        "gaussian": Gaussian(2.0, 0.5),
        "gaussian_trunc": Gaussian(2.0, 0.5, lo=1.0, hi=3.0),
        "loguniform": LogUniform(1e-2, 1e2),
        "lognormal": LogNormal(0.5, 0.4),
        "studentt_df2": StudentT(2.0, 0.5, df=2),
        "laplace": Laplace(1.0, 0.3),
    }
    # Even point count: u=0 is Laplace's quantile-map kink (the median), where
    # autodiff returns the one-sided derivative of a piecewise branch and the
    # comparison spuriously reads +inf. Measured: every off-kink point agrees
    # to ~1e-16; only the kink itself disagrees. Avoid it rather than widen
    # the tolerance.
    u_grid = jnp.linspace(-1.8, 1.8, 8)
    log_phi = -0.5 * u_grid**2 - 0.5 * jnp.log(2 * jnp.pi)
    for name, dist in dists.items():
        theta = jax.vmap(dist.unstandardize)(u_grid)
        dtheta_du = jax.vmap(jax.grad(lambda u, d=dist: d.unstandardize(u)))(u_grid)
        implied = log_phi - jnp.log(jnp.abs(dtheta_du))
        declared = jax.vmap(dist.log_prob)(theta)
        assert jnp.allclose(implied, declared, atol=1e-5), (
            f"{name}: unstandardize's pushforward density disagrees with "
            f"log_prob — max |diff| = {float(jnp.max(jnp.abs(implied - declared))):.2e}"
        )


def test_a_frozen_chain_is_refused_not_returned():
    """Every MCMC driver must refuse a chain that never moved.

    #1530's lesson generalized: MAP-echo draws look like a plausible answer.
    ``_require_finite_tuning`` catches the MCLMC starved-tuner cause; this is
    the effect-side net for every driver — a retained chain whose draws are
    all identical is not a posterior, whatever produced it.
    """
    import jax.numpy as jnp

    from tengri.inference._hierarchical_flat import _require_moving_chain
    from tengri.inference.hierarchical import DegenerateChainError

    moving = jnp.array([[0.0, 1.0], [0.1, 1.0], [0.2, 0.9]])
    _require_moving_chain(moving, "mcmc_nuts")  # passes silently

    frozen = jnp.tile(jnp.array([[0.5, -1.2]]), (60, 1))
    with pytest.raises(DegenerateChainError) as exc:
        _require_moving_chain(frozen, "mcmc_nuts")
    assert "mcmc_nuts" in str(exc.value)


def test_an_unknown_fit_kwarg_is_refused_not_swallowed():
    """``run_flat_sampler(..., **_ignored)`` was a silent kwarg sink.

    The #1378 standard: a typo'd fit option must fail loudly. Before this
    guard, ``n_samplse=1000`` (or ``init_from=...``, which the hierarchical
    surface documents as unsupported) vanished silently and the fit ran with
    defaults while the caller believed otherwise.
    """
    from tengri.inference._hierarchical_flat import run_flat_sampler

    with pytest.raises(TypeError) as exc:
        run_flat_sampler(object(), "map", key=None, n_samplse=1000)
    msg = str(exc.value)
    assert "n_samplse" in msg, "the error must name the unknown kwarg"
    assert "n_samples" in msg, "the error must show the accepted options"


def test_newton_polish_reaches_the_mode_adam_crawls_toward():
    """Second-order polish converges where first-order plateaus.

    Measured on the D=516 reference fixture: Adam's max |grad| was 1.13e3
    after 300 steps and still 84.6 after 8000 — first-order alone cannot
    reach a Laplace-grade mode there in any practical budget. The polish
    reuses the exact Hessian the covariance needs, so it costs ~one
    Hessian per iteration and converges quadratically inside the basin.
    """
    import jax.numpy as jnp

    from tengri.inference._hierarchical_flat import _newton_polish

    a = jnp.array([3.0, 0.5, 10.0])
    center = jnp.array([1.0, -2.0, 0.3])

    class _Prob:
        n_dim = 3

        @staticmethod
        def log_prob(x):
            return -0.5 * jnp.sum(a * (x - center) ** 2)

    start = center + jnp.array([0.8, -1.5, 0.2])
    mode = _newton_polish(_Prob, start, tol=1e-8, max_iters=12)
    import jax

    gmax = float(jnp.max(jnp.abs(jax.grad(_Prob.log_prob)(mode))))
    assert gmax <= 1e-8, f"polish left max |grad| = {gmax:.2e}"
    assert jnp.allclose(mode, center, atol=1e-8)


def test_laplace_refuses_curvature_off_a_mode():
    """#1537's lesson, enforced: no covariance without a verified mode.

    The single-galaxy laplace expanded about non-modes with no grad=0 check
    and returned plausible wrong answers (#1537). The hierarchical driver
    verifies the gradient at the reached point and refuses loudly — naming
    the knob (map_steps) — rather than handing back error bars measured
    around a point that is not the posterior's mode.
    """
    import jax.numpy as jnp

    from tengri.inference._hierarchical_flat import _require_converged_mode

    _require_converged_mode(jnp.array([1e-5, -2e-5]), "laplace", 300, tol=1e-3)  # passes

    with pytest.raises(RuntimeError) as exc:
        _require_converged_mode(jnp.array([0.5, -0.01]), "laplace", 300, tol=1e-3)
    msg = str(exc.value)
    assert "map_steps" in msg, "the error must name the knob that fixes it"
    assert "mode" in msg


def test_laplace_refuses_non_negative_definite_curvature():
    """A Cholesky that NaNs means the reached point is not a maximum.

    ``jnp.linalg.cholesky`` returns NaNs (no exception) for a non-PSD input;
    sampling from those NaNs would return a posterior of NaNs or, worse,
    garbage that passes a finite-check downstream. Refuse by name instead.
    """
    import jax.numpy as jnp

    from tengri.inference._hierarchical_flat import _require_psd_curvature

    good = jnp.linalg.cholesky(jnp.eye(3))
    _require_psd_curvature(good, "laplace")  # passes

    bad = jnp.linalg.cholesky(jnp.array([[1.0, 0.0], [0.0, -1.0]]))  # NaNs
    with pytest.raises(RuntimeError) as exc:
        _require_psd_curvature(bad, "laplace")
    assert "mode" in str(exc.value) or "definite" in str(exc.value)


def test_a_non_finite_tuning_is_refused_not_returned():
    """A starved MCLMC tuner must raise, not hand back a frozen chain.

    Measured on the 2-galaxy D=516 fixture: adjusted MCLMC with
    ``n_warmup=60`` returned ``L=nan``, ``step_size=nan`` and 60 copies of
    the init point — the #1530 MAP-echo failure, one driver over. The
    fraction-based tuning phases (``frac_tune1=0.1`` etc.) starve below a
    few hundred steps; at ``n_warmup=500`` the same fixture tunes finite.
    A frozen chain that LOOKS like a populated posterior must be a loud
    ``DegenerateChainError``, exactly as #1569 made raytrace's degeneracy.
    """
    import jax.numpy as jnp

    from tengri.inference._hierarchical_flat import _require_finite_tuning
    from tengri.inference.hierarchical import DegenerateChainError

    # finite tuning passes silently
    _require_finite_tuning(jnp.array(2.45), jnp.array(1.23), "mcmc_adjusted_mclmc", 500)

    with pytest.raises(DegenerateChainError) as exc:
        _require_finite_tuning(jnp.array(jnp.nan), jnp.array(jnp.nan), "mcmc_adjusted_mclmc", 60)
    msg = str(exc.value)
    assert "n_warmup" in msg, "the error must name the knob that fixes it"
    assert "mcmc_adjusted_mclmc" in msg, "the error must name the method asked for"


def test_nss_refuses_a_live_set_smaller_than_the_dimension():
    """HRSS directions come from the live points' covariance -- rank matters.

    ``n_live`` points give the empirical covariance rank at most
    ``n_live - 1``; with ``n_live <= D`` every slice direction lies in a
    proper subspace and the orthogonal complement is NEVER explored --
    silent bias, not slowness. The default n_live=500 therefore refuses the
    D=516 reference fixture by construction, with an error naming the knob
    (nss_n_live) and the working alternatives, instead of returning samples
    confined to a hyperplane that pass every finite-check downstream.
    """
    from tengri.inference._hierarchical_flat import _require_nondegenerate_live_set

    _require_nondegenerate_live_set(100, 20, "nss")  # passes: 100 live points span D=20

    with pytest.raises(ValueError) as exc:
        _require_nondegenerate_live_set(500, 516, "nss")
    msg = str(exc.value)
    assert "nss_n_live" in msg, "the error must name the knob that fixes it"
    assert "516" in msg, "the error must name the dimension it cannot span"
    assert "nss" in msg, "the error must name the method asked for"


def test_nss_refuses_an_unconverged_evidence_integral():
    """#1429's founding failure, as a runtime invariant rather than a probe.

    The rejection stand-in's defect was terminating early and returning a
    silently truncated -- therefore biased -- sample set. The real sampler
    can reach the same state by a different road: hitting
    ``nss_max_iterations`` while ``log(Z_live/Z)`` still exceeds tolerance
    means the live set still holds unintegrated posterior mass, and
    resampling then systematically misses the peak. Loud refusal naming the
    knob, not a diagnostics field nobody reads.
    """
    from tengri.inference._hierarchical_flat import _require_converged_evidence

    # converged: remaining evidence fraction below tolerance passes silently
    _require_converged_evidence(412, 10000, -3.5, -3.0, "nss")

    with pytest.raises(RuntimeError) as exc:
        _require_converged_evidence(10000, 10000, 4.2, -3.0, "nss")
    msg = str(exc.value)
    assert "nss_max_iterations" in msg, "the error must name the knob that fixes it"
    assert "#1429" in msg, "the error must cite the failure class it prevents"


def test_flat_problem_exposes_a_separable_posterior():
    """log_prob must be log_likelihood + log_prior, or nested sampling is wrong.

    Nested sampling handles the prior via the unit-cube transform and must be
    given the LIKELIHOOD alone. If the two were entangled, ``nss`` would be
    double-counting the prior and silently sampling the wrong distribution.
    """
    sig = inspect.signature(build_flat_problem)
    assert {"key", "memory_mode"} <= set(sig.parameters)
    fields = build_flat_problem.__doc__
    assert "FlatProblem" in fields
    from tengri.inference._hierarchical_flat import FlatProblem

    ann = set(FlatProblem.__dataclass_fields__)
    assert {"log_likelihood", "log_prior", "log_prob", "prior_transform"} <= ann

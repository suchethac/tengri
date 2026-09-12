# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for the ``Fitter(..., extra_log_prior=...)`` hook.

Surface protected: ``Fitter.__init__``'s ``extra_log_prior`` parameter and
its wiring into BOTH ``tengri.inference.loss_functions.build_logprior_fn``
(physical-space prior, used by nested sampling / evidence) AND
``build_loss_fn`` (the standardized-space objective that MAP, VI, and MCMC
all minimize via ``InferenceContext.neg_log_posterior_fn`` /
``Fitter._get_or_build_loss_fn``) -- the task-7/D7 fix: the informative AGN
priors in ``tengri.agn.priors`` were "unreachable from the public API," and
review round 1 (RULING R10) found that wiring the hook into only
``build_logprior_fn`` left it a silent no-op on every primary inference
backend. The physics-correctness of any specific prior (e.g.
``prior_energy_balance``) is covered in
``tests/crossval/test_agn_priors_vs_agnfitter.py`` and
``tests/contract/test_agn_priors.py``; this file only checks that a supplied
``extra_log_prior`` callable is actually invoked, on every objective, with
the documented ``(params, state)`` signature, and its return value folded in
by exact addition/subtraction -- not silently dropped, and (round 1) not
silently swapped between two Fitters sharing one Model object (the engine
cache is keyed on the Model, and ``extra_log_prior`` was not originally part
of ``Fitter._engine_cache_key()``).

Uses the session-scoped ``synthetic_ssp`` fixture (no ``data/ssp_*.h5``
needed, #613) so this runs in the default fast tier.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter
from tengri.inference.loss_functions import (
    _unstandardize_parameters,
    build_logprior_fn,
    build_loss_fn,
)
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


_PHOT = Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))


def _build_model(ssp):
    obs = Observation(photometry=_PHOT)
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8, 12)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    ), obs


def _extra_term(params, state):
    """A deliberately simple, non-physical extra term.

    Uses BOTH ``params`` (a free parameter) and ``state`` (a core
    ``ForwardState`` field guaranteed present regardless of which components
    are configured) so the test exercises the full documented contract
    without depending on any specific component's ``state.derived`` keys.
    """
    log_total_mass = params["sfh_dpl_log_total_mass"]
    return -0.5 * (log_total_mass - 10.0) ** 2 - 1e-6 * jnp.sum(state.sed_intrinsic**2)


@pytest.fixture(scope="module")
def _fitters(synthetic_ssp):
    model_a, _obs = _build_model(synthetic_ssp)
    model_b, _ = _build_model(synthetic_ssp)
    data = jnp.ones(len(_PHOT.filters))
    noise = 0.1 * jnp.ones_like(data)
    plain = Fitter(model_a, data, noise, data_type="photometry")
    hooked = Fitter(model_b, data, noise, data_type="photometry", extra_log_prior=_extra_term)
    return plain, hooked


def test_extra_log_prior_changes_logprior_by_exactly_the_extra_term(_fitters):
    plain, hooked = _fitters
    free_name = next(iter(plain._free_names))
    assert free_name == "sfh_dpl_log_total_mass"
    free_params = {free_name: 9.3}

    lp_plain = float(build_logprior_fn(plain)(free_params))
    lp_hooked = float(build_logprior_fn(hooked)(free_params))

    # Independently reconstruct what the extra term SHOULD be: merge fixed
    # values exactly as build_logprior_fn's hook path does, predict_state,
    # and evaluate _extra_term directly.
    params = dict(free_params)
    for name, val in hooked._fixed_values.items():
        params[name] = val
    params = hooked.spec.resolve_mirrors(params)
    state = hooked.model.predict_state(params)
    expected_extra = float(_extra_term(params, state))

    assert (lp_hooked - lp_plain) == pytest.approx(expected_extra, rel=1e-9, abs=1e-9)
    # And the extra term must be genuinely nonzero for this input, otherwise
    # the assertion above would pass vacuously even if the hook were ignored.
    assert abs(expected_extra) > 1e-6


def test_extra_log_prior_none_is_bit_identical_to_no_hook(synthetic_ssp):
    """Default ``extra_log_prior=None`` must not perturb the existing log-prior."""
    model_a, _obs = _build_model(synthetic_ssp)
    model_b, _ = _build_model(synthetic_ssp)
    data = jnp.ones(len(_PHOT.filters))
    noise = 0.1 * jnp.ones_like(data)
    plain = Fitter(model_a, data, noise, data_type="photometry")
    explicit_none = Fitter(model_b, data, noise, data_type="photometry", extra_log_prior=None)
    free_params = {"sfh_dpl_log_total_mass": 9.3}
    lp_plain = float(build_logprior_fn(plain)(free_params))
    lp_explicit_none = float(build_logprior_fn(explicit_none)(free_params))
    assert lp_plain == lp_explicit_none


def test_hooked_logprior_fn_is_jit_compiled(_fitters):
    _, hooked = _fitters
    logprior_fn = build_logprior_fn(hooked)
    free_params = {"sfh_dpl_log_total_mass": 9.3}
    eager = float(logprior_fn(free_params))
    jitted = float(jax.jit(logprior_fn)(free_params))
    assert jitted == pytest.approx(eager, rel=1e-9, abs=1e-9)


def test_non_callable_extra_log_prior_raises_type_error(synthetic_ssp):
    model, _obs = _build_model(synthetic_ssp)
    data = jnp.ones(len(_PHOT.filters))
    noise = 0.1 * jnp.ones_like(data)
    with pytest.raises(TypeError, match="extra_log_prior must be callable"):
        Fitter(model, data, noise, data_type="photometry", extra_log_prior="not callable")


# ── RULING R10: build_loss_fn / MAP / VI / MCMC objective ───────────────────


def _reconstruct_extra_term(fitter, params_unbounded):
    """Independently recompute what build_loss_fn's hook path should add.

    Mirrors ``_unstandardize_parameters`` + ``predict_state`` + ``_extra_term``
    exactly as the production code path does, so this checks WIRING (is the
    term actually folded in, with the right sign) rather than re-deriving the
    unstandardization math.
    """
    params = _unstandardize_parameters(
        params_unbounded,
        fitter.spec,
        fitter._free_names,
        fitter._fixed_values,
        fitter.spec.stochastic,
    )
    state = fitter.model.predict_state(params)
    return float(_extra_term(params, state))


def test_extra_log_prior_changes_loss_fn_by_exactly_the_extra_term(_fitters):
    """RULING R10: the standardized-space objective (MAP/VI/MCMC) must see
    the same extra term as build_logprior_fn (nested sampling), with the
    OPPOSITE sign (loss_fn is a quantity to MINIMIZE)."""
    plain, hooked = _fitters
    params_unbounded = {"sfh_dpl_log_total_mass": jnp.array(0.3)}

    loss_plain = float(build_loss_fn(plain)(params_unbounded, plain._data_args))
    loss_hooked = float(build_loss_fn(hooked)(params_unbounded, hooked._data_args))

    expected_extra = _reconstruct_extra_term(hooked, params_unbounded)

    assert (loss_plain - loss_hooked) == pytest.approx(expected_extra, rel=1e-9, abs=1e-9)
    assert abs(expected_extra) > 1e-6


def test_extra_log_prior_none_is_bit_identical_on_loss_fn(synthetic_ssp):
    model_a, _obs = _build_model(synthetic_ssp)
    model_b, _ = _build_model(synthetic_ssp)
    data = jnp.ones(len(_PHOT.filters))
    noise = 0.1 * jnp.ones_like(data)
    plain = Fitter(model_a, data, noise, data_type="photometry")
    explicit_none = Fitter(model_b, data, noise, data_type="photometry", extra_log_prior=None)
    params_unbounded = {"sfh_dpl_log_total_mass": jnp.array(0.3)}
    loss_plain = float(build_loss_fn(plain)(params_unbounded, plain._data_args))
    loss_none = float(build_loss_fn(explicit_none)(params_unbounded, explicit_none._data_args))
    assert loss_plain == loss_none


def test_hooked_loss_fn_is_jit_and_grad_compatible(_fitters):
    _, hooked = _fitters
    loss_fn = build_loss_fn(hooked)
    params_unbounded = {"sfh_dpl_log_total_mass": jnp.array(0.3)}
    data_args = hooked._data_args

    eager = float(loss_fn(params_unbounded, data_args))
    jitted = float(jax.jit(loss_fn)(params_unbounded, data_args))
    assert jitted == pytest.approx(eager, rel=1e-9, abs=1e-9)

    grad_val = jax.grad(lambda p: loss_fn(p, data_args))(params_unbounded)
    assert jnp.isfinite(grad_val["sfh_dpl_log_total_mass"])
    assert float(grad_val["sfh_dpl_log_total_mass"]) != 0.0


def test_map_objective_via_inference_context_honors_extra_log_prior(synthetic_ssp):
    """The literal MAP path: ``InferenceContext.neg_log_posterior_fn`` (used
    by ``inference/backends/map_dispatch.py``) delegates to
    ``Fitter._get_or_build_loss_fn()`` -- confirm the hook reaches it there,
    not just via the module-level ``build_loss_fn`` builder."""
    model_a, _obs = _build_model(synthetic_ssp)
    model_b, _ = _build_model(synthetic_ssp)
    data = jnp.ones(len(_PHOT.filters))
    noise = 0.1 * jnp.ones_like(data)
    plain = Fitter(model_a, data, noise, data_type="photometry")
    hooked = Fitter(model_b, data, noise, data_type="photometry", extra_log_prior=_extra_term)

    ctx_plain = InferenceContext.from_target(plain)
    ctx_hooked = InferenceContext.from_target(hooked)

    params_unbounded = {"sfh_dpl_log_total_mass": jnp.array(0.3)}
    loss_plain = float(ctx_plain.neg_log_posterior_fn(params_unbounded, plain._data_args))
    loss_hooked = float(ctx_hooked.neg_log_posterior_fn(params_unbounded, hooked._data_args))

    expected_extra = _reconstruct_extra_term(hooked, params_unbounded)
    assert (loss_plain - loss_hooked) == pytest.approx(expected_extra, rel=1e-9, abs=1e-9)


def test_extra_log_prior_does_not_leak_across_fitters_sharing_a_model(synthetic_ssp):
    """Regression for a bug found while wiring RULING R10: the loss-fn engine
    cache is keyed on the Model OBJECT
    (``_model_cache_owner.get_or_compile_model(self.model)``), so two
    Fitters sharing one Model -- the documented, encouraged pattern for
    reusing compiled XLA programs -- would silently share whichever
    ``build_loss_fn`` closure compiled first if ``extra_log_prior`` were not
    part of ``Fitter._engine_cache_key()``: a plain Fitter built after a
    hooked one on the SAME model would run the HOOKED objective (or vice
    versa), with no error and no warning."""
    model, _obs = _build_model(synthetic_ssp)
    data = jnp.ones(len(_PHOT.filters))
    noise = 0.1 * jnp.ones_like(data)

    plain = Fitter(model, data, noise, data_type="photometry")
    hooked = Fitter(model, data, noise, data_type="photometry", extra_log_prior=_extra_term)

    params_unbounded = {"sfh_dpl_log_total_mass": jnp.array(0.3)}
    loss_plain = float(plain._get_or_build_loss_fn()(params_unbounded, plain._data_args))
    loss_hooked = float(hooked._get_or_build_loss_fn()(params_unbounded, hooked._data_args))
    assert loss_plain != loss_hooked


def test_extra_log_prior_is_an_engine_ledger_row(synthetic_ssp):
    """The hook is keyed by ``ENGINE_POLICY``, as a ``content`` row (#2163 E.5).

    The hand-written key tuple that carried ``self._extra_log_prior`` by
    identity is gone; under the ledger the row is what separates the engines
    of two Fitters sharing one Model that differ only in the hook. Pinned at
    the key level (not only through the loss values above): ``None`` vs a
    hook differ, two Fitters handed the SAME function agree, and the entry is
    ``baked()``'s module-qualified name, never an address. Two distinct
    closures returned by one factory therefore share a key (same qualified
    name), which the identity keying did not do; distinct hooks should be
    distinct functions.
    """
    from tengri.inference._engine_policy import ENGINE_POLICY, FINGERPRINT_POLICY

    model, _obs = _build_model(synthetic_ssp)
    data = jnp.ones(len(_PHOT.filters))
    noise = 0.1 * jnp.ones_like(data)
    plain = Fitter(model, data, noise, data_type="photometry")
    hooked = Fitter(model, data, noise, data_type="photometry", extra_log_prior=_extra_term)
    hooked_again = Fitter(model, data, noise, data_type="photometry", extra_log_prior=_extra_term)

    assert ENGINE_POLICY["_extra_log_prior"][0] == "content"
    assert FINGERPRINT_POLICY["_extra_log_prior"][0] == "exclude"

    assert plain._engine_cache_key() != hooked._engine_cache_key()
    assert hooked._engine_cache_key() == hooked_again._engine_cache_key()

    entries = dict(hooked._engine_cache_key()[2])
    assert entries["_extra_log_prior"] == (
        "callable",
        f"{_extra_term.__module__}.{_extra_term.__qualname__}",
    )
    assert dict(plain._engine_cache_key()[2])["_extra_log_prior"] is None

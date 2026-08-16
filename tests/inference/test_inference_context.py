# SPDX-License-Identifier: BSD-3-Clause
"""Tests for InferenceContext — the Python-level seam between Fitter and backends.

Covers:
1. Construction from a Fitter and accessor wiring (spec, model, data_args).
2. Frozenness (cannot mutate attributes after construction).
3. JAX trace guard (``__jax_array__`` raises rather than silently leaking
   the context into a JIT key).
4. Dispatch through ``Fitter.run`` with a ``legacy_fitter=False`` backend
   actually receives an ``InferenceContext`` (not a ``Fitter``).
5. Dispatch through ``Fitter.run`` with a default backend
   (``legacy_fitter=True``) still receives the ``Fitter`` (no regression).

No SSP data required — uses MagicMock model and Parameters spec.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

jax.config.update("jax_enable_x64", True)

from tengri.inference._backend_registry import (
    _BACKENDS,
    BackendEntry,
    register_backend,
)
from tengri.inference.context import InferenceContext
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform


def _build_fitter():
    """Construct a minimal Fitter with a MagicMock model and tiny spec."""
    from tengri.inference.fitter import Fitter

    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_diff=Fixed(0.3),
        redshift=Fixed(0.1),
    )
    model = MagicMock()
    model.spec = spec
    model.predict_photometry.return_value = jnp.ones(3) * 1e-18
    data = jnp.ones(3) * 1e-18
    noise = jnp.ones(3) * 1e-19
    return Fitter(model, data, noise, data_type="photometry")


class TestContextAccessors:
    def test_spec_is_fitter_spec(self):
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        assert ctx.spec is fitter.spec

    def test_model_is_fitter_model(self):
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        assert ctx.model is fitter.model

    def test_data_args_is_fitter_data_args(self):
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        assert ctx.data_args is fitter._data_args

    def test_memory_mode_defaults_to_fast(self):
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        assert ctx.memory_mode == "fast"

    def test_posterior_chunk_size_defaults_to_none(self):
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        assert ctx.posterior_chunk_size is None

    def test_free_names_matches_fitter(self):
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        assert ctx.free_names == list(fitter._free_names)

    def test_likelihood_config_properties_delegate_to_fitter(self):
        """Step-D-prime properties surface Fitter's likelihood config so the
        ``Likelihood`` module (``inference/likelihood.py``) can read context
        instead of reaching into ``fitter._*`` private state.

        Verifies each new property reads the same value as the underlying
        Fitter attribute. If a future refactor moves storage off the Fitter,
        the property's getter changes but this test stays meaningful (the
        property still has to surface the same value).
        """
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        # Data + noise + masks
        assert ctx.data is fitter.data
        assert ctx.noise is fitter.noise
        assert ctx.data_mask is fitter.data_mask
        assert ctx.data_type == fitter.data_type
        assert ctx.has_spectroscopy == fitter._has_spectroscopy
        assert ctx.fixed_values is fitter._fixed_values
        # Calibration marginalization config
        assert ctx.calibration_marginalize == fitter._calibration_marginalize
        assert ctx.cal_n_poly == fitter._cal_n_poly
        assert ctx.cal_prior_sigma == fitter._cal_prior_sigma
        # E-line config
        assert ctx.eline_marginalize == fitter._eline_marginalize
        assert ctx.eline_fitted == fitter._eline_fitted
        assert ctx.eline_prior_type == fitter._eline_prior_type
        assert ctx.eline_prior_sigma == fitter._eline_prior_sigma
        assert ctx.eline_prior_width_dex == fitter._eline_prior_width_dex
        assert ctx.eline_independent_wavelengths is fitter._eline_independent_wavelengths
        assert ctx.eline_amplitude_names == fitter._eline_amplitude_names
        assert ctx.eline_wavelengths is fitter._eline_wavelengths
        assert ctx.eline_constraint_matrix is fitter._eline_constraint_matrix
        # CompileCache discoverability
        assert ctx.cache is fitter.cache


class TestHamiltonianSplit:
    """log_likelihood_fn + log_prior_fn split — paper §2 + §4 alignment.

    Inference methods like Nested Sampling need the data term and the
    prior term exposed separately, not bundled into neg_log_posterior_fn.
    These tests pin the split contract.
    """

    def test_log_prior_fn_is_standard_normal_quadratic(self):
        """log p(xi) = -0.5 * sum(xi^2) for the standardized N(0,I) prior.

        Paper §2 'Standardized Inference': every free parameter lives
        in the N(0,I) latent space after ``_unstandardize_parameters``.
        The log-prior is therefore the quadratic form irrespective of
        physical-prior type (Uniform, Gaussian, LogUniform).
        """
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        log_prior = ctx.log_prior_fn

        # At xi=0 (the mode), log-prior should be 0 (the constant
        # offset -D/2 * log(2pi) is dropped).
        xi_zero = {name: jnp.float64(0.0) for name in fitter._free_names}
        assert float(log_prior(xi_zero)) == 0.0

        # At xi=1 for every dim, log_prior = -0.5 * D
        xi_one = {name: jnp.float64(1.0) for name in fitter._free_names}
        D = len(fitter._free_names)
        assert float(log_prior(xi_one)) == pytest.approx(-0.5 * D)

        # At xi=2 for one dim, log_prior += -0.5 * (4 - 1) = -1.5
        xi_two_one = {name: jnp.float64(1.0) for name in fitter._free_names}
        xi_two_one[fitter._free_names[0]] = jnp.float64(2.0)
        assert float(log_prior(xi_two_one)) == pytest.approx(-0.5 * (D + 3))

    def test_log_prior_fn_is_pure_function(self):
        """log_prior_fn returns the same value across calls — pure JAX-friendly."""
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        xi = {name: jnp.float64(0.5) for name in fitter._free_names}
        v1 = float(ctx.log_prior_fn(xi))
        v2 = float(ctx.log_prior_fn(xi))
        assert v1 == v2

    def test_log_likelihood_fn_property_exists(self):
        """log_likelihood_fn is exposed and returns a callable."""
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        loglik_fn = ctx.log_likelihood_fn
        assert callable(loglik_fn)


class TestObjectiveIsJitCached:
    """neg_log_posterior_fn must be a JIT-compiled callable, not a raw closure.

    Regression for the 2026-07 precompute audit. ``neg_log_posterior_fn``
    (and ``loss_fn``) are documented as *"JIT-compiled" / "JIT-cached"* on the
    property and in ADR-0010, and callers pull the primitive straight out of
    the context to evaluate the objective. But the accessor returned a raw
    ``build_loss_fn`` closure, so a direct ``neg_log_posterior_fn(params,
    data_args)`` call re-ran the per-component chain dispatcher at Python level
    every evaluation. On a joint fit with a spectral-index / line-flux channel
    (where the feature forward ``predict_state`` is otherwise never fused) that
    was ~20x slower than the fused objective (27 ms -> 1.4 ms, value-identical),
    silently swamping the WavePrecomp LUT speedup for the DESI-style
    photometry + emission-line + Dn4000 use case.

    ``grad_fn`` was already protected (it wraps ``value_and_grad`` in its own
    ``jax.jit``); these tests pin the same guarantee on the objective itself so
    VI / nested-sampling / MAP-logging / custom loops don't regress. A
    ``jax.jit``-wrapped callable exposes ``.lower()`` / ``.trace()`` staging
    methods; a plain Python function does not.
    """

    def test_neg_log_posterior_fn_is_jit_compiled(self):
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        fn = ctx.neg_log_posterior_fn
        assert hasattr(fn, "lower") and hasattr(fn, "trace"), (
            "neg_log_posterior_fn must be a jax.jit-wrapped callable "
            "(exposes .lower()/.trace()); got a raw Python function, so the "
            "objective would run the component chain un-fused every evaluation."
        )

    def test_grad_fn_is_jit_compiled(self):
        """grad_fn parity — the already-protected sibling stays jit-wrapped."""
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        assert hasattr(ctx.grad_fn, "lower"), "grad_fn must stay jax.jit-wrapped"

    def test_deprecated_loss_fn_alias_is_also_jit_compiled(self):
        """The deprecated ``loss_fn`` alias returns the same jit'd callable."""
        import warnings

        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fn = ctx.loss_fn
        assert hasattr(fn, "lower") and hasattr(fn, "trace")


class TestContextFrozenness:
    def test_cannot_replace_fitter(self):
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        with pytest.raises((AttributeError, TypeError)):
            ctx.fitter = MagicMock()  # type: ignore[misc]

    def test_jax_trace_guard_raises(self):
        """Context must refuse to be traced by JAX — pulling it into a
        jit/vmap/scan boundary is a programming error we want loud."""
        fitter = _build_fitter()
        ctx = InferenceContext(fitter=fitter)
        with pytest.raises(TypeError, match="must not be traced by JAX"):
            ctx.__jax_array__()


class TestDispatchSwitch:
    """Fitter.run must pass the context (not the Fitter) when ``legacy_fitter=False``.

    Probes hijack the ``map`` registry entry (which is always present and
    canonical) so we don't have to monkeypatch ``_CANONICAL_METHODS``.
    """

    @staticmethod
    def _swap_entry(name: str, runner, *, legacy_fitter: bool):
        """Replace the registry entry for ``name`` and return the original."""
        original = _BACKENDS[name]
        _BACKENDS[name] = BackendEntry(
            name=name,
            runner=runner,
            tier=original.tier,
            short_doc=original.short_doc,
            requires=(),  # drop the requires-check so we don't need optax for a probe
            legacy_fitter=legacy_fitter,
        )
        return original

    def test_legacy_backend_receives_fitter(self):
        fitter = _build_fitter()
        received = {}

        def runner(context, *, key, init_from=None, **kw):
            received["target"] = context
            from tengri.inference.posterior import Posterior

            return Posterior(
                samples=None,
                params={},
                method="map",
                wall_time_s=0.0,
                diagnostics={},
                _model=fitter.model,
                _fitter=fitter,
            )

        original = self._swap_entry("map", runner, legacy_fitter=True)
        try:
            fitter.run("map", key=jax.random.PRNGKey(0))
        finally:
            _BACKENDS["map"] = original

        assert received["target"] is fitter

    def test_migrated_backend_receives_context(self):
        fitter = _build_fitter()
        received = {}

        def runner(context, *, key, init_from=None, **kw):
            received["target"] = context
            from tengri.inference.posterior import Posterior

            return Posterior(
                samples=None,
                params={},
                method="map",
                wall_time_s=0.0,
                diagnostics={},
                _model=fitter.model,
                _fitter=fitter,
            )

        original = self._swap_entry("map", runner, legacy_fitter=False)
        try:
            fitter.run("map", key=jax.random.PRNGKey(0))
        finally:
            _BACKENDS["map"] = original

        target = received["target"]
        assert isinstance(target, InferenceContext)
        assert target.fitter is fitter

    def test_register_backend_decorator_defaults_to_legacy(self):
        """Existing @register_backend(...) call sites should not change behavior."""

        @register_backend("_default_probe")
        def runner(target, *, key, init_from=None, **kw):
            from tengri.inference.posterior import Posterior

            return Posterior(
                samples=None,
                params={"target_type": type(target).__name__},
                method="_default_probe",
                wall_time_s=0.0,
                diagnostics={},
                _model=None,
                _fitter=None,
            )

        try:
            entry = _BACKENDS["_default_probe"]
            assert entry.legacy_fitter is True, (
                "@register_backend without legacy_fitter= must default to True "
                "so existing registrations stay unchanged."
            )
        finally:
            _BACKENDS.pop("_default_probe", None)

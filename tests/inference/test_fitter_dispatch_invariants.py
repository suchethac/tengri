# SPDX-License-Identifier: BSD-3-Clause
"""Invariant tests for Fitter method dispatch.

Covers:
1. resolve_method() — canonical-method validation and unknown-method errors.
2. "auto" and "mcmc" dimensionality-based routing — mocked _run_* dispatch.
3. _engine_cache_key() — stability and sensitivity to spec changes.

No SSP data required. resolve_method() is tested as a pure function.
Cache-key tests use a MagicMock model with a real Parameters spec.
Dispatch routing tests use a real Fitter with mocked _run_* methods.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


import warnings
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest

from tengri.config.exceptions import ParameterError
from tengri.inference.fitter import (
    _AUTO_D_THRESHOLD,
    _CANONICAL_METHODS,
    _MCMC_AUTO_D_THRESHOLD,
    resolve_method,
)
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

# ── Fixtures ──────────────────────────────────────────────────────


def _make_mock_model(spec: Parameters, n_filters: int = 3) -> MagicMock:
    """Return a MagicMock SEDModel wired to the given spec."""
    model = MagicMock()
    model.spec = spec
    model.predict_photometry.return_value = jnp.ones(n_filters) * 1e-18
    model.predict_spectrum.return_value = jnp.ones(50) * 1e-18
    return model


@pytest.fixture(scope="module")
def low_d_spec():
    """A small DPL spec with few free params (D < _AUTO_D_THRESHOLD)."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_diff=Fixed(0.3),
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def high_d_spec():
    """A spec with many free params (D > _AUTO_D_THRESHOLD)."""
    params: dict = {f"sfh_dpl_dummy_{i}": Uniform(0.0, 1.0) for i in range(25)}
    # We build a real dense_basis spec to get enough free parameters naturally
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_diff=Uniform(0.0, 2.0),
        redshift=Fixed(0.1),
    )


# ── resolve_method — unknown methods ──────────────────────────────


class TestResolveMethodUnknownMethod:
    @pytest.mark.parametrize(
        "bad_method",
        ["xyzzy", "mcmc_bayes", "VI", "NUTS", "", "nuts_fast", "geovi", "mgvi", "raytrace"],
    )
    def test_unknown_method_raises_parameter_error(self, bad_method: str) -> None:
        with pytest.raises(ParameterError):
            resolve_method(bad_method)

    def test_none_method_raises_parameter_error(self) -> None:
        with pytest.raises(ParameterError):
            resolve_method(None)  # type: ignore[arg-type]

    def test_error_message_lists_canonical_names(self) -> None:
        with pytest.raises(ParameterError, match="vi"):
            resolve_method("totally_invalid")


# ── resolve_method — canonical methods ────────────────────────────


class TestResolveMethodCanonical:
    @pytest.mark.parametrize("canonical", sorted(_CANONICAL_METHODS))
    def test_canonical_method_returns_unchanged(self, canonical: str) -> None:
        result = resolve_method(canonical)
        assert result == canonical, (
            f"resolve_method({canonical!r}) returned {result!r} instead of passing through"
        )

    @pytest.mark.parametrize("canonical", sorted(_CANONICAL_METHODS))
    def test_canonical_method_no_warning(self, canonical: str) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_method(canonical)

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not dep_warnings, (
            f"Canonical method {canonical!r} unexpectedly emitted DeprecationWarning"
        )


# ── Threshold constants — documentation contract ──────────────────


class TestThresholdConstants:
    """Threshold constants must match the documented values in CLAUDE.md / docstrings."""

    def test_auto_threshold_is_20(self) -> None:
        assert _AUTO_D_THRESHOLD == 20, (
            f"_AUTO_D_THRESHOLD changed from documented 20 to {_AUTO_D_THRESHOLD}"
        )

    def test_mcmc_threshold_is_20(self) -> None:
        assert _MCMC_AUTO_D_THRESHOLD == 20, (
            f"_MCMC_AUTO_D_THRESHOLD changed from documented 20 to {_MCMC_AUTO_D_THRESHOLD}"
        )

    def test_auto_threshold_matches_mcmc_threshold(self) -> None:
        """Both thresholds use the same value so 'auto' and 'mcmc' agree on the boundary."""
        assert _AUTO_D_THRESHOLD == _MCMC_AUTO_D_THRESHOLD


# ── "auto" dispatch routing — mock patching ───────────────────────


class TestAutoDispatchRouting:
    """'auto' must route to mcmc_nuts for D<=threshold, vi for D>threshold."""

    def _build_fitter(self, spec: Parameters):
        """Build a Fitter using a MagicMock model (no SSP data needed)."""
        from tengri.inference.fitter import Fitter

        model = _make_mock_model(spec, n_filters=3)
        noise = jnp.ones(3) * 1e-19
        data = jnp.ones(3) * 1e-18
        return Fitter(model, data, noise, data_type="photometry")

    def test_auto_low_d_routes_to_nuts(self, low_d_spec: Parameters) -> None:
        """auto with D <= threshold should dispatch the mcmc_nuts backend."""
        assert low_d_spec.n_free <= _AUTO_D_THRESHOLD, (
            f"Precondition: low_d_spec.n_free={low_d_spec.n_free} > {_AUTO_D_THRESHOLD}"
        )
        fitter = self._build_fitter(low_d_spec)
        # Mock the runner at the registry — that's the seam dispatch
        # actually goes through now (migrated backends bypass Fitter._run_*).
        from tengri.inference._backend_registry import _BACKENDS, BackendEntry

        sentinel = MagicMock(name="posterior_sentinel")
        sentinel._fitter = fitter
        calls: dict[str, int] = {"mcmc_nuts": 0, "vi_nonlinear_fast": 0}

        def make_runner(name):
            def runner(context, *, key, init_from=None, **kw):
                calls[name] += 1
                return sentinel

            return runner

        original_nuts = _BACKENDS["mcmc_nuts"]
        original_vi = _BACKENDS["vi_nonlinear_fast"]
        _BACKENDS["mcmc_nuts"] = BackendEntry(
            name="mcmc_nuts",
            runner=make_runner("mcmc_nuts"),
            tier="primary",
            short_doc="",
            requires=(),
            legacy_fitter=False,
        )
        _BACKENDS["vi_nonlinear_fast"] = BackendEntry(
            name="vi_nonlinear_fast",
            runner=make_runner("vi_nonlinear_fast"),
            tier="primary",
            short_doc="",
            requires=(),
            legacy_fitter=False,
        )
        try:
            fitter.run("auto", key=jax.random.PRNGKey(0))
        finally:
            _BACKENDS["mcmc_nuts"] = original_nuts
            _BACKENDS["vi_nonlinear_fast"] = original_vi

        assert calls["mcmc_nuts"] == 1, f"Expected mcmc_nuts called once, got {calls}"
        assert calls["vi_nonlinear_fast"] == 0

    def test_auto_high_d_routes_to_vi(self) -> None:
        """auto with D > threshold should dispatch the vi_nonlinear_fast backend.

        Pre-registry semantics (and the current registry-backed dispatch)
        send high-D ``auto`` to the ``vi_nonlinear_fast`` backend.
        """
        from tengri.inference._backend_registry import _BACKENDS, BackendEntry
        from tengri.inference.fitter import Fitter

        mock_spec = MagicMock()
        mock_spec.n_free = _AUTO_D_THRESHOLD + 5
        mock_spec.free_params = [f"p{i}" for i in range(_AUTO_D_THRESHOLD + 5)]
        mock_model = _make_mock_model(mock_spec, n_filters=3)
        noise = jnp.ones(3) * 1e-19
        data = jnp.ones(3) * 1e-18
        fitter = Fitter(mock_model, data, noise, data_type="photometry")

        calls: dict[str, int] = {"mcmc_nuts": 0, "vi_nonlinear_fast": 0}

        def make_runner(name):
            def runner(context, *, key, init_from=None, **kw):
                calls[name] += 1
                return MagicMock(name=f"{name}_posterior")

            return runner

        originals = {n: _BACKENDS[n] for n in calls}
        for n in originals:
            _BACKENDS[n] = BackendEntry(
                name=n,
                runner=make_runner(n),
                tier="primary",
                short_doc="",
                requires=(),
                legacy_fitter=False,
            )
        try:
            fitter.run("auto", key=jax.random.PRNGKey(0))
        finally:
            for n, e in originals.items():
                _BACKENDS[n] = e

        assert calls["vi_nonlinear_fast"] == 1
        assert calls["mcmc_nuts"] == 0


# ── "mcmc" dispatch routing — mock patching ───────────────────────


class TestMcmcDispatchRouting:
    """'mcmc' must route to _run_nuts for D<=threshold, _run_raytrace for D>threshold."""

    def _build_fitter_with_spec(self, spec: Parameters):
        from tengri.inference.fitter import Fitter

        model = _make_mock_model(spec, n_filters=3)
        noise = jnp.ones(3) * 1e-19
        data = jnp.ones(3) * 1e-18
        return Fitter(model, data, noise, data_type="photometry")

    @staticmethod
    def _swap_mcmc_runners():
        """Swap mcmc_nuts and mcmc_raytrace registry entries with probes.

        Returns ``(call_counts, restore_fn)``.
        """
        from tengri.inference._backend_registry import _BACKENDS, BackendEntry

        calls: dict[str, int] = {"mcmc_nuts": 0, "mcmc_raytrace": 0}

        def make_runner(name):
            def runner(context, *, key, init_from=None, **kw):
                calls[name] += 1
                return MagicMock(name=f"{name}_posterior")

            return runner

        originals = {n: _BACKENDS[n] for n in ("mcmc_nuts", "mcmc_raytrace")}
        for n in originals:
            _BACKENDS[n] = BackendEntry(
                name=n,
                runner=make_runner(n),
                tier="primary",
                short_doc="",
                requires=(),
                legacy_fitter=False,
            )

        def restore():
            for n, e in originals.items():
                _BACKENDS[n] = e

        return calls, restore

    def test_mcmc_low_d_routes_to_nuts(self, low_d_spec: Parameters) -> None:
        assert low_d_spec.n_free <= _MCMC_AUTO_D_THRESHOLD
        fitter = self._build_fitter_with_spec(low_d_spec)
        calls, restore = self._swap_mcmc_runners()
        try:
            fitter.run("mcmc", key=jax.random.PRNGKey(0))
        finally:
            restore()
        assert calls["mcmc_nuts"] == 1
        assert calls["mcmc_raytrace"] == 0

    def test_mcmc_high_d_routes_to_raytrace(self) -> None:
        from tengri.inference.fitter import Fitter

        mock_spec = MagicMock()
        mock_spec.n_free = _MCMC_AUTO_D_THRESHOLD + 5
        mock_spec.free_params = [f"p{i}" for i in range(_MCMC_AUTO_D_THRESHOLD + 5)]
        mock_model = _make_mock_model(mock_spec, n_filters=3)
        noise = jnp.ones(3) * 1e-19
        data = jnp.ones(3) * 1e-18
        fitter = Fitter(mock_model, data, noise, data_type="photometry")

        calls, restore = self._swap_mcmc_runners()
        try:
            fitter.run("mcmc", key=jax.random.PRNGKey(0))
        finally:
            restore()
        assert calls["mcmc_raytrace"] == 1
        assert calls["mcmc_nuts"] == 0


# ── _engine_cache_key — stability and sensitivity ─────────────────


class TestEngineCacheKey:
    """Cache key must be stable for identical config and differ when config changes."""

    def _make_fitter(self, spec: Parameters):
        from tengri.inference.fitter import Fitter

        model = _make_mock_model(spec, n_filters=3)
        noise = jnp.ones(3) * 1e-19
        data = jnp.ones(3) * 1e-18
        return Fitter(model, data, noise, data_type="photometry")

    def test_cache_key_stable_for_identical_spec(self, low_d_spec: Parameters) -> None:
        """Two Fitters with the same spec and data shape must produce equal cache keys."""
        f1 = self._make_fitter(low_d_spec)
        f2 = self._make_fitter(low_d_spec)
        assert f1._engine_cache_key() == f2._engine_cache_key()

    def test_cache_key_differs_when_free_param_added(self, low_d_spec: Parameters) -> None:
        """Adding a new free parameter must change the cache key."""
        spec_extended = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 5.0),
            sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
            met_logzsol=Uniform(-2.0, 0.5),
            dust_tau_diff=Uniform(0.0, 2.0),  # was Fixed in low_d_spec
            redshift=Fixed(0.1),
        )
        f_base = self._make_fitter(low_d_spec)
        f_extended = self._make_fitter(spec_extended)
        assert f_base._engine_cache_key() != f_extended._engine_cache_key(), (
            "Cache key did not change when a new free param was added"
        )

    def test_cache_key_is_hashable(self, low_d_spec: Parameters) -> None:
        """Cache key must be usable as a dict key (hashable)."""
        fitter = self._make_fitter(low_d_spec)
        key = fitter._engine_cache_key()
        d: dict = {}
        d[key] = "value"
        assert d[key] == "value"

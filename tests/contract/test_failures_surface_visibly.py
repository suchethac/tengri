# SPDX-License-Identifier: BSD-3-Clause
"""Failures that used to pass silently now reach the user.

Each test pins one site where a dropped result, a non-finite count, a missing
upstream input or a bad cache file was invisible: the user got fewer draws,
zero output or an unrelated AttributeError, with no warning that names the
cause. The assertions are on the public effect (a warning, a ValueError, an
unchanged global filter list), not on the internals that carry it.
"""

from __future__ import annotations

import pickle
import sys
import types
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri

pytestmark = pytest.mark.contract


# ── hierarchical geoVI: dropped posterior draws ───────────────────────────────


def _fake_jft_draw_fails_at(fail_index: int, exc: BaseException):
    """A NIFTy stand-in whose linear-residual draw raises on call ``fail_index``."""
    calls = {"n": 0}

    def draw_linear_residual(likelihood, pos, key, cg_kwargs):
        i = calls["n"]
        calls["n"] += 1
        if i == fail_index:
            raise exc
        return types.SimpleNamespace(tree={"x": jnp.asarray(0.5)}), None

    return types.SimpleNamespace(draw_linear_residual=draw_linear_residual)


def test_short_draw_collection_warns_with_requested_and_collected():
    from tengri.inference.hierarchical import _collect_linear_draws, _warn_short_draws

    jft = _fake_jft_draw_fails_at(2, TypeError("cg API mismatch"))
    pos = {"x": jnp.asarray(1.0)}
    draws, failure = _collect_linear_draws(jft, None, pos, jax.random.PRNGKey(0), 10)

    assert len(draws) == 2
    assert isinstance(failure, TypeError)

    with pytest.warns(UserWarning, match=r"requested 10.*collected 2.*TypeError"):
        _warn_short_draws(requested=10, collected=len(draws), failure=failure)


def test_full_draw_collection_is_silent():
    from tengri.inference.hierarchical import _warn_short_draws

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_short_draws(requested=5, collected=5, failure=None)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


# ── hierarchical geoVI: NIFTy call scoping ────────────────────────────────────


def test_quiet_optimize_kl_leaves_global_filters_and_stdout_intact(capsys):
    from tengri.inference.hierarchical import _quiet_optimize_kl

    def optimize_kl(*args, **kwargs):
        print("noisy NIFTy progress")
        warnings.warn("inner NIFTy warning", UserWarning, stacklevel=1)
        raise RuntimeError("linear solver failed")

    jft = types.SimpleNamespace(optimize_kl=optimize_kl)
    filters_before = list(warnings.filters)
    stdout_before = sys.stdout

    with pytest.raises(RuntimeError, match="linear solver"):
        _quiet_optimize_kl(jft)

    assert list(warnings.filters) == filters_before
    assert sys.stdout is stdout_before
    assert "noisy NIFTy progress" not in capsys.readouterr().out


def test_quiet_optimize_kl_suppresses_inner_warnings_and_returns():
    from tengri.inference.hierarchical import _quiet_optimize_kl

    def optimize_kl(*args, **kwargs):
        warnings.warn("inner NIFTy warning", UserWarning, stacklevel=1)
        return "samples", "state"

    jft = types.SimpleNamespace(optimize_kl=optimize_kl)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _quiet_optimize_kl(jft)
    assert out == ("samples", "state")
    assert not [w for w in caught if "inner NIFTy" in str(w.message)]


# ── VI backends: non-finite counts ────────────────────────────────────────────


def test_pathfinder_nonfinite_draws_warn(monkeypatch):
    import tengri.inference.backends.mcmc._shared as shared
    from tengri.inference.backends import pathfinder

    monkeypatch.setattr(shared, "_check_blackjax_floor", lambda: None)

    class _Pathfinder:
        @staticmethod
        def approximate(key, logp, init, **kwargs):
            return None, None

        @staticmethod
        def sample(key, state, n):
            samples = np.zeros((n, 2))
            samples[0, 0] = np.nan
            return jnp.asarray(samples), jnp.zeros(n)

    fake_blackjax = types.SimpleNamespace(pathfinder=_Pathfinder)
    monkeypatch.setitem(sys.modules, "blackjax", fake_blackjax)

    with pytest.warns(UserWarning, match=r"non-finite"):
        pathfinder.run_pathfinder(
            key=jax.random.PRNGKey(0),
            log_posterior_flat=lambda x: -0.5 * jnp.sum(x**2),
            init_flat=jnp.zeros(2),
            unravel_fn=lambda v: {"x": v},
            to_physical_fn=lambda d: d,
            model=None,
            n_samples=8,
            n_elbo_draws=2,
            verbose=False,
        )


def test_gaussian_vi_nonfinite_elbo_steps_warn(monkeypatch):
    import tengri.inference.backends.mcmc._shared as shared
    from tengri.inference.backends.vi import gaussian

    monkeypatch.setattr(shared, "_check_blackjax_floor", lambda: None)

    class _Algorithm:
        @staticmethod
        def init(init_flat, optimizer):
            return init_flat

        @staticmethod
        def step(key, state, logp, optimizer, num_samples):
            return state, types.SimpleNamespace(elbo=jnp.asarray(jnp.nan))

        @staticmethod
        def sample(key, state, n):
            return jnp.zeros((n, state.shape[0]))

    monkeypatch.setattr(gaussian, "_family", lambda name: _Algorithm)

    with pytest.warns(UserWarning, match=r"non-finite"):
        gaussian.run_gaussian_vi(
            key=jax.random.PRNGKey(0),
            log_posterior_flat=lambda x: -0.5 * jnp.sum(x**2),
            init_flat=jnp.zeros(2),
            unravel_fn=lambda v: {"x": v},
            to_physical_fn=lambda d: d,
            model=None,
            n_steps=4,
            n_samples=4,
            verbose=False,
        )


# ── cache loaders: stale or foreign pickles ───────────────────────────────────


def test_fitter_load_cache_rejects_non_dict_payload(tmp_path):
    from tengri.inference.fitter import Fitter

    path = tmp_path / "cache.pkl"
    path.write_bytes(pickle.dumps(["not", "a", "cache"]))
    with pytest.raises(ValueError, match=r"cache\.pkl.*dict"):
        Fitter.load_cache(types.SimpleNamespace(), path)


def test_fitter_load_cache_names_missing_keys(tmp_path):
    from tengri.inference.fitter import Fitter

    path = tmp_path / "cache.pkl"
    path.write_bytes(pickle.dumps({"spec_fingerprint": "abc"}))
    with pytest.raises(ValueError, match=r"cache\.pkl.*adaptation"):
        Fitter.load_cache(types.SimpleNamespace(), path)


def test_sbi_from_file_names_missing_posterior_key(tmp_path):
    from tengri.inference.backends.sbi import SBIPosterior

    path = tmp_path / "posterior.pkl"
    path.write_bytes(pickle.dumps({"param_names": ["a"]}))
    with pytest.raises(ValueError, match=r"posterior\.pkl.*posterior"):
        SBIPosterior.from_file(str(path))


# ── radio: missing L_ir and L_agn_bol ─────────────────────────────────────────


def test_radio_without_dust_or_agn_warns_at_build():
    from tengri.config.exceptions import AdvisoryWarning
    from tengri.forward.component_factory import build_components

    ssp = tengri.load_ssp()
    with pytest.warns(AdvisoryWarning, match=r"L_ir.*L_agn_bol"):
        build_components(ssp_data=ssp, use_radio=True, use_dust=False)


def test_radio_with_dust_publisher_is_silent_at_build():
    from tengri.config.exceptions import AdvisoryWarning
    from tengri.forward.component_factory import build_components

    ssp = tengri.load_ssp()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_components(ssp_data=ssp, use_radio=True, use_dust=True)
    assert not [
        w for w in caught if issubclass(w.category, AdvisoryWarning) and "L_ir" in str(w.message)
    ]


# ── filter discovery: suggest() skips broken curves loudly ────────────────────


def test_suggest_warns_once_per_skipped_filter(monkeypatch):
    import tengri.observation.filters as filters

    names = filters.list_filter_aliases().names()
    broken = names[0]
    real_load = filters.load_filter_set

    def load_filter_set(names_arg, *args, **kwargs):
        if list(names_arg) == [broken]:
            raise KeyError(broken)
        return real_load(names_arg, *args, **kwargs)

    monkeypatch.setattr(filters, "load_filter_set", load_filter_set)
    with pytest.warns(UserWarning, match=broken):
        result = filters.suggest(redshift=0.0, coverage="uv_to_ir")
    assert broken not in result


def test_suggest_does_not_swallow_unexpected_errors(monkeypatch):
    import tengri.observation.filters as filters

    def load_filter_set(names_arg, *args, **kwargs):
        raise RuntimeError("unrelated bug")

    monkeypatch.setattr(filters, "load_filter_set", load_filter_set)
    with pytest.raises(RuntimeError, match="unrelated bug"):
        filters.suggest(redshift=0.0, coverage="uv_to_ir")

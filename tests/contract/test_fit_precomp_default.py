# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for precomp-by-default fits + auto-prewarm (2026-07).

Covers the fit-time approximation policy added in the ``approx="auto"`` feature:
model build/prediction stay exact, fits route through the precompute LUT by
default, ``approx=None`` opts out, an explicit config overrides, a build-time
``approx`` is respected, and the user's model is never mutated. Also guards the
``with_approx`` clone (bit-identical to a directly-built precomp model) and the
``prewarm`` recursion guard.

Design: docs/internal/specs/2026-07-15-fit-precomp-default-design.md
"""

import chex
import jax
import pytest

from tengri import FIXED, Fixed, ForwardModel, SEDModel, Uniform, WavePrecomp

pytestmark = pytest.mark.contract


def _model(ssp, obs, approx=None):
    """A small 2-free-parameter photometry model at fixed redshift.

    Two dust parameters are free (``dust_tau_diff``, ``dust_tau_bc``) — enough
    for a non-degenerate MAP without depending on optional builder groups.
    """
    return SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "delayed", "*": FIXED},
        dust={
            "law": "power_law",
            "type": "two_component",
            "*": FIXED,
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": Uniform(0.0, 1.0),
            "emission": None,
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
        approx=approx,
    )


# ── with_approx clone ────────────────────────────────────────────────────


def test_with_approx_clone_matches_direct(synthetic_ssp_wide, synthetic_tophat_obs):
    """A cloned WavePrecomp model predicts identically to a directly-built one."""
    exact = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    direct = _model(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp())
    clone = exact.with_approx(WavePrecomp())
    p = exact.spec.sample(jax.random.PRNGKey(0))
    chex.assert_trees_all_close(
        clone.predict_photometry(p),
        direct.predict_photometry(p),
        rtol=1e-6,
        custom_message="with_approx clone != directly-built WavePrecomp model",
    )
    assert clone._has_modern_approx()
    assert not exact._has_modern_approx()


def test_with_approx_none_is_noop_on_exact(synthetic_ssp_wide, synthetic_tophat_obs):
    """with_approx(None) on an already-exact model returns self (no rebuild)."""
    exact = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    assert exact.with_approx(None) is exact


def test_with_approx_none_reverts_lut_to_exact(synthetic_ssp_wide, synthetic_tophat_obs):
    """with_approx(None) on a LUT model rebuilds the exact path, matching exact."""
    exact = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    lut = _model(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp())
    back = lut.with_approx(None)
    assert not back._has_modern_approx()
    p = exact.spec.sample(jax.random.PRNGKey(1))
    chex.assert_trees_all_close(back.predict_photometry(p), exact.predict_photometry(p), rtol=1e-6)


# ── Fitter approx resolution ─────────────────────────────────────────────


def _fitter(model, ssp, obs, **kw):
    from tengri.inference.fitter import Fitter

    truth = model.spec.sample(jax.random.PRNGKey(0))
    mock = model.mock(truth, snr=30.0, key=jax.random.PRNGKey(0))
    fwd = ForwardModel.build(sed=model, observation=obs)
    return Fitter(fwd, mock.flux_obs, mock.noise, **kw), mock


def test_auto_enables_precomp(synthetic_ssp_wide, synthetic_tophat_obs):
    """Default approx='auto' routes an exact-built photometry fit through the LUT."""
    exact = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    f, _ = _fitter(exact, synthetic_ssp_wide, synthetic_tophat_obs)  # auto default
    assert f.model._has_modern_approx()


def test_approx_none_forces_exact_and_is_identity(synthetic_ssp_wide, synthetic_tophat_obs):
    """approx=None keeps the exact path — the fit model IS the passed model."""
    exact = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    fwd = ForwardModel.build(sed=exact, observation=synthetic_tophat_obs)
    from tengri.inference.fitter import Fitter

    truth = exact.spec.sample(jax.random.PRNGKey(0))
    mock = exact.mock(truth, snr=30.0, key=jax.random.PRNGKey(0))
    f = Fitter(fwd, mock.flux_obs, mock.noise, approx=None)
    assert not f.model._has_modern_approx()
    assert f.model is fwd  # exact path leaves the model object untouched


def test_build_time_approx_respected(synthetic_ssp_wide, synthetic_tophat_obs):
    """A model built with a modern approx is not re-cloned by approx='auto'."""
    built = ForwardModel.build(
        sed=_model(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp()),
        observation=synthetic_tophat_obs,
    )
    from tengri.inference.fitter import Fitter

    truth = built.spec.sample(jax.random.PRNGKey(0))
    mock = built.populations[0].sed.mock(truth, snr=30.0, key=jax.random.PRNGKey(0))
    f = Fitter(built, mock.flux_obs, mock.noise)  # auto
    assert f.model is built


def test_explicit_override_reaches_model(synthetic_ssp_wide, synthetic_tophat_obs):
    """An explicit WavePrecomp(n_z=...) is threaded to the fit model."""
    exact = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    f, _ = _fitter(exact, synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp(n_z=37))
    inner = f.model._inner_sed_for_delegation()
    assert inner._approx_config_wave.n_z == 37


def test_original_model_untouched(synthetic_ssp_wide, synthetic_tophat_obs):
    """Auto-cloning for the fit does not mutate the user's exact model."""
    exact = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    _fitter(exact, synthetic_ssp_wide, synthetic_tophat_obs)  # auto
    assert not exact._has_modern_approx()


def test_bad_approx_string_rejected(synthetic_ssp_wide, synthetic_tophat_obs):
    """A non-'auto' approx string is a clear error, not a silent pass-through."""
    exact = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    with pytest.raises(ValueError, match="approx="):
        _fitter(exact, synthetic_ssp_wide, synthetic_tophat_obs, approx="wave_precomp")


# ── Agreement + prewarm (small MAP) ──────────────────────────────────────


def test_auto_matches_exact_map(synthetic_ssp_wide, synthetic_tophat_obs):
    """approx='auto' (LUT) and approx=None (exact) recover the same MAP point."""
    exact = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    truth = exact.spec.sample(jax.random.PRNGKey(0))
    mock = exact.mock(truth, snr=30.0, key=jax.random.PRNGKey(0))
    fwd = ForwardModel.build(sed=exact, observation=synthetic_tophat_obs)
    kf = jax.random.PRNGKey(1)
    p_auto = fwd.fit(mock.flux_obs, mock.noise, method="map", n_steps=150, key=kf)
    p_exact = fwd.fit(mock.flux_obs, mock.noise, method="map", n_steps=150, key=kf, approx=None)
    for name in exact.spec.free_params:
        a, e = float(p_auto.params[name]), float(p_exact.params[name])
        assert abs(a - e) < 5e-3, f"{name}: auto={a} exact={e}"


def test_prewarm_completes_without_recursion(synthetic_ssp_wide, synthetic_tophat_obs):
    """prewarm=True (default) fits cleanly — guards run->prewarm->run recursion."""
    exact = _model(synthetic_ssp_wide, synthetic_tophat_obs)
    truth = exact.spec.sample(jax.random.PRNGKey(0))
    mock = exact.mock(truth, snr=30.0, key=jax.random.PRNGKey(0))
    fwd = ForwardModel.build(sed=exact, observation=synthetic_tophat_obs)
    post = fwd.fit(
        mock.flux_obs,
        mock.noise,
        method="map",
        n_steps=50,
        key=jax.random.PRNGKey(2),
        prewarm=True,
    )
    assert post.params  # a result came back
    # The returned posterior references the LUT fit clone (fast predict).
    assert post._model._has_modern_approx()


# The legacy ``ensure_photometry_precomputed`` chain was deleted: it returned a
# constant ``False`` that every caller discarded, so a "no double-precompute"
# guard has nothing left to guard. ``approx=`` is now the only precompute path.

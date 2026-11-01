# SPDX-License-Identifier: BSD-3-Clause
"""Contract: CueBackend's weighted Cue parameters are computed in the log domain (#1206).

The old linear-domain combination overflows in pure float32 when logqion_all ~ 44-52,
which the isfinite guard then silently zeroes. The log-domain formulation must stay
finite and must agree with the linear-domain arithmetic in float64.

Weights are given on the SSP age grid of the backend (the only grid the weighted
path accepts), and the reference for the ionizing shape is the linear composite
spectrum fitted with :func:`fit_ionizing_spectrum`.

1. f64 exactness: Q_H and the unclipped shape equal the linear composite reference.
2. Pure float32 is finite and matches float64 (backend rebuilt inside x64 off).
3. A degenerate population (all-old or all-zero weights) gives Q_H = -99 and a
   finite shape.
4. The gradient with respect to the weights is non-zero and finite on a young
   population, and finite on an all-old one.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import load_ssp_data
from tengri.components.nebular.cue import _MAX_NEB_LOG_AGE, CueBackend
from tengri.components.nebular.ionizing_spectrum import fit_ionizing_spectrum

pytestmark = pytest.mark.regression_bug

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_CUE_WEIGHTS = "data/cue_weights.npz"
_IONSPEC_KEYS = (
    "ionspec_index1",
    "ionspec_index2",
    "ionspec_index3",
    "ionspec_index4",
    "ionspec_logLratio1",
    "ionspec_logLratio2",
    "ionspec_logLratio3",
)


def _load_ssp():
    if not Path(_BARE).is_file():
        pytest.skip(f"missing bare SSP {_BARE}")
    return load_ssp_data(_BARE)


def _grid(be):
    """SSP age grid of the backend, log10(age/yr), and its young mask."""
    ages = np.asarray(be._ssp_log_age_yr)
    return ages, ages <= _MAX_NEB_LOG_AGE


def _weights(ages, kind):
    """Mass weights on the SSP age grid (unnormalized)."""
    n = ages.shape[0]
    young = ages <= _MAX_NEB_LOG_AGE
    if kind == "young_burst":
        t_myr = 10.0 ** (ages - 6.0)
        w = np.exp(-0.5 * ((t_myr - 5.0) / 2.0) ** 2)
    elif kind == "older_population":
        w = np.exp(-np.abs(np.arange(n) - n / 3) / 5.0)
    elif kind == "old_only":
        w = np.where(young, 0.0, 1.0)
    else:  # pragma: no cover - guard against a typo
        raise ValueError(kind)
    return w if kind == "old_only" else np.where(young, w, 0.0)


def _linear_composite_reference(ssp, im, weights):
    """Linear f64 composite at metallicity node ``im``, fitted with the reference recipe.

    Returns the unclipped 7-vector and the reference log10 Q_H. Both come from the
    linear composite, so they are the ground truth for the log-domain path.
    """
    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    flux = np.asarray(ssp.ssp_flux, dtype=np.float64)[im]  # (n_age, n_wave)
    comp = np.sum(np.asarray(weights, dtype=np.float64)[:, None] * flux, axis=0)
    ref = fit_ionizing_spectrum(wave, comp)
    pl = np.asarray(ref["powerlaw_params"])
    i7 = np.concatenate([pl[:, 0], np.diff(np.asarray(ref["log_seglum"]))])
    return i7, float(ref["gas_logqion"])


def _raw_seven(be, weights, ages, log_z):
    raw = be._weighted_cue_params_raw(jnp.asarray(weights), jnp.asarray(ages), log_z)
    return raw, jnp.stack([raw[k] for k in _IONSPEC_KEYS])


def test_f64_exactness_vs_linear_composite_reference():
    """Q_H at rtol 1e-10 and the unclipped shape at 1e-8, against the linear composite."""
    ssp = _load_ssp()
    be = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)
    ages, young = _grid(be)
    lgmet = np.asarray(ssp.ssp_lgmet)

    for im in (1, 7, 12):  # grid metallicity nodes, no interpolation
        log_z = float(lgmet[im])
        for kind in ("young_burst", "older_population"):
            w = _weights(ages, kind)
            w = w / np.sum(w)
            w_eff = np.where(young & (w > 0), w, 0.0)  # the window the weighted path uses
            ref7, ref_q = _linear_composite_reference(ssp, im, w_eff)
            raw, got7 = _raw_seven(be, w, ages, log_z)
            ctx = f"met node {im} (log_z={log_z:.3f}), sfh={kind}"
            np.testing.assert_allclose(
                float(raw["gas_logqion"]), ref_q, rtol=1e-10, err_msg=f"Q_H at {ctx}"
            )
            np.testing.assert_allclose(
                np.asarray(got7), ref7, rtol=0, atol=1e-8, err_msg=f"shape at {ctx}"
            )


def test_pure_float32_correctness():
    """Pure f32 must be finite and match f64 (anti-silent-zeroing gate).

    The backend is rebuilt inside ``jax.enable_x64(False)``, or the f64 tables would
    mask any overflow, and the dtype is asserted.
    """
    ssp = _load_ssp()
    be64 = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)
    ages, _ = _grid(be64)
    n_age = ages.shape[0]
    ssp_weights_f64 = np.ones(n_age) / n_age
    log_z = -2.0

    raw64, i7_f64 = _raw_seven(be64, ssp_weights_f64, ages, log_z)
    gas_logqion_f64 = raw64["gas_logqion"]

    with jax.enable_x64(False):
        be32 = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)
        assert be32._logqion_table.dtype == jnp.float32, (
            f"CueBackend was NOT rebuilt in f32 context: dtype={be32._logqion_table.dtype}"
        )
        raw32 = be32._weighted_cue_params_raw(
            jnp.asarray(ssp_weights_f64, dtype=jnp.float32),
            jnp.asarray(ages, dtype=jnp.float32),
            jnp.float32(log_z),
        )
        gas_logqion_f32 = raw32["gas_logqion"]
        i7_f32 = jnp.stack([raw32[k] for k in _IONSPEC_KEYS])

    assert jnp.all(jnp.isfinite(gas_logqion_f32)), f"f32 Q_H non-finite: {gas_logqion_f32}"
    assert jnp.all(jnp.isfinite(i7_f32)), f"f32 shape non-finite: {i7_f32}"
    assert jnp.allclose(gas_logqion_f32, gas_logqion_f64, atol=5e-3), (
        f"f32/f64 gas_logqion mismatch: f32={gas_logqion_f32}, f64={gas_logqion_f64}"
    )
    assert jnp.allclose(i7_f32, i7_f64, atol=0.02), (
        f"f32/f64 ionspec mismatch: f32={i7_f32}, f64={i7_f64}"
    )


def test_degenerate_population():
    """All-old or all-zero weights: gas_logqion = -99.0 and a finite shape.

    All-old weight is only degenerate under the 100 Myr cut; by default every age is
    counted (see ``test_all_old_population_counts_its_photons_by_default``).
    """
    ssp = _load_ssp()
    be = CueBackend(_CUE_WEIGHTS, ssp_data=ssp, lyc_max_age_yr=1.0e8)
    ages, young = _grid(be)

    weights_old = np.where(young, 0.0, 1.0)
    result_old = be._compute_weighted_cue_params(
        jnp.asarray(weights_old), jnp.asarray(ages), log_z=-2.0
    )
    assert result_old["gas_logqion"] == -99.0, (
        f"all-old population should have gas_logqion=-99.0, got {result_old['gas_logqion']}"
    )
    i7_old = jnp.stack([result_old[k] for k in _IONSPEC_KEYS])
    assert jnp.all(jnp.isfinite(i7_old)), f"all-old ionspec should be finite, got {i7_old}"

    result_zero = be._compute_weighted_cue_params(
        jnp.zeros(ages.shape[0]), jnp.asarray(ages), log_z=-2.0
    )
    assert result_zero["gas_logqion"] == -99.0, (
        f"zero weights should have gas_logqion=-99.0, got {result_zero['gas_logqion']}"
    )
    i7_zero = jnp.stack([result_zero[k] for k in _IONSPEC_KEYS])
    assert jnp.all(jnp.isfinite(i7_zero)), f"zero-weight ionspec should be finite: {i7_zero}"


def test_gradient_nonzero_and_smooth():
    """d gas_logqion / d weights: non-zero and finite on a young population."""
    ssp = _load_ssp()
    be = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)
    ages, _ = _grid(be)
    ssp_weights = jnp.ones(ages.shape[0]) / ages.shape[0]
    log_z = -2.0

    def gas_logqion_fn(w):
        return be._compute_weighted_cue_params(w, jnp.asarray(ages), log_z)["gas_logqion"]

    g_young = jax.grad(gas_logqion_fn)(ssp_weights)
    assert jnp.all(jnp.isfinite(g_young)), f"gradient has NaN/inf (young): {g_young}"
    assert jnp.any(jnp.abs(g_young) > 1e-10), (
        f"gradient w.r.t. ssp_weights is zero for a young population: {g_young}. "
        "This indicates SFH has no effect on Q_H, which is wrong."
    )


def test_gradient_finite_on_all_old_population():
    """No young bins: the gradient is finite (the emission is zero on purpose)."""
    ssp = _load_ssp()
    be = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)
    ages, young = _grid(be)
    log_z = -2.0
    w_old = jnp.asarray(np.where(young, 0.0, 1.0) / np.sum(~young))

    def gas_logqion_fn(w):
        return be._compute_weighted_cue_params(w, jnp.asarray(ages), log_z)["gas_logqion"]

    g_old = jax.grad(gas_logqion_fn)(w_old)
    # grad-assert: finite-only - this arm is the degenerate one on purpose
    assert jnp.all(jnp.isfinite(g_old)), f"gradient has NaN/inf for all-old population: {g_old}"


def test_all_old_population_counts_its_photons_by_default():
    """With every age counted an all-old population has a finite Q_H, not the -99 of the cut."""
    ssp = _load_ssp()
    be = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)
    ages, _ = _grid(be)
    weights = _weights(ages, "old_only")
    weights = weights / np.sum(weights)
    result = be._compute_weighted_cue_params(jnp.asarray(weights), jnp.asarray(ages), log_z=-2.0)
    assert float(result["gas_logqion"]) > -99.0
    assert all(np.isfinite(float(v)) for v in result.values())


def test_cut_composite_uses_exactly_the_counted_bins():
    """With lyc_max_age_yr=1e8 the shape is the fit of the young bins alone."""
    ssp = _load_ssp()
    be_all = CueBackend(_CUE_WEIGHTS, ssp_data=ssp)
    be_cut = CueBackend(_CUE_WEIGHTS, ssp_data=ssp, lyc_max_age_yr=1.0e8)
    ages, young = _grid(be_all)
    w = _weights(ages, "young_burst") + 0.5 * _weights(ages, "old_only")
    log_z = float(np.asarray(ssp.ssp_lgmet)[7])
    raw_cut, got_cut = _raw_seven(be_cut, w, ages, log_z)
    raw_ref, got_ref = _raw_seven(be_all, np.where(young, w, 0.0), ages, log_z)
    np.testing.assert_allclose(
        float(raw_cut["gas_logqion"]), float(raw_ref["gas_logqion"]), rtol=1e-10
    )
    np.testing.assert_allclose(np.asarray(got_cut), np.asarray(got_ref), rtol=0, atol=1e-8)

# SPDX-License-Identifier: BSD-3-Clause
r"""The SpectrumPrecomp LUT applies the exact path's sigma_v and instrument stages (#2958, #1688).

Before the fix the per-pixel families were point samples of the un-convolved rest
SED: ``predict_spectrum`` under ``approx=SpectrumPrecomp()`` ignored the instrument
resolution and the galaxy velocity dispersion, so ``d spectrum / d sigma_v`` was
exactly zero and unresolved lines stayed unbroadened below R = 3000. The exact path
(``project_spectrum_kernel_split``) applies both; the LUT now runs the same stages on
its pixel grid. The tests pin, on the synthetic wide SSP (continuum) and on the real
wNE SSP with baked-in nebular lines (lines), at R = 2000 and sigma_v free:

(a) the gradient of the summed spectrum w.r.t. ``sigma_v_kms`` is finite, non-zero,
    of the exact path's sign, and within 10 % of it on the line-rich fixture;
(b) the LUT spectrum at sigma_v = 300 km/s differs from sigma_v = 0;
(c) the LUT matches the exact path at sigma_v = 300 km/s to 2e-3 at every pixel, and
    within 5 Angstrom of H-alpha and H-beta;
(d) with no instrument (``resolution=None``) the LUT is bit-for-bit the pre-fix LUT;
(e) the exact path is bit-for-bit the pre-extraction exact path;
(f) a banded resolution matrix falls back to the exact path with a warning and is
    bit-for-bit equal to it.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, SpectrumPrecomp, Uniform
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.regression_bug

_WAVE_OBS = np.linspace(4000.0, 9000.0, 2000)
_Z = 0.1
_HALPHA_OBS = 6562.8 * (1.0 + _Z)
_HBETA_OBS = 4861.3 * (1.0 + _Z)
_IDX = [0, 333, 666, 999, 1332, 1666, 1999]
_MADAU = {"type": "madau"}


def _spec(resolution=2000.0):
    return Spectroscopy(resample="point", wave_obs=_WAVE_OBS, resolution=resolution)


def _build(ssp, approx, *, spectro, igm=None, sigma_v=None):
    obs = Observation(spectroscopy=spectro)
    base = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=approx,
        redshift=Fixed(_Z),
        igm=igm if igm is not None else {"type": "none"},
        sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
    )
    spec = base.spec if sigma_v is None else base.spec.merge_observation_params(sigma_v_kms=sigma_v)
    return SEDModel(spec, ssp, observation=obs, approx=approx)


def _spectrum(model, sigma_v):
    return np.asarray(model.predict_spectrum({"sigma_v_kms": sigma_v}))


def _grad(model, sigma_v):
    return jax.grad(lambda s: jnp.sum(model.predict_spectrum({"sigma_v_kms": s})))(
        jnp.asarray(sigma_v, dtype=jnp.float64)
    )


def _max_rel_dev(s_lut, s_exact, mask):
    return float(np.max(np.abs(s_lut[mask] / s_exact[mask] - 1.0)))


def _mask(s_exact):
    return np.abs(s_exact) > 1e-4 * np.max(np.abs(s_exact))


# ── (a) gradient w.r.t. sigma_v matches the exact path ─────────────────────────


def test_a_gradient_live_on_continuum_fixture(synthetic_ssp_wide):
    sv = Uniform(0.0, 2000.0)
    exact = _build(synthetic_ssp_wide, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    lut = _build(synthetic_ssp_wide, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=sv)
    g_lut = float(_grad(lut, 150.0))
    g_exact = float(_grad(exact, 150.0))
    assert np.isfinite(g_lut)
    assert g_lut != 0.0
    assert np.sign(g_lut) == np.sign(g_exact)


def test_a_gradient_matches_exact_on_line_fixture(ssp_data_wne):
    sv = Uniform(0.0, 2000.0)
    exact = _build(ssp_data_wne, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    lut = _build(ssp_data_wne, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=sv)
    g_lut = float(_grad(lut, 150.0))
    g_exact = float(_grad(exact, 150.0))
    assert np.isfinite(g_lut)
    assert g_lut != 0.0
    rel = abs(g_lut - g_exact) / abs(g_exact)
    assert rel <= 0.10, f"LUT grad {g_lut} vs exact {g_exact} (rel {rel:.3g})"


# ── (b) sigma_v changes the spectrum ───────────────────────────────────────────


def test_b_spectrum_depends_on_sigma_v(ssp_data_wne):
    lut = _build(ssp_data_wne, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=Uniform(0, 2000))
    s0 = _spectrum(lut, 0.0)
    s300 = _spectrum(lut, 300.0)
    assert float(np.max(np.abs(s300 - s0))) > 1e-6 * float(np.max(np.abs(s0)))


# ── (c) LUT equals exact at identical parameters, including the lines ─────────


def test_c_lut_matches_exact_continuum(synthetic_ssp_wide):
    sv = Uniform(0.0, 2000.0)
    exact = _build(synthetic_ssp_wide, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    lut = _build(synthetic_ssp_wide, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=sv)
    s_exact, s_lut = _spectrum(exact, 300.0), _spectrum(lut, 300.0)
    assert _max_rel_dev(s_lut, s_exact, _mask(s_exact)) <= 2e-3


def test_c_lut_matches_exact_lines_at_halpha_hbeta(ssp_data_wne):
    sv = Uniform(0.0, 2000.0)
    exact = _build(ssp_data_wne, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    lut = _build(ssp_data_wne, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=sv)
    s_exact, s_lut = _spectrum(exact, 300.0), _spectrum(lut, 300.0)
    mask = _mask(s_exact)
    assert _max_rel_dev(s_lut, s_exact, mask) <= 2e-3
    for centre in (_HALPHA_OBS, _HBETA_OBS):
        near = mask & (np.abs(_WAVE_OBS - centre) <= 5.0)
        assert near.any()
        assert _max_rel_dev(s_lut, s_exact, near) <= 2e-3


# ── (d) no instrument: the LUT is bit-for-bit the pre-fix LUT ─────────────────

# Pre-fix LUT values (resolution=None, sigma_v=0, madau IGM), at _IDX, captured from the
# conftest fixtures on origin/main before the kernel was applied. With no resolution there is no kernel to
# apply, so the fixed LUT must reproduce them exactly.
_PREFIX_LUT_WNE_RNONE = [
        7.585321879860206e-29,
        2.7566442699605755e-28,
        4.476516855282938e-28,
        6.528615494672773e-28,
        7.666995745049181e-28,
        1.0511964695858449e-27,
        1.2729797128126482e-27,
    ]


def test_d_resolution_none_is_bit_for_bit_prefix(ssp_data_wne):
    lut = _build(ssp_data_wne, SpectrumPrecomp(), spectro=_spec(resolution=None), igm=_MADAU,
                 sigma_v=Fixed(0.0))
    s = np.asarray(lut.predict_spectrum({}))
    np.testing.assert_array_equal(s[_IDX], np.asarray(_PREFIX_LUT_WNE_RNONE))


# ── (e) exact path bit-for-bit across the extraction ──────────────────────────

_PINNED_EXACT = {
    "synth_B": [
        2.616654422948485e-12,
        1.9176771331791432e-12,
        1.4729137600973475e-12,
        1.1699882140946834e-12,
        9.526091195785729e-13,
        7.922208055230021e-13,
        6.692538866622282e-13,
    ],
    "wne_B": [
        7.59369890747518e-29,
        2.70378918815045e-28,
        4.39145217903001e-28,
        6.586601334966522e-28,
        7.890869728131413e-28,
        1.0505243755535244e-27,
        1.2730962692858322e-27,
    ],
}


def test_e_exact_path_bit_for_bit_across_extraction(synthetic_ssp_wide, ssp_data_wne):
    sv = Uniform(0.0, 2000.0)
    exact = _build(synthetic_ssp_wide, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    np.testing.assert_array_equal(
        _spectrum(exact, 150.0)[_IDX], np.asarray(_PINNED_EXACT["synth_B"])
    )
    exact_w = _build(ssp_data_wne, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    np.testing.assert_array_equal(
        _spectrum(exact_w, 150.0)[_IDX], np.asarray(_PINNED_EXACT["wne_B"])
    )


# ── (f) a banded resolution matrix falls back to the exact path ────────────────


def test_f_resolution_matrix_falls_back_bit_exact(synthetic_ssp_wide):
    rm = gaussian_resolution_bands(jnp.asarray(_WAVE_OBS), 2000.0, n_diag=11)
    spectro = Spectroscopy(resample="point", wave_obs=_WAVE_OBS, resolution_matrix=rm)
    sv = Uniform(0.0, 2000.0)
    exact = _build(synthetic_ssp_wide, None, spectro=spectro, igm=_MADAU, sigma_v=sv)
    with pytest.warns(UserWarning, match="resolution matrix"):
        lut = _build(synthetic_ssp_wide, SpectrumPrecomp(), spectro=spectro, igm=_MADAU,
                     sigma_v=sv)
    np.testing.assert_array_equal(_spectrum(lut, 150.0), _spectrum(exact, 150.0))

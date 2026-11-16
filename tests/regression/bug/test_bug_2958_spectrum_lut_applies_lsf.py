# SPDX-License-Identifier: BSD-3-Clause
r"""The SpectrumPrecomp LUT applies the exact path's sigma_v and instrument stages (#2958, #1688).

Before the fix the per-pixel families were point samples of the un-convolved rest
SED: ``predict_spectrum`` under ``approx=SpectrumPrecomp()`` ignored the instrument
resolution and the galaxy velocity dispersion, so ``d spectrum / d sigma_v`` was
exactly zero and unresolved lines stayed unbroadened below R = 3000. The exact path
(``project_spectrum_kernel_split``) applies both; the LUT now runs the same stages in
the same order: sigma_v on the model grid inside the stellar family (a pre-broadened
template table for fixed sigma_v, a full-grid stellar sum otherwise), then IGM, then the
instrument kernel on the pixels. The tests pin, on the synthetic wide SSP (continuum) and on
the real wNE SSP with baked-in nebular lines (lines), at R = 2000 and sigma_v free:

(a) the gradient of a chi-square against the sigma_v = 100 spectrum w.r.t.
    ``sigma_v_kms`` is within 1e-3 (relative) of the exact path's at 50 and 300 km/s;
(b) the LUT spectrum at sigma_v = 300 km/s differs from sigma_v = 0;
(c) the LUT matches the exact path at sigma_v = 300 km/s to 2e-3 at every pixel,
    including the Ca II K core, and within 5 Angstrom of H-alpha and H-beta;
(d) with no instrument (``resolution=None``) the LUT is bit-for-bit the pre-fix LUT;
(e) the exact path is bit-for-bit the pre-extraction exact path;
(f) a banded resolution matrix falls back to the exact path with a warning and is
    bit-for-bit equal to it;
(g) the fixed-sigma_v pre-broadened table equals the full-grid sum route;
(h) the LUT matches the exact path at R = 5000 with no resolution cap;
(i) free redshift with a fixed sigma_v uses a cube broadened once at build and equals the
    per-draw sum route at two redshifts;
(j) a joint observation with approx=WavePrecomp() promotes the spectrum channel onto the
    LUT for a fixed sigma_v and keeps it exact, saying so once, for a free one (approx="auto"
    keeps spectroscopy exact in any case); an explicit SpectrumPrecomp() with a free sigma_v
    engages the LUT and warns once about the cost, on a spectroscopy-only and on a joint
    observation alike.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    SpectrumPrecomp,
    Uniform,
    WavePrecomp,
)
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.regression_bug

_WAVE_OBS = np.linspace(4000.0, 9000.0, 2000)
_Z = 0.1
_HALPHA_OBS = 6562.8 * (1.0 + _Z)
_HBETA_OBS = 4861.3 * (1.0 + _Z)
# The pins were captured on one platform and are checked on others (linux CI, macOS, other
# jaxlib builds), where the FFT and the reductions round differently in the last bits:
# measured up to 1.7e-15 relative on the CI runner (test e 1.67e-15, test d 3.1e-16) against
# the pins captured on macOS. 1e-12 sits about 600 times above that and well below a real
# change: a 1 % change in sigma_v moves the pinned spectra by 6.7e-9 (synth_B) to 1.6e-4
# (wne_B), so the pins still catch it. synth_B cannot see the instrument-R stage at all.
_PIN_RTOL = 1e-12
_IDX = [0, 333, 666, 999, 1332, 1666, 1999]
_MADAU = {"type": "madau"}


def _spec(resolution=2000.0, resample="point"):
    return Spectroscopy(resample=resample, wave_obs=_WAVE_OBS, resolution=resolution)


def _build(ssp, approx, *, spectro, igm=None, sigma_v=None, photometry=None, law="calzetti"):
    obs = Observation(photometry=photometry, spectroscopy=spectro)
    dust = {"type": "two_component", "law": law, "all_params": Fixed(DEFAULT)}
    if law == "power_law":
        dust["tau_diff"] = 0.3
    # The base model only supplies the spec; the approximation is resolved on the model
    # that carries the declared dispersion.
    base = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(_Z),
        igm=igm if igm is not None else {"type": "none"},
        sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
        dust_attenuation=dust,
        neb={"type": "none"},
    )
    spec = (
        base.spec if sigma_v is None else base.spec.merge_observation_params(sigma_v_kms=sigma_v)
    )
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


def _chi2_grad(model, ref, sigma_v):
    """d chi2 / d sigma_v of the spectrum against ``ref`` with 5 % errors."""

    def chi2(s):
        spec = model.predict_spectrum({"sigma_v_kms": s})
        return jnp.sum(((spec - ref) / (0.05 * ref)) ** 2)

    return float(jax.grad(chi2)(jnp.asarray(sigma_v, dtype=jnp.float64)))


@pytest.mark.parametrize("sigma_v", [50.0, 300.0])
def test_a_chi2_gradient_matches_exact(ssp_data_wne, sigma_v):
    sv = _FREE_SERVED
    exact = _build(ssp_data_wne, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    lut = _build(ssp_data_wne, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=sv)
    assert lut._approx.get("spectrum_precomp")
    ref = jnp.asarray(_spectrum(exact, 100.0))
    g_exact = _chi2_grad(exact, ref, sigma_v)
    g_lut = _chi2_grad(lut, ref, sigma_v)
    assert np.isfinite(g_lut)
    assert g_lut != 0.0
    assert abs(g_lut - g_exact) <= 1e-3 * abs(g_exact), f"LUT {g_lut} vs exact {g_exact}"


def test_a_gradient_live_on_continuum_fixture(synthetic_ssp_wide):
    sv = _FREE_SERVED
    exact = _build(synthetic_ssp_wide, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    lut = _build(synthetic_ssp_wide, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=sv)
    assert lut._approx.get("spectrum_precomp")
    ref = jnp.asarray(_spectrum(exact, 100.0))
    g_lut = _chi2_grad(lut, ref, 300.0)
    g_exact = _chi2_grad(exact, ref, 300.0)
    assert np.isfinite(g_lut)
    assert g_lut != 0.0
    assert np.sign(g_lut) == np.sign(g_exact)


# ── (b) sigma_v changes the spectrum ───────────────────────────────────────────


def test_b_spectrum_depends_on_sigma_v(ssp_data_wne):
    lut = _build(
        ssp_data_wne, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=_FREE_SERVED
    )
    assert lut._approx.get("spectrum_precomp")
    s0 = _spectrum(lut, 0.0)
    s300 = _spectrum(lut, 300.0)
    assert float(np.max(np.abs(s300 - s0))) > 1e-6 * float(np.max(np.abs(s0)))


# ── (c) LUT equals exact at identical parameters, including the lines ─────────


def test_c_lut_matches_exact_continuum(synthetic_ssp_wide):
    sv = _FREE_SERVED
    exact = _build(synthetic_ssp_wide, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    lut = _build(synthetic_ssp_wide, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=sv)
    assert lut._approx.get("spectrum_precomp")
    s_exact, s_lut = _spectrum(exact, 300.0), _spectrum(lut, 300.0)
    assert _max_rel_dev(s_lut, s_exact, _mask(s_exact)) <= 2e-3


def test_c_lut_matches_exact_lines_at_halpha_hbeta(ssp_data_wne):
    sv = _FREE_SERVED
    exact = _build(ssp_data_wne, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    lut = _build(ssp_data_wne, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=sv)
    assert lut._approx.get("spectrum_precomp")
    s_exact, s_lut = _spectrum(exact, 300.0), _spectrum(lut, 300.0)
    mask = _mask(s_exact)
    assert _max_rel_dev(s_lut, s_exact, mask) <= 2e-3
    for center in (_HALPHA_OBS, _HBETA_OBS):
        near = mask & (np.abs(_WAVE_OBS - center) <= 5.0)
        assert near.any()
        assert _max_rel_dev(s_lut, s_exact, near) <= 2e-3


def test_c_fixed_sigma_v_table_matches_exact(ssp_data_wne):
    exact = _build(ssp_data_wne, None, spectro=_spec(), igm=_MADAU, sigma_v=Fixed(300.0))
    lut = _build(
        ssp_data_wne, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=Fixed(300.0)
    )
    s_exact = np.asarray(exact.predict_spectrum({}))
    s_lut = np.asarray(lut.predict_spectrum({}))
    assert _max_rel_dev(s_lut, s_exact, _mask(s_exact)) <= 2e-3


# ── (g) fixed-sigma_v pre-broadened table equals the full-grid sum route ──────


#: A free dispersion the spectrum table serves on point-sampled pixels (#3500).
_FREE_SERVED = Uniform(20.0, 2000.0)


def test_g_prebroadened_table_equals_sum_route(ssp_data_wne):
    table = _build(
        ssp_data_wne, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=Fixed(300.0)
    )
    # The per-draw route's dispersion is floored at 20 km/s: the table declines point-sampled
    # pixels whose kernel can reach zero (#3500), and the comparison would be against exact.
    summed = _build(
        ssp_data_wne, SpectrumPrecomp(), spectro=_spec(), igm=_MADAU, sigma_v=_FREE_SERVED
    )
    assert table._approx.get("spectrum_precomp") and summed._approx.get("spectrum_precomp")
    s_table = np.asarray(table.predict_spectrum({}))
    s_sum = _spectrum(summed, 300.0)
    # Same operation in two orders of linear ops: relative to the median flux so FFT
    # roundoff in the deepest absorption pixels does not dominate.
    # Route bar, 3x the measured 5.39e-8 with the native-slope Hermite read (#3541, 2026-11-15;
    # it was 1.21e-5 when the table read PCHIP of its own broadened columns).
    assert float(np.max(np.abs(s_table - s_sum))) <= 1.6e-7 * float(np.median(np.abs(s_sum)))


# ── (h) no resolution cap: the LUT holds at R = 5000 ──────────────────────────


def test_h_lut_matches_exact_at_r5000_without_fallback(ssp_data_wne):
    sv = _FREE_SERVED
    exact = _build(ssp_data_wne, None, spectro=_spec(5000.0), igm=_MADAU, sigma_v=sv)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*falls back.*")
        lut = _build(
            ssp_data_wne, SpectrumPrecomp(), spectro=_spec(5000.0), igm=_MADAU, sigma_v=sv
        )
    assert lut._approx.get("spectrum_precomp")
    s_exact, s_lut = _spectrum(exact, 300.0), _spectrum(lut, 300.0)
    mask = _mask(s_exact)
    assert _max_rel_dev(s_lut, s_exact, mask) <= 2e-3


# ── (d) no instrument: the LUT is bit-for-bit the pre-fix LUT ─────────────────

# Pre-fix LUT values (resolution=None, sigma_v=0, madau IGM), at _IDX, captured from the
# conftest fixtures on origin/main before the kernel was applied. With no resolution there
# is no kernel to apply, so the fixed LUT must reproduce them exactly.
# Re-captured on main ac75d42fe: the SFH timing parameters became fractions of
# age(z) (#2748), which moves the fixture's default tsnorm SFH at z = 0.1, so the
# arrays pinned before it (and the ones below in _PINNED_EXACT) no longer apply.
# Re-captured again on main debe80a55 (2026-11-15) after #3434 (in #3462): fixed
# metallicity carries no metallicity scatter, which moves these spectra by up to
# 0.36% (test d), 0.37% (wne_B) and 3.6% (synth_B, the synthetic grid's coarse Z nodes).
# The pins still guard the LUT-with-no-instrument and the exact path against
# unintended change; a ruled physics change re-captures them.
# Re-pinned 2026-11-15: monotone PCHIP metallicity read (#3434, owner ruling 'True PCHIP
# on the spectra'). Main's values are 8.800566e-29 ... 1.410505e-27; the PCHIP read moves the
# low-flux pixels by up to 2.4e-4 relative. Control (Hermite read) reproduces main to 1e-15.
# Re-pinned 2026-11-15 (#3541, owner ruling "Exact: native slopes"): the spectrum table now reads
# the native PCHIP slopes by Hermite instead of PCHIP of its own resampled values. The move is up
# to 2.8e-6 relative (index 0: 2.0e-7). Measured against the exact path on this grid, the LUT
# moved from 7.2e-6 to 4.4e-6 at these pixels; the rest (about 6e-6 at every pixel, with or
# without Madau IGM) is a separate LUT residual, not the metallicity read, and is not closed here.
_PREFIX_LUT_WNE_RNONE = [
    8.801269134494637e-29,
    3.1571986038432897e-28,
    5.065771732046372e-28,
    7.30496310383873e-28,
    8.53745948244721e-28,
    1.1664301615680506e-27,
    1.4114095304577797e-27,
]


def test_d_no_kernel_point_pixels_decline_to_the_exact_path(ssp_data_wne):
    """Point-sampled pixels with no instrument and no dispersion: the table declines (#3500).

    The table holds lattice cell means, which do not give the pixel-center value of an
    unbroadened model. The model says so and is the exact path, bit for bit. The values
    pinned above were the pixel table's, 4.4e-6 from the exact path at these pixels.
    """
    kwargs = dict(spectro=_spec(resolution=None), igm=_MADAU, sigma_v=Fixed(0.0))
    exact = _build(ssp_data_wne, None, **kwargs)
    with pytest.warns(UserWarning, match="spectrum table cannot serve"):
        lut = _build(ssp_data_wne, SpectrumPrecomp(), **kwargs)
    assert not lut._approx.get("spectrum_precomp")
    s_lut, s_exact = np.asarray(lut.predict_spectrum({})), np.asarray(exact.predict_spectrum({}))
    np.testing.assert_array_equal(s_lut, s_exact)
    rel = np.max(np.abs(s_lut[_IDX] / np.asarray(_PREFIX_LUT_WNE_RNONE) - 1.0))
    assert 1e-6 < rel < 1e-5, "the old pixel-table pins sit 4.4e-6 from the exact path"


def _brute_ln_lambda_broaden(wave_rest, sed_rest, sigma_v_kms, *, step=2e-6):
    """Gaussian in ln(lambda) on lambda F_lambda, on a fine uniform ln-lambda grid (#3436).

    The same brute-force reference as test_bug_3436: a direct kernel on a grid far finer
    than the lattice, applied by FFT, with no lattice. Returns ln(lambda) and lambda F.
    """
    sig = sigma_v_kms / 299792.458
    x = np.arange(np.log(wave_rest[0]), np.log(wave_rest[-1]), step)
    g = np.interp(np.exp(x), wave_rest, sed_rest) * np.exp(x)
    half = int(np.ceil(12.0 * sig / step))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * step / sig) ** 2)
    k /= k.sum()
    n = g.size + k.size
    conv = np.fft.irfft(np.fft.rfft(g, n) * np.fft.rfft(k, n), n)
    return x, conv[half : half + g.size]


def _igm_case_reference(exact_model, sigma_v_kms):
    """Brute-force LUT-free reference for the exact model with IGM, point-sampled.

    The stellar light of this case is all of ``state.sed_intrinsic`` (no nebular or
    instrument-only group), so the reference is T * B(S) on the rest grid, read at the
    observed pixel centers, scaled per pixel to the model's sigma_v = 0 output.
    """
    state = exact_model.predict_state({"sigma_v_kms": 0.0})
    wave_rest = np.asarray(state.wave)
    sed = np.asarray(state.sed_intrinsic)
    igm = np.asarray(state.derived["igm_transmission"])
    x, g_b = _brute_ln_lambda_broaden(wave_rest, sed, sigma_v_kms)
    _, g_0 = _brute_ln_lambda_broaden(wave_rest, sed, 1e-9)
    t_f = np.interp(np.exp(x), wave_rest, igm)
    wq = np.asarray(_WAVE_OBS) / (1.0 + _Z)
    ref_b = np.interp(np.log(wq), x, g_b * t_f / np.exp(x))
    ref_0 = np.interp(np.log(wq), x, g_0 * t_f / np.exp(x))
    base = np.asarray(exact_model.predict_spectrum({"sigma_v_kms": 0.0}))
    return ref_b * (base / ref_0)


def test_d_sigma_v_exact_path_matches_brute_force_without_resolution(ssp_data_wne):
    """#3436: the exact path with IGM and no resolution broadens at library resolution.

    Resample mode: point (``_spec`` uses ``resample="point"``). Reference: the brute-force
    ln-lambda broadening of the stellar light, times the IGM, at the pixel centers (1e-3).
    The point readout is the closed-form Gaussian of #3512 (measured 7.2e-5 of peak).
    """
    exact = _build(
        ssp_data_wne,
        None,
        spectro=_spec(resolution=None),
        igm=_MADAU,
        sigma_v=Uniform(0.0, 2000.0),
    )
    s150 = _spectrum(exact, 150.0)
    ref = _igm_case_reference(exact, 150.0)
    rel = float(np.max(np.abs(s150 - ref)) / np.max(np.abs(ref)))
    print(f"NUMBERS #3436 exact vs brute-force reference at sigma_v=150: {rel:.6e} of peak")
    assert rel <= 1e-3


def test_d_sigma_v_zero_lut_agrees_with_exact_without_resolution(ssp_data_wne):
    """The sigma_v = 0 LUT matches the exact path (point sampling, IGM, no resolution)."""
    sv = _FREE_SERVED
    exact = _build(ssp_data_wne, None, spectro=_spec(resolution=None), igm=_MADAU, sigma_v=sv)
    lut = _build(
        ssp_data_wne, SpectrumPrecomp(), spectro=_spec(resolution=None), igm=_MADAU, sigma_v=sv
    )
    s_exact, s_lut = _spectrum(exact, 0.0), _spectrum(lut, 0.0)
    rel = float(np.max(np.abs(s_lut - s_exact)) / np.max(np.abs(s_exact)))
    print(f"NUMBERS #3436 LUT vs exact at sigma_v=0, resolution=None: {rel:.6e} of peak")
    assert rel <= 1e-5


def test_d_sigma_v_lut_matches_brute_force_without_resolution(ssp_data_wne):
    """The LUT at sigma_v = 150 without resolution matches the brute-force reference (#3500).

    Resample mode: point. Same reference and bar as the exact-path test above. The table
    convolves its dispersion on the lattice, so this is a real check, not a known gap.
    """
    exact = _build(
        ssp_data_wne,
        None,
        spectro=_spec(resolution=None),
        igm=_MADAU,
        sigma_v=_FREE_SERVED,
    )
    lut = _build(
        ssp_data_wne,
        SpectrumPrecomp(),
        spectro=_spec(resolution=None),
        igm=_MADAU,
        sigma_v=_FREE_SERVED,
    )
    ref = _igm_case_reference(exact, 150.0)
    rel = float(np.max(np.abs(_spectrum(lut, 150.0) - ref)) / np.max(np.abs(ref)))
    print(f"NUMBERS #3436 LUT vs brute-force reference at sigma_v=150: {rel:.6e} of peak")
    assert rel <= 1e-3


# ── (e) exact path bit-for-bit across the extraction ──────────────────────────

_PINNED_EXACT = {
    "synth_B": [
        2.5081139118082243e-12,
        1.8381297451705434e-12,
        1.4118156511701318e-12,
        1.1214557967824752e-12,
        9.130938870797045e-13,
        7.593586524698117e-13,
        6.414925459336969e-13,
    ],
    # Re-pinned 2026-11-15: monotone PCHIP metallicity read (#3434, owner ruling 'True PCHIP on
    # the spectra'). Main's wne_B pins moved by up to 7.8e-5 relative (index 0); synth_B unchanged
    # to 1e-15 (no metallicity read in that grid's exact path beyond round-off).
    "wne_B": [
        8.905642404244887e-29,
        3.0968951597561074e-28,
        4.97183102730229e-28,
        7.368426424015946e-28,
        8.780656340097938e-28,
        1.1662876927396995e-27,
        1.4114629446223282e-27,
    ],
}
# Re-pinned 2026-11-16 (#3500): point mode applies the kernel on the ln-lambda lattice before
# the pixels are read. Old values (main 268d84602): synth_B 2.5081147137035423e-12 ...
# 6.414930444347694e-13, moved by up to 8.4e-7 relative; wne_B 8.809892332215404e-29 ...
# 1.4115376352795165e-27, moved by up to 1.09e-2 relative at index 0, the first pixel, where
# convolving after sampling mirrored the spectrum at the pixel grid's end (the 1-2% end-pixel
# error of #3500), and by at most 5.1e-4 at the other six. Reference for the new order: the
# analytic value in test_bug_3500_point_mode_kernel_before_sampling.py (at most 6e-5 of peak).


def test_e_exact_path_bit_for_bit_across_extraction(synthetic_ssp_wide, ssp_data_wne):
    sv = Uniform(0.0, 2000.0)
    exact = _build(synthetic_ssp_wide, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    np.testing.assert_allclose(
        _spectrum(exact, 150.0)[_IDX], np.asarray(_PINNED_EXACT["synth_B"]), rtol=_PIN_RTOL, atol=0
    )
    exact_w = _build(ssp_data_wne, None, spectro=_spec(), igm=_MADAU, sigma_v=sv)
    np.testing.assert_allclose(
        _spectrum(exact_w, 150.0)[_IDX], np.asarray(_PINNED_EXACT["wne_B"]), rtol=_PIN_RTOL, atol=0
    )


# ── (f) a banded resolution matrix falls back to the exact path ────────────────


def test_f_resolution_matrix_falls_back_bit_exact(synthetic_ssp_wide):
    rm = gaussian_resolution_bands(jnp.asarray(_WAVE_OBS), 2000.0, n_diag=11)
    spectro = Spectroscopy(resample="point", wave_obs=_WAVE_OBS, resolution_matrix=rm)
    sv = Uniform(0.0, 2000.0)
    exact = _build(synthetic_ssp_wide, None, spectro=spectro, igm=_MADAU, sigma_v=sv)
    with pytest.warns(UserWarning, match="resolution matrix"):
        lut = _build(
            synthetic_ssp_wide, SpectrumPrecomp(), spectro=spectro, igm=_MADAU, sigma_v=sv
        )
    np.testing.assert_array_equal(_spectrum(lut, 150.0), _spectrum(exact, 150.0))


# ── (i) free z + fixed sigma_v: the build-time broadened cube ──────────────────


def _build_free_z(ssp, sigma_v):
    obs = Observation(spectroscopy=_spec())
    base = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Uniform(0.05, 0.15),
        igm={"type": "none"},
        sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
    )
    spec = base.spec.merge_observation_params(sigma_v_kms=sigma_v)
    return SEDModel(
        spec, ssp, observation=obs, approx=SpectrumPrecomp(n_z=16, z_min=0.04, z_max=0.16)
    )


@pytest.mark.parametrize("z", [0.08, 0.12])
def test_i_free_z_fixed_sigma_v_cube_equals_sum_route(ssp_data_wne, z):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cube = _build_free_z(ssp_data_wne, Fixed(300.0))
        summed = _build_free_z(ssp_data_wne, _FREE_SERVED)
    chain = cube._cached_component_chain or cube._build_component_chain()
    assert chain[0]._state.ssp_spec_lut is not None
    assert cube._approx.get("spectrum_precomp") and summed._approx.get("spectrum_precomp")
    s_cube = np.asarray(cube.predict_spectrum({"redshift": z}))
    s_sum = np.asarray(summed.predict_spectrum({"redshift": z, "sigma_v_kms": 300.0}))
    # Route bar at 3x the measured residual with the native-slope Hermite read (#3541: z=0.08
    # 5.08e-8, z=0.12 5.56e-8, measured 2026-11-15). Before #3541 the PCHIP read of the stored
    # broadened cube gave 1.05e-5 and 1.18e-5. Same cause as test_g: PCHIP is non-linear.
    bar = {0.08: 1.6e-7, 0.12: 1.7e-7}[z]
    assert float(np.max(np.abs(s_cube - s_sum))) <= bar * float(np.median(np.abs(s_sum)))


# ── (j) routing of a free sigma_v ──────────────────────────────────────────────


def _joint(ssp, approx, *, sigma_v, resample="point", law="calzetti"):
    return _build(
        ssp,
        approx,
        spectro=_spec(resample=resample),
        igm=_MADAU,
        sigma_v=sigma_v,
        photometry=Photometry.from_names(["des_g"]),
        law=law,
    )


def test_j_joint_wave_precomp_keeps_free_sigma_v_spectrum_exact(ssp_data_wne):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = _joint(ssp_data_wne, None, sigma_v=Uniform(0.0, 2000.0))
    # The point-sampled kernel can be narrower than the table resolves, so the table declines
    # and the promotion is blocked without the cost warning (#3500).
    joint = _joint(ssp_data_wne, WavePrecomp(), sigma_v=Uniform(0.0, 2000.0))
    assert joint._approx.get("wave_precomp")
    assert not joint._approx.get("spectrum_precomp")
    np.testing.assert_array_equal(_spectrum(joint, 150.0), _spectrum(exact, 150.0))


def test_j_joint_wave_precomp_promotes_fixed_sigma_v_spectrum(ssp_data_wne):
    # A Calzetti step inside the spectrum keeps the promotion off (#3500); power_law has none.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        joint = _joint(ssp_data_wne, WavePrecomp(), sigma_v=Fixed(150.0), law="power_law")
    assert joint._approx.get("wave_precomp")
    assert joint._approx.get("spectrum_precomp")
    assert not any("sigma_v_kms is free" in str(w.message) for w in caught)


def test_j_joint_explicit_lut_with_free_sigma_v_warns_once_about_cost(ssp_data_wne):
    # Conserving pixels are served at any kernel width, so the explicit table engages (#3500).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        joint = _joint(
            ssp_data_wne, SpectrumPrecomp(), sigma_v=Uniform(0.0, 2000.0), resample="conserving"
        )
    assert joint._approx.get("wave_precomp")
    assert joint._approx.get("spectrum_precomp")
    assert (
        sum(
            "convolves the stellar spectrum on the lattice every draw" in str(w.message)
            for w in caught
        )
        == 1
    )
    assert not any("sigma_v_kms is free" in str(w.message) for w in caught)


def test_j_explicit_lut_with_free_sigma_v_engages_and_warns_once(ssp_data_wne):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        lut = _build(
            ssp_data_wne,
            SpectrumPrecomp(),
            spectro=_spec(resample="conserving"),
            igm=_MADAU,
            sigma_v=Uniform(0.0, 2000.0),
        )
    assert lut._approx.get("spectrum_precomp")
    assert (
        sum(
            "convolves the stellar spectrum on the lattice every draw" in str(w.message)
            for w in caught
        )
        == 1
    )

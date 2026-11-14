# SPDX-License-Identifier: BSD-3-Clause
"""#3436: sigma_v broadens at library resolution when no instrument resolution is set.

With ``resolution=None`` the stellar velocity dispersion used to have no effect at all
(the exact path dropped it, and the spectrum table disagreed with it). Owner ruling
2026-11-13: the dispersion is the galaxy's, so it broadens the stellar group at library
resolution whatever the instrument. The projection adds the dispersion's correction on the
ln-lambda lattice to the unbroadened projection, so the output depends on sigma_v, its
gradient is non-zero and matches finite differences, and at sigma_v = 0 (concrete or traced)
the output is the exact unbroadened projection, bit for bit.

The brute-force reference below is independent of tengri's broadening code: the unbroadened
SED convolved by direct quadrature with a Gaussian in ln(lambda), sampled at the pixels.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.forward.sed_model import SpectrumPrecomp
from tengri.observation.spectrum import compute_spectrum_conserving, project_spectrum
from tengri.parameters import Fixed
from tengri.utils.cosmology import luminosity_distance
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.regression_bug

_C_KMS = 299792.458
_Z = 0.1
_DL = float(luminosity_distance(jnp.asarray(_Z)))
_BARE_SSP = "data/fsps_prsc_miles_chabrier.h5"


def _sed_on_grid():
    """A smooth continuum with two emission lines on a log-uniform rest grid."""
    wave = np.geomspace(1000.0, 2.0e4, 4000)
    cont = 1e28 * (wave / 5000.0) ** -1.0
    lines = 3e29 * np.exp(-0.5 * ((wave - 5007.0) / 2.0) ** 2) + 1e29 * np.exp(
        -0.5 * ((wave - 6563.0) / 2.0) ** 2
    )
    return jnp.asarray(wave), jnp.asarray(cont + lines)


_WAVE_REST, _SED = _sed_on_grid()
_WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)
# 1 Angstrom pixels around the 5007 line (observed 5508 at z = 0.1): sensitive to the broadening.
_FINE_OBS = jnp.linspace(5480.0, 5540.0, 61)


def test_sigma_v_changes_spectrum_without_resolution():
    base = project_spectrum(_SED, _WAVE_REST, _WAVE_OBS, _Z, _DL, sigma_v_kms=0.0, conserving=True)
    broad = project_spectrum(
        _SED, _WAVE_REST, _WAVE_OBS, _Z, _DL, sigma_v_kms=150.0, conserving=True
    )
    rel = float(jnp.max(jnp.abs(broad - base)) / jnp.max(jnp.abs(base)))
    assert rel > 1e-3, f"sigma_v = 150 km/s changed the spectrum by only {rel:.2e} of peak"


def test_sigma_v_gradient_is_nonzero_and_matches_finite_differences():
    weights = jnp.linspace(0.0, 1.0, _WAVE_OBS.shape[0])

    def pixel_sum(sigma):
        flux = project_spectrum(
            _SED, _WAVE_REST, _WAVE_OBS, _Z, _DL, sigma_v_kms=sigma, conserving=True
        )
        return jnp.sum(weights * flux)

    sigma0 = jnp.asarray(150.0)
    grad = float(jax.grad(pixel_sum)(sigma0))
    h = 1e-2
    fd = (float(pixel_sum(sigma0 + h)) - float(pixel_sum(sigma0 - h))) / (2 * h)
    assert np.isfinite(grad), "a non-finite sigma_v gradient"
    assert grad != 0.0, "an identically zero sigma_v gradient: sigma_v is inert"
    assert abs(grad - fd) / abs(fd) <= 1e-4


def test_zero_sigma_v_is_the_exact_unbroadened_projection_bit_for_bit():
    exact = compute_spectrum_conserving(_SED, _WAVE_REST, _WAVE_OBS, _Z, _DL)
    zero = project_spectrum(_SED, _WAVE_REST, _WAVE_OBS, _Z, _DL, sigma_v_kms=0.0, conserving=True)
    np.testing.assert_array_equal(np.asarray(zero), np.asarray(exact))


@pytest.mark.parametrize("conserving", [True, False], ids=["pixel-means", "point"])
def test_traced_zero_sigma_v_is_bit_identical_to_sigma_v_absent(conserving):
    """A traced sigma_v of exactly 0 under jit gives the sigma_v-absent output."""

    def absent():
        return project_spectrum(_SED, _WAVE_REST, _WAVE_OBS, _Z, _DL, conserving=conserving)

    def traced(sigma):
        return project_spectrum(
            _SED, _WAVE_REST, _WAVE_OBS, _Z, _DL, sigma_v_kms=sigma, conserving=conserving
        )

    zero = jax.jit(traced)(jnp.asarray(0.0))
    np.testing.assert_array_equal(np.asarray(zero), np.asarray(jax.jit(absent)()))


def _synthetic_ssp() -> SSPData:
    """Smooth continuum with sharp absorption near both window ends and in the middle."""
    wave = np.geomspace(3500.0, 7500.0, 7200)
    ages = np.linspace(-1.0, 1.14, 12)
    lgmet = np.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN
    absorb = np.ones_like(wave)
    for center in (4699.5, 5140.0, 5585.5):
        absorb *= 1.0 - 0.6 * np.exp(-0.5 * ((wave - center) / 0.8) ** 2)
    base = (wave / 5000.0) ** -1.5 * absorb
    age_scale = (1.0 + np.arange(ages.size) * 0.05)[:, None]
    met_scale = np.array([1.0, 1.3, 1.6])[:, None, None]
    return SSPData(
        ssp_wave=jnp.asarray(wave),
        ssp_flux=jnp.asarray(met_scale * age_scale[None, :, :] * base[None, None, :]),
        ssp_lg_age_gyr=jnp.asarray(ages),
        ssp_lgmet=jnp.asarray(lgmet),
    )


_MODEL_Z = 0.021
_SIGMA_V_KMS = 200.0
_MODEL_WAVE_OBS = np.linspace(4800.0, 5700.0, 450)
_REL_TOL = 1e-3  # largest per-pixel error, relative to the largest reference pixel


@pytest.fixture(scope="module")
def ssp():
    return _synthetic_ssp()


def _observation():
    return tengri.Observation(
        spectroscopy=tengri.Spectroscopy(
            wave_obs=jnp.asarray(_MODEL_WAVE_OBS), resolution=None, resample="point"
        )
    )


def _model(ssp, sigma, approx=None):
    kwargs = dict(
        ssp_data=ssp,
        observation=_observation(),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(10.0),
            "log_total_mass": Fixed(10.0),
        },
        met={"logzsol": Fixed(-0.2)},
        redshift=Fixed(_MODEL_Z),
        neb={"type": "none"},
    )
    base = SEDModel.build(**kwargs, approx=None)
    if sigma is None:
        return base
    spec = base.spec.merge_observation_params(sigma_v_kms=sigma)
    return SEDModel(spec, ssp, observation=_observation(), approx=approx)


def _brute_force_reference(ssp):
    """Brute-force Gaussian-in-ln(lambda) broadening of the sigma = 0 SED, scaled to the model."""
    model0 = _model(ssp, None)
    state = model0.predict_state({})
    wave_rest = np.asarray(state.wave)
    sed0 = np.asarray(state.derived["lnu_age"].sum(axis=0))
    sigma_ln = _SIGMA_V_KMS / _C_KMS
    reach = 12.0 * sigma_ln
    wq = _MODEL_WAVE_OBS / (1.0 + _MODEL_Z)
    out = (wave_rest >= wq.min() * np.exp(-reach)) & (wave_rest <= wq.max() * np.exp(reach))
    src = (wave_rest >= wq.min() * np.exp(-2 * reach)) & (
        wave_rest <= wq.max() * np.exp(2 * reach)
    )
    x, dx = np.log(wave_rest[src]), np.gradient(np.log(wave_rest))[src]
    kernel = np.exp(-0.5 * ((np.log(wave_rest[out])[:, None] - x[None, :]) / sigma_ln) ** 2)
    kernel = kernel * dx[None, :]
    kernel /= kernel.sum(axis=1, keepdims=True)
    brute = np.interp(wq, wave_rest[out], kernel @ sed0[src])
    no_sigma = np.asarray(model0.predict_observables({}).spec_fnu)
    return brute * no_sigma / np.interp(wq, wave_rest, sed0)


def _max_rel(a, ref):
    return float(np.max(np.abs(np.asarray(a) - ref)) / np.max(np.abs(ref)))


def test_exact_path_matches_brute_force_ln_lambda_broadening(ssp):
    exact = np.asarray(_model(ssp, Fixed(_SIGMA_V_KMS)).predict_observables({}).spec_fnu)
    assert _max_rel(exact, _brute_force_reference(ssp)) < _REL_TOL


def test_table_and_exact_agree_without_a_resolution(ssp):
    """Table vs exact without a resolution (#3413 owns the table rule; see the PR body)."""
    exact = np.asarray(_model(ssp, Fixed(_SIGMA_V_KMS)).predict_observables({}).spec_fnu)
    table = np.asarray(
        _model(ssp, Fixed(_SIGMA_V_KMS), SpectrumPrecomp()).predict_observables({}).spec_fnu
    )
    assert _max_rel(table, exact) < _REL_TOL


def test_fixed_zero_sigma_v_is_unchanged_without_a_resolution(ssp):
    without = np.asarray(_model(ssp, None).predict_observables({}).spec_fnu)
    zero = np.asarray(_model(ssp, Fixed(0.0)).predict_observables({}).spec_fnu)
    np.testing.assert_array_equal(zero, without)


def test_traced_zero_model_sigma_v_is_bit_identical_to_absent(ssp):
    absent_model = _model(ssp, None)
    model = _model(ssp, Uniform(0.0, 500.0))

    @jax.jit
    def absent():
        return absent_model.predict_observables({}).spec_fnu

    @jax.jit
    def spec(s):
        return model.predict_observables({"sigma_v_kms": s}).spec_fnu

    np.testing.assert_array_equal(np.asarray(spec(jnp.asarray(0.0))), np.asarray(absent()))


def test_free_sigma_v_has_a_live_gradient_without_a_resolution(ssp):
    model = _model(ssp, Uniform(0.0, 500.0))

    def first_pixels(s):
        return jnp.sum(model.predict_observables({"sigma_v_kms": s}).spec_fnu[:5])

    g = float(jax.grad(first_pixels)(jnp.asarray(150.0)))
    h = 1e-2
    fd = float((first_pixels(150.0 + h) - first_pixels(150.0 - h)) / (2 * h))
    assert np.isfinite(g) and g != 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-4)


@pytest.mark.skipif(not os.path.isfile(_BARE_SSP), reason="bare-stellar SSP not under data/")
def test_model_sigma_v_without_resolution_moves_the_output():
    from tengri import (
        Fixed as FixedP,
        Observation,
        Parameters,
        SEDModel as Model,
        Spectroscopy,
        load_ssp_data,
    )

    ssp_file = load_ssp_data(_BARE_SSP)
    wave = jnp.linspace(3800.0, 9200.0, 120)

    def model(sigma):
        spec = Parameters(
            sfh_dpl_alpha=FixedP(1.5),
            dust_tau_bc=Uniform(0.0, 1.0),
            dust_tau_diff=Uniform(0.0, 1.0),
            redshift=FixedP(_Z),
            sigma_v_kms=FixedP(sigma),
        )
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave))
        return Model(spec, ssp_file, observation=obs)

    m0, m150 = model(0.0), model(150.0)
    params = {
        **m0.spec.sample(jax.random.PRNGKey(0)),
        "dust_tau_bc": jnp.asarray(0.3),
        "dust_tau_diff": jnp.asarray(0.2),
    }
    flux0 = m0.predict_spectrum(params)
    flux150 = m150.predict_spectrum(params)
    rel = float(jnp.max(jnp.abs(flux150 - flux0)) / jnp.max(jnp.abs(flux0)))
    assert rel > 1e-3


def _brute_ln_lambda_broaden(wave_rest, sed_rest, sigma_v_kms, *, step=2e-6):
    """Gaussian in ln(lambda) on lambda F_lambda, on a fine uniform ln-lambda grid.

    The reference convolves the linearly interpolated SED on a grid far finer than the
    lattice (step 2e-6, against a 5e-5 lattice and a 8e-5 line), with a direct Gaussian
    kernel applied by FFT and no lattice. Returns the fine grid (ln lambda) and the
    broadened lambda F_lambda on it, so that a caller can integrate pixel means.
    """
    sig = sigma_v_kms / _C_KMS
    x = np.arange(np.log(wave_rest[0]), np.log(wave_rest[-1]), step)
    g = np.interp(np.exp(x), wave_rest, sed_rest) * np.exp(x)
    half = int(np.ceil(12.0 * sig / step))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * step / sig) ** 2)
    k /= k.sum()
    n = g.size + k.size
    conv = np.fft.irfft(np.fft.rfft(g, n) * np.fft.rfft(k, n), n)
    return x, conv[half : half + g.size]


def _brute_pixel_means(x, g_broad, wave_obs, z):
    """Pixel means of F_lambda over the pixel edges, from the fine ln-lambda integral."""
    wq = np.asarray(wave_obs) / (1.0 + z)
    mid = 0.5 * (wq[1:] + wq[:-1])
    edges = np.concatenate([[wq[0] - (mid[0] - wq[0])], mid, [wq[-1] + (wq[-1] - mid[-1])]])
    # Trapezoid cumulative of lambda F in ln lambda: the integral of F dlambda.
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (g_broad[1:] + g_broad[:-1]) * np.diff(x))])
    cum_at = np.interp(np.log(edges), x, cum, left=0.0)
    return np.diff(cum_at) / np.diff(edges)


def test_conserving_pixel_means_match_brute_force_ln_lambda_broadening():
    """Pixel means of the sigma_v-broadened SED, against the fine ln-lambda reference.

    A 1 Angstrom line inside the window exercises the cubic readout of the lattice's
    cumulative integral. A 0.4 Angstrom line (about 1.5 lattice cells per sigma) reads 4e-3 of
    peak off the reference at this spacing, so the test line is not narrower than that.
    """
    wave = np.geomspace(1000.0, 2.0e4, 400000)
    sed = 1e28 * (wave / 5000.0) ** -1.0 + 3e29 * np.exp(-0.5 * ((wave - 5007.0) / 1.0) ** 2)
    sigma = 150.0
    x, g_broad = _brute_ln_lambda_broaden(wave, sed, sigma)
    ref_plain = _brute_pixel_means(x, np.interp(np.exp(x), wave, sed) * np.exp(x), _FINE_OBS, _Z)
    ref = _brute_pixel_means(x, g_broad, _FINE_OBS, _Z)
    base = np.asarray(
        project_spectrum(jnp.asarray(sed), jnp.asarray(wave), _FINE_OBS, _Z, _DL, conserving=True)
    )
    flux = np.asarray(
        project_spectrum(
            jnp.asarray(sed),
            jnp.asarray(wave),
            _FINE_OBS,
            _Z,
            _DL,
            sigma_v_kms=sigma,
            conserving=True,
        )
    )
    # The rest-to-observed unit factor is per pixel; take it from the sigma = 0 projection.
    expected = ref * (base / ref_plain)
    rel = np.max(np.abs(flux - expected)) / np.max(np.abs(expected))
    assert rel <= 1e-3, f"pixel means differ from the ln-lambda reference by {rel:.2e} of peak"


def test_igm_is_applied_after_the_dispersion_without_resolution():
    """The IGM multiplies the dispersed stellar light: T * B(S), not B(T * S)."""
    from types import SimpleNamespace

    from tengri.observation.observation import project_spectrum_kernel_split

    wave = np.geomspace(1000.0, 2.0e4, 400000)
    sed = 1e28 * (wave / 5000.0) ** -1.0
    # A narrow absorption line well inside the window: the order of T and B matters here.
    igm = 1.0 - 0.9 * np.exp(-0.5 * ((wave - 5000.0) / 3.0) ** 2)
    sigma = 150.0
    x, g_broad = _brute_ln_lambda_broaden(wave, sed, sigma)
    t_fine = np.interp(np.exp(x), wave, igm)
    expected = _brute_pixel_means(x, g_broad * t_fine, _FINE_OBS, _Z)  # T after B
    _, g_swapped = _brute_ln_lambda_broaden(wave, sed * igm, sigma)  # B after T
    swapped = _brute_pixel_means(x, g_swapped, _FINE_OBS, _Z)
    plain = _brute_pixel_means(x, np.interp(np.exp(x), wave, sed) * np.exp(x), _FINE_OBS, _Z)
    base = np.asarray(
        project_spectrum(jnp.asarray(sed), jnp.asarray(wave), _FINE_OBS, _Z, _DL, conserving=True)
    )
    unit = base / plain  # the rest-to-observed factor, per pixel
    expected, swapped = expected * unit, swapped * unit
    state = SimpleNamespace(derived={})
    got = np.asarray(
        project_spectrum_kernel_split(
            state,
            jnp.asarray(sed),
            jnp.asarray(igm),
            jnp.asarray(wave),
            _FINE_OBS,
            _Z,
            _DL,
            resolution=None,
            sigma_lib_kms=0.0,
            sigma_v_kms=sigma,
            conserving=True,
        )
    )
    rel = np.max(np.abs(got - expected)) / np.max(np.abs(expected))
    rel_swapped = np.max(np.abs(swapped - expected)) / np.max(np.abs(expected))
    assert rel_swapped > 1e-2, "the IGM test would not tell the two orders apart"
    assert rel <= 1e-3, f"IGM-after-dispersion differs from the reference by {rel:.2e} of peak"

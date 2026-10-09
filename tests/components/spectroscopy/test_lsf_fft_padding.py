# SPDX-License-Identifier: BSD-3-Clause
"""Padded, fast-length, windowed LSF convolution and the rest-grid sigma_v crop (#2832).

The three Gaussian FFT sites in ``observation/spectrum.py`` convolve
circularly over the raw array length, so the kernel wrapped from the red end onto
the blue end (#2712), and a prime pixel count ran at a slow transform length
(#2832). Three kinds of reference are used:

- ``_ref_real_space_*``: a real-space sampled Gaussian, written with ``np.convolve``
  on a reflecting (symmetric) extension of the spectrum. It shares no code with the
  FFT path and is the independent check, valid where sigma_pix >= 3.
- ``_ref_gaussian_conv`` / ``_ref_variable_conv``: the same Fourier kernel on a
  convolution with a reflecting (symmetric) boundary over a margin of 16 n, which is
  converged for these kernels. They check the Fourier path at any sigma.
- Layout comparisons force the full-length transform and compare it with the
  windowed one on the same padding.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FREE, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.observation import spectrum as spectrum_mod
from tengri.observation.spectrum import (
    _C_KM_S,
    _FWHM_TO_SIGMA,
    apply_lsf,
    broaden_velocity_only,
    project_spectrum,
    velocity_broaden,
)

pytestmark = pytest.mark.contract

#: Reference margin, in multiples of n. The old kernel is sampled in frequency, so it
#: has tails that decay only like h**-2 in the margin h; the margin here is 16 n, where
#: doubling it changes the reference by less than 1e-10 relative.
_REF_MARGIN_FACTOR = 16
_R_SCALAR = 3000.0


def _rel_tol(n: int) -> float:
    """Tolerance relative to max|flux| for a grid of ``n`` pixels.

    On the production grid (n = 7909, R = 3000, sigma about 1.1 px) the result agrees
    with the converged reference to 1.3e-8 (constant R) and 2.2e-8 (variable R) on
    this steep spectrum at the padding of :func:`_lsf_pad_pixels`. The residue is the
    Fourier kernel's tail beyond that padding, so the production grid is held to
    1e-7. The n = 1000 grid samples the LSF at sigma of about 0.1 to 0.2 px, where
    the ringing tail is larger: even a full-length margin (n - 1 pixels) misses 1e-8
    there, so that grid is held to 1e-5.
    """
    return 1e-7 if n >= 5000 else 1e-5


def _sigma_inst_kms(resolution):
    return _C_KM_S / (_FWHM_TO_SIGMA * np.asarray(resolution, dtype=np.float64))


def _spectrum_with_edge_features(n: int) -> np.ndarray:
    """Steep slope plus narrow lines near both ends and in the interior."""
    x = np.arange(n, dtype=np.float64) / n
    flux = 1.0e3 * np.exp(-3.0 * x)
    pix = np.arange(n, dtype=np.float64)
    for center, amp in ((15.0, 500.0), (n - 20.0, 300.0), (n // 2, 200.0)):
        flux += amp * np.exp(-0.5 * ((pix - center) / 1.5) ** 2)
    return flux


def _ref_real_space(spec: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Real-space sampled Gaussian on a reflecting (symmetric) boundary.

    Independent of the FFT path: the taps are the Gaussian sampled at integers,
    truncated at 8 sigma and normalized, applied with ``np.convolve`` in ``valid``
    mode to a spectrum padded by reflection. The sampled and Fourier kernels agree
    to about 1e-10 for sigma_pix >= 3, so this is only used there.
    """
    half = int(np.ceil(8.0 * sigma_pix))
    t = np.arange(-half, half + 1, dtype=np.float64)
    taps = np.exp(-(t**2) / (2.0 * sigma_pix**2))
    taps /= taps.sum()
    return np.convolve(np.pad(spec, half, mode="symmetric"), taps, mode="valid")


def _ref_real_space_variable(spec, wave, sigma_eff, n_bins=16):
    """Piecewise-bin real-space reference: the bin geometry of the full array."""
    n = spec.size
    dln_local = np.gradient(np.log(wave))
    width = n / n_bins
    half_w = 0.75 * width
    pix = np.arange(n, dtype=np.float64)
    acc = np.zeros(n)
    total = np.zeros(n)
    for k in range(n_bins):
        center = (k + 0.5) * width
        mask = (np.abs(pix - center) < width).astype(np.float64)
        count = max(mask.sum(), 1.0)
        sigma_mean = (sigma_eff * mask).sum() / count
        dln_mean = (dln_local * mask).sum() / count
        sigma_pix = (sigma_mean / _C_KM_S) / dln_mean
        dist = np.abs(pix - center) / half_w
        weight = np.where(dist < 1.0, 0.5 * (1.0 + np.cos(np.pi * dist)), 0.0)
        acc += weight * _ref_real_space(spec, sigma_pix)
        total += weight
    return acc / np.maximum(total, 1e-30)


def _ref_gaussian_conv(spec: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Fourier Gaussian convolution on a reflecting boundary over a 16 n margin."""
    n = spec.size
    margin = _REF_MARGIN_FACTOR * n
    padded = np.pad(spec, margin, mode="symmetric")
    length = padded.size
    freq = np.fft.rfftfreq(length)
    kernel = np.exp(-2.0 * np.pi**2 * sigma_pix**2 * freq**2)
    conv = np.fft.irfft(np.fft.rfft(padded) * kernel, n=length)
    return conv[margin : margin + n]


def _ref_variable_conv(
    spec: np.ndarray, wave: np.ndarray, sigma_eff: np.ndarray, n_bins: int = 16
) -> np.ndarray:
    """Piecewise-bin Fourier convolution, full-array bin geometry, reflecting boundary."""
    n = spec.size
    dln_local = np.gradient(np.log(wave))
    width = n / n_bins
    half_w = 0.75 * width
    pix = np.arange(n, dtype=np.float64)
    margin = _REF_MARGIN_FACTOR * n
    padded = np.pad(spec, margin, mode="symmetric")
    length = padded.size
    freq = np.fft.rfftfreq(length)
    flux_ft = np.fft.rfft(padded)
    acc = np.zeros(n)
    total = np.zeros(n)
    for k in range(n_bins):
        center = (k + 0.5) * width
        mask = (np.abs(pix - center) < width).astype(np.float64)
        count = max(mask.sum(), 1.0)
        sigma_mean = (sigma_eff * mask).sum() / count
        dln_mean = (dln_local * mask).sum() / count
        sigma_pix = (sigma_mean / _C_KM_S) / dln_mean
        kernel = np.exp(-2.0 * np.pi**2 * sigma_pix**2 * freq**2)
        conv = np.fft.irfft(flux_ft * kernel, n=length)[margin : margin + n]
        dist = np.abs(pix - center) / half_w
        weight = np.where(dist < 1.0, 0.5 * (1.0 + np.cos(np.pi * dist)), 0.0)
        acc += weight * conv
        total += weight
    return acc / np.maximum(total, 1e-30)


def _assert_close_to_reference(out, ref, label):
    out = np.asarray(out)
    tol = _rel_tol(out.size)
    scale = np.max(np.abs(ref))
    err = np.max(np.abs(out - ref)) / scale
    assert err <= tol, f"{label}: relative error {err:.3e} > {tol:g}"
    edge = np.concatenate([out[:20] - ref[:20], out[-20:] - ref[-20:]])
    edge_err = np.max(np.abs(edge)) / scale
    assert edge_err <= tol, f"{label}: edge relative error {edge_err:.3e} > {tol:g}"


# ── 1. Sweep over the three FFT sites against the big-margin reference ──


@pytest.mark.parametrize("n", [7909, 1000])
def test_constant_r_log_grid_matches_big_margin_reference(n):
    wave = np.geomspace(3600.0, 9824.0, n)
    spec = _spectrum_with_edge_features(n)
    out = apply_lsf(jnp.asarray(spec), jnp.asarray(wave), resolution=_R_SCALAR)
    sigma_eff = float(_sigma_inst_kms(_R_SCALAR))
    dln = np.log(wave[1] / wave[0])
    ref = _ref_gaussian_conv(spec, (sigma_eff / _C_KM_S) / dln)
    _assert_close_to_reference(out, ref, f"constant R, n={n}")


@pytest.mark.parametrize("n", [7909, 1000])
def test_variable_r_linear_grid_matches_big_margin_reference(n):
    wave = np.linspace(3600.0, 9824.0, n)
    spec = _spectrum_with_edge_features(n)
    resolution = np.full(n, _R_SCALAR)
    out = apply_lsf(jnp.asarray(spec), jnp.asarray(wave), resolution=jnp.asarray(resolution))
    ref = _ref_variable_conv(spec, wave, _sigma_inst_kms(resolution))
    _assert_close_to_reference(out, ref, f"variable R, n={n}")


@pytest.mark.parametrize("n", [7909, 1000])
def test_velocity_broaden_matches_big_margin_reference(n):
    wave = np.geomspace(3600.0, 9824.0, n)
    spec = _spectrum_with_edge_features(n)
    out = velocity_broaden(jnp.asarray(spec), jnp.asarray(wave), 60.0)
    dln = np.log(wave[1] / wave[0])
    ref = _ref_gaussian_conv(spec, (60.0 / _C_KM_S) / dln)
    _assert_close_to_reference(out, ref, f"velocity_broaden, n={n}")


@pytest.mark.parametrize("n", [7909, 1000])
def test_broaden_velocity_only_matches_big_margin_reference(n):
    wave = np.linspace(3600.0, 9824.0, n)
    spec = _spectrum_with_edge_features(n)
    out = broaden_velocity_only(jnp.asarray(spec), jnp.asarray(wave), 120.0, 16)
    ref = _ref_variable_conv(spec, wave, np.full(n, 120.0))
    _assert_close_to_reference(out, ref, f"broaden_velocity_only, n={n}")


def test_single_bin_full_length_path_matches_reference():
    """n_bins=1 cannot save work by windowing, so it takes the full-length path."""
    n = 1000
    wave = np.linspace(3600.0, 9824.0, n)
    spec = _spectrum_with_edge_features(n)
    out = broaden_velocity_only(jnp.asarray(spec), jnp.asarray(wave), 120.0, 1)
    ref = _ref_variable_conv(spec, wave, np.full(n, 120.0), n_bins=1)
    _assert_close_to_reference(out, ref, "broaden_velocity_only, one bin")


# ── 2. The edge wrap is gone ──


def test_first_pixel_after_broadening_is_not_wrapped_from_red_end():
    n = 7909
    wave = np.geomspace(3600.0, 9824.0, n)
    spec = np.linspace(1.0, 100.0, n)
    out = np.asarray(velocity_broaden(jnp.asarray(spec), jnp.asarray(wave), 300.0))
    dln = np.log(wave[1] / wave[0])
    ref = _ref_gaussian_conv(spec, (300.0 / _C_KM_S) / dln)
    assert abs(out[0] - ref[0]) <= 0.01 * abs(ref[0]), (out[0], ref[0])
    assert abs(out[-1] - ref[-1]) <= 0.01 * abs(ref[-1]), (out[-1], ref[-1])


def test_variable_path_first_pixel_is_not_wrapped_from_red_end():
    n = 7909
    wave = np.linspace(3600.0, 9824.0, n)
    spec = np.linspace(1.0, 100.0, n)
    resolution = np.full(n, _R_SCALAR)
    out = np.asarray(
        apply_lsf(jnp.asarray(spec), jnp.asarray(wave), resolution=jnp.asarray(resolution))
    )
    ref = _ref_variable_conv(spec, wave, _sigma_inst_kms(resolution))
    assert abs(out[0] - ref[0]) <= 0.01 * abs(ref[0]), (out[0], ref[0])


# ── 3. Static skip for a concrete non-positive sigma_v ──


def test_literal_zero_sigma_v_emits_no_fft():
    n = 512
    wave = jnp.asarray(np.linspace(3600.0, 9824.0, n))
    jaxpr = jax.make_jaxpr(lambda f: broaden_velocity_only(f, wave, 0.0, 16))(jnp.ones(n))
    assert "fft" not in str(jaxpr)


def test_traced_sigma_v_still_emits_fft():
    n = 512
    wave = jnp.asarray(np.linspace(3600.0, 9824.0, n))
    jaxpr = jax.make_jaxpr(lambda f, s: broaden_velocity_only(f, wave, s, 16))(
        jnp.ones(n), jnp.asarray(0.0)
    )
    assert "fft" in str(jaxpr)


def test_literal_zero_sigma_v_returns_input_unchanged():
    n = 64
    wave = np.linspace(3600.0, 9824.0, n)
    flux = np.random.default_rng(0).normal(size=n)
    out = broaden_velocity_only(flux, wave, 0.0, 16)
    np.testing.assert_array_equal(np.asarray(out), flux)


def test_zero_sigma_v_on_jax_array_keeps_where_guard_result():
    """A jax-array sigma is traced-like: the jnp.where guard still gives the identity."""
    n = 256
    wave = jnp.asarray(np.linspace(3600.0, 9824.0, n))
    flux = jnp.asarray(np.random.default_rng(1).normal(size=n))
    out = broaden_velocity_only(flux, wave, jnp.asarray(0.0), 16)
    np.testing.assert_allclose(np.asarray(out), np.asarray(flux), rtol=0, atol=0)


# ── 3b. Windowing is actually used (kills a forced full-length path) ──


def test_windowed_path_uses_short_transforms_on_large_array():
    n = 7909
    wave = np.linspace(3600.0, 9824.0, n)
    spec = jnp.asarray(_spectrum_with_edge_features(n))
    # Built outside the trace: a resolution created inside make_jaxpr is a tracer, and
    # its sigma is then bounded as traced, as it would be for a fitted resolution.
    resolution = np.full(n, _R_SCALAR)
    wave_j = jnp.asarray(wave)
    jaxpr = str(jax.make_jaxpr(lambda s: apply_lsf(spec, wave_j, resolution, sigma_v_kms=s))(60.0))
    lengths = [int(m) for m in re.findall(r"fft_lengths=\((\d+)", jaxpr)]
    assert lengths, "no FFT in the traced LSF"
    pad = spectrum_mod._lsf_pad_pixels(wave, n, spectrum_mod._LSF_MAX_SIGMA_KMS, 16)
    full = spectrum_mod._next_fast_fft_len(n + 2 * pad)
    assert max(lengths) < full, (max(lengths), full)


# ── 4. Gradients with respect to sigma through the variable path ──


def test_grad_wrt_sigma_v_matches_central_differences():
    n = 2000
    wave = jnp.asarray(np.linspace(3600.0, 9824.0, n))
    spec = jnp.asarray(_spectrum_with_edge_features(n))
    resolution = jnp.asarray(np.full(n, _R_SCALAR))
    weights = jnp.asarray(np.cos(np.arange(n) / 37.0))

    def loss(s):
        return jnp.sum(weights * apply_lsf(spec, wave, resolution, sigma_v_kms=s))

    s0, step = 80.0, 0.25
    grad = float(jax.grad(loss)(jnp.asarray(s0)))
    fd = float((loss(s0 + step) - loss(s0 - step)) / (2.0 * step))
    assert abs(grad - fd) <= 1e-5 * abs(fd), (grad, fd)


# ── 5. Rest-grid sigma_v crop (item 3) ──


def test_rest_grid_window_brackets_observed_range():
    wave_rest = np.geomspace(1000.0, 20000.0, 4000)
    wave_obs = np.linspace(3600.0, 9824.0, 1500)
    lo, hi = spectrum_mod.rest_grid_observed_window(wave_rest, wave_obs, 1.0)
    assert 0 < lo < hi < wave_rest.size
    assert wave_rest[lo] <= wave_obs.min() / 2.0
    assert wave_rest[hi - 1] >= wave_obs.max() / 2.0


def test_rest_grid_window_is_none_when_redshift_is_traced():
    wave_rest = jnp.asarray(np.geomspace(1000.0, 20000.0, 400))
    wave_obs = jnp.asarray(np.linspace(3600.0, 9824.0, 150))

    seen = []

    def probe(z):
        seen.append(spectrum_mod.rest_grid_observed_window(wave_rest, wave_obs, z))
        return z

    jax.make_jaxpr(probe)(jnp.asarray(1.0))
    assert seen == [None]


def test_cropped_rest_grid_pass_equals_full_grid_on_observed_grid():
    n = 4000
    z = 1.0
    wave_rest = np.geomspace(1000.0, 20000.0, n)
    wave_obs = np.linspace(3600.0, 9824.0, 1500)
    bump = 0.5 * np.exp(-0.5 * ((wave_rest - 2800.0) / 4.0) ** 2)
    sed = 1.0e20 * (wave_rest / 5000.0) ** -1.5 * (1.0 + bump)
    sigma_v = 150.0
    dl_cm = 1.0e28

    full = broaden_velocity_only(jnp.asarray(sed), jnp.asarray(wave_rest), sigma_v, 16)
    flux_full = project_spectrum(
        full, jnp.asarray(wave_rest), jnp.asarray(wave_obs), z, dl_cm, resolution=_R_SCALAR
    )

    lo, hi = spectrum_mod.rest_grid_observed_window(wave_rest, wave_obs, z)
    assert (lo, hi) != (0, n), "the crop must remove something to test anything"
    cropped = spectrum_mod.broaden_velocity_only_window(
        jnp.asarray(sed), jnp.asarray(wave_rest), sigma_v, 16, lo, hi
    )
    flux_cropped = project_spectrum(
        cropped, jnp.asarray(wave_rest), jnp.asarray(wave_obs), z, dl_cm, resolution=_R_SCALAR
    )
    ref = np.asarray(flux_full)
    err = np.max(np.abs(np.asarray(flux_cropped) - ref)) / np.max(np.abs(ref))
    assert err <= 1e-10, err


def test_observation_branch_uses_cropped_pass_for_concrete_redshift(monkeypatch):
    """The IGM branch hands the rest-grid pass to the window helper when z is concrete."""
    from tengri.observation import observation as obs_mod

    calls = []
    real = spectrum_mod.broaden_velocity_only_window

    def spy(*args, **kwargs):
        calls.append(args[4:6])
        return real(*args, **kwargs)

    monkeypatch.setattr(spectrum_mod, "broaden_velocity_only_window", spy)
    n = 4000
    wave_rest = np.geomspace(1000.0, 20000.0, n)
    wave_obs = np.linspace(3600.0, 9824.0, 1500)
    window = spectrum_mod.rest_grid_observed_window(wave_rest, wave_obs, 1.0)
    assert window is not None
    sed = np.ones(n)
    igm = np.ones(n)
    state = _FakeState(sed_rest=sed)
    obs_mod.project_spectrum_kernel_split(
        state,
        jnp.asarray(sed),
        jnp.asarray(igm),
        jnp.asarray(wave_rest),
        jnp.asarray(wave_obs),
        1.0,
        1.0e28,
        resolution=_R_SCALAR,
        sigma_lib_kms=0.0,
        sigma_v_kms=150.0,
        window_z=1.0,
    )
    assert calls == [window]


class _FakeState:
    """Minimal ForwardState stand-in: only the fields the stellar split reads."""

    def __init__(self, sed_rest):
        self.sed_intrinsic = sed_rest
        zeros = np.zeros_like(sed_rest)
        self.derived = {
            "sed_nebular": zeros,
            "sed_shock": zeros,
            "sed_agn_lines_attenuated": zeros,
        }


# ── 6. Helpers ──


def _is_5_smooth(m: int) -> bool:
    for p in (2, 3, 5):
        while m % p == 0:
            m //= p
    return m == 1


def test_next_fast_fft_len_returns_smallest_5_smooth_length():
    fast = spectrum_mod._next_fast_fft_len
    assert fast(1) == 1
    assert fast(7) == 8
    assert fast(11) == 12
    for n in range(1, 2000):
        m = fast(n)
        assert m >= n and _is_5_smooth(m)
        assert all(not _is_5_smooth(k) for k in range(n, m))


def test_lsf_pad_pixels_concrete_and_traced_branches():
    n = 7909
    wave = np.geomspace(3600.0, 9824.0, n)
    dln = np.log(wave[1] / wave[0])
    five_sigma = int(np.ceil(5.0 * 2000.0 / (_C_KM_S * dln)))
    expected = max(five_sigma, spectrum_mod._LSF_RINGING_FLOOR_PIXELS)
    assert spectrum_mod._lsf_pad_pixels(wave, n, 2000.0) == expected
    assert spectrum_mod._lsf_pad_pixels(jnp.asarray(wave), n, 2000.0) == expected

    seen = []

    def probe(w):
        seen.append(spectrum_mod._lsf_pad_pixels(w, n, 2000.0))
        return w

    jax.make_jaxpr(probe)(jnp.asarray(wave))
    assert seen == [n - 1], "a traced grid has no pixel scale, so the exact choice is n - 1"


# ── 6. The window_z plumbing (#2832) ──


def _kernel_split_inputs(n: int = 4000, z: float = 1.0):
    wave_rest = np.geomspace(1000.0, 20000.0, n)
    wave_obs = np.linspace(3600.0, 9824.0, 1500)
    sed = (
        1.0e20
        * (wave_rest / 5000.0) ** -1.5
        * (1.0 + 0.5 * np.exp(-0.5 * ((wave_rest - 2800.0) / 4.0) ** 2))
    )
    return wave_rest, wave_obs, sed, z


def _kernel_split(sed, wave_rest, wave_obs, z, **kwargs):
    from tengri.observation.observation import project_spectrum_kernel_split
    from tengri.protocols.component import ForwardState

    n = sed.size
    state = ForwardState(
        wave=jnp.asarray(wave_rest),
        sed_intrinsic=jnp.asarray(sed),
        derived={"sed_nebular": jnp.zeros(n), "sed_shock": jnp.zeros(n)},
    )
    dl_cm = float(np.sqrt((1.0 + z) / (4.0 * np.pi)))
    return np.asarray(
        project_spectrum_kernel_split(
            state,
            jnp.asarray(sed),
            jnp.ones(n),
            jnp.asarray(wave_rest),
            jnp.asarray(wave_obs),
            z,
            dl_cm,
            resolution=_R_SCALAR,
            sigma_lib_kms=0.0,
            sigma_v_kms=150.0,
            **kwargs,
        )
    )


@pytest.mark.parametrize("z", [1.0, 3.0])
def test_window_z_cropped_split_matches_full_grid(z, monkeypatch):
    """A concrete window_z crops the rest pass; the observed flux is unchanged to 1e-10."""
    wave_rest, wave_obs, sed, _ = _kernel_split_inputs(z=z)
    calls = []
    real = spectrum_mod.broaden_velocity_only_window

    def spy(*args, **kwargs):
        calls.append(args[4:6])
        return real(*args, **kwargs)

    monkeypatch.setattr(spectrum_mod, "broaden_velocity_only_window", spy)
    full = _kernel_split(sed, wave_rest, wave_obs, z, window_z=None)
    assert calls == [], "window_z=None must keep the full-grid pass"
    cropped = _kernel_split(sed, wave_rest, wave_obs, z, window_z=z)
    assert len(calls) == 1, "a concrete window_z must run the cropped pass"
    lo, hi = calls[0]
    assert (lo, hi) != (0, wave_rest.size)
    err = np.max(np.abs(cropped - full)) / np.max(np.abs(full))
    assert err <= 1e-10, err


def test_window_z_wrong_redshift_is_not_interchangeable():
    """A window built at a different redshift changes the answer, so window_z must be z.

    Measured: a window at z = 1 applied to a z = 3 spectrum is off by 2.4e-5 relative,
    which is why the observation layer keys the window to the Fixed redshift and not to
    the lower bound of a free redshift prior.
    """
    wave_rest, wave_obs, sed, _ = _kernel_split_inputs(z=3.0)
    full = _kernel_split(sed, wave_rest, wave_obs, 3.0, window_z=None)
    wrong = _kernel_split(sed, wave_rest, wave_obs, 3.0, window_z=1.0)
    assert np.max(np.abs(wrong - full)) / np.max(np.abs(full)) > 1e-8


def test_window_z_observation_predict_on_fixed_z_igm_model(synthetic_ssp_wide, monkeypatch):
    """Observation.predict(window_z=z) on a Fixed-z SEDModel with IGM: crop equals full."""
    import warnings

    from tengri.parameters.resolve import merge_fixed_params

    z = 1.0
    wave_obs = jnp.linspace(2000.0, 3200.0, 300)
    spec = Spectroscopy(wave_obs=wave_obs, resolution=3000.0, sigma_lib_kms=70.0)
    obs = Observation(spectroscopy=spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
            neb={"type": "none"},
            igm={"type": "inoue14"},
            redshift=Fixed(z),
        )
        merged = base.spec.merge_observation_params(sigma_v_kms=Uniform(0.0, 2000.0))
        model = SEDModel(merged, synthetic_ssp_wide, observation=obs)
    p = dict(model.spec.sample(jax.random.PRNGKey(11)))
    p["sfh_dpl_log_total_mass"] = jnp.asarray(10.0)
    p["dust_tau_bc"] = jnp.asarray(0.0)
    p["dust_tau_diff"] = jnp.asarray(0.0)
    p["sigma_v_kms"] = jnp.asarray(300.0)

    state = model.predict_state(p)
    full_params = merge_fixed_params(model.spec, p)
    kwargs = model._observation_predict_kwargs(p)
    calls = []
    real = spectrum_mod.broaden_velocity_only_window

    def spy(*args, **kw):
        calls.append(args[4:6])
        return real(*args, **kw)

    monkeypatch.setattr(spectrum_mod, "broaden_velocity_only_window", spy)
    base_out = model.observation.predict(state, full_params, wave_obs=wave_obs, **kwargs)
    assert calls == []
    crop_out = model.observation.predict(
        state, full_params, wave_obs=wave_obs, window_z=z, **kwargs
    )
    assert len(calls) == 1, "the cropped rest-grid pass did not run"
    a = np.asarray(base_out["spec_fnu"])
    b = np.asarray(crop_out["spec_fnu"])
    assert np.max(np.abs(a)) > 0.0
    err = np.max(np.abs(b - a)) / np.max(np.abs(a))
    assert err <= 1e-10, err


# ── 7. The public predict_spectrum path crops for a Fixed redshift (#2832) ──


def _igm_model_for_public_path(ssp, wave_obs, redshift):
    """A photometry-and-spectroscopy IGM model with the given redshift spec."""
    import warnings

    spec = Spectroscopy(wave_obs=wave_obs, resolution=3000.0, sigma_lib_kms=70.0)
    obs = Observation(spectroscopy=spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
            neb={"type": "none"},
            igm={"type": "inoue14"},
            redshift=redshift,
        )
        merged = base.spec.merge_observation_params(sigma_v_kms=Uniform(0.0, 2000.0))
        return SEDModel(merged, ssp, observation=obs)


def _public_params(model, sigma_v):
    p = dict(model.spec.sample(jax.random.PRNGKey(5)))
    p["sfh_dpl_log_total_mass"] = jnp.asarray(10.0)
    p["dust_tau_bc"] = jnp.asarray(0.0)
    p["dust_tau_diff"] = jnp.asarray(0.0)
    p["sigma_v_kms"] = jnp.asarray(sigma_v)
    return p


def test_fixed_redshift_public_predict_spectrum_crops_and_matches_full_grid(
    synthetic_ssp_wide, monkeypatch
):
    """Fixed(z) + IGM: model.predict_spectrum(p) takes the cropped pass; equals the full grid."""
    z = 0.73
    wave_obs = jnp.linspace(2600.0, 4200.0, 240)
    model = _igm_model_for_public_path(synthetic_ssp_wide, wave_obs, Fixed(z))
    assert model.z_fixed == pytest.approx(z)
    p = _public_params(model, 300.0)

    calls = []
    real = spectrum_mod.broaden_velocity_only_window

    def spy(*args, **kwargs):
        calls.append(args[4:6])
        return real(*args, **kwargs)

    monkeypatch.setattr(spectrum_mod, "broaden_velocity_only_window", spy)
    public = np.asarray(model.predict_spectrum(p))
    assert len(calls) >= 1, "the public predict_spectrum path did not take the cropped pass"
    lo, hi = calls[0]
    assert (lo, hi) != (0, synthetic_ssp_wide.ssp_wave.shape[0]), "the crop removed nothing"
    # The explicit wave_obs path passes no window: it is the full-grid reference.
    full = np.asarray(model.predict_spectrum(p, wave_obs=wave_obs))
    assert np.max(np.abs(public)) > 0.0
    err = np.max(np.abs(public - full)) / np.max(np.abs(public))
    assert err <= 1e-10, err


def test_free_redshift_public_predict_spectrum_never_crops(synthetic_ssp_wide, monkeypatch):
    """A free redshift has no single value, so the public path keeps the full grid."""
    wave_obs = jnp.linspace(2600.0, 4200.0, 240)
    model = _igm_model_for_public_path(synthetic_ssp_wide, wave_obs, Uniform(0.6, 0.9))
    assert model.z_fixed is None
    p = _public_params(model, 300.0)
    p["redshift"] = jnp.asarray(0.75)

    calls = []
    real = spectrum_mod.broaden_velocity_only_window

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(spectrum_mod, "broaden_velocity_only_window", spy)
    out = np.asarray(model.predict_spectrum(p))
    assert np.all(np.isfinite(out))
    assert calls == []


# ── 8. Padding from the kernel's own sigma and pixel scale (#2832) ──


def _log_grid_and_spectrum(n: int = 2000):
    wave = np.geomspace(3600.0, 9824.0, n)
    x = np.arange(n, dtype=np.float64)
    spec = 1.0e3 * np.exp(-3.0 * x / n) + 300.0 * np.exp(-0.5 * ((x - 10) / 1.5) ** 2)
    spec += 200.0 * (x > n // 2)
    return wave, spec


def _rel_err(out, ref):
    return float(np.max(np.abs(np.asarray(out) - ref)) / np.max(np.abs(ref)))


@pytest.mark.parametrize("sigma_pix", [3.0, 5.0, 8.0])
def test_three_sites_match_independent_real_space_reference(sigma_pix):
    """All three FFT sites agree with a real-space sampled Gaussian to 1e-10 (sigma_pix >= 3)."""
    wave, spec = _log_grid_and_spectrum()
    dln = np.log(wave[1] / wave[0])
    sigma_kms = sigma_pix * dln * _C_KM_S
    resolution = _C_KM_S / (spectrum_mod._FWHM_TO_SIGMA * sigma_kms)
    ref = _ref_real_space(spec, sigma_pix)
    out_v = velocity_broaden(jnp.asarray(spec), jnp.asarray(wave), sigma_kms)
    assert _rel_err(out_v, ref) <= 1e-10
    out_c = apply_lsf(jnp.asarray(spec), jnp.asarray(wave), float(resolution))
    assert _rel_err(out_c, ref) <= 1e-10
    out_b = apply_lsf(jnp.asarray(spec), jnp.asarray(wave), np.full(wave.size, resolution))
    ref_b = _ref_real_space_variable(spec, wave, np.full(wave.size, sigma_kms))
    assert _rel_err(out_b, ref_b) <= 1e-10


@pytest.mark.parametrize("sigma_kms", [3000.0, 5000.0, 10000.0])
def test_wide_kernels_are_not_truncated_by_a_fixed_bound(sigma_kms):
    """A velocity sigma beyond the old 2000 km/s bound is still exact (was 3e-4 to 0.12)."""
    wave, spec = _log_grid_and_spectrum()
    sigma_pix = sigma_kms / _C_KM_S / np.log(wave[1] / wave[0])
    out = velocity_broaden(jnp.asarray(spec), jnp.asarray(wave), sigma_kms)
    assert _rel_err(out, _ref_real_space(spec, sigma_pix)) <= 1e-6


def test_apply_lsf_low_resolution_is_not_truncated():
    """R = 30 (sigma about 4244 km/s) was 7e-3 off under the fixed 2000 km/s bound."""
    wave, spec = _log_grid_and_spectrum()
    dln = np.log(wave[1] / wave[0])
    resolution = 30.0
    sigma_kms = _C_KM_S / (spectrum_mod._FWHM_TO_SIGMA * resolution)
    out = apply_lsf(jnp.asarray(spec), jnp.asarray(wave), resolution)
    assert _rel_err(out, _ref_real_space(spec, sigma_kms / _C_KM_S / dln)) <= 1e-6


def test_sigma_bound_is_exact_when_concrete_and_maximal_when_traced():
    concrete = spectrum_mod._lsf_sigma_bound_kms(30.0, None, None)
    assert concrete == pytest.approx(_C_KM_S / (spectrum_mod._FWHM_TO_SIGMA * 30.0))
    # sigma_lib only subtracts, so a concrete sigma_lib gives the exact quadrature.
    # R = 300 gives sigma_inst = 424 km/s, above sigma_lib = 70 km/s. At R = 3000
    # (sigma_inst = 42) the deficit clamps to zero, which the next check covers.
    lib = spectrum_mod._lsf_sigma_bound_kms(300.0, 70.0, 0.0)
    inst = _C_KM_S / (spectrum_mod._FWHM_TO_SIGMA * 300.0)
    assert lib == pytest.approx(np.sqrt(inst**2 - 70.0**2))
    assert spectrum_mod._lsf_sigma_bound_kms(3000.0, 70.0, 0.0) == 0.0
    assert spectrum_mod._lsf_sigma_bound_kms(None, None, 0.0) == 0.0
    seen = []

    def probe(s):
        seen.append(spectrum_mod._lsf_sigma_bound_kms(None, None, s))
        return s

    jax.make_jaxpr(probe)(jnp.asarray(60.0))
    assert seen == [spectrum_mod._LSF_MAX_SIGMA_KMS]


@pytest.mark.parametrize("resolution", [10.0, 30.0, 100.0, 300.0])
def test_traced_grid_is_exact(resolution):
    """A traced grid pads by n - 1, which is exact: no ringing or truncation is lost."""
    wave, spec = _log_grid_and_spectrum()
    sigma_kms = _C_KM_S / (spectrum_mod._FWHM_TO_SIGMA * resolution)
    sigma_pix = sigma_kms / _C_KM_S / np.log(wave[1] / wave[0])
    ref = _ref_real_space(spec, sigma_pix)
    spec_j = jnp.asarray(spec)
    wave_j = jnp.asarray(wave)

    traced = jax.jit(lambda w: apply_lsf(spec_j, w, resolution))(wave_j)
    assert _rel_err(traced, ref) <= 1e-9
    # The explicit override with the same width is the same computation.
    override = apply_lsf(spec_j, wave_j, resolution, pad_pixels=wave.size - 1)
    assert _rel_err(override, np.asarray(traced)) <= 1e-12


def test_pad_pixels_override_is_taken_literally():
    wave, spec = _log_grid_and_spectrum()
    resolution = 300.0
    sigma_kms = _C_KM_S / (spectrum_mod._FWHM_TO_SIGMA * resolution)
    sigma_pix = sigma_kms / _C_KM_S / np.log(wave[1] / wave[0])
    ref = _ref_real_space(spec, sigma_pix)
    narrow = apply_lsf(jnp.asarray(spec), jnp.asarray(wave), resolution, pad_pixels=8)
    assert _rel_err(narrow, ref) > 1e-5, "an 8-pixel override must be used as given"


def test_project_spectrum_threads_lsf_pad_pixels():
    wave, spec = _log_grid_and_spectrum()
    obs = jnp.asarray(wave)
    rest = jnp.asarray(wave)
    default = project_spectrum(jnp.asarray(spec), rest, obs, 0.0, 1.0, resolution=300.0)
    same = project_spectrum(
        jnp.asarray(spec), rest, obs, 0.0, 1.0, resolution=300.0, lsf_pad_pixels=wave.size - 1
    )
    assert _rel_err(same, np.asarray(default)) <= 1e-12
    narrow = project_spectrum(
        jnp.asarray(spec), rest, obs, 0.0, 1.0, resolution=300.0, lsf_pad_pixels=8
    )
    assert _rel_err(narrow, np.asarray(default)) > 1e-5


def test_near_duplicate_pixel_keeps_padding_and_result():
    """One near-duplicate pixel must not shrink the variable path's speedup (#2832)."""
    n = 7909
    clean = np.linspace(3600.0, 9824.0, n)
    dup = clean.copy()
    dup[3001] = dup[3000] + 1e-3
    assert np.all(np.diff(dup) > 0)
    sigma = 2000.0
    h_clean = spectrum_mod._lsf_pad_pixels(clean, n, sigma, 16)
    h_dup = spectrum_mod._lsf_pad_pixels(dup, n, sigma, 16)
    assert h_dup == h_clean
    assert spectrum_mod._variable_r_layout(n, 16, h_dup, None).windowed
    spec = _spectrum_with_edge_features(n)
    resolution = np.full(n, _R_SCALAR)
    out = apply_lsf(jnp.asarray(spec), jnp.asarray(dup), jnp.asarray(resolution))
    ref = _ref_variable_conv(spec, dup, _sigma_inst_kms(resolution))
    assert _rel_err(out, ref) <= 1e-7


def test_variable_padding_uses_the_smallest_bin_mean_pixel_scale():
    n = 7909
    wave = np.linspace(3600.0, 9824.0, n)
    dln = np.gradient(np.log(wave))
    width = n / 16
    means = []
    for k in range(16):
        c = (k + 0.5) * width
        mask = np.abs(np.arange(n) - c) < width
        means.append((dln * mask).sum() / mask.sum())
    expected = int(np.ceil(5.0 * 2000.0 / (_C_KM_S * min(means))))
    assert spectrum_mod._lsf_pad_pixels(wave, n, 2000.0, 16) == expected


@pytest.mark.parametrize(
    ("sigma_pix", "n_bins", "tol"),
    [
        (3.0, 16, 1e-10),
        (3.0, 64, 1e-10),
        (5.0, 64, 1e-10),
        (1.1, 16, 1e-6),
        (1.1, 64, 1e-6),
        (0.3, 16, 5e-6),
        (0.3, 64, 5e-6),
        (0.14, 64, 5e-6),
    ],
)
def test_windowed_and_full_layouts_agree_on_a_rough_spectrum(sigma_pix, n_bins, tol, monkeypatch):
    """Rough spectrum (white noise plus lines): windowed vs full-length transforms.

    Both use the same padding, so the gap is the Fourier kernel's ringing tail beyond
    that padding. It is below 1e-10 for sigma_pix >= 3 (measured 2e-12 at 3). Below 3
    it is 1.3e-7 to 1.2e-6 (measured at 0.14, 0.3, 1.1 and 64 bins), so the bounds
    are 1e-6 for 1.1 and 5e-6 for the sub-pixel cases.
    """
    n = 7909
    wave = np.linspace(3600.0, 9824.0, n)
    rng = np.random.default_rng(1)
    x = np.arange(n, dtype=np.float64)
    rough = rng.normal(size=n) * 5.0 + 1.0 + 3.0 * np.exp(-3.0 * x / n)
    rough += 200.0 * np.exp(-0.5 * ((x - 3000) / 1.5) ** 2)
    dln = np.mean(np.diff(np.log(wave)))
    sigma_kms = sigma_pix * dln * _C_KM_S
    sigma = jnp.full(n, sigma_kms)
    pad = spectrum_mod._lsf_pad_pixels(wave, n, sigma_kms, n_bins)
    fn = spectrum_mod._apply_lsf_variable_r.__wrapped__
    windowed = np.asarray(fn(jnp.asarray(rough), jnp.asarray(wave), sigma, n_bins, pad=pad))
    orig = spectrum_mod._variable_r_layout

    def full_only(n_pix, nb, pad_, ids):
        layout = orig(n_pix, nb, pad_, ids)
        if not layout.windowed:
            return layout
        seg_full = spectrum_mod._next_fast_fft_len(n_pix + 2 * pad_)
        return spectrum_mod._BinLayout(
            False, layout.centers, layout.half_w, layout.bin_width, None, n_pix, seg_full, pad_
        )

    monkeypatch.setattr(spectrum_mod, "_variable_r_layout", full_only)
    full = np.asarray(fn(jnp.asarray(rough), jnp.asarray(wave), sigma, n_bins, pad=pad))
    assert _rel_err(windowed, full) <= tol


def test_ringing_floor_matches_its_measurement():
    """The floor is the padding at which sub-pixel kernels reach 1e-7 (measured 1.5e-7).

    The reference is the Fourier-kernel convolution: the sampled Gaussian used for the
    sigma_pix >= 3 checks is not the same kernel at sigma_pix below 1.
    """
    wave, spec = _log_grid_and_spectrum()
    dln = np.log(wave[1] / wave[0])
    for sigma_kms in (42.4, 100.0):
        sigma_pix = sigma_kms / _C_KM_S / dln
        ref = _ref_gaussian_conv(spec, sigma_pix)
        out = spectrum_mod._gaussian_fft_convolve(
            jnp.asarray(spec), sigma_pix, spectrum_mod._LSF_RINGING_FLOOR_PIXELS
        )
        assert _rel_err(out, ref) <= 2e-7

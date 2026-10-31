# SPDX-License-Identifier: BSD-3-Clause
"""Per-segment calibration scales and the scale-free coefficient prior (#3134).

The calibration multiplier of a multi-camera spectrum is one free scale per camera
segment plus a shared Chebyshev tilt. The marginal over those columns goes through
:func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`, the same
primitive as every other linear nuisance.

Runs on synthetic toy spectra with known truth (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.calibration import (
    calibration_basis,
    marginalize_calibration_segments,
)

pytestmark = pytest.mark.contract

_N_PER_SEG = 120


def _toy_grid(n_seg: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a wavelength grid and per-pixel segment ids for ``n_seg`` equal segments."""
    wave = np.linspace(4000.0, 7000.0, n_seg * _N_PER_SEG)
    segments = np.repeat(np.arange(n_seg), _N_PER_SEG)
    return wave, segments


def _toy_model(wave: np.ndarray) -> np.ndarray:
    """Return a smooth positive model spectrum of order unity."""
    return 1.0 + 0.3 * np.sin(wave / 300.0)


def test_two_segments_equal_independent_camera_fits_and_recover_scales():
    wave, segments = _toy_grid(2)
    model = _toy_model(wave)
    true_scales = np.array([1.15, 0.87])
    sigma = np.full_like(wave, 1e-3)
    data = true_scales[segments] * model

    joint = marginalize_calibration_segments(
        jnp.asarray(model),
        jnp.asarray(data),
        jnp.asarray(sigma),
        jnp.asarray(wave),
        jnp.asarray(segments),
        n_segments=2,
        n_poly=0,
    )
    per_camera = 0.0
    for j in range(2):
        mask = (segments == j).astype(np.float64)
        single = marginalize_calibration_segments(
            jnp.asarray(model),
            jnp.asarray(data),
            jnp.asarray(sigma),
            jnp.asarray(wave),
            jnp.zeros_like(jnp.asarray(segments)),
            n_segments=1,
            n_poly=0,
            pix_mask=jnp.asarray(mask),
        )
        per_camera += float(single[0])

    assert np.isfinite(float(joint[0]))
    np.testing.assert_allclose(float(joint[0]), per_camera, rtol=1e-10)
    np.testing.assert_allclose(1.0 + np.asarray(joint[1]), true_scales, atol=1e-6)


def test_segment_indicator_columns_are_present_in_basis():
    wave, segments = _toy_grid(3)
    basis = calibration_basis(
        jnp.asarray(wave),
        float(wave.min()),
        float(wave.max()),
        2,
        segments=jnp.asarray(segments),
        n_segments=3,
    )
    assert basis.shape == (wave.size, 5)
    np.testing.assert_array_equal(np.asarray(basis[:, :3]).sum(axis=1), np.ones(wave.size))
    np.testing.assert_array_equal(np.asarray(basis[:, 0]), (segments == 0).astype(float))


def test_scale_free_marginal_invariant_to_model_rescaling_alone():
    wave, segments = _toy_grid(2)
    model = _toy_model(wave)
    sigma = np.full_like(wave, 0.02)
    data = 1.05 * model + 0.01 * np.cos(wave / 150.0)
    n_poly = 2

    def run(lam: float):
        return marginalize_calibration_segments(
            jnp.asarray(lam * model),
            jnp.asarray(data),
            jnp.asarray(sigma),
            jnp.asarray(wave),
            jnp.asarray(segments),
            n_segments=2,
            n_poly=n_poly,
            cal_prior="scale_free",
        )

    base = run(1.0)
    scaled = run(2.0)
    np.testing.assert_allclose(float(scaled[0]), float(base[0]), rtol=1e-10)


def test_scale_free_marginal_under_joint_data_model_rescaling():
    wave, segments = _toy_grid(2)
    model = _toy_model(wave)
    sigma = np.full_like(wave, 0.02)
    data = 1.05 * model + 0.01 * np.cos(wave / 150.0)
    n_seg, n_poly = 2, 1
    n_pix = wave.size

    def run(lam: float):
        return marginalize_calibration_segments(
            jnp.asarray(lam * model),
            jnp.asarray(lam * data),
            jnp.asarray(lam * sigma),
            jnp.asarray(wave),
            jnp.asarray(segments),
            n_segments=n_seg,
            n_poly=n_poly,
            cal_prior="scale_free",
        )

    base = run(1.0)
    scaled = run(2.0)
    expected = (n_seg + n_poly - n_pix) * math.log(2.0)
    np.testing.assert_allclose(float(scaled[0]) - float(base[0]), expected, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(scaled[1]), 2.0 * np.asarray(base[1]), rtol=1e-8)


def test_float32_value_and_gradient_are_finite_and_nonzero():
    wave, segments = _toy_grid(2)
    shape = _toy_model(wave)
    amplitude = 1e-25

    def loss(log_amp):
        amp = amplitude * jnp.exp(log_amp - jnp.log(amplitude))
        model = amp * jnp.asarray(shape, jnp.float32)
        data = amp * jnp.asarray(1.02 * shape + 0.01, jnp.float32)
        sigma = jnp.full(wave.shape, 0.01 * amplitude, jnp.float32)
        return marginalize_calibration_segments(
            model,
            data,
            sigma,
            jnp.asarray(wave, jnp.float32),
            jnp.asarray(segments),
            n_segments=2,
            n_poly=1,
            cal_prior="scale_free",
        )[0]

    with jax.enable_x64(False):
        value, grad = jax.value_and_grad(loss)(jnp.float32(math.log(amplitude)))
        assert np.isfinite(float(value))
        assert np.isfinite(float(grad))
        assert float(grad) != 0.0


# ── Configuration and batch leaves ──────────────────────────────────────────


def test_spectroscopy_accepts_sizes_or_ids_and_validates_layout():
    from tengri import Spectroscopy

    wave = np.linspace(4000.0, 7000.0, 200)
    by_sizes = Spectroscopy(wave_obs=wave, calibration_segments=(120, 80))
    ids = np.repeat([0, 1], [120, 80])
    by_ids = Spectroscopy(wave_obs=wave, calibration_segments=ids)
    np.testing.assert_array_equal(by_sizes.segment_ids, by_ids.segment_ids)
    assert by_sizes.n_cal_segments == 2
    assert Spectroscopy(wave_obs=wave).n_cal_segments == 0

    with pytest.raises(ValueError, match="sum"):
        Spectroscopy(wave_obs=wave, calibration_segments=(120, 70))
    with pytest.raises(ValueError, match="contiguous"):
        Spectroscopy(wave_obs=wave, calibration_segments=np.repeat([0, 2], [120, 80]))
    with pytest.raises(ValueError, match="cal_prior"):
        Spectroscopy(wave_obs=wave, cal_prior="flat")
    with pytest.raises(ValueError, match="leaves the wavelength grid"):
        Spectroscopy(wave_obs=wave, cal_scale_prior=(0.0, 0.3), cal_scale_window=(3000.0, 5000.0))
    with pytest.raises(ValueError, match="needs cal_scale_prior"):
        Spectroscopy(wave_obs=wave, cal_scale_window=(4500.0, 5000.0))


def test_spectroscopy_cache_key_tracks_calibration_options():
    from tengri import Spectroscopy

    wave = np.linspace(4000.0, 7000.0, 200)
    base = Spectroscopy(wave_obs=wave, calibration_segments=(120, 80))
    same = Spectroscopy(wave_obs=wave, calibration_segments=(120, 80))
    other = Spectroscopy(wave_obs=wave, calibration_segments=(100, 100))
    free = Spectroscopy(wave_obs=wave, calibration_segments=(120, 80), cal_prior="scale_free")
    assert base.cache_key() == same.cache_key()
    assert base.cache_key() != other.cache_key()
    assert base.cache_key() != free.cache_key()


def test_batch_carries_segment_and_prior_leaves_and_buckets_by_count():
    from tengri.observation.batched import GalaxySpectrum, build_spectro_batches

    def galaxy(n, ids, prior):
        wave = np.linspace(4000.0, 7000.0, n)
        return GalaxySpectrum(
            wave=wave,
            flux=np.ones(n),
            ivar=np.full(n, 4.0),
            resolution=3000.0,
            z=0.1,
            segment_ids=ids,
            cal_scale_prior=prior,
        )

    a = galaxy(100, np.repeat([0, 1], 50), (0.0, 0.3))
    b = galaxy(100, np.repeat([0, 1, 2], [30, 40, 30]), (0.1, 0.2))
    c = galaxy(100, None, None)
    buckets = build_spectro_batches([a, b, c], quantum=128)
    by_count = {spec.n_segments: (batch, idx) for spec, batch, idx in buckets}
    assert set(by_count) == {0, 2, 3}
    batch2, _ = by_count[2]
    assert batch2.segment.shape == (1, 128)
    np.testing.assert_array_equal(np.asarray(batch2.segment[0, 98:]), np.ones(30, dtype=int))
    np.testing.assert_allclose(np.asarray(batch2.cal_scale_center), [0.0])
    np.testing.assert_allclose(np.asarray(by_count[3][0].cal_scale_sigma), [0.2])
    np.testing.assert_allclose(np.asarray(by_count[0][0].cal_scale_sigma), [1.0])


# ── (a) default bit-compatibility and the one-segment rulings ───────────────


def _cheb_columns(wave: np.ndarray, n_poly: int) -> np.ndarray:
    """Independent Chebyshev T_1..T_n on the grid extremes, via numpy."""
    x = 2.0 * (wave - wave.min()) / (wave.max() - wave.min()) - 1.0
    return np.stack(
        [np.polynomial.chebyshev.chebval(x, [0] * k + [1]) for k in range(1, n_poly + 1)], axis=1
    )


def _toy_data(wave: np.ndarray, model: np.ndarray, sigma: float) -> np.ndarray:
    return 1.07 * model + 0.03 * np.cos(wave / 170.0) + 0.5 * sigma * np.sin(wave / 11.0)


def test_no_segments_form_is_bit_identical_to_the_present_marginal():
    from tengri.observation.calibration import marginalize_calibration

    wave, _ = _toy_grid(2)
    model, sigma = _toy_model(wave), np.full_like(wave, 0.02)
    data = _toy_data(wave, model, 0.02)
    present = marginalize_calibration(
        jnp.asarray(model), jnp.asarray(data), jnp.asarray(sigma), jnp.asarray(wave), n_poly=3
    )
    direct = marginalize_calibration(
        jnp.asarray(model), jnp.asarray(data), jnp.asarray(sigma), jnp.asarray(wave), 3, 1.0
    )
    np.testing.assert_allclose(float(present[0]), float(direct[0]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(present[1]), np.asarray(direct[1]), rtol=1e-12)


def test_one_segment_with_pinned_scale_equals_the_present_marginal():
    from tengri.observation.calibration import marginalize_calibration

    wave, _ = _toy_grid(2)
    model, sigma = _toy_model(wave), np.full_like(wave, 0.02)
    data = _toy_data(wave, model, 0.02)
    pinned = 1e-6
    present = marginalize_calibration(
        jnp.asarray(model),
        jnp.asarray(data),
        jnp.asarray(sigma),
        jnp.asarray(wave),
        n_poly=2,
        prior_sigma=pinned,
    )
    one_seg = marginalize_calibration_segments(
        jnp.asarray(model),
        jnp.asarray(data),
        jnp.asarray(sigma),
        jnp.asarray(wave),
        jnp.zeros(wave.size, dtype=int),
        n_segments=1,
        n_poly=2,
        prior_sigma=pinned,
    )
    np.testing.assert_allclose(float(one_seg[0]), float(present[0]), rtol=1e-6)


def test_one_segment_with_wide_prior_equals_hand_computation_with_the_scale_column():
    from tengri.observation.linear_marginal import marginalize_linear_nuisance

    wave, _ = _toy_grid(2)
    model, sigma = _toy_model(wave), np.full_like(wave, 0.02)
    data = _toy_data(wave, model, 0.02)
    prior_sigma = 2.0
    design = model[:, None] * np.concatenate([np.ones((wave.size, 1)), _cheb_columns(wave, 2)], 1)
    by_hand = marginalize_linear_nuisance(
        jnp.asarray(data),
        jnp.asarray(model),
        jnp.asarray(design),
        sigma=jnp.asarray(sigma),
        prior_precision=jnp.eye(3) / prior_sigma**2,
    )
    one_seg = marginalize_calibration_segments(
        jnp.asarray(model),
        jnp.asarray(data),
        jnp.asarray(sigma),
        jnp.asarray(wave),
        jnp.zeros(wave.size, dtype=int),
        n_segments=1,
        n_poly=2,
        prior_sigma=prior_sigma,
    )
    np.testing.assert_allclose(float(one_seg[0]), float(by_hand.log_like), rtol=1e-10)


# ── (c) scale-free with line columns present ────────────────────────────────


def _line_block(wave: np.ndarray, sigma: np.ndarray, width: float, lam: float):
    from tengri.observation.eline_marginalization import unit_line_prior

    centers = np.array([4500.0, 5500.0, 6500.0])
    profile = np.exp(-0.5 * ((wave[:, None] - centers) / 12.0) ** 2)
    return unit_line_prior(jnp.asarray(profile), width * lam, noise=jnp.asarray(lam * sigma))


def test_scale_free_with_lines_invariant_to_model_rescale_and_counts_only_calibration():
    wave, segments = _toy_grid(2)
    model, sigma = _toy_model(wave), np.full_like(wave, 0.02)
    data = _toy_data(wave, model, 0.02)
    n_cal, n_pix = 2 + 1, wave.size

    def run(lam_model: float, lam_all: float, with_lines: bool = True):
        lines = _line_block(wave, sigma, 0.5, lam_all) if with_lines else None
        return marginalize_calibration_segments(
            jnp.asarray(lam_all * lam_model * model),
            jnp.asarray(lam_all * data),
            jnp.asarray(lam_all * sigma),
            jnp.asarray(wave),
            jnp.asarray(segments),
            n_segments=2,
            n_poly=1,
            cal_prior="scale_free",
            lines=lines,
        )[0]

    base = float(run(1.0, 1.0))
    np.testing.assert_allclose(float(run(2.0, 1.0)), base, rtol=1e-10)
    np.testing.assert_allclose(
        float(run(1.0, 2.0)) - base, (n_cal - n_pix) * math.log(2.0), rtol=1e-8
    )


# ── (d) the fiber-fraction latent against direct quadrature ─────────────────


def _q_reference(sigma_p: float, center: float, sigma_dex: float, window: np.ndarray):
    """Gaussian-prior marginal, its q posterior, and the grid in ln q (independent path)."""
    from tengri.observation.linear_marginal import marginalize_linear_nuisance

    wave, segments = _toy_grid(2)
    model, sigma = _toy_model(wave), np.full_like(wave, 0.05)
    data = _toy_data(wave, model, 0.05)
    basis = np.concatenate([np.eye(2)[segments], _cheb_columns(wave, 1)], axis=1)
    full = marginalize_linear_nuisance(
        jnp.asarray(data),
        jnp.asarray(model),
        jnp.asarray(model[:, None] * basis),
        sigma=jnp.asarray(sigma),
        prior_precision=jnp.eye(3) / sigma_p**2,
    )
    v = (window @ basis) / window.sum()
    q_mean = 1.0 + float(v @ np.asarray(full.coef_hat))
    q_var = float(v @ np.asarray(full.coef_cov) @ v)
    ln_c, ln_s = math.log(10.0) * center, math.log(10.0) * sigma_dex
    grid = math.log(q_mean) + np.linspace(-8, 8, 4001) * math.sqrt(q_var) / q_mean
    q = np.exp(grid)
    p_g = np.exp(-0.5 * (q - q_mean) ** 2 / q_var) / math.sqrt(2 * math.pi * q_var)
    ln_prior = np.exp(-0.5 * ((grid - ln_c) / ln_s) ** 2) / (ln_s * math.sqrt(2 * math.pi))
    return wave, segments, model, sigma, data, v, float(full.log_like), grid, p_g * ln_prior * q


def _new_path_log_post(
    sigma_p, center, sigma_dex, window, grid, wave, segments, model, sigma, data
):
    from tengri.observation.calibration import calibration_q_log_prior, mean_multiplier_vector

    basis = calibration_basis(
        jnp.asarray(wave),
        float(wave.min()),
        float(wave.max()),
        1,
        segments=jnp.asarray(segments),
        n_segments=2,
    )
    v = mean_multiplier_vector(basis, jnp.ones(wave.size), jnp.asarray(window))
    ln_c, ln_s = math.log(10.0) * center, math.log(10.0) * sigma_dex
    values = []
    for lq in grid:
        like = marginalize_calibration_segments(
            jnp.asarray(model),
            jnp.asarray(data),
            jnp.asarray(sigma),
            jnp.asarray(wave),
            jnp.asarray(segments),
            n_segments=2,
            n_poly=1,
            prior_sigma=sigma_p,
            log_q=jnp.asarray(lq),
            q_window=jnp.asarray(window),
        )[0]
        prior = calibration_q_log_prior(jnp.asarray(lq), v, prior_sigma=sigma_p)
        log_norm = -0.5 * ((lq - ln_c) / ln_s) ** 2 - math.log(ln_s * math.sqrt(2 * math.pi))
        values.append(float(like + prior) + log_norm)
    return np.asarray(values)


def test_log_q_posterior_equals_direct_quadrature_over_q():
    sigma_p, center, sigma_dex = 1.5, 0.02, 0.3
    wave = _toy_grid(2)[0]
    window = ((wave > 4500.0) & (wave < 5600.0)).astype(float)
    wave, segments, model, sigma, data, _v, z_g, grid, reference = _q_reference(
        sigma_p, center, sigma_dex, window
    )
    log_post = _new_path_log_post(
        sigma_p, center, sigma_dex, window, grid, wave, segments, model, sigma, data
    )
    unnormalized = np.exp(log_post - z_g)
    ref_integral = np.trapezoid(reference, grid)
    np.testing.assert_allclose(np.trapezoid(unnormalized, grid), ref_integral, rtol=1e-6)
    np.testing.assert_allclose(
        unnormalized / np.trapezoid(unnormalized, grid),
        reference / np.trapezoid(reference, grid),
        rtol=1e-6,
        atol=1e-12,
    )

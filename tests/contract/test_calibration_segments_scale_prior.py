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

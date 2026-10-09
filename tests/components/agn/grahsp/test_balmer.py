# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the GRAHSP Balmer continuum (Grandi 1982)."""

from __future__ import annotations

from pathlib import Path

import chex
import numpy as np
import pytest

from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.bounds

FIXTURE = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "grahsp" / "balmer.npz"


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURE)


def test_fixture_shapes(fixture):
    """Check that the fixture has the expected structure."""
    assert fixture["balmer_spectra"].shape[0] == 4
    assert fixture["balmer_spectra"].shape[1] == fixture["wave_nm"].size


def test_balmer_matches_upstream(fixture):
    """Compare JAX implementation against upstream reference."""
    from tengri.components.agn.grahsp.balmer import balmer_continuum

    wave_nm = fixture["wave_nm"]
    params = fixture["params"]
    expected = fixture["balmer_spectra"]

    for i, p in enumerate(params):
        out = np.asarray(
            balmer_continuum(
                wave_nm=wave_nm,
                l5100=float(p["lum5100A"]),
                a_bc=float(p["ABC"]),
                linewidth_kms=float(p["linewidth_kms"]),
            )
        )
        # Same wave grid as upstream — should match to numerical precision.
        # atol: the Gaussian tail underflows toward 1e-275, where scipy and JAX erfc differ.
        np.testing.assert_allclose(
            out,
            expected[i],
            rtol=1e-9,
            atol=1e-12 * float(np.max(expected[i])),
            err_msg=f"case {i}",
        )


def test_balmer_finite(fixture):
    """Balmer continuum must be finite everywhere."""
    from tengri.components.agn.grahsp.balmer import balmer_continuum

    wave_nm = fixture["wave_nm"]
    params = fixture["params"]

    for i, p in enumerate(params):
        out = np.asarray(
            balmer_continuum(
                wave_nm=wave_nm,
                l5100=float(p["lum5100A"]),
                a_bc=float(p["ABC"]),
                linewidth_kms=float(p["linewidth_kms"]),
            )
        )
        assert np.all(np.isfinite(out)), f"Non-finite values in case {i}"


def test_balmer_zero_abc_zero(fixture):
    """With ABC=0, the BC should be zero everywhere."""
    from tengri.components.agn.grahsp.balmer import balmer_continuum

    wave_nm = fixture["wave_nm"]
    out = np.asarray(
        balmer_continuum(
            wave_nm=wave_nm,
            l5100=1.0e36,
            a_bc=0.0,  # Disabled
            linewidth_kms=5000.0,
        )
    )
    np.testing.assert_allclose(out, 0.0, atol=0.0, rtol=0.0)


def test_jit_compatible(fixture):
    """Balmer continuum must be JIT-compilable."""
    import jax.numpy as jnp

    from tengri.components.agn.grahsp.balmer import balmer_continuum

    out = assert_jit_matches_eager(
        balmer_continuum,
        wave_nm=jnp.linspace(200.0, 400.0, 100),
        l5100=1.0e36,
        a_bc=0.5,
        linewidth_kms=5000.0,
    )
    chex.assert_shape(out, (100,))
    chex.assert_tree_all_finite(out)


def _bc_truncation_and_weight():
    """Blackbody weight and truncation normalization of the BC shape (module constants)."""
    from tengri.components.agn.grahsp import balmer as m

    def weight(wave_nm):
        bb = wave_nm ** (-5.0) / np.expm1(m._H_C_PER_K_B_NM_K / (m._BC_TEMPERATURE_K * wave_nm))
        bb_edge = m._BALMER_EDGE_NM ** (-5.0) / np.expm1(
            m._H_C_PER_K_B_NM_K / (m._BC_TEMPERATURE_K * m._BALMER_EDGE_NM)
        )
        return bb / bb_edge

    return weight, -np.expm1(-m._BC_TAU), m._BALMER_EDGE_NM


@pytest.mark.parametrize("linewidth_kms", [500.0, 2000.0])
def test_smoothing_conserves_the_truncation_energy(linewidth_kms):
    """Smoothing the edge conserves the (blackbody-free) truncation energy to rtol 1e-4.

    With the blackbody factor divided out, the BC is a linear-approximated truncation
    ``(alpha x + beta)(1 - exp(-1))`` for ``x = lambda/364.6 nm < 1`` convolved with a
    Gaussian, so its integral over ``x >= 250/364.6`` equals the sharp one, tail included.
    A Gaussian dropped or mis-widened (FWHM as sigma) breaks this.
    """
    from tengri.components.agn.grahsp.balmer import balmer_continuum

    weight, trunc_edge, edge = _bc_truncation_and_weight()
    wave = np.linspace(250.0, 520.0, 400001)
    out = np.asarray(
        balmer_continuum(wave_nm=wave, l5100=510.0, a_bc=1.0, linewidth_kms=linewidth_kms)
    )
    shape = out * trunc_edge / weight(wave)  # (alpha x + beta)(1 - e^-1) convolved
    smoothed = float(np.sum(0.5 * (shape[1:] + shape[:-1]) * np.diff(wave)))
    x_a = 250.0 / edge
    alpha, beta = 1.8, -0.8
    sharp = edge * (1.0 - np.exp(-1.0)) * (alpha * (1.0 - x_a**2) / 2.0 + beta * (1.0 - x_a))
    np.testing.assert_allclose(smoothed, sharp, rtol=1e-4)


def test_smoothing_keeps_a_tail_above_the_edge_of_gaussian_width():
    """Just above 364.6 nm the smoothed BC is nonzero; one FWHM above it is negligible.

    At the edge itself the smoothed truncation is about half of the sharp value.
    """
    from tengri.components.agn.grahsp.balmer import balmer_continuum

    weight, trunc_edge, edge = _bc_truncation_and_weight()
    v = 5000.0
    sigma_nm = edge * v / 299792.458 / 2.3548200450309493
    wave = np.array([edge, edge + sigma_nm, edge + 10.0 * sigma_nm])
    out = np.asarray(balmer_continuum(wave_nm=wave, l5100=510.0, a_bc=1.0, linewidth_kms=v))
    assert out[1] > 0.0
    assert out[2] < 1e-12 * out[0]
    # Exact Gaussian-smoothed value at x = 1: (beta/2 + alpha (1/2 - s/sqrt(2 pi))) (1 - e^-1)
    # with s the Gaussian sigma in x = lambda / edge.
    s_x = v / 299792.458 / 2.3548200450309493
    expected = (-0.8 * 0.5 + 1.8 * (0.5 - s_x / np.sqrt(2.0 * np.pi))) * (1.0 - np.exp(-1.0))
    np.testing.assert_allclose(out[0] * trunc_edge / weight(wave[0]), expected, rtol=1e-9)

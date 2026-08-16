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
        np.testing.assert_allclose(out, expected[i], rtol=1e-9, atol=0.0, err_msg=f"case {i}")


def test_balmer_zero_above_edge(fixture):
    """Balmer continuum must be exactly zero above the Balmer edge (364.6 nm)."""
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
        # All output above Balmer edge must be zero.
        above_edge = wave_nm > 364.6
        assert np.all(out[above_edge] == 0.0), f"Non-zero above edge in case {i}"


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

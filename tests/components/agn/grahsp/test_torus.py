# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the GRAHSP torus (cool + hot log-Gaussian + Si feature)."""

from __future__ import annotations

from pathlib import Path

import chex
import numpy as np
import pytest

pytestmark = pytest.mark.bounds

FIXTURE = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "grahsp" / "torus.npz"


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURE)


def test_fixture_shapes(fixture):
    assert fixture["torus_spectra"].shape[0] == 4
    assert fixture["torus_spectra"].shape[1] == fixture["wave_torus_nm"].size


def test_torus_dust_matches_upstream(fixture):
    from tengri.components.agn.grahsp.torus import torus_dust_continuum

    wave_nm = fixture["wave_torus_nm"]
    params = fixture["params"]
    expected = fixture["torus_spectra"]
    for i, p in enumerate(params):
        out = np.asarray(
            torus_dust_continuum(
                wave_nm=wave_nm,
                l5100=float(p["lum5100A"]),
                fcov=float(p["fcov"]),
                cool_lam_um=float(p["COOLlam"]),
                cool_width=float(p["COOLwidth"]),
                hot_lam_um=float(p["HOTlam"]),
                hot_width=float(p["HOTwidth"]),
                hot_fcov=float(p["HOTfcov"]),
            )
        )
        # Same wave grid as upstream — should match to numerical precision.
        np.testing.assert_allclose(out, expected[i], rtol=1e-10, atol=0.0, err_msg=f"case {i}")


def test_si_feature_matches_upstream(fixture):
    from tengri.components.agn.grahsp.torus import si_feature

    wave_si_nm = fixture["wave_si_nm"]
    params = fixture["params"]
    expected = fixture["si_spectra"]
    for i, p in enumerate(params):
        out = np.asarray(
            si_feature(
                wave_nm=wave_si_nm,
                l5100=float(p["lum5100A"]),
                fcov=float(p["fcov"]),
                si=float(p["Si"]),
            )
        )
        np.testing.assert_allclose(out, expected[i], rtol=1e-10, atol=0.0, err_msg=f"case {i}")


def test_torus_normalization_at_12um(fixture):
    """Cool+hot peak at 12 um must equal lambda*L_lambda(12um) = 2.5 * lum5100A * fcov / 12 um."""
    from tengri.components.agn.grahsp.torus import torus_dust_continuum

    wave_nm = fixture["wave_torus_nm"]
    norm_idx = int(np.argmin(np.abs(wave_nm - 12000.0)))
    params = fixture["params"]
    for _i, p in enumerate(params):
        out = np.asarray(
            torus_dust_continuum(
                wave_nm=wave_nm,
                l5100=float(p["lum5100A"]),
                fcov=float(p["fcov"]),
                cool_lam_um=float(p["COOLlam"]),
                cool_width=float(p["COOLwidth"]),
                hot_lam_um=float(p["HOTlam"]),
                hot_width=float(p["HOTwidth"]),
                hot_fcov=float(p["HOTfcov"]),
            )
        )
        # By eq. fcov: lambda*L_lambda(12um) = 2.5 * lum5100A * fcov
        # so L_lambda(12um) = 2.5 * lum5100A * fcov / 12000nm
        expected_lam_Llam = 2.5 * float(p["lum5100A"]) * float(p["fcov"])
        np.testing.assert_allclose(out[norm_idx] * 12000.0, expected_lam_Llam, rtol=1e-10)


def test_jit_compatible():
    import jax
    import jax.numpy as jnp

    from tengri.components.agn.grahsp.torus import (
        si_feature,
        torus_dust_continuum,
    )

    fn1 = jax.jit(torus_dust_continuum)
    out = fn1(jnp.array([1000.0, 12000.0, 50000.0]), 1.0e36, 0.4, 20.0, 0.5, 3.0, 0.5, 1.0)
    chex.assert_shape(out, (3,))
    chex.assert_tree_all_finite(out)
    fn2 = jax.jit(si_feature)
    out2 = fn2(jnp.array([8000.0, 9841.0, 14224.0, 20000.0]), 1.0e36, 0.4, 1.0)
    chex.assert_tree_all_finite(out2)


@pytest.mark.parametrize("width", [0.2, 0.45, 0.65])
def test_log_gaussian_width_is_the_standard_deviation(width):
    """The width W is the standard deviation in dex: L(peak * 10**W) / L(peak) = exp(-1/2).

    Buchner et al. 2024 write L_cool proportional to exp[-(x)^2 / (2 W^2)] with W in dex.
    The hot component is switched off (hot_fcov = 0) so only the cool Gaussian remains.
    """
    import jax.numpy as jnp

    from tengri.components.agn.grahsp.torus import torus_dust_continuum

    lam_um = 17.0
    wave_nm = jnp.asarray(
        [lam_um * 1000.0, lam_um * 1000.0 * 10.0**width, lam_um * 1000.0 * 10.0 ** (-2 * width)]
    )
    out = np.asarray(
        torus_dust_continuum(
            wave_nm=wave_nm,
            l5100=1.0,
            fcov=0.4,
            cool_lam_um=lam_um,
            cool_width=width,
            hot_lam_um=2.0,
            hot_width=0.5,
            hot_fcov=0.0,
        )
    )
    np.testing.assert_allclose(out[1] / out[0], np.exp(-0.5), rtol=1e-12)
    np.testing.assert_allclose(out[2] / out[0], np.exp(-2.0), rtol=1e-12)

# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Mor & Netzer 2012 template-based torus (MN12 + Si feature)."""

from __future__ import annotations

from pathlib import Path

import chex
import numpy as np
import pytest

pytestmark = pytest.mark.bounds

FIXTURE = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "grahsp" / "torus_mn12.npz"


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURE)


def test_fixture_shapes(fixture):
    assert fixture["torus_spectra_native"].shape[0] == 4
    assert fixture["torus_spectra_native"].shape[1] == fixture["wave_mn12_nm"].size
    assert fixture["si_spectra_native"].shape[0] == 4
    assert fixture["si_spectra_native"].shape[1] == fixture["wave_si_nm"].size


def test_torus_mn12_continuum_native_grid(fixture):
    """Test torus_mn12_continuum matches numpy reference on the native grid."""
    from tengri.components.agn.grahsp.torus import torus_mn12_continuum

    wave_nm = fixture["wave_mn12_nm"]
    params = fixture["params"]
    expected = fixture["torus_spectra_native"]

    for i, p in enumerate(params):
        out = np.asarray(
            torus_mn12_continuum(
                wave_nm=wave_nm,
                l5100=float(p["l5100"]),
                fcov=float(p["fcov"]),
                tor_temp=float(p["tor_temp"]),
                tor_cutoff_um=float(p["tor_cutoff_um"]),
                mn12_wave_nm=fixture["wave_mn12_nm"],
                mn12_avg=fixture["mn12_avg"],
                mn12_lo=fixture["mn12_lo"],
                mn12_hi=fixture["mn12_hi"],
            )
        )
        # On native grid, should match to near-machine precision (interpolation path not used)
        np.testing.assert_allclose(out, expected[i], rtol=1e-9, atol=0.0, err_msg=f"case {i}")


def test_torus_mn12_continuum_interpolation(fixture):
    """Test torus_mn12_continuum gives finite results on a finer grid."""
    from tengri.components.agn.grahsp.torus import torus_mn12_continuum

    # Create a finer output grid
    wave_fine = np.linspace(100.0, 300000.0, 5000)
    params = fixture["params"]

    for i, p in enumerate(params):
        out = np.asarray(
            torus_mn12_continuum(
                wave_nm=wave_fine,
                l5100=float(p["l5100"]),
                fcov=float(p["fcov"]),
                tor_temp=float(p["tor_temp"]),
                tor_cutoff_um=float(p["tor_cutoff_um"]),
                mn12_wave_nm=fixture["wave_mn12_nm"],
                mn12_avg=fixture["mn12_avg"],
                mn12_lo=fixture["mn12_lo"],
                mn12_hi=fixture["mn12_hi"],
            )
        )
        # Should be finite and non-negative
        assert np.all(np.isfinite(out)), f"case {i} has non-finite values"
        # Most values should be non-negative (cutoff enforces this)
        assert np.sum(out >= 0.0) > len(out) * 0.99, f"case {i} has unexpected negative values"


def test_torus_mn12_si_native_grid(fixture):
    """Test torus_mn12_si matches numpy reference on the native grid."""
    from tengri.components.agn.grahsp.torus import torus_mn12_si

    wave_nm = fixture["wave_si_nm"]
    params = fixture["params"]
    expected = fixture["si_spectra_native"]

    for i, p in enumerate(params):
        out = np.asarray(
            torus_mn12_si(
                wave_nm=wave_nm,
                l5100=float(p["l5100"]),
                fcov=float(p["fcov"]),
                si=float(p["si"]),
                si_wave_nm=fixture["wave_si_nm"],
                si_lumin=fixture["mn12_si_lumin"],
            )
        )
        # On native grid, should match to near-machine precision
        np.testing.assert_allclose(out, expected[i], rtol=1e-9, atol=0.0, err_msg=f"case {i}")


def test_torus_mn12_si_interpolation(fixture):
    """Test torus_mn12_si gives finite results on a finer grid."""
    from tengri.components.agn.grahsp.torus import torus_mn12_si

    # Create a finer output grid
    wave_fine = np.linspace(100.0, 300000.0, 5000)
    params = fixture["params"]

    for i, p in enumerate(params):
        out = np.asarray(
            torus_mn12_si(
                wave_nm=wave_fine,
                l5100=float(p["l5100"]),
                fcov=float(p["fcov"]),
                si=float(p["si"]),
                si_wave_nm=fixture["wave_si_nm"],
                si_lumin=fixture["mn12_si_lumin"],
            )
        )
        # Should be finite; can be positive or negative (emission vs absorption)
        assert np.all(np.isfinite(out)), f"case {i} has non-finite values"


def test_normalization_at_12um(fixture):
    """Verify normalization at 12 um: L_lambda(12 um) = 2.5*l5100*fcov/12000 nm.

    The cutoff multiplies this value.
    """
    from tengri.components.agn.grahsp.torus import torus_mn12_continuum

    wave_nm = fixture["wave_mn12_nm"]
    params = fixture["params"]

    # Find index closest to 12 µm
    norm_idx = int(np.argmin(np.abs(wave_nm - 12000.0)))

    for _i, p in enumerate(params):
        out = np.asarray(
            torus_mn12_continuum(
                wave_nm=wave_nm,
                l5100=float(p["l5100"]),
                fcov=float(p["fcov"]),
                tor_temp=float(p["tor_temp"]),
                tor_cutoff_um=float(p["tor_cutoff_um"]),
                mn12_wave_nm=fixture["wave_mn12_nm"],
                mn12_avg=fixture["mn12_avg"],
                mn12_lo=fixture["mn12_lo"],
                mn12_hi=fixture["mn12_hi"],
            )
        )
        # lambda*L_lambda(12 um) = 2.5 * l5100 * fcov (Netzer relation); at 12 um the
        # avg/lo/hi templates equal 1, so out(12 um) = l_torus / 12000 nm * cutoff(12 um).
        l_torus = 2.5 * float(p["l5100"]) * float(p["fcov"]) / 12000.0
        cutoff_12 = 1.0 - np.exp(-((wave_nm[norm_idx] / 1000.0 / float(p["tor_cutoff_um"])) ** 2))
        np.testing.assert_allclose(out[norm_idx], l_torus * cutoff_12, rtol=1e-9)


def test_jit_compatible_mn12():
    """Test torus_mn12_continuum and torus_mn12_si are JIT-compilable."""
    import jax
    import jax.numpy as jnp

    from tengri.components.agn.grahsp.templates import load_grahsp_templates
    from tengri.components.agn.grahsp.torus import (
        torus_mn12_continuum,
        torus_mn12_si,
    )

    templates = load_grahsp_templates()

    # Compile torus_mn12_continuum
    fn_cont = jax.jit(torus_mn12_continuum)
    wave_test = jnp.array([1000.0, 12000.0, 50000.0])
    out_cont = fn_cont(
        wave_nm=wave_test,
        l5100=1.0e36,
        fcov=0.4,
        tor_temp=0.5,
        tor_cutoff_um=1.2,
        mn12_wave_nm=templates.torus_mn12_wave_nm,
        mn12_avg=templates.torus_mn12_avg,
        mn12_lo=templates.torus_mn12_lo,
        mn12_hi=templates.torus_mn12_hi,
    )
    chex.assert_shape(out_cont, (3,))
    chex.assert_tree_all_finite(out_cont)

    # Compile torus_mn12_si
    fn_si = jax.jit(torus_mn12_si)
    out_si = fn_si(
        wave_nm=wave_test,
        l5100=1.0e36,
        fcov=0.4,
        si=0.5,
        si_wave_nm=templates.torus_mn12_si_wave_nm,
        si_lumin=templates.torus_mn12_si_lumin,
    )
    chex.assert_tree_all_finite(out_si)


def test_temperature_branch_differentiability():
    """Test that temperature branching via jnp.where is differentiable."""
    import jax
    import jax.numpy as jnp

    from tengri.components.agn.grahsp.templates import load_grahsp_templates
    from tengri.components.agn.grahsp.torus import torus_mn12_continuum

    templates = load_grahsp_templates()
    wave = jnp.array([5000.0, 12000.0, 20000.0])

    def fn(tor_temp):
        return jnp.sum(
            torus_mn12_continuum(
                wave_nm=wave,
                l5100=1.0e36,
                fcov=0.4,
                tor_temp=tor_temp,
                tor_cutoff_um=1.2,
                mn12_wave_nm=templates.torus_mn12_wave_nm,
                mn12_avg=templates.torus_mn12_avg,
                mn12_lo=templates.torus_mn12_lo,
                mn12_hi=templates.torus_mn12_hi,
            )
        )

    # Should be able to take gradient w.r.t. tor_temp
    grad_fn = jax.grad(fn)
    grad_val = grad_fn(0.3)
    assert np.isfinite(grad_val)
    assert np.any(grad_val != 0.0), (
        "`grad_val` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


@pytest.mark.parametrize("tor_temp", [0.0, 0.6, -0.6])
@pytest.mark.parametrize("variant", ["gaussian", "mn12"])
def test_torus_12um_luminosity_is_2p5_fcov_l5100(variant, tor_temp, fixture):
    """lambda*L_lambda(12 um) = 2.5 * fcov * l5100 for both GRAHSP torus variants.

    This is the Netzer (2013) relation quoted in the paper: (lambda L_lambda)(12 um) /
    (lambda L_lambda)(5100 A) = 2.5 f_cov. The two variants must agree on it.
    """
    from tengri.components.agn.grahsp.torus import torus_dust_continuum, torus_mn12_continuum

    l5100, fcov = 1.0e44, 0.4
    wave_nm = np.union1d(np.logspace(2.5, 5.0, 400), [12000.0])
    if variant == "gaussian":
        out = torus_dust_continuum(
            wave_nm=wave_nm,
            l5100=l5100,
            fcov=fcov,
            cool_lam_um=17.0,
            cool_width=0.45,
            hot_lam_um=2.0,
            hot_width=0.5,
            hot_fcov=1.0,
        )
    else:
        out = torus_mn12_continuum(
            wave_nm=wave_nm,
            l5100=l5100,
            fcov=fcov,
            tor_temp=tor_temp,
            tor_cutoff_um=1.2,
            mn12_wave_nm=fixture["wave_mn12_nm"],
            mn12_avg=fixture["mn12_avg"],
            mn12_lo=fixture["mn12_lo"],
            mn12_hi=fixture["mn12_hi"],
        )
    at_12um = float(np.interp(12000.0, wave_nm, np.asarray(out))) * 12000.0
    np.testing.assert_allclose(at_12um, 2.5 * fcov * l5100, rtol=1e-3)

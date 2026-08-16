# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the GRAHSP SMC-like bi-attenuation curve.

Validates against fixtures generated from the upstream
``biattenuation.BiAttenuationLaw.get_attenuation`` formula.
"""

from __future__ import annotations

from pathlib import Path

import chex
import numpy as np
import pytest

from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.bounds

FIXTURE = (
    Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "grahsp" / "biattenuation.npz"
)


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURE)


def test_smc_curve_matches_upstream(fixture):
    from tengri.components.agn.grahsp.attenuation import smc_attenuation_curve

    wave_nm = fixture["wave_nm"]
    expected = fixture["curve_default"]
    out = np.asarray(
        smc_attenuation_curve(
            wave_nm, opt_index=-1.2, nir_index=-3.0, norm=1.2, lam_break_nm=1100.0
        )
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_fawcett_curve_matches_upstream(fixture):
    from tengri.components.agn.grahsp.attenuation import smc_attenuation_curve

    wave_nm = fixture["wave_nm"]
    expected = fixture["curve_fawcett"]
    out = np.asarray(
        smc_attenuation_curve(
            wave_nm, opt_index=-1.0, nir_index=-2.6, norm=1.0, lam_break_nm=1100.0
        )
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_curve_continuity_at_break():
    """At lambda = lam_break, the curve must equal `norm` regardless of indices."""
    from tengri.components.agn.grahsp.attenuation import smc_attenuation_curve

    out = smc_attenuation_curve(
        np.array([1100.0]),
        opt_index=-1.2,
        nir_index=-3.0,
        norm=1.2,
        lam_break_nm=1100.0,
    )
    np.testing.assert_allclose(out, 1.2, rtol=1e-12)


def test_attenuation_factors(fixture):
    from tengri.components.agn.grahsp.attenuation import (
        attenuation_factors,
    )

    wave_nm = fixture["wave_nm"]
    cases = fixture["cases"]
    expected_gal = fixture["factor_gal"]
    expected_agn = fixture["factor_agn"]
    for i, case in enumerate(cases):
        f_gal, f_agn = attenuation_factors(
            wave_nm,
            ebv=float(case["ebv"]),
            ebv_agn=float(case["ebv_agn"]),
            opt_index=-1.2,
            nir_index=-3.0,
            norm=1.2,
            lam_break_nm=1100.0,
        )
        np.testing.assert_allclose(
            np.asarray(f_gal), expected_gal[i], rtol=1e-12, err_msg=f"gal case {i}"
        )
        np.testing.assert_allclose(
            np.asarray(f_agn), expected_agn[i], rtol=1e-12, err_msg=f"agn case {i}"
        )


def test_jit_compatible():
    import jax.numpy as jnp

    from tengri.components.agn.grahsp.attenuation import (
        attenuation_factors,
    )

    f_gal, f_agn = assert_jit_matches_eager(
        attenuation_factors, jnp.array([100.0, 1100.0, 10000.0]), 0.5, 0.3, -1.2, -3.0, 1.2, 1100.0
    )
    chex.assert_shape(f_gal, (3,))
    chex.assert_tree_all_finite(f_gal)
    chex.assert_tree_all_finite(f_agn)

# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2596: graybody closure carries extra (nu/nu_ref)^beta factor.

The graybody opacity factor (1 - exp(-(lambda0/lambda)^beta)) was being multiplied by an
extra optically-thin emissivity factor (nu/nu_ref)^beta, which is correct for
modified_blackbody but not for the general-opacity graybody model (Casey 2012
Eq. 1, CIGALE mbb.py:78-80, Synthesizer Greybody(optically_thin=False)).

Fix: remove the emissivity factor from the graybody closure, so the shape is
opacity * B_nu only, matching Casey (2012), CIGALE, and Synthesizer.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.emission import DUST_EMISSION_MODELS as M

pytestmark = pytest.mark.regression_bug


def _bnu(T, wave):
    """Planck function nu^3 / (exp(h*nu/k*T) - 1)."""
    h = 6.62607015e-27
    k = 1.380649e-16
    c = 2.99792458e10
    nu = c / (wave * 1e-8)
    x = h * nu / (k * T)
    return nu**3 / np.expm1(x)


def _reference_graybody(T, beta, lam0_um, wave):
    """Reference: (1 - exp(-(lambda0/lambda)^beta)) * B_nu, normalized to unit power.

    Casey (2012) Eq. 1, second term (general opacity, no extra nu^beta factor).
    Matches CIGALE mbb.py:78-80 and Synthesizer Greybody(optically_thin=False).
    """
    tau = (lam0_um * 1e4 / wave) ** beta
    bnu_val = _bnu(T, wave)
    s = -np.expm1(-tau) * bnu_val
    c = 2.99792458e10
    nu = c / (wave * 1e-8)
    return s / -np.trapezoid(s, nu)


@pytest.mark.parametrize(
    "T,beta,lam0",
    [
        (50.0, 1.5, 200.0),
        (35.0, 1.6, 200.0),
        (25.0, 2.0, 100.0),
        (20.0, 1.5, 100.0),
        (60.0, 1.2, 300.0),
    ],
)
def test_graybody_equals_reference_formula(T, beta, lam0):
    """graybody matches independent numpy expression (1 - exp(-(lambda0/lambda)^beta)) * B_nu.

    Max relative error < 1e-2 in float64 over 8-1000 um.
    References Casey 2012 Eq. 1, CIGALE mbb.py:78-80, Synthesizer Greybody(optically_thin=False).
    """
    wave = np.logspace(4, 7.3, 4000)  # 1 um .. 2 mm in Angstrom

    # Compute graybody via closure
    tg = np.asarray(
        M["graybody"](
            jnp.asarray(wave),
            1.0,
            dust_T=T,
            dust_beta_ir=beta,
            dust_lambda_0_um=lam0,
            dust_epsilon_mbb=1.0,
        )
    )
    c = 2.99792458e10
    nu = c / (wave * 1e-8)
    tg_norm = tg / -np.trapezoid(tg, nu)

    # Reference formula
    ref = _reference_graybody(T, beta, lam0, wave)

    # Select 8-1000 um
    sel = (wave >= 8e4) & (wave <= 1e7)

    # Relative error
    rel_err = np.abs((tg_norm[sel] - ref[sel]) / (ref[sel] + 1e-30))
    max_rel_err = np.max(rel_err)

    assert max_rel_err < 1e-2, (
        f"Relative error {max_rel_err:.3e} exceeds 1e-2 at T={T}, beta={beta}, lam0={lam0}"
    )


def test_graybody_peak_position():
    """nu*L_nu peak for (50 K, 1.5, 200 um) lies in 71.5-72.5 um."""
    T, beta, lam0 = 50.0, 1.5, 200.0
    wave = np.logspace(4, 7.3, 4000)

    tg = np.asarray(
        M["graybody"](
            jnp.asarray(wave),
            1.0,
            dust_T=T,
            dust_beta_ir=beta,
            dust_lambda_0_um=lam0,
            dust_epsilon_mbb=1.0,
        )
    )
    c = 2.99792458e10
    nu = c / (wave * 1e-8)

    peak_idx = np.argmax(tg * nu)
    peak_um = wave[peak_idx] / 1e4

    assert 71.5 <= peak_um <= 72.5, f"Peak position {peak_um:.1f} um outside 71.5-72.5 um"

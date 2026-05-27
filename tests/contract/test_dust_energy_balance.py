# SPDX-License-Identifier: BSD-3-Clause
"""Issue #437: dust UV-absorbed energy ~= dust FIR re-emission.

Two-component Calzetti attenuation + a Draine & Li 2007 / 2014 /
Dale 2014 / THEMIS dust IR template must conserve energy to better
than a few percent across a tau sweep:

    int L_dust_emission dnu  (8-1000 um)
    ~  int (L_intrinsic - L_attenuated) dnu  (912 A - 3 um)

Issue #437 reported a ~10x energy gap (ratio ~0.1) with the
``dl07`` alias, suggesting the IR normalisation was decoupled
from the UV-absorbed energy. The fix has since landed in the
component chain; this test pins the invariant so we cannot
regress.

A small (~3%) deficit is allowed because (i) trapezoid
integration on a log-spaced wavelength grid undersamples the
peak of the FIR template, and (ii) some absorbed photons go to
emission-line + PAH features outside the 8-1000 um band.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tengri

pytestmark = pytest.mark.contract


def _lnu_integrate(L_nu: np.ndarray, wave: np.ndarray, wmin: float, wmax: float) -> float:
    """Integrate L_nu over [wmin, wmax] in frequency space."""
    c_aa_s = 2.998e18  # speed of light in Angstrom/s
    mask = (wave >= wmin) & (wave <= wmax)
    w = wave[mask]
    L = L_nu[mask]
    nu = c_aa_s / w
    order = np.argsort(nu)
    return float(np.trapezoid(L[order], nu[order]))


@pytest.fixture(scope="module")
def intrinsic_sed():
    try:
        ssp = tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "tsnorm", "*": tengri.FIXED},
        dust={
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 0.0,
            "tau_bc": 0.0,
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    sed = np.asarray(m.predict_rest_sed(p).sed)
    wave = np.asarray(m.ssp_data.ssp_wave)
    return sed, wave, ssp


@pytest.mark.parametrize(
    "emission_type",
    ["draine_li2007", "draine_li2014", "dale2014", "themis"],
)
@pytest.mark.parametrize("tau", [0.3, 1.0])
def test_dust_energy_balance(intrinsic_sed, emission_type, tau):
    """L_emit_FIR must equal L_abs_UV-Opt to within ~5%."""
    sed_intr, wave, ssp = intrinsic_sed
    m = tengri.SEDModel.build(
        ssp,
        sfh={"type": "tsnorm", "*": tengri.FIXED},
        dust={
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": tau,
            "tau_bc": tau,
            "emission": {"type": emission_type, "*": tengri.FIXED},
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    sed_d = np.asarray(m.predict_rest_sed(p).sed)

    L_abs = _lnu_integrate(sed_intr - sed_d, wave, 912.0, 3.0e4)
    L_emit = _lnu_integrate(sed_d, wave, 8.0e4, 1.0e7)
    ratio = L_emit / L_abs
    assert 0.90 < ratio < 1.10, (
        f"Energy balance violated for {emission_type}, tau={tau}: "
        f"L_emit/L_abs = {ratio:.3f} (expected ~1.0)"
    )

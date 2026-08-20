# SPDX-License-Identifier: BSD-3-Clause
"""Energy-balance behavior across galaxy types, not just one fiducial (#1206).

The log-domain reformulation of ``L_absorbed`` changed the arithmetic that
produces ``L_ir`` for *every* galaxy, so the properties that must survive it
are checked across star-forming, bursty, quiescent and constant-SFR histories,
over a range of optical depths, redshifts, masses and metallicities.

These are physics assertions, not parity assertions: a reformulation can
reproduce a frozen reference bit-for-bit on one test galaxy and still be wrong
where that galaxy did not probe (no dust at all, extreme optical depth, the
sign of the boundary correction).
"""

from itertools import pairwise

import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.regression_bug

# Lookback convention: ``start_gyr`` is SF onset (older), ``end_gyr`` cessation
# (younger). start < end silently yields an empty SFH -- keep start > end.
SFHS = {
    "star_forming": {"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0)},
    "bursty": {"type": "delayed", "tau_gyr": Fixed(0.05), "age_gyr": Fixed(0.1)},
    "quiescent": {"type": "delayed", "tau_gyr": Fixed(0.3), "age_gyr": Fixed(12.0)},
    "constant": {"type": "const", "start_gyr": Fixed(8.0), "end_gyr": Fixed(0.5)},
}


def _build(ssp, sfh, tau_bc, tau_diff, *, z=0.0, log_mass=10.0, logzsol=0.0, emission=True):
    """Two-component dust model at a given optical depth and galaxy config."""
    sfh = dict(sfh)
    sfh["log_total_mass"] = Fixed(log_mass)
    sfh["*"] = FIXED
    dust_attenuation = {
        "type": "two_component",
        "law_bc": "calzetti",
        "law_diff": "calzetti",
        "tau_bc": Fixed(tau_bc),
        "tau_diff": Fixed(tau_diff),
        "*": FIXED,
    }
    kwargs = {
        "ssp_data": ssp,
        "met": {"logzsol": Fixed(logzsol), "*": FIXED},
        "sfh": sfh,
        "dust_attenuation": dust_attenuation,
        "redshift": Fixed(z),
    }
    if emission:
        kwargs["dust_emission"] = {"type": "dale2014", "*": FIXED}
    return SEDModel.build(**kwargs)


@pytest.mark.parametrize("name", sorted(SFHS))
def test_no_dust_absorbs_exactly_nothing(synthetic_ssp_wide, name):
    """tau == 0 -> ``L_ir`` exactly 0.0 and ``log_L_ir`` -inf, with no IR emitted.

    This is the branch the log form introduces (``-inf`` powering back to zero);
    a naive ``log10(0.0)`` would give ``-inf`` that propagates as NaN through the
    dust-emission normalization instead.
    """
    state = _build(synthetic_ssp_wide, SFHS[name], 0.0, 0.0).predict_state({})
    assert float(np.asarray(state.derived["L_ir"])) == 0.0
    assert float(np.asarray(state.derived["log_L_ir"])) == -np.inf
    ir = np.asarray(state.derived["sed_dust_ir"])
    assert np.all(ir == 0.0), "dust emitted IR with nothing absorbed"


@pytest.mark.parametrize("name", sorted(SFHS))
def test_absorbed_luminosity_grows_with_optical_depth(synthetic_ssp_wide, name):
    """More dust absorbs more energy — monotone in tau for every SFH."""
    values = [
        float(
            np.asarray(
                _build(synthetic_ssp_wide, SFHS[name], tau, tau).predict_state({}).derived["L_ir"]
            )
        )
        for tau in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)
    ]
    assert all(b >= a for a, b in pairwise(values)), f"non-monotonic: {values}"
    assert values[-1] > values[1] > 0.0, f"no growth with tau: {values}"


@pytest.mark.parametrize("name", sorted(SFHS))
@pytest.mark.parametrize("tau", [0.5, 2.0, 8.0])
def test_absorbed_never_exceeds_intrinsic_bolometric(synthetic_ssp_wide, name, tau):
    """Energy conservation: dust cannot absorb more than the stars emit."""
    free = _build(synthetic_ssp_wide, SFHS[name], 0.0, 0.0, emission=False).predict_state({})
    wave = np.asarray(free.wave, dtype=np.float64)
    sed_intrinsic = np.asarray(free.sed_intrinsic, dtype=np.float64)
    # Same LyC convention as the energy balance itself (#922).
    l_bol = abs(np.trapezoid(np.where(wave >= 912.0, sed_intrinsic, 0.0), C_AA / wave))

    state = _build(synthetic_ssp_wide, SFHS[name], tau, tau, emission=False).predict_state({})
    l_ir = float(np.asarray(state.derived["L_ir"]))
    assert 0.0 <= l_ir <= l_bol * (1.0 + 1e-9), f"L_ir={l_ir:.6e} exceeds L_bol={l_bol:.6e}"


@pytest.mark.parametrize("name", sorted(SFHS))
def test_log_and_linear_forms_agree(synthetic_ssp_wide, name):
    """``log_L_ir`` must equal log10 of the published linear ``L_ir``."""
    for tau in (0.3, 1.5, 6.0):
        for z, log_mass, logzsol in ((0.0, 10.0, 0.0), (2.0, 8.5, -1.0), (5.0, 11.5, 0.3)):
            state = _build(
                synthetic_ssp_wide,
                SFHS[name],
                tau,
                tau * 0.7,
                z=z,
                log_mass=log_mass,
                logzsol=logzsol,
            ).predict_state({})
            linear = float(np.asarray(state.derived["L_ir"]))
            log_form = float(np.asarray(state.derived["log_L_ir"]))
            assert linear > 0.0, f"setup: expected absorption at tau={tau}"
            np.testing.assert_allclose(
                log_form, np.log10(linear), rtol=1e-12, err_msg=f"{name} tau={tau} z={z}"
            )

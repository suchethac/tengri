# SPDX-License-Identifier: BSD-3-Clause
"""The cic gradient of a post-starburst bin edge is continuous through dense-grid knots (#2715).

The dense integrand carries a pair of knots at e(1 -+ 1e-6) for every bin edge. A dense
node (an SSP node or a refined knot) lying inside that bracket split it into a
(SFR_lo, SFR_hi) and a (SFR_hi, SFR_hi) cell, so the mass moved by half the jump per
unit edge shift: the gradient with respect to ``tflex_gyr`` / ``tlast_gyr`` was 28.5 %
off the finite difference at 1.0000 Gyr (an SSP node) and 47 % off at 1.2065 Gyr (a dense
knot) while 0.0035 Gyr away it matched to 1e-8. Nodes inside a bracket now move just
outside it, so one cell holds the whole jump.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri.components.stellar.component as C
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

H = 1e-4
#: Relative AD-vs-FD bound. The dense integrand is piecewise-linear in the edge
#: with a slope change each time a moving knot crosses another node, so an
#: h = 1e-4 difference differs from the exact derivative by up to 4e-6 even
#: 3e-4 Gyr away from any knot (2e-9 typical); at the knots the worst of the
#: 121 / 209 measured is 5.3e-6 / 1.1e-5. The defect was 0.28 and 0.47.
AT_KNOT_TOL = 2e-5
Z = 0.1
BASE = {
    "tlast_gyr": 0.5,
    "tflex_gyr": 2.0,
    "ratio_young": 1.0,
    "ratio_flex_0": 0.8,
    "ratio_flex_1": -0.5,
    "ratio_flex_2": 0.6,
    "ratio_flex_3": -0.3,
    "ratio_old_0": 0.3,
    "ratio_old_1": -0.2,
    "ratio_old_2": 0.2,
}
#: log F_r at three generic edge positions, measured with main's source (the
#: construction changes nothing away from a knot).
LOG_FR_AWAY = {
    "tflex_gyr": ((1.5, 1.77, 2.2), (-59.86834783997114, -59.8749645822331, -59.885871746508094)),
    "tlast_gyr": (
        (0.33, 0.57, 0.81),
        (-59.762773461102135, -59.926139655984436, -60.06583485572039),
    ),
}
SWEEPS = {"tflex_gyr": (1.0, 2.4), "tlast_gyr": (0.2, 0.9)}


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(["galex_fuv", "sdss_u", "sdss_r"]))


def _model(ssp, obs, swept, bounds):
    sfh = {"type": "psb_flex", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)}
    sfh.update({k: Fixed(v) for k, v in BASE.items() if k != swept})
    sfh[swept] = Uniform(*bounds)
    sfh["age_kernel"] = "cic"
    return SEDModel.build(
        ssp_data=ssp, observation=obs, neb={"type": "none"}, redshift=Fixed(Z), sfh=sfh
    )


def _log_fr(model, swept):
    name = f"sfh_psb_flex_{swept}"
    return lambda x: jnp.log(model.predict_photometry({name: x})[2])


def _ad_fd(model, swept, xs):
    f = _log_fr(model, swept)
    ad = jax.jit(jax.vmap(jax.grad(f)))(xs)
    fv = jax.jit(jax.vmap(f))
    fd = (fv(xs + H) - fv(xs - H)) / (2 * H)
    return np.asarray(ad), np.asarray(fd)


def _dense_knots_gyr(ssp, lo, hi):
    ages_yr = jnp.asarray(10.0 ** np.asarray(ssp.ssp_lg_age_gyr) * 1e9)
    grid = np.asarray(C._refine_sfh_table_ages(ages_yr, C.INTEGRAND_FACTOR_PARAMETRIC)) / 1e9
    return grid[(grid >= lo) & (grid <= hi)]


@pytest.mark.parametrize("swept", sorted(SWEEPS))
def test_ad_matches_fd_at_every_dense_knot(ssp_data_fsps, obs, swept):
    lo, hi = SWEEPS[swept]
    knots = _dense_knots_gyr(ssp_data_fsps, lo + 2 * H, hi - 2 * H)
    assert len(knots) >= 50, "the scan must cover many dense knots, enumerated from the grid"
    ad, fd = _ad_fd(_model(ssp_data_fsps, obs, swept, (lo, hi)), swept, jnp.asarray(knots))
    rel = np.abs(ad - fd) / np.abs(fd)
    worst = int(np.argmax(rel))
    assert rel[worst] <= AT_KNOT_TOL, (
        f"{swept}={knots[worst]:.6f} Gyr: AD {ad[worst]:.6e} vs FD {fd[worst]:.6e}, "
        f"rel {rel[worst]:.2e} over {len(knots)} knots"
    )


@pytest.mark.parametrize("x0", [1.0000, 1.2065])
def test_the_reported_tflex_points(ssp_data_fsps, obs, x0):
    """1.0000 Gyr (SSP node) and 1.2065 Gyr (dense knot).

    AD was -1.59e-2 / -1.23e-2 against FD -2.22e-2 / -2.33e-2.
    """
    ad, fd = _ad_fd(
        _model(ssp_data_fsps, obs, "tflex_gyr", (1.0, 2.4)), "tflex_gyr", jnp.asarray([x0])
    )
    assert abs(ad[0] - fd[0]) / abs(fd[0]) <= AT_KNOT_TOL, (x0, ad[0], fd[0])


@pytest.mark.parametrize("swept", sorted(SWEEPS))
def test_values_away_from_knots_are_unchanged(ssp_data_fsps, obs, swept):
    xs, expected = LOG_FR_AWAY[swept]
    f = jax.jit(_log_fr(_model(ssp_data_fsps, obs, swept, SWEEPS[swept]), swept))
    got = np.array([float(f(jnp.asarray(x))) for x in xs])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("swept", sorted(SWEEPS))
def test_value_is_continuous_through_a_knot(ssp_data_fsps, obs, swept):
    lo, hi = SWEEPS[swept]
    knot = float(
        _dense_knots_gyr(ssp_data_fsps, lo + 0.1, hi - 0.1)[
            len(_dense_knots_gyr(ssp_data_fsps, lo + 0.1, hi - 0.1)) // 2
        ]
    )
    f = jax.jit(_log_fr(_model(ssp_data_fsps, obs, swept, (lo, hi)), swept))
    d = 1e-8
    jump = float(f(jnp.asarray(knot + d)) - f(jnp.asarray(knot - d)))
    slope = float(
        jax.grad(_log_fr(_model(ssp_data_fsps, obs, swept, (lo, hi)), swept))(jnp.asarray(knot))
    )
    assert abs(jump - 2 * d * slope) <= 1e-3 * abs(2 * d * slope) + 1e-14

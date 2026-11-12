# SPDX-License-Identifier: BSD-3-Clause
"""WavePrecomp photometry must meet its 3e-4 contract in narrow bands and under dust (#3268).

Measured on main (2026-11-07, issue #3268) the narrow-band error with a live nebular
Lyman-continuum mask (Cue) was 2.3e-3 for ``hsc_nb816`` at z = 0.244, and the
``two_component`` dust case at tau_diff = 4 was larger still. The stars-only model
is exact (1e-14), and the nebular band itself is exact: the nebular LUT equals the
exact filter integral of ``sed_nebular`` to the bit. The error sits in the stellar
LUT. A live Lyman mask routes the stellar SSP photometry through
``lyc_augment_fold_T``, which inserts the Lyman edge (911.76 A x (1+z), below the
filter) as a zero-width node pair. The union grid is trimmed to the filter support,
so the inserted edge sorts in front of the band and the trapezoid draws a spurious
ramp across the gap to the first band node, where the transmission is nonzero for a
narrow band with sharp edges.

Each case takes the worst relative error over six prior draws (``FREE`` SFH), since
the bias depends on the SFH and a single draw can sit under the tolerance by luck.
"""

import jax
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, WavePrecomp

pytestmark = pytest.mark.regression_bug

#: The documented WavePrecomp photometry contract (relative error, worst band, worst draw).
CONTRACT = 3e-4
REDSHIFT = 0.244
N_DRAWS = 6


def _build(ssp, obs, approx, *, dust):
    dust_cfg = (
        {
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(1.0),
            "tau_diff": Fixed(4.0),
            "other_params": Fixed(DEFAULT),
        }
        if dust
        else {"type": "none"}
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        redshift=Fixed(REDSHIFT),
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_cfg,
        approx=approx,
    )


def _worst_relative_error(ssp, obs, *, dust):
    exact = _build(ssp, obs, None, dust=dust)
    lut = _build(ssp, obs, WavePrecomp(), dust=dust)
    worst = 0.0
    for key in range(N_DRAWS):
        params = exact.spec.sample(jax.random.PRNGKey(key))
        a = np.asarray(exact.predict_photometry(params), dtype=np.float64)
        b = np.asarray(lut.predict_photometry(params), dtype=np.float64)
        worst = max(worst, float(np.max(np.abs(b / a - 1.0))))
    return worst


@pytest.fixture(scope="module")
def nb816_obs():
    return Observation(photometry=Photometry.from_names(["hsc_nb816"]))


def test_narrow_band_with_cue_meets_contract(ssp_data_fsps, nb816_obs):
    """hsc_nb816 at z = 0.244 with Cue (nebular Lyman mask live) meets 3e-4.

    Neuter: restore the augmented Lyman edge at the observed edge outside the
    filter's support and this measures ~6e-4 (main), above the contract.
    """
    err = _worst_relative_error(ssp_data_fsps, nb816_obs, dust=False)
    print(f"hsc_nb816 z={REDSHIFT} cue worst relative error: {err:.3e}")
    assert err < CONTRACT


def test_dust_tau_diff_4_with_cue_meets_contract(ssp_data_fsps, nb816_obs):
    """two_component dust at tau_diff = 4 with Cue meets 3e-4 (see xfail note if not)."""
    err = _worst_relative_error(ssp_data_fsps, nb816_obs, dust=True)
    print(f"hsc_nb816 z={REDSHIFT} cue+dust tau_diff=4 worst relative error: {err:.3e}")
    assert err < CONTRACT


# ── Broad bands under dust (the dust-driven half of #3268) ──────────────────────────
#
# Single deterministic points from the issue's worst-point table (z = 0.01, Cue +
# two_component Calzetti, IGM off). Measured (2026-11-12, float64, Mac CPU):
#
#     case              main (origin/main e4454ec12)   this fix
#     galex_nuv         -7.540e-03                     -7.540e-03
#     des_g             -2.474e-03                     -2.474e-03
#     des_r             -3.250e-03                     -3.250e-03
#
# The Lyman-edge fix does not touch these: the error is the dust screen's sub-band
# quadrature over the broad band, which the issue's cause measurements localise to
# blue bands. Strict xfail until that design change lands, so an accidental pass
# is reported.

_BROAD_CASES = [
    # band, z, tau_bc, tau_diff, sfh (tau_gyr, age_frac), logU, logZ, measured on this fix
    pytest.param(
        "galex_nuv",
        0.01,
        2.0,
        4.0,
        (0.1, 0.1),
        -3.5,
        0.2,
        "-7.540e-03",
        id="galex_nuv-young",
    ),
    pytest.param("des_g", 0.01, 2.0, 4.0, (5.0, 0.9), -3.5, 0.2, "-2.474e-03", id="des_g-old"),
    pytest.param("des_r", 0.01, 4.0, 4.0, (0.1, 0.1), -1.0, -1.0, "-3.250e-03", id="des_r-young"),
]


def _build_broad(ssp, band, z, tau_bc, tau_diff, sfh, logU, logZ, approx):
    from tengri.observation.photometry import Photometry as _P

    obs = Observation(photometry=_P.from_names([band]))
    tau_gyr, age_frac = sfh
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "delayed", "tau_gyr": Fixed(tau_gyr), "age_frac": Fixed(age_frac)},
        met={"logzsol": Fixed(logZ)},
        redshift=Fixed(z),
        neb={"type": "cue", "logU": Fixed(logU)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(tau_bc),
            "tau_diff": Fixed(tau_diff),
            "other_params": Fixed(DEFAULT),
        },
        approx=approx,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#3268 dust-driven half, not fixed by the Lyman-edge guard: broad-band "
        "two_component error stays at the main value (see the measurement table "
        "above the test). Remaining: the dust screen's sub-band quadrature design."
    ),
)
@pytest.mark.parametrize("band, z, tau_bc, tau_diff, sfh, logU, logZ, measured", _BROAD_CASES)
def test_broad_band_dust_meets_contract(
    ssp_data_fsps, band, z, tau_bc, tau_diff, sfh, logU, logZ, measured
):
    exact = _build_broad(ssp_data_fsps, band, z, tau_bc, tau_diff, sfh, logU, logZ, None)
    lut = _build_broad(ssp_data_fsps, band, z, tau_bc, tau_diff, sfh, logU, logZ, WavePrecomp())
    params = exact.spec.sample(jax.random.PRNGKey(0))
    a = np.asarray(exact.predict_photometry(params), dtype=np.float64)
    b = np.asarray(lut.predict_photometry(params), dtype=np.float64)
    err = float(np.max(np.abs(b / a - 1.0)))
    print(
        f"{band} z={z} tau_bc={tau_bc} tau_diff={tau_diff} error {err:.3e} (measured {measured})"
    )
    assert err < CONTRACT

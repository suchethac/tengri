# SPDX-License-Identifier: BSD-3-Clause
"""The conserving composable path reproduces the monolithic energy-conserving
AGN models across the FULL SED (not just the peak).

Regression lock for PR #916: the composable presets diverged 0.5-4x from the
monolithic models because the disc was not debited by ``agn_torus_frac``. Under
``agn_norm='conserving'`` the composable runner debits the disc so
``disc(1-f) + torus(f)`` conserves ``L_bol`` — reproducing each monolithic model
to floating-point precision. This test green-lights the Phase-4 retirement of
the monolithic registrations.
"""

import numpy as np
import pytest

import tengri.components.agn.unified as U
from tengri.components.agn.blocks.registry import composable

pytestmark = pytest.mark.regression_bug

_WAVE = np.geomspace(1e2, 1e7, 3000)

_CASES = [
    ("silva04", U.silva04_agn, dict(agn_disc_block="powerlaw", agn_torus_block="silva04")),
    (
        "cat3d_wind",
        U.cat3d_wind_agn,
        dict(agn_disc_block="powerlaw", agn_torus_block="cat3d_wind"),
    ),
    # adaf is exact again (#898): both the monolithic ``adaf_agn`` and the
    # composable disc='adaf' preset now use the faithful Mahadevan 1997
    # ``adaf_spectrum``, so the conserving composable reproduces the monolithic.
    ("adaf", U.adaf_agn, dict(agn_disc_block="adaf", agn_torus_block="silva04")),
]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("name,mono_fn,blocks", _CASES)
def test_conserving_reproduces_monolithic_full_sed(name, mono_fn, blocks):
    # The residual floor is the Stage-4.5 Type-1/2 sigmoid visibility mask the
    # composable path applies and the monolithic models lack: at the default
    # Type-1 viewing (i=30, theta_torus=30) the mask = 0.99999969, so the
    # max peak-relative residual is ~3e-7 — a documented superset (the
    # composable path adds inclination geometry), not a discrepancy. The 1e-6
    # tolerance sits an order of magnitude above that floor. Both agn_lum_ratio axes
    # are covered so the Phase-4 retirement gate holds at partial AGN fractions
    # too (unified.py:581 scales the total by agn_lum_ratio).
    for agn_lum_ratio in (0.5, 1.0):
        for tf in (0.3, 0.5, 0.7):
            mono = np.asarray(mono_fn(_WAVE, 45.0, agn_lum_ratio=agn_lum_ratio, agn_torus_frac=tf))
            comp = np.asarray(
                composable(
                    _WAVE,
                    45.0,
                    agn_norm="conserving",
                    agn_lum_ratio=agn_lum_ratio,
                    agn_torus_frac=tf,
                    **blocks,
                )
            )
            scale = np.max(np.abs(mono))
            np.testing.assert_allclose(
                comp / scale,
                mono / scale,
                atol=1e-6,
                rtol=0,
                err_msg=(
                    f"{name}: conserving != monolithic at "
                    f"agn_lum_ratio={agn_lum_ratio}, torus_frac={tf}"
                ),
            )

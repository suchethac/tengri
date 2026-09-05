# SPDX-License-Identifier: BSD-3-Clause
"""Grammar-level contract for the Feltre NLR block (nlr='feltre').

The block claims a *public-API capability*: selectable through the composable
grammar with drivable photoionization axes. So it must be verified through
``SEDModel.build`` — not just direct ``composable()`` calls, where ``**params``
forwards everything and hides the ``agn_``-prefix routing filter
(``component.py:348``). This test would have caught the original silent-no-op:
the Feltre grid axes were named ``neb_*`` and never reached the runner.
"""

import os

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel

_GRID = os.path.join("data", "feltre_grid.h5")
pytestmark = [
    pytest.mark.contract,
    pytest.mark.skipif(
        not os.path.exists(_GRID), reason="data/feltre_grid.h5 absent (user-built grid)"
    ),
]


def _build_feltre(synthetic_ssp_wide, logU):
    return SEDModel.build(
        synthetic_ssp_wide,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        agn={
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "log_lbol": 13.0,
            "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
            "nlr": {"type": "feltre", "nlr_logU": Fixed(logU), "all_params": Fixed(DEFAULT)},
        },
        redshift=Fixed(0.0),
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_feltre_nlr_lines_appear_and_logU_is_not_a_noop(synthetic_ssp_wide):
    """nlr='feltre' emits a non-zero AGN SED, and its logU grid axis measurably
    changes the prediction *through the public grammar* (not a frozen default)."""
    sed_a = np.asarray(
        _build_feltre(synthetic_ssp_wide, -2.0).predict_state({}).derived["sed_agn"]
    )
    sed_b = np.asarray(
        _build_feltre(synthetic_ssp_wide, -3.5).predict_state({}).derived["sed_agn"]
    )

    assert sed_a.max() > 0.0, "Feltre NLR produced a zero AGN SED"
    assert not np.allclose(sed_a, sed_b), (
        "agn_nlr_logU (nlr='feltre', logU) is a silent no-op through "
        "SEDModel.build — the Feltre grid axis is frozen at its default because "
        "the param does not reach the runner (agn_ prefix / declaration missing)."
    )

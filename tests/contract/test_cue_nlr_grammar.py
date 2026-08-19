# SPDX-License-Identifier: BSD-3-Clause
"""Grammar-level contract for the Cue AGN-ionized NLR block (nlr='cue').

Mirrors ``test_feltre_nlr_grammar.py``. The block advertises a public-API
capability — selectable through the composable grammar with a drivable
photoionization axis — so it must be verified through ``SEDModel.build``, not
just direct ``composable()`` calls where ``**params`` forwards everything and
hides the ``agn_``-prefix routing filter (``component.py:348``). This is the
test the Feltre block's first cut was missing (#930): it proves the grid axis
actually moves the SED and is not a frozen default.
"""

import os

import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel

_WEIGHTS = os.path.join("data", "cue_weights.npz")
pytestmark = [
    pytest.mark.contract,
    pytest.mark.skipif(
        not os.path.exists(_WEIGHTS), reason="data/cue_weights.npz absent (data-gated)"
    ),
]


def _build_cue(ssp, logU):
    return SEDModel.build(
        ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "*": FIXED,
        },
        dust={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "*": FIXED,
        },
        agn={
            "type": "composable",
            "*": FIXED,
            "log_lbol": 13.0,
            "disc": {"type": "multicolor", "*": FIXED},
            "nlr": {"type": "cue", "nlr_logU": Fixed(logU), "*": FIXED},
        },
        redshift=Fixed(0.0),
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cue_nlr_lines_appear_and_logU_is_not_a_noop(synthetic_ssp_wide):
    """nlr='cue' emits a non-zero AGN SED, and its logU axis measurably changes
    the prediction *through the public grammar* (not a frozen default)."""
    sed_a = np.asarray(_build_cue(synthetic_ssp_wide, -2.0).predict_state({}).derived["sed_agn"])
    sed_b = np.asarray(_build_cue(synthetic_ssp_wide, -3.5).predict_state({}).derived["sed_agn"])

    assert sed_a.max() > 0.0, "Cue NLR produced a zero AGN SED"
    assert not np.allclose(sed_a, sed_b), (
        "agn_nlr_logU (nlr='cue', logU) is a silent no-op through SEDModel.build "
        "— the Cue ionization axis is frozen at its default because the param "
        "does not reach the runner (agn_ prefix / declaration missing)."
    )

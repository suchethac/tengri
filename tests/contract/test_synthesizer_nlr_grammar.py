# SPDX-License-Identifier: BSD-3-Clause
"""Grammar-level contract for the Synthesizer NLR/BLR photoionization axes.

The ``nlr='synthesizer'`` / ``blr='synthesizer'`` blocks took ``neb_logU`` /
``neb_logZ_gas`` (galaxy-nebular names). Through ``SEDModel.build`` only
``agn_``-prefixed params reach the AGN runner (``component.py:348``), so those
axes were frozen at their block defaults — a silent no-op (#931). After renaming
to ``agn_nlr_*`` / ``agn_blr_*`` they must move the SED through the grammar.

Grid-gated: skips cleanly where the Synthesizer AGN grids are absent (they are
data-gated and not shipped); runs in grid-equipped environments.
"""

import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel


def _synth_grid_available() -> bool:
    from tengri.components.agn.blocks.nlr import _resolve_synthesizer_grid

    try:
        _resolve_synthesizer_grid("nlr")
        return True
    except (FileNotFoundError, OSError):
        return False


pytestmark = [
    pytest.mark.contract,
    pytest.mark.skipif(
        not _synth_grid_available(), reason="Synthesizer AGN grids absent (data-gated)"
    ),
]


def _build(ssp, block, logU):
    return SEDModel.build(
        ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "all_params": FIXED,
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": FIXED,
        },
        agn={
            "type": "composable",
            "all_params": FIXED,
            "log_lbol": 13.0,
            "disc": {"type": "multicolor", "all_params": FIXED},
            "nlr": {"type": block, "nlr_logU": Fixed(logU), "all_params": FIXED},
        },
        redshift=Fixed(0.0),
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("block", ["synthesizer", "synthesizer_spectra"])
def test_synthesizer_nlr_logU_is_not_a_noop(synthetic_ssp_wide, block):
    """nlr='synthesizer[_spectra]' logU measurably changes the SED through the
    public grammar — regression for the frozen-neb_logU silent no-op (#931)."""

    def _sed(logU):
        return np.asarray(
            _build(synthetic_ssp_wide, block, logU).predict_state({}).derived["sed_agn"]
        )

    sed_a = _sed(-1.5)
    sed_b = _sed(-3.0)
    assert sed_a.max() > 0.0, f"nlr='{block}' produced a zero AGN SED"
    assert not np.allclose(sed_a, sed_b), (
        f"agn_nlr_logU is a silent no-op for nlr='{block}' through SEDModel.build "
        "— the Synthesizer grid axis is frozen at its default."
    )

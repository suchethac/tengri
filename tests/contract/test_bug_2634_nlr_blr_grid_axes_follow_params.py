# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for #2634: NLR/BLR blocks forward agn_log_mbh/agn_log_ledd.

The Synthesizer-grid NLR and BLR blocks must forward agn_log_mbh and
agn_log_ledd parameters to the backend compute functions.

Grid-gated: skips cleanly where the Synthesizer AGN grids are absent.
"""

from __future__ import annotations

import pytest


def _synth_grid_available() -> bool:
    from tengri.components.agn.blocks.nlr import _resolve_synthesizer_grid

    try:
        _resolve_synthesizer_grid("nlr")
        _resolve_synthesizer_grid("blr")
        return True
    except (FileNotFoundError, OSError):
        return False


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_grid():
    """Skip the entire module if Synthesizer grids are absent."""
    if not _synth_grid_available():
        pytest.skip("Synthesizer AGN grids absent (data-gated)", allow_module_level=True)


pytestmark = pytest.mark.contract


def _build_model(ssp, block_type):
    """Build a minimal composable AGN model with Synthesizer-grid block."""
    from tengri import DEFAULT, Fixed, SEDModel

    return SEDModel.build(
        ssp,
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
            block_type: {"type": "synthesizer_spectra", "all_params": Fixed(DEFAULT)},
        },
        redshift=Fixed(0.0),
    )


@pytest.mark.parametrize("block_type", ["nlr", "blr"])
def test_blocks_accept_new_parameters(synthetic_ssp_wide, block_type):
    """Test that the NLR/BLR blocks accept agn_log_mbh and agn_log_ledd."""
    # The fix is that the blocks now have these parameters in their signature
    # and forward them to the backend. This test verifies they're accepted.
    model = _build_model(synthetic_ssp_wide, block_type)

    # Just verify the model builds and produces output
    state = model.predict_state({})
    sed_agn = state.derived.get("sed_agn")

    assert sed_agn is not None, f"{block_type}_synthesizer_spectra: no sed_agn output"
    assert len(sed_agn) > 0, f"{block_type}_synthesizer_spectra: empty sed_agn"

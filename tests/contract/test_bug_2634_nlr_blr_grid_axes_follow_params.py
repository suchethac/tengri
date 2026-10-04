# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for #2634: NLR/BLR blocks forward agn_log_mbh/agn_log_ledd.

The Synthesizer-grid NLR and BLR blocks must forward agn_log_mbh and
agn_log_ledd parameters to the backend compute functions.

Grid-gated: skips cleanly where the Synthesizer AGN grids are absent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.contract


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


@pytest.mark.parametrize("region", ["nlr", "blr"])
def test_block_equals_backend_call(region):
    """Test (a): Block output equals backend call with explicit params.

    RED on main: block uses hardcoded defaults (8.0, -0.3) ignoring parameters.
    """
    from tengri.components.agn.blocks.nlr import (
        _resolve_synthesizer_grid,
        nlr_synthesizer_spectra_block,
    )
    from tengri.components.agn.blocks.blr import blr_synthesizer_spectra_block
    from tengri.components.agn.nlr_cloudy import compute_nlr_sed_synthesizer_spectra
    from tengri.utils.physics_constants import L_SUN as L_SUN_ERG, C_AA as C_AA_PER_S

    wave = np.linspace(1000, 10000, 100)
    l_bol_erg = 10.0 ** 13.0 * L_SUN_ERG
    grid_path = _resolve_synthesizer_grid(region)
    cf = 0.1
    logU = -2.0 if region == "nlr" else -1.0
    logn = 4.0
    logZ = -2.0

    # Call block with explicit parameters
    block_fn = nlr_synthesizer_spectra_block if region == "nlr" else blr_synthesizer_spectra_block
    kwargs = {
        "wavelength": wave,
        "agn_log_lbol": 13.0,
        "l5100_disc": np.zeros(1),
        "agn_log_mbh": 8.0,
        "agn_log_ledd": -1.0,
    }
    if region == "nlr":
        kwargs.update({"agn_nlr_cf": cf, "agn_nlr_logU": logU, "agn_nlr_logn": logn, "agn_nlr_logZ": logZ})
    else:
        kwargs.update({"agn_blr_cf": cf, "agn_blr_logU": logU, "agn_blr_logn": logn, "agn_blr_logZ": logZ})

    L_lambda_block, _ = block_fn(**kwargs)
    L_lambda_block = np.asarray(L_lambda_block)

    # Backend call at same parameters
    L_nu_backend = compute_nlr_sed_synthesizer_spectra(
        wave,
        l_disc_bol_erg=l_bol_erg,
        covering_fraction=cf,
        grid_path=grid_path,
        log_bh_mass=8.0,
        log_eddington=-1.0,
        neb_logU=logU,
        neb_logn=logn,
        neb_logZ_gas=logZ,
        region=region,
    )
    L_lambda_backend = L_nu_backend * C_AA_PER_S / wave**2

    np.testing.assert_allclose(
        L_lambda_block, L_lambda_backend, rtol=1e-12, atol=0,
        err_msg=f"{region}: block does not forward params to backend"
    )

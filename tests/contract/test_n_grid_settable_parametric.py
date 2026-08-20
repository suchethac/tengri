# SPDX-License-Identifier: BSD-3-Clause
"""User-set ``n_grid`` is honored for parametric (non-stochastic) SFHs.

Previously ``SEDModel`` forced ``n_grid = 256`` for any non-stochastic SFH
(``spec.n_grid if spec.stochastic else 256``), silently ignoring the user's
``SEDModel.build(..., n_grid=N)``. The SFH×SSP integral is n_grid-invariant for
parametric SFHs (it converges well before 64 points — see the #499 quadrature
check), so this is a control/perf knob, not a correctness change; the default
(256) is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri

pytestmark = [pytest.mark.contract]


def _build(ssp, obs, **extra):
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "delayed", "*": tengri.FIXED},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "*": tengri.FIXED,
        },
        dust_emission=None,
        neb={"type": "none"},
        redshift=tengri.Fixed(0.05),
        **extra,
    )


def test_n_grid_honored_for_parametric_sfh(synthetic_ssp_wide, synthetic_tophat_obs):
    """build(n_grid=N) now sets the SFH grid for a parametric SFH (was forced 256)."""
    assert _build(synthetic_ssp_wide, synthetic_tophat_obs).n_grid == 256  # default
    assert _build(synthetic_ssp_wide, synthetic_tophat_obs, n_grid=512).n_grid == 512
    assert _build(synthetic_ssp_wide, synthetic_tophat_obs, n_grid=64).n_grid == 64


def test_parametric_sed_is_n_grid_invariant(synthetic_ssp_wide, synthetic_tophat_obs):
    """Honoring n_grid must not change the parametric default result."""
    m256 = _build(synthetic_ssp_wide, synthetic_tophat_obs)
    m512 = _build(synthetic_ssp_wide, synthetic_tophat_obs, n_grid=512)
    p = {}
    np.testing.assert_allclose(
        np.asarray(m256.predict_photometry(p)),
        np.asarray(m512.predict_photometry(p)),
        rtol=1e-6,
    )

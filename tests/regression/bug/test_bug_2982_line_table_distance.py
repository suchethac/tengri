# SPDX-License-Identifier: BSD-3-Clause
r"""Regression: the per-Q_H line table is the same at a catalog build as at a plain build.

``precompute_line_per_qh`` evaluates the model at its own ``Fixed`` redshift, so the
distance it divides out must be the one at that redshift. A catalog build
(``WavePrecomp(catalog_z_range=...)``) took its reference redshift from the catalog
lower bound instead, so the table was low by ``4 pi d_L^2(z_fixed) / 4 pi d_L^2(z_lo)``:
113x at ``z = 0.1`` against a lower bound of 0.01. The reference redshift still sets the
reference SFH timing, which is pinned separately (#2748).

This is a VALUE test: a catalog-built model's table is compared with a plain model's table
at the same ``Fixed`` redshift, where the plain divisor is already correct by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import Observation, WavePrecomp
from tengri.components.nebular.line_precompute import precompute_line_per_qh
from tengri.observation.photometry_config import Photometry
from tests.regression.bug.test_bug_2748_reference_sfh_timing import _build_model_free_met
from tests.regression.bug.test_catalog_z_range_line_redshift import (
    BANDS,
    line_data,
    synthetic_ssp_wide,
)

pytestmark = pytest.mark.regression_bug

RTOL = 1e-6
Z_FIXED = 0.1
MET_LO, MET_HI, N_MET = -1.0, 0.3, 3


def _line_table(approx):
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())
    model = _build_model_free_met(ssp, obs, Z_FIXED, approx)
    lines = line_data().wavelengths
    return precompute_line_per_qh(model, lines, met_lo=MET_LO, met_hi=MET_HI, n_met=N_MET)


def test_catalog_line_table_matches_plain_table_at_fixed_redshift():
    """A catalog build's per-Q_H table equals a plain build's at the same Fixed z.

    The plain build's reference redshift is the Fixed value, so its divisor is already the
    distance at the redshift the flux is evaluated at. The catalog build must agree.
    """
    catalog = _line_table(WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=50))
    plain = _line_table(WavePrecomp())

    np.testing.assert_allclose(
        np.asarray(catalog.line_per_qh), np.asarray(plain.line_per_qh), rtol=RTOL
    )

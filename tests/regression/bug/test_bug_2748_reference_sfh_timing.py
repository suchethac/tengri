# SPDX-License-Identifier: BSD-3-Clause
r"""Regression: the FeaturePrecomp reference SFH has one absolute timing at every build redshift.

The per-Q_H nebular grid is built from a reference SFH drawn at a reference redshift
``ref_z``: the catalog lower bound for a ``catalog_z_range`` build (0.01 here) and
the ``Fixed`` value for a plain build (0.1 here). A sampled ``*_frac`` timing converts
to Gyr at the build redshift, so the same fraction gave two different absolute onsets,
``age(0.01)`` and ``age(0.1)`` apart by about 1.2 Gyr. The two grids then disagreed
by up to 7 % in line flux. The reference is a convention, so its timing is fixed in
the z = 0 frame: the grid build must see the same absolute onset at every ``ref_z``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import FeaturePrecomp, Observation, SEDModel, WavePrecomp
from tengri.cosmology import age_at_z
from tengri.observation.photometry_config import Photometry
from tests.regression.bug.test_catalog_z_range_line_redshift import (
    BANDS,
    _build_model,
    line_data,
    synthetic_ssp_wide,
)

pytestmark = pytest.mark.regression_bug


def _grid_reference_onsets(monkeypatch, approx):
    """Build one model and return the absolute onset [Gyr] the grid build reads.

    The per-Q_H grid build calls ``model._line_igm_transmission(ref_params, wave,
    ref_z)`` once with the reference parameters it is built from, so that call is
    recorded and its formation onset is ``sfh_dpl_age_frac * age(ref_z)``.
    """
    recorded = []
    original = SEDModel._line_igm_transmission

    def record(self, params, line_wave_rest, z):
        frac = params.get("sfh_dpl_age_frac")
        # Only the build's concrete reference call; a traced evaluation has no value.
        concrete = not isinstance(frac, jax.core.Tracer) and not isinstance(z, jax.core.Tracer)
        if frac is not None and concrete:
            recorded.append((float(frac), float(z)))
        return original(self, params, line_wave_rest, z)

    monkeypatch.setattr(SEDModel, "_line_igm_transmission", record)
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())
    _build_model(ssp, obs, 0.1, approx)
    assert recorded, "the grid build did not read a reference dpl onset"
    # The age is taken after the build, outside the grid's JAX trace.
    return np.asarray([frac * float(age_at_z(z)) for frac, z in recorded])


def test_feature_precomp_reference_onset_matches_across_build_redshifts(monkeypatch):
    """Catalog (lower bound 0.01) and plain (0.1) grids read the same absolute onset."""
    catalog = _grid_reference_onsets(
        monkeypatch,
        (WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=50), FeaturePrecomp(n_grid=4)),
    )
    plain = _grid_reference_onsets(monkeypatch, (WavePrecomp(), FeaturePrecomp(n_grid=4)))

    np.testing.assert_allclose(catalog, plain, rtol=1e-12)

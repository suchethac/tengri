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

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    FeaturePrecomp,
    Fixed,
    Observation,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.components.nebular.line_precompute import precompute_line_per_qh
from tengri.cosmology import age_at_z
from tengri.forward.sed_model import _snap_to_nebular_catalog
from tengri.observation.photometry_config import Photometry
from tests.regression.bug.test_catalog_z_range_line_redshift import (
    BANDS,
    _build_model,
    _build_model_fixed_dust,
    line_data,
    synthetic_ssp_wide,
)

pytestmark = pytest.mark.regression_bug


def _build_model_free_met(ssp, obs, z, approx):
    """Fixed nebular and dust, metallicity free: the line table sweeps ``met_logzsol``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(z),
            sfh={"type": "dpl", "all_params": FREE},
            met={"logzsol": Uniform(-1.0, 0.3)},
            neb={
                "type": "cue",
                "all_params": Fixed(DEFAULT),
                "logU": Fixed(-2.5),
                "logZ_gas": Fixed(0.0),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_bc": Fixed(1.0),
                "tau_diff": Fixed(0.2),
            },
            dust_emission={"type": "none"},
            approx=approx,
        )


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


def test_line_precompute_reference_onset_matches_across_build_redshifts(monkeypatch):
    """The per-Q_H line table reads the same absolute reference onset at each ref_z.

    The table's distance divisor is set by its reference redshift, which differs
    between the catalog and plain builds, so its values are not compared here. The
    onset it writes into the reference SFH is the quantity this test pins.
    """
    from tengri.parameters import timing

    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())
    lines = line_data().wavelengths
    real = timing.reference_timing_fracs
    onsets = []
    for approx in (WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=50), WavePrecomp()):
        model = _build_model_free_met(ssp, obs, 0.1, approx)
        seen = []

        def spy(spec, params, ref_z, _seen=seen):
            out = real(spec, params, ref_z)
            _seen.append(float(out["sfh_dpl_age_frac"]) * float(age_at_z(ref_z)))
            return out

        monkeypatch.setattr(timing, "reference_timing_fracs", spy)
        precompute_line_per_qh(model, lines, met_lo=-1.0, met_hi=0.3, n_met=3)
        monkeypatch.undo()
        assert seen, "the line table did not build a reference SFH"
        onsets.append(seen)
    np.testing.assert_allclose(onsets[0], onsets[1], rtol=1e-12)


def _recording_predict_state(original, recorded):
    """A ``predict_state`` that logs the concrete (fraction, redshift) it is given."""

    def record(self, params, *args, **kwargs):
        frac = params.get("sfh_dpl_age_frac")
        if frac is not None and not isinstance(frac, jax.core.Tracer):
            recorded.append((float(frac), float(self._get_redshift(params))))
        return original(self, params, *args, **kwargs)

    return record


def test_catalog_snap_reads_the_z0_reference_onset(monkeypatch):
    """The catalog line-wavelength snap reads the z = 0 frame reference onset.

    The snap evaluates the reference at the model's own redshift, so its onset is the
    sampled fraction times ``age(0)`` for a catalog build and a plain one alike.
    """
    ssp = synthetic_ssp_wide()
    obs = Observation(photometry=Photometry.from_names(BANDS), line_fluxes=line_data())
    targets = jnp.asarray([6564.61, 5006.84])
    for approx in (WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=50), WavePrecomp()):
        model = _build_model_fixed_dust(ssp, obs, 0.1, approx, tau_bc=1.0)
        sampled = float(model.spec.sample(jax.random.PRNGKey(0))["sfh_dpl_age_frac"])
        recorded = []
        monkeypatch.setattr(
            SEDModel,
            "predict_state",
            _recording_predict_state(SEDModel.predict_state, recorded),
        )
        _snap_to_nebular_catalog(model, targets)
        monkeypatch.undo()
        assert recorded, "the catalog snap did not read a reference dpl onset"
        onsets = [frac * float(age_at_z(z)) for frac, z in recorded]
        np.testing.assert_allclose(onsets, sampled * float(age_at_z(0.0)), rtol=1e-12)

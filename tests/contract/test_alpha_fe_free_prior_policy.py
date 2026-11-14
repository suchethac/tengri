#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""``met_alpha_fe`` is free only on an [alpha/Fe] grid with a spectroscopic channel (#2250).

Broadband photometry cannot constrain [alpha/Fe] at fixed total metallicity, so the
policy frees it only when the SSP grid has an [alpha/Fe] axis AND the observation
carries a spectrum or spectral indices. The free prior is the grid's own node range.

The synthetic 4-D grid has two features that respond to [alpha/Fe] in opposite
directions: a Mg b-like trough that deepens with alpha and an Fe-like pair of troughs
(5270 and 5335 A) that shallows with alpha. Mgb alone is degenerate between alpha and
total metallicity; Mgb with the two Fe indices is not. The acceptance test checks that
the MAP fit recovers both, and that the degeneracy is real.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from tengri import (
    DEFAULT,
    FREE,
    Fitter,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    Uniform,
)
from tengri.components.stellar.sps.dsps_wrapper import SSPData, has_alpha_grid
from tengri.config.exceptions import ParameterError
from tengri.observation.spectral_indices import STANDARD_INDICES, SpectralIndexData
from tengri.parameters import Fixed
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

ALPHA_NODES = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
LGMET_NODES = np.array([-1.5, -0.5, 0.0])  # log10(Z/Zsun) of the grid nodes
INDEX_NAMES = ("Mgb", "Fe5270", "Fe5335")
TRUTH = {"met_alpha_fe": 0.3, "met_logzsol": -0.2}


def _build_ssp(n_wave: int = 6000, *, with_alpha: bool = True) -> SSPData:
    n_met, n_age = 3, 20
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages = jnp.linspace(-1.0, 1.14, n_age)
    w = np.asarray(wave)
    mg = np.exp(-0.5 * ((w - 5175.0) / 8.0) ** 2)
    fe = np.exp(-0.5 * ((w - 5270.0) / 6.0) ** 2) + np.exp(-0.5 * ((w - 5335.0) / 6.0) ** 2)
    cont = 1.0 + 0.5 * (5500.0 / w)  # alpha- and Z-independent continuum shape
    cubes = []
    for a in ALPHA_NODES:
        per_met = []
        for lg in LGMET_NODES:
            f_z = 1.0 + 0.5 * (lg + 1.0)  # both features deepen with Z
            depth = 0.25 * (1.0 + a) * f_z * mg + 0.20 * (1.0 - a) * f_z * fe
            per_met.append(np.broadcast_to(cont * (1.0 - depth), (n_age, n_wave)))
        cubes.append(np.stack(per_met))
    flux = np.stack(cubes, axis=1)  # (n_met, n_alpha, n_age, n_wave)
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.asarray(flux),
        ssp_lg_age_gyr=ages,
        ssp_lgmet=jnp.asarray(LGMET_NODES + LOG10_ZSUN),
        ssp_alpha_fe=jnp.asarray(ALPHA_NODES) if with_alpha else None,
    )


@pytest.fixture(scope="module")
def ssp4d() -> SSPData:
    return _build_ssp()


@pytest.fixture(scope="module")
def ssp3d(ssp4d) -> SSPData:
    """The alpha=0 slab, a grid with no [alpha/Fe] axis."""
    return SSPData(
        ssp_wave=ssp4d.ssp_wave,
        ssp_flux=ssp4d.ssp_flux[:, 1],
        ssp_lg_age_gyr=ssp4d.ssp_lg_age_gyr,
        ssp_lgmet=ssp4d.ssp_lgmet,
    )


def _spectro_obs() -> Observation:
    wave_obs = jnp.linspace(4000.0, 8000.0, 400)
    return Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r"]),
        spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=jnp.full(400, 2000.0)),
    )


def _index_defs():
    return tuple(STANDARD_INDICES[n] for n in INDEX_NAMES)


def _index_data(values=None, errors=None) -> SpectralIndexData:
    n = len(INDEX_NAMES)
    return SpectralIndexData(
        index_defs=_index_defs(),
        values=jnp.zeros(n) if values is None else jnp.asarray(values),
        errors=jnp.full(n, 0.05) if errors is None else jnp.asarray(errors),
    )


def _index_obs(values=None, errors=None) -> Observation:
    return Observation(spectral_indices=_index_data(values, errors))


def _photo_obs() -> Observation:
    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))


def _model(ssp, observation, met):
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
        },
        met=met,
        redshift=Fixed(0.05),
        neb={"type": "ssp"},
    )


# -- the gate: which observation/grid combinations free met_alpha_fe ----------------


def test_fixture_is_4d_and_3d(ssp4d, ssp3d):
    assert has_alpha_grid(ssp4d)
    assert not has_alpha_grid(ssp3d)


def test_4d_spectroscopy_frees_alpha_with_grid_range(ssp4d, capsys):
    m = _model(ssp4d, _spectro_obs(), {"alpha_fe": FREE, "logzsol": Fixed(-0.3)})
    assert "met_alpha_fe" in m.spec.free_params
    dist = m.spec._distributions["met_alpha_fe"]
    assert dist.bounds == pytest.approx((ALPHA_NODES.min(), ALPHA_NODES.max()))
    m.spec.summary()
    summary = capsys.readouterr().out
    free_rows = [ln for ln in summary.splitlines() if "met_alpha_fe" in ln]
    assert free_rows and "Uniform" in free_rows[0], free_rows


def test_4d_spectral_indices_frees_alpha_with_grid_range(ssp4d):
    m = _model(ssp4d, _index_obs(), {"alpha_fe": FREE, "logzsol": Fixed(-0.3)})
    assert "met_alpha_fe" in m.spec.free_params
    dist = m.spec._distributions["met_alpha_fe"]
    assert dist.bounds == pytest.approx((ALPHA_NODES.min(), ALPHA_NODES.max()))


def test_4d_photometry_only_refuses_explicit_free(ssp4d):
    with pytest.raises(ParameterError, match="met_alpha_fe") as exc:
        _model(ssp4d, _photo_obs(), {"alpha_fe": FREE, "logzsol": Fixed(-0.3)})
    assert "spectrum" in str(exc.value)


def test_4d_photometry_only_all_params_leaves_alpha_at_zero(ssp4d):
    m = _model(ssp4d, _photo_obs(), {"all_params": FREE})
    assert "met_alpha_fe" not in m.spec.free_params
    dist = m.spec._distributions["met_alpha_fe"]
    assert dist.is_fixed and float(dist.value) == 0.0
    assert "met_logzsol" in m.spec.free_params


def test_4d_photometry_only_explicit_prior_is_honored(ssp4d):
    m = _model(ssp4d, _photo_obs(), {"alpha_fe": Uniform(-0.2, 0.4), "logzsol": Fixed(-0.3)})
    assert "met_alpha_fe" in m.spec.free_params
    assert m.spec._distributions["met_alpha_fe"].bounds == pytest.approx((-0.2, 0.4))


def test_3d_spectroscopy_refuses_explicit_free(ssp3d):
    with pytest.raises(ParameterError, match="met_alpha_fe"):
        _model(ssp3d, _spectro_obs(), {"alpha_fe": FREE, "logzsol": Fixed(-0.3)})


def test_explicit_prior_outside_the_nodes_is_refused_by_name(ssp4d):
    with pytest.raises(ParameterError, match="met_alpha_fe"):
        _model(ssp4d, _spectro_obs(), {"alpha_fe": Uniform(0.8, 1.0), "logzsol": Fixed(-0.3)})


def test_explicit_prior_past_the_nodes_warns_once(ssp4d):
    from tengri.config.exceptions import GridSupportWarning

    with pytest.warns(GridSupportWarning, match="met_alpha_fe"):
        m = _model(ssp4d, _spectro_obs(), {"alpha_fe": Uniform(-0.4, 0.4), "logzsol": Fixed(-0.3)})
    assert m.spec._distributions["met_alpha_fe"].bounds == pytest.approx((-0.4, 0.4))


# -- acceptance: the spectroscopic channel actually constrains the pair ------------


def _truth_model(ssp4d, observation):
    return _model(
        ssp4d,
        observation,
        {"alpha_fe": FREE, "logzsol": Uniform(-1.5, 0.0)},
    )


def _index_predictions(model, params):
    return np.asarray(model.predict_spectral_indices(params, _index_defs()))


def _log_like(pred, data, sigma):
    return -0.5 * float(np.sum(((pred - data) / sigma) ** 2))


def test_three_indices_recover_alpha_and_metallicity(ssp4d):
    """MAP with both free recovers the truth within 0.02 (alpha) and 0.02 dex (Z).

    Index likelihoods are constraint terms on top of a base data set, so the fit
    carries photometry as its base (noise-free at the truth, 5 % errors), exactly
    as the joint photometry + index fits in the suite do.
    """
    truth_model = _truth_model(ssp4d, _index_obs())
    idx = _index_predictions(truth_model, TRUTH)
    sigma = np.full(len(INDEX_NAMES), 0.01)
    obs = Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r"]),
        spectral_indices=_index_data(values=idx, errors=sigma),
    )
    fitted = _truth_model(ssp4d, obs)
    phot = np.asarray(fitted.predict_photometry(TRUTH))
    fm = ForwardModel.build(sed=fitted, observation=obs)
    post = Fitter(
        fm, data=phot, noise=0.05 * np.abs(phot), data_type="photometry", approx=None
    ).run("map")
    got = {name: float(np.asarray(post.params[name]).ravel()[0]) for name in TRUTH}
    assert abs(got["met_alpha_fe"] - TRUTH["met_alpha_fe"]) < 0.02, got
    assert abs(got["met_logzsol"] - TRUTH["met_logzsol"]) < 0.02, got


def test_mgb_alone_is_degenerate_and_the_three_indices_are_not(ssp4d):
    """Mgb-only has a flat ridge in (alpha, Z); the three indices break it."""
    model = _truth_model(ssp4d, _index_obs())
    sigma3 = np.full(len(INDEX_NAMES), 0.01)
    truth = _index_predictions(model, TRUTH)

    def mgb(alpha, z):
        return _index_predictions(model, {"met_alpha_fe": alpha, "met_logzsol": z})[0]

    # Second point on the Mgb ridge: same Mgb at a lower metallicity node.
    z2 = -0.5
    target = truth[0]
    alpha2 = brentq(lambda a: mgb(a, z2) - target, ALPHA_NODES.min(), ALPHA_NODES.max())
    pair2 = {"met_alpha_fe": alpha2, "met_logzsol": z2}
    pred2 = _index_predictions(model, pair2)

    assert abs(pred2[0] - truth[0]) < 1e-6 * abs(truth[0])
    assert abs(pred2[1] - truth[1]) > 0.05 or abs(pred2[2] - truth[2]) > 0.05

    chi2_1 = float(np.sum(((truth - truth) / sigma3) ** 2))
    chi2_2 = float(np.sum(((pred2 - truth) / sigma3) ** 2))
    assert chi2_2 - chi2_1 > 10.0
    # Mgb-only likelihood is identical at both pairs.
    assert _log_like(pred2[:1], truth[:1], sigma3[:1]) == pytest.approx(
        _log_like(truth[:1], truth[:1], sigma3[:1]), rel=1e-6
    )

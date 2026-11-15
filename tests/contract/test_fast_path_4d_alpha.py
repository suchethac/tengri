#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""The fast path (``compute_joint_weights`` and its consumers) supports a 4D [alpha/Fe] grid.

The joint (met, age) weights do not depend on [alpha/Fe]; the alpha dependence sits in what the
weights are contracted with. The exact forward contracts the 4D cube over metallicity per age and
then collapses the [alpha/Fe] axis at ``met_alpha_fe``. The fast path must do the same, so that:

a. ``compute_joint_weights`` on a 4D grid equals the weights on the 3D alpha = 0 slab;
b. ``compute_log_nion`` equals the ``log_nion`` the exact forward publishes;
c. the window-LUT spectral indices (``approx=True``) equal the exact indices, and the Mgb index
   moves with [alpha/Fe] (so the test cannot pass on a grid that ignores alpha);
d. the fast-path Mgb index has a finite, nonzero gradient in ``met_alpha_fe`` that matches a
   central finite difference;
e. ``measure_line_fluxes`` (approx=True) equals the exact path on the 4D grid;
f. on the 3D alpha = 0 slab the fast path still equals the exact path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, Uniform
from tengri.components.stellar.component import StellarSEDComponent
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.spectral_indices import STANDARD_INDICES
from tengri.parameters import Fixed
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

ALPHA_NODES = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
MGB = STANDARD_INDICES["Mgb"]


@pytest.fixture(scope="module")
def ssp4d():
    """Build 3 Z x 5 [alpha/Fe] x 20 ages x 1200 wavelengths, 500 to 10000 A.

    The grid starts below the Lyman limit so the ionizing slice is populated (test b). The
    alpha axis deepens a Mg b-like trough at 5175 A, so every node differs."""
    n_met, n_age, n_wave = 3, 20, 1200
    wave = jnp.linspace(500.0, 10000.0, n_wave)
    ages = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(7)
    base = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-3
    feature = jnp.exp(-0.5 * ((wave - 5175.0) / 40.0) ** 2)
    cubes = [base * (1.0 - 0.25 * (1.0 + a) * feature)[None, None, :] for a in ALPHA_NODES]
    flux = jnp.stack(cubes, axis=1)  # (n_met, n_alpha, n_age, n_wave)
    lgmet = jnp.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=lgmet,
        ssp_alpha_fe=jnp.asarray(ALPHA_NODES),
    )


def _slab(ssp4d, k):
    return SSPData(
        ssp_wave=ssp4d.ssp_wave,
        ssp_flux=ssp4d.ssp_flux[:, k],
        ssp_lg_age_gyr=ssp4d.ssp_lg_age_gyr,
        ssp_lgmet=ssp4d.ssp_lgmet,
    )


@pytest.fixture(scope="module")
def observation():
    wave_obs = jnp.linspace(4000.0, 8000.0, 400)
    return tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]),
        spectroscopy=tengri.Spectroscopy(wave_obs=wave_obs, resolution=jnp.full(400, 2000.0)),
    )


def _model(ssp, observation, alpha_spec):
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
        met={
            "logzsol": Fixed(-0.3),
            **({"alpha_fe": alpha_spec} if alpha_spec is not None else {}),
        },
        redshift=Fixed(0.05),
        neb={"type": "ssp"},
    )


@pytest.fixture(scope="module")
def model4d(ssp4d, observation):
    return _model(ssp4d, observation, Uniform(-0.2, 0.6))


@pytest.fixture(scope="module")
def model3d_alpha0(ssp4d, observation):
    return _model(_slab(ssp4d, 1), observation, None)


def _stellar(model) -> StellarSEDComponent:
    return next(c for c in model._feature_chain() if isinstance(c, StellarSEDComponent))


def _full(model, alpha):
    params = {} if alpha is None else {"met_alpha_fe": alpha}
    return model._evaluation_params(params, None)


# ── a. joint weights are alpha independent ─────────────────────────────


@pytest.mark.parametrize("alpha", [0.0, 0.4])
def test_joint_weights_match_the_alpha0_slab(model4d, model3d_alpha0, alpha):
    jw4, tm4, ages4 = _stellar(model4d).compute_joint_weights(_full(model4d, alpha))
    jw3, tm3, ages3 = _stellar(model3d_alpha0).compute_joint_weights(_full(model3d_alpha0, None))
    assert np.array_equal(np.asarray(jw4), np.asarray(jw3))
    assert np.array_equal(np.asarray(tm4), np.asarray(tm3))
    assert np.array_equal(np.asarray(ages4), np.asarray(ages3))


# ── b. compute_log_nion equals the exact log_nion ──────────────────────


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.6])
def test_log_nion_matches_the_exact_forward(model4d, alpha):
    fast = float(_stellar(model4d).compute_log_nion(_full(model4d, alpha)))
    exact = float(np.asarray(model4d.predict_state({"met_alpha_fe": alpha}).derived["log_nion"]))
    assert np.isfinite(exact)
    np.testing.assert_allclose(fast, exact, rtol=1e-12)


# ── c. window-LUT spectral indices equal the exact path ────────────────


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.6])
def test_mgb_fast_matches_exact(model4d, alpha):
    fast = np.asarray(
        model4d.predict_spectral_indices({"met_alpha_fe": alpha}, (MGB,), approx=True)
    )
    exact = np.asarray(
        model4d.predict_spectral_indices({"met_alpha_fe": alpha}, (MGB,), approx=False)
    )
    np.testing.assert_allclose(fast, exact, rtol=1e-10)


def test_mgb_moves_with_alpha(model4d):
    lo = float(model4d.predict_spectral_indices({"met_alpha_fe": 0.0}, (MGB,), approx=True)[0])
    hi = float(model4d.predict_spectral_indices({"met_alpha_fe": 0.6}, (MGB,), approx=True)[0])
    assert abs(hi - lo) > 1e-3 * abs(lo)


# ── d. gradient of the fast-path index in met_alpha_fe ─────────────────


def test_mgb_fast_gradient_matches_finite_difference(model4d):
    def mgb(a):
        return model4d.predict_spectral_indices({"met_alpha_fe": a}, (MGB,), approx=True)[0]

    a0, h = 0.25, 1e-4
    g = float(jax.grad(mgb)(a0))
    fd = float((mgb(a0 + h) - mgb(a0 - h)) / (2 * h))
    assert np.isfinite(g)
    assert g != 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-4)


# ── e. line fluxes from the window LUT ─────────────────────────────────


def test_measure_line_fluxes_fast_matches_exact(model4d):
    from tengri.observation.line_measurement import default_line_defs

    line_defs = default_line_defs([5175.0 * 1.05])
    fast = np.asarray(model4d.measure_line_fluxes({"met_alpha_fe": 0.3}, line_defs, approx=True))
    exact = np.asarray(model4d.measure_line_fluxes({"met_alpha_fe": 0.3}, line_defs, approx=False))
    assert np.all(np.isfinite(fast))
    np.testing.assert_allclose(fast, exact, rtol=1e-10)


# ── f. 3D control: the fast path is unchanged on the alpha = 0 slab ────


def test_mgb_fast_matches_exact_on_3d_slab(model3d_alpha0):
    fast = np.asarray(model3d_alpha0.predict_spectral_indices({}, (MGB,), approx=True))
    exact = np.asarray(model3d_alpha0.predict_spectral_indices({}, (MGB,), approx=False))
    np.testing.assert_allclose(fast, exact, rtol=1e-12)

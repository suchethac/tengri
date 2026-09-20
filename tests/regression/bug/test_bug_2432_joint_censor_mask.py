# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2432 — the censor mask did not follow the joint data.

Issue: ``ForwardModel.fit(Data(...))`` builds the joint fit by concatenating
photometry and spectroscopy into one vector, photometry first, because that is
the order the joint likelihood splits on. It concatenated ``data`` and
``noise`` and left ``data_mask`` at the photometry length. The censored
likelihood then applied an ``(n_filters,)`` mask to an
``(n_filters + n_pix,)`` prediction and raised ``Incompatible shapes for
broadcasting`` from inside a ``jnp.where``, three frames below ``fit()``, with
no mention of ``censor``.

Neither end was individually wrong, which is why neither suite caught it:
``Data.validate_against`` correctly refuses a mask that is not per-band, and
``censored_neg_log_likelihood`` correctly requires a mask matching
``predicted``. Together they made the configuration unreachable -- no value of
``censor`` satisfied both -- so a joint fit carrying any upper limit could not
run at all. Upper limits are routine in exactly the bands that make a fit
panchromatic, so this is not an exotic combination.

Solution: extend the mask where the fluxes are extended, marking every spectral
pixel DETECTED. A censored pixel would be a limit on one resolution element,
which no instrument reports; line limits go through ``Data.lines``.

https://github.com/suchethac/tengri/issues/2432
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tengri
from tengri import (
    DEFAULT,
    Data,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    Uniform,
)
from tengri.observation.noise import DETECTED, UPPER_LIMIT

pytestmark = pytest.mark.regression_bug

FILTERS = ["sdss_g", "sdss_r", "sdss_i"]
WAVE_OBS = np.linspace(4000.0, 7000.0, 40)


@pytest.fixture(scope="module")
def joint_model():
    """A small joint photometry + spectroscopy model."""
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
    obs = Observation(
        photometry=Photometry.from_names(FILTERS),
        spectroscopy=Spectroscopy(wave_obs=WAVE_OBS, resolution=1000.0),
    )
    sed = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )
    return ForwardModel.build(sed=sed)


def _joint_data(censor):
    """Synthetic joint measurements with the given per-band censor flags."""
    flux = np.array([1.0e-28, 1.2e-28, 1.4e-28])
    spec = np.full(WAVE_OBS.shape, 1.0e-28)
    return Data(
        photometry=(flux, 0.05 * flux),
        spectrum=(spec, 0.05 * spec),
        censor=np.asarray(censor, dtype=np.int32),
    )


def test_joint_fit_with_censored_band_runs(joint_model):
    """A joint fit carrying an upper limit reaches a finite MAP.

    Before the fix this raised ``Incompatible shapes for broadcasting:
    shapes=[(3,), (43,), (43,)]`` -- 3 bands against 3 + 40 pixels.
    """
    data = _joint_data([DETECTED, DETECTED, UPPER_LIMIT])
    post = joint_model.fit(data, method="map", key=jax.random.PRNGKey(0))
    value = float(post.params["sfh_delayed_log_total_mass"])
    assert np.isfinite(value), f"MAP returned non-finite mass {value}"


def test_censoring_still_changes_the_answer(joint_model):
    """The extension must not quietly neutralize the mask.

    Padding with DETECTED is only correct for the *spectral* half. A fix that
    reached the right shape by flattening the photometric flags too would make
    this test's two fits identical, and the first test alone would pass.
    """
    key = jax.random.PRNGKey(0)
    all_detected = joint_model.fit(_joint_data([DETECTED] * 3), method="map", key=key).params[
        "sfh_delayed_log_total_mass"
    ]
    one_limit = joint_model.fit(
        _joint_data([DETECTED, DETECTED, UPPER_LIMIT]), method="map", key=key
    ).params["sfh_delayed_log_total_mass"]

    assert not np.isclose(float(all_detected), float(one_limit), rtol=1e-6), (
        "flagging a band as an upper limit left the MAP unchanged, so the "
        "censoring is not reaching the likelihood"
    )


def test_photometry_only_censoring_unaffected(joint_model):
    """The photometry-only path, which always worked, still does."""
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
    obs = Observation(photometry=Photometry.from_names(FILTERS))
    sed = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )
    flux = np.array([1.0e-28, 1.2e-28, 1.4e-28])
    data = Data(
        photometry=(flux, 0.05 * flux),
        censor=np.array([DETECTED, DETECTED, UPPER_LIMIT], dtype=np.int32),
    )
    post = ForwardModel.build(sed=sed).fit(data, method="map", key=jax.random.PRNGKey(0))
    assert np.isfinite(float(post.params["sfh_delayed_log_total_mass"]))

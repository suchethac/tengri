# SPDX-License-Identifier: BSD-3-Clause
"""Presence on a joint photometry + spectrum fit, through the Fitter.

``presence`` has one entry per photometric band and masks the photometric block; the spectrum
pixels stay present. A joint fit with one band absent (stored as zero flux and zero error)
equals the joint fit on data from which that band was removed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fitter, Parameters, SEDModel, Uniform
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectroscopy import Spectroscopy
from tests.contract import _channel_fixtures as fixtures
from tests.contract.test_inference_cache_keys import _small_ssp

pytestmark = pytest.mark.contract

BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "2mass_j"]
N_PIX = 50
ABSENT = 2
WAVE = np.linspace(4000.0, 8000.0, N_PIX)
#: MAP parameters of the two fits agree to this relative tolerance: the objectives are equal,
#: so the optimizer follows the same path up to round-off.
MAP_RTOL = 1e-6


def _model(bands):
    observation = Observation(
        photometry=Photometry.from_names(bands), spectroscopy=Spectroscopy(wave_obs=WAVE)
    )
    spec = Parameters(
        redshift=0.1, sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.3, 3.0)
    )
    return SEDModel(spec, _small_ssp(), observation=observation)


@pytest.fixture(scope="module")
def cells():
    full = _model(BANDS)
    kept = [b for i, b in enumerate(BANDS) if i != ABSENT]
    point = dict(full.spec.sample(jax.random.PRNGKey(0)))
    phot, spec = (
        np.asarray(full.predict_photometry(point)),
        np.asarray(full.predict_spectrum(point)),
    )
    wiggle = lambda a: a * (1.0 + 0.05 * np.sin(np.arange(a.size)))  # noqa: E731
    phot, spec = wiggle(phot), wiggle(spec)
    phot_err, spec_err = 0.1 * np.abs(phot), 0.1 * np.abs(spec)
    present = np.arange(len(BANDS)) != ABSENT
    return {
        "full": full,
        "reduced": _model(kept),
        "presence": present.astype(float),
        "absent_data": np.r_[np.where(present, phot, 0.0), spec],
        "absent_noise": np.r_[np.where(present, phot_err, 0.0), spec_err],
        "reduced_data": np.r_[phot[present], spec],
        "reduced_noise": np.r_[phot_err[present], spec_err],
    }


def _absent_fitter(cells, presence=None):
    return Fitter(
        cells["full"],
        jnp.asarray(cells["absent_data"]),
        jnp.asarray(cells["absent_noise"]),
        data_type="joint",
        presence=cells["presence"] if presence is None else presence,
    )


def _reduced_fitter(cells):
    return Fitter(
        cells["reduced"],
        jnp.asarray(cells["reduced_data"]),
        jnp.asarray(cells["reduced_noise"]),
        data_type="joint",
    )


def test_a_joint_fit_takes_one_presence_entry_per_photometric_band(cells):
    fitter = _absent_fitter(cells)
    assert fitter.presence.shape == (len(BANDS),)
    term, physical = fixtures.build_term(fitter)
    assert np.isfinite(float(term.log_prob(physical, fitter._data_args)))


@pytest.mark.parametrize("length", (3, len(BANDS) + N_PIX))
def test_any_other_presence_length_is_refused_with_the_expected_length(cells, length):
    with pytest.raises(
        ValueError, match=rf"expected \({len(BANDS)},\), one entry per photometric"
    ):
        _absent_fitter(cells, presence=np.ones(length))


def test_the_data_term_of_an_absent_band_is_the_term_of_the_band_removed(cells):
    absent, reduced = _absent_fitter(cells), _reduced_fitter(cells)
    term_a, physical = fixtures.build_term(absent)
    term_r, _ = fixtures.build_term(reduced)
    np.testing.assert_allclose(
        float(term_a.log_prob(physical, absent._data_args)),
        float(term_r.log_prob(physical, reduced._data_args)),
        rtol=1e-12,
    )


def test_a_joint_map_fit_with_an_absent_band_equals_the_fit_without_the_band(cells):
    fit_absent = _absent_fitter(cells).run("map")
    fit_reduced = _reduced_fitter(cells).run("map")
    assert set(fit_absent.params) == set(fit_reduced.params)
    for name, value in fit_absent.params.items():
        np.testing.assert_allclose(
            np.asarray(value), np.asarray(fit_reduced.params[name]), rtol=MAP_RTOL, err_msg=name
        )

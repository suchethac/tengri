# SPDX-License-Identifier: BSD-3-Clause
r"""A model that projects a spectrum keeps its nebular continuum under the per-Q_H grid (#3227).

Serving photometry from the per-Q_H nebular grid zeroes ``sed_nebular``, which since
#1281 is allowed only when nothing downstream reads the continuum. The census asked the
component chain alone, and a spectrum is projected by the observation, not by a
component. So a dust-free Cue model with ``approx=(WavePrecomp(), FeaturePrecomp())``
and a spectrum ran its observables kernel on a state with ``sed_nebular`` exactly zero:
the joint photometry, projected off that state through ``observation.predict``, was
-6.9 %, -2.2 % and -14.4 % in g, r and i against ``approx=None`` (Halpha at z = 0.1 sits
in i), and the spectrum was off by up to 55 % per pixel. The photometry-only grid model
and the lazy ``predict(p).spectrum()`` were both correct, which is why it went unseen.

Pinned here, against ``approx=None`` on the same model:

* joint photometry and spectrum, and a spectrum-only model;
* ``nebular_grid_can_serve_photometry()`` answers False whenever a spectrum is configured,
  and still True for the photometry-only control, so the fitter advice and the behaviour
  agree.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_FILTERS = ["sdss_g", "sdss_r", "sdss_i"]
_SPEC_WAVE = jnp.linspace(4000.0, 7000.0, 40)
_PARAMS = ({"sfh_delayed_log_total_mass": 10.0}, {"sfh_delayed_log_total_mass": 9.3})
# The exact spectrum and the exact photometry are the same arithmetic on the same state
# once the continuum is present, so only float64 round-off separates them.
_RTOL = 1e-10


def _build(ssp, *, phot: bool, spec: bool, approx):
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Spectroscopy, Uniform

    observation = Observation(
        photometry=Photometry.from_names(_FILTERS) if phot else None,
        spectroscopy=Spectroscopy(wave_obs=_SPEC_WAVE) if spec else None,
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
        },
        dust_attenuation={"type": "none"},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=approx,
    )


@pytest.fixture(scope="module")
def models(ssp_data_fsps):
    from tengri import FeaturePrecomp, WavePrecomp

    grid = (WavePrecomp(), FeaturePrecomp())
    out = {
        "joint": _build(ssp_data_fsps, phot=True, spec=True, approx=grid),
        "joint_exact": _build(ssp_data_fsps, phot=True, spec=True, approx=None),
        "spec": _build(ssp_data_fsps, phot=False, spec=True, approx=grid),
        "spec_exact": _build(ssp_data_fsps, phot=False, spec=True, approx=None),
        "phot": _build(ssp_data_fsps, phot=True, spec=False, approx=grid),
    }
    for name in ("joint", "spec", "phot"):
        assert out[name]._nebular_grid_table is not None, f"setup: no grid on {name}"
    return out


@pytest.mark.parametrize("params", _PARAMS)
def test_joint_photometry_matches_the_exact_path(models, params):
    np.testing.assert_allclose(
        np.asarray(models["joint"].predict_photometry(params)),
        np.asarray(models["joint_exact"].predict_photometry(params)),
        rtol=_RTOL,
    )


@pytest.mark.parametrize("params", _PARAMS)
def test_joint_spectrum_matches_the_exact_path(models, params):
    np.testing.assert_allclose(
        np.asarray(models["joint"].predict_spectrum(params)),
        np.asarray(models["joint_exact"].predict_spectrum(params)),
        rtol=_RTOL,
    )


@pytest.mark.parametrize("params", _PARAMS)
def test_spectrum_only_matches_the_exact_path(models, params):
    np.testing.assert_allclose(
        np.asarray(models["spec"].predict_spectrum(params)),
        np.asarray(models["spec_exact"].predict_spectrum(params)),
        rtol=_RTOL,
    )


def test_a_spectrum_disarms_grid_photometry_and_nothing_else_does(models):
    assert not models["joint"].nebular_grid_can_serve_photometry()
    assert not models["spec"].nebular_grid_can_serve_photometry()
    assert models["phot"].nebular_grid_can_serve_photometry()

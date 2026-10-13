# SPDX-License-Identifier: BSD-3-Clause
"""Catalog mode with photometry + spectroscopy must use the runtime redshift (#2858).

With ``catalog_z_range`` set, a ``Fixed`` redshift is a placeholder: the real z arrives
per galaxy as a runtime input. The spectroscopy branch of the z-table setup used to test
only ``is_fixed``, so it built a fixed-z spec at the placeholder and overwrote the shared
``redshift_spec``, which also wiped the photometry free-z spec. Photometry and spectrum
then both used the placeholder z (errors of 2.5x and 7.9x in the issue).

The test takes data from a fixed-z model at the true z and scores the catalog model,
evaluated at that runtime z, against it. A correct model reproduces its own data.
"""

import warnings

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Spectroscopy, load_ssp_data
from tengri.forward.sed_model import SpectrumPrecomp, WavePrecomp
from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug

_SSP_CANDIDATES = (
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    "data/ssp_prsc_bc03_chabrier.h5",
)
_PLACEHOLDER_Z = 0.1
_TRUE_Z = 0.5
_CATALOG_RANGE = (0.01, 1.0)
# 5% errors. A model at the wrong z misses by 2.5x (photometry) and 7.9x (spectrum),
# which is a loss in the hundreds to hundreds of thousands; a correct one is ~1e-6.
_ERR_FRACTION = 0.05
_LOSS_CEILING = 1e-3


@pytest.fixture(scope="module")
def ssp():
    from pathlib import Path

    path = next((p for p in _SSP_CANDIDATES if Path(p).is_file()), None)
    if path is None:
        pytest.skip("no SSP grid on disk")
    return load_ssp_data(path)


def _obs():
    wave_obs = np.linspace(4000.0, 7000.0, 300)
    return Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r"]),
        spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=2000.0, resample="point"),
    )


def _build(ssp, redshift, approx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=_obs(),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)},
            redshift=redshift,
            approx=approx,
        )


def _loss_at_runtime_z(catalog, data_type, data, err, z):
    """Negative log posterior of the catalog model at runtime redshift ``z``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fit = Fitter(catalog, data, err, data_type=data_type, params_override={"redshift": z})
    ctx = InferenceContext.from_target(fit)
    p = ctx.initial_params(jax.random.PRNGKey(0))
    return float(ctx.neg_log_posterior_fn(p, ctx.data_args))


@pytest.fixture(scope="module")
def reference_and_catalog(ssp):
    catalog = _build(
        ssp,
        Fixed(_PLACEHOLDER_Z),
        (WavePrecomp(catalog_z_range=_CATALOG_RANGE), SpectrumPrecomp()),
    )
    reference = _build(ssp, Fixed(_TRUE_Z), None)
    return reference, catalog


@pytest.mark.parametrize("data_type", ["photometry", "spectroscopy"])
def test_catalog_model_at_runtime_z_reproduces_fixed_z_data(reference_and_catalog, data_type):
    reference, catalog = reference_and_catalog
    p_ref = dict(reference.spec.sample(jax.random.PRNGKey(0)))
    if data_type == "photometry":
        data = np.asarray(reference.predict_photometry(p_ref))
    else:
        data = np.asarray(reference.predict_spectrum(p_ref))
    err = _ERR_FRACTION * np.abs(data)

    loss = _loss_at_runtime_z(catalog, data_type, data, err, _TRUE_Z)
    assert loss < _LOSS_CEILING, (
        f"{data_type}: catalog model at runtime z misses its data ({loss})"
    )


@pytest.mark.parametrize("data_type", ["photometry", "spectroscopy"])
def test_placeholder_z_is_visibly_wrong(reference_and_catalog, data_type):
    """Guard against a vacuous pass: the placeholder z must score clearly worse."""
    reference, catalog = reference_and_catalog
    p_ref = dict(reference.spec.sample(jax.random.PRNGKey(0)))
    if data_type == "photometry":
        data = np.asarray(reference.predict_photometry(p_ref))
    else:
        data = np.asarray(reference.predict_spectrum(p_ref))
    err = _ERR_FRACTION * np.abs(data)

    at_placeholder = _loss_at_runtime_z(catalog, data_type, data, err, _PLACEHOLDER_Z)
    assert at_placeholder > 100.0 * _LOSS_CEILING

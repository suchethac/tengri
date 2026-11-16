# SPDX-License-Identifier: BSD-3-Clause
"""A saved and reloaded catalog posterior keeps its galaxy's redshift.

A catalog fit with per-galaxy redshifts stores each galaxy's value on its
posterior as ``runtime_values``. ``Posterior.save`` did not write it and
``Posterior.load`` did not restore it, so a reloaded posterior evaluated the
model at the build-time placeholder redshift: stellar mass and every derived
property came out wrong with no warning.

The round trip must be exact: the reloaded runtime values, the derived
properties and the posterior photometry are bit-identical to the live
posterior's.
"""

from __future__ import annotations

import warnings
from functools import cache, lru_cache

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

from tengri import (
    DEFAULT,
    Catalog,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp_data,
)
from tengri.inference.posterior import Posterior

SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
Z_PLACEHOLDER = 0.1
FIT_ZS = (0.06, 0.32)
FIT_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1", "wise_w2"]
FIT_TRUTH = {"sfh_dpl_log_total_mass": 10.0, "sfh_dpl_tau_gyr": 3.0, "dust_tau_diff": 0.3}
PROPERTIES = ("stellar_mass", "sfr_100myr", "mass_weighted_age_gyr")


@lru_cache(maxsize=1)
def _ssp():
    return load_ssp_data(SSP_PATH)


def _fit_model(redshift):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=_ssp(),
            observation=Observation(photometry=Photometry.from_names(FIT_BANDS)),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(8.0, 12.5),
                "tau_gyr": Uniform(0.5, 10.0),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_diff": Uniform(0.0, 1.5),
            },
            neb={"type": "ssp"},
            redshift=redshift,
        )


@cache
def _mock_fluxes():
    """Photometry and errors of the truth at each galaxy's redshift."""
    out = []
    for z in FIT_ZS:
        flux = np.asarray(_fit_model(Fixed(z)).predict_photometry(FIT_TRUTH))
        out.append((flux, 0.05 * flux))
    return out


def _catalog_table():
    fluxes = _mock_fluxes()
    table = {"z": np.asarray(FIT_ZS)}
    for j in range(len(FIT_BANDS)):
        table[f"f{j}"] = np.asarray([f[0][j] for f in fluxes])
        table[f"e{j}"] = np.asarray([f[1][j] for f in fluxes])
    return table


@cache
def _catalog_fit():
    """MAP catalog fit of two galaxies at different redshifts, built at a placeholder."""
    fwd = ForwardModel.build(sed=_fit_model(Fixed(Z_PLACEHOLDER)))
    cat = Catalog(
        fwd,
        _catalog_table(),
        flux_unit="cgs_fnu",
        redshift_col="z",
        flux_cols=[f"f{j}" for j in range(len(FIT_BANDS))],
        err_cols=[f"e{j}" for j in range(len(FIT_BANDS))],
    )
    return cat.fit(key=jax.random.PRNGKey(0), method="map", approx=None, properties=PROPERTIES)


def test_saved_posterior_keeps_runtime_redshift(tmp_path):
    fit = _catalog_fit()
    assert len(fit.posteriors) == len(FIT_ZS)

    for i, (live, z) in enumerate(zip(fit.posteriors, FIT_ZS, strict=True)):
        path = tmp_path / f"galaxy_{i}.h5"
        live.save(str(path))
        # The model the fit evaluated with, which declares the redshift as a runtime
        # key. A fixed-redshift rebuild would not, and would hide the round trip.
        loaded = Posterior.load(str(path), model=live._model)

        live_rt = live.runtime_values
        loaded_rt = loaded.runtime_values
        assert set(loaded_rt) == set(live_rt), f"galaxy {i}: runtime keys differ"
        assert float(live_rt["redshift"]) == pytest.approx(z)
        np.testing.assert_array_equal(
            np.asarray(loaded_rt["redshift"]),
            np.asarray(live_rt["redshift"]),
            err_msg=f"galaxy {i}: runtime redshift",
        )

        for name in PROPERTIES:
            np.testing.assert_array_equal(
                np.asarray(loaded.properties[name]),
                np.asarray(live.properties[name]),
                err_msg=f"galaxy {i}: {name} after reload",
            )

        np.testing.assert_array_equal(
            np.asarray(loaded.observables()),
            np.asarray(live.observables()),
            err_msg=f"galaxy {i}: posterior photometry after reload",
        )

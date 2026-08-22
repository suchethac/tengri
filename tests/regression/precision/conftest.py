# SPDX-License-Identifier: BSD-3-Clause
import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

Z_MASS_GRID = [(0.01, 10.0), (0.1, 8.0), (0.5, 11.0), (1.0, 10.0), (3.0, 10.0), (6.0, 9.0)]


@pytest.fixture(scope="module")
def ssp_bare():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data("data/fsps_prsc_miles_chabrier.h5")


def build_model(ssp, forward_dtype):
    """Build the panchromatic parity model (stellar + dust + Cue + AGN + radio + X-ray).

    Parameters
    ----------
    ssp : SSPData
        Bare-stellar SSP library (Cue nebular backend).
    forward_dtype : str
        Forward-model compute dtype, ``"float64"`` or ``"float32"``.

    Returns
    -------
    SEDModel
        The exact-path (``approx=None``) panchromatic model used by every
        float32 parity test as its shared subject.
    """
    from tengri import SEDModel, recipes
    from tengri.observation import Observation, Photometry

    obs = Observation(
        photometry=Photometry.from_names(
            ["galex_fuv", "sdss_r", "wise_w1", "wise_w4", "herschel_250"]
        )
    )
    recipe = recipes.agn_panchromatic()
    recipe["approx"] = None
    return SEDModel.build(ssp_data=ssp, observation=obs, forward_dtype=forward_dtype, **recipe)


def build_minimal_cue_model(ssp, forward_dtype):
    """Build a minimal stellar+Cue model (no AGN, dust IR, radio, xray).

    Used to isolate the ionizing-SED float32 safety check without AGN
    SKIRTOR interpolation failures that affect pure-f32 tests.
    """
    from tengri import FIXED, Fixed, SEDModel
    from tengri.observation import Observation, Photometry

    obs = Observation(photometry=Photometry.from_names(["sdss_r", "sdss_i"]))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        neb={"type": "cue", "all_params": FIXED},
        redshift=Fixed(1.0),
        approx=None,
        forward_dtype=forward_dtype,
    )


def forward_outputs(model, z, log10_mass):
    """Run one forward pass and return the parity observables as float64 arrays.

    Parameters
    ----------
    model : SEDModel
        A model from :func:`build_model` (any ``forward_dtype``).
    z : float
        Redshift to evaluate at.
    log10_mass : float
        ``sfh_dpl_log_total_mass`` [log10 Msun] to evaluate at.

    Returns
    -------
    dict
        Keys ``"photometry"``, ``"rest_sed"``, ``"halpha"``, ``"l_tir"``,
        ``"q_h"`` — all cast to float64 for cross-dtype comparison.
    """
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    p["redshift"] = z
    p["sfh_dpl_log_total_mass"] = log10_mass
    pred = model.predict(p)
    return {
        "photometry": np.asarray(pred.photometry(), dtype=np.float64),
        "rest_sed": np.asarray(pred.rest_sed(), dtype=np.float64),
        "halpha": np.float64(pred.lines.halpha),
        "l_tir": np.float64(pred.properties["l_tir"]),
        "q_h": np.float64(pred.properties["q_h"]),
    }

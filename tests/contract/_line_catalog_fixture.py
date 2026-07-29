# SPDX-License-Identifier: BSD-3-Clause
"""Minimal two-galaxy catalog carrying per-galaxy emission-line fluxes.

NOTE: synthetic_ssp_wide (conftest.py:167) is documented as a SMOOTH CONTINUUM
with NO nebular emission. This is intentional — we test the PLUMBING
(per-galaxy line data reaches likelihood), not the physics (model predicts
non-zero lines). With predicted lines = 0, the chi-squared term
((obs - 0) / err)**2 genuinely differs between obs=1e-16 and obs=4e-16,
proving the threading works.
"""

from __future__ import annotations

import jax
import numpy as np

from tengri import FIXED, FREE, Fixed, Observation, SEDModel
from tengri.inference.catalog import Catalog
from tengri.observation.line_flux_data import LineFluxData


def build_two_galaxy_catalog(*, halpha, ssp, obs_base, n_line_cols=None):
    """Build a two-row catalog whose galaxies differ only in Halpha flux.

    Parameters
    ----------
    halpha : tuple of float
        Halpha flux for each galaxy [erg/s/cm2].
    ssp : SSPData
        SSP data (e.g., synthetic_ssp_wide; does NOT need line emission).
    obs_base : Observation
        Observation (e.g., synthetic_tophat_obs).
    n_line_cols : int or None
        If specified, intentionally mismatch line column count for testing.

    Returns
    -------
    catalog : tengri.Catalog
    truth : dict
        The parameter dictionary both galaxies were generated from.
    """
    # Add line flux data to the observation
    line_names = ("Halpha",)
    line_wave = (6564.61,)
    line_fluxes_template = np.array([1.0e-16])  # template flux
    line_errors = np.array([0.1e-16])

    line_data = LineFluxData(
        names=line_names,
        fluxes=line_fluxes_template,
        errors=line_errors,
        wavelengths=np.array(line_wave),
    )
    obs = Observation(
        photometry=obs_base.photometry,
        line_fluxes=line_data,
    )

    # Build model with simple configuration; use "none" nebular since
    # synthetic_ssp_wide has no nebular emission anyway.
    z = 0.0
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "*": FREE},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(z),
    )

    # Generate truth parameters for galaxy 0
    key = jax.random.PRNGKey(42)
    truth_g0 = model.spec.sample(key)

    # Generate data for galaxy 0
    pred_g0 = model.predict(truth_g0)
    flux_g0 = pred_g0.photometry()
    noise_g0 = np.abs(flux_g0) * 0.05 + 1e-30

    # Galaxy 1: same photometry as galaxy 0, differ only in observed Halpha.
    # Even though model predicts Halpha=0, chi-squared term ((obs - 0) / err)**2
    # differs between obs=1e-16 and obs=4e-16, proving per-galaxy data reaches.
    flux_g1 = flux_g0.copy()
    noise_g1 = noise_g0.copy()

    # Build catalog table with per-galaxy line fluxes
    table = {
        "flux_1": np.array([flux_g0[0], flux_g1[0]]),
        "flux_2": np.array([flux_g0[1], flux_g1[1]]),
        "flux_3": np.array([flux_g0[2], flux_g1[2]]),
        "flux_4": np.array([flux_g0[3], flux_g1[3]]),
        "flux_5": np.array([flux_g0[4], flux_g1[4]]),
        "flux_1_err": np.array([noise_g0[0], noise_g1[0]]),
        "flux_2_err": np.array([noise_g0[1], noise_g1[1]]),
        "flux_3_err": np.array([noise_g0[2], noise_g1[2]]),
        "flux_4_err": np.array([noise_g0[3], noise_g1[3]]),
        "flux_5_err": np.array([noise_g0[4], noise_g1[4]]),
        "halpha_flux": np.array(halpha),
        "halpha_err": np.array([0.1e-16, 0.1e-16]),
    }

    # Create catalog with line columns
    if n_line_cols is not None and n_line_cols != len(line_names):
        # Intentionally wrong count for testing validation
        line_cols = [f"halpha_flux_{i}" for i in range(n_line_cols)]
    else:
        line_cols = ["halpha_flux"]

    cat = Catalog(
        model,
        table,
        flux_unit="cgs_fnu",
        flux_cols=["flux_1", "flux_2", "flux_3", "flux_4", "flux_5"],
        err_cols=["flux_1_err", "flux_2_err", "flux_3_err", "flux_4_err",
                  "flux_5_err"],
        line_cols=line_cols,
        line_err_cols=["halpha_err"],
    )

    return cat, truth_g0

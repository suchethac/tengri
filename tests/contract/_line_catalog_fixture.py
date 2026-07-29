# SPDX-License-Identifier: BSD-3-Clause
"""Minimal two-galaxy catalog carrying per-galaxy emission-line fluxes.

CRITICAL: Uses the EXISTING synthetic_ssp_wide and synthetic_tophat_obs fixtures
from conftest.py, which have proven wavelength coverage for Halpha line flux
prediction at z=0.05. Do NOT build custom SSPs — they risk omitting wavelength
ranges needed for physical line predictions.
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
        SSP data from synthetic_ssp_wide fixture (REQUIRED, not optional).
        Must have wavelength coverage that includes Halpha (6564.61 A) over
        the redshift range used (z=0.05 here).
    obs_base : Observation
        Observation from synthetic_tophat_obs fixture (REQUIRED, not optional).
        Provides the photometry bands.
    n_line_cols : int or None
        If specified, intentionally mismatch line column count for testing.

    Returns
    -------
    catalog : tengri.Catalog
    truth : dict
        The parameter dictionary both galaxies were generated from.

    Raises
    ------
    AssertionError
        If the model predicts zero line fluxes for the truth parameters
        (indicating wavelength coverage or model configuration issue).
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

    # Build the model
    # Use "cb19" nebular backend which actually computes line luminosities
    # (unlike "ssp" which is BakedInBackend and returns NaN for lines).
    # Use z=0 to keep Halpha at rest wavelength (6564.61 Å).
    z = 0.0  # Rest frame
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "*": FREE},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "cb19", "*": FREE},
        redshift=Fixed(z),
    )

    # Generate truth parameters for galaxy 0
    key = jax.random.PRNGKey(42)
    truth_g0 = model.spec.sample(key)

    # CRITICAL ASSERTION: the model MUST predict non-zero line fluxes.
    # If this fails, the SSP or redshift lacks wavelength coverage for lines.
    pred_g0 = model.predict(truth_g0)
    predicted_halpha = pred_g0.halpha
    assert predicted_halpha > 0, (
        f"Model predicts zero Halpha ({predicted_halpha}) for truth parameters. "
        f"Check SSP wavelength coverage, redshift z={z}, and nebular backend. "
        f"Without non-zero predictions, the line flux likelihood cannot constrain the fit."
    )

    # Generate data for galaxy 0
    flux_g0 = pred_g0.photometry()
    noise_g0 = np.abs(flux_g0) * 0.05 + 1e-30

    # Galaxy 1: same photometry as galaxy 0, but with different Halpha.
    # Both galaxies fitted against same photometry data, differ only in observed Halpha.
    # If per-galaxy line fluxes reach the likelihood, the fits should differ.
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

# SPDX-License-Identifier: BSD-3-Clause
"""Minimal two-galaxy catalog carrying per-galaxy emission-line fluxes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel
from tengri.inference.catalog import Catalog
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.photometry import FilterCurve


def build_two_galaxy_catalog(*, halpha, n_line_cols=None, ssp=None, obs_base=None):
    """Build a two-row catalog whose galaxies differ only in Halpha flux.

    Parameters
    ----------
    halpha : tuple of float
        Halpha flux for each galaxy [erg/s/cm2].
    n_line_cols : int or None
        If specified, intentionally mismatch line column count for testing.
    ssp : SSPData or None
        Synthetic SSP. If None, uses synthetic_ssp_wide.
    obs_base : Observation or None
        Base observation. If None, uses synthetic_tophat_obs.

    Returns
    -------
    catalog : tengri.Catalog
    truth : dict
        The parameter dictionary both galaxies were generated from.
    """
    if ssp is None:
        n_met, n_age = 3, 25
        wave = jnp.logspace(2.0, 7.0, 1600)
        ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
        lgmet = jnp.array([-4.0, -2.65, -1.3])
        base = (5000.0 / wave) ** 2
        flux = (
            base[None, None, :]
            * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
            * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
        )
        flux = jnp.abs(flux) + 1e-12

        from tengri.components.stellar.sps.dsps_wrapper import SSPData

        ssp = SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)

    if obs_base is None:

        def _tophat(center, frac=0.16, n=40):
            wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
            trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
            return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

        curves = tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0, 7600.0, 9000.0))
        obs_base = Observation(photometry=Photometry(filters=curves))

    # Add line flux data to the observation
    line_names = ("Halpha",)
    line_wave = (6564.61,)
    line_fluxes_template = jnp.array([1.0e-16])  # template flux
    line_errors = jnp.array([0.1e-16])

    line_data = LineFluxData(
        names=line_names,
        fluxes=line_fluxes_template,
        errors=line_errors,
        wavelengths=jnp.array(line_wave),
    )
    obs = Observation(
        photometry=obs_base.photometry,
        line_fluxes=line_data,
    )

    # Build the model
    z = 0.05
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "*": FREE},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "cue", "*": FREE},
        redshift=Fixed(z),
    )

    # Generate truth parameters for galaxy 0
    key = jax.random.PRNGKey(42)
    truth_g0 = model.spec.sample(key)

    # Generate data for galaxy 0
    pred_g0 = model.predict(truth_g0)
    flux_g0 = pred_g0.photometry()
    noise_g0 = jnp.abs(flux_g0) * 0.05 + 1e-30

    # Galaxy 1: same photometry as galaxy 0, but with 4x higher Halpha.
    # This creates a scenario where the line flux strongly suggests higher SFR
    # than the photometry alone would indicate. The fitter must adjust
    # nebular parameters (or other params) to produce 4x Halpha while keeping
    # the same photometry. Higher Halpha relative to the SED should favor
    # parameters that produce more young ionizing photons (higher SFR proxy).
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

    # For the test, return the first galaxy's truth
    truth = truth_g0

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
        err_cols=["flux_1_err", "flux_2_err", "flux_3_err", "flux_4_err", "flux_5_err"],
        line_cols=line_cols,
        line_err_cols=["halpha_err"],
    )

    return cat, truth

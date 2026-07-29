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

        from tengri.observation.ssp import SSPData

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
        neb={"type": "none"},
        redshift=Fixed(z),
    )

    # Generate truth
    key = jax.random.PRNGKey(42)
    truth = model.spec.sample(key)

    # Generate synthetic data for two galaxies with different Halpha
    flux_g1 = model.predict_photometry(truth)
    flux_g2 = flux_g1.copy()
    noise = jnp.abs(flux_g1) * 0.05 + 1e-30

    # Build catalog table with per-galaxy line fluxes
    table = {
        "flux_1": np.array([flux_g1[0], flux_g2[0]]),
        "flux_2": np.array([flux_g1[1], flux_g2[1]]),
        "flux_3": np.array([flux_g1[2], flux_g2[2]]),
        "flux_4": np.array([flux_g1[3], flux_g2[3]]),
        "flux_5": np.array([flux_g1[4], flux_g2[4]]),
        "flux_1_err": np.array([noise[0], noise[0]]),
        "flux_2_err": np.array([noise[1], noise[1]]),
        "flux_3_err": np.array([noise[2], noise[2]]),
        "flux_4_err": np.array([noise[3], noise[3]]),
        "flux_5_err": np.array([noise[4], noise[4]]),
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
        err_cols=["flux_1_err", "flux_2_err", "flux_3_err", "flux_4_err", "flux_5_err"],
        line_cols=line_cols,
        line_err_cols=["halpha_err"],
    )

    return cat, truth

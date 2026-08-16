# SPDX-License-Identifier: BSD-3-Clause
"""Conservation tests for stellar population synthesis: mass and SED balance.

Validates that integrated stellar mass and SED luminosity are self-consistent
across stellar population models.
"""

import pathlib
import warnings

import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.conservation

_SSP = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def ssp():
    if not _SSP.exists():
        pytest.skip(f"SSP not available at {_SSP}")
    return load_ssp_data(str(_SSP))


@pytest.fixture(scope="module")
def stellar_only_model(ssp):
    """Stellar-only SED model for conservation tests."""
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )
    phot = Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    obs = Observation(photometry=phot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs, approx=WavePrecomp())


_PARAMS = {
    "sfh_tsnorm_log_total_mass": 1.0,
    "sfh_tsnorm_peak_lbt_gyr": 2.0,
    "sfh_tsnorm_width_gyr": 1.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 3.0,
}


# ── Conservation: Stellar photometry LUT vs direct integration ─────


class TestStellarPhotometryConsistency:
    """Conservation test: stellar photometry LUT path matches direct integration.

    Energy conservation: precomputed filter-convolved SED (LUT) must give
    identical photometry to direct numerical integration of the intrinsic SED
    through the same filter response.
    """

    def test_stellar_phot_lnu_precomp_within_tolerance(self, stellar_only_model):
        """stellar_phot_lnu_precomp (LUT) matches direct integration to <0.5%.

        Phase 3c-3a: the LUT is built at the source's z; direct comparison
        uses the same z. Conservation check: ∫ SED × Filter via precompute
        path = ∫ SED × Filter via direct path (within numerical integration tol).
        """
        import math

        m = stellar_only_model
        state = m.predict_state(_PARAMS)
        lut_lnu = state.derived["stellar_phot_lnu_precomp"]
        from tengri.observation.photometry import compute_flux_density

        z = 0.05  # matches stellar_only_model's Fixed(0.05)
        dl_cm = 1.0
        cosmology = (1.0 + z) / (4.0 * math.pi * dl_cm**2)
        lut_fnu = lut_lnu * cosmology
        direct_fnu = jnp.asarray(
            [
                compute_flux_density(
                    state.sed_intrinsic,
                    state.wave,
                    fw,
                    ft,
                    redshift=z,
                    dl_cm=dl_cm,
                )
                for fw, ft in zip(
                    m.observation.photometry.filter_waves,
                    m.observation.photometry.filter_trans,
                    strict=False,
                )
            ]
        )
        rel_err = jnp.abs(lut_fnu - direct_fnu) / jnp.abs(direct_fnu)
        assert float(jnp.max(rel_err)) < 5e-3, (
            f"LUT diverges from direct: max rel err = {float(jnp.max(rel_err)):.4%}"
        )

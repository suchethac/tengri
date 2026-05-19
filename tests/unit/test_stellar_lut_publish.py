"""Tolerance test for Phase 3b stellar wave_precomp LUT publish.

Compares stellar_phot_lnu_lut (LUT path) against direct filter integration
of sed_intrinsic (exact path) on a stellar-only SDSS model.
"""

import pathlib
import warnings

import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed, Uniform

_SSP = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def ssp():
    if not _SSP.exists():
        pytest.skip(f"SSP not available at {_SSP}")
    return load_ssp_data(str(_SSP))


@pytest.fixture(scope="module")
def stellar_only_model(ssp):
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_peak_sfr=Uniform(-1, 3),
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
        return SEDModel(spec, ssp, observation=obs, approx={"wave_precomp": True})


_PARAMS = {
    "sfh_tsnorm_log_peak_sfr": 1.0,
    "sfh_tsnorm_peak_lbt_gyr": 2.0,
    "sfh_tsnorm_width_gyr": 1.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 3.0,
}


def test_stellar_phot_lnu_lut_within_tolerance(stellar_only_model):
    """stellar_phot_lnu_lut from LUT path matches direct integration to <0.5%.

    Comparison: both quantities are rest-frame Lν of the stellar component
    in erg/s/Hz, integrated through the same filters. The LUT path uses
    the precomputed SSP × filter table; the direct path uses the full
    state.sed_intrinsic. compute_flux_density applies a (1+z)/(4π·dl²)
    cosmology factor so we undo it with ``× 4π × dl_cm²`` to recover Lν.
    """
    import math

    m = stellar_only_model
    state = m.predict_via_orchestrator(_PARAMS)
    lut_path = state.derived["stellar_phot_lnu_lut"]  # erg/s/Hz, rest-frame Lν
    from tengri.observation.photometry import compute_flux_density

    # Undo compute_flux_density's (1+z)/(4π·dl_cm²) factor to recover bare Lν.
    dl_cm = 1.0
    inv_cosmology = 4.0 * math.pi * dl_cm**2  # at z=0 this is just 4π·dl²
    direct = (
        jnp.asarray(
            [
                compute_flux_density(
                    state.sed_intrinsic,
                    state.wave,
                    fw,
                    ft,
                    redshift=0.0,
                    dl_cm=dl_cm,
                )
                for fw, ft in zip(
                    m.observation.photometry.filter_waves,
                    m.observation.photometry.filter_trans,
                    strict=False,
                )
            ]
        )
        * inv_cosmology
    )
    # Tolerance: 0.5% per the documented hybrid-kernel accuracy
    # (Zacharegkas+2025; see docs/dev/optimization-architecture.md).
    rel_err = jnp.abs(lut_path - direct) / jnp.abs(direct)
    print(f"max rel_err = {float(jnp.max(rel_err)):.4%}")
    assert float(jnp.max(rel_err)) < 5e-3, (
        f"LUT diverges from direct: max rel err = {float(jnp.max(rel_err)):.4%}"
    )


def test_lut_only_published_when_wave_precomp_on(ssp):
    """state.derived does NOT contain stellar_phot_lnu_lut when wave_precomp=False (default)."""
    spec = Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_peak_sfr=Uniform(-1, 3),
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
    phot = Photometry.from_names(["sdss_u"])
    obs = Observation(photometry=phot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel(spec, ssp, observation=obs)  # wave_precomp default False
    state = m.predict_via_orchestrator(_PARAMS)
    assert "stellar_phot_lnu_lut" not in state.derived


def test_observation_predict_bit_exact_with_wave_precomp_on(stellar_only_model):
    """Phase 3b invariant: observation.predict is UNCHANGED when wave_precomp=True.

    The LUT is computed and published into derived but observation.predict
    continues integrating sed_intrinsic through filters as before. This
    test pins that invariant.
    """
    m = stellar_only_model
    # Phase 3b invariant: phot_fnu from predict_observables must equal the
    # legacy predict_photometry_via_orchestrator output bit-for-bit, even
    # though state.derived["stellar_phot_lnu_lut"] is now populated. The
    # LUT is internal; observation.predict still integrates sed_intrinsic.
    o = m.predict_observables(_PARAMS)
    legacy = m.predict_photometry_via_orchestrator(_PARAMS)
    diff = float(jnp.max(jnp.abs(o.phot_fnu - legacy)))
    assert diff < 1e-10, f"predict_observables drifted when wave_precomp=True: max diff = {diff}"

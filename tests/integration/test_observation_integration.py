# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for Observation with SEDModel and Fitter.

Requires SSP data — skips gracefully if not found.
"""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.forward.sed_model import SEDModel
from tengri.inference.fitter import Fitter
from tengri.observation.noise_model import NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry import FilterCurve
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectroscopy import Spectroscopy
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="module")
def base_spec():
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        redshift=Fixed(0.5),
    )


@pytest.fixture(scope="module")
def phot_obs():
    return Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
    )


def _make_synthetic_filters(n=3):
    """Create synthetic filters (no network required)."""
    filters = []
    for _i, (cen, name) in enumerate(
        [(4500, "synth_b"), (6200, "synth_r"), (8000, "synth_i")][:n]
    ):
        wave = jnp.linspace(cen - 500, cen + 500, 50)
        trans = jnp.exp(-0.5 * ((wave - cen) / 200.0) ** 2)
        filters.append(FilterCurve(wave=wave, trans=trans, name=name))
    return filters


# ── SEDModel integration ─────────────────────────────────────────────


class TestObservationWithModel:
    def test_model_accepts_observation_photometry(self, ssp, base_spec):
        """SEDModel(spec, ssp, observation=obs) with photometry works."""
        filters = _make_synthetic_filters()
        obs = Observation(photometry=Photometry(filters=tuple(filters)))
        model = SEDModel(base_spec, ssp, observation=obs)
        assert model.observation is not None
        assert model.observation.can_do_photometry
        assert model.filter_waves is not None
        assert len(model.filter_waves) == 3

    def test_model_observation_backward_compat(self, ssp, base_spec):
        """SEDModel(spec, ssp, filters=...) still works, creates Observation."""
        filters = _make_synthetic_filters()
        # 3-tuple format (simulating load_filter_set output)
        filter_data = (
            [f.wave for f in filters],
            [f.trans for f in filters],
            filters,
        )
        model = SEDModel(base_spec, ssp, filters=filter_data)
        assert model.observation is not None
        assert model.observation.can_do_photometry
        assert model.observation.photometry.n_filters == 3

    def test_model_rejects_both_filters_and_observation(self, ssp, base_spec):
        """ValueError if both filters= and observation= provided."""
        filters = _make_synthetic_filters()
        obs = Observation(photometry=Photometry(filters=tuple(filters)))
        with pytest.raises(ValueError, match="Cannot specify both"):
            SEDModel(base_spec, ssp, filters=filters, observation=obs)

    def test_auto_merge_adds_calibration_params(self, ssp, base_spec):
        """model.spec.free_params includes cal_c1..cN from spectroscopy."""
        wave_obs = jnp.linspace(10000, 50000, 100)
        obs = Observation(
            photometry=Photometry(filters=tuple(_make_synthetic_filters())),
            spectroscopy=Spectroscopy(wave_obs=wave_obs, calibration_order=2),
        )
        model = SEDModel(base_spec, ssp, observation=obs)
        assert "cal_c1" in model.spec.free_params
        assert "cal_c2" in model.spec.free_params

    def test_auto_merge_adds_noise_params(self, ssp, base_spec):
        """model.spec.free_params includes noise_frac_cal from noise config."""
        obs = Observation(
            photometry=Photometry(filters=tuple(_make_synthetic_filters())),
            noise=NoiseModel(calibration_floor=Uniform(0.01, 0.1)),
        )
        model = SEDModel(base_spec, ssp, observation=obs)
        assert "noise_frac_cal" in model.spec.free_params

    def test_auto_merge_user_precedence(self, ssp):
        """User-provided noise_frac_cal isn't overridden by auto-merge."""
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
            met_logzsol=Uniform(-1.5, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            redshift=Fixed(0.5),
            # User explicitly sets noise_frac_cal as a wider Uniform
            noise_frac_cal=Uniform(0.0, 0.5),
        )
        obs = Observation(
            photometry=Photometry(filters=tuple(_make_synthetic_filters())),
            # NoiseModel tries to auto-merge a different distribution
            noise=NoiseModel(calibration_floor=Uniform(0.01, 0.1)),
        )
        model = SEDModel(spec, ssp, observation=obs)
        # User's Uniform(0.0, 0.5) should win over NoiseModel's Uniform(0.01, 0.1)
        dist = model.spec.get_distribution("noise_frac_cal")
        assert isinstance(dist, Uniform)
        assert dist.bounds[1] == 0.5  # User's upper bound, not 0.1

    # Removed: test_auto_precompute_spectroscopy / test_no_precompute_when_z_free.
    # They asserted on ``model._precomputed.spectroscopy`` — the PrecomputedData
    # container retired in #620. Spectroscopy precompute is no longer auto-triggered
    # by a fixed-z Observation; it is opt-in via ``approx=SpectrumPrecomp()``, which
    # is covered by 31 tests (tests/contract/test_spectrum_lut.py,
    # test_spectrum_precomp_includes_agn.py, test_composite_approx.py, ...). There is
    # no auto-precompute behavior left to assert here.

    def test_lsf_settings_from_observation(self, ssp, base_spec):
        """LSF resolution/sigma_lib from Spectroscopy override spec attrs."""
        wave_obs = jnp.linspace(10000, 50000, 100)
        obs = Observation(
            spectroscopy=Spectroscopy(
                wave_obs=wave_obs,
                resolution=2000.0,
                sigma_lib_kms=15.0,
                lsf_n_bins=8,
            ),
        )
        model = SEDModel(base_spec, ssp, observation=obs)
        assert model._lsf_resolution == 2000.0
        assert model._sigma_lib_kms == 15.0
        assert model._lsf_n_bins == 8


# ── Fitter integration ────────────────────────────────────────────


class TestObservationWithFitter:
    def test_fitter_infers_photometry_type(self, ssp, base_spec):
        """Fitter(model, data, noise) with phot-only obs infers photometry."""
        obs = Observation(photometry=Photometry(filters=tuple(_make_synthetic_filters())))
        model = SEDModel(base_spec, ssp, observation=obs)
        # Generate fake data
        key = jax.random.PRNGKey(0)
        params = base_spec.sample(key)
        flux = model.predict_photometry(params)
        noise = jnp.ones_like(flux) * 0.1

        fitter = Fitter(model, flux, noise)
        assert fitter.data_type == "photometry"

    def test_fitter_explicit_data_type_still_works(self, ssp, base_spec):
        """Explicit data_type= overrides observation inference."""
        obs = Observation(photometry=Photometry(filters=tuple(_make_synthetic_filters())))
        model = SEDModel(base_spec, ssp, observation=obs)
        flux = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter = Fitter(model, flux, noise, data_type="photometry")
        assert fitter.data_type == "photometry"

    def test_fitter_no_observation_defaults_photometry(self, ssp, base_spec):
        """No observation, no data_type → defaults to 'photometry'."""
        model = SEDModel(base_spec, ssp)
        assert model.observation is None
        fitter = Fitter(model, jnp.ones(3), jnp.ones(3) * 0.1)
        assert fitter.data_type == "photometry"


# ── End-to-end ────────────────────────────────────────────────────


class TestObservationEndToEnd:
    def test_photometry_map_fit(self, ssp, base_spec):
        """Full: Observation → SEDModel → predict → Fitter → MAP."""
        obs = Observation(photometry=Photometry(filters=tuple(_make_synthetic_filters())))
        model = SEDModel(base_spec, ssp, observation=obs)

        # Generate mock photometry
        key = jax.random.PRNGKey(42)
        params = base_spec.sample(key)
        flux_true = model.predict_photometry(params)
        noise = jnp.abs(flux_true) * 0.05 + 1e-32

        fitter = Fitter(model, flux_true, noise)
        posterior = fitter.run("map", n_steps=20, optimizer="adam")
        assert posterior is not None
        assert posterior.params is not None

    def test_joint_pack_unpack_roundtrip(self, ssp, base_spec):
        """obs.pack_data → predict → obs.unpack_prediction consistency."""
        wave_obs = jnp.linspace(10000, 50000, 50)
        obs = Observation(
            photometry=Photometry(filters=tuple(_make_synthetic_filters())),
            spectroscopy=Spectroscopy(wave_obs=wave_obs),
        )
        model = SEDModel(base_spec, ssp, observation=obs)

        # Check pack/unpack shapes
        phot_data = jnp.ones(3) * 1e-29
        spec_data = jnp.ones(50) * 1e-30
        packed = obs.pack_data(phot=phot_data, spec=spec_data)
        chex.assert_shape(packed, (53,))
        unpacked = obs.unpack_prediction(packed)
        assert unpacked["photometry"].shape == (3,)
        assert unpacked["spectroscopy"].shape == (50,)

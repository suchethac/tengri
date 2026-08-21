# SPDX-License-Identifier: BSD-3-Clause
"""Tests for SEDModel.compile_signature() and Fitter.compile_signature().

Verifies that:
1. Signatures are hashable and deterministic
2. Identical models produce identical signatures
3. Different configurations produce different signatures
4. Signatures can be used as dictionary keys
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import Fitter, Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.contract


@pytest.fixture
def mock_ssp_data():
    """Return a minimal SSPData for testing."""
    n_met, n_age, n_wave = 8, 15, 200
    ssp = SSPData(
        ssp_wave=jnp.logspace(3, 4.5, n_wave),
        ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
    )
    return ssp


@pytest.fixture
def photometry():
    """Return a basic photometry observation."""
    return Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))


@pytest.fixture
def spec_dpl():
    """Return a simple DPL SFH spec."""
    return Parameters(
        redshift=0.1,
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.3, 3.0),
    )


class TestSEDModelCompileSignature:
    """Tests for SEDModel.compile_signature()."""

    def test_signature_is_hashable(self, mock_ssp_data, photometry, spec_dpl):
        """Signature should be hashable and usable as dict key."""
        model = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        sig = model.compile_signature()

        # Should be hashable
        hash(sig)

        # Should work as dict key
        test_dict = {sig: "value"}
        assert test_dict[sig] == "value"

    def test_signature_is_deterministic(self, mock_ssp_data, photometry, spec_dpl):
        """Signature should be identical across multiple calls."""
        model = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        sig1 = model.compile_signature()
        sig2 = model.compile_signature()
        assert sig1 == sig2

    def test_identical_models_same_signature(self, mock_ssp_data, photometry, spec_dpl):
        """Two models with identical shape should produce identical signatures."""
        model1 = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        model2 = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)

        sig1 = model1.compile_signature()
        sig2 = model2.compile_signature()
        assert sig1 == sig2

    def test_different_ssp_shapes_different_signature(self, photometry, spec_dpl):
        """Models with different SSP array shapes should have different signatures."""
        # Shape (8, 15, 200)
        ssp1 = SSPData(
            ssp_wave=jnp.logspace(3, 4.5, 200),
            ssp_flux=jnp.ones((8, 15, 200), dtype=jnp.float64),
            ssp_lg_age_gyr=jnp.linspace(6, 10.1, 15),
            ssp_lgmet=jnp.linspace(-2.0, 0.3, 8),
        )

        # Shape (8, 15, 150) - different wavelength count
        ssp2 = SSPData(
            ssp_wave=jnp.logspace(3, 4.5, 150),
            ssp_flux=jnp.ones((8, 15, 150), dtype=jnp.float64),
            ssp_lg_age_gyr=jnp.linspace(6, 10.1, 15),
            ssp_lgmet=jnp.linspace(-2.0, 0.3, 8),
        )

        model1 = SEDModel(spec_dpl, ssp1, observation=photometry)
        model2 = SEDModel(spec_dpl, ssp2, observation=photometry)

        sig1 = model1.compile_signature()
        sig2 = model2.compile_signature()
        assert sig1 != sig2

    def test_different_dust_model_different_signature(self, mock_ssp_data, photometry):
        """Models with different dust configurations should have different signatures."""
        spec1 = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            dust_model="single_component",
        )
        spec2 = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            dust_model="two_component",
        )

        model1 = SEDModel(spec1, mock_ssp_data, observation=photometry)
        model2 = SEDModel(spec2, mock_ssp_data, observation=photometry)

        sig1 = model1.compile_signature()
        sig2 = model2.compile_signature()
        assert sig1 != sig2

    def test_igm_flag_affects_signature(self, mock_ssp_data, photometry):
        """Models with different IGM settings should have different signatures."""
        spec1 = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            apply_igm=True,
        )
        spec2 = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            apply_igm=False,
        )

        model1 = SEDModel(spec1, mock_ssp_data, observation=photometry)
        model2 = SEDModel(spec2, mock_ssp_data, observation=photometry)

        # Different IGM settings should produce different signatures
        sig1 = model1.compile_signature()
        sig2 = model2.compile_signature()
        assert sig1 != sig2

    def test_different_filters_different_signature(self, mock_ssp_data, spec_dpl):
        """Models with different filter counts should have different signatures."""
        phot1 = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g"]))
        phot2 = Observation(
            photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i"])
        )

        model1 = SEDModel(spec_dpl, mock_ssp_data, observation=phot1)
        model2 = SEDModel(spec_dpl, mock_ssp_data, observation=phot2)

        sig1 = model1.compile_signature()
        sig2 = model2.compile_signature()
        assert sig1 != sig2

    def test_signature_is_picklable(self, mock_ssp_data, photometry, spec_dpl):
        """Signature should be picklable."""
        import pickle

        model = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        sig = model.compile_signature()

        # Should pickle and unpickle successfully
        pickled = pickle.dumps(sig)
        unpickled = pickle.loads(pickled)
        assert unpickled == sig

    def test_astrodust_settings_affect_signature(self, mock_ssp_data, photometry, spec_dpl):
        """Astrodust spinning_dust and f_cnm settings must enter compile_signature.

        Models differing ONLY in spinning_dust or f_cnm settings must have
        different signatures to prevent silent cache collisions. This guards
        against M4 (deletion of the two signature entries) and enforces the
        structural-setting wiring in #1093.

        Dropping the astrodust_spinning_dust and astrodust_f_cnm entries from
        compile_signature() return will turn this red.
        """
        # Base spec with astrodust dust emission
        base_spec = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            dust_model="two_component",
            dust_emission="astrodust",
            astrodust_spinning_dust=False,
            astrodust_f_cnm=0.28,
        )

        # Test spinning_dust difference
        spec_no_spd = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            dust_model="two_component",
            dust_emission="astrodust",
            astrodust_spinning_dust=False,
            astrodust_f_cnm=0.28,
        )
        spec_yes_spd = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            dust_model="two_component",
            dust_emission="astrodust",
            astrodust_spinning_dust=True,
            astrodust_f_cnm=0.28,
        )

        model_no_spd = SEDModel(spec_no_spd, mock_ssp_data, observation=photometry)
        model_yes_spd = SEDModel(spec_yes_spd, mock_ssp_data, observation=photometry)

        sig_no_spd = model_no_spd.compile_signature()
        sig_yes_spd = model_yes_spd.compile_signature()

        assert sig_no_spd != sig_yes_spd, (
            "spinning_dust difference must produce different signatures"
        )

        # Test f_cnm difference (both with spinning_dust=True)
        spec_fcnm_low = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            dust_model="two_component",
            dust_emission="astrodust",
            astrodust_spinning_dust=True,
            astrodust_f_cnm=0.1,
        )
        spec_fcnm_high = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            dust_model="two_component",
            dust_emission="astrodust",
            astrodust_spinning_dust=True,
            astrodust_f_cnm=0.5,
        )

        model_fcnm_low = SEDModel(spec_fcnm_low, mock_ssp_data, observation=photometry)
        model_fcnm_high = SEDModel(spec_fcnm_high, mock_ssp_data, observation=photometry)

        sig_fcnm_low = model_fcnm_low.compile_signature()
        sig_fcnm_high = model_fcnm_high.compile_signature()

        assert sig_fcnm_low != sig_fcnm_high, "f_cnm difference must produce different signatures"


class TestFitterCompileSignature:
    """Tests for Fitter.compile_signature()."""

    def test_fitter_signature_hashable(self, mock_ssp_data, photometry, spec_dpl):
        """Fitter signature should be hashable."""
        model = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1
        fitter = Fitter(model, data, noise, data_type="photometry")

        sig = fitter.compile_signature()
        hash(sig)

        # Should work as dict key
        test_dict = {sig: "fitter"}
        assert test_dict[sig] == "fitter"

    def test_fitter_signature_deterministic(self, mock_ssp_data, photometry, spec_dpl):
        """Fitter signature should be deterministic."""
        model = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1
        fitter = Fitter(model, data, noise, data_type="photometry")

        sig1 = fitter.compile_signature()
        sig2 = fitter.compile_signature()
        assert sig1 == sig2

    def test_identical_fitters_same_signature(self, mock_ssp_data, photometry, spec_dpl):
        """Two fitters with identical config should produce identical signatures."""
        model1 = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        model2 = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)

        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter1 = Fitter(model1, data, noise, data_type="photometry")
        fitter2 = Fitter(model2, data, noise, data_type="photometry")

        sig1 = fitter1.compile_signature()
        sig2 = fitter2.compile_signature()
        assert sig1 == sig2

    def test_different_data_length_different_signature(self, mock_ssp_data, photometry, spec_dpl):
        """Fitters with different data lengths should have different signatures."""
        model = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)

        data1 = jnp.ones(3)
        data2 = jnp.ones(4)  # Different length
        noise = jnp.ones(4) * 0.1

        fitter1 = Fitter(model, data1, noise[:3], data_type="photometry")
        fitter2 = Fitter(model, data2, noise, data_type="photometry")

        sig1 = fitter1.compile_signature()
        sig2 = fitter2.compile_signature()
        assert sig1 != sig2

    def test_different_data_type_different_signature(self, mock_ssp_data, photometry, spec_dpl):
        """Fitters with different data types should have different signatures."""
        model = SEDModel(spec_dpl, mock_ssp_data, observation=photometry)
        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter1 = Fitter(model, data, noise, data_type="photometry")
        fitter2 = Fitter(model, data, noise, data_type="joint")

        sig1 = fitter1.compile_signature()
        sig2 = fitter2.compile_signature()
        # Different data_type → different signature
        assert sig1 != sig2

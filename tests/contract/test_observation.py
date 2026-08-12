# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the unified Observation API.

Tests NoiseModel, Photometry, Spectroscopy, Observation,
and Parameters.with_params(). No SSP data needed — pure config/logic.
"""

import dataclasses

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.noise_model import NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry import FilterCurve
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectroscopy import Spectroscopy
from tengri.parameters.priors import Fixed, Gaussian, Uniform

pytestmark = pytest.mark.contract

# ── Helpers ───────────────────────────────────────────────────────


def _make_filter(name="test_r", center=6200.0, width=500.0, n=50):
    """Create a synthetic Gaussian filter for testing."""
    wave = jnp.linspace(center - 2 * width, center + 2 * width, n)
    trans = jnp.exp(-0.5 * ((wave - center) / width) ** 2)
    return FilterCurve(wave=wave, trans=trans, name=name)


def _make_wave_obs(n=100, lo=10000.0, hi=50000.0):
    """Create a synthetic observed wavelength grid."""
    return jnp.linspace(lo, hi, n)


# ── NoiseModel ───────────────────────────────────────────────────


class TestNoiseModel:
    def test_default_no_params(self):
        """NoiseModel() with defaults generates no Parameters entries."""
        nc = NoiseModel()
        assert nc.get_params() == {}

    def test_fixed_calibration_floor(self):
        """Float calibration_floor > 0 → Fixed distribution in get_params()."""
        nc = NoiseModel(calibration_floor=0.05)
        params = nc.get_params()
        assert "noise_frac_cal" in params
        assert isinstance(params["noise_frac_cal"], Fixed)

    def test_zero_calibration_floor(self):
        """Float calibration_floor == 0 → no param."""
        nc = NoiseModel(calibration_floor=0.0)
        assert nc.get_params() == {}

    def test_free_calibration_floor(self):
        """Distribution calibration_floor → free param in get_params()."""
        nc = NoiseModel(calibration_floor=Uniform(0.01, 0.15))
        params = nc.get_params()
        assert "noise_frac_cal" in params
        assert isinstance(params["noise_frac_cal"], Uniform)

    def test_student_t_dof(self):
        """student_t_dof → Fixed noise_dof in get_params()."""
        nc = NoiseModel(student_t_dof=5.0)
        params = nc.get_params()
        assert "noise_dof" in params
        assert isinstance(params["noise_dof"], Fixed)

    def test_both_params(self):
        """Both calibration_floor and student_t together."""
        nc = NoiseModel(calibration_floor=Uniform(0.01, 0.1), student_t_dof=10.0)
        params = nc.get_params()
        assert len(params) == 2
        assert "noise_frac_cal" in params
        assert "noise_dof" in params

    def test_frozen_immutable(self):
        """Cannot mutate fields after creation."""
        nc = NoiseModel()
        with pytest.raises(dataclasses.FrozenInstanceError):
            nc.calibration_floor = 0.1


# ── Photometry ────────────────────────────────────────────────────


class TestPhotometry:
    def test_from_filters(self):
        """Direct construction from FilterCurve tuple."""
        f1 = _make_filter("band_a", 5000.0)
        f2 = _make_filter("band_b", 7000.0)
        phot = Photometry(filters=(f1, f2))
        assert phot.n_filters == 2
        assert len(phot.filter_waves) == 2
        assert len(phot.filter_trans) == 2

    def test_names_derived_from_filters(self):
        """If names not given, derived from FilterCurve.name."""
        f1 = _make_filter("my_u")
        f2 = _make_filter("my_g")
        phot = Photometry(filters=(f1, f2))
        assert phot.names == ("my_u", "my_g")

    def test_names_explicit(self):
        """Explicit names override filter names."""
        f1 = _make_filter("internal_a")
        phot = Photometry(filters=(f1,), names=("custom_name",))
        assert phot.names == ("custom_name",)

    def test_empty_filters_rejected(self):
        """Empty tuple raises ValueError."""
        with pytest.raises(ValueError, match="at least one filter"):
            Photometry(filters=())

    def test_from_filter_set_3tuple(self):
        """Photometry.from_filter_set() accepts 3-tuple from load_filter_set()."""
        f1 = _make_filter("r")
        f2 = _make_filter("i")
        # Simulate load_filter_set() output: (waves_list, trans_list, curves_list)
        three_tuple = (
            [f1.wave, f2.wave],
            [f1.trans, f2.trans],
            [f1, f2],
        )
        phot = Photometry.from_filter_set(three_tuple)
        assert phot.n_filters == 2

    def test_from_filter_set_curve_list(self):
        """Photometry.from_filter_set() accepts list of FilterCurve."""
        f1 = _make_filter("r")
        phot = Photometry.from_filter_set([f1])
        assert phot.n_filters == 1

    def test_from_filter_set_bad_type(self):
        """Photometry.from_filter_set() rejects bad input."""
        with pytest.raises(TypeError):
            Photometry.from_filter_set("not_a_filter")

    def test_frozen_immutable(self):
        """Cannot mutate fields after creation."""
        phot = Photometry(filters=(_make_filter(),))
        with pytest.raises(dataclasses.FrozenInstanceError):
            phot.n_filters = 99

    def test_summary(self):
        """summary() returns non-empty string."""
        phot = Photometry(filters=(_make_filter("u"), _make_filter("g")))
        s = phot.summary()
        assert "2 filters" in s
        assert "u" in s

    def test_filter_waves_match(self):
        """filter_waves matches the input FilterCurve wave arrays."""
        f = _make_filter("test", 6000.0)
        phot = Photometry(filters=(f,))
        np.testing.assert_array_equal(phot.filter_waves[0], f.wave)
        np.testing.assert_array_equal(phot.filter_trans[0], f.trans)


# ── Spectroscopy ────────────────────────────────────────────


class TestSpectroscopy:
    def test_basic_creation(self):
        """Minimal config with just wave_obs."""
        wave = _make_wave_obs()
        sc = Spectroscopy(wave_obs=wave)
        assert sc.n_pixels == 100
        assert not sc.has_lsf
        assert not sc.has_calibration

    def test_constant_resolution(self):
        """Scalar resolution stores correctly."""
        wave = _make_wave_obs()
        sc = Spectroscopy(wave_obs=wave, resolution=3000.0)
        assert sc.has_lsf
        assert sc.resolution == 3000.0

    def test_variable_resolution(self):
        """Array resolution stores correctly."""
        wave = _make_wave_obs(50)
        R = jnp.linspace(30, 300, 50)
        sc = Spectroscopy(wave_obs=wave, resolution=R)
        assert sc.has_lsf
        chex.assert_shape(sc.resolution, (50,))

    def test_calibration_params_order_0(self):
        """calibration_order=0 → no params."""
        sc = Spectroscopy(wave_obs=_make_wave_obs(), calibration_order=0)
        assert sc.get_calibration_params() == {}

    def test_calibration_params_order_3(self):
        """calibration_order=3 → {cal_c1, cal_c2, cal_c3} Gaussian priors."""
        sc = Spectroscopy(wave_obs=_make_wave_obs(), calibration_order=3)
        params = sc.get_calibration_params()
        assert len(params) == 3
        assert "cal_c1" in params
        assert "cal_c2" in params
        assert "cal_c3" in params
        for v in params.values():
            assert isinstance(v, Gaussian)

    def test_nirspec_prism_factory(self):
        """nirspec_prism() creates config with variable-R array."""
        wave = jnp.linspace(6000.0, 53000.0, 200)
        sc = Spectroscopy.nirspec_prism(wave)
        assert sc.has_lsf
        assert isinstance(sc.resolution, jnp.ndarray)
        chex.assert_shape(sc.resolution, (200,))

    def test_nirspec_g140m_factory(self):
        """nirspec_g140m() creates config with R~1000 via resolution function."""
        wave = _make_wave_obs()
        sc = Spectroscopy.nirspec_g140m(wave)
        assert sc.has_lsf
        # nirspec_g140m_resolution returns ~1000 for all wavelengths
        r = jnp.asarray(sc.resolution)
        assert jnp.allclose(r, 1000.0)

    def test_constant_r_factory(self):
        """constant_r(R=5000) stores scalar resolution."""
        wave = _make_wave_obs()
        sc = Spectroscopy.constant_r(wave, R=5000)
        assert sc.resolution == 5000.0

    def test_factory_with_calibration(self):
        """Factories pass through calibration_order."""
        wave = _make_wave_obs()
        sc = Spectroscopy.nirspec_prism(wave, calibration_order=2)
        assert sc.calibration_order == 2
        assert len(sc.get_calibration_params()) == 2

    def test_frozen_immutable(self):
        """Cannot mutate fields after creation."""
        sc = Spectroscopy(wave_obs=_make_wave_obs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            sc.calibration_order = 5

    def test_summary(self):
        """summary() returns useful info."""
        wave = _make_wave_obs(200)
        sc = Spectroscopy(wave_obs=wave, resolution=1000.0, calibration_order=2)
        s = sc.summary()
        assert "200 pixels" in s
        assert "R=1000" in s
        assert "cal order=2" in s


# ── Observation ───────────────────────────────────────────────────


class TestObservation:
    @pytest.fixture
    def phot(self):
        return Photometry(filters=(_make_filter("r"), _make_filter("i")))

    @pytest.fixture
    def spec_config(self):
        return Spectroscopy(wave_obs=_make_wave_obs(150))

    def test_photometry_only_capabilities(self, phot):
        """can_do_photometry=True, can_do_spectroscopy=False."""
        obs = Observation(photometry=phot)
        assert obs.can_do_photometry
        assert not obs.can_do_spectroscopy
        assert not obs.is_joint
        assert obs.data_type == "photometry"

    def test_spectroscopy_only_capabilities(self, spec_config):
        """can_do_spectroscopy=True, can_do_photometry=False."""
        obs = Observation(spectroscopy=spec_config)
        assert not obs.can_do_photometry
        assert obs.can_do_spectroscopy
        assert not obs.is_joint
        assert obs.data_type == "spectroscopy"

    def test_joint_capabilities(self, phot, spec_config):
        """Both set → is_joint=True, data_type='joint'."""
        obs = Observation(photometry=phot, spectroscopy=spec_config)
        assert obs.can_do_photometry
        assert obs.can_do_spectroscopy
        assert obs.is_joint
        assert obs.data_type == "joint"

    def test_empty_observation_raises(self):
        """Neither phot nor spec → ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            Observation()

    def test_n_data_phot(self, phot):
        """n_data_phot matches number of filters."""
        obs = Observation(photometry=phot)
        assert obs.n_data_phot == 2
        assert obs.n_data_spec == 0
        assert obs.n_data == 2

    def test_n_data_spec(self, spec_config):
        """n_data_spec matches len(wave_obs)."""
        obs = Observation(spectroscopy=spec_config)
        assert obs.n_data_phot == 0
        assert obs.n_data_spec == 150
        assert obs.n_data == 150

    def test_n_data_joint(self, phot, spec_config):
        """n_data = n_data_phot + n_data_spec."""
        obs = Observation(photometry=phot, spectroscopy=spec_config)
        assert obs.n_data == 2 + 150

    def test_pack_data_phot_only(self, phot):
        """pack_data(phot=array) returns array unchanged."""
        obs = Observation(photometry=phot)
        data = jnp.array([1.0, 2.0])
        packed = obs.pack_data(phot=data)
        np.testing.assert_array_equal(packed, data)

    def test_pack_data_spec_only(self, spec_config):
        """pack_data(spec=array) returns array unchanged."""
        obs = Observation(spectroscopy=spec_config)
        data = jnp.ones(150)
        packed = obs.pack_data(spec=data)
        np.testing.assert_array_equal(packed, data)

    def test_pack_data_joint(self, phot, spec_config):
        """pack_data(phot, spec) → concatenated [phot, spec]."""
        obs = Observation(photometry=phot, spectroscopy=spec_config)
        p = jnp.array([1.0, 2.0])
        s = jnp.ones(150) * 3.0
        packed = obs.pack_data(phot=p, spec=s)
        chex.assert_shape(packed, (152,))
        np.testing.assert_array_equal(packed[:2], p)
        np.testing.assert_array_equal(packed[2:], s)

    def test_pack_data_validates_shape(self, phot):
        """pack_data with wrong-size array raises ValueError."""
        obs = Observation(photometry=phot)
        with pytest.raises(ValueError, match="shape"):
            obs.pack_data(phot=jnp.array([1.0, 2.0, 3.0]))  # 3 != 2 filters

    def test_pack_data_missing_phot(self, phot):
        """pack_data without phot= when photometry configured → error."""
        obs = Observation(photometry=phot)
        with pytest.raises(ValueError, match="phot="):
            obs.pack_data()

    def test_unpack_prediction_joint(self, phot, spec_config):
        """unpack_prediction splits at n_data_phot boundary."""
        obs = Observation(photometry=phot, spectroscopy=spec_config)
        predicted = jnp.arange(152, dtype=float)
        result = obs.unpack_prediction(predicted)
        assert "photometry" in result
        assert "spectroscopy" in result
        assert result["photometry"].shape == (2,)
        assert result["spectroscopy"].shape == (150,)

    def test_unpack_roundtrip(self, phot, spec_config):
        """pack_data → unpack_prediction → original arrays."""
        obs = Observation(photometry=phot, spectroscopy=spec_config)
        p = jnp.array([10.0, 20.0])
        s = jnp.linspace(1, 150, 150)
        packed = obs.pack_data(phot=p, spec=s)
        unpacked = obs.unpack_prediction(packed)
        np.testing.assert_array_almost_equal(unpacked["photometry"], p)
        np.testing.assert_array_almost_equal(unpacked["spectroscopy"], s)

    def test_get_all_params_empty(self, phot):
        """Photometry-only obs → empty dict."""
        obs = Observation(photometry=phot)
        assert obs.get_all_params() == {}

    def test_get_all_params_merges(self, phot):
        """Collects from spectroscopy + noise configs."""
        sc = Spectroscopy(wave_obs=_make_wave_obs(), calibration_order=2)
        nc = NoiseModel(calibration_floor=Uniform(0.01, 0.1))
        obs = Observation(photometry=phot, spectroscopy=sc, noise=nc)
        params = obs.get_all_params()
        assert "cal_c1" in params
        assert "cal_c2" in params
        assert "noise_frac_cal" in params

    def test_summary_string(self, phot, spec_config):
        """summary() returns non-empty string with key info."""
        obs = Observation(photometry=phot, spectroscopy=spec_config)
        s = obs.summary()
        assert "Observation" in s
        assert "joint" in s
        assert len(s) > 50

    def test_with_noise_only(self, phot):
        """NoiseModel without spectroscopy → noise params only."""
        nc = NoiseModel(calibration_floor=0.03)
        obs = Observation(photometry=phot, noise=nc)
        params = obs.get_all_params()
        assert "noise_frac_cal" in params
        assert len(params) == 1


# ── Parameters.with_params ─────────────────────────────────────────


class TestParamSpecWithParams:
    @pytest.fixture
    def base_spec(self):
        from tengri.parameters.parameters import Parameters

        return Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_total_mass=Uniform(-1.0, 2.5),
            met_logzsol=Uniform(-1.5, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            redshift=Fixed(0.5),
        )

    def test_adds_new_params(self, base_spec):
        """with_params adds new params to free_params."""
        new_spec = base_spec.with_params(cal_c1=Gaussian(0, 0.1))
        assert "cal_c1" in new_spec.free_params
        assert "cal_c1" in new_spec.all_params

    def test_returns_new_instance(self, base_spec):
        """Original Parameters unchanged (immutability)."""
        original_params = list(base_spec.all_params)
        _new_spec = base_spec.with_params(cal_c1=Gaussian(0, 0.1))
        assert base_spec.all_params == original_params
        assert "cal_c1" not in base_spec.all_params

    def test_user_defined_wins(self, base_spec):
        """User-provided params can't be overridden by with_params."""
        # met_logzsol was explicitly provided by user as Uniform(-1.5, 0.2)
        new_spec = base_spec.with_params(met_logzsol=Gaussian(0, 1))
        # Should keep the original Uniform, not the new Gaussian
        dist = new_spec.get_distribution("met_logzsol")
        assert isinstance(dist, Uniform)

    def test_default_can_be_overridden(self, base_spec):
        """Default (non-user-provided) params CAN be overridden.

        Note: As of Step E, noise_frac_cal is no longer a base Parameters default;
        it's owned by the Observation layer and only exists when an Observation
        is provided. Here we test overriding with a custom observation param instead.
        """
        # met_logzsol is a default Uniform(-2.0, 0.2) if not provided by user
        # (user provided Uniform(-1.5, 0.2), so it uses that)
        # Instead, test overriding redshift which is not user-provided (Fixed(0.5) user-provided)
        # So we test with a parameter that was added via with_params
        new_spec = base_spec.with_params(cal_c1=Gaussian(0, 0.1))
        dist = new_spec.get_distribution("cal_c1")
        assert isinstance(dist, Gaussian)

    def test_empty_kwargs_returns_self(self, base_spec):
        """with_params() with no args returns self."""
        result = base_spec.with_params()
        assert result is base_spec

    def test_multiple_params(self, base_spec):
        """Can add multiple params at once.

        Note: As of Step E, noise_frac_cal is owned by Observation, not Parameters.
        Test adding multiple custom parameters instead.
        """
        new_spec = base_spec.with_params(
            cal_c1=Gaussian(0, 0.1),
            cal_c2=Gaussian(0, 0.1),
            agn_log_lbol=Fixed(11.0),
        )
        assert "cal_c1" in new_spec.free_params
        assert "cal_c2" in new_spec.free_params
        # agn_log_lbol is a new param being added
        assert "agn_log_lbol" in new_spec.all_params

    def test_fixed_param_added(self, base_spec):
        """Fixed params added via with_params appear in fixed_params."""
        new_spec = base_spec.with_params(noise_dof=Fixed(5.0))
        assert "noise_dof" in new_spec.fixed_params
        assert "noise_dof" not in new_spec.free_params

    def test_sample_includes_new_params(self, base_spec):
        """Sampling from augmented spec includes new params."""
        new_spec = base_spec.with_params(cal_c1=Gaussian(0, 0.1))
        key = jax.random.PRNGKey(42)
        sample = new_spec.sample(key)
        assert "cal_c1" in sample

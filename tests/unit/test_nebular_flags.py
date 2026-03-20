"""Tests for nebular backend flag validation in ParamSpec."""

import warnings

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from diffsed.distributions import Fixed, Uniform
from diffsed.param_spec import ParamSpec


class TestNebularFlagConflicts:
    """Mutually exclusive flag validation."""

    def test_no_nebular_is_default(self):
        spec = ParamSpec()
        assert spec.nebular_mode == "off"

    def test_nebular_ssp_sets_mode(self):
        spec = ParamSpec(nebular_ssp=True)
        assert spec.nebular_mode == "ssp"

    def test_nebular_cloudy_requires_grid_path(self):
        with pytest.raises(ValueError, match="cloudy_grid_path"):
            ParamSpec(nebular=True)

    def test_nebular_cloudy_with_path(self):
        spec = ParamSpec(
            nebular=True,
            cloudy_grid_path="data/cloudy_grid_mist.h5",
        )
        assert spec.nebular_mode == "cloudy"

    def test_nebular_cue_sets_mode(self):
        spec = ParamSpec(nebular_cue=True)
        assert spec.nebular_mode == "cue"

    def test_nebular_cue_default_weights_path(self):
        spec = ParamSpec(nebular_cue=True)
        assert spec.cue_weights_path is not None
        assert "cue_weights" in str(spec.cue_weights_path)

    def test_nebular_cue_custom_weights_path(self):
        spec = ParamSpec(nebular_cue=True, cue_weights_path="/custom/path.npz")
        assert spec.cue_weights_path == "/custom/path.npz"

    def test_conflict_ssp_and_cloudy(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ParamSpec(nebular_ssp=True, nebular=True, cloudy_grid_path="x.h5")

    def test_conflict_ssp_and_cue(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ParamSpec(nebular_ssp=True, nebular_cue=True)

    def test_conflict_cloudy_and_cue(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ParamSpec(nebular=True, nebular_cue=True, cloudy_grid_path="x.h5")


class TestNebularSspWarnings:
    """BakedIn mode warns when user sets nebular params."""

    def test_warns_on_neb_logU(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ParamSpec(nebular_ssp=True, neb_logU=Uniform(-4, -1))
            neb_warnings = [x for x in w if "nebular_ssp" in str(x.message)]
            assert len(neb_warnings) > 0

    def test_no_neb_params_registered(self):
        spec = ParamSpec(nebular_ssp=True)
        assert "neb_logU" not in spec.free_params
        assert "neb_logU" not in spec.all_params


class TestNebularCloudyParams:
    """CLOUDY mode registers standard nebular params."""

    def test_registers_neb_params(self):
        spec = ParamSpec(nebular=True, cloudy_grid_path="data/cloudy_grid_mist.h5")
        assert "neb_logU" in spec.all_params
        assert "neb_fesc" in spec.all_params

    def test_does_not_register_ionspec(self):
        spec = ParamSpec(nebular=True, cloudy_grid_path="data/cloudy_grid_mist.h5")
        assert "ionspec_index1" not in spec.all_params


class TestNebularCueParams:
    """Cue mode registers nebular params + optional ionspec."""

    def test_registers_standard_neb_params(self):
        spec = ParamSpec(nebular_cue=True)
        assert "neb_logU" in spec.all_params
        assert "neb_fesc" in spec.all_params

    def test_ionspec_not_registered_by_default(self):
        spec = ParamSpec(nebular_cue=True)
        assert "ionspec_index1" not in spec.all_params
        assert "ionspec_index1" not in spec.free_params

    def test_ionspec_fixed_registered(self):
        spec = ParamSpec(nebular_cue=True, ionspec_index1=Fixed(5.0))
        assert "ionspec_index1" in spec.all_params
        assert "ionspec_index1" not in spec.free_params

    def test_ionspec_free_registered(self):
        spec = ParamSpec(nebular_cue=True, ionspec_index1=Uniform(1, 42))
        assert "ionspec_index1" in spec.free_params

    def test_gas_extra_params_optional(self):
        spec = ParamSpec(nebular_cue=True, gas_logn=Fixed(2.5))
        assert "gas_logn" in spec.all_params

    def test_ionspec_on_cloudy_raises(self):
        with pytest.raises(ValueError, match="ionspec"):
            ParamSpec(
                nebular=True,
                cloudy_grid_path="x.h5",
                ionspec_index1=Uniform(1, 42),
            )


class TestNebularIonization:
    """neb_ionization flag for Cue."""

    def test_default_is_ssp(self):
        spec = ParamSpec(nebular_cue=True)
        assert spec.neb_ionization == "ssp"

    def test_agn_not_implemented(self):
        with pytest.raises(NotImplementedError, match="AGN ionization"):
            ParamSpec(nebular_cue=True, neb_ionization="agn")

    def test_ssp_agn_not_implemented(self):
        with pytest.raises(NotImplementedError, match="AGN ionization"):
            ParamSpec(nebular_cue=True, neb_ionization="ssp+agn")


class TestBackwardCompat:
    """Old-style nebular flags still work."""

    def test_nebular_string_cue(self):
        spec = ParamSpec(nebular="cue")
        assert spec.nebular_mode == "cue"

    def test_cue_weights_path_implies_cue(self):
        spec = ParamSpec(cue_weights_path="data/cue_weights.npz")
        assert spec.nebular_mode == "cue"

    def test_cloudy_grid_path_implies_cloudy(self):
        spec = ParamSpec(cloudy_grid_path="data/cloudy_grid_mist.h5")
        assert spec.nebular_mode == "cloudy"

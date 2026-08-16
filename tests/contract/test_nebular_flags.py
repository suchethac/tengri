# SPDX-License-Identifier: BSD-3-Clause
"""Tests for nebular backend flag validation in Parameters."""

import warnings

import pytest

pytestmark = pytest.mark.contract

from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform


class TestNebularFlagConflicts:
    """Mutually exclusive flag validation."""

    def test_no_nebular_is_default(self):
        spec = Parameters()
        assert spec.nebular_mode == "off"

    def test_nebular_ssp_sets_mode(self):
        spec = Parameters(nebular_ssp=True)
        assert spec.nebular_mode == "ssp"

    def test_nebular_cloudy_requires_grid_path(self, monkeypatch):
        """With no grid discoverable, ``nebular=True`` must fail loudly.

        ``nebular=True`` auto-discovers a default CLOUDY grid when one is on
        disk (#1015). This test used to assert the error unconditionally, so
        its outcome depended on whether ``data/cloudy_grid_mist.h5`` happened
        to exist: it passed on a bare CI runner and failed on any machine with
        the real grids. Pin the contract instead — when discovery finds
        nothing, the error names the knob the user has to set.
        """
        monkeypatch.setattr(Parameters, "_default_cloudy_grid", lambda self: None)
        with pytest.raises(ValueError, match="cloudy_grid_path"):
            Parameters(nebular=True)

    def test_nebular_cloudy_uses_discovered_default(self, monkeypatch):
        """When a grid IS discoverable, ``nebular=True`` adopts it (#1015)."""
        monkeypatch.setattr(
            Parameters, "_default_cloudy_grid", lambda self: "/discovered/cloudy_grid.h5"
        )
        spec = Parameters(nebular=True)
        assert spec.nebular_mode == "cloudy"
        assert spec.cloudy_grid_path == "/discovered/cloudy_grid.h5"

    def test_nebular_cloudy_with_path(self):
        spec = Parameters(
            nebular=True,
            cloudy_grid_path="data/cloudy_grid_mist.h5",
        )
        assert spec.nebular_mode == "cloudy"

    def test_nebular_cue_sets_mode(self):
        spec = Parameters(nebular_cue=True)
        assert spec.nebular_mode == "cue"

    def test_nebular_cue_default_weights_path(self):
        spec = Parameters(nebular_cue=True)
        assert spec.cue_weights_path is not None
        assert "cue_weights" in str(spec.cue_weights_path)

    def test_nebular_cue_custom_weights_path(self):
        spec = Parameters(nebular_cue=True, cue_weights_path="/custom/path.npz")
        assert spec.cue_weights_path == "/custom/path.npz"

    def test_conflict_ssp_and_cloudy(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            Parameters(nebular_ssp=True, nebular=True, cloudy_grid_path="x.h5")

    def test_conflict_ssp_and_cue(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            Parameters(nebular_ssp=True, nebular_cue=True)

    def test_conflict_cloudy_and_cue(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            Parameters(nebular=True, nebular_cue=True, cloudy_grid_path="x.h5")


class TestNebularSspWarnings:
    """BakedIn mode warns when user sets nebular params."""

    def test_warns_on_neb_logU(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Parameters(nebular_ssp=True, neb_logU=Uniform(-4, -1))
            neb_warnings = [x for x in w if "nebular_ssp" in str(x.message)]
            assert len(neb_warnings) > 0

    def test_no_neb_params_registered(self):
        spec = Parameters(nebular_ssp=True)
        assert "neb_logU" not in spec.free_params
        assert "neb_logU" not in spec.all_params


class TestNebularCloudyParams:
    """CLOUDY mode registers standard nebular params."""

    def test_registers_neb_params(self):
        spec = Parameters(nebular=True, cloudy_grid_path="data/cloudy_grid_mist.h5")
        assert "neb_logU" in spec.all_params
        assert "neb_fesc" in spec.all_params

    def test_does_not_register_ionspec(self):
        spec = Parameters(nebular=True, cloudy_grid_path="data/cloudy_grid_mist.h5")
        assert "ionspec_index1" not in spec.all_params


class TestNebularCueParams:
    """Cue mode registers nebular params + optional ionspec."""

    def test_registers_standard_neb_params(self):
        spec = Parameters(nebular_cue=True)
        assert "neb_logU" in spec.all_params
        assert "neb_fesc" in spec.all_params

    def test_ionspec_not_registered_by_default(self):
        spec = Parameters(nebular_cue=True)
        assert "ionspec_index1" not in spec.all_params
        assert "ionspec_index1" not in spec.free_params

    def test_ionspec_fixed_registered(self):
        spec = Parameters(nebular_cue=True, ionspec_index1=Fixed(5.0))
        assert "ionspec_index1" in spec.all_params
        assert "ionspec_index1" not in spec.free_params

    def test_ionspec_free_registered(self):
        spec = Parameters(nebular_cue=True, ionspec_index1=Uniform(1, 42))
        assert "ionspec_index1" in spec.free_params

    def test_gas_extra_params_optional(self):
        spec = Parameters(nebular_cue=True, gas_logn=Fixed(2.5))
        assert "gas_logn" in spec.all_params

    def test_ionspec_on_cloudy_raises(self):
        with pytest.raises(ValueError, match="ionspec"):
            Parameters(
                nebular=True,
                cloudy_grid_path="x.h5",
                ionspec_index1=Uniform(1, 42),
            )


class TestNebularIonization:
    """neb_ionization flag for Cue."""

    def test_default_is_ssp(self):
        spec = Parameters(nebular_cue=True)
        assert spec.neb_ionization == "ssp"

    def test_agn_not_implemented(self):
        with pytest.raises(NotImplementedError, match="AGN ionization"):
            Parameters(nebular_cue=True, neb_ionization="agn")

    def test_ssp_agn_not_implemented(self):
        with pytest.raises(NotImplementedError, match="AGN ionization"):
            Parameters(nebular_cue=True, neb_ionization="ssp+agn")


class TestBackwardCompat:
    """Old-style nebular flags still work."""

    def test_nebular_string_cue(self):
        spec = Parameters(nebular="cue")
        assert spec.nebular_mode == "cue"

    def test_cue_weights_path_implies_cue(self):
        spec = Parameters(cue_weights_path="data/cue_weights.npz")
        assert spec.nebular_mode == "cue"

    def test_cloudy_grid_path_implies_cloudy(self):
        spec = Parameters(cloudy_grid_path="data/cloudy_grid_mist.h5")
        assert spec.nebular_mode == "cloudy"

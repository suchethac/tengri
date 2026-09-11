# SPDX-License-Identifier: BSD-3-Clause
"""Test that clone helpers (with_params, merge_observation_params) carry fresh provenance.

with_params() and merge_observation_params() should build a new _flat_provenance mapping
so that a parameter merged into a flat spec is tracked as user-provided, and the mapping
is immutable (MappingProxyType).
"""

import types

import pytest

from tengri import Fixed, Uniform
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.contract


class TestFlatProvenanceClones:
    """Test that clone helpers carry fresh provenance maps."""

    def test_with_params_creates_new_provenance_mapping(self):
        """with_params() should create a new, independent _flat_provenance mapping."""
        spec = Parameters(redshift=Fixed(0.5))
        new = spec.with_params(dust_delta=Uniform(-1.0, 0.4))

        # The mappings should be different objects
        assert new._flat_provenance is not spec._flat_provenance

    def test_with_params_adds_merged_param_to_provenance(self):
        """with_params() should add the merged parameter to the new _flat_provenance."""
        spec = Parameters(redshift=Fixed(0.5))
        new = spec.with_params(dust_delta=Uniform(-1.0, 0.4))

        assert "dust_delta" in new._flat_provenance
        assert new._flat_provenance["dust_delta"] == "user_prior"

    def test_with_params_omits_merged_param_from_original_provenance(self):
        """with_params() should not add the merged parameter to the original's _flat_provenance."""
        spec = Parameters(redshift=Fixed(0.5))
        new = spec.with_params(dust_delta=Uniform(-1.0, 0.4))

        assert "dust_delta" not in spec._flat_provenance

    def test_merge_observation_params_creates_new_provenance(self):
        """merge_observation_params() should create a new _flat_provenance mapping."""
        spec = Parameters(redshift=Fixed(0.5))
        new = spec.merge_observation_params(dust_delta=Fixed(-0.3))

        # The mappings should be different objects
        assert new._flat_provenance is not spec._flat_provenance

    def test_merge_observation_params_tracks_as_user_fixed(self):
        """merge_observation_params() should track a Fixed param as 'user_fixed'."""
        spec = Parameters(redshift=Fixed(0.5))
        new = spec.merge_observation_params(dust_delta=Fixed(-0.3))

        assert "dust_delta" in new._flat_provenance
        assert new._flat_provenance["dust_delta"] == "user_fixed"

    def test_merge_observation_params_tracks_as_user_prior(self):
        """merge_observation_params() should track a non-Fixed param as 'user_prior'."""
        spec = Parameters(redshift=Fixed(0.5))
        new = spec.merge_observation_params(dust_delta=Uniform(-1.0, 0.4))

        assert "dust_delta" in new._flat_provenance
        assert new._flat_provenance["dust_delta"] == "user_prior"

    def test_provenance_is_mappingproxy(self):
        """_flat_provenance should be a MappingProxyType (immutable)."""
        spec = Parameters(redshift=Fixed(0.5))
        assert isinstance(spec._flat_provenance, types.MappingProxyType)

    def test_provenance_mapping_is_immutable(self):
        """Attempting to write to _flat_provenance should raise TypeError."""
        spec = Parameters(redshift=Fixed(0.5))
        with pytest.raises(TypeError):
            spec._flat_provenance["dust_delta"] = "user_prior"

    def test_initial_provenance_tracks_user_fixed(self):
        """Initial _flat_provenance should track a Fixed param as 'user_fixed'."""
        spec = Parameters(redshift=Fixed(0.5))
        assert spec._flat_provenance.get("redshift") == "user_fixed"

    def test_initial_provenance_tracks_user_prior(self):
        """Initial _flat_provenance should track a non-Fixed param as 'user_prior'."""
        spec = Parameters(redshift=Uniform(0.1, 1.0))
        assert spec._flat_provenance.get("redshift") == "user_prior"

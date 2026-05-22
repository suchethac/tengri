# SPDX-License-Identifier: BSD-3-Clause
"""Tests for dust emission registry and public API contract.

Verify model registration, dispatcher routing, and module-level aliases.
"""

import jax.numpy as jnp
import pytest

from tengri.components.dust import emission as em
from tengri.components.dust.emission import (
    DUST_EMISSION_MODELS as _DUST_EMISSION_MODELS,
    preload_emission_model,
    resolve_emission_model,
)

DUST_EMISSION_MODELS = _DUST_EMISSION_MODELS

pytestmark = pytest.mark.contract

# ── Registration ──────────────────────────────────────────────────


class TestRegistration:
    """Verify the models are registered in the emission model registry."""

    def test_energy_balance_split_in_registry(self):
        from tengri.components.dust.emission import DUST_EMISSION_MODELS

        assert "energy_balance_split" in DUST_EMISSION_MODELS

    def test_resolve_emission_model_energy_balance_split(self):
        from tengri.components.dust.emission import resolve_emission_model

        fn = resolve_emission_model("energy_balance_split")
        assert callable(fn)

    def test_modified_blackbody_in_registry(self):

        assert "modified_blackbody" in DUST_EMISSION_MODELS

    def test_casey2012_in_registry(self):

        assert "casey2012" in DUST_EMISSION_MODELS

    def test_in_registry(self):

        assert "energy_balance_split" in DUST_EMISSION_MODELS

    def test_resolve_emission_model(self):

        fn = resolve_emission_model("energy_balance_split")
        assert callable(fn)


# ── Registry utilities ────────────────────────────────────────────


class TestRegistryUtilities:
    """Tests for resolve_emission_model, preload_emission_model."""

    def test_resolve_returns_callable(self):
        """resolve_emission_model returns a callable for known models."""

        for name in ("modified_blackbody", "energy_balance_split", "casey2012"):
            fn = resolve_emission_model(name)
            assert callable(fn), f"resolve_emission_model('{name}') is not callable"

    def test_resolve_unknown_raises_value_error(self):
        """resolve_emission_model raises ValueError for unknown model names."""

        with pytest.raises(ValueError, match="Unknown dust emission model"):
            resolve_emission_model("definitely_not_a_model_12345")

    def test_preload_unknown_raises_value_error(self):
        """preload_emission_model raises ValueError for unknown model names."""
        from tengri.components.dust.emission import preload_emission_model

        with pytest.raises(ValueError, match="Unknown emission model"):
            preload_emission_model("not_registered_xyz")

    def test_preload_known_returns_callable(self):
        """preload_emission_model returns a callable for the MBB model (no data needed)."""

        fn = preload_emission_model("modified_blackbody")
        assert callable(fn)

    def test_find_data_file_missing(self):
        """_find_data_file returns None for nonexistent files."""
        from tengri.components.dust.emission import _find_data_file

        result = _find_data_file("__definitely_not_here__.npz")
        assert result is None

    def test_find_data_file_present(self, tmp_path, monkeypatch):
        """_find_data_file returns the path when file exists in data/."""
        from tengri.components.dust import emission as em

        # Temporarily add tmp_path to search candidates
        original = em._DATA_CANDIDATES[:]
        fake_file = tmp_path / "test_dummy.npz"
        fake_file.write_bytes(b"")
        monkeypatch.setattr(em, "_DATA_CANDIDATES", [tmp_path])
        result = em._find_data_file("test_dummy.npz")
        monkeypatch.setattr(em, "_DATA_CANDIDATES", original)
        assert result == str(fake_file)

    def test_all_lazy_models_in_registry(self):
        """All expected lazy-loaded models are present in DUST_EMISSION_MODELS."""

        for name in ("draine_li2007", "dale2014", "draine_li2014", "astrodust", "bosa", "themis"):
            assert name in DUST_EMISSION_MODELS, f"'{name}' missing from registry"


# ── apply_dust_emission dispatcher ────────────────────────────────


class TestApplyDustEmission:
    """Tests for the apply_dust_emission high-level dispatcher.

    Public surface: dispatcher correctly routes to underlying models.
    """

    def test_delegates_to_modified_blackbody(self):
        """apply_dust_emission with 'modified_blackbody' matches direct call."""
        import jax.numpy as jnp

        from tengri.components.dust.emission import apply_dust_emission, modified_blackbody

        wave = jnp.logspace(5, 8, 100)
        L_abs = 1e10

        direct = modified_blackbody(wave, L_abs, dust_T=35.0)
        via_dispatcher = apply_dust_emission("modified_blackbody", wave, L_abs, dust_T=35.0)
        assert jnp.allclose(direct, via_dispatcher)

    def test_delegates_to_casey2012(self):
        """apply_dust_emission with 'casey2012' matches direct call."""

        from tengri.components.dust.emission import apply_dust_emission, casey2012

        wave = jnp.logspace(5, 8, 100)
        L_abs = 5e9

        direct = casey2012(wave, L_abs, dust_T=40.0)
        via_dispatcher = apply_dust_emission("casey2012", wave, L_abs, dust_T=40.0)
        assert jnp.allclose(direct, via_dispatcher)

    def test_unknown_name_raises(self):
        """apply_dust_emission raises ValueError for unknown model names."""

        from tengri.components.dust.emission import apply_dust_emission

        wave = jnp.logspace(5, 8, 50)
        with pytest.raises(ValueError, match="Unknown dust emission model"):
            apply_dust_emission("no_such_model", wave, 1e10)


# ── Module-level aliases ──────────────────────────────────────────


class TestModuleLevelAliases:
    """Module-level functions like draine_li2007() dispatch to the registry.

    Public surface: module-level convenience functions are callable.
    """

    def test_draine_li2007_alias_callable(self):
        """draine_li2007 module-level function is callable."""

        assert callable(em.draine_li2007)

    def test_dale2014_alias_callable(self):
        """dale2014 module-level function is callable."""

        assert callable(em.dale2014)

    def test_astrodust_alias_callable(self):
        """astrodust module-level function is callable."""

        assert callable(em.astrodust)

    def test_bosa_alias_callable(self):
        """bosa module-level function is callable."""

        assert callable(em.bosa)

    def test_themis_alias_callable(self):
        """themis module-level function is callable."""

        assert callable(em.themis)

    def test_draine_li2014_alias_callable(self):
        """draine_li2014 module-level function is callable."""

        assert callable(em.draine_li2014)


# ── TestModifiedBlackbody registry check ──────────────────────────


class TestModifiedBlackbody:
    """Standalone tests for modified_blackbody."""

    def test_registered_in_models(self):

        assert "modified_blackbody" in DUST_EMISSION_MODELS


# ── TestCasey2012 registry check ──────────────────────────────────


class TestCasey2012:
    """Standalone tests for casey2012 (MBB + mid-IR power law)."""

    def test_registered_in_models(self):

        assert "casey2012" in DUST_EMISSION_MODELS

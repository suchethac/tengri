# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the dust emission loader cache and public convenience API.

Dispatch is single via ``_REGISTRY`` (SEDModelComponents); the
surviving ``DUST_EMISSION_MODELS`` dict is an internal loader cache for the
tabulated HDF5 templates the grid components call into (NOT a dispatch table).
These tests pin the loader-cache contract and the public
``apply_dust_emission`` / module-alias convenience surface.
"""

import jax.numpy as jnp
import pytest

from tengri.components.dust import emission as em
from tengri.components.dust.emission import preload_emission_model
from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS

pytestmark = pytest.mark.contract


# ── Loader-cache registration ─────────────────────────────────────


class TestRegistration:
    """The eagerly- and lazily-registered emission models are in the cache."""

    def test_analytic_models_registered(self):
        # energy_balance_split is intentionally NOT here: its component
        # (EnergyBalanceSplitIRSEDComponent) calls the closure with a
        # non-default eta_balance=1.0, so a loader-cache entry would be a
        # divergent second dispatch path — removed for single dispatch (#850).
        # See test_dust_emission_single_dispatch.py for the component-side assertion.
        for name in ("graybody", "modified_blackbody", "casey2012"):
            assert name in DUST_EMISSION_MODELS
            assert callable(DUST_EMISSION_MODELS[name])

    def test_lazy_models_registered(self):
        for name in ("draine_li2007", "dale2014", "draine_li2014", "astrodust", "bosa", "themis"):
            assert name in DUST_EMISSION_MODELS, f"'{name}' missing from loader cache"

    def test_unknown_name_key_error(self):
        with pytest.raises(KeyError):
            DUST_EMISSION_MODELS["definitely_not_a_model_12345"]


# ── preload + data-file helpers ───────────────────────────────────


class TestLoaderUtilities:
    def test_preload_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown emission model"):
            preload_emission_model("not_registered_xyz")

    def test_preload_known_returns_callable(self):
        # MBB is analytic — no data needed to preload.
        assert callable(preload_emission_model("modified_blackbody"))

    def test_find_data_file_missing(self):
        from tengri._data_setup import find_data_str

        assert find_data_str("__definitely_not_here__.npz") is None

    def test_find_data_file_present(self, tmp_path, monkeypatch):
        # Redirect via $TENGRI_DATA_DIR rather than by patching a module-level
        # candidate list: the env var is the mechanism users actually have, so
        # this now exercises the real lookup instead of a test-only seam.
        from tengri._data_setup import find_data_str

        fake_file = tmp_path / "test_dummy.npz"
        fake_file.write_bytes(b"")
        monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))
        assert find_data_str("test_dummy.npz") == str(fake_file)


# ── apply_dust_emission convenience dispatcher ────────────────────


class TestApplyDustEmission:
    """The high-level ``apply_dust_emission(name, ...)`` convenience wrapper.

    This routes to the loader cache (not the pipeline dispatch); the pipeline
    itself dispatches only through ``_REGISTRY`` components.
    """

    def test_delegates_to_modified_blackbody(self):
        from tengri.components.dust.emission import apply_dust_emission, modified_blackbody

        wave = jnp.logspace(5, 8, 100)
        L_abs = 1e10
        direct = modified_blackbody(wave, L_abs, dust_T=35.0)
        via = apply_dust_emission("modified_blackbody", wave, L_abs, dust_T=35.0)
        assert jnp.allclose(direct, via)

    def test_delegates_to_casey2012(self):
        from tengri.components.dust.emission import apply_dust_emission, casey2012

        wave = jnp.logspace(5, 8, 100)
        L_abs = 5e9
        direct = casey2012(wave, L_abs, dust_T=40.0)
        via = apply_dust_emission("casey2012", wave, L_abs, dust_T=40.0)
        assert jnp.allclose(direct, via)

    def test_unknown_name_raises(self):
        from tengri.components.dust.emission import apply_dust_emission

        wave = jnp.logspace(5, 8, 50)
        with pytest.raises(ValueError, match="Unknown dust emission model"):
            apply_dust_emission("no_such_model", wave, 1e10)


# ── Module-level closure aliases ──────────────────────────────────


class TestModuleLevelAliases:
    def test_aliases_callable(self):
        for name in ("draine_li2007", "dale2014", "astrodust", "bosa", "themis", "draine_li2014"):
            assert callable(getattr(em, name))

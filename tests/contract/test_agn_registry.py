# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for AGN model registry: resolve and registration."""

from __future__ import annotations

import warnings

import chex
import jax.numpy as jnp
import pytest

from tengri.components.agn.unified import (
    AGN_MODELS as _AGN_MODELS,
    register_agn_model,
    resolve_agn_model,
)

AGN_MODELS = _AGN_MODELS

pytestmark = pytest.mark.contract


@pytest.fixture()
def wavelength():
    """Broad wavelength grid from radio (1 cm) to hard X-ray (1 A)."""
    return jnp.logspace(0, 8, 500)  # 1 A to 10^8 A (= 1 cm)


class TestResolveAgn:
    """Tests for resolve_agn_model error/warning branches."""

    def test_unknown_model_raises_value_error(self):
        """resolve_agn_model raises ValueError for unknown names."""
        from tengri.components.agn.unified import resolve_agn_model

        with pytest.raises(ValueError, match="Unknown AGN model"):
            resolve_agn_model("not_a_real_model_xyz_abc")

    def test_kubota_done_emits_deprecation_warning(self, wavelength):
        """resolve_agn_model('kubota_done') emits DeprecationWarning."""

        with pytest.warns(DeprecationWarning, match="kubota_done.*deprecated"):
            fn = resolve_agn_model("kubota_done")
        assert callable(fn)

    def test_kubota_done_still_returns_valid_function(self, wavelength):
        """Despite the deprecation warning, the returned function produces finite output."""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fn = resolve_agn_model("kubota_done")

        l_nu = fn(wavelength, agn_log_lbol=44.0)
        chex.assert_tree_all_finite(l_nu)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_all_canonical_models_in_registry(self):
        """All canonical model names resolve via resolve_agn_model."""
        from tengri.components.agn.unified import resolve_agn_model

        for name in (
            "multicolor_agn",
            "kubota_done",
            "kubota_done_full",
            "adaf",
            "unified_nlr_blr",
        ):
            with pytest.warns(DeprecationWarning):
                fn = resolve_agn_model(name)
            assert callable(fn), f"'{name}' failed to resolve"


class TestRegisterAgn:
    """Tests for the register_agn_model decorator factory."""

    def test_decorator_adds_model_to_registry(self):
        """register_agn_model adds the decorated function to AGN_MODELS."""
        from tengri.components.agn.unified import AGN_MODELS, register_agn_model

        @register_agn_model("_test_unit_model_a1b2")
        def _dummy(wavelength, agn_log_lbol, **_kw):
            return jnp.zeros_like(wavelength)

        try:
            assert "_test_unit_model_a1b2" in AGN_MODELS
            # Registry now stores AGNRegistryEntry which wraps the function
            entry = AGN_MODELS["_test_unit_model_a1b2"]
            assert callable(entry)
            assert entry.callable is _dummy
        finally:
            AGN_MODELS.pop("_test_unit_model_a1b2", None)

    def test_decorator_returns_original_function(self):
        """The decorator is transparent: the decorated function is returned unchanged."""

        def _raw(wavelength, agn_log_lbol, **_kw):
            return wavelength * agn_log_lbol

        decorated = register_agn_model("_test_unit_identity_c3d4")(_raw)
        try:
            assert decorated is _raw
        finally:
            AGN_MODELS.pop("_test_unit_identity_c3d4", None)

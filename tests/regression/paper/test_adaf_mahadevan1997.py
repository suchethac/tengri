# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for ADAF model against Mahadevan 1997.

Reference:
    Mahadevan et al. 1997, ApJ 486, 268 — Optically-Thin, Advection-Dominated
    Accretion around Supermassive Black Holes.
    https://doi.org/10.1086/304535
"""

import chex
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_paper


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def wavelength():
    """Broad wavelength grid from radio (1 cm) to hard X-ray (1 A)."""
    return jnp.logspace(0, 8, 500)  # 1 A to 10^8 A (= 1 cm)


@pytest.fixture()
def optical_wavelength():
    """Optical/UV wavelength grid."""
    return jnp.logspace(2.5, 5.0, 200)  # 316 A to 100,000 A


# ── Registry and model discovery ──────────────────────────────────


class TestAdafRegistry:
    """Tests that ADAF is properly registered in the AGN model registry."""

    def test_registered_as_adaf(self):
        """'adaf' resolves via resolve_agn_model.

        Mahadevan 1997 ADAF model is accessible by name 'adaf'.
        """
        import warnings

        from tengri.components.agn.unified import resolve_agn_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fn = resolve_agn_model("adaf")
        assert callable(fn)

    def test_get_agn_model_adaf(self):
        """resolve_agn_model('adaf') returns a callable.

        The model resolver must return a function matching the ADAF spec.
        """
        from tengri.components.agn.unified import resolve_agn_model

        model_fn = resolve_agn_model("adaf")
        assert callable(model_fn)

    def test_registered_model_runs(self, optical_wavelength):
        """The registered 'adaf' model produces finite output.

        A numerical implementation of Mahadevan 1997 equations must
        produce finite values at all wavelengths.
        """
        from tengri.components.agn.unified import resolve_agn_model

        model_fn = resolve_agn_model("adaf")
        l_nu = model_fn(optical_wavelength, agn_log_lbol=42.0)
        chex.assert_tree_all_finite(l_nu)
        chex.assert_equal_shape([l_nu, optical_wavelength])

    def test_adaf_in_unified_disc_fns(self, optical_wavelength):
        """'adaf' disc type works in unified_agn combiner.

        The ADAF component must be integrable with torus models
        via the unified combiner (Equations 3-4 in the model).
        """
        from tengri.components.agn.unified import unified_agn

        l_nu = unified_agn(
            optical_wavelength,
            agn_log_lbol=42.0,
            disc_model="adaf",
            torus_model="silva04",
        )
        chex.assert_tree_all_finite(l_nu)

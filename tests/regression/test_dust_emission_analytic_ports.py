# SPDX-License-Identifier: BSD-3-Clause
"""Bit-exact regression tests for analytic dust emission SEDModelComponent ports.

Tests that the SEDModelComponent wrappers produce identical output to the
original pure closures (zero tolerance) and that CMB parity is preserved
across redshifts.
"""

import jax.numpy as jnp
import numpy as np
import pytest

# Must import before the components to set up the registry
import tengri  # noqa: F401


@pytest.mark.regression_bug
class TestDustEmissionAnalyticPorts:
    """Bit-exact regression suite for dust emission model ports."""

    @pytest.fixture
    def wave_grid(self):
        """Standard wavelength grid for testing."""
        return jnp.linspace(1e3, 1e7, 512)

    @pytest.fixture
    def L_ir(self):
        """Standard absorbed luminosity for testing."""
        return 1e44  # erg/s (arbitrary scale)

    def test_registry_contains_analytic_names(self):
        """Verify all 4 analytic dust models are registered."""
        from tengri.components.sed_model_component import _REGISTRY

        required_names = {
            "modified_blackbody",
            "casey2012",
            "pah_drude",
            "schreiber2016",
        }
        registered_names = set(_REGISTRY.keys())
        assert required_names.issubset(registered_names), (
            f"Missing from registry: {required_names - registered_names}"
        )

    def test_modified_blackbody_z0_exact(self, wave_grid, L_ir):
        """Modified blackbody: z=0 golden test (exact match to closure)."""
        from tengri.components.dust.emission import (
            modified_blackbody as closure_fn,
        )
        from tengri.components.sed_model_component import _REGISTRY

        # Instantiate component
        comp = _REGISTRY["modified_blackbody"]()

        # Closure call (golden truth)
        golden = closure_fn(
            wave_grid,
            L_ir,
            dust_T=30.0,
            dust_beta_ir=1.8,
            dust_epsilon_mbb=1.0,
            redshift=0.0,
        )

        # Component call
        p = {"T": 30.0, "beta_ir": 1.8, "epsilon_mbb": 1.0, "redshift": 0.0}
        sed_out, _published = comp.predict(
            p,
            jnp.zeros_like(wave_grid),
            wave_grid,
            L_ir=L_ir,
        )

        # Extract emission SED (component adds to input, which is zeros)
        emission_component = sed_out

        # Exact match (same function, same inputs)
        np.testing.assert_allclose(
            emission_component,
            golden,
            rtol=0.0,
            atol=0.0,
            err_msg="Modified blackbody component does not match closure exactly",
        )
        # Also check the published dict
        assert np.allclose(_published["sed_dust_ir"], golden, rtol=0.0, atol=0.0)

    def test_modified_blackbody_cmb_parity(self, wave_grid, L_ir):
        """Modified blackbody: z>0 CMB parity (component vs direct closure)."""
        from tengri.components.dust.emission import (
            modified_blackbody as closure_fn,
        )
        from tengri.components.sed_model_component import _REGISTRY

        comp = _REGISTRY["modified_blackbody"]()

        # Test at z=6.0 (high redshift for strong CMB effect)
        z = 6.0

        # Closure call
        golden = closure_fn(
            wave_grid,
            L_ir,
            dust_T=30.0,
            dust_beta_ir=1.8,
            dust_epsilon_mbb=1.0,
            redshift=z,
        )

        # Component call
        p = {"T": 30.0, "beta_ir": 1.8, "epsilon_mbb": 1.0, "redshift": z}
        sed_out, _published = comp.predict(
            p,
            jnp.zeros_like(wave_grid),
            wave_grid,
            L_ir=L_ir,
        )

        # Exact match
        np.testing.assert_allclose(
            sed_out,
            golden,
            rtol=0.0,
            atol=0.0,
            err_msg=f"Modified blackbody CMB parity broken at z={z}",
        )

    def test_casey2012_z0_exact(self, wave_grid, L_ir):
        """Casey2012: z=0 golden test (exact match to closure)."""
        from tengri.components.dust.emission import casey2012 as closure_fn
        from tengri.components.sed_model_component import _REGISTRY

        comp = _REGISTRY["casey2012"]()

        # Closure call (golden truth)
        golden = closure_fn(
            wave_grid,
            L_ir,
            dust_T=35.0,
            dust_beta_ir=1.8,
            dust_alpha_mir=2.0,
            optically_thin=False,
            redshift=0.0,
        )

        # Component call
        p = {"T": 35.0, "beta_ir": 1.8, "alpha_mir": 2.0, "redshift": 0.0}
        sed_out, _published = comp.predict(
            p,
            jnp.zeros_like(wave_grid),
            wave_grid,
            L_ir=L_ir,
        )

        # Exact match
        np.testing.assert_allclose(
            sed_out,
            golden,
            rtol=0.0,
            atol=0.0,
            err_msg="Casey2012 component does not match closure exactly",
        )

    def test_casey2012_cmb_parity(self, wave_grid, L_ir):
        """Casey2012: z>0 CMB parity (component vs direct closure)."""
        from tengri.components.dust.emission import casey2012 as closure_fn
        from tengri.components.sed_model_component import _REGISTRY

        comp = _REGISTRY["casey2012"]()

        z = 6.0

        # Closure call
        golden = closure_fn(
            wave_grid,
            L_ir,
            dust_T=35.0,
            dust_beta_ir=1.8,
            dust_alpha_mir=2.0,
            optically_thin=False,
            redshift=z,
        )

        # Component call
        p = {"T": 35.0, "beta_ir": 1.8, "alpha_mir": 2.0, "redshift": z}
        sed_out, _published = comp.predict(
            p,
            jnp.zeros_like(wave_grid),
            wave_grid,
            L_ir=L_ir,
        )

        # Exact match
        np.testing.assert_allclose(
            sed_out,
            golden,
            rtol=0.0,
            atol=0.0,
            err_msg=f"Casey2012 CMB parity broken at z={z}",
        )

    def test_pah_drude_z0_exact(self, wave_grid, L_ir):
        """PAH Drude: z=0 golden test (exact match to closure)."""
        from tengri.components.dust.emission import pah_drude as closure_fn
        from tengri.components.sed_model_component import _REGISTRY

        comp = _REGISTRY["pah_drude"]()

        # Closure call (golden truth)
        golden = closure_fn(wave_grid, L_ir, redshift=0.0)

        # Component call (no free params)
        p = {"redshift": 0.0}
        sed_out, _published = comp.predict(
            p,
            jnp.zeros_like(wave_grid),
            wave_grid,
            L_ir=L_ir,
        )

        # Exact match
        np.testing.assert_allclose(
            sed_out,
            golden,
            rtol=0.0,
            atol=0.0,
            err_msg="PAH Drude component does not match closure exactly",
        )

    def test_pah_drude_redshift_invariance(self, wave_grid, L_ir):
        """PAH Drude: redshift should not affect rest-frame shape."""
        from tengri.components.sed_model_component import _REGISTRY

        comp = _REGISTRY["pah_drude"]()

        # Component call at z=0
        p0 = {"redshift": 0.0}
        sed_z0, _ = comp.predict(p0, jnp.zeros_like(wave_grid), wave_grid, L_ir=L_ir)

        # Component call at z=6.0
        p6 = {"redshift": 6.0}
        sed_z6, _ = comp.predict(p6, jnp.zeros_like(wave_grid), wave_grid, L_ir=L_ir)

        # PAH Drude shape should be unchanged (redshift unused)
        np.testing.assert_allclose(
            sed_z0,
            sed_z6,
            rtol=0.0,
            atol=0.0,
            err_msg="PAH Drude redshift scaling broken (should be invariant)",
        )

    def test_schreiber2016_z0_exact(self, wave_grid, L_ir):
        """Schreiber2016: z=0 golden test (exact match to closure)."""
        from tengri.components.dust.emission import schreiber2016 as closure_fn
        from tengri.components.sed_model_component import _REGISTRY

        comp = _REGISTRY["schreiber2016"]()

        # Closure call (golden truth, note: dust_f_pah in closure)
        golden = closure_fn(
            wave_grid,
            L_ir,
            dust_T=30.0,
            dust_f_pah=0.05,
            redshift=0.0,
        )

        # Component call (canonical f_pah, #849)
        p = {"T": 30.0, "f_pah": 0.05, "redshift": 0.0}
        sed_out, _published = comp.predict(
            p,
            jnp.zeros_like(wave_grid),
            wave_grid,
            L_ir=L_ir,
        )

        # Exact match
        np.testing.assert_allclose(
            sed_out,
            golden,
            rtol=0.0,
            atol=0.0,
            err_msg="Schreiber2016 component does not match closure exactly",
        )

    def test_schreiber2016_cmb_parity(self, wave_grid, L_ir):
        """Schreiber2016: z>0 CMB parity (component vs direct closure)."""
        from tengri.components.dust.emission import schreiber2016 as closure_fn
        from tengri.components.sed_model_component import _REGISTRY

        comp = _REGISTRY["schreiber2016"]()

        z = 6.0

        # Closure call
        golden = closure_fn(
            wave_grid,
            L_ir,
            dust_T=30.0,
            dust_f_pah=0.05,
            redshift=z,
        )

        # Component call
        p = {"T": 30.0, "f_pah": 0.05, "redshift": z}
        sed_out, _published = comp.predict(
            p,
            jnp.zeros_like(wave_grid),
            wave_grid,
            L_ir=L_ir,
        )

        # Exact match
        np.testing.assert_allclose(
            sed_out,
            golden,
            rtol=0.0,
            atol=0.0,
            err_msg=f"Schreiber2016 CMB parity broken at z={z}",
        )

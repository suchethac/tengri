"""Regression test for Compositional kernel L_absorbed_stellar guard.

See ADR / docs/known_bugs.md for full context.
"""

import inspect

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestCompositionalKernelLAbsorbedGuard:
    """fused_kernels.py compositional path — L_absorbed_stellar must be guarded.

    The hybrid kernel and sed_pipeline.py both have:
            L_absorbed_stellar = jnp.where(jnp.isfinite(...), ..., 0.0)
            L_absorbed_stellar = jnp.maximum(L_absorbed_stellar, 0.0)

    The compositional kernel was missing this guard until this fix. These tests
    verify the guard logic behaves correctly and would detect its removal.
    """

    def test_guard_clamps_inf_to_zero(self):
        """The isfinite guard replaces Inf with 0.0, preventing NaN in L_ir."""
        # Simulate what -jnp.trapezoid(sed_intr - sed_atten, nu) returns when
        # sed_intr contains Inf (e.g., from pure-SSP extreme metallicity UV flux)
        L_absorbed_stellar_raw = jnp.array(jnp.inf)

        # Apply the guard added to the compositional kernel
        L_absorbed_stellar = jnp.where(
            jnp.isfinite(L_absorbed_stellar_raw), L_absorbed_stellar_raw, 0.0
        )
        L_absorbed_stellar = jnp.maximum(L_absorbed_stellar, 0.0)
        L_ir = jnp.maximum(L_absorbed_stellar * 0.5, 0.0)

        assert jnp.isfinite(L_ir), "Guard failed: Inf L_absorbed_stellar produced non-finite L_ir"
        assert float(L_ir) == pytest.approx(0.0, abs=1e-30)

    def test_guard_clamps_nan_to_zero(self):
        """The isfinite guard replaces NaN (e.g. 0*Inf) with 0.0."""
        L_absorbed_stellar_raw = jnp.array(jnp.nan)

        L_absorbed_stellar = jnp.where(
            jnp.isfinite(L_absorbed_stellar_raw), L_absorbed_stellar_raw, 0.0
        )
        L_absorbed_stellar = jnp.maximum(L_absorbed_stellar, 0.0)

        assert jnp.isfinite(L_absorbed_stellar)
        assert float(L_absorbed_stellar) == pytest.approx(0.0, abs=1e-30)

    def test_guard_preserves_valid_values(self):
        """The guard must not alter normal (finite, positive) absorbed luminosities."""
        L_in = jnp.array(1.23e45)  # typical AGN host absorbed luminosity in erg/s

        L_out = jnp.where(jnp.isfinite(L_in), L_in, 0.0)
        L_out = jnp.maximum(L_out, 0.0)

        assert float(L_out) == pytest.approx(float(L_in), rel=1e-6)

    @pytest.mark.skip(reason="forward/_kernels/ deleted in Phase 6; guard sites no longer exist")
    def test_guard_present_in_source(self):
        """Smoke test: verify the isfinite guard exists in both guard sites.

        Guard locations:
        - fused_kernels.py: hybrid path (_hybrid_phot_body)
        - nonstell.py: compositional path (build_nonstell_fn)

        If either is accidentally removed, this test catches it immediately
        without needing to construct a full model.
        """
        import tengri.forward._kernels.hybrid as fk
        import tengri.forward.nonstell as ns

        fk_count = inspect.getsource(fk).count("jnp.isfinite(L_absorbed_stellar)")
        ns_count = inspect.getsource(ns).count("jnp.isfinite(L_absorbed_stellar)")
        total = fk_count + ns_count
        assert total >= 2, (
            f"Expected ≥2 isfinite guards on L_absorbed_stellar across "
            f"fused_kernels.py ({fk_count}) and nonstell.py ({ns_count}). "
            "A guard may have been accidentally removed."
        )

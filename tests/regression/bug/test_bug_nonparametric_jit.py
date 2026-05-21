"""Regression test for nonparametric.py JIT-safe bug.

Bug: nonparametric.py:74,210 — len(bin_edges_gyr) raises ConcretizationTypeError in JIT.
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestNonparametricJITSafe:
    """Bug: nonparametric.py:74,210 — len(bin_edges_gyr) not JIT-safe."""

    def test_continuity_sfh_jit(self):
        """continuity should JIT-compile with JAX array bin_edges."""
        from tengri.components.stellar.sfh.nonparametric import DEFAULT_BIN_EDGES_GYR, continuity

        age_yr = jnp.linspace(1e7, 13e9, 100)

        @jax.jit
        def _eval(edges):
            kwargs = {f"ratio_{i}": 0.0 for i in range(edges.shape[0] - 2)}
            return continuity(age_yr, log_total_mass=10.0, bin_edges_gyr=edges, **kwargs)

        sfr = _eval(DEFAULT_BIN_EDGES_GYR)
        assert sfr.shape == age_yr.shape
        assert jnp.all(jnp.isfinite(sfr))

    def test_dirichlet_sfh_jit(self):
        """dirichlet should JIT-compile with JAX array bin_edges."""
        from tengri.components.stellar.sfh.nonparametric import DEFAULT_BIN_EDGES_GYR, dirichlet

        age_yr = jnp.linspace(1e7, 13e9, 100)

        @jax.jit
        def _eval(edges):
            kwargs = {f"z_frac_{i}": 0.5 for i in range(edges.shape[0] - 2)}
            return dirichlet(age_yr, log_total_mass=10.0, bin_edges_gyr=edges, **kwargs)

        sfr = _eval(DEFAULT_BIN_EDGES_GYR)
        assert sfr.shape == age_yr.shape

    def test_continuity_sfh_piecewise_constant(self):
        """continuity should return piecewise-constant SFR (step function per Leja+2019)."""
        from tengri.components.stellar.sfh.nonparametric import continuity

        edges = jnp.array([0.0, 1.0, 5.0, 13.7])  # 3 bins in Gyr
        # Age points within the same bin should have identical SFR
        age_in_bin0 = jnp.array([0.1e9, 0.5e9, 0.9e9])  # all in [0, 1] Gyr bin
        age_in_bin1 = jnp.array([1.5e9, 2.0e9, 4.0e9])  # all in [1, 5] Gyr bin

        sfr_bin0 = continuity(
            age_in_bin0, log_total_mass=10.0, bin_edges_gyr=edges, ratio_0=0.5, ratio_1=0.0
        )
        sfr_bin1 = continuity(
            age_in_bin1, log_total_mass=10.0, bin_edges_gyr=edges, ratio_0=0.5, ratio_1=0.0
        )

        # Within each bin, SFR must be exactly constant
        assert jnp.allclose(sfr_bin0, sfr_bin0[0]), "SFR not constant within bin 0"
        assert jnp.allclose(sfr_bin1, sfr_bin1[0]), "SFR not constant within bin 1"

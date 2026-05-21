"""Regression test for unified.py torus_frac float equality safe bug.

Bug: unified.py:813 — jnp.where(agn_torus_frac == 0.5, ...) JIT-unsafe for traced values.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestUnifiedAGNTorusFrac:
    """Bug: unified.py:813 — float == comparison not JIT-safe."""

    def test_unified_agn_jit_safe(self):
        """unified_nlr_blr should JIT-compile without issues from float == comparison."""
        pytest.importorskip("tengri.components.agn.unified")
        import jax

        from tengri.components.agn.unified import unified_nlr_blr

        wave = jnp.logspace(2.5, 5.0, 200)

        @jax.jit
        def _eval(torus_frac):
            return unified_nlr_blr(
                wave,
                agn_log_lbol=12.0,
                agn_torus_frac=torus_frac,
                agn_theta_torus=60.0,
                agn_cos_inc=0.5,
                agn_log_mbh=8.0,
                agn_log_ledd=-1.0,
            )

        # Default sentinel value (0.5) — should activate geometric derivation
        l_nu_default = _eval(0.5)
        # Non-default value — should use the provided value
        l_nu_custom = _eval(0.3)

        assert jnp.all(jnp.isfinite(l_nu_default))
        assert jnp.all(jnp.isfinite(l_nu_custom))

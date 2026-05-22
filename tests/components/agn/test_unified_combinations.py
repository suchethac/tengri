"""Tests for unified_agn combiner across disc/torus model combinations."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.bounds


@pytest.fixture()
def wavelength():
    """Broad wavelength grid from radio (1 cm) to hard X-ray (1 A)."""
    return jnp.logspace(0, 8, 500)  # 1 A to 10^8 A (= 1 cm)


class TestUnifiedAgnCombinations:
    """Tests for the unified_agn combiner across disc/torus model combinations."""

    @pytest.mark.parametrize(
        "disc_model,torus_model",
        [
            ("powerlaw", "simple"),
            ("powerlaw", "two_temperature"),
            ("multicolor", "simple"),
            ("multicolor", "two_temperature"),
            ("adaf", "simple"),
        ],
    )
    def test_all_combinations_produce_finite_output(self, wavelength, disc_model, torus_model):
        """Every supported disc × torus combination gives finite, non-negative output."""
        from tengri.components.agn.unified import unified_agn

        l_nu = unified_agn(
            wavelength,
            agn_log_lbol=44.0,
            disc_model=disc_model,
            torus_model=torus_model,
        )
        assert jnp.all(jnp.isfinite(l_nu)), (
            f"Non-finite SED for disc={disc_model}, torus={torus_model}"
        )
        assert jnp.all(l_nu >= 0.0), f"Negative SED for disc={disc_model}, torus={torus_model}"
        chex.assert_equal_shape([l_nu, wavelength])

    def test_unknown_disc_raises(self, wavelength):
        """Unknown disc_model raises KeyError."""
        from tengri.components.agn.unified import unified_agn

        with pytest.raises(KeyError):
            unified_agn(wavelength, agn_log_lbol=44.0, disc_model="nonexistent")

    def test_unknown_torus_raises(self, wavelength):
        """Unknown torus_model raises KeyError."""
        from tengri.components.agn.unified import unified_agn

        with pytest.raises(KeyError):
            unified_agn(wavelength, agn_log_lbol=44.0, torus_model="nonexistent")

    def test_torus_dominated_sed_uv_suppressed(self, wavelength):
        """With torus_frac=1.0, IR dominates over UV by Wien suppression.

        The 1000 K torus has Wien tail at λ=5000 Å: x = hν/kT ≈ 28.8,
        giving B_nu ∝ exp(-28.8) ≈ 3e-13 relative to peak.  The disc
        contributes nothing (agn_frac=0).  So IR >> UV by a factor >>1000.
        """
        from tengri.components.agn.unified import unified_agn

        uv_mask = (wavelength > 500.0) & (wavelength < 5000.0)
        ir_mask = (wavelength > 2e4) & (wavelength < 1e6)
        l_nu = unified_agn(wavelength, agn_log_lbol=44.0, agn_torus_frac=1.0)
        ir_sum = float(jnp.sum(l_nu[ir_mask]))
        uv_sum = float(jnp.sum(l_nu[uv_mask]))
        assert ir_sum > 0.0, "IR sum is zero — torus emitted nothing"
        assert ir_sum > 1000.0 * uv_sum, (
            f"Expected IR >> UV (Wien suppression); got IR={ir_sum:.3e}, UV={uv_sum:.3e}"
        )

    def test_torus_frac_one_sets_disc_to_zero(self, wavelength):
        """agn_torus_frac=0.0 yields exactly the disc alone (no torus).

        Torus is called with agn_torus_frac=0, so simple_torus returns 0;
        unified_agn output equals disc-only.
        """
        from tengri.components.agn.disc import powerlaw_disc
        from tengri.components.agn.unified import unified_agn

        l_unified = unified_agn(wavelength, agn_log_lbol=44.0, agn_torus_frac=0.0)
        # Disc gets agn_frac = 1 - 0 = 1.0
        l_disc = powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_frac=1.0)
        np.testing.assert_allclose(
            np.array(l_unified),
            np.array(l_disc),
            rtol=1e-6,
            err_msg="unified_agn with torus_frac=0 should equal disc-only",
        )

    def test_jit_over_default_combination(self, wavelength):
        """The default powerlaw+simple combination is JIT-compilable."""
        from tengri.components.agn.unified import unified_agn

        @jax.jit
        def _run(wave):
            return unified_agn(wave, agn_log_lbol=44.0)

        chex.assert_tree_all_finite(_run(wavelength))

    def test_total_is_disc_plus_torus(self, wavelength):
        """unified_agn output equals disc + torus computed separately."""
        from tengri.components.agn.disc import powerlaw_disc
        from tengri.components.agn.torus import simple_torus
        from tengri.components.agn.unified import unified_agn

        frac = 0.3
        l_unified = unified_agn(wavelength, agn_log_lbol=44.0, agn_torus_frac=frac)
        l_disc = powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_frac=1.0 - frac)
        l_torus = simple_torus(wavelength, agn_log_lbol=44.0, agn_torus_frac=frac)
        np.testing.assert_allclose(
            np.array(l_unified),
            np.array(l_disc + l_torus),
            rtol=1e-6,
            err_msg="unified_agn should equal disc + torus",
        )

# SPDX-License-Identifier: BSD-3-Clause
"""Tests for unified_agn combiner across disc/torus model combinations."""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


def _real_silva04_grid_present() -> bool:
    """Distinguish the real Silva+04 grid from the synthetic zero-template
    fixture that ``tests/conftest.py`` creates for CI.

    The synthetic grid lets the orchestration code path execute, but its
    template is uniformly zero — so physics-level assertions (e.g. "torus
    dominates IR") can only be exercised with the real grid.
    """
    import h5py

    grid = Path(__file__).resolve().parents[2] / "data" / "silva04_torus_grid.h5"
    if not grid.is_file():
        return False
    try:
        with h5py.File(grid, "r") as f:
            template = f["silva04/template"][...]
    except (KeyError, OSError):
        return False
    return bool(np.any(template != 0.0))


@pytest.fixture()
def wavelength():
    """Broad wavelength grid from radio (1 cm) to hard X-ray (1 A)."""
    return jnp.logspace(0, 8, 500)  # 1 A to 10^8 A (= 1 cm)


class TestUnifiedAgnCombinations:
    """Tests for the unified_agn combiner across disc/torus model combinations."""

    @pytest.mark.parametrize(
        "disc_model,torus_model",
        [
            ("powerlaw", "silva04"),
            ("multicolor", "silva04"),
            ("adaf", "silva04"),
            ("powerlaw", "skirtor"),
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

    @pytest.mark.skipif(
        not _real_silva04_grid_present(),
        reason="needs real Silva+04 torus grid (synthetic fixture is zero-template)",
    )
    def test_torus_dominated_sed_uv_suppressed(self, wavelength):
        """With torus_frac=1.0, IR dominates over UV by Wien suppression.

        The 1000 K torus has Wien tail at λ=5000 Å: x = hν/kT ≈ 28.8,
        giving B_nu ∝ exp(-28.8) ≈ 3e-13 relative to peak.  The disc
        contributes nothing (agn_lum_ratio=0).  So IR >> UV by a factor >>1000.
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

        Torus is called with agn_torus_frac=0, so silva04_analytic returns 0;
        unified_agn output equals disc-only.
        """
        from tengri.components.agn.disc import powerlaw_disc
        from tengri.components.agn.unified import unified_agn

        l_unified = unified_agn(wavelength, agn_log_lbol=44.0, agn_torus_frac=0.0)
        # Disc gets agn_lum_ratio = 1 - 0 = 1.0
        l_disc = powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_lum_ratio=1.0)
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
        from tengri.components.agn.silva04 import silva04_analytic
        from tengri.components.agn.unified import unified_agn

        frac = 0.3
        l_unified = unified_agn(
            wavelength,
            agn_log_lbol=44.0,
            disc_model="powerlaw",
            torus_model="silva04",
            agn_torus_frac=frac,
        )
        l_disc = powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_lum_ratio=1.0 - frac)
        l_torus = silva04_analytic(wavelength, agn_log_lbol=44.0, agn_torus_frac=frac)
        np.testing.assert_allclose(
            np.array(l_unified),
            np.array(l_disc + l_torus),
            rtol=1e-6,
            err_msg="unified_agn should equal disc + torus",
        )

# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the composable AGN block subsystem.

Covers:
- registry sanity (categories + builtin none/grahsp/alternates)
- runner equivalence: all-grahsp recipe ≡ compute_grahsp_sed
- mix-and-match: GRAHSP BBB + simple two-T torus + SMC Prevot atten
- recipe validation: warnings on suspicious combos, ValueError on typos
- JIT compatibility
"""

from __future__ import annotations

import warnings

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds

from tengri.components.agn import AGN_MODELS, resolve_agn_model
from tengri.components.agn.blocks import (
    AGN_BLOCKS,
    BLOCK_CATEGORIES,
    RecipeWarning,
    composable_agn_l_nu,
    register_agn_block,
    resolve_agn_block,
    validate_block_recipe,
)
from tests._data_skip import requires_grahsp

# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────


def test_all_categories_have_none_block():
    for cat in BLOCK_CATEGORIES:
        assert "none" in AGN_BLOCKS[cat], f"{cat} missing 'none' block"


def test_grahsp_blocks_registered_for_all_categories():
    """GRAHSP must provide an impl for every pipeline stage."""
    expected = {
        "disc": "grahsp_sbpl",
        "nlr": "grahsp",
        "blr": "grahsp",
        "feii": "grahsp",
        "torus": "grahsp",
        "attenuation": "grahsp_biatten",
    }
    for cat, name in expected.items():
        assert name in AGN_BLOCKS[cat], f"{cat}/{name} not registered"


def test_at_least_one_alternate_per_category():
    """For mix-and-match to be meaningful, need >= 1 non-GRAHSP, non-none impl."""
    for cat in ("disc", "torus", "attenuation"):
        non_default = [n for n in AGN_BLOCKS[cat] if n != "none" and not n.startswith("grahsp")]
        assert non_default, f"{cat} has no non-GRAHSP alternate"


def test_composable_in_agn_models():
    assert "composable" in AGN_MODELS


def test_resolve_unknown_block_raises():
    with pytest.raises(ValueError, match="Unknown disc block"):
        resolve_agn_block("disc", "bogus_does_not_exist")


def test_register_duplicate_raises():
    """register_agn_block must reject duplicate (category, name) pairs."""
    with pytest.raises(ValueError, match="already registered"):

        @register_agn_block("disc", "grahsp_sbpl")
        def _redefine(*args, **kwargs):
            return None


# ──────────────────────────────────────────────────────────────────────
# Recipe validation: warnings for suspicious combos
# ──────────────────────────────────────────────────────────────────────


def test_validate_clean_recipe_emits_no_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RecipeWarning)
        issues = validate_block_recipe(
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
        )
    assert issues == []
    assert not any(issubclass(x.category, RecipeWarning) for x in w)


def test_validate_all_none_warns():
    with pytest.warns(RecipeWarning, match="every block selector is 'none'"):
        validate_block_recipe(
            agn_disc_block="none",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
        )


def test_validate_active_downstream_no_disc_warns():
    """Forgetting the disc but enabling nlr/blr/feii/torus must warn loudly."""
    with pytest.warns(RecipeWarning, match="agn_disc_block='none'"):
        validate_block_recipe(
            agn_disc_block="none",
            agn_nlr_block="grahsp",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="grahsp",
            agn_attenuation_block="none",
        )


def test_validate_unknown_block_raises_not_warns():
    """Typo in selector should be a hard error."""
    with pytest.raises(ValueError, match="Unknown disc block"):
        validate_block_recipe(
            agn_disc_block="grahps_sbpl",  # deliberate typo
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
        )


# ──────────────────────────────────────────────────────────────────────
# Runner: all-grahsp recipe equals compute_grahsp_sed
# ──────────────────────────────────────────────────────────────────────


def _grahsp_params():
    """Single-source GRAHSP parameter set used by both pipelines.

    ``agn_grahsp_l5100`` is **explicit** here. Without it, the two paths
    use different conventions:

    - ``compute_grahsp_sed`` (monolithic): rescales l5100 so the total
      AGN-side bolometric integral matches ``10**agn_log_lbol * L_sun``.
    - ``composable_agn_l_nu`` (block runner): each block self-normalises
      from its own params, with no post-hoc bolometric coupling.

    Setting ``agn_grahsp_l5100`` directly bypasses both the monolithic
    rescale and the composable disc auto-norm, so the two paths agree
    bit-for-bit.
    """
    return dict(
        agn_log_lbol=44.5,
        agn_grahsp_l5100=1.0e44,
        agn_grahsp_uvslope=0.0,
        agn_grahsp_plslope=-1.7,
        agn_grahsp_plbendloc_nm=100.0,
        agn_grahsp_plbendwidth=1.0,
        agn_grahsp_cutoff_nm=10000.0,
        agn_grahsp_a_lines=1.0,
        agn_grahsp_a_feii=5.0,
        agn_grahsp_linewidth_kms=5000.0,
        agn_grahsp_fcov=0.4,
        agn_grahsp_si=0.5,
        agn_grahsp_cool_lam_um=17.0,
        agn_grahsp_cool_width=0.45,
        agn_grahsp_hot_lam_um=2.0,
        agn_grahsp_hot_width=0.5,
        agn_grahsp_hot_fcov=1.0,
        agn_grahsp_ebv=0.05,
        agn_grahsp_ebv_agn=0.05,
        agn_type=1,
    )


@requires_grahsp
def test_all_grahsp_recipe_matches_compute_grahsp_sed():
    """Every selector = 'grahsp*' should reproduce compute_grahsp_sed exactly."""
    from tengri.components.agn.grahsp import compute_grahsp_sed

    wave_aa = jnp.logspace(2, 6, 400)
    p = _grahsp_params()

    out_composable = composable_agn_l_nu(
        wave_aa,
        agn_disc_block="grahsp_sbpl",
        agn_nlr_block="grahsp",
        agn_blr_block="grahsp",
        agn_feii_block="grahsp",
        agn_torus_block="grahsp",
        agn_attenuation_block="grahsp_biatten",
        **p,
    )
    out_monolithic = compute_grahsp_sed(wave_aa, **p)

    # Both paths exercise the same physics. The runner derives l5100_disc
    # via ``jnp.interp(5100Å, wave, L_λ_disc) × 5100`` from the disc grid;
    # the monolithic path uses the analytic l5100 directly. With explicit
    # ``agn_grahsp_l5100`` they agree to grid-interpolation precision.
    np.testing.assert_allclose(
        np.asarray(out_composable),
        np.asarray(out_monolithic),
        rtol=1e-3,
        atol=0.0,
    )


def test_mix_grahsp_disc_with_simple_torus():
    """GRAHSP BBB + simple greybody torus + SMC atten — runs and is finite."""
    wave_aa = jnp.logspace(2, 6, 400)
    out = composable_agn_l_nu(
        wave_aa,
        agn_disc_block="grahsp_sbpl",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="two_temperature",
        agn_attenuation_block="smc_prevot",
        agn_log_lbol=45.0,
        agn_grahsp_l5100=1.0e44,
        agn_T_hot=1200.0,
        agn_T_warm=300.0,
        agn_frac_hot=0.3,
        agn_torus_frac=0.5,
        agn_attenuation_ebv=0.2,
    )
    chex.assert_equal_shape([out, wave_aa])
    chex.assert_tree_all_finite(out)
    # Some flux must come through after attenuation.
    assert float(out.sum()) > 0


def test_disc_only_recipe_is_pure_continuum():
    """No nlr/blr/feii/torus → output is just the disc continuum (in L_nu)."""
    wave_aa = jnp.logspace(2, 5, 200)
    out = composable_agn_l_nu(
        wave_aa,
        agn_disc_block="grahsp_sbpl",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="none",
        agn_attenuation_block="none",
        agn_grahsp_l5100=1.0e44,
    )
    # Should be smooth (no line spikes) and positive.
    chex.assert_tree_all_finite(out)
    assert jnp.all(out > 0)


# ──────────────────────────────────────────────────────────────────────
# JIT
# ──────────────────────────────────────────────────────────────────────


@requires_grahsp
def test_runner_jit_compatible():
    """Selectors are static; param values are dynamic."""
    wave_aa = jnp.logspace(2, 6, 200)

    @jax.jit
    def fwd(l5100, ebv):
        return composable_agn_l_nu(
            wave_aa,
            agn_log_lbol=44.5,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_l5100=l5100,
            agn_grahsp_ebv=ebv,
        )

    out = fwd(jnp.array(1.0e44), jnp.array(0.1))
    chex.assert_equal_shape([out, wave_aa])
    chex.assert_tree_all_finite(out)


def test_resolve_via_agn_models_registry():
    """Going through resolve_agn_model('composable') must match direct call."""
    fn_via_registry = resolve_agn_model("composable")
    wave_aa = jnp.logspace(2, 6, 100)
    p = dict(
        agn_disc_block="grahsp_sbpl",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="grahsp",
        agn_attenuation_block="none",
        agn_grahsp_l5100=1.0e44,
    )
    out_registry = fn_via_registry(wave_aa, agn_log_lbol=44.5, **p)
    out_direct = composable_agn_l_nu(wave_aa, agn_log_lbol=44.5, **p)
    np.testing.assert_allclose(np.asarray(out_registry), np.asarray(out_direct), rtol=1e-12)

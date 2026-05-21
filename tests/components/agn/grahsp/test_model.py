"""Integration test: full GRAHSP AGN forward model on a sample wave grid."""

from __future__ import annotations

import chex
import pytest

pytestmark = pytest.mark.bounds

import jax.numpy as jnp
import numpy as np


def test_full_pipeline_runs():
    from tengri.components.agn.grahsp import GRAHSPParams, evaluate_grahsp_agn

    wave_nm = jnp.logspace(2, 5, 200)  # 100 nm to 100 um
    p = GRAHSPParams(l5100=1.0e44, ebv=0.1, ebv_agn=0.0)
    sed = evaluate_grahsp_agn(wave_nm, p)
    chex.assert_equal_shape([sed.bbb, wave_nm])
    chex.assert_tree_all_finite(sed.bbb)
    chex.assert_tree_all_finite(sed.torus)
    chex.assert_tree_all_finite(sed.bbb_attenuated)
    assert jnp.all(sed.bbb_attenuated <= sed.bbb + sed.broad_lines + sed.narrow_lines + sed.feii)
    assert sed.l_bol_bbb > 0
    assert sed.l_bol_torus > 0


def test_no_attenuation_recovers_intrinsic():
    from tengri.components.agn.grahsp import GRAHSPParams, evaluate_grahsp_agn

    wave_nm = jnp.logspace(2, 5, 200)
    p = GRAHSPParams(l5100=1.0e44, ebv=0.0, ebv_agn=0.0)
    sed = evaluate_grahsp_agn(wave_nm, p)
    intrinsic = sed.bbb + sed.broad_lines + sed.narrow_lines + sed.feii
    np.testing.assert_allclose(np.asarray(sed.bbb_attenuated), np.asarray(intrinsic), rtol=1e-12)


def test_higher_extinction_lowers_bbb():
    from tengri.components.agn.grahsp import GRAHSPParams, evaluate_grahsp_agn

    wave_nm = jnp.logspace(2, 4, 200)
    p_lo = GRAHSPParams(l5100=1.0e44, ebv=0.1, ebv_agn=0.0)
    p_hi = GRAHSPParams(l5100=1.0e44, ebv=0.5, ebv_agn=0.0)
    sed_lo = evaluate_grahsp_agn(wave_nm, p_lo)
    sed_hi = evaluate_grahsp_agn(wave_nm, p_hi)
    # AGN attenuated SED should be strictly lower under stronger extinction.
    assert jnp.all(sed_hi.bbb_attenuated <= sed_lo.bbb_attenuated + 1e-10)


def test_jit_full_pipeline():
    """End-to-end JIT compatibility (agn_type as static)."""
    import jax

    from tengri.components.agn.grahsp import (
        GRAHSPParams,
        evaluate_grahsp_agn,
        load_grahsp_templates,
    )

    templates = load_grahsp_templates()

    @jax.jit
    def fwd(lum, ebv):
        p = GRAHSPParams(l5100=lum, ebv=ebv)
        return evaluate_grahsp_agn(jnp.logspace(2, 5, 100), p, templates).bbb_attenuated

    out = fwd(1.0e44, 0.2)
    chex.assert_shape(out, (100,))
    chex.assert_tree_all_finite(out)

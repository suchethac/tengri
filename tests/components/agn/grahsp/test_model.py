# SPDX-License-Identifier: BSD-3-Clause
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


def test_monolith_lbol_is_the_accretion_luminosity_and_independent_of_fcov():
    """``agn_log_lbol`` normalizes the accretion (BBB) luminosity only.

    The torus re-radiates absorbed accretion light, so adding it to the normalized
    L_bol counts that light twice; the paper defines lumBolBBB (everything but the
    torus, above 91.2 nm) and lumBolTOR separately. The implied ``l5100`` must then
    not move with the torus covering factor.
    """
    import warnings

    from tengri.components.agn.grahsp import GRAHSPParams, evaluate_grahsp_agn
    from tengri.components.agn.grahsp.model import compute_grahsp_sed
    from tengri.utils.physics_constants import C_AA, L_SUN

    wave_aa = jnp.logspace(2.5, 6.0, 3000)
    wave_nm = wave_aa * 0.1
    l5100_by_fcov = []
    for fcov in (0.1, 0.9):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l_nu = np.asarray(compute_grahsp_sed(wave_aa, agn_log_lbol=45.0, agn_grahsp_fcov=fcov))
        unit = evaluate_grahsp_agn(wave_nm, GRAHSPParams(l5100=1.0, fcov=fcov))
        unit_l_nu = (
            np.asarray(unit.bbb_attenuated + unit.torus_attenuated) * 0.1 * wave_aa**2 / C_AA
        )
        l5100 = l_nu / unit_l_nu
        np.testing.assert_allclose(l5100, l5100[0], rtol=1e-9)  # pure rescale
        np.testing.assert_allclose(l5100[0] * float(unit.l_bol_bbb), 1e45 * L_SUN, rtol=1e-9)
        l5100_by_fcov.append(l5100[0])
    np.testing.assert_allclose(l5100_by_fcov[0], l5100_by_fcov[1], rtol=1e-12)

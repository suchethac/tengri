"""Tests for the composable GRAHSPSEDComponent adapter."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tengri.components.agn.grahsp import (
    GRAHSPSEDComponent,
    GRAHSPSEDComponentConfig,
)
from tengri.protocols.component import ForwardState


def _default_params():
    return {
        "agn_grahsp_l5100": jnp.array(1.0e44),
        "agn_grahsp_uvslope": jnp.array(0.0),
        "agn_grahsp_plslope": jnp.array(-1.7),
        "agn_grahsp_plbendloc_nm": jnp.array(100.0),
        "agn_grahsp_plbendwidth": jnp.array(1.0),
        "agn_grahsp_cutoff_nm": jnp.array(10000.0),
        "agn_grahsp_a_lines": jnp.array(1.0),
        "agn_grahsp_a_feii": jnp.array(5.0),
        "agn_grahsp_linewidth_kms": jnp.array(5000.0),
        "agn_grahsp_fcov": jnp.array(0.4),
        "agn_grahsp_si": jnp.array(0.0),
        "agn_grahsp_cool_lam_um": jnp.array(17.0),
        "agn_grahsp_cool_width": jnp.array(0.45),
        "agn_grahsp_hot_lam_um": jnp.array(2.0),
        "agn_grahsp_hot_width": jnp.array(0.5),
        "agn_grahsp_hot_fcov": jnp.array(1.0),
        "agn_grahsp_ebv": jnp.array(0.05),
        "agn_grahsp_ebv_agn": jnp.array(0.05),
    }


def test_apply_runs_end_to_end():
    component = GRAHSPSEDComponent()
    state = ForwardState(wave=jnp.logspace(3, 6, 200))  # 1000 Å to 1e6 Å
    out = component.apply(state, _default_params())
    assert out.sed_intrinsic.shape == state.wave.shape
    assert jnp.all(jnp.isfinite(out.sed_intrinsic))
    assert "L_agn_bol" in out.derived
    assert "L_agn_torus" in out.derived
    assert out.derived["L_agn_bol"] > 0
    assert out.derived["L_agn_torus"] > 0


def test_compose_disable_torus():
    """Disabling torus should reduce mid-IR SED to zero."""
    cfg = GRAHSPSEDComponentConfig(include_torus=False, include_si=False)
    component = GRAHSPSEDComponent(config=cfg)
    state = ForwardState(wave=jnp.logspace(4, 6, 200))  # 10000 Å to 1e6 Å (1-100 um)
    out = component.apply(state, _default_params())
    # In the mid-IR (~10 um) BBB+lines should be tiny; torus is the dominant
    # contributor when on. Without torus, sed_intrinsic must be much smaller.
    cfg_with = GRAHSPSEDComponentConfig(include_torus=True, include_si=True)
    out_with = GRAHSPSEDComponent(config=cfg_with).apply(state, _default_params())
    # At 5 um (50000 Å) the torus dominates by orders of magnitude.
    idx = int(np.argmin(np.abs(np.asarray(state.wave) - 5e4)))
    assert float(out.sed_intrinsic[idx]) < float(out_with.sed_intrinsic[idx]) / 5.0


def test_compose_disable_attenuation():
    cfg_no_atten = GRAHSPSEDComponentConfig(apply_attenuation=False)
    cfg_atten = GRAHSPSEDComponentConfig(apply_attenuation=True)
    state = ForwardState(wave=jnp.logspace(3, 5, 200))
    out_no = GRAHSPSEDComponent(config=cfg_no_atten).apply(state, _default_params())
    out_atten = GRAHSPSEDComponent(config=cfg_atten).apply(state, _default_params())
    # With attenuation, the SED in the UV must be lower.
    uv_idx = int(np.argmin(np.abs(np.asarray(state.wave) - 1500)))
    assert float(out_atten.sed_intrinsic[uv_idx]) < float(out_no.sed_intrinsic[uv_idx])


def test_compose_only_bbb():
    """A BBB-only configuration is useful for swapping in non-GRAHSP torus."""
    cfg = GRAHSPSEDComponentConfig(
        include_bbb=True,
        include_lines=False,
        include_feii=False,
        include_torus=False,
        include_si=False,
        apply_attenuation=False,
    )
    state = ForwardState(wave=jnp.logspace(3, 6, 200))
    out = GRAHSPSEDComponent(config=cfg).apply(state, _default_params())
    # Mid-IR contribution should be that of the BBB alone, not the torus.
    # At 12 um (120000 Å) the BBB is many orders of magnitude below the
    # torus-on case, so just check positive and finite.
    assert jnp.all(jnp.isfinite(out.sed_intrinsic))
    assert float(out.sed_intrinsic.max()) > 0
    assert out.derived["L_agn_torus"] == 0.0


def test_additive_to_existing_sed():
    component = GRAHSPSEDComponent()
    wave = jnp.logspace(3, 5, 200)
    base = jnp.full_like(wave, 1.0e30)
    state = ForwardState(wave=wave, sed_intrinsic=base)
    out = component.apply(state, _default_params())
    # Component is purely additive.
    assert jnp.all(out.sed_intrinsic >= base - 1e10)


def test_parameter_prefix_invariant():
    component = GRAHSPSEDComponent()
    for decl in component.declared_parameters():
        assert decl.name.startswith("agn_grahsp_") or decl.name in ("redshift",)


def test_publishes_l_agn_absorbed():
    """L_agn_absorbed must be non-negative and zero when attenuation is off."""
    component_atten = GRAHSPSEDComponent(config=GRAHSPSEDComponentConfig(apply_attenuation=True))
    component_no_atten = GRAHSPSEDComponent(
        config=GRAHSPSEDComponentConfig(apply_attenuation=False)
    )
    state = ForwardState(wave=jnp.logspace(3, 6, 200))
    out_atten = component_atten.apply(state, _default_params())
    out_no_atten = component_no_atten.apply(state, _default_params())
    # Attenuated: positive absorption; unattenuated: exactly zero.
    assert "L_agn_absorbed" in out_atten.derived
    assert float(out_atten.derived["L_agn_absorbed"]) > 0
    assert float(out_no_atten.derived["L_agn_absorbed"]) == 0.0


def test_jit_apply():
    import jax

    component = GRAHSPSEDComponent()
    state = ForwardState(wave=jnp.logspace(3, 5, 100))
    templates_state = component.precompute()

    # agn_type / toggles are static via the closure; params are dynamic.
    @jax.jit
    def fwd(p):
        return component.apply(state, p, templates_state).sed_intrinsic

    out = fwd(_default_params())
    assert out.shape == (100,)
    assert jnp.all(jnp.isfinite(out))

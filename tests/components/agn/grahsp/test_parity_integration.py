# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for the GRAHSP parity implementation (Balmer continuum, MN12
template torus, Veron-Cetty FeII, Netzer disc).

Verifies the four upstream-faithful variants flow through the full
``model`` / ``component`` / ``registry`` / ``precompute`` chain — not just
their isolated physics functions — so they survive inference and the
``WavePrecomp`` photometry path.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract


WAVE_AA = jnp.logspace(2.0, 6.0, 400)  # 100 Å -> 1e6 Å


def _compute(**kw):
    from tengri.components.agn.grahsp.model import compute_grahsp_sed

    return np.asarray(compute_grahsp_sed(WAVE_AA, agn_log_lbol=45.0, **kw))


def test_all_variants_finite_and_nonzero():
    base = _compute()
    chex.assert_tree_all_finite(base)
    assert (base > 0).any()
    for label, kw in [
        ("balmer", dict(agn_grahsp_a_bc=1.0)),
        ("mn12_torus", dict(torus_model="mn12")),
        ("mn12_tor_temp_pos", dict(torus_model="mn12", agn_grahsp_tor_temp=0.5)),
        ("mn12_tor_temp_neg", dict(torus_model="mn12", agn_grahsp_tor_temp=-0.5)),
        ("veroncetty", dict(feii_template="veroncetty2004")),
        ("netzer_disc", dict(disc_model="netzer")),
        ("netzer_disc_hi", dict(disc_model="netzer", disc_m="9.0", disc_a="0.998")),
    ]:
        out = _compute(**kw)
        chex.assert_tree_all_finite(out)
        assert (out > 0).any(), f"{label} produced an all-zero SED"


def test_balmer_adds_flux_below_edge():
    """Turning on the Balmer continuum must add flux blueward of 3646 Å."""
    off = _compute(agn_grahsp_a_bc=0.0)
    on = _compute(agn_grahsp_a_bc=2.0)
    below = np.asarray(WAVE_AA) < 3646.0
    # Net flux below the Balmer edge increases (the BB+truncation is positive).
    assert on[below].sum() > off[below].sum()


def test_veroncetty_differs_from_bruhweiler():
    """The two FeII templates must produce measurably different SEDs."""
    bv = _compute(feii_template="bruhweiler2008", agn_grahsp_a_feii=10.0)
    vc = _compute(feii_template="veroncetty2004", agn_grahsp_a_feii=10.0)
    assert not np.allclose(bv, vc, rtol=1e-3)


def test_registry_forwards_new_params():
    """The registered ``grahsp`` entry point must forward the new kwargs."""
    import warnings

    from tengri.components.agn.unified import resolve_agn_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn = resolve_agn_model("grahsp")
    assert callable(fn)
    out_default = np.asarray(fn(WAVE_AA, agn_log_lbol=45.0))
    out_mn12 = np.asarray(fn(WAVE_AA, agn_log_lbol=45.0, torus_model="mn12"))
    out_disc = np.asarray(fn(WAVE_AA, agn_log_lbol=45.0, disc_model="netzer"))
    chex.assert_tree_all_finite(out_mn12)
    chex.assert_tree_all_finite(out_disc)
    # Variant selection actually changes the SED.
    assert not np.allclose(out_default, out_mn12)
    assert not np.allclose(out_default, out_disc)


def test_grad_flows_through_new_params():
    """Gradients w.r.t. the new continuous params are finite."""
    from tengri.components.agn.grahsp.model import compute_grahsp_sed

    g_bc = jax.grad(
        lambda a: jnp.sum(compute_grahsp_sed(WAVE_AA, agn_grahsp_l5100=1e44, agn_grahsp_a_bc=a))
    )(0.5)
    g_tt = jax.grad(
        lambda t: jnp.sum(
            compute_grahsp_sed(
                WAVE_AA, agn_grahsp_l5100=1e44, torus_model="mn12", agn_grahsp_tor_temp=t
            )
        )
    )(0.3)
    assert np.isfinite(float(g_bc))
    assert np.any(float(g_bc) != 0.0), (
        "`float(g_bc)` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    assert np.isfinite(float(g_tt))
    assert np.any(float(g_tt) != 0.0), (
        "`float(g_tt)` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


def _toy_filters():
    # Two broad filters spanning optical and mid-IR (rest frame).
    w1 = np.linspace(4000.0, 7000.0, 200)
    t1 = np.exp(-0.5 * ((w1 - 5500.0) / 500.0) ** 2)
    w2 = np.linspace(40000.0, 200000.0, 300)  # ~4-20 um
    t2 = np.exp(-0.5 * ((w2 - 1.2e5) / 3e4) ** 2)
    return [w1, w2], [t1, t2]


def _precompute_lookup(torus_model):
    """Build the production precompute lookup for a torus model and evaluate it
    at the GRAHSP default (plslope=-1.7, ebv=0); returns per-filter L_nu."""
    from tengri.components.agn.grahsp.precompute import build_lookup, precompute

    fws, fts = _toy_filters()
    pre = precompute(
        filter_waves=fws,
        filter_trans=fts,
        redshift=0.0,
        plslope_grid=np.array([-1.7], dtype=np.float64),
        ebv_grid=np.array([0.0], dtype=np.float64),
        torus_model=torus_model,
    )
    fn = build_lookup(pre)
    # Default GRAHSP normalization scale is 1.0; grid axes are (plslope, ebv).
    return np.asarray(fn(jnp.array(1.0), jnp.array(-1.7), jnp.array(0.0)))


@pytest.mark.parametrize("torus_model", ["gaussian", "mn12"])
def test_precompute_lookup_finite_positive(torus_model):
    """The WavePrecomp lookup yields finite, positive photometry per variant.

    Guards the #629-class bug: a new torus variant must flow through the
    precompute path (not silently default / zero out).
    """
    phot = _precompute_lookup(torus_model)
    chex.assert_tree_all_finite(phot)
    assert (phot > 0).all(), f"{torus_model} precompute photometry not all-positive"


def test_precompute_torus_model_is_honored_in_midIR():
    """Selecting ``torus_model`` changes the precomputed mid-IR photometry more
    than the optical — i.e. the selector reaches the torus, not just a no-op.

    The optical band (filter 0) is BBB+line dominated and nearly identical
    between torus models; the mid-IR band (filter 1) is torus-dominated and
    must differ. This is convention-independent (a within-build comparison).
    """
    g = _precompute_lookup("gaussian")
    m = _precompute_lookup("mn12")
    optical_frac = abs(m[0] - g[0]) / g[0]
    midir_frac = abs(m[1] - g[1]) / g[1]
    assert midir_frac > optical_frac
    assert midir_frac > 0.05, "torus_model selection had negligible mid-IR effect"


@pytest.mark.parametrize("torus_model", ["gaussian", "mn12"])
def test_strong_negative_si_never_drives_torus_negative(torus_model):
    """A strong silicate-absorption fit must not make the total torus L_lambda
    negative — upstream's ``mask_negative`` clip is preserved at the component."""
    from tengri.components.agn.grahsp.model import GRAHSPParams, evaluate_grahsp_agn
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    wave = jnp.logspace(3.0, 5.5, 600)  # 1000 Å - ~30 um
    params = GRAHSPParams(l5100=1e44, si=-4.0, fcov=0.9, torus_model=torus_model)
    sed = evaluate_grahsp_agn(wave, params, load_grahsp_templates())
    total_torus = np.asarray(sed.torus + sed.si)
    assert (total_torus >= -1e-6 * np.abs(total_torus).max()).all()


def test_ratio_tor_bbb_and_frac_agn_tor():
    """The new bolometric diagnostics behave sensibly."""
    from tengri.components.agn.grahsp.bolometric import frac_agn_tor, ratio_tor_bbb

    assert float(ratio_tor_bbb(2.0, 4.0)) == pytest.approx(0.5)
    assert float(ratio_tor_bbb(1.0, 0.0)) == 0.0  # guarded denominator
    assert float(frac_agn_tor(3.0, 1.0)) == pytest.approx(0.75)
    assert float(frac_agn_tor(0.0, 0.0)) == 0.0


def test_precompute_mn12_differs_from_gaussian():
    """The two torus models must produce different precomputed photometry."""
    from tengri.components.agn.grahsp.precompute import build_lookup, precompute

    fws, fts = _toy_filters()
    kw = dict(
        filter_waves=fws,
        filter_trans=fts,
        redshift=0.0,
        plslope_grid=np.array([-1.7], dtype=np.float64),
        ebv_grid=np.array([0.0], dtype=np.float64),
    )
    g = np.asarray(
        build_lookup(precompute(torus_model="gaussian", **kw))(
            *[jnp.array(v) for v in (1.0, -1.7, 0.0)]
        )
    )
    m = np.asarray(
        build_lookup(precompute(torus_model="mn12", **kw))(
            *[jnp.array(v) for v in (1.0, -1.7, 0.0)]
        )
    )
    assert not np.allclose(g, m)

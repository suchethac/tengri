# SPDX-License-Identifier: BSD-3-Clause
"""The photometry LUT reddened the wrong stars, and extrapolated the screen (#1122).

Four bugs, each with its own guard here. Every accuracy assertion compares the LUT
against the **exact path** — never against another preintegral. Validating a
precompute against a preintegral is tautological: both sides share the machinery
under test, so it is structurally blind to any error they have in common. That is
exactly how the sigmoid bug below survived, and how a first draft of this fix
measured itself as ~40x better than it was.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, WavePrecomp
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "galex_nuv", "des_g", "des_r", "wise_w1"]
KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


def _build(ssp, approx, tau_diff=0.5, tau_bc=1.0, z=0.2):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(BANDS)),
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_diff": tau_diff,
            "tau_bc": tau_bc,
        },
        redshift=Fixed(z),
        approx=approx,
    )


def _lut_vs_exact(ssp, approx, **kw):
    """max |LUT / exact - 1| over the bands. ONE param dict, shared."""
    m_exact = _build(ssp, None, **kw)
    p = dict(m_exact.spec.sample(KEY))
    exact = np.asarray(m_exact.predict_photometry(p))
    lut = np.asarray(_build(ssp, approx, **kw).predict_photometry(p))
    return float(np.max(np.abs(lut / exact - 1.0)))


# ── 1. the birth cloud must reddened the SAME stars on both paths ──────────────


def test_young_indicator_matches_the_exact_screen():
    """The LUT's young indicator must be the logistic the exact screen uses.

    It was ``1 / (1 + 10**u)`` while ``two_component_dust`` used
    ``jax.nn.sigmoid``. Since 10^u = e^(u*ln10), the LUT's birth-cloud transition
    was 2.3x sharper: a different stellar population sat behind the birth cloud in
    the fast path. An AGE-domain error, so no wavelength refinement could see it.
    """
    from tengri.components.dust.two_component import _young_indicator

    ages = np.logspace(6.0, 10.1, 64)
    t_birth, width = 1e7, 0.3

    got = np.asarray(_young_indicator(jnp.asarray(ages), t_birth, width))
    want = np.asarray(jax.nn.sigmoid(-(jnp.log10(ages) - jnp.log10(t_birth)) / width))
    np.testing.assert_allclose(got, want, rtol=1e-12)

    # Non-vacuity: the OLD base-10 spelling must be visibly different, or this
    # test would pass against the bug it exists to catch.
    old = 1.0 / (1.0 + 10.0 ** ((np.log10(ages) - np.log10(t_birth)) / width))
    assert np.max(np.abs(old - want)) > 0.1


# ── 2. the screen is EVALUATED, not extrapolated ───────────────────────────────


def test_quadrature_beats_taylor_against_the_exact_path(ssp):
    """K=5 must be materially closer to the exact path than the Taylor form."""
    taylor = _lut_vs_exact(ssp, WavePrecomp(n_subbands=0, taylor_correction=True))
    quad = _lut_vs_exact(ssp, WavePrecomp(n_subbands=5))
    assert quad < taylor, f"quadrature {quad:.2%} not better than Taylor {taylor:.2%}"
    assert quad < 0.01, f"K=5 vs exact = {quad:.2%}, expected < 1%"


def test_quadrature_converges_with_k(ssp):
    """Error must fall with K. If it does not, the nodes or the partition are wrong.

    A first draft dropped the fractional interval at each sub-band edge, so the
    sub-integrals no longer summed to the whole and the error GREW with K.
    """
    errs = [_lut_vs_exact(ssp, WavePrecomp(n_subbands=k)) for k in (1, 3, 8)]
    assert errs[0] > errs[1] > errs[2], f"not converging in K: {errs}"


def test_subband_partition_conserves_flux(ssp):
    """Sum over sub-bands must reproduce the full band integral exactly."""
    m = _build(ssp, WavePrecomp(n_subbands=5))
    p = dict(m.spec.sample(KEY))
    st = m.predict_state(p)
    per_age = np.asarray(st.derived["stellar_phot_lnu_per_age_precomp"])
    sub = np.asarray(st.derived["stellar_phot_lnu_per_age_subband_precomp"])
    np.testing.assert_allclose(sub.sum(axis=-1), per_age, rtol=1e-10)


def test_quadrature_nodes_are_inside_their_band_and_positive(ssp):
    """A zero node would be finite in the forward pass and inf in the gradient.

    Where a template has no flux in a sub-band its weight is zero, so the node
    cannot change the result — but it is still fed through the dust law, which
    goes as 1/lambda.
    """
    m = _build(ssp, WavePrecomp(n_subbands=5))
    st = m.predict_state(dict(m.spec.sample(KEY)))
    nodes = np.asarray(st.derived["stellar_subband_waves_rest_precomp"])
    assert np.all(np.isfinite(nodes))
    assert np.all(nodes > 0.0)
    assert np.all(np.diff(nodes, axis=-1) >= 0.0), "nodes not ordered within a band"


def test_gradient_is_finite_through_the_quadrature(ssp):
    m = _build(ssp, WavePrecomp(n_subbands=5))
    p = dict(m.spec.sample(KEY))
    g = assert_grad_matches_fd(lambda q: jnp.sum(m.predict_photometry(q)), p)
    for v in jax.tree.leaves(g):
        assert bool(jnp.all(jnp.isfinite(v)))
        assert jnp.any(v != 0.0), (
            "`v` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )


# ── 3. n_subbands must color the compiled-kernel cache ────────────────────────


def test_n_subbands_changes_the_compiled_kernel(ssp):
    """Two models differing ONLY in K must not share a compiled kernel.

    ``approx`` flags are filtered with ``isinstance(v, bool)`` and n_subbands is an
    int, so it was dropped from the signature: the second model silently reused the
    first's kernel. Only bites when one process builds both — i.e. a benchmark or a
    sweep, never a single fit, which is why it hid.
    """
    m3 = _build(ssp, WavePrecomp(n_subbands=3))
    m8 = _build(ssp, WavePrecomp(n_subbands=8))
    assert m3.compile_signature() != m8.compile_signature()

    # BIT-identity, not allclose: a leaked cache entry returns the *same floats*.
    # At a converged config K=3 and K=8 agree to well inside allclose's default
    # tolerance, so allclose cannot tell a shared kernel from a converged one.
    # Use a high-tau, high-z config where K genuinely bites, and demand the two
    # differ at all.
    m3 = _build(ssp, WavePrecomp(n_subbands=3), tau_diff=2.0, tau_bc=3.0, z=1.0)
    m8 = _build(ssp, WavePrecomp(n_subbands=8), tau_diff=2.0, tau_bc=3.0, z=1.0)
    p = dict(m3.spec.sample(KEY))
    f3 = np.asarray(m3.predict_photometry(p))
    f8 = np.asarray(m8.predict_photometry(p))
    assert not np.array_equal(f3, f8), (
        "K=3 and K=8 returned bit-identical photometry — they are sharing a "
        "compiled kernel, i.e. n_subbands has fallen out of the cache key again"
    )


# ── 4. the single-component screen must not silently lose its correction ───────


def test_single_component_screen_is_also_quadratured(ssp):
    """``taylor_correction`` now defaults off, so an unwired screen would fall back
    to the bare A(lam_eff) form and be WORSE than before the fix."""
    kw = {"ssp_data": ssp, "observation": Observation(photometry=Photometry.from_names(BANDS))}
    m_exact = SEDModel.build(
        **kw,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.4)},
        redshift=Fixed(0.1),
        approx=None,
    )
    m_lut = SEDModel.build(
        **kw,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.4)},
        redshift=Fixed(0.1),
        approx=WavePrecomp(),
    )
    p = dict(m_exact.spec.sample(KEY))
    st = m_lut.predict_state(p)
    assert st.derived.get("dust_attenuation_subband_precomp") is not None, (
        "single-component screen published no sub-band tensors"
    )
    exact = np.asarray(m_exact.predict_photometry(p))
    lut = np.asarray(m_lut.predict_photometry(p))
    assert float(np.max(np.abs(lut / exact - 1.0))) < 5e-3


# ── 5. the free-z (ztable) path must carry the quadrature too ─────────────────


def test_free_z_path_publishes_the_subband_tensors(ssp):
    """A free redshift must not silently fall back to the Taylor extrapolation."""
    from tengri import Uniform

    m = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(BANDS)),
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.5,
            "tau_bc": 1.0,
        },
        redshift=Uniform(0.01, 1.5),
        approx=WavePrecomp(n_z=40),
    )
    p = dict(m.spec.sample(KEY))
    p["redshift"] = 0.3
    st = m.predict_state(p)
    sub = st.derived.get("stellar_phot_lnu_per_age_subband_precomp")
    assert sub is not None, "free-z path published no sub-band tensors"
    assert np.asarray(sub).shape[-1] == 5
    assert st.derived.get("dust_bc_attenuation_subband_precomp") is not None


def test_ztable_disk_cache_key_includes_n_subbands(ssp):
    """A cached K=0 table must not be reused for a K=5 model.

    The ztable is cached to DISK by content hash. n_subbands changes the table's
    content, so it must change the hash — otherwise the quadrature silently
    no-ops across processes, persistently, which is worse than the in-process
    kernel-cache leak.
    """
    from tengri.components.stellar.sps.precompute import _ztable_cache_key
    from tengri.observation.filters import load_filter
    from tengri.utils.filter_convention import FilterConvention

    fs = [load_filter(b) for b in BANDS[:2]]
    args = (
        ssp,
        [f.wave for f in fs],
        [f.trans for f in fs],
        np.linspace(0.01, 1.5, 8),
        False,
        False,
        FilterConvention.BESSELL,
    )
    assert _ztable_cache_key(*args, 0) != _ztable_cache_key(*args, 5)
    assert _ztable_cache_key(*args, 3) != _ztable_cache_key(*args, 5)

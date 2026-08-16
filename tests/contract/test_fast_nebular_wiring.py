# SPDX-License-Identifier: BSD-3-Clause
"""Contract: SEDModel.enable_fast_nebular wires the per-Q_H grid into the live
forward so photometry AND line fluxes reconstruct without the Cue forward (#950).

Validated end-to-end:
  * the grid actually attaches to the chain nebular component (no silent no-op);
  * predict_photometry(fast) and predict_line_fluxes(fast) match the exact model;
  * the fast model gets its own compile_signature (no kernel-cache color-leak);
  * predict_spectrum is guarded (a fast model would omit the nebular continuum);
  * gas-phase (neb_logZ_gas) and stellar (met_logzsol) metallicity are SEPARATE
    grid axes;
  * the fast joint objective is jit + grad safe;
  * the nion decoupling (Q_H from the ionizing slice) is bit-exact with the
    full-SED integral — the enabler for pruning the stellar SED under WavePrecomp.

Data-gated (needs a bare SSP + cue_weights.npz); skips in CI.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    FIXED,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.components.nebular.component import NebularSEDComponent
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.contract

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_BANDS = ["galex_fuv", "galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1", "wise_w2"]
_LINES = ("Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717")
_LINE_DATA = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
_LW = _LINE_DATA.wavelengths
Z = 0.15


def _require():
    if not Path(_BARE).is_file():
        pytest.skip(f"missing bare SSP {_BARE}")
    if not Path("data/cue_weights.npz").is_file():
        pytest.skip("Cue weights (data/cue_weights.npz) not present")


def _build(neb, *, dust_on=True, precomp=True):
    import warnings

    _require()
    ssp = load_ssp_data(_BARE)
    obs = Observation(photometry=Photometry.from_names(_BANDS), line_fluxes=_LINE_DATA)
    taus = (0.25, 0.4) if dust_on else (0.0, 0.0)
    dust = {
        "type": "two_component",
        "law_bc": "calzetti",
        "*": FIXED,
        "tau_diff": Fixed(taus[0]),
        "tau_bc": Fixed(taus[1]),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": FREE},
            dust=dust,
            neb=neb,
            redshift=Fixed(Z),
            approx=WavePrecomp() if precomp else None,
        )


_CUE = {"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0)}


def test_enable_attaches_grid_no_silent_noop():
    m = _build(_CUE)
    m.enable_fast_nebular(_LW, n_grid=12)
    neb = [c for c in m._cached_component_chain if isinstance(c, NebularSEDComponent)]
    assert neb, "no NebularSEDComponent in the chain"
    assert all(c.grid_table is not None for c in neb), "grid_table not attached (silent no-op)"
    assert m._nebular_grid_table is not None


def test_fast_photometry_and_lines_match_exact():
    m_exact = _build(_CUE)
    m_fast = _build(_CUE)
    m_fast.enable_fast_nebular(_LW, n_grid=16)
    wp = wl = 0.0
    worst = None
    for i in range(8):
        p = dict(m_exact.spec.sample(jax.random.PRNGKey(600 + i)))
        pe = np.asarray(m_exact.predict_photometry(p))
        pf = np.asarray(m_fast.predict_photometry(p))
        rel = np.abs(pf - pe) / (np.abs(pe) + 1e-40)
        if rel.max() > wp:
            wp = rel.max()
            worst = (i, int(rel.argmax()), pe, pf, rel, p)
        le = np.asarray(m_exact.predict_line_fluxes(p, target_wavelengths=_LW, redden=True))
        lf = np.asarray(m_fast.predict_line_fluxes(p, target_wavelengths=_LW, redden=True))
        strong = np.abs(le) > 1e-3 * np.max(np.abs(le))
        wl = max(wl, np.max(np.abs(lf - le)[strong] / (np.abs(le)[strong] + 1e-40)))
    assert wp < 3e-2, _photometry_report(wp, worst)
    assert wl < 5e-2, f"fast line fluxes off by {wl:.2e}"


def _photometry_report(wp, worst):
    """Turn the bare max-relative-error into something #1154 can be worked from.

    ``fast photometry off by 2.09e-01`` names no band, no draw and no flux scale, so a
    CI log cannot distinguish "the fast path is wrong" from "this band carries almost
    no light and a near-zero denominator inflated a harmless difference" (the #1134
    trap). Since the test is xfailed on linux and XPASSes on arm64, the log is the only
    window anyone gets onto the platform split — so print the whole table: the offending
    band and draw, both predictions, and each band's share of the des_r flux.
    """
    i, b, pe, pf, rel, p = worst
    r = abs(pe[_BANDS.index("des_r")]) + 1e-300
    out = [
        f"fast photometry off by {wp:.2e} (tolerance 3e-2)  [#1154]",
        f"  worst: band {_BANDS[b]!r} on prior draw {i}",
        f"    exact = {pe[b]:.6e}   fast = {pf[b]:.6e}   rel = {rel[b]:.3%}",
        f"    that band carries {abs(pe[b]) / r:.2e} x the des_r flux",
        f"  params: { {k: round(float(v), 6) for k, v in sorted(p.items())} }",
        f"    {'band':>10} {'exact':>13} {'fast':>13} {'rel':>9} {'/des_r':>9}",
    ]
    for k, band in enumerate(_BANDS):
        out.append(
            f"    {band:>10} {pe[k]:13.5e} {pf[k]:13.5e} {rel[k]:9.3%} {abs(pe[k]) / r:9.2e}"
        )
    return "\n".join(out)


def test_signature_differs_fast_vs_exact():
    """A fast model must not share a compiled kernel slot with the exact one."""
    m_exact = _build(_CUE)
    m_fast = _build(_CUE)
    m_fast.enable_fast_nebular(_LW, n_grid=8)
    assert m_exact.compile_signature() != m_fast.compile_signature()


def test_predict_spectrum_on_a_fast_model_matches_the_exact_model():
    """A fast model must still return the exact spectrum, not refuse (#950 -> #1673).

    ``predict_spectrum`` refused here from #950, because the fast path zeroed
    ``nebular_sed`` and the spectrum came back missing the nebular continuum and
    lines. #1673 fixed the cause instead: ``predict_state`` materializes the
    nebular component, so a rich consumer reads a complete forward state while
    ``predict_photometry`` keeps reconstructing from the grid.

    Asserting equality rather than a raise is the stronger statement -- a refusal
    passes whether or not the values are right, and would turn red the moment the
    values became available, which is exactly what happened here. Measured
    bit-exact (rel 0.0) against ``approx=None`` when the fix landed.
    """
    wave = np.linspace(4000.0, 7000.0, 50)
    p = dict(_build(_CUE).spec.sample(jax.random.PRNGKey(0)))

    exact = np.asarray(_build(_CUE).predict_spectrum(p, wave_obs=wave), dtype=np.float64)
    m_fast = _build(_CUE)
    m_fast.enable_fast_nebular(_LW, n_grid=8)
    got = np.asarray(m_fast.predict_spectrum(p, wave_obs=wave), dtype=np.float64)

    assert np.isfinite(got).all(), "fast-model spectrum is not finite"
    rel = np.max(np.abs(got - exact) / np.maximum(np.abs(exact), np.abs(exact).max() * 1e-30))
    assert rel < 1e-10, (
        f"fast-model spectrum moved {rel:.3e} from the exact path -- the nebular "
        "continuum is not reaching predict_spectrum (#950/#1673)."
    )


def test_gas_and_stellar_metallicity_are_separate_axes():
    """neb_logZ_gas (gas) and met_logzsol (stellar) are independent grid axes,
    reconstructed correctly even when they differ strongly."""
    neb = {"type": "cue", "*": FIXED, "logU": Fixed(-2.5), "logZ_gas": Uniform(-1.0, 0.4)}
    m_fast = _build(neb)
    m_fast.enable_fast_nebular(_LW, n_grid=10)
    axes = m_fast._nebular_grid_table.axis_names
    assert "met_logzsol" in axes and "neb_logZ_gas" in axes, axes
    m_exact = _build(neb)
    p = dict(m_exact.spec.sample(jax.random.PRNGKey(11)))
    p["met_logzsol"] = jnp.asarray(-1.5)  # metal-poor stars
    p["neb_logZ_gas"] = jnp.asarray(0.3)  # metal-rich gas — decoupled
    le = np.asarray(m_exact.predict_line_fluxes(p, target_wavelengths=_LW, redden=True))
    lf = np.asarray(m_fast.predict_line_fluxes(p, target_wavelengths=_LW, redden=True))
    strong = np.abs(le) > 1e-3 * np.max(np.abs(le))
    rel = np.max(np.abs(lf - le)[strong] / (np.abs(le)[strong] + 1e-40))
    assert rel < 6e-2, f"decoupled met/gasZ reconstruction off by {rel:.2e}"


def test_fast_joint_objective_jit_and_grad_safe():
    """The fast forward is JIT + gradient safe from ONE shared predict_state —
    the HMC hot path."""
    m = _build(_CUE)
    m.enable_fast_nebular(_LW, n_grid=8)
    p = dict(m.spec.sample(jax.random.PRNGKey(3)))
    fr = list(m.spec.free_params)
    fp = {k: p[k] for k in fr}
    fx = {k: v for k, v in p.items() if k not in fr}

    def obj(q):
        pp = {**fx, **q}
        state = m.predict_state(pp)
        phot = m.observation.predict_via_precomp(state, pp)
        lines = m.predict_line_fluxes(pp, target_wavelengths=_LW, state=state)
        return jnp.sum(phot["phot_fnu"]) + jnp.sum(lines)

    val = jax.jit(obj)(fp)
    grad = jax.jit(jax.grad(obj))(fp)
    assert np.isfinite(float(val))
    assert all(np.all(np.isfinite(np.asarray(g))) for g in grad.values())


def test_fast_line_fluxes_jit_safe_without_state():
    """predict_line_fluxes on a fast model is jit+grad safe with NO state passed.

    The standalone path (no feature_state from the joint loss) routes through the
    SED-free ``_compute_nion``. Regression: ``compute_nion`` recomputed the
    ionizing-bin count via ``int(jnp.sum(wave < ...))`` — a ConcretizationTypeError
    under jit when ``ssp_wave`` is a traced input — instead of the static
    build-time ``_state.n_ion_bins``. All the other fast tests pass a shared state
    (state.derived['nion']), so only this exercises the standalone jit path."""
    m = _build(_CUE)
    m.enable_fast_nebular(_LW, n_grid=8)
    p = dict(m.spec.sample(jax.random.PRNGKey(1)))
    fr = list(m.spec.free_params)
    fp = {k: p[k] for k in fr}
    fx = {k: v for k, v in p.items() if k not in fr}

    def line_sum(q):
        return jnp.sum(m.predict_line_fluxes({**fx, **q}, target_wavelengths=_LW))

    g = jax.jit(jax.grad(line_sum))(fp)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in g.values())


def test_nion_decoupling_is_bit_exact_with_full_sed():
    """predict_state's nion (integrated over the ionizing SLICE) is bit-exact with
    the full-SED integral. Guards the decoupling that lets the WavePrecomp LUT
    prune the stellar SED — an INDEPENDENT check (the compute_nion parity test now
    compares slice-vs-slice, so it can no longer catch a slice/full divergence)."""
    from tengri.components.stellar.component import _integrate_nion

    # no dust, no nebular add -> the final sed_intrinsic IS the stellar SED, so a
    # full-grid integral is a clean independent reference for the published nion.
    m = _build({"type": "none"}, dust_on=False, precomp=False)
    p = dict(m.spec.sample(jax.random.PRNGKey(5)))
    st = m.predict_state(p)
    nion_published = float(np.sum(np.asarray(st.derived["nion"])))
    nion_full = float(_integrate_nion(jnp.asarray(st.sed_intrinsic), jnp.asarray(st.wave)))
    assert abs(nion_published - nion_full) <= 1e-6 * abs(nion_full), (
        f"sliced nion {nion_published:.6e} != full-SED nion {nion_full:.6e}"
    )

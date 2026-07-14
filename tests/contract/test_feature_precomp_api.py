# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``approx=FeaturePrecomp()`` serves emission lines from a build-time LUT.

The emission-line precompute existed but no fit could reach it. The likelihood called
``measure_line_fluxes(params, defs, state=...)`` and never passed ``fast=True``, so the
baked-in window LUT was unreachable from any fit; and it computed ``predict_state`` —
the full-grid forward — whenever a line channel was active, which is precisely the cost
the precompute exists to skip. Meanwhile the Cue grid could only be switched on by the
imperative ``enable_fast_nebular()``, with no way to *declare* a fast-line model under
the ``SEDModel.build`` grammar.

The load-bearing test here is therefore not "does it run" but
``test_the_likelihood_actually_reaches_the_fast_line_path``: it compares compiled FLOPs of
the real fit objective with and without the precompute. A fast path the likelihood cannot
reach is a no-op, and a no-op is what this whole seam shipped as.

Structural checks (raises, flags, FLOP guards, dust-off identity) run anywhere. Physics
parity with dust ON is gated to the real SSP: CI injects a coarse synthetic grid (#613) on
which the window LUT's age-resolved window-center dust approximation is badly inaccurate,
so a parity assertion there would fail for reasons that have nothing to do with this code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    FIXED,
    FREE,
    FeaturePrecomp,
    Fixed,
    Observation,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp,
)
from tengri.inference.fitter import Fitter
from tengri.inference.loss_functions import build_loss_fn
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_list import LineList
from tengri.observation.line_measurement import default_line_defs
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.contract

BANDS = ["des_g", "des_r", "des_z", "wise_w1", "wise_w2"]
LINES = ["OII_3726", "Hbeta", "OIII_5007", "Halpha", "SII_6717"]
Z = 0.1

# A star-forming SFH. A declining 9 Gyr history puts the Balmer lines into stellar
# ABSORPTION (Halpha ~ -7e-20), and every relative-error assertion taken against a
# flux that is ~zero passes or fails for reasons unrelated to the code under test.
TRUTH = {
    "sfh_dpl_log_total_mass": 10.5,
    "sfh_dpl_age_gyr": 12.0,
    "sfh_dpl_tau_gyr": 11.0,
    "sfh_dpl_alpha": 1.0,
    "sfh_dpl_beta": 3.0,
    "met_logzsol": 0.0,
    "dust_tau_bc": 1.0,
    "dust_tau_diff": 0.4,
    "neb_logU": -2.8,
    "neb_logZ_gas": -0.2,
}


def _line_waves():
    cat = LineList.default_optical()
    return jnp.asarray([float(w) for n, w in zip(cat.names, cat.wavelengths) if n in LINES])


def _line_data(fluxes=None):
    waves = _line_waves()
    f = jnp.ones(len(LINES)) if fluxes is None else jnp.asarray(fluxes)
    return LineFluxData(
        names=tuple(LINES), fluxes=f, errors=jnp.abs(f) * 0.1 + 1e-30, wavelengths=waves
    )


def _build(ssp, *, cue, approx, line_data=None, emission=True, dust=True):
    kw = {}
    if cue:
        kw["neb"] = {
            "type": "cue",
            "*": FIXED,
            "logU": Uniform(-4.0, -1.0),
            "logZ_gas": Uniform(-1.5, 0.3),
        }
    if dust:
        d = {
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "tau_bc": Uniform(0.0, 4.0),
            "tau_diff": Uniform(0.0, 3.0),
        }
        if emission:
            d["emission"] = {"type": "dale2014", "*": FIXED}
        kw["dust"] = d
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(
            photometry=Photometry.from_names(BANDS),
            line_fluxes=_line_data() if line_data is None else line_data,
        ),
        redshift=Fixed(Z),
        sfh={"type": "dpl", "*": FREE},
        stellar={"met_logzsol": Uniform(-1.5, 0.3)},
        approx=approx,
        **kw,
    )


def _params(m, **override):
    p = {k: jnp.asarray(float(v)) for k, v in m.spec.get_fixed_values().items()}
    for k in m.spec.free_params:
        p[k] = jnp.asarray(float(TRUTH[k]))
    p.update({k: jnp.asarray(float(v)) for k, v in override.items()})
    return p


def _loss_grad_flops(m):
    """Compiled FLOPs of one gradient of the REAL fit objective — what a step pays."""
    n = len(BANDS)
    f = Fitter(m, jnp.ones(n), jnp.ones(n))
    loss_fn = build_loss_fn(f)
    data_args = dict(f._data_args)
    init = f._initialize_unbounded(jax.random.PRNGKey(0))
    g = jax.jit(jax.grad(lambda p: loss_fn(p, data_args)))
    return g.lower(init).compile().cost_analysis()["flops"]


# ── the bug: the likelihood could not reach the fast path ──────────────────


def test_the_likelihood_actually_reaches_the_fast_line_path(synthetic_ssp_wide):
    """A fast path the fit cannot reach is a no-op — which is what this shipped as.

    Compares the compiled cost of the real objective's gradient with and without
    ``FeaturePrecomp``. Asserted on FLOPs, not wall clock, so it cannot flake on a
    loaded machine; and against the *same* model otherwise, so it cannot go stale.
    """
    ssp = synthetic_ssp_wide
    slow = _loss_grad_flops(_build(ssp, cue=False, approx=WavePrecomp()))
    fast = _loss_grad_flops(_build(ssp, cue=False, approx=(WavePrecomp(), FeaturePrecomp())))
    assert fast < slow / 5, (
        f"the likelihood is not using the line precompute: {fast:,.0f} FLOPs with "
        f"FeaturePrecomp vs {slow:,.0f} without — expected a large drop, since the "
        f"window LUT exists to skip the full-grid forward"
    )


def test_feature_precomp_sets_the_baked_in_measurement_flag(synthetic_ssp_wide):
    """Baked-in has no line catalog, so it must route to the window LUT."""
    m = _build(synthetic_ssp_wide, cue=False, approx=FeaturePrecomp())
    assert m._fast_line_measurement is True
    assert not m._has_line_catalog()


def test_no_precomp_leaves_the_exact_path_alone(synthetic_ssp_wide):
    """Without the opt-in, nothing changes — the approximation never self-activates."""
    m = _build(synthetic_ssp_wide, cue=False, approx=WavePrecomp())
    assert m._fast_line_measurement is False


# ── the API surface ────────────────────────────────────────────────────────


def test_feature_precomp_composes_with_wave_precomp(synthetic_ssp_wide):
    """The line LUT and the photometry LUT are independent and stack."""
    m = _build(synthetic_ssp_wide, cue=False, approx=(WavePrecomp(), FeaturePrecomp()))
    assert m._approx_config_feature is not None
    assert m._approx["wave_precomp"] is True
    assert m._fast_line_measurement is True


def test_unknown_approx_member_still_raises(synthetic_ssp_wide):
    with pytest.raises(TypeError, match="not a legal value"):
        _build(synthetic_ssp_wide, cue=False, approx="feature_precomp")


def test_two_feature_precomps_raise(synthetic_ssp_wide):
    with pytest.raises(TypeError, match="at most one of each"):
        _build(synthetic_ssp_wide, cue=False, approx=(FeaturePrecomp(), FeaturePrecomp()))


def test_feature_precomp_without_lines_names_the_problem(synthetic_ssp_wide):
    """No lines to tabulate is a build error, not a silent no-op."""
    with pytest.raises(ValueError, match="no emission lines to tabulate"):
        SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=Observation(photometry=Photometry.from_names(BANDS)),
            redshift=Fixed(Z),
            sfh={"type": "dpl", "*": FREE},
            approx=FeaturePrecomp(),
        )


def test_explicit_lines_override_the_observation(synthetic_ssp_wide):
    m = _build(
        synthetic_ssp_wide,
        cue=False,
        approx=FeaturePrecomp(lines=(4862.68, 6564.61)),
    )
    assert m._fast_line_measurement is True
    assert m._feature_precomp_lines.shape[0] == 2


# ── the guard: dust IR admitted for lines, still barred from indices ───────


def test_dust_ir_is_admitted_for_lines(synthetic_ssp_wide):
    """A dust-IR component must not block the line LUT.

    Its emission is a *smooth* continuum common to the line window and its sidebands,
    so it cancels in the continuum subtraction: the measured bias on the ten DESI
    optical lines is < 1e-7 even at tau_bc=4, tau_diff=3, where the IR term already
    contributes 3% of the continuum *level*. Before this, a chain with dust emission
    raised — which is every realistic model.
    """
    m = _build(synthetic_ssp_wide, cue=False, approx=FeaturePrecomp(), emission=True)
    out = m.measure_line_fluxes(
        _params(m), default_line_defs(np.asarray(_line_waves())), fast=True
    )
    assert np.all(np.isfinite(np.asarray(out)))


def test_dust_ir_is_still_barred_from_spectral_indices(synthetic_ssp_wide):
    """A break index is a flux RATIO — a smooth additive offset does not cancel there.

    The asymmetry with lines is physical, not an oversight: without a sideband
    subtraction there is nothing for the IR continuum to cancel against.
    """
    from tengri.observation.spectral_indices import STANDARD_INDICES

    m = _build(synthetic_ssp_wide, cue=False, approx=FeaturePrecomp(), emission=True)
    # Dn4000 is a break: the ratio of two band fluxes, with no sideband subtraction
    # for a smooth IR continuum to cancel against.
    with pytest.raises(ValueError, match="predict_spectral_indices"):
        m.predict_spectral_indices(_params(m), (STANDARD_INDICES["Dn4000"],), fast=True)


# ── physics parity (real SSP only — the synthetic grid is too coarse) ──────


@pytest.mark.usefixtures("real_ssp_only")
def test_fast_lines_match_the_exact_path_cue():
    """Cue's per-Q_H grid must reproduce the exact Cue forward."""
    ssp = load_ssp("fsps_prsc_miles_chabrier")  # bare-stellar: a wNE grid double-counts
    waves = _line_waves()
    m_ex = _build(ssp, cue=True, approx=None)
    ref = np.asarray(m_ex.predict_line_fluxes(_params(m_ex), target_wavelengths=waves))
    assert ref[LINES.index("Halpha")] > 0, "Halpha must be in emission or the test is vacuous"

    m_fa = _build(ssp, cue=True, approx=(WavePrecomp(), FeaturePrecomp(n_grid=8)))
    got = np.asarray(m_fa.predict_line_fluxes(_params(m_fa), target_wavelengths=waves))

    rel = np.abs(got - ref) / np.abs(ref)
    assert rel.max() < 0.02, f"Cue grid drifted {rel.max():.2%} from the exact forward"


@pytest.mark.usefixtures("real_ssp_only")
def test_fast_lines_match_the_exact_path_baked_in():
    """The window LUT must reproduce the measured lines.

    Restricted to lines that are actually EMITTING and not negligible. In this SSP
    [NII] comes out near zero and in absorption (~1% of Halpha), and a relative error
    against ~zero is a ratio of two noise floors — the same trap as quoting a huge
    percentage on a sub-Lyman band whose flux is 1e-7 of the optical.
    """
    ssp = load_ssp()  # wNE: lines baked into the templates
    defs = default_line_defs(np.asarray(_line_waves()))
    m_ex = _build(ssp, cue=False, approx=None)
    ref = np.asarray(m_ex.measure_line_fluxes(_params(m_ex), defs))

    m_fa = _build(ssp, cue=False, approx=(WavePrecomp(), FeaturePrecomp()))
    got = np.asarray(m_fa.measure_line_fluxes(_params(m_fa), defs, fast=True))

    strong = np.abs(ref) > 0.05 * np.abs(ref[LINES.index("Halpha")])
    assert strong.sum() >= 3, "too few emitting lines to make this a real test"
    rel = np.abs(got[strong] - ref[strong]) / np.abs(ref[strong])
    assert rel.max() < 0.02, f"window LUT drifted {rel.max():.2%} on the emitting lines"


@pytest.mark.usefixtures("real_ssp_only")
def test_the_dust_ir_term_does_not_change_the_measured_lines():
    """The bound the relaxed guard rests on, asserted rather than asserted-about.

    If admitting dust IR into the window LUT ever starts biasing the measured line
    fluxes, this fails — rather than the bias quietly propagating into a catalog.
    """
    ssp = load_ssp()
    defs = default_line_defs(np.asarray(_line_waves()))
    m_on = _build(ssp, cue=False, approx=None, emission=True)
    m_off = _build(ssp, cue=False, approx=None, emission=False)

    for tau_bc, tau_diff in [(1.0, 0.4), (4.0, 3.0)]:
        kw = {"dust_tau_bc": tau_bc, "dust_tau_diff": tau_diff}
        a = np.asarray(m_on.measure_line_fluxes(_params(m_on, **kw), defs))
        b = np.asarray(m_off.measure_line_fluxes(_params(m_off, **kw), defs))
        rel = np.abs(a - b) / np.abs(a)
        assert rel.max() < 1e-5, (
            f"dust IR shifted the measured line fluxes by {rel.max():.2e} at "
            f"tau_bc={tau_bc}, tau_diff={tau_diff} — the continuum subtraction is no "
            f"longer canceling it, so the window LUT must not admit dust emission"
        )

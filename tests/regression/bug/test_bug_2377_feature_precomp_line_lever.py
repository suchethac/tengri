# SPDX-License-Identifier: BSD-3-Clause
r"""Cue ``FeaturePrecomp`` attachment has one lever, not two (#2377).

``FeaturePrecomp`` has two independent levers. A Cue backend's per-Q_H grid serves
photometry (disarmed by dust), while the SSP-window LUT serves line fluxes (unaffected by
dust). Four call sites in ``fitter.py`` used to check ``fast_nebular_can_engage`` in
isolation, which checks only the photometry lever. So dusty Cue fits were offered (and
auto-attached) a line LUT that measured 1.00x in compiled FLOPs — a second compiled kernel
for nothing.

**The root cause:** a Cue backend does not run the SSP-window branch of
``_resolve_feature_precomp``, so there is no ``_fast_line_measurement`` flag and no line
lever at all. The per-Q_H grid serves the photometry half, and that half is disarmed only
when the dust cannot take its nebular from the grid. Since #3267 a dusty chain with no IR
emission can take it (two-component dust with no ``dust_emission``), so a dusty Cue
line-flux fit is no longer a grid worth 0 compiled FLOPs. Measured on this file's
fixture (3 bands and 3 line fluxes, gradient FLOPs of the fit objective, float64):

==========================  ====================  ====================  ========
model                       ``WavePrecomp``       ``+FeaturePrecomp``   ratio
==========================  ====================  ====================  ========
dust-free                   54,103,816            2,241,922             **24.13x**
two-component, no IR        61,318,708            12,859,558            **4.77x**
WG00, shape free (control)  54,128,280            54,128,280            **1.00x**
==========================  ====================  ====================  ========

The explicit pair is what the grid buys. The ``auto`` policy still refuses the top-up on
the two-component, no-IR row (``test_the_top_up_is_refused_on_a_dusty_cue_line_fit``):
``fast_nebular_can_engage`` reads the model before ``FeaturePrecomp`` is configured, when
the grid's dust flag is not yet set, so it reads False there. That refusal is a known gap
(a missed 4.77x), not a justified zero; widening ``auto`` is a later PR. The WG00 row is
the control: where the grid cannot engage the two objectives are equal to the digit.

Before #3267 the two-component row measured 1.00x (58,497,272 both arms), and the grid's
build cost (7.2 s ``enable_fast_nebular``, an in-process re-trace) bought nothing. That
was the grid disarmed by dust, not the line channel.

This is not a regression to undo. Pre-#2377 a dusty Cue line-flux fit paid:

- 7.2 s for the ``enable_fast_nebular`` grid build.
- A changed ``compile_signature()`` despite no compiled-graph benefit.
- Silently zero gradient FLOPs improvement.

The guard ``feature_lut_serves_line_channel`` asks "does the SSP-window-LUT branch
run?" (true only when the backend is not Q_H-linear). On Cue it returns False, so the
line lever does not exist. Callers must ask ``fast_nebular_can_engage`` to decide
photometry-only leverage.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    FeaturePrecomp,
    Fitter,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.inference.fitter import _resolve_batch_fit_approx
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.regression_bug

_LINES = ("Halpha", "Hbeta", "OIII_5007")
_WAVES = jnp.array([6564.61, 4862.71, 5008.24])


def _model(ssp, obs, approx, *, dust_attenuation: bool, wg00: bool = False):
    """Build a Cue SED model with optional dust.

    ``wg00=True`` swaps the two-component screen for WG00 with its shape free: a
    dust that has no stellar energy-balance LUT, so the per-Q_H grid cannot take its
    nebular and stays disarmed (the control arm of the compiled-graph test).
    """
    dust_block = (
        {
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": 0.0,
        }
        if dust_attenuation
        else {"type": "none"}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if wg00:
            dust_block = {"type": "wg00", "all_params": FREE}
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(9.0, 11.0),
                "tau_gyr": 1.0,
                "age_gyr": 5.0,
            },
            dust_attenuation=dust_block,
            neb={"type": "cue", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
            approx=approx,
        )


def _line_flux_setup(ssp, *, dust_attenuation: bool):
    """A photometry + 3-line-flux observation with data at truth."""
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    base = _model(ssp, obs, WavePrecomp(), dust_attenuation=dust_attenuation)
    truth = base.spec.sample(jax.random.PRNGKey(0))
    phot = np.asarray(base.predict_photometry(truth))
    lf = np.asarray(base.measure_line_fluxes(truth, approx=False))[:3]
    obs = Observation(
        photometry=obs.photometry,
        line_fluxes=LineFluxData(
            names=_LINES,
            fluxes=jnp.asarray(lf),
            errors=jnp.asarray(np.abs(lf) * 0.05 + 1e-30),
            wavelengths=_WAVES,
        ),
    )
    return obs, phot


def _fitter(model, phot, **kw):
    """Construct a Fitter, suppressing warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Fitter(
            model,
            data=phot,
            noise=0.05 * np.abs(phot) + 1e-31,
            data_type="photometry",
            **kw,
        )


def _objective_flops(fitter) -> int:
    """Gradient FLOPs of the real fit objective, not of ``predict_photometry``."""
    from tengri.inference.context import InferenceContext

    ctx = InferenceContext.from_target(fitter)
    nlp, args = ctx.neg_log_posterior_fn, ctx.data_args
    p0 = ctx.initial_params(jax.random.PRNGKey(1))
    return int(
        jax.jit(jax.grad(lambda p: nlp(p, args))).lower(p0).compile().cost_analysis()["flops"]
    )


def _arm(ssp, obs, phot, policy, *, dust_attenuation: bool, wg00: bool = False):
    """One benchmark arm, set at the surface it is measured on.

    The ``approx=`` that governs a FIT is the fit-time one: ``Fitter`` defaults to
    ``"auto"`` and re-resolves whatever the model was built with, so two models
    differing only in their BUILD-time ``approx=`` are one configuration wearing two
    labels. That is the defect this file's sibling PR fixed in notebook 10, and it
    reappeared in the first draft of this file -- every arm read 1.00x because every
    arm was the same graph.

    Returns ``(flops, feature_precomp_resolved, fits_lines)`` so a caller can prove
    the arms differ before it compares their cost.
    """
    model = _model(ssp, obs, None, dust_attenuation=dust_attenuation, wg00=wg00)
    fitter = _fitter(model, phot, approx=policy)
    state = getattr(fitter.model, "approx", None)
    return (
        _objective_flops(fitter),
        bool(state is not None and getattr(state, "feature_precomp", False)),
        "line_flux_waves" in fitter._data_args,
    )


def test_setup_the_control_is_really_dust_free(ssp_data_fsps):
    """Guard the guard: if both arms carry dust, every ratio below is meaningless."""
    from tengri.forward.sed_model import _nebular_continuum_consumers

    obs, _ = _line_flux_setup(ssp_data_fsps, dust_attenuation=True)
    dusty = _model(ssp_data_fsps, obs, None, dust_attenuation=True)
    clean = _model(ssp_data_fsps, obs, None, dust_attenuation=False)

    dusty_consumers = _nebular_continuum_consumers(dusty._build_component_chain())
    clean_consumers = _nebular_continuum_consumers(clean._build_component_chain())

    assert dusty_consumers, "the dusty arm has no sed_nebular consumer — it is not dusty"
    assert not clean_consumers, (
        f"the control still consumes sed_nebular ({[type(c).__name__ for c in clean_consumers]}), "
        "so it is not a control. Omitting dust= builds a dust component; pass "
        "dust={'type': 'none'}."
    )


def test_feature_precomp_still_pays_on_a_dust_free_cue_line_fit(ssp_data_fsps):
    """The lever must still work where it is legitimately available.

    Asserted before the dusty cases, so a fix that disabled the LUT everywhere
    cannot pass this file.
    """
    obs, phot = _line_flux_setup(ssp_data_fsps, dust_attenuation=False)

    wave, wave_feat, wave_lines = _arm(
        ssp_data_fsps, obs, phot, WavePrecomp(), dust_attenuation=False
    )
    pair, pair_feat, pair_lines = _arm(
        ssp_data_fsps, obs, phot, (WavePrecomp(), FeaturePrecomp()), dust_attenuation=False
    )

    # Prove the arms are two configurations before comparing their cost. Without
    # this, an equal-FLOPs reading is uninterpretable: it is what a benchmark
    # measuring one configuration twice looks like.
    assert wave_lines and pair_lines, "the line-flux channel never reached the objective"
    assert not wave_feat, "the WavePrecomp arm resolved WITH FeaturePrecomp"
    assert pair_feat, "the pair arm resolved WITHOUT FeaturePrecomp"

    # The dust-free Cue model: per-Q_H grid engages in photometry, and the line
    # fluxes benefit from the faster state. Measured 17.15x on the real SSP.
    assert pair < wave / 5.0, (
        f"FeaturePrecomp buys only {wave / pair:.2f}x on a dust-free Cue line-flux "
        f"fit ({wave:,} -> {pair:,} gradient FLOPs of the objective). Measured at "
        "17.15x when #2377 landed; the refusal added there is conditioned on dust, "
        "so if this collapses the predicate has over-reached."
    )
    print(f"Dust-free Cue line-flux fit: {wave:,} -> {pair:,} FLOPs ({wave / pair:.2f}x speedup)")


def _feature_attached(model) -> bool:
    """The one read of "FeaturePrecomp is attached": ``model.approx.feature_precomp``."""
    state = getattr(model, "approx", None)
    return bool(state is not None and getattr(state, "feature_precomp", False))


@pytest.mark.parametrize(
    "build_approx", [None, WavePrecomp()], ids=["build-None", "build-WavePrecomp"]
)
def test_the_top_up_is_refused_on_a_dusty_cue_line_fit(ssp_data_fsps, build_approx):
    """``auto`` keeps refusing the top-up on a dusty no-IR Cue line-flux fit (a known gap).

    Pinned on both build paths. ``build_approx=None`` followed by ``Fitter`` with the
    default ``approx="auto"`` is the default path; ``WavePrecomp()`` is the explicit-build
    path. Both must refuse. The refusal is no longer justified by the 1.00x measured in
    #2377: the explicit pair buys 4.77x here (see the compiled-graph test). It is a gap
    because ``fast_nebular_can_engage`` reads the model before ``FeaturePrecomp`` is
    configured, and the dusty grid flag is only set once it is. Closing it is a later PR,
    and this test flips then (#3267).
    """
    obs, phot = _line_flux_setup(ssp_data_fsps, dust_attenuation=True)

    resolved = _fitter(_model(ssp_data_fsps, obs, build_approx, dust_attenuation=True), phot).model

    assert not _feature_attached(resolved), (
        "the resolver now attaches FeaturePrecomp to a dusty no-IR Cue line-flux fit. "
        "That closes the known gap (#3267): flip this pin and the premise in the "
        "compiled-graph test together."
    )


def test_positive_control_the_attached_read_can_be_true(ssp_data_fsps):
    """The read used above must be able to report True, or the refusal pins prove nothing.

    Two cases read through the same ``_feature_attached``: a dust-free Cue line-flux fit
    under ``auto`` (attaches, measured 24x), and an explicitly built dusty pair (attaches
    by construction). If either reads False, the refusal assertions are vacuous.
    """
    obs_clean, phot_clean = _line_flux_setup(ssp_data_fsps, dust_attenuation=False)
    clean_auto = _fitter(
        _model(ssp_data_fsps, obs_clean, WavePrecomp(), dust_attenuation=False), phot_clean
    ).model
    assert _feature_attached(clean_auto), (
        "positive control failed: auto did not attach FeaturePrecomp to a dust-free Cue "
        "line-flux fit, so _feature_attached cannot be trusted to read True"
    )

    obs, _phot = _line_flux_setup(ssp_data_fsps, dust_attenuation=True)
    dusty_pair = _model(
        ssp_data_fsps, obs, (WavePrecomp(), FeaturePrecomp()), dust_attenuation=True
    )
    assert _feature_attached(dusty_pair), (
        "positive control failed: an explicitly built pair does not read as attached"
    )


def test_the_grid_pays_on_dusty_no_ir_cue_and_equality_holds_where_disarmed(ssp_data_fsps):
    """The ENGAGEMENT half, measured on the objective (not the photometry alone).

    Dusty no-IR two-component Cue: the explicit pair pays 4.77x in gradient FLOPs (the
    photometry half, served from the per-Q_H grid since #3267). Control: WG00 with a
    free shape, where the dust has no stellar LUT and the grid stays disarmed, the two
    objectives are equal. If the control stops being equal, the grid engages where it
    should not; if the ratio collapses, the grid stopped serving.
    """
    obs, phot = _line_flux_setup(ssp_data_fsps, dust_attenuation=True)

    # Force the grid explicitly and check it measures 1.00x. Both arms are set at
    # the FIT surface, and the resolutions are asserted to differ, so the equality
    # below is a statement about the compiled graph rather than about the harness.
    explicit, explicit_feat, explicit_lines = _arm(
        ssp_data_fsps, obs, phot, (WavePrecomp(), FeaturePrecomp()), dust_attenuation=True
    )
    auto, auto_feat, auto_lines = _arm(
        ssp_data_fsps, obs, phot, WavePrecomp(), dust_attenuation=True
    )

    assert explicit_lines and auto_lines, "the line-flux channel never reached the objective"
    assert explicit_feat, "the explicit arm resolved WITHOUT FeaturePrecomp"
    assert not auto_feat, "the WavePrecomp arm resolved WITH FeaturePrecomp"

    assert explicit * 4 <= auto, (
        f"explicit FeaturePrecomp on a dusty no-IR Cue line-flux fit buys only "
        f"{auto / explicit:.2f}x ({auto:,} -> {explicit:,} gradient FLOPs of the "
        "objective); measured 4.77x under #3267. If the grid no longer serves this "
        "photometry, revisit the top-up predicate (#2377)."
    )
    print(f"Dusty no-IR Cue line-flux fit: {auto:,} -> {explicit:,} FLOPs")

    # Control: WG00 (free shape, no stellar LUT) disarms the grid, so the pair and the
    # build-time WavePrecomp must compile to the same objective.
    wg_wave, wg_feat_w, wg_lines_w = _arm(
        ssp_data_fsps, obs, phot, WavePrecomp(), dust_attenuation=True, wg00=True
    )
    wg_pair, wg_feat_p, wg_lines_p = _arm(
        ssp_data_fsps,
        obs,
        phot,
        (WavePrecomp(), FeaturePrecomp()),
        dust_attenuation=True,
        wg00=True,
    )
    assert wg_lines_w and wg_lines_p, "the line-flux channel never reached the objective"
    assert wg_feat_p and not wg_feat_w, "the WG00 arms did not resolve as two configurations"
    assert wg_pair == wg_wave, (
        f"on WG00 (grid disarmed) FeaturePrecomp changed the objective's gradient FLOPs: "
        f"{wg_wave:,} vs {wg_pair:,}. The grid engaged where it cannot serve (#3267)."
    )


def test_the_batch_surface_refuses_it_too(ssp_data_fsps):
    """The two surfaces must reach the same verdict on the same model."""
    obs, _phot = _line_flux_setup(ssp_data_fsps, dust_attenuation=True)
    model = _model(ssp_data_fsps, obs, None, dust_attenuation=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = _resolve_batch_fit_approx(model, "auto", "photometry")

    state = getattr(batch, "approx", None)
    attached = state is not None and getattr(state, "feature_precomp", False)
    assert not attached, (
        "the batch resolver attached FeaturePrecomp to a dusty Cue model, "
        "where the single-galaxy resolver refused it. The two surfaces must agree "
        "on the same model (#2377)."
    )


def test_no_warning_promises_a_speed_up_that_measures_one_x(ssp_data_fsps):
    """A Cue line-flux fit should not warn about a non-existent LUT remedy.

    The warning ``_warn_lines_without_lut`` was added to point out a 21x speedup.
    On a Cue model the line lever does not exist, so the warning would point at a
    remedy that cannot be applied.
    """
    obs, phot = _line_flux_setup(ssp_data_fsps, dust_attenuation=True)
    model = _model(ssp_data_fsps, obs, WavePrecomp(), dust_attenuation=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _fitter(model, phot)
        # Filter to the specific warning about lines without LUT
        lut_warnings = [
            wi
            for wi in w
            if issubclass(wi.category, UserWarning) and "without FeaturePrecomp" in str(wi.message)
        ]
    assert not lut_warnings, (
        f"expected no warning about lines without FeaturePrecomp on a Cue model, "
        f"but got {len(lut_warnings)}: {[str(wi.message) for wi in lut_warnings]}"
    )

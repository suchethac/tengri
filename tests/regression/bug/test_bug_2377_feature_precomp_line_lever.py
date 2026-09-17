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
lever at all. The per-Q_H grid serves photometry only, and that lever is disarmed by dust.
A decision "dusty Cue fits should get the line LUT" is "the dusty Cue fits should get a
grid worth 0 compiled FLOPs".

**Measured on a Cue model with 3 bands and 3 line fluxes, gradient FLOPs of the fit
objective:**

==========  ====================  ====================  ========
dust        ``WavePrecomp``       ``+FeaturePrecomp``   ratio
==========  ====================  ====================  ========
none        50,664,616            2,954,374             **17.15x**
yes         58,497,272            58,497,272            **1.00x**
==========  ====================  ====================  ========

The dust-free Cue model shows the per-Q_H grid's line channel benefit (17.15x). Dust
disarms the grid's photometry lever, leaving only an inert line attachment. The grid is
built and changes ``compile_signature()`` despite producing byte-identical compiled HLO
(measured same SHA-256 as exact path), costing a 7.2 s ``enable_fast_nebular`` build
and an in-process re-trace.

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


def _model(ssp, obs, approx, *, dust_attenuation: bool):
    """Build a Cue SED model with optional dust."""
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


def _arm(ssp, obs, phot, policy, *, dust_attenuation: bool):
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
    model = _model(ssp, obs, None, dust_attenuation=dust_attenuation)
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


def test_the_top_up_is_refused_on_a_dusty_cue_line_fit(ssp_data_fsps):
    """The resolver must refuse the grid on dusty Cue, where it measures 1.00x.

    This is the assertion that #2377 added: a Cue model has only one lever
    (photometry, disarmed by dust), and the line channel should not get offered
    a grid with zero compiled FLOPs.
    """
    obs, phot = _line_flux_setup(ssp_data_fsps, dust_attenuation=True)

    # Resolve under the default "auto" policy
    resolved = _fitter(_model(ssp_data_fsps, obs, None, dust_attenuation=True), phot).model

    state = getattr(resolved, "approx", None)
    attached = state is not None and getattr(state, "feature_precomp", False)
    assert not attached, (
        "the resolver attached FeaturePrecomp to a dusty Cue model's line-flux fit, "
        "where it measures 1.00x in compiled FLOPs — a second compiled kernel for "
        "nothing (#2377)."
    )


def test_the_refusal_is_justified_by_the_compiled_graph(ssp_data_fsps):
    """The ENGAGEMENT half: on dusty Cue the grid brings exactly zero compiled FLOPs.

    This is the objective measurement that makes the refusal in the previous test
    more than an arbitrary rule. If this ever stops being equal, the refusal must
    be revisited.
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

    assert explicit == auto, (
        f"appending FeaturePrecomp to a dusty Cue line-flux fit changed the "
        f"objective's compiled gradient FLOPs: {auto:,} (auto) vs {explicit:,} "
        f"(explicit). If the line lever became available for a chain that reads "
        "sed_nebular, that is a real change. Measure and update the guards that "
        "refuse the top-up (#2377)."
    )
    print(f"Dusty Cue line-flux fit: {auto:,} FLOPs (exactly equal with/without FeaturePrecomp)")


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

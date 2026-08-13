# SPDX-License-Identifier: BSD-3-Clause
r"""``FeaturePrecomp`` must not be attached to a model it cannot help (#1748).

Since #1281 the fast nebular grid may serve photometry only when nothing downstream
reads ``sed_nebular``, because serving it requires zeroing the continuum and the dust
energy balance reads it to size the absorbed budget. ``DustSEDComponent`` declares it
as an input, so **any model with dust disarms that channel** — and for months three
warnings, two shipped resolvers and ``CLAUDE.md`` went on advertising a ~21x line
speedup that was measured at **1.00x, bit-identical compiled FLOPs**.

**``FeaturePrecomp`` has two consumers, and this file tests both** (#1770). The
nebular continuum shortcut above is one. The other is the line channel: the LUT
flips ``needs_state`` in ``loss_functions``, so the likelihood stops rebuilding the
full-grid SED to read a handful of line fluxes (#1477). Dust does not disarm that —
measured on the fit objective of a **dusty** model, 1.08x (Cue) and 7.19x (baked-in),
against a *photometry* gradient on the same model that is bit-identical. #1760 gated
both consumers on the continuum predicate and silently withdrew the second; every
test here stayed green, because every fixture was photometry-only. **A guard is only
as wide as the channels its fixtures carry.**

Note the two thresholds: exact equality proves a config never reached the compiled
graph, and is the right assertion for the photometry case. 8 % is not equality — a
thin margin still means the config engaged. Asserting "inert" where the truth is
"thin" is how the line consumer got written off.

The #1281 correctness fix is not a regression to undo. On the pre-#1281 tree the fast
pair's photometry for a dusty model differed from exact by **0.41 %** against
0.0115 % for a dust-free control — the shortcut bought a biased answer, and a
constant forward bias enters the gradient multiplied by SNR (#1671). What was wrong
was the advertising.

**Why FLOPs and not wall clock.** A timing guard on this would be folded away: XLA
already proved it can compile the "exact" arm down to the fast one and make a
wall-clock guard pass on the cost it was checking (#1696). FLOPs read off
``compile().cost_analysis()`` are a property of the compiled graph, so exact equality
is unambiguous evidence that a config never reached it.

**The dust-free control is not optional.** Without it, "the ratio is 1.00x" is equally
consistent with "the LUT is inert" and "this fixture has nothing to tabulate", and the
second reading would let the defect through.
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug

_PARAMS_BASE = {"sfh_delayed_log_total_mass": 10.0}


_LINES = ("Halpha", "Hbeta", "OIII_5007")
_LINE_WAVES = (6564.61, 4862.71, 5008.24)


def _observation(*, lines: bool):
    """Photometry, optionally plus a measured line-flux channel.

    The channel is the whole point of the ``lines=True`` arm: ``FeaturePrecomp``'s
    second consumer only exists when the loss has line fluxes to serve, so a
    photometry-only fixture cannot see it — which is exactly how #1760 shipped past
    the rest of this file.
    """
    import warnings

    from tengri import Observation, Photometry
    from tengri.observation.line_flux_data import LineFluxData

    phot = Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])
    if not lines:
        return Observation(photometry=phot)
    with warnings.catch_warnings():
        # Observation(line_fluxes=...) is deprecated in favor of Data(lines=...)
        # (#1321); the channel-presence contract under test is unchanged by that.
        warnings.simplefilter("ignore", DeprecationWarning)
        return Observation(
            photometry=phot,
            line_fluxes=LineFluxData(
                names=_LINES,
                fluxes=jnp.asarray([1e-16, 3e-17, 5e-17]),
                errors=jnp.asarray([1e-17, 3e-18, 5e-18]),
                wavelengths=jnp.asarray(_LINE_WAVES),
            ),
        )


def _build(ssp, approx, *, dust: bool, lines: bool = False, neb: str = "cue"):
    """Build a model. ``neb`` selects which ``FeaturePrecomp`` ROUTE the model takes.

    ``'cue'`` publishes a Q_H-linear catalog and takes the ``'grid'`` route; ``'none'``
    on a baked-in/wNE SSP leaves the lines inside the templates and takes the
    ``'window'`` route. The two have different engagement conditions, so a fixture
    that only ever exercises one cannot see a gate that conflates them.
    """
    from tengri import FIXED, Fixed, SEDModel, Uniform

    dust_group = (
        {
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_diff": Uniform(0.0, 1.5),
            "tau_bc": 0.0,
        }
        if dust
        # Omitting `dust=` still builds a DustSEDComponent, which would make the
        # control identical to the treatment in the one respect under test.
        else {"type": "none"}
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=_observation(lines=lines),
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust=dust_group,
        neb={"type": "none"} if neb == "none" else {"type": neb, "all_params": FIXED},
        redshift=Fixed(0.1),
        approx=approx,
    )


def _grad_flops(model):
    """FLOPs of the compiled photometry gradient, from the HLO cost analysis."""
    free = list(model.spec.free_params)
    defaults = {k: 10.0 if "log_total_mass" in k else 0.5 for k in free}

    def loss(v):
        params = dict(defaults)
        params["sfh_delayed_log_total_mass"] = v
        return jnp.sum(model.predict_photometry(params))

    return int(jax.jit(jax.grad(loss)).lower(jnp.asarray(10.0)).compile().cost_analysis()["flops"])


def test_setup_the_control_is_really_dust_free(ssp_data_fsps):
    """Guard the guard: if both arms carry dust, every ratio below is meaningless."""
    from tengri.forward.sed_model import _nebular_continuum_consumers

    dusty = _build(ssp_data_fsps, None, dust=True)
    clean = _build(ssp_data_fsps, None, dust=False)

    dusty_consumers = _nebular_continuum_consumers(dusty._build_component_chain())
    clean_consumers = _nebular_continuum_consumers(clean._build_component_chain())

    assert dusty_consumers, "the dusty arm has no sed_nebular consumer — it is not dusty"
    assert not clean_consumers, (
        f"the control still consumes sed_nebular ({[type(c).__name__ for c in clean_consumers]}), "
        "so it is not a control. Omitting dust= builds a dust component; pass "
        "dust={'type': 'none'}."
    )


def test_feature_precomp_still_pays_on_a_dust_free_model(ssp_data_fsps):
    """The lever must still work where it is legitimately available.

    Asserted before the dusty case, because a guard that only checks "no effect on
    dusty models" would pass if ``FeaturePrecomp`` were broken everywhere.
    """
    from tengri import FeaturePrecomp, WavePrecomp

    wave = _grad_flops(_build(ssp_data_fsps, (WavePrecomp(),), dust=False))
    pair = _grad_flops(_build(ssp_data_fsps, (WavePrecomp(), FeaturePrecomp()), dust=False))
    assert pair < wave / 5.0, (
        f"FeaturePrecomp buys only {wave / pair:.2f}x on a dust-free model "
        f"({wave:,} -> {pair:,} gradient FLOPs). It should be an order of magnitude; "
        "the fast nebular grid is not engaging even where it is allowed to."
    )


def test_feature_precomp_is_inert_on_a_dusty_model(ssp_data_fsps):
    """Pins the fact the advice depends on, so it cannot silently change again."""
    from tengri import FeaturePrecomp, WavePrecomp

    wave = _grad_flops(_build(ssp_data_fsps, (WavePrecomp(),), dust=True))
    pair = _grad_flops(_build(ssp_data_fsps, (WavePrecomp(), FeaturePrecomp()), dust=True))
    assert pair == wave, (
        f"FeaturePrecomp changed the compiled gradient on a dusty model "
        f"({wave:,} -> {pair:,}). If the fast path became available for a chain that "
        "reads sed_nebular, check that the nebular continuum is still correct there "
        "(#1673) before relaxing this — and update the resolvers and warnings, which "
        "currently skip the top-up on exactly this predicate (#1748)."
    )


def test_the_resolver_does_not_attach_a_config_that_cannot_engage(ssp_data_fsps):
    """No fit may pay a second compiled kernel for a no-op config.

    ``compile_signature()`` includes the approx state, so appending an inert
    ``FeaturePrecomp`` costs a distinct compiled kernel while changing no FLOPs.
    """
    import jax.numpy as jnp_

    from tengri.inference.fitter import Fitter, fast_nebular_can_engage

    dusty = _build(ssp_data_fsps, None, dust=True)
    assert not fast_nebular_can_engage(dusty)

    n_bands = 3
    fitter = Fitter(dusty, jnp_.ones(n_bands), jnp_.ones(n_bands))
    # "auto" is the default policy every fit surface resolves through.
    resolved = fitter._resolve_fit_approx(dusty, "auto")

    state = getattr(resolved, "approx", None)
    attached = state is not None and getattr(state, "feature_precomp", False)
    assert not attached, (
        "the resolver attached FeaturePrecomp to a dusty model, where it is measured "
        "bit-identical in compiled FLOPs — a separate compiled kernel for no effect "
        "(#1748)."
    )


def _objective_flops(model, approx, *, n_bands: int = 3):
    """FLOPs of the compiled gradient of the FIT OBJECTIVE under ``approx``.

    Distinct from :func:`_grad_flops` on purpose. That one differentiates
    ``predict_photometry``, which has no line channel at all — so it cannot see
    ``FeaturePrecomp``'s second consumer, and reading it as the whole story is what
    #1760 did. The line LUT acts inside the likelihood, via ``needs_state``.

    ``approx`` goes to the **Fitter**, not to ``SEDModel.build``. A build-time config
    with the fitter left on its ``"auto"`` default is not a controlled comparison:
    "auto" TOPS UP, so a WavePrecomp-only arm silently acquires FeaturePrecomp and
    both arms resolve to the same config. That reads as bit-identical FLOPs and looks
    exactly like the inertness this file exists to detect. An explicit config is
    respected as given.
    """
    import warnings

    from tengri.inference.context import InferenceContext
    from tengri.inference.fitter import Fitter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter = Fitter(
            model,
            jnp.full(n_bands, 1e-17),
            jnp.full(n_bands, 1e-18),
            data_type="photometry",
            approx=approx,
        )
        resolved = fitter._resolve_fit_approx(model, approx)
        state = getattr(resolved, "approx", None)
        wanted = any(type(c).__name__ == "FeaturePrecomp" for c in approx)
        got = bool(state is not None and getattr(state, "feature_precomp", False))
        assert got == wanted, (
            f"the arm is not the arm it claims to be: asked for feature_precomp="
            f"{wanted}, resolved to {got}. Assert the treatment arm is LIVE before "
            "reading any ratio off it."
        )
        ctx = InferenceContext.from_target(fitter)
        x0 = ctx.initial_params(jax.random.PRNGKey(0))
        args = ctx.data_args
        grad = jax.grad(ctx.neg_log_posterior_fn, argnums=0)
        analysis = jax.jit(grad).lower(x0, args).compile().cost_analysis()
    if isinstance(analysis, (list, tuple)):
        analysis = analysis[0]
    return int(analysis["flops"])


@pytest.mark.parametrize(
    ("neb", "ssp_fixture", "expect"),
    [
        # Cue publishes a Q_H-linear catalog; a baked-in/wNE SSP keeps its lines in
        # the templates and is measured off the spectrum. Different mechanisms, same
        # verdict once a line channel exists — which is why the fix is a disjunction
        # over consumers rather than a dispatch on the backend.
        ("cue", "ssp_data_fsps", 1.05),
        ("none", "synthetic_ssp_wide", 3.0),
    ],
    ids=["cue-grid", "bakedin-window"],
)
def test_a_line_channel_revives_the_lut_on_a_dusty_model(request, neb, ssp_fixture, expect):
    """The regression #1760 shipped: dust does not make the LUT inert for LINES.

    #1748 measured a dusty model's *photometry* gradient bit-identical with and
    without ``FeaturePrecomp`` and gated the config on that. But the line channel is
    a second consumer — ``loss_functions`` sets ``needs_state = ... or
    (has_line_fluxes and not fast_lines)`` — and it is not disarmed by dust.

    Measured on the fit objective, dusty, fitting three line fluxes:
    Cue **46,607,124 -> 43,211,740 (1.08x)**, baked-in
    **1,986,819 -> 276,186 (7.19x)**. The Cue margin is thin, and that is the point:
    thin is not *bit-identical*. Exact equality is what proves a config never reached
    the graph; 8 % proves it reached it and paid for itself.
    """
    from tengri import FeaturePrecomp, WavePrecomp

    ssp = request.getfixturevalue(ssp_fixture)
    model = _build(ssp, None, dust=True, lines=True, neb=neb)

    wave = _objective_flops(model, (WavePrecomp(),))
    pair = _objective_flops(model, (WavePrecomp(), FeaturePrecomp()))
    assert pair * expect < wave, (
        f"FeaturePrecomp bought only {wave / pair:.2f}x on a DUSTY {neb} model that "
        f"fits line fluxes ({wave:,} -> {pair:,} objective gradient FLOPs); expected "
        f"at least {expect:.2f}x. Dust disarms the nebular continuum shortcut, not "
        "the line channel — do not gate the two on one predicate (#1770)."
    )


def test_the_resolver_attaches_the_lut_for_a_dusty_line_fit(ssp_data_fsps):
    """The same resolver, the same dusty model — opposite verdicts, decided by lines.

    Companion to ``test_the_resolver_does_not_attach_a_config_that_cannot_engage``.
    Pinning only the "does not attach" half is what let a gate that *never* attaches
    look correct: both halves have to be asserted, or the guard cannot tell a correct
    refusal from a broken one.
    """
    import warnings

    from tengri.inference.fitter import Fitter, fast_nebular_can_engage, feature_precomp_can_engage

    dusty_lines = _build(ssp_data_fsps, None, dust=True, lines=True)

    # The narrow predicate still says no — the continuum shortcut really is disarmed.
    assert not fast_nebular_can_engage(dusty_lines)
    # The question a resolver must actually ask says yes: the line consumer is live.
    assert feature_precomp_can_engage(dusty_lines)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter = Fitter(dusty_lines, jnp.full(3, 1e-17), jnp.full(3, 1e-18))
        resolved = fitter._resolve_fit_approx(dusty_lines, "auto")

    state = getattr(resolved, "approx", None)
    assert state is not None and getattr(state, "feature_precomp", False), (
        "a dusty line-flux fit did not get FeaturePrecomp under approx='auto' — every "
        "likelihood evaluation reconstructs the full-grid SED to read a few line "
        "fluxes (#1770)."
    )

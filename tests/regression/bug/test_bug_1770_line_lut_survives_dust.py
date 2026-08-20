# SPDX-License-Identifier: BSD-3-Clause
"""A dusty line-flux fit must still get the line LUT (#1770).

#1748 established that ``FeaturePrecomp``'s *photometry* shortcut is disarmed by
dust: serving photometry from the per-Q_H grid requires zeroing ``sed_nebular``,
and ``DustSEDComponent`` reads it. #1760 acted on that with
``fast_nebular_can_engage()`` — correctly — but consulted it on the *line*
channel too, and its guard measured ``jnp.sum(model.predict_photometry(params))``,
a photometry objective that cannot observe the line-channel saving.

The line channel is served by a different mechanism: the LUT supplies the line
fluxes directly, so ``loss_functions`` need not set ``needs_state=True`` and
rebuild the full-grid SED through ``predict_state`` on every likelihood
evaluation. Dust does not touch that, so gating it cost every dusty line-flux
fit a measured 4.77x.

FLOPs off ``compile().cost_analysis()``, not wall clock, for the reason #1760
gives: XLA has been shown to compile the "exact" arm down to the fast one and
make a timing guard pass on the cost it was checking (#1696).

**The #1748 control is not optional.** Without it this file would pass just as
well if someone "fixed" #1770 by reverting #1760 wholesale, which would restore
a config that is genuinely inert on a photometry-only dusty fit.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    FIXED,
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
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.inference.fitter import _resolve_batch_fit_approx
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

_LINES = ("Halpha", "Hbeta", "OIII_5007")
_WAVES = jnp.array([6564.61, 4862.71, 5008.24])

#: Measured 4.77x on this fixture. Assert well below it: the point is that a
#: large saving exists, not the exact number, which is XLA-version sensitive.
_MIN_SPEEDUP = 2.0


@pytest.fixture(scope="module")
def wide_ssp():
    wave = jnp.logspace(2.0, 7.0, 1600)
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(jax.random.normal(jax.random.PRNGKey(3), (3, 25, wave.size))) * 1e-3
        + 1e-5,
        ssp_lg_age_gyr=jnp.linspace(-3.0, 1.14, 25),
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
    )


@pytest.fixture(scope="module")
def phot_obs():
    curves = []
    for i, c in enumerate([3500.0, 4800.0, 6200.0, 7600.0, 9000.0]):
        wv = np.linspace(c - 400, c + 400, 32)
        curves.append(FilterCurve(wave=wv, trans=np.ones_like(wv), name=f"b{i}"))
    return Observation(photometry=Photometry(filters=tuple(curves)))


def _model(ssp, obs, approx, *, dust: bool):
    dust_block = (
        {
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_diff": Uniform(0.0, 2.0),
        }
        if dust
        else {"type": "none"}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation=dust_block,
            neb={"type": "none"},
            redshift=Fixed(0.1),
            approx=approx,
        )


def _line_flux_setup(ssp, phot_obs, *, dust: bool):
    """A photometry + 3-line-flux observation with data at truth."""
    base = _model(ssp, phot_obs, WavePrecomp(), dust_attenuation=dust)
    truth = base.spec.sample(jax.random.PRNGKey(0))
    phot = np.asarray(base.predict_photometry(truth))
    lf = np.asarray(base.measure_line_fluxes(truth, approx=False))[:3]
    obs = Observation(
        photometry=phot_obs.photometry,
        line_fluxes=LineFluxData(
            names=_LINES,
            fluxes=jnp.asarray(lf),
            errors=jnp.asarray(np.abs(lf) * 0.05 + 1e-30),
            wavelengths=_WAVES,
        ),
    )
    return obs, phot


def _fitter(model, phot, **kw):
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
    """Gradient FLOPs of the real fit objective, not of ``predict_photometry``.

    Measuring the photometry surface instead is precisely how #1770 shipped.
    """
    from tengri.inference.context import InferenceContext

    ctx = InferenceContext.from_target(fitter)
    nlp, args = ctx.neg_log_posterior_fn, ctx.data_args
    p0 = ctx.initial_params(jax.random.PRNGKey(1))
    return int(
        jax.jit(jax.grad(lambda p: nlp(p, args))).lower(p0).compile().cost_analysis()["flops"]
    )


def test_dusty_line_flux_fit_gets_the_lut(wide_ssp, phot_obs):
    """The resolver must attach the LUT even though the model carries dust."""
    obs, phot = _line_flux_setup(wide_ssp, phot_obs, dust_attenuation=True)
    fitter = _fitter(_model(wide_ssp, obs, None, dust_attenuation=True), phot)

    assert fitter.model.approx.feature_precomp, (
        "a dusty line-flux fit did not get FeaturePrecomp — fast_nebular_can_engage "
        "gates the photometry shortcut, not the line channel (#1770)"
    )


def test_the_lut_is_worth_attaching_on_a_dusty_model(wide_ssp, phot_obs):
    """And it must actually be cheaper — the assertion #1760's guard could not make.

    Compares the resolver's own choice against the same fit forced to
    ``WavePrecomp`` alone, so this measures what a user gets rather than what a
    hand-built config does.
    """
    obs, phot = _line_flux_setup(wide_ssp, phot_obs, dust_attenuation=True)

    resolved = _objective_flops(_fitter(_model(wide_ssp, obs, None, dust_attenuation=True), phot))
    # The control must pass ``approx=`` to the Fitter, not merely build the model
    # with it: under the default "auto" policy a build-time WavePrecomp is topped
    # up with the LUT (#1683), so a model-only control silently measures the same
    # arm twice. An explicit config means what it says and is not topped up.
    wave_only = _objective_flops(
        _fitter(_model(wide_ssp, obs, None, dust_attenuation=True), phot, approx=WavePrecomp())
    )

    assert wave_only > resolved * _MIN_SPEEDUP, (
        f"the line LUT saved little on a dusty model: WavePrecomp alone "
        f"{wave_only:,} vs resolved {resolved:,} FLOPs "
        f"({wave_only / max(resolved, 1):.2f}x, expected > {_MIN_SPEEDUP}x). Either "
        f"the LUT stopped being attached, or it stopped helping"
    )


def test_control_photometry_only_dusty_fit_stays_without_the_lut(wide_ssp, phot_obs):
    """#1748 must stay fixed: no line channel, dusty model, no LUT.

    This is what stops #1770 being "fixed" by reverting #1760. There the config
    genuinely buys a second compiled kernel for no effect.
    """
    fitter = _fitter(_model(wide_ssp, phot_obs, None, dust_attenuation=True), np.ones(5))

    assert not fitter.model.approx.feature_precomp, (
        "a photometry-only fit on a dusty model attached FeaturePrecomp, which "
        "#1748 measured as bit-identical in compiled FLOPs"
    )


def test_control_the_dust_free_path_is_unchanged(wide_ssp, phot_obs):
    """A dust-free line-flux fit was already getting the LUT and must still."""
    obs, phot = _line_flux_setup(wide_ssp, phot_obs, dust_attenuation=False)
    fitter = _fitter(_model(wide_ssp, obs, None, dust_attenuation=False), phot)

    assert fitter.model.approx.feature_precomp
    assert fitter.model.approx.wave_precomp


def test_explicit_feature_precomp_is_still_honored(wide_ssp, phot_obs):
    """Naming the LUT explicitly on a dusty model must not be silently dropped."""
    obs, phot = _line_flux_setup(wide_ssp, phot_obs, dust_attenuation=True)
    model = _model(wide_ssp, obs, (WavePrecomp(), FeaturePrecomp()), dust_attenuation=True)

    assert _fitter(model, phot).model.approx.feature_precomp


@pytest.mark.parametrize("dust", [True, False], ids=["dusty", "dust-free"])
@pytest.mark.parametrize("lines", [True, False], ids=["lines", "no-lines"])
def test_the_batch_resolver_agrees_with_the_single_galaxy_one(wide_ssp, phot_obs, dust, lines):
    """The two surfaces must reach the same verdict on the same model.

    ``CatalogFitter`` and ``PopulationFitter`` resolve their precompute through
    :func:`_resolve_batch_fit_approx`, not through ``Fitter._resolve_fit_approx``.
    The first fix for #1770 corrected the single-galaxy resolver and left the batch
    one gated on ``fast_nebular_can_engage`` with no line-channel check at all, so
    the two disagreed on exactly one cell:

    ========  =======  =====================  ======================
    dust      lines    batch attached?        single-galaxy attached?
    ========  =======  =====================  ======================
    **yes**   **yes**  **no**                 **yes**
    yes       no       no                     no
    no        yes      yes                    yes
    no        no       no                     no
    ========  =======  =====================  ======================

    One diverging cell out of four is the catalog channel matrix's signature
    failure (#1460/#1480/#1599): the surfaces agree everywhere the fixtures
    happened to look. Asserted as the full matrix rather than the one broken cell,
    so a future gate cannot fix this row by breaking another.

    Batch surfaces are the ones that matter most here — they evaluate the forward
    model per galaxy, per likelihood call.
    """
    obs, phot = _line_flux_setup(wide_ssp, phot_obs, dust_attenuation=dust)
    model = _model(wide_ssp, obs if lines else phot_obs, None, dust_attenuation=dust)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = _resolve_batch_fit_approx(model, "auto", "photometry")
    single = _fitter(model, phot).model

    def _attached(m):
        state = getattr(m, "approx", None)
        return bool(state is not None and getattr(state, "feature_precomp", False))

    assert _attached(batch) == _attached(single), (
        f"batch and single-galaxy resolvers disagree on dust={dust}, lines={lines}: "
        f"batch attached={_attached(batch)}, single attached={_attached(single)}. "
        "A catalog fit must not silently run a different precompute from the "
        "identical single-galaxy fit (#1770)."
    )

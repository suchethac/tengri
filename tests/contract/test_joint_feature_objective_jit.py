# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the joint feature objective is JIT-cached, finite, and non-silent.

Regression for the 2026-07 precompute audit. The DESI-style joint fit —
GALEX/DES/WISE photometry + emission-line fluxes + Dn4000 / Balmer indices —
exercises the feature channels (``predict_line_fluxes`` /
``predict_spectral_indices``) that force a full-grid ``predict_state`` forward.

Two guarantees pinned here:

1. ``neg_log_posterior_fn`` is a genuine ``jax.jit``-wrapped callable, so a
   direct objective evaluation is fused (~1-3 ms) rather than a Python-level
   per-component chain dispatch (~27 ms — 20x slower, which silently drowned
   the WavePrecomp LUT speedup when a line / index channel was present).
2. The line-flux and spectral-index channels actually feed the objective —
   corrupting the observed feature data must move the posterior, so a silent
   drop of a channel (wrong-shape data, mis-wired cohort) is caught.

Data-gated (needs real SSP grids); skips in CI. The bare-stellar FSPS grid is
required for the Cue line channel; the wNE grid suffices for indices.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    FIXED,
    FREE,
    Fitter,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.inference.context import InferenceContext
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.spectral_indices import STANDARD_INDICES, SpectralIndexData

pytestmark = pytest.mark.contract

_BANDS = [
    "galex_fuv",
    "galex_nuv",
    "des_g",
    "des_r",
    "des_i",
    "des_z",
    "des_y",
    "wise_w1",
    "wise_w2",
    "wise_w3",
    "wise_w4",
]
_INDEX = [STANDARD_INDICES["Dn4000"], STANDARD_INDICES["HdA"]]
_LINES = ["Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717"]


def _fitter(model, phot):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Fitter(
            model,
            data=np.asarray(phot),
            noise=0.05 * np.abs(np.asarray(phot)) + 1e-31,
            data_type="photometry",
        )


def _grad_max(gfn, x0, da):
    g = gfn(x0, da)
    return float(max(float(jnp.max(jnp.abs(v))) for v in g.values()))


def test_phot_plus_dn4000_objective_is_jit_and_nonsilent(ssp_data_wne):
    """phot + Dn4000/HdA: objective jit'd + finite; the index channel moves it."""
    import warnings

    sid = SpectralIndexData(
        index_defs=tuple(_INDEX), values=jnp.array([1.4, 5.0]), errors=jnp.array([0.02, 0.3])
    )
    obs = Observation(photometry=Photometry.from_names(_BANDS), spectral_indices=sid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=obs,
            sfh={"type": "dpl", "*": FREE},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_diff": Uniform(0.0, 2.0),
            },
            neb={"type": "none"},
            redshift=Fixed(0.1),
            approx=WavePrecomp(),
        )
    truth = model.spec.sample(jax.random.PRNGKey(0))
    phot = model.predict_photometry(truth)
    # mock index data at truth so chi2 is well posed
    idx_truth = np.asarray(model.predict_spectral_indices(truth, _INDEX))
    sid_obs = SpectralIndexData(
        index_defs=tuple(_INDEX), values=jnp.array(idx_truth), errors=jnp.array([0.02, 0.3])
    )
    obs = Observation(photometry=Photometry.from_names(_BANDS), spectral_indices=sid_obs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=obs,
            sfh={"type": "dpl", "*": FREE},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_diff": Uniform(0.0, 2.0),
            },
            neb={"type": "none"},
            redshift=Fixed(0.1),
            approx=WavePrecomp(),
        )
    ctx = InferenceContext.from_target(_fitter(model, phot))
    nlp, da = ctx.neg_log_posterior_fn, ctx.data_args
    x0 = ctx.initial_params(jax.random.PRNGKey(1))

    # (1) JIT-cached — the perf guarantee
    assert hasattr(nlp, "lower") and hasattr(nlp, "trace")
    # (2) finite objective + gradient
    v = float(nlp(x0, da))
    assert np.isfinite(v)
    gfn = jax.jit(jax.grad(lambda p, d: nlp(p, d)))
    assert np.isfinite(_grad_max(gfn, x0, da))

    # (3) the index channel is not a silent drop: corrupt Dn4000 -> objective moves
    sid_bad = SpectralIndexData(
        index_defs=tuple(_INDEX),
        values=jnp.array([idx_truth[0] + 1.0, idx_truth[1]]),  # +1 in Dn4000 = ~20 sigma
        errors=jnp.array([0.02, 0.3]),
    )
    obs_bad = Observation(photometry=Photometry.from_names(_BANDS), spectral_indices=sid_bad)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_bad = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=obs_bad,
            sfh={"type": "dpl", "*": FREE},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_diff": Uniform(0.0, 2.0),
            },
            neb={"type": "none"},
            redshift=Fixed(0.1),
            approx=WavePrecomp(),
        )
    ctx_bad = InferenceContext.from_target(_fitter(model_bad, phot))
    v_bad = float(ctx_bad.neg_log_posterior_fn(x0, ctx_bad.data_args))
    assert abs(v - v_bad) > 1.0, "Dn4000 channel is a silent no-op in the objective"


def test_desi_joint_phot_lines_dn4000_objective(ssp_data_fsps):
    """Full DESI joint (phot + 5 lines + Dn4000, Cue): jit'd, finite, lines move it."""
    import warnings

    dummy = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
    sid = SpectralIndexData.from_names(["Dn4000"], [1.4], [0.05])

    def build(obs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh={"type": "dpl", "*": FREE},
                dust={
                    "type": "two_component",
                    "law": "calzetti",
                    "*": FIXED,
                    "tau_diff": Uniform(0.0, 2.0),
                },
                neb={"type": "cue", "*": FIXED},
                redshift=Fixed(0.1),
                approx=WavePrecomp(),
            )

    m0 = build(
        Observation(
            photometry=Photometry.from_names(_BANDS), line_fluxes=dummy, spectral_indices=sid
        )
    )
    truth = m0.spec.sample(jax.random.PRNGKey(2))
    phot = m0.predict_photometry(truth)
    line_truth = np.asarray(m0.predict_line_fluxes(truth, target_wavelengths=dummy.wavelengths))
    idx_truth = np.asarray(m0.predict_spectral_indices(truth, [STANDARD_INDICES["Dn4000"]]))

    lines = LineFluxData(
        names=tuple(_LINES),
        fluxes=jnp.array(line_truth),
        errors=jnp.array(0.1 * np.abs(line_truth) + 1e-19),
        wavelengths=dummy.wavelengths,
    )
    sid_obs = SpectralIndexData(
        index_defs=(STANDARD_INDICES["Dn4000"],),
        values=jnp.array(idx_truth),
        errors=jnp.array([0.02]),
    )
    obs = Observation(
        photometry=Photometry.from_names(_BANDS), line_fluxes=lines, spectral_indices=sid_obs
    )
    ctx = InferenceContext.from_target(_fitter(build(obs), phot))
    nlp, da = ctx.neg_log_posterior_fn, ctx.data_args
    x0 = ctx.initial_params(jax.random.PRNGKey(3))

    assert hasattr(nlp, "lower") and hasattr(nlp, "trace")  # jit-cached
    v = float(nlp(x0, da))
    assert np.isfinite(v)
    gfn = jax.jit(jax.grad(lambda p, d: nlp(p, d)))
    assert np.isfinite(_grad_max(gfn, x0, da))

    # lines contribute: 3x the observed line fluxes -> objective must move
    lines_bad = LineFluxData(
        names=tuple(_LINES),
        fluxes=jnp.array(line_truth * 3.0),
        errors=jnp.array(0.1 * np.abs(line_truth) + 1e-19),
        wavelengths=dummy.wavelengths,
    )
    obs_bad = Observation(
        photometry=Photometry.from_names(_BANDS), line_fluxes=lines_bad, spectral_indices=sid_obs
    )
    ctx_bad = InferenceContext.from_target(_fitter(build(obs_bad), phot))
    v_bad = float(ctx_bad.neg_log_posterior_fn(x0, ctx_bad.data_args))
    assert abs(v - v_bad) > 1.0, "line-flux channel is a silent no-op in the objective"

# SPDX-License-Identifier: BSD-3-Clause
"""Composite dense_basis SFH: tx_frac never reaches the forward model (#1031).

``mean_sfh_type=["dense_basis", "field"]`` registers ``sfh_dbp_tx_frac_*``
correctly in the spec and moves them through ``predict_sfh`` — but the forward
model never sees them. Photometry is **bit-identical** across the whole
``tx_frac`` range:

===================  ===================  ==========================
SFH                  tx_frac moves SFH?   tx_frac moves photometry?
===================  ===================  ==========================
``dense_basis``      yes                  yes  (correct)
``[dense_basis,      yes                  **no** (silent no-op)
  field]``
===================  ===================  ==========================

The composed SFH function dispatches per-component on **public** names
(``sfh_dbp_tx_frac_0``) — deliberately, so two additive SFH components that
share an internal kwarg (both carry a ``log_total_mass``) cannot collide. The
kwargs handed to it are keyed by *internal* name, so the composer matches
nothing and dense_basis is evaluated with no parameters at all. In some
configurations that surfaces loudly (``ValueError: dense_basis requires at least
one tx_frac_* parameter``, seen in the slow-tier VI fits and behind the skipped
Stochastic-Field benchmark sections of #925); in the plain photometry path it is
silent, which is worse — a sampler explores three free parameters that change
nothing and reports a confident, meaningless posterior.

**The xfail below is the acceptance test for any fix.** Re-keying
``_compute_sfr``'s kwargs to public names makes the ValueError disappear and the
model predict happily, while leaving the photometry no-op fully intact — a fix
that only stops the exception has not fixed anything. ``_compute_sfr`` is not
the only place the SFH kwargs are assembled; the forward path has its own, and
the values must be carried all the way to ``predict_photometry``.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import pytest

from tengri import FIXED, FREE, Fixed, SEDModel

pytestmark = pytest.mark.regression_bug

_TX_SWEEP = (0.05, 0.5, 0.95)


def _build(ssp_data, observation, sfh_type):
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={"type": sfh_type, "*": FREE},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


def _sweep_tx(model, key=0):
    """Photometry across the tx_frac range, holding every other param fixed."""
    params = dict(model.spec.sample(jax.random.PRNGKey(key)))
    tx = next(p for p in model.spec.free_params if "tx_frac" in p)
    return tx, [model.predict_photometry({**params, tx: jnp.asarray(v)}) for v in _TX_SWEEP]


def _moves(fluxes):
    """True if sweeping the parameter changed the observable at all."""
    return not all(jnp.allclose(a, b, atol=0.0, rtol=1e-9) for a, b in itertools.pairwise(fluxes))


@pytest.mark.parametrize("sfh_type", ["dexp", "dpl", "dense_basis", ["tsnorm", "field"]])
def test_working_sfh_topologies_still_predict(synthetic_ssp_wide, synthetic_tophat_obs, sfh_type):
    """The topologies that work — a guard for any fix to the composite.

    The SFH kwargs dict is assembled on a hot path shared by every topology, so
    a fix that repairs ``dense_basis+field`` by breaking ``dexp`` fails here.
    """
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs, sfh_type)
    flux = model.predict_photometry(model.spec.sample(jax.random.PRNGKey(0)))

    assert jnp.all(jnp.isfinite(flux))
    assert jnp.all(flux > 0)


def test_dense_basis_alone_is_not_a_no_op(synthetic_ssp_wide, synthetic_tophat_obs):
    """Control: the single-component path DOES carry tx_frac into the SED.

    This is what makes the composite failure a wiring bug rather than a
    dense_basis bug, and it pins the working half so a fix cannot regress it.
    """
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs, "dense_basis")
    tx, fluxes = _sweep_tx(model)

    assert _moves(fluxes), f"{tx} should move the photometry for a bare dense_basis SFH"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#1031: composite dense_basis+field drops tx_frac in the forward model. "
        "Photometry is bit-identical across the full tx_frac range while "
        "predict_sfh responds — a silent no-op, not a crash. Any fix must turn "
        "THIS green, not merely stop the ValueError seen in the fit path."
    ),
)
def test_tx_frac_reaches_the_forward_model_in_composite(synthetic_ssp_wide, synthetic_tophat_obs):
    """Acceptance test for #1031: tx_frac must move the *observable*.

    Asserting on ``spec.free_params`` would NOT catch this — the spec was always
    correct. Nor would ``predict_sfh``, which responds to tx_frac even while the
    forward model ignores it. Only sweeping the parameter and watching the
    prediction move proves the value crossed the seam.
    """
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs, ["dense_basis", "field"])
    tx, fluxes = _sweep_tx(model)

    assert _moves(fluxes), (
        f"{tx} is a silent no-op: photometry is bit-identical across "
        f"tx_frac = {_TX_SWEEP}, so a sampler would explore it for free"
    )

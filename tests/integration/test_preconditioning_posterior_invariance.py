# SPDX-License-Identifier: BSD-3-Clause
"""Two arms of the same posterior must agree on their means (#1301, #1498).

The companion to ``tests/contract/test_preconditioning_roundtrip.py``, which holds
the *primary* guard: preconditioned draws must explain the data, measured as a
log-posterior deficit. That guard is unconditional, runs against every capable
backend, and gates every pull request.

This file holds the second, weaker angle, and it is here rather than beside that
guard for a measured reason. It fits the model **twice** — once with
``precondition=False`` — and the unpreconditioned arm samples an unwhitened
``cond ~ 1e5`` posterior, which NUTS answers by building deeper trajectories
rather than by failing. Measured 2026-08-04: 155 s against 53 s for the entire
contract-tier file. Since the mapping is already guarded unconditionally there,
paying that on every pull request buys very little, so it runs on the nightly
slow tier instead.

What it adds that the deficit cannot see: a *partial* mapping error that still
lands the draws in the typical set. Such draws explain the data perfectly well —
the deficit stays near ``D/2`` — but the posterior they describe is subtly the
wrong one, and only a comparison against a second arm reveals it.

Two properties make it stable where the #1498 failure was not:

* the gap is scaled by its **Monte Carlo standard error** rather than by the
  posterior width, so poor mixing widens the bar instead of tripping it
  (:data:`_Z_GATE`);
* it stays gated on convergence, because an unconverged chain is somewhere else
  entirely and the MCSE does not describe that.

Preconditioning is documented to mix *worse* on exactly this configuration
(D=8 dust, median ESS/s ratio 0.62, range 0.10–1.13), so both matter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FREE, Fixed, ForwardModel, NoiseModel, Observation, Photometry, SEDModel

pytestmark = pytest.mark.contract

BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "2mass_j", "2mass_h"]
OVERRIDES = {
    "sfh_dpl_alpha": 2.0,
    "sfh_dpl_beta": 1.5,
    "sfh_dpl_log_total_mass": 10.5,
    "dust_tau_diff": 0.2,
}


def _model(ssp):
    obs = Observation(
        photometry=Photometry.from_names(BANDS),
        noise=NoiseModel(calibration_floor=0.02, student_t_dof=None),
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        met={"logzsol": Fixed(-0.3)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FREE,
        },
        neb={"type": "none"},
        redshift=Fixed(0.1),
        igm={"type": "none"},
    )


def _mock(model):
    """A valid full parameter set: a prior draw with a few values pinned."""
    drawn = model.spec.sample(jax.random.PRNGKey(0))
    params = {
        **model.spec.get_fixed_values(),
        **drawn,
        **{k: jnp.array(v) for k, v in OVERRIDES.items() if k in drawn},
    }
    mock = model.mock(params, snr=30.0, key=jax.random.PRNGKey(1))
    return np.asarray(mock.flux_obs), np.asarray(mock.noise)


def _fit(model, data, noise, *, precondition, method="mcmc_nuts"):
    return ForwardModel.build(sed=model).fit(
        data,
        noise,
        method=method,
        key=jax.random.PRNGKey(7),
        n_warmup=250,
        n_samples=250,
        n_chains=1,
        dense_mass_matrix=False,
        precondition=precondition,
        verbose=False,
    )


#: Split-R-hat above which a chain is not converged enough for its posterior mean
#: to mean anything. The usual 1.05.
_RHAT_GATE = 1.05

#: Largest gap between the two arms' posterior means, in units of the Monte Carlo
#: standard error *of that gap* (#1498).
#:
#: The previous form divided by the posterior **width**, which is the wrong scale. It
#: asked "did the mean move a lot?" when the question is "did it move further than
#: two finite chains could have moved it by chance?". Dividing by the MCSE is better
#: on both sides at once: a badly-mixed chain has a large MCSE and so *widens* the bar
#: instead of tripping it, while a real basis error clears any bar by an order of
#: magnitude. Measured: **0.58** between two healthy arms, **31.2** against a
#: forgotten inverse map, and ~3.2 for the historical CI failure that opened #1498 —
#: which the old ``< 1.0`` bar reported as a bug at 2.13 sd.
_Z_GATE = 5.0

#: Minimum number of free parameters the mean comparison must actually reach.
#:
#: The model has 7. This loop used to iterate ``plain.samples``, which is 19 entries
#: of which **13 are ``Fixed`` constants** — zero variance, scoring ``0 / 1e-3 = 0``
#: and passing for free. It looked like broad coverage and was six parameters. There
#: was no counter at all, so a loop that reached nothing would have passed silently.
_MIN_COMPARED = 5


@pytest.fixture(scope="module")
def target(ssp_data_wne):
    """Model and mock photometry — nothing more.

    Deliberately lighter than the ``objective`` fixture in the contract-tier file,
    which additionally builds a MAP expansion point and two Hessians. This test
    never touched any of that; it only needs something to fit twice.
    """
    model = _model(ssp_data_wne)
    data, noise = _mock(model)
    return model, data, noise


def test_preconditioning_leaves_the_posterior_unchanged(target):
    """Same posterior, different coordinates — a second angle on the invariant.

    Weaker than the deficit guard, and deliberately secondary. Two independent
    250/250 NUTS runs buy an effective sample size of about **2.5** on this model, so
    a comparison of their means has little power however it is scaled — and a *skip*
    here costs nothing that matters, because the mapping is guarded unconditionally
    by ``test_preconditioned_samples_respect_the_prior_support`` in the contract
    tier. It is kept because it is sensitive to a different failure: a *partial*
    mapping error that still lands the draws in the typical set, which the deficit
    would not notice.
    """
    model, data, noise = target
    plain = _fit(model, data, noise, precondition=False)
    preconditioned = _fit(model, data, noise, precondition=True)

    unconverged = {
        arm: {k: round(v, 3) for k, v in fit.rhat().items() if v > _RHAT_GATE}
        for arm, fit in (("plain", plain), ("preconditioned", preconditioned))
    }
    if any(unconverged.values()):
        pytest.skip(
            "posterior means are not comparable — split-R-hat above "
            f"{_RHAT_GATE} in: { {k: v for k, v in unconverged.items() if v} }. "
            "A mean shift here would be mixing, not a whitening-mapping bug; the "
            "mapping itself is covered unconditionally by the deficit guard."
        )

    ess_plain = plain.effective_sample_size()
    ess_pre = preconditioned.effective_sample_size()

    def _ess(table, name):
        """ESS, floored at 1. An unknown ESS must widen the bar, never narrow it."""
        value = float(table.get(name, 1.0))
        return value if np.isfinite(value) and value >= 1.0 else 1.0

    compared = 0
    for name in model.spec.free_params:
        if name not in plain.samples or name not in preconditioned.samples:
            continue
        a = np.asarray(plain.samples[name])
        b = np.asarray(preconditioned.samples[name])
        if a.ndim != 1:
            continue
        # MCSE of the difference between two independent chain means.
        mcse = float(np.sqrt(a.var() / _ess(ess_plain, name) + b.var() / _ess(ess_pre, name)))
        if not np.isfinite(mcse) or mcse <= 0.0:
            continue
        z = abs(float(np.mean(a) - np.mean(b))) / mcse
        compared += 1
        assert z < _Z_GATE, (
            f"{name}: posterior means differ by {z:.1f} Monte Carlo standard errors "
            f"({np.mean(a):.4f} vs {np.mean(b):.4f}, ESS {_ess(ess_plain, name):.0f} "
            f"and {_ess(ess_pre, name):.0f}) — too large to be sampling noise, so the "
            "draws are probably still in whitened coordinates"
        )

    assert compared >= _MIN_COMPARED, (
        f"only {compared} free parameters were compared, below the {_MIN_COMPARED} "
        f"this model should reach ({len(model.spec.free_params)} are free) — the "
        "comparison has quietly stopped covering the posterior"
    )

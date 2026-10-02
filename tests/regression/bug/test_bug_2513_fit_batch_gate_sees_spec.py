# SPDX-License-Identifier: BSD-3-Clause
"""#2513: fit_batch forwards the spec to the dense-mass gate.

``fit_batch``'s shared vmap adaptation was the one ``resolve_dense_mass_gate``
caller that did not forward ``spec``. With ``spec=None`` the auto-policy's
final ``return not _spec_uses_dense_basis(spec)`` evaluates True at
``n_dim <= 12``, so the batch path actively GRANTED a dense mass matrix to the
one spec shape measured to OOM (the 22.78 GB dense_basis spike of #319) — and
on this seam the one shared adaptation serves every galaxy in the batch.

The test spies on the REAL gate (wrapping, not replacing: the verdicts below
are the production policy's own) at the consuming module —
``_fit_batch_vmap_mcmc`` imports it function-scope from ``nuts``, so the
module attribute is resolved at call time. The spy firing is also the proof
the batch took the vmap route rather than the sequential fallback. The DPL
contrast arm pins that the diagonal verdict comes from the SPEC, not from
dimensionality or the method: same bands, same free-parameter scale, auto
policy below the cap — dense_basis gets diagonal, DPL gets dense.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    SSPData,
    Uniform,
)

pytestmark = pytest.mark.regression_bug


def _synthetic_ssp():
    n_age = 20
    wave = jnp.logspace(2.0, 5.5, 400)
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=ages_gyr,
        ssp_lgmet=lgmet,
    )


def _build(sfh_group):
    obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i"]))
    return SEDModel.build(
        ssp_data=_synthetic_ssp(),
        observation=obs,
        sfh=sfh_group,
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


def _run_batch_and_spy(model, monkeypatch):
    """fit_batch under the auto policy, spying on the real gate's traffic."""
    from tengri.inference.backends.mcmc import nuts as nuts_mod
    from tengri.inference.fitter import Fitter

    truth = {k: float(v) for k, v in model.spec.sample(jax.random.PRNGKey(0)).items()}
    flux = np.asarray(model.predict_photometry(truth))
    noise = np.abs(flux) / 10.0

    calls = []
    real_gate = nuts_mod.resolve_dense_mass_gate

    def spy(dense_mass_matrix, n_dim, **kwargs):
        verdict = real_gate(dense_mass_matrix, n_dim, **kwargs)
        calls.append(
            {
                "n_dim": n_dim,
                "spec": kwargs.get("spec"),
                "method": kwargs.get("method"),
                "verdict": verdict,
            }
        )
        return verdict

    monkeypatch.setattr(nuts_mod, "resolve_dense_mass_gate", spy)

    fitter = Fitter(model, data=jnp.asarray(flux), noise=jnp.asarray(noise))
    batch = [
        {"flux_obs": jnp.asarray(flux), "noise": jnp.asarray(noise)},
        {"flux_obs": jnp.asarray(flux) * 1.1, "noise": jnp.asarray(noise)},
    ]
    posteriors = fitter.fit_batch(
        batch,
        method="mcmc_hmc",
        key=jax.random.PRNGKey(1),
        verbose=False,
        dense_mass_matrix=None,  # the auto policy — the branch under test
        n_warmup=60,
        n_samples=20,
        precondition=False,
    )
    assert len(posteriors) == 2
    batch_calls = [c for c in calls if c["method"] == "fit_batch"]
    assert batch_calls, (
        "the fit_batch gate never fired — the batch fell back to the "
        "sequential path and this test is not exercising the seam"
    )
    return batch_calls[0], fitter


def test_dense_basis_spec_comes_back_diagonal_through_fit_batch(monkeypatch):
    model = _build({"type": "dense_basis", "all_params": FREE})
    call, fitter = _run_batch_and_spy(model, monkeypatch)
    assert call["n_dim"] <= 12, (
        f"fixture drifted: n_dim={call['n_dim']} is above the auto-dense window, "
        "so the dense_basis exception cannot be observed"
    )
    assert call["spec"] is fitter.spec, (
        "fit_batch did not forward its own spec to the dense-mass gate "
        f"(got {type(call['spec']).__name__})"
    )
    assert call["verdict"] is False, (
        "a dense_basis spec at n_dim <= 12 was GRANTED a dense mass matrix "
        "through fit_batch — the #319 OOM shape, shared across the whole batch"
    )


def test_dpl_spec_still_gets_dense_through_fit_batch(monkeypatch):
    """The contrast arm: the diagonal verdict above comes from the spec."""
    model = _build(
        {
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": FREE,
            "alpha": FREE,
        }
    )
    call, fitter = _run_batch_and_spy(model, monkeypatch)
    assert call["n_dim"] <= 12
    assert call["spec"] is fitter.spec
    assert call["verdict"] is True, (
        "a non-dense_basis spec below the cap should resolve dense under the "
        "auto policy; if this fails the gate's policy changed and the "
        "dense_basis arm above is no longer a spec-driven contrast"
    )

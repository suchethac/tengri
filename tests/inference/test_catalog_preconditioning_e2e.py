# SPDX-License-Identifier: BSD-3-Clause
"""The analytic metric, threaded through a real batched catalog fit.

``tests/contract/test_catalog_preconditioning.py`` pins the shape of the seam
from the outside. This suite runs it: three galaxies, two free parameters,
``mcmc_hmc`` and ``mcmc_chees`` on the batched path, with and without the metric.

Three things can only be caught here, and each of them is silent:

1. **The draws must come back in the standardized latent basis.** Sampling
   happens in ``zeta`` and the posterior is reported in ``xi = A zeta``. A
   missing restore returns finite, correctly-shaped, correctly-ordered draws in
   the wrong coordinates -- no exception, no warning, a posterior quietly
   reported in a basis nobody asked for. The check is that the preconditioned and
   unpreconditioned fits agree on the *same* galaxies.
2. **The metric must be per galaxy.** One matrix hoisted out of the ``lax.map``
   and broadcast would run without error and whiten every galaxy against
   whichever one built it. The check is that galaxies with different data get
   different measured metric condition numbers.
3. **The reported conditioning must be the run's own.** A fit that whitened but
   cannot say how much is not reportable.

Auto-marked ``slow`` by the ``tests/inference`` path rule in ``conftest.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

#: Declared support for ``sfh_dpl_log_total_mass``. Named once so the prior and
#: the injected truths cannot drift apart -- they did once (#369 / c66c0aff0),
#: and every galaxy clamped to the boundary while the failure read as a
#: batch-routing bug.
_MASS_LO, _MASS_HI = 7.0, 12.5

#: Two decades apart, well inside the support, so the three galaxies have
#: genuinely different posterior curvature and a per-galaxy metric has something
#: to be per-galaxy *about*.
_MASSES = (9.0, 10.0, 11.0)


def _build_model(synthetic_ssp, simple_observation):
    """A dpl-SFH photometry model with mass and metallicity free (D = 2).

    Two free parameters rather than one on purpose: at D = 1 the transform is a
    scalar rescaling and cannot rotate, so a bug that dropped the off-diagonal
    structure would pass. Mass and metallicity are the pair broadband photometry
    genuinely couples, which is what gives the metric a non-trivial off-diagonal
    to whiten.
    """
    from tengri import Fixed, Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(_MASS_LO, _MASS_HI),
        met_logzsol=Uniform(0.4, 1.8),  # in-grid for synthetic_ssp
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    return SEDModel(spec, synthetic_ssp, observation=simple_observation)


def _catalog(model, key, noise_frac=0.02):
    """Three galaxies a decade apart in mass, fractional noise only.

    The synthetic SSP produces absolute fluxes ~1e-25, so any additive floor
    would swamp the signal and leave every galaxy prior-dominated -- and a
    prior-dominated posterior has metric ``I``, which would make a per-galaxy
    metric test vacuous.
    """
    galaxies = []
    for i, mass in enumerate(_MASSES):
        k = jax.random.fold_in(key, i)
        params = dict(model.spec.sample(k))
        params["sfh_dpl_log_total_mass"] = jnp.array(mass)
        params["met_logzsol"] = jnp.array(1.0)
        flux = model.predict_photometry(params)
        noise = jnp.abs(flux) * noise_frac
        obs = flux + noise * jax.random.normal(jax.random.fold_in(k, 1), shape=flux.shape)
        galaxies.append({"flux_obs": obs, "noise": noise})
    return galaxies


def _run(cat, method, *, precondition, key, **extra):
    return cat.run(
        method,
        key=key,
        forward_chunk_size=3,
        n_warmup=80,
        n_burnin=20,
        n_samples=200,
        precondition=precondition,
        verbose=False,
        **extra,
    )


@pytest.fixture(scope="module")
def _fixture(synthetic_ssp, simple_observation):
    from tengri import CatalogFitter

    model = _build_model(synthetic_ssp, simple_observation)
    galaxies = _catalog(model, jax.random.PRNGKey(0))
    return CatalogFitter(model, galaxies, data_type="photometry")


class TestTheMetricActuallyEngages:
    def test_it_reports_what_it_whitened(self, _fixture):
        cp = _run(_fixture, "mcmc_hmc", precondition=True, key=jax.random.PRNGKey(1))
        d = cp.diagnostics
        assert d["whitening_strength"] == 0.5
        assert d["preconditioned"] == 3, "every galaxy should have been whitened"
        assert d["whitened_condition_median"] < d["metric_condition_median"]

    def test_off_reports_nothing_rather_than_zero(self, _fixture):
        """``None`` and ``0`` are different claims; a run that did not whiten
        must not report a condition number it never measured."""
        cp = _run(_fixture, "mcmc_hmc", precondition=None, key=jax.random.PRNGKey(1))
        assert cp.diagnostics["whitening_strength"] is None
        assert cp.diagnostics["preconditioned"] is None
        assert "metric_condition_median" not in cp.diagnostics

    def test_half_whitening_leaves_the_square_root_of_the_conditioning(self, _fixture):
        """``kappa ** |1 - alpha|`` at alpha = 0.5, if the metric is exact.

        Not a tautology: it holds only when the modal Hessian *is* the curvature,
        so a large departure here would say the posterior is strongly
        non-Gaussian at the MAP -- worth knowing, and worth noticing if it
        changes.
        """
        cp = _run(_fixture, "mcmc_hmc", precondition=0.5, key=jax.random.PRNGKey(1))
        for post in cp.posteriors:
            raw = post.diagnostics["metric_condition"]
            whitened = post.diagnostics["whitened_condition"]
            assert whitened == pytest.approx(raw**0.5, rel=0.05)


class TestTheMetricIsPerGalaxy:
    def test_galaxies_with_different_data_get_different_metrics(self, _fixture):
        """One matrix broadcast to every lane would make these identical.

        Three galaxies a decade apart in mass have measurably different
        curvature; if their reported condition numbers coincide, the metric was
        built once and shared, which is a silent correctness failure rather than
        a performance one.
        """
        cp = _run(_fixture, "mcmc_hmc", precondition=True, key=jax.random.PRNGKey(1))
        conds = [p.diagnostics["metric_condition"] for p in cp.posteriors]
        assert len(set(np.round(conds, 6))) == len(conds), (
            f"per-galaxy metric condition numbers coincide ({conds}); the metric "
            "is being shared across lanes rather than built per galaxy"
        )


@pytest.mark.parametrize("method", ["mcmc_hmc", "mcmc_chees"])
class TestTheDrawsComeBackInTheLatentBasis:
    def test_preconditioned_and_not_agree_on_the_posterior(self, _fixture, method):
        """The one check a missing ``restore`` cannot survive.

        A linear reparametrization changes the density only by a constant
        Jacobian, so the sampled distribution is unchanged and the two arms must
        agree to within Monte-Carlo error. Draws left in ``zeta`` would be off by
        the whitening transform -- finite, correctly shaped, and wrong.

        The tolerance is deliberately loose: these are short chains on a
        synthetic SSP and the claim being tested is "same distribution", not
        "same numbers". A dropped restore misses by the condition number, which
        is orders of magnitude, not percent.
        """
        extra = {"n_ensemble": 4, "max_leapfrog_steps": 32} if method == "mcmc_chees" else {}
        off = _run(_fixture, method, precondition=None, key=jax.random.PRNGKey(4), **extra)
        on = _run(_fixture, method, precondition=True, key=jax.random.PRNGKey(4), **extra)

        name = "sfh_dpl_log_total_mass"
        for i, (a, b) in enumerate(zip(off.posteriors, on.posteriors, strict=True)):
            xa = np.asarray(a.samples[name])
            xb = np.asarray(b.samples[name])
            assert np.all(np.isfinite(xb))
            spread = max(float(np.std(xa)), 0.05)
            assert abs(float(np.mean(xa)) - float(np.mean(xb))) < 4.0 * spread, (
                f"galaxy {i}: preconditioned and unpreconditioned posteriors "
                f"disagree ({np.mean(xa):.3f} vs {np.mean(xb):.3f}, spread "
                f"{spread:.3f}). Draws are most likely still in the whitened basis."
            )

    def test_the_draws_stay_inside_the_declared_support(self, _fixture, method):
        """A basis error usually shows up here first, as an impossible mass."""
        extra = {"n_ensemble": 4, "max_leapfrog_steps": 32} if method == "mcmc_chees" else {}
        on = _run(_fixture, method, precondition=True, key=jax.random.PRNGKey(5), **extra)
        for post in on.posteriors:
            x = np.asarray(post.samples["sfh_dpl_log_total_mass"])
            assert x.min() >= _MASS_LO - 1e-6
            assert x.max() <= _MASS_HI + 1e-6

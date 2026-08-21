# SPDX-License-Identifier: BSD-3-Clause
"""Vectorized catalog posterior sampling (NUTS/HMC) through ``CatalogFitter.run``.

Track A of the GPU catalog/hierarchical plan: catalog fits want *sampled*
per-galaxy posteriors, not VI. Before this feature, ``CatalogFitter.run(
"mcmc_nuts", forward_chunk_size=K)`` warned and fell back to a Python
for-loop (one NUTS warmup per galaxy). This suite pins the vectorized
behavior:

- ``forward_chunk_size`` is *honored* for ``mcmc_nuts`` / ``mcmc_hmc`` — no
  "ignored" warning — and the run reports a vectorized diagnostic marker.
- **Different galaxies stay distinct.** Galaxies with well-separated injected
  parameters are recovered in the right slot (order preserved, each near its
  own truth) — a mis-routed batch axis would scramble this.
- **Chunk-invariance.** ``forward_chunk_size=1`` and ``=N`` with the same key
  give the same samples up to float round-off (~1e-8) — the
  ``lax.map(batch_size=K)`` chunk width is a performance detail, not a
  numerical one.
- Per-galaxy ``Posterior`` objects carry finite ``samples`` of the right shape.

Runs on the synthetic SSP (CI-runnable). Auto-marked ``slow`` by the
``tests/inference`` path rule in ``conftest.py``.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

#: Declared support for ``sfh_dpl_log_total_mass``, named once so the prior, the
#: injected truths and the recovery check cannot drift apart.
#:
#: They did. #369 renamed ``sfh_*_log_peak_sfr`` to ``log_total_mass``, turning
#: ``log10(SFR)`` into ``log10(M*)``, and c66c0aff0 (#1839) converted this prior
#: from ``Uniform(-1.0, 2.5)`` to the declared range — but the three truths below
#: stayed at 0/1/2, every one of them below the new floor. All three galaxies
#: then clamped to the boundary and the order check read
#: ``mass not monotonic across slots: [7. 7. 7.]``, which looks like a
#: batch-routing bug and is not one. ``tools/check_param_ranges.py`` compares
#: call-site *priors* to the declaration, so injected truths are invisible to it.
_MASS_LO, _MASS_HI = 7.0, 12.5


def _build_model(synthetic_ssp, simple_observation):
    """A small dpl-SFH photometry model with stellar mass as the one free param.

    Mass is the clean amplitude parameter that broadband photometry constrains
    directly; everything that could trade against it (age, alpha, beta, tau, dust
    via M/L, metallicity, redshift) is pinned so the recovery is unambiguous. The
    point of these tests is the *vectorization* (per-galaxy warmup + ``lax.map``
    data routing), not the dimensionality of the posterior.
    """
    from tengri import Fixed, Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(_MASS_LO, _MASS_HI),
        sfh_dpl_alpha=Fixed(2.0),  # pin shape params — else degenerate with mass (M/L)
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(1.0),  # in-grid for synthetic_ssp (logzsol range [0.348, 1.848])
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    return SEDModel(spec, synthetic_ssp, observation=simple_observation)


def _catalog_from_truths(model, truths, key, noise_frac=0.02):
    """Inject galaxies with the given free-param truths (a list of dicts).

    Noise is purely fractional (no additive floor): the synthetic SSP produces
    absolute fluxes ~1e-25, so any fixed floor would swamp the signal and leave
    every galaxy prior-dominated. Fractional noise keeps SNR ~ 1/noise_frac at
    every mass scale, so mass is genuinely constrained.
    """
    galaxies = []
    for i, overrides in enumerate(truths):
        k = jax.random.fold_in(key, i)
        true_params = dict(model.spec.sample(k))
        true_params.update(overrides)
        flux = model.predict_photometry(true_params)
        noise = jnp.abs(flux) * noise_frac
        flux_obs = flux + noise * jax.random.normal(jax.random.fold_in(k, 1), shape=flux.shape)
        galaxies.append({"flux_obs": flux_obs, "noise": noise})
    return galaxies


# Distinct galaxies: monotonically increasing stellar mass (the amplitude
# parameter that 3-band photometry cleanly constrains), same alpha. A mis-routed
# batch axis — galaxy i sampling galaxy j's data — is caught by an order check.
# (alpha, the SFH shape, is left free but is only weakly constrained by broadband
# photometry, so it is not used as a recovery target.)
#: 1e9 / 1e10 / 1e11 Msun — a decade apart, well inside ``_MASS_LO.._MASS_HI``,
#: so monotonicity across slots is a routing check and not a prior-clamp artifact.
_DISTINCT_MASSES = (9.0, 10.0, 11.0)

_DISTINCT_TRUTHS = [
    {"sfh_dpl_log_total_mass": jnp.array(m), "sfh_dpl_alpha": jnp.array(2.0)}
    for m in _DISTINCT_MASSES
]


def test_forward_chunk_size_honored_for_nuts(synthetic_ssp, simple_observation):
    """mcmc_nuts + forward_chunk_size must NOT warn, and must take the vectorized path."""
    from tengri import CatalogFitter

    model = _build_model(synthetic_ssp, simple_observation)
    galaxies = _catalog_from_truths(model, _DISTINCT_TRUTHS, jax.random.PRNGKey(0))
    cat = CatalogFitter(model, galaxies, data_type="photometry")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cp = cat.run(
            "mcmc_nuts",
            key=jax.random.PRNGKey(1),
            forward_chunk_size=3,
            n_warmup=15,
            n_burnin=5,
            n_samples=15,
            verbose=False,
        )

    chunk_warns = [w for w in caught if "forward_chunk_size" in str(w.message)]
    assert chunk_warns == [], f"forward_chunk_size should be honored, got: {chunk_warns}"
    assert cp.diagnostics.get("vectorized") is True
    assert cp.diagnostics.get("forward_chunk_size") == 3
    assert cp.n_galaxies == 3


def test_distinct_galaxies_recovered_in_order(synthetic_ssp, simple_observation):
    """Well-separated galaxies are recovered in the right slot (no batch mix-up).

    The synthetic SSP is unphysical by construction, so absolute recovery is
    biased — the point here is *distinctness*: galaxies injected with
    monotonically increasing mass and alpha must come back monotonic and
    well-separated per slot. A mis-routed batch axis (galaxy i sampling galaxy
    j's data) scrambles the order or collapses the spread.
    """
    from tengri import CatalogFitter

    model = _build_model(synthetic_ssp, simple_observation)
    galaxies = _catalog_from_truths(model, _DISTINCT_TRUTHS, jax.random.PRNGKey(2))
    cat = CatalogFitter(model, galaxies, data_type="photometry")

    cp = cat.run(
        "mcmc_nuts",
        key=jax.random.PRNGKey(3),
        forward_chunk_size=3,
        n_warmup=120,
        n_burnin=40,
        n_samples=300,
        verbose=False,
    )
    assert cp.diagnostics.get("vectorized") is True

    mass = np.array([float(np.mean(cp[i].samples["sfh_dpl_log_total_mass"])) for i in range(3)])

    # Order preserved: galaxy i's data landed in slot i (not shuffled by the
    # batch axis), and each galaxy recovered near its own injected mass.
    assert mass[0] < mass[1] < mass[2], f"mass not monotonic across slots: {mass}"
    # Read the targets off the truths that were injected, rather than restating
    # them — a third copy of these numbers is how the first two fell out of step.
    for i, truth in enumerate(_DISTINCT_MASSES):
        assert abs(mass[i] - truth) < 0.4, f"galaxy {i} mass {mass[i]:.3f} far from truth {truth}"
    assert np.all(np.isfinite(mass))


def test_chunk_invariance(synthetic_ssp, simple_observation):
    """K=1 and K=N with the same key give the same samples up to float round-off.

    ``forward_chunk_size`` (the ``lax.map(batch_size=K)`` chunk width) is a
    performance knob, not a numerical one: changing it must not change the
    science. It is not bit-exact — XLA vectorizes reductions differently for
    K=1 vs K=3 — but agreement to ~1e-8 (far tighter than the VI ``n_pad``
    path's 1e-5) confirms each galaxy's chain is independent of the chunking.
    """
    from tengri import CatalogFitter

    model = _build_model(synthetic_ssp, simple_observation)
    galaxies = _catalog_from_truths(model, _DISTINCT_TRUTHS, jax.random.PRNGKey(4))
    cat = CatalogFitter(model, galaxies, data_type="photometry")

    common = dict(
        key=jax.random.PRNGKey(5),
        n_warmup=12,
        n_burnin=4,
        n_samples=12,
        verbose=False,
    )
    cp_k1 = cat.run("mcmc_nuts", forward_chunk_size=1, **common)
    cp_kN = cat.run("mcmc_nuts", forward_chunk_size=3, **common)

    for i in range(3):
        for name in cp_k1[i].samples:
            np.testing.assert_allclose(
                np.asarray(cp_k1[i].samples[name]),
                np.asarray(cp_kN[i].samples[name]),
                rtol=1e-8,
                atol=1e-8,
                err_msg=f"galaxy {i} param {name}: K=1 vs K=3 differ beyond round-off",
            )


def test_samples_shape_and_finite(synthetic_ssp, simple_observation):
    """Each per-galaxy Posterior carries finite samples of length n_samples."""
    from tengri import CatalogFitter

    model = _build_model(synthetic_ssp, simple_observation)
    galaxies = _catalog_from_truths(model, _DISTINCT_TRUTHS, jax.random.PRNGKey(6))
    cat = CatalogFitter(model, galaxies, data_type="photometry")

    n_samples = 20
    cp = cat.run(
        "mcmc_nuts",
        key=jax.random.PRNGKey(7),
        forward_chunk_size=2,
        n_warmup=15,
        n_burnin=5,
        n_samples=n_samples,
        verbose=False,
    )
    assert len(cp) == 3
    for i in range(3):
        samples = cp[i].samples
        assert "sfh_dpl_log_total_mass" in samples
        arr = np.asarray(samples["sfh_dpl_log_total_mass"])
        assert arr.shape[0] == n_samples, (
            f"galaxy {i}: expected {n_samples} draws, got {arr.shape}"
        )
        assert np.all(np.isfinite(arr)), f"galaxy {i}: non-finite samples"


def test_mcmc_hmc_vectorizes_too(synthetic_ssp, simple_observation):
    """The HMC sampler shares the vectorized machinery — smoke it end to end."""
    from tengri import CatalogFitter

    model = _build_model(synthetic_ssp, simple_observation)
    galaxies = _catalog_from_truths(model, _DISTINCT_TRUTHS, jax.random.PRNGKey(8))
    cat = CatalogFitter(model, galaxies, data_type="photometry")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cp = cat.run(
            "mcmc_hmc",
            key=jax.random.PRNGKey(9),
            forward_chunk_size=3,
            n_warmup=15,
            n_burnin=5,
            n_samples=15,
            n_leapfrog_steps=8,
            verbose=False,
        )

    assert [w for w in caught if "forward_chunk_size" in str(w.message)] == []
    assert cp.diagnostics.get("vectorized") is True
    assert cp.diagnostics.get("sampler") == "hmc"
    for i in range(3):
        arr = np.asarray(cp[i].samples["sfh_dpl_log_total_mass"])
        assert arr.shape[0] == 15
        assert np.all(np.isfinite(arr))


def _build_model_multi_d(synthetic_ssp, simple_observation):
    """A D=3 model: mass plus two parameters that trade against it.

    ``_build_model`` pins six of seven parameters, so it is D=1 and cannot show
    a mixing failure -- a frozen chain there is indistinguishable from a
    converged one by shape and finiteness alone (#2026). These tests need a
    posterior with somewhere to fail.
    """
    from tengri import Fixed, Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(_MASS_LO, _MASS_HI),
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(1.0),
        dust_tau_bc=Uniform(0.0, 1.0),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    return SEDModel(spec, synthetic_ssp, observation=simple_observation)


_MULTI_D_TRUTHS = [
    {"sfh_dpl_log_total_mass": 9.0, "sfh_dpl_alpha": 2.0, "dust_tau_bc": 0.3},
    {"sfh_dpl_log_total_mass": 10.5, "sfh_dpl_alpha": 2.0, "dust_tau_bc": 0.3},
]


def test_catalog_hmc_draws_actually_move(synthetic_ssp, simple_observation):
    """Every free parameter of every galaxy must *move*, not merely be finite.

    A frozen chain is finite and correctly shaped, and split R-hat reads ~1.0 on
    it because both variances are zero (#1438), so shape-and-finite assertions
    cannot see this failure. The convention is the one
    ``docs/dev/hierarchical-flat-seam.md`` prescribes and
    ``test_hierarchical_backends_actually_run.py`` already applies to
    ``PopulationFitter``; #2026 is that no catalog test applied it.

    Scope, so this is not over-read: on this synthetic D=3 model the pre-PR-#2031
    configuration (prior-center init, forced diagonal mass) also passes -- the
    smallest range across parameters is 0.0070 against 0.0145 with the MAP warm
    start. This is the standing guard the convention asks for, not a reproducer
    for that change. The configurations that do freeze are real models at
    catalog scale (#1999).
    """
    from tengri import CatalogFitter

    model = _build_model_multi_d(synthetic_ssp, simple_observation)
    galaxies = _catalog_from_truths(model, _MULTI_D_TRUTHS, jax.random.PRNGKey(0))

    cp = CatalogFitter(model, galaxies, data_type="photometry").run(
        "mcmc_hmc",
        key=jax.random.PRNGKey(1),
        forward_chunk_size=2,
        n_warmup=60,
        n_burnin=0,
        n_samples=60,
        n_leapfrog_steps=16,
        verbose=False,
    )

    for i in range(len(galaxies)):
        for name in model.spec.free_params:
            draws = np.asarray(cp[i].samples[name])
            assert np.all(np.isfinite(draws)), f"galaxy {i}, {name}: non-finite draws"
            assert np.ptp(draws) > 0, f"galaxy {i}, {name}: the chain never moved"


def test_catalog_mass_matrix_follows_the_single_fit_policy():
    """``dense_mass_matrix=None`` resolves through the shared auto-policy (PR #2031).

    The catalog path hardcoded ``False`` and consumed it as
    ``bool(dense_mass_matrix)``, so a D<8 catalog silently got a diagonal mass
    where a single fit of the same model got a dense one -- and passing the
    documented ``None`` default selected diagonal rather than the policy.
    """
    from tengri.inference.backends.mcmc.nuts import _resolve_dense_mass_matrix

    assert _resolve_dense_mass_matrix(None, 3) is True
    assert _resolve_dense_mass_matrix(None, 8) is False
    # An explicit choice still wins, in both directions.
    assert _resolve_dense_mass_matrix(False, 3) is False
    assert _resolve_dense_mass_matrix(True, 20) is True


def test_catalog_init_from_accepts_user_starting_points(synthetic_ssp, simple_observation):
    """``init_from`` takes an array, one dict, or one dict per galaxy.

    Mirrors the single-galaxy contract in ``_maybe_map_init``: ``None`` means
    "find me a starting point", not "start at the prior center".
    """
    from tengri import CatalogFitter

    model = _build_model_multi_d(synthetic_ssp, simple_observation)
    galaxies = _catalog_from_truths(model, _MULTI_D_TRUTHS, jax.random.PRNGKey(0))
    fitter = CatalogFitter(model, galaxies, data_type="photometry")
    kw = {
        "key": jax.random.PRNGKey(1),
        "forward_chunk_size": 2,
        "n_warmup": 20,
        "n_burnin": 0,
        "n_samples": 20,
        "n_leapfrog_steps": 8,
        "verbose": False,
    }
    d_free = len(model.spec.free_params)

    for init_from in ("prior", np.zeros((2, d_free)), _MULTI_D_TRUTHS[0], _MULTI_D_TRUTHS):
        cp = fitter.run("mcmc_hmc", init_from=init_from, **kw)
        for i in range(2):
            arr = np.asarray(cp[i].samples["sfh_dpl_log_total_mass"])
            assert arr.shape[0] == 20
            assert np.all(np.isfinite(arr))


def test_catalog_init_from_rejects_mismatched_counts(synthetic_ssp, simple_observation):
    """A wrong-length ``init_from`` raises rather than broadcasting.

    Reusing one galaxy's starting point for another is invisible downstream: the
    fit runs, the shapes are right, and the posteriors are quietly wrong.
    """
    from tengri import CatalogFitter

    model = _build_model_multi_d(synthetic_ssp, simple_observation)
    galaxies = _catalog_from_truths(model, _MULTI_D_TRUTHS, jax.random.PRNGKey(0))
    fitter = CatalogFitter(model, galaxies, data_type="photometry")
    kw = {
        "key": jax.random.PRNGKey(1),
        "forward_chunk_size": 2,
        "n_warmup": 10,
        "n_samples": 10,
        "verbose": False,
    }

    with pytest.raises(ValueError, match="shape"):
        fitter.run("mcmc_hmc", init_from=np.zeros((3, len(model.spec.free_params))), **kw)
    with pytest.raises(ValueError, match="entries for 2 galaxies"):
        fitter.run("mcmc_hmc", init_from=[_MULTI_D_TRUTHS[0]], **kw)
    with pytest.raises(ValueError, match="init_from must be"):
        fitter.run("mcmc_hmc", init_from="nonsense", **kw)

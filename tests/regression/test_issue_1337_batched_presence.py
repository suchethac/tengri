# SPDX-License-Identifier: BSD-3-Clause
"""Batched MCMC presence masking for heterogeneous catalogs (#1337 Phase 1).

Three adversarial tests (neuter-checked) verifying per-galaxy presence masks
in the batched ``mcmc_nuts`` / ``mcmc_hmc`` catalog path:

1. Gradient-zero inside the BATCHED loss: the per-galaxy logdensity returns zero
   gradient w.r.t. an absent (presence=0) band's data.

2. All-present == no-presence, bit-identical END-TO-END: a catalog fit with
   presence=all-ones on every galaxy produces bit-identical posterior samples
   to the same fit run with no presence key at all.

3. A masked band's DATA VALUE is irrelevant to the batched fit: the same
   galaxy fit through the real ``run_one`` path with band ``k`` masked gives
   bit-identical chains whether band ``k``'s flux is clean or wildly corrupted,
   and diverges from the same corrupted data left unmasked. This is the
   load-bearing guard for ``data_args["presence"] = presence`` in ``run_one``
   (an all-ones mask can never neuter-check masking).
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine
from tengri.inference.catalog_fitter import _CatalogFitterOriginal as CatalogFitter
from tengri.inference.fitter import Fitter


@pytest.mark.regression_bug
class TestBatchedPresenceMasking:
    """Batched presence masking in MCMC catalog path."""

    def test_gradient_zero_on_absent_band(self, ssp_data_wne, synthetic_tophat_obs):
        """Gradient w.r.t. an absent band's data must be zero (load-bearing).

        Verifies that the batched MCMC engine correctly routes presence masks
        through to the likelihood, so an absent band contributes zero to gradients.
        """
        from jax.flatten_util import ravel_pytree

        from tengri import FIXED, FREE, SEDModel

        # Minimal model: all SFH free, everything else fixed.
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust={"type": "two_component", "all_params": FIXED},
            redshift=0.05,
        )

        key = jax.random.PRNGKey(42)
        n_bands = model.observation.n_data

        # Sample random parameters for prediction.
        test_params = model.spec.sample(key)
        pred = model.predict(test_params)
        flux_obs = np.asarray(pred.photometry(), dtype=np.float64)
        noise = 0.05 * np.abs(flux_obs) + 1e-16
        noise = np.maximum(noise, 1e-18)

        # Create fitter for this galaxy.
        fitter = Fitter(model, flux_obs, noise)

        # Build the MCMC engine (NUTS, tiny warmup for speed).
        _run_one, _unravel_fn = build_catalog_mcmc_engine(
            fitter,
            "nuts",
            n_warmup=5,
            n_burnin=2,
            n_samples=1,
            use_dense=False,
        )

        # Construct presence mask: first band absent (presence[0] = 0.0).
        presence = np.ones(n_bands, dtype=np.float64)
        presence[0] = 0.0

        # Initialize parameters.
        init_params = fitter._initialize_unbounded(key)
        init_flat, _ = ravel_pytree(init_params)

        # Extract the per-galaxy log-posterior by calling run_one's internals.
        from tengri.inference.backends.mcmc._shared import _get_flat_logdensity

        log_posterior_flat_2arg, _, _, template_data_args = _get_flat_logdensity(
            fitter, init_params
        )

        # Assemble data_args with presence mask.
        noise_inv = 1.0 / (noise**2)
        data_args = dict(template_data_args)
        data_args["data"] = flux_obs
        data_args["noise"] = noise
        data_args["noise_inv"] = noise_inv
        data_args["sqrt_noise_inv"] = jnp.sqrt(noise_inv)
        data_args["presence"] = presence

        # Compute gradient w.r.t. data[0] (absent band) at the initialization.
        def logposterior_vs_data(data_flat):
            da = dict(data_args)
            da["data"] = data_flat
            return log_posterior_flat_2arg(init_flat, da)

        # Finiteness only, deliberately: the observed fluxes here are ~1e-26, so
        # a step proportional to them is ~1e-31 and f(x+h) comes back
        # bit-identical to f(x-h) — the numerical derivative underflows to zero
        # while the analytic one is ~-3.6e5. That is the probe hitting the
        # floating-point floor, not evidence about the gradient.
        grad_wrt_data = jax.grad(logposterior_vs_data)(flux_obs)

        # For an absent band (presence[0] = 0.0), the gradient must be exactly zero.
        print(f"Gradient w.r.t. absent band data[0]: {grad_wrt_data[0]}")
        assert np.abs(grad_wrt_data[0]) < 1e-14, (
            f"Gradient w.r.t. absent band data should be zero, got {grad_wrt_data[0]}"
        )
        # For a present band, the gradient should be nonzero (sanity check).
        print(f"Gradient w.r.t. present band data[1]: {grad_wrt_data[1]}")
        assert np.abs(grad_wrt_data[1]) > 1e-10, (
            f"Gradient w.r.t. present band data should be nonzero, got {grad_wrt_data[1]}"
        )

    def test_all_present_bit_identical_to_no_presence(self, ssp_data_wne, synthetic_tophat_obs):
        """All-ones presence must give bit-identical results to no presence (END-TO-END).

        A catalog fit with presence=all-ones on every galaxy must produce
        bit-identical posterior samples to the same fit run with no presence key.
        """
        from tengri import FIXED, FREE, SEDModel

        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust={"type": "two_component", "all_params": FIXED},
            redshift=0.05,
        )

        key = jax.random.PRNGKey(42)
        n_bands = model.observation.n_data

        # Sample random parameters for prediction.
        test_params = model.spec.sample(key)
        pred = model.predict(test_params)
        flux_obs = np.asarray(pred.photometry(), dtype=np.float64)
        noise = 0.05 * np.abs(flux_obs) + 1e-16
        noise = np.maximum(noise, 1e-18)

        # Build two catalogs: one with presence=all-ones, one with no presence.
        cat_with_presence = [
            {
                "flux_obs": flux_obs,
                "noise": noise,
                "presence": np.ones(n_bands),  # All bands present
            }
        ]
        cat_without_presence = [
            {
                "flux_obs": flux_obs,
                "noise": noise,
                # No presence key
            }
        ]

        # Fit both with the same key and tiny NUTS (5 warmup, 5 samples for speed).
        key1 = jax.random.fold_in(key, 1)

        fitter_with = CatalogFitter(model, cat_with_presence)
        result_with = fitter_with.run(
            "mcmc_nuts",
            key=key1,
            n_warmup=5,
            n_burnin=2,
            n_samples=5,
            verbose=False,
        )

        fitter_without = CatalogFitter(model, cat_without_presence)
        result_without = fitter_without.run(
            "mcmc_nuts",
            key=key1,  # Same key
            n_warmup=5,
            n_burnin=2,
            n_samples=5,
            verbose=False,
        )

        # Compare per-galaxy posterior samples.
        # Both should have one galaxy with the same samples.
        assert len(result_with) == 1
        assert len(result_without) == 1

        samples_with = result_with[0].samples
        samples_without = result_without[0].samples

        # Samples must match exactly (bit-identical within floating-point epsilon).
        print("Checking bit-identical samples...")
        chex.assert_trees_all_equal(samples_with, samples_without)
        print("✓ Samples are bit-identical")

    def test_masked_band_value_irrelevant_through_run_one(
        self, ssp_data_wne, synthetic_tophat_obs
    ):
        """A masked band's DATA VALUE must not affect the batched fit (LOAD-BEARING).

        This is the test that actually guards this task's change
        (``data_args["presence"] = presence`` inside the batched ``run_one``).
        It drives the real ``CatalogFitter.run("mcmc_nuts")`` path — not a
        hand-assembled ``data_args`` — and uses an *absent band whose flux is
        wildly corrupted*, so applying vs. ignoring the mask gives divergent
        objectives (an all-ones mask can never neuter-check masking; see #1337).

        Two independent, deterministic assertions:

        1. **Masked value irrelevant.** The same galaxy fit twice, band ``k``
           masked in both, differing only in band ``k``'s flux (corrupt vs.
           clean): a correctly-applied mask makes band ``k``'s value contribute
           nothing, so the two chains are **bit-identical**. If ``run_one``
           drops the mask, band ``k`` leaks in and the corrupt/clean chains
           diverge — this assertion fails.
        2. **Mask has a real effect.** Masked vs. all-ones on the *same*
           corrupted data must diverge (the mask removed band ``k``'s large
           residual). Under the neuter both collapse to all-ones → identical.
        """
        from tengri import FIXED, FREE, SEDModel

        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust={"type": "two_component", "all_params": FIXED},
            redshift=0.05,
        )

        key = jax.random.PRNGKey(7)
        n_bands = model.observation.n_data

        # Model-scale data (the campaign's degenerate-data trap: ~1e-15 data
        # makes the mask invisible). Generate observed flux from the model.
        pred = model.predict(model.spec.sample(key))
        flux_clean = np.asarray(pred.photometry(), dtype=np.float64)
        noise = np.maximum(0.05 * np.abs(flux_clean) + 1e-16, 1e-18)

        # Corrupt the last band so its residual is enormous when NOT masked.
        k = n_bands - 1
        flux_corrupt = flux_clean.copy()
        flux_corrupt[k] = flux_clean[k] * 50.0 + 100.0 * noise[k]

        presence_masked = np.ones(n_bands)
        presence_masked[k] = 0.0
        presence_allones = np.ones(n_bands)

        fit_key = jax.random.fold_in(key, 3)
        nuts_kw = dict(n_warmup=20, n_burnin=5, n_samples=20, verbose=False)

        def _fit(flux, presence):
            # Same noise (from the clean flux) in every case, so the ONLY input
            # difference is flux[k] and/or the presence vector.
            cat = [{"flux_obs": flux, "noise": noise, "presence": presence}]
            return CatalogFitter(model, cat).run("mcmc_nuts", key=fit_key, **nuts_kw)[0].samples

        s_masked_corrupt = _fit(flux_corrupt, presence_masked)
        s_masked_clean = _fit(flux_clean, presence_masked)
        s_allones_corrupt = _fit(flux_corrupt, presence_allones)

        # (1) Masking band k makes its VALUE irrelevant → corrupt ≡ clean.
        #     Neuter (run_one ignores presence): band k leaks → NOT identical → fail.
        chex.assert_trees_all_equal(s_masked_corrupt, s_masked_clean)
        print("✓ masked band's value is irrelevant (corrupt ≡ clean, bit-identical)")

        # (2) The mask has a measurable effect on the SAME corrupted data.
        #     Neuter: masked collapses to all-ones → identical → max_diff == 0 → fail.
        leaves_m = jax.tree_util.tree_leaves(s_masked_corrupt)
        leaves_a = jax.tree_util.tree_leaves(s_allones_corrupt)
        max_diff = max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(leaves_m, leaves_a)
        )
        print(f"max |masked − all-ones| over samples (corrupted band): {max_diff}")
        assert max_diff > 1e-6, (
            "Masking a corrupted band had NO effect on the batched fit → run_one is "
            "not applying the per-galaxy presence mask (data_args['presence'] plumbing bug)."
        )

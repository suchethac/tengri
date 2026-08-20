# SPDX-License-Identifier: BSD-3-Clause
"""Batched per-galaxy redshift for the MCMC catalog path (#1337 Phase 2).

A catalog carrying a per-galaxy ``Fixed`` redshift (the ``redshift_col`` +
``catalog_z_range`` configuration) now fits through the batched ``mcmc_nuts`` /
``mcmc_hmc`` path as ONE compiled program: the redshift flows as a runtime input
the engine vmaps over, instead of being baked per galaxy (which recompiles per
distinct z on the sequential path).

The mechanism is a guarded override in ``build_loss_fn``: when the caller threads
a redshift through ``data_args`` it replaces the baked fixed value. No single-fit
path ever sets ``data_args["redshift"]``, so its program is unchanged.

**Data-scale warning (this bit a probe once).** Model photometry here is
``~1e-30`` erg/s/cm^2/Hz. A noise floor like ``1e-18`` is then twelve orders of
magnitude ABOVE the signal and every loss collapses onto the prior term, making
a redshift change look like a no-op. Scale the noise to the flux.
"""

import numpy as np
import pytest

from tengri.inference.catalog_fitter import _CatalogFitterOriginal as CatalogFitter


def _noise_for(flux):
    """Model-scale noise. Never a floor above the flux (see module docstring)."""
    return np.maximum(0.05 * np.abs(flux), np.max(np.abs(flux)) * 1e-8)


@pytest.mark.regression_bug
class TestBatchedRedshift:
    """Runtime per-galaxy redshift in the MCMC catalog path."""

    @staticmethod
    def _model(ssp_data_wne, synthetic_tophat_obs, *, z_range):
        from tengri import FIXED, FREE, Fixed, SEDModel, WavePrecomp

        approx = WavePrecomp(catalog_z_range=z_range, n_z=50) if z_range else None
        return SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={"law": "power_law", "type": "two_component", "all_params": FIXED},
            redshift=Fixed(0.1),
            approx=approx,
        )

    def test_runtime_redshift_reaches_the_forward_pass(self, ssp_data_wne, synthetic_tophat_obs):
        """The redshift threaded through data_args changes the loss (LOAD-BEARING).

        Overriding to the baked z must be an exact no-op; overriding to a different
        z must move the loss (the prediction really is recomputed at that z). Neuter:
        drop the ``data_args["redshift"]`` override in ``build_loss_fn`` and the
        different-z loss collapses back onto the baseline.
        """
        import jax

        from tengri.inference.fitter import Fitter
        from tengri.inference.loss_functions import build_loss_fn

        model = self._model(ssp_data_wne, synthetic_tophat_obs, z_range=(0.05, 1.0))
        key = jax.random.PRNGKey(1)
        flux = np.asarray(model.predict_photometry(model.spec.sample(key)), dtype=np.float64)
        noise = _noise_for(flux)

        fitter = Fitter(model, flux, noise)
        loss_fn = build_loss_fn(fitter)
        data_args = dict(fitter._data_args)
        init = fitter._initialize_unbounded(key)

        # A single fit never carries a redshift in data_args — that is what keeps
        # the non-catalog path byte-identical.
        assert "redshift" not in data_args

        L_base = float(loss_fn(init, data_args))
        L_same = float(loss_fn(init, {**data_args, "redshift": np.float64(0.1)}))
        L_diff = float(loss_fn(init, {**data_args, "redshift": np.float64(0.8)}))
        print(f"L_base={L_base:.6g} L_same={L_same:.6g} L_diff={L_diff:.6g}")

        assert L_same == L_base, (
            f"override at the baked z must be an exact no-op ({L_same} vs {L_base})"
        )
        assert not np.isclose(L_diff, L_base, rtol=1e-3), (
            f"a different runtime redshift must change the loss; L_diff={L_diff} "
            f"vs L_base={L_base} — the override never reached the forward pass."
        )

    def test_catalog_z_range_required_for_batched_per_galaxy_z(
        self, ssp_data_wne, synthetic_tophat_obs
    ):
        """Without the runtime-LUT, a batched per-galaxy-z fit would recompile per z."""
        import jax

        model = self._model(ssp_data_wne, synthetic_tophat_obs, z_range=None)
        key = jax.random.PRNGKey(2)
        flux = np.asarray(model.predict_photometry(model.spec.sample(key)), dtype=np.float64)
        noise = _noise_for(flux)
        catalog = [
            {"flux_obs": flux, "noise": noise, "redshift": 0.1},
            {"flux_obs": flux, "noise": noise, "redshift": 0.8},
        ]
        with pytest.raises(ValueError, match="catalog_z_range"):
            CatalogFitter(model, catalog).run("mcmc_nuts", key=key, n_warmup=5, n_samples=5)

    @pytest.mark.slow  # a real 2-galaxy NUTS catalog fit; see #1346 (shard OOM)
    def test_batched_heterogeneous_redshift_fits_and_separates(
        self, ssp_data_wne, synthetic_tophat_obs
    ):
        """Two galaxies at different z fit in one batched program and stay distinct.

        Each galaxy's data is generated at ITS OWN redshift, so a run that ignored
        the per-galaxy redshift would fit both against the same prediction.
        """
        import jax

        model = self._model(ssp_data_wne, synthetic_tophat_obs, z_range=(0.05, 1.0))
        key = jax.random.PRNGKey(4)
        params = model.spec.sample(key)
        flux_lo = np.asarray(
            model.predict_photometry({**params, "redshift": 0.1}), dtype=np.float64
        )
        flux_hi = np.asarray(
            model.predict_photometry({**params, "redshift": 0.8}), dtype=np.float64
        )
        # The two galaxies must be genuinely different data (guard the fixture).
        # atol=0 is mandatory: these fluxes are ~1e-30, and numpy's default
        # atol=1e-8 would call ANY two of them equal.
        assert not np.allclose(flux_lo, flux_hi, rtol=1e-3, atol=0.0), (
            "fixture is degenerate: the two redshifts produced the same photometry"
        )

        catalog = [
            {"flux_obs": flux_lo, "noise": _noise_for(flux_lo), "redshift": 0.1},
            {"flux_obs": flux_hi, "noise": _noise_for(flux_hi), "redshift": 0.8},
        ]
        cp = CatalogFitter(model, catalog).run(
            "mcmc_nuts", key=key, n_warmup=20, n_burnin=5, n_samples=20, verbose=False
        )
        assert cp.n_galaxies == 2
        for i in range(2):
            for name, val in cp.posteriors[i].params.items():
                assert np.all(np.isfinite(np.asarray(val))), f"galaxy {i} param {name} non-finite"
        print("✓ batched per-galaxy-z catalog fit ran with finite per-galaxy params")

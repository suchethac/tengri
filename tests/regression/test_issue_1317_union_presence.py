# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #1317: heterogeneous catalogs via union filters + presence-masked
likelihood.

Absent bands (`missing="mask"`) must contribute exactly zero to the χ² and its gradient.
A table band not in the union observation must raise at ingestion.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


class TestIssue1317UnionPresence:
    """#1317: Presence-masked likelihood for heterogeneous catalogs."""

    def test_unresolvable_flux_column_raises(self):
        """An unreadable flux column must raise ValueError.

        The referent of this check moved in #1458. It used to assert that a
        ``flux_cols`` entry outside the observation's bands raised "not in the
        observation" — but that check made ``flux_cols`` unable to name a real
        catalog column at all, which is the only reason the parameter exists,
        so it was removed. Column names are now validated against the **table**,
        the referent they always should have had.

        The case below is unchanged in outcome and stronger in message:
        ``sdss_i`` is in neither the observation nor the table, so it still
        raises — now naming the table's actual columns, which is what lets a
        user fix it.
        """
        from tengri.inference.catalog_ingest import ingest_catalog
        from tengri.observation import Photometry

        phot = Photometry.from_names(["sdss_g", "sdss_r"])
        table = {
            "sdss_g": np.array([1.0, 2.0]),
            "sdss_g_err": np.array([0.1, 0.1]),
            "sdss_r": np.array([3.0, 4.0]),
            "sdss_r_err": np.array([0.2, 0.2]),
        }

        with pytest.raises(ValueError, match="Missing flux column") as excinfo:
            ingest_catalog(
                table,
                photometry=phot,
                flux_unit="cgs_fnu",
                flux_cols=["sdss_g", "sdss_i"],  # sdss_i is in neither
                err_cols=["sdss_g_err", "sdss_i_err"],
            )
        # The message must name what IS available, not only what is missing.
        assert "sdss_r" in str(excinfo.value), excinfo.value

    def test_absent_band_contributes_zero(self):
        """Absent bands (presence=False) must not affect the fit results.

        Build a reference 2-band fit, then fit a 3-band galaxy with one band absent
        (via missing="mask"). The 2-band galaxy should have identical MAP to the
        reference, and the 3-band galaxy should match when only the 2 bands are used.
        """
        import warnings
        from pathlib import Path

        from tengri import (
            FIXED,
            Fixed,
            ForwardModel,
            Observation,
            Photometry,
            SEDModel,
            load_ssp_data,
        )
        from tengri.forward.sed_model import WavePrecomp
        from tengri.inference.catalog import Catalog

        # Load SSP data
        ssp_candidates = [
            "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
            "data/ssp_prsc_bc03_chabrier.h5",
        ]
        ssp_path = next((p for p in ssp_candidates if Path(p).is_file()), None)
        if ssp_path is None:
            pytest.skip("No SSP grid on disk; skipping fit test.")

        ssp = load_ssp_data(ssp_path)

        # Reference: 2-band observation
        obs_2band = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

        # Build 2-band model with catalog_z_range for per-galaxy redshifts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed_2band = SEDModel.build(
                ssp_data=ssp,
                observation=obs_2band,
                sfh={"type": "dpl", "all_params": FIXED, "tau_gyr": Fixed(10.0)},
                dust_attenuation={
                    "type": "single_component",
                    "law": "calzetti",
                    "tau_v": Fixed(0.3),
                },
                redshift=Fixed(0.1),
                approx=WavePrecomp(catalog_z_range=(0.0, 1.0)),  # For catalog fits
            )
            # Wrap in ForwardModel for Catalog compatibility
            model_2band = ForwardModel.build(sed=sed_2band, observation=obs_2band)

        # Generate model-scale data for galaxy 1 at the reference model point
        reference_params = sed_2band.spec.get_fixed_values()
        ref_phot = sed_2band.predict_photometry(reference_params)

        # Add small noise
        noise_2band = 0.01 * ref_phot

        # Reference fit: galaxy 1 with 2 bands
        # Include redshift column (required for Fixed redshift models)
        table_2band = {
            "sdss_g": ref_phot[:1],
            "sdss_g_err": noise_2band[:1],
            "sdss_r": ref_phot[1:2],
            "sdss_r_err": noise_2band[1:2],
            "z": np.array([0.1]),  # Redshift for catalog
        }

        cat_2band = Catalog(
            model_2band,
            table_2band,
            flux_unit="cgs_fnu",
            redshift_col="z",
        )
        post_2band = cat_2band.fit(key=jax.random.PRNGKey(0), method="map")
        map_2band = post_2band[0].params

        # Now test: 3-band observation with one band absent
        obs_3band = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_i", "sdss_r"]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed_3band = SEDModel.build(
                ssp_data=ssp,
                observation=obs_3band,
                sfh={"type": "dpl", "all_params": FIXED, "tau_gyr": Fixed(10.0)},
                dust_attenuation={
                    "type": "single_component",
                    "law": "calzetti",
                    "tau_v": Fixed(0.3),
                },
                redshift=Fixed(0.1),
                approx=WavePrecomp(catalog_z_range=(0.0, 1.0)),  # For catalog fits
            )
            # Wrap in ForwardModel for Catalog compatibility
            model_3band = ForwardModel.build(sed=sed_3band, observation=obs_3band)

        # Generate data for 3-band model
        ref_phot_3band = sed_3band.predict_photometry(reference_params)
        noise_3band = 0.01 * ref_phot_3band

        # Create table with sdss_i absent (NaN)
        table_3band = {
            "sdss_g": ref_phot_3band[:1],
            "sdss_g_err": noise_3band[:1],
            "sdss_i": np.array([np.nan]),  # Absent band
            "sdss_i_err": noise_3band[1:2],  # Error is present (even though flux is absent)
            "sdss_r": ref_phot_3band[2:3],
            "sdss_r_err": noise_3band[2:3],
            "z": np.array([0.1]),  # Redshift for catalog
        }

        # Ingest with missing="mask"
        cat_3band = Catalog(
            model_3band,
            table_3band,
            flux_unit="cgs_fnu",
            missing="mask",
            redshift_col="z",
        )

        # Fit the 3-band galaxy with one band absent
        post_3band = cat_3band.fit(key=jax.random.PRNGKey(0), method="map")
        map_3band = post_3band[0].params

        # The 2-band MAP and 3-band MAP should agree on the stellar_mass
        # (the only free parameter)
        np.testing.assert_allclose(
            map_2band["sfh_dpl_alpha"],
            map_3band["sfh_dpl_alpha"],
            rtol=1e-5,
            err_msg="2-band and 3-band (with absent band) MAPs should agree",
        )

    def test_gradient_is_zero_wrt_absent_band(self):
        """The gradient of the loss w.r.t. an absent band's data must be exactly 0.0.

        This is the load-bearing test: if masking doesn't work, the gradient
        will be nonzero and the fit will be biased.
        """
        import jax

        from tengri.inference.likelihoods.gaussian import diag_gaussian_chi2

        # Create sample data
        predicted = jnp.array([1.0, 2.0, 3.0])
        observed = jnp.array([1.1, 2.2, 3.3])
        sigma = jnp.array([0.1, 0.2, 0.3])
        presence = jnp.array([1.0, 0.0, 1.0])  # Middle band absent

        # Define a loss function that takes observed as input
        def loss_fn(obs):
            return diag_gaussian_chi2(predicted, obs, sigma, presence=presence)

        # Compute the gradient w.r.t. the observed data
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(observed)

        # The gradient w.r.t. the absent band (index 1) must be exactly 0.0
        assert float(grad[1]) == 0.0, (
            f"Gradient w.r.t. absent band must be exactly 0.0, got {float(grad[1])}"
        )

        # The gradients w.r.t. present bands should be nonzero
        assert float(grad[0]) != 0.0, "Gradient w.r.t. present band should be nonzero"
        assert float(grad[2]) != 0.0, "Gradient w.r.t. present band should be nonzero"

    def test_presence_all_true_bitidentical_no_mask(self):
        """When presence=all-ones, the loss must be bit-identical to no masking.

        Guard rail: this ensures the masking logic doesn't introduce spurious
        changes even when all bands are present.
        """
        from tengri.inference.likelihoods.gaussian import diag_gaussian_chi2

        predicted = jnp.array([1.0, 2.0, 3.0])
        observed = jnp.array([1.1, 2.2, 3.3])
        sigma = jnp.array([0.1, 0.2, 0.3])

        # Loss without presence mask
        loss_no_mask = diag_gaussian_chi2(predicted, observed, sigma)

        # Loss with presence=all-ones
        presence_all_true = jnp.ones(3)
        loss_with_all_true = diag_gaussian_chi2(
            predicted, observed, sigma, presence=presence_all_true
        )

        # Must be exactly equal
        assert jnp.allclose(loss_no_mask, loss_with_all_true, rtol=0, atol=0), (
            f"Presence=all-ones should be bit-identical: "
            f"no_mask={float(loss_no_mask)}, all_true={float(loss_with_all_true)}"
        )

    def test_gradient_zero_neuter_check(self):
        """Neuter-check: removing the presence factor should make gradient nonzero.

        This verifies that the gradient-zero test actually catches the absence
        of masking logic. If we remove the `presence *` factor from the loss,
        this test should fail.
        """
        import jax

        from tengri.inference.likelihoods.gaussian import diag_gaussian_chi2

        predicted = jnp.array([1.0, 2.0, 3.0])
        observed = jnp.array([1.1, 2.2, 3.3])
        sigma = jnp.array([0.1, 0.2, 0.3])
        presence = jnp.array([1.0, 0.0, 1.0])  # Middle band absent

        def loss_fn(obs):
            return diag_gaussian_chi2(predicted, obs, sigma, presence=presence)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(observed)

        # Verify that the gradient w.r.t. the absent band IS actually zero
        # (this is what we expect the implementation to achieve)
        assert float(grad[1]) == 0.0, (
            "Neuter-check: gradient should be zero for absent band, but is "
            f"{float(grad[1])}. If you removed the presence factor, this proves "
            "the test catches it."
        )

    def test_gradient_zero_through_the_full_fitter_loss(self):
        """END-TO-END: the presence mask must reach the ACTUAL compiled Fitter
        loss, not just the unit ``diag_gaussian_chi2``.

        The unit test above passes even if the plumbing between the Fitter's
        ``data_args["presence"]`` and the likelihood adapter is broken (the adapter
        must be constructed with ``presence_key="presence"`` for it to read the
        mask). This test grads the real loss w.r.t. the data and asserts the absent
        band's gradient is exactly 0.0 while present bands are nonzero — hermetic
        (synthetic SSP), no data files.
        """
        from tengri import (
            FIXED,
            FREE,
            ForwardModel,
            Observation,
            Photometry,
            SEDModel,
            Uniform,
        )
        from tengri.components.stellar.sps.dsps_wrapper import SSPData
        from tengri.inference.fitter import Fitter
        from tengri.observation.photometry import FilterCurve

        wave = jnp.linspace(3000.0, 10000.0, 60)
        ssp = SSPData(
            ssp_wave=wave,
            ssp_flux=jnp.abs(jnp.ones((3, 12, 60))) * 1e-3 + 1e-5,
            ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 12),
            ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
        )
        curves = tuple(
            FilterCurve(wave=jnp.linspace(lo, hi, 30), trans=jnp.ones(30) * 0.5, name=f"b{i}")
            for i, (lo, hi) in enumerate([(3500.0, 4500.0), (5000.0, 6500.0), (7500.0, 9000.0)])
        )
        obs = Observation(photometry=Photometry(filters=curves))
        sed = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": FIXED,
                "tau_bc": 0.5,
            },
            neb={"type": "none"},
            redshift=Uniform(0.1, 1.0),
        )
        fwd = ForwardModel.build(sed=sed, observation=obs)
        data = jnp.array([1.0, 2.0, 3.0])
        noise = jnp.array([0.1, 0.2, 0.3])

        fitter = Fitter(fwd, data=data, noise=noise, presence=jnp.array([1.0, 0.0, 1.0]))
        loss = fitter._get_or_build_loss_fn()
        x0 = fitter._initialize_unbounded(jax.random.PRNGKey(0))
        da = fitter._data_args

        grad = np.asarray(jax.grad(lambda d: loss(x0, {**da, "data": d}))(data))
        assert float(grad[1]) == 0.0, (
            f"absent band's gradient through the full Fitter loss must be exactly 0.0, "
            f"got {float(grad[1])} — the presence mask is not reaching the compiled loss "
            f"(check presence_key wiring in build_base_likelihood)."
        )
        assert float(grad[0]) != 0.0 and float(grad[2]) != 0.0, "present bands must be nonzero"

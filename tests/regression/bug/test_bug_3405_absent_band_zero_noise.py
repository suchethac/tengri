# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #3405: an absent band with zero noise must not make the likelihood NaN.

``catalog_ingest`` stores an absent band (``missing="mask"``) with flux 0, noise 0 and
presence 0. Whitening that band divides by zero, and ``0 * NaN`` is NaN, so the value and
the gradient both went non-finite. An absent band must contribute exactly zero, and a
galaxy whose MAP comes back non-finite must be reported as a failed fit, not as frozen.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference import mass_profile
from tengri.inference.catalog_convergence import catalog_convergence, galaxy_health
from tengri.inference.likelihoods.gaussian import diag_gaussian_chi2
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.regression_bug


class TestGaussianChi2AbsentZeroNoise:
    """Site 2: the diagonal-Gaussian chi-square with an absent zero-noise band."""

    def test_value_is_finite_and_equals_two_band_value(self):
        predicted = jnp.array([1.0, 2.0, 3.0])
        observed = jnp.array([1.1, 2.2, 3.3])
        sigma = jnp.array([1.0, 0.0, 1.0])
        presence = jnp.array([1.0, 0.0, 1.0])

        chi2_3 = diag_gaussian_chi2(predicted, observed, sigma, presence=presence)
        chi2_2 = diag_gaussian_chi2(
            jnp.array([1.0, 3.0]),
            jnp.array([1.1, 3.3]),
            jnp.array([1.0, 1.0]),
        )

        assert np.isfinite(float(chi2_3))
        np.testing.assert_allclose(float(chi2_3), float(chi2_2), rtol=1e-12)

    def test_nan_flux_in_absent_band_does_not_leak_into_value(self):
        predicted = jnp.array([1.0, 2.0, 3.0])
        observed = jnp.array([1.1, jnp.nan, 3.3])
        sigma = jnp.array([1.0, 0.0, 1.0])
        presence = jnp.array([1.0, 0.0, 1.0])

        chi2_3 = diag_gaussian_chi2(predicted, observed, sigma, presence=presence)
        chi2_2 = diag_gaussian_chi2(
            jnp.array([1.0, 3.0]),
            jnp.array([1.1, 3.3]),
            jnp.array([1.0, 1.0]),
        )

        np.testing.assert_allclose(float(chi2_3), float(chi2_2), rtol=1e-12)

    def test_gradient_wrt_predicted_is_finite_and_zero_on_absent_band(self):
        predicted = jnp.array([1.0, 2.0, 3.0])
        observed = jnp.array([1.1, 2.2, 3.3])
        sigma = jnp.array([1.0, 0.0, 1.0])
        presence = jnp.array([1.0, 0.0, 1.0])

        grad = jax.grad(lambda mu: diag_gaussian_chi2(mu, observed, sigma, presence=presence))(
            predicted
        )

        assert np.all(np.isfinite(np.asarray(grad)))
        assert float(grad[1]) == 0.0


class TestMultivariateGaussianAbsentBand:
    """Site 3: the banded/dense multivariate-Gaussian likelihood with a NaN absent pixel."""

    def test_absent_pixel_with_nan_data_gives_two_pixel_value(self):
        from tengri.inference.likelihoods.protocol import MultivariateGaussianLikelihood

        lik = MultivariateGaussianLikelihood(
            obs=jnp.array([1.0, jnp.nan, 3.0]),
            cov_inv=jnp.eye(3),
            presence_key="presence",
        )
        pred = {"spec_fnu": jnp.array([1.5, 2.0, 2.5])}
        data_args = {"presence": jnp.array([1.0, 0.0, 1.0])}

        got = float(lik.log_prob(pred, None, data_args))

        np.testing.assert_allclose(got, -0.5 * (0.5**2 + 0.5**2), rtol=1e-12)


class TestProfileStatsAbsentZeroNoise:
    """Site 1: the profiled mass statistics with an absent zero-noise band."""

    def test_stats_finite_and_equal_to_band_deleted(self, monkeypatch):
        phys = {"logM": jnp.asarray(10.0)}

        def stats_with_unit_prediction(unit_pred, data, noise, presence):
            # Linear in M = 10**phys["logM"]: pred = M * unit_pred, as the profiled path assumes.
            def fake_predict(model, data_type, phys, **kwargs):
                return (10.0 ** phys["logM"]) * jnp.asarray(unit_pred)

            monkeypatch.setattr(mass_profile, "_predict_full_vector", fake_predict)
            return mass_profile._profile_stats(
                None, "logM", phys, data, noise, presence, data_type="phot"
            )

        A, a_star, chi2_min, _ = stats_with_unit_prediction(
            np.array([0.5, 0.8, 1.3]),
            jnp.array([2.0, 0.0, 4.0]),
            jnp.array([0.5, 0.0, 1.0]),
            jnp.array([1.0, 0.0, 1.0]),
        )
        A2, a2, chi2_2, _ = stats_with_unit_prediction(
            np.array([0.5, 1.3]),
            jnp.array([2.0, 4.0]),
            jnp.array([0.5, 1.0]),
            None,
        )

        for value in (A, a_star, chi2_min):
            assert np.isfinite(float(value))
        np.testing.assert_allclose(float(A), float(A2), rtol=1e-12)
        np.testing.assert_allclose(float(a_star), float(a2), rtol=1e-12)
        np.testing.assert_allclose(float(chi2_min), float(chi2_2), rtol=1e-12)


def _fit_posterior(params, final_loss, *, map_converged=False):
    """A MAP-shaped Posterior, as ``map_dispatch`` builds one: no samples, loss in diagnostics."""
    return Posterior(
        samples=None,
        params=params,
        method="MAP (test)",
        wall_time_s=0.0,
        diagnostics={
            "final_loss": final_loss,
            "map_converged": map_converged,
            "n_steps": 10,
        },
    )


class TestNonFiniteFitIsFailed:
    """The convergence summary must not call a NaN MAP 'frozen'."""

    def test_non_finite_loss_is_unconverged_not_frozen(self):
        post = _fit_posterior({"sfh_dpl_log_total_mass": 10.0}, float("nan"))

        health = galaxy_health(post, index=0)

        assert health.verdict == "unconverged"
        assert "returned a non-finite fit" in health.reason

    def test_non_finite_parameter_is_unconverged(self):
        post = _fit_posterior({"sfh_dpl_log_total_mass": float("nan")}, 1.0)

        health = galaxy_health(post, index=0)

        assert health.verdict == "unconverged"
        assert "returned a non-finite fit" in health.reason

    def test_catalog_summary_counts_non_finite_fit_as_failed(self):
        posts = [_fit_posterior({"a": float("nan")}, float("nan"))]

        conv = catalog_convergence(posts)

        assert conv.n_frozen == 0
        assert conv.n_unconverged == 1
        assert "returned a non-finite fit" in conv.summary()


class TestCatalogAbsentBandEndToEnd:
    """A catalog with one absent zero-noise band fits to a finite MAP."""

    def test_map_fit_with_absent_band_is_finite_and_converged(self):
        import warnings

        from tengri import (
            DEFAULT,
            Fixed,
            ForwardModel,
            Observation,
            Photometry,
            SEDModel,
            Uniform,
            load_ssp_data,
        )
        from tengri.forward.sed_model import WavePrecomp
        from tengri.inference.catalog import Catalog

        ssp_path = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
        if not ssp_path.is_file():
            pytest.skip("No SSP grid on disk; skipping fit test.")
        ssp = load_ssp_data(str(ssp_path))

        names = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1"]
        obs = Observation(photometry=Photometry.from_names(names))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed = SEDModel.build(
                ssp_data=ssp,
                observation=obs,
                sfh={
                    "type": "dpl",
                    "all_params": Fixed(DEFAULT),
                    "tau_gyr": Fixed(10.0),
                    "log_total_mass": Uniform(8.0, 12.0),
                },
                dust_attenuation={
                    "type": "single_component",
                    "law": "calzetti",
                    "tau_v": Fixed(0.3),
                },
                redshift=Fixed(0.1),
                approx=WavePrecomp(catalog_z_range=(0.0, 1.0)),
            )
            model = ForwardModel.build(sed=sed, observation=obs)

        ref = np.asarray(sed.predict_photometry({"sfh_dpl_log_total_mass": 10.0}))
        noise = 0.02 * ref
        absent = names.index("sdss_i")

        table = {"z": np.array([0.1, 0.1])}
        for j, name in enumerate(names):
            flux = np.array([ref[j], ref[j]], dtype=float)
            err = np.array([noise[j], noise[j]], dtype=float)
            if j == absent:
                flux[0] = np.nan
                err[0] = np.nan
            table[name] = flux
            table[f"{name}_err"] = err

        cat = Catalog(model, table, flux_unit="cgs_fnu", missing="mask", redshift_col="z")
        post = cat.fit(key=jax.random.PRNGKey(0), method="map")

        galaxy0 = post[0]
        for value in galaxy0.params.values():
            assert np.all(np.isfinite(np.asarray(value)))
        assert galaxy0.diagnostics["map_converged"] is True


def _ingest_phot(names):
    from tengri.observation import Photometry

    return Photometry.from_names(names)


class TestNonFiniteFluxIsAbsent:
    """A non-finite flux under ``missing='mask'`` is an absent band, exactly like NaN."""

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_non_finite_flux_and_error_ingest_like_nan(self, bad):
        from tengri.inference.catalog_ingest import ingest_catalog

        names = ["sdss_g", "sdss_r"]

        def table(flux_bad, err_bad):
            return {
                "sdss_g": np.array([1.0, 2.0]),
                "sdss_g_err": np.array([0.1, 0.1]),
                "sdss_r": np.array([3.0, flux_bad]),
                "sdss_r_err": np.array([0.2, err_bad]),
            }

        ca_nan = ingest_catalog(
            table(np.nan, np.nan),
            photometry=_ingest_phot(names),
            flux_unit="cgs_fnu",
            missing="mask",
        )
        ca_bad = ingest_catalog(
            table(bad, bad),
            photometry=_ingest_phot(names),
            flux_unit="cgs_fnu",
            missing="mask",
        )

        assert not bool(ca_bad.presence[1, 1])
        np.testing.assert_array_equal(ca_bad.presence, ca_nan.presence)
        np.testing.assert_array_equal(ca_bad.flux, ca_nan.flux)
        np.testing.assert_array_equal(ca_bad.noise, ca_nan.noise)
        assert np.all(np.isfinite(ca_bad.flux)) and np.all(np.isfinite(ca_bad.noise))

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_non_finite_flux_is_refused_under_error_policy(self, bad):
        from tengri.inference.catalog_ingest import ingest_catalog

        t = {
            "sdss_g": np.array([1.0, 2.0]),
            "sdss_g_err": np.array([0.1, 0.1]),
            "sdss_r": np.array([3.0, bad]),
            "sdss_r_err": np.array([0.2, 0.2]),
        }
        with pytest.raises(ValueError, match=r"missing=.mask."):
            ingest_catalog(t, photometry=_ingest_phot(["sdss_g", "sdss_r"]), flux_unit="cgs_fnu")

    def test_finite_flux_with_infinite_error_is_still_an_error(self):
        from tengri.inference.catalog_ingest import ingest_catalog

        t = {
            "sdss_g": np.array([1.0, 2.0]),
            "sdss_g_err": np.array([0.1, 0.1]),
            "sdss_r": np.array([3.0, 4.0]),
            "sdss_r_err": np.array([0.2, np.inf]),
        }
        with pytest.raises(ValueError, match="NaN error"):
            ingest_catalog(
                t,
                photometry=_ingest_phot(["sdss_g", "sdss_r"]),
                flux_unit="cgs_fnu",
                missing="mask",
            )


_WNE_NAMES = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1"]


def _catalog_model(ssp):
    import warnings

    from tengri import DEFAULT, Fixed, ForwardModel, Observation, Photometry, SEDModel, Uniform
    from tengri.forward.sed_model import WavePrecomp

    obs = Observation(photometry=Photometry.from_names(_WNE_NAMES))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sed = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "tau_gyr": Fixed(10.0),
                "log_total_mass": Uniform(8.0, 12.0),
            },
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "tau_v": Fixed(0.3),
            },
            redshift=Fixed(0.1),
            approx=WavePrecomp(catalog_z_range=(0.0, 1.0)),
        )
        model = ForwardModel.build(sed=sed, observation=obs)
    ref = np.asarray(sed.predict_photometry({"sfh_dpl_log_total_mass": 10.0}))
    return model, ref, 0.02 * ref


def _two_galaxy_table(ref, noise, *, row0_flux, row0_err):
    """Galaxy 0 takes ``row0_flux``/``row0_err``; galaxy 1 is the clean reference."""
    table = {"z": np.array([0.1, 0.1])}
    for j, name in enumerate(_WNE_NAMES):
        table[name] = np.array([row0_flux[j], ref[j]], dtype=float)
        table[f"{name}_err"] = np.array([row0_err[j], noise[j]], dtype=float)
    return table


class TestCatalogAllAbsentGalaxy:
    """A galaxy with every band absent is refused, never fitted to a NaN logM."""

    def test_all_absent_galaxy_is_refused_and_other_galaxy_fits(self, ssp_data_wne):
        from tengri.inference.catalog import Catalog

        model, ref, noise = _catalog_model(ssp_data_wne)
        nan6 = np.full(len(_WNE_NAMES), np.nan)
        table = _two_galaxy_table(ref, noise, row0_flux=nan6, row0_err=nan6)
        cat = Catalog(model, table, flux_unit="cgs_fnu", missing="mask", redshift_col="z")

        post = cat.fit(key=jax.random.PRNGKey(0), method="map", record_refusals=True)

        assert set(post.refusals) == {0}
        assert "absent" in post.refusals[0]
        assert len(post.posteriors) == 1
        for value in post.posteriors[0].params.values():
            assert np.all(np.isfinite(np.asarray(value)))

    def test_all_absent_galaxy_raises_by_default(self, ssp_data_wne):
        from tengri.inference.catalog import Catalog

        model, ref, noise = _catalog_model(ssp_data_wne)
        nan6 = np.full(len(_WNE_NAMES), np.nan)
        table = _two_galaxy_table(ref, noise, row0_flux=nan6, row0_err=nan6)
        cat = Catalog(model, table, flux_unit="cgs_fnu", missing="mask", redshift_col="z")

        with pytest.raises(ValueError, match="absent"):
            cat.fit(key=jax.random.PRNGKey(0), method="map")


class TestCatalogInfiniteFluxMapMatchesNaN:
    """+inf flux in one band fits to the same MAP as NaN in that band."""

    def test_inf_flux_map_equals_nan_flux_map(self, ssp_data_wne):
        from tengri.inference.catalog import Catalog

        model, ref, noise = _catalog_model(ssp_data_wne)
        absent = _WNE_NAMES.index("sdss_i")

        def fit(bad):
            flux0 = ref.astype(float).copy()
            err0 = noise.astype(float).copy()
            flux0[absent] = bad
            err0[absent] = bad
            table = _two_galaxy_table(ref, noise, row0_flux=flux0, row0_err=err0)
            cat = Catalog(model, table, flux_unit="cgs_fnu", missing="mask", redshift_col="z")
            return cat.fit(key=jax.random.PRNGKey(0), method="map")

        post_nan = fit(np.nan)
        post_inf = fit(np.inf)

        for name, value in post_nan[0].params.items():
            assert np.all(np.isfinite(np.asarray(value)))
            np.testing.assert_allclose(
                np.asarray(post_inf[0].params[name]), np.asarray(value), rtol=1e-10
            )

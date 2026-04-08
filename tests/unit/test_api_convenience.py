"""Tests for the new convenience API.

No-SSP tests run unconditionally. SSP-required tests are skipped when data is absent.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# SSP availability gate
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILES = sorted(_DATA_DIR.glob("ssp_*.h5"))
_SSP_FILE = _SSP_FILES[0] if _SSP_FILES else None
_SSP_EXISTS = _SSP_FILE is not None and _SSP_FILE.is_file()
_needs_ssp = pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_posterior(method="vi", n_samples=10):
    from tengri.inference.posterior import Posterior

    key = jax.random.PRNGKey(0)
    samples = {
        "met_logzsol": jax.random.normal(key, (n_samples,)) * 0.1 - 0.3,
        "dust_tau_bc": jax.random.uniform(key, (n_samples,)) * 0.5 + 0.2,
    }
    return Posterior(
        samples=samples,
        params={"met_logzsol": jnp.array([-0.3]), "dust_tau_bc": jnp.array([0.5])},
        method=method,
        wall_time_s=1.0,
        diagnostics={},
    )


# ---------------------------------------------------------------------------
# Task 1: resolve_short_names
# ---------------------------------------------------------------------------


class TestResolveShortNames:
    def test_tsnorm_short_names(self):
        from tengri.core.param_translate import resolve_short_names
        from tengri.distributions import Uniform

        expanded = resolve_short_names(
            "tsnorm", {"log_peak_sfr": Uniform(-1, 2.5), "logzsol": Uniform(-2, 0.2)}
        )
        assert "sfh_tsnorm_log_peak_sfr" in expanded
        assert "met_logzsol" in expanded
        assert "log_peak_sfr" not in expanded

    def test_dpl_plus_field_compound(self):
        from tengri.core.param_translate import resolve_short_names
        from tengri.distributions import Uniform

        expanded = resolve_short_names(
            "dpl+field",
            {"log_peak_sfr": Uniform(-1, 2.5), "psd_sigma": Uniform(0.1, 2.0)},
        )
        assert "sfh_dpl_log_peak_sfr" in expanded
        assert "sfh_field_psd_sigma" in expanded

    def test_full_names_pass_through(self):
        from tengri.core.param_translate import resolve_short_names
        from tengri.distributions import Uniform

        expanded = resolve_short_names("tsnorm", {"sfh_tsnorm_log_peak_sfr": Uniform(-1, 2.5)})
        assert "sfh_tsnorm_log_peak_sfr" in expanded

    def test_list_input(self):
        from tengri.core.param_translate import resolve_short_names
        from tengri.distributions import Uniform

        r1 = resolve_short_names("tsnorm", {"log_peak_sfr": Uniform(-1, 2.5)})
        r2 = resolve_short_names(["tsnorm"], {"log_peak_sfr": Uniform(-1, 2.5)})
        assert set(r1.keys()) == set(r2.keys())


# ---------------------------------------------------------------------------
# Task 2: Method unification — deprecation warnings
# ---------------------------------------------------------------------------


class TestDeprecationWarnings:
    def test_deprecated_alias_dict_exists(self):
        import tengri.inference.fitter as fitter_module

        assert hasattr(fitter_module, "_DEPRECATED_METHOD_ALIASES")
        assert isinstance(fitter_module._DEPRECATED_METHOD_ALIASES, dict)

    @pytest.mark.parametrize(
        "old_name,canonical",
        [
            ("geovi", "vi"),
            ("native_geovi", "vi"),
            ("mgvi", "vi_linear"),
            ("raytrace", "mcmc_raytrace"),
            ("nuts", "mcmc_nuts"),
            ("elliptical_slice", "mcmc_ess"),
            ("nss", "evidence"),
        ],
    )
    def test_alias_maps_to_canonical(self, old_name, canonical):
        from tengri.inference.fitter import _DEPRECATED_METHOD_ALIASES

        assert old_name in _DEPRECATED_METHOD_ALIASES
        assert _DEPRECATED_METHOD_ALIASES[old_name] == canonical

    def test_canonical_names_in_docstring(self):
        from tengri.inference.fitter import Fitter

        doc = Fitter.run.__doc__
        for name in ("vi", "vi_linear", "mcmc_raytrace", "mcmc_nuts", "auto"):
            assert name in doc, f"'{name}' not found in Fitter.run() docstring"


# ---------------------------------------------------------------------------
# Task 3: Posterior._fitter + refine() + validate()
# ---------------------------------------------------------------------------


class TestPosteriorRefine:
    def test_fitter_field_default_none(self):
        p = _make_minimal_posterior()
        assert p._fitter is None

    def test_fitter_field_settable(self):
        p = _make_minimal_posterior()
        mock = MagicMock()
        p._fitter = mock
        assert p._fitter is mock

    def test_refine_raises_without_fitter(self):
        p = _make_minimal_posterior()
        with pytest.raises(RuntimeError):
            p.refine("mcmc_raytrace")

    def test_refine_calls_fitter_run(self):
        p = _make_minimal_posterior()
        mock_fitter = MagicMock()
        expected = _make_minimal_posterior(method="mcmc_raytrace")
        mock_fitter.run.return_value = expected
        p._fitter = mock_fitter

        result = p.refine("mcmc_raytrace", n_steps=100)

        mock_fitter.run.assert_called_once_with("mcmc_raytrace", init_from=p, n_steps=100)
        assert result is expected

    def test_validate_raises_without_fitter(self):
        p = _make_minimal_posterior()
        with pytest.raises(RuntimeError):
            p.validate()

    def test_validate_returns_required_keys(self):
        p = _make_minimal_posterior()
        mock_fitter = MagicMock()
        mock_fitter.spec.n_free = 5
        mock_fitter.run.return_value = _make_minimal_posterior("mcmc_nuts")
        p._fitter = mock_fitter

        result = p.validate(n_steps=5)
        assert "mcmc_result" in result
        assert "overlap" in result
        assert "passed" in result


# ---------------------------------------------------------------------------
# Task 6: PriorPredictive
# ---------------------------------------------------------------------------


class TestPriorPredictive:
    def test_prior_predictive_importable(self):
        import tengri

        assert hasattr(tengri, "PriorPredictive")

    def test_check_finite_with_none_flux(self):
        from tengri.core.model import PriorPredictive

        ppc = PriorPredictive(flux=None, sfh=jnp.zeros((10, 50)), params={})
        result = ppc.check_finite()
        assert result["ok"] is True
        assert result["n_nan"] == 0

    def test_check_finite_with_nan_flux(self):
        from tengri.core.model import PriorPredictive

        flux = jnp.array([[float("nan"), 1.0], [2.0, 3.0]])
        ppc = PriorPredictive(flux=flux, sfh=jnp.zeros((2, 50)), params={})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ppc.check_finite()
        assert result["n_nan"] >= 1
        assert result["ok"] is False
        assert len(w) >= 1


# ---------------------------------------------------------------------------
# Task 7: posteriors_to_dataframe
# ---------------------------------------------------------------------------


class TestPostersToDataframe:
    def test_returns_dataframe_with_correct_shape(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        import tengri

        results = [_make_minimal_posterior() for _ in range(3)]
        df = tengri.posteriors_to_dataframe(results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_params_filter(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not installed")

        import tengri

        results = [_make_minimal_posterior()]
        df = tengri.posteriors_to_dataframe(results, params=["met_logzsol"])

        assert any("met_logzsol" in c for c in df.columns)
        assert not any("dust_tau_bc" in c for c in df.columns)

    def test_raises_without_pandas(self):
        import tengri

        results = [_make_minimal_posterior()]
        with (
            patch.dict("sys.modules", {"pandas": None}),
            pytest.raises(ImportError, match="pandas"),
        ):
            tengri.posteriors_to_dataframe(results)


# ---------------------------------------------------------------------------
# Task 8: HierarchicalResult.individual
# ---------------------------------------------------------------------------


class TestHierarchicalResultIndividual:
    def test_individual_returns_empty_when_none(self):
        from tengri.inference.hierarchical import HierarchicalResult

        result = HierarchicalResult(
            shared_samples={},
            shared_params={},
            individual_samples=None,
            method="geovi",
            wall_time_s=0.0,
            diagnostics={},
        )
        assert result.individual == []

    def test_individual_returns_list_of_namespace(self):
        from types import SimpleNamespace

        from tengri.inference.hierarchical import HierarchicalResult

        individual_samples = [
            {"met_logzsol": jnp.array([-0.3, -0.2, -0.1])},
            {"met_logzsol": jnp.array([-0.5, -0.4, -0.3])},
        ]
        result = HierarchicalResult(
            shared_samples={},
            shared_params={},
            individual_samples=individual_samples,
            method="geovi",
            wall_time_s=0.0,
            diagnostics={},
        )
        ind = result.individual
        assert len(ind) == 2
        assert isinstance(ind[0], SimpleNamespace)
        assert hasattr(ind[0], "samples")
        assert hasattr(ind[0], "params")


# ---------------------------------------------------------------------------
# Task 5: Model.from_config (no-SSP: test classmethod exists)
# ---------------------------------------------------------------------------


class TestModelFromConfigInterface:
    def test_from_config_is_classmethod(self):
        import inspect

        from tengri.core.model import Model

        assert hasattr(Model, "from_config")
        assert isinstance(inspect.getattr_static(Model, "from_config"), classmethod)

    def test_from_config_signature(self):
        import inspect

        from tengri.core.defaults import get_from_config_defaults
        from tengri.core.model import Model

        sig = inspect.signature(Model.from_config)
        assert "ssp" in sig.parameters
        assert "sfh" in sig.parameters
        assert "priors" in sig.parameters
        # Defaults now come from defaults.toml, not hardcoded in the signature.
        # The signature uses Ellipsis as a sentinel; verify TOML drives the value.
        assert sig.parameters["sfh"].default is ...
        defs = get_from_config_defaults()
        assert isinstance(defs["sfh"], str)
        assert len(defs["sfh"]) > 0


# ---------------------------------------------------------------------------
# SSP-required integration tests
# ---------------------------------------------------------------------------


@_needs_ssp
class TestModelFitIntegration:
    @pytest.fixture(scope="class")
    def model_and_mock(self):
        import tengri

        model = tengri.Model.from_config(
            ssp=str(_SSP_FILE),
            sfh="dpl",
            filters=["sdss_u", "sdss_g", "sdss_r"],
            redshift=0.1,
            priors=dict(
                alpha=tengri.Uniform(0.5, 3.0),
                beta=tengri.Uniform(0.3, 2.0),
                tau_gyr=tengri.Uniform(0.5, 10.0),
                log_peak_sfr=tengri.Uniform(-1, 2.5),
                logzsol=tengri.Uniform(-1.5, 0.2),
                tau_bc=tengri.Uniform(0, 3.0),
            ),
        )
        true_params = {
            "sfh_dpl_alpha": 1.2,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 4.0,
            "sfh_dpl_log_peak_sfr": 0.9,
            "met_logzsol": -0.3,
            "dust_tau_bc": 1.0,
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 0.1,
        }
        mock = model.mock(true_params, snr=10.0, key=jax.random.PRNGKey(0))
        return model, mock

    def test_model_fit_map_returns_posterior(self, model_and_mock):
        from tengri.inference.posterior import Posterior

        model, mock = model_and_mock
        result = model.fit(mock.flux_obs, mock.noise, method="map")
        assert isinstance(result, Posterior)

    def test_model_fit_sets_fitter_attribute(self, model_and_mock):
        model, mock = model_and_mock
        model.fit(mock.flux_obs, mock.noise, method="map")
        assert hasattr(model, "fitter_")

    def test_model_fit_result_has_fitter_backref(self, model_and_mock):
        model, mock = model_and_mock
        result = model.fit(mock.flux_obs, mock.noise, method="map")
        assert result._fitter is not None

    def test_prior_predictive(self, model_and_mock):
        from tengri.core.model import PriorPredictive

        model, _ = model_and_mock
        ppc = model.prior_predictive(n=20, seed=0)
        assert isinstance(ppc, PriorPredictive)
        assert ppc.sfh is not None
        result = ppc.check_finite()
        assert isinstance(result["ok"], bool)

    def test_fit_catalog_list_of_dicts(self, model_and_mock):
        model, mock = model_and_mock
        catalog = [
            {
                "flux_u": float(mock.flux_obs[0]),
                "flux_g": float(mock.flux_obs[1]),
                "flux_r": float(mock.flux_obs[2]),
                "err_u": float(mock.noise[0]),
                "err_g": float(mock.noise[1]),
                "err_r": float(mock.noise[2]),
            }
        ]
        results = model.fit_catalog(
            catalog,
            flux_cols=["flux_u", "flux_g", "flux_r"],
            err_cols=["err_u", "err_g", "err_r"],
            method="map",
            verbose=False,
        )
        assert len(results) == 1

    def test_recommend_method_returns_valid_string(self, model_and_mock):
        model, _ = model_and_mock
        method = model.recommend_method()
        valid = {"vi", "vi_linear", "laplace", "vi_nifty", "map"}
        assert method in valid, f"Unexpected method: {method}"

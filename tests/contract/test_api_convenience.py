# SPDX-License-Identifier: BSD-3-Clause
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

pytestmark = pytest.mark.contract

# ── SSP availability gate ─────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILES = sorted(_DATA_DIR.glob("ssp_*.h5"))
_SSP_FILE = _SSP_FILES[0] if _SSP_FILES else None
_SSP_EXISTS = _SSP_FILE is not None and _SSP_FILE.is_file()
_needs_ssp = pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")

# ── Helpers ───────────────────────────────────────────────────────


class _FakeModel:
    """Minimal model stub with predict_photometry (no SSP needed)."""

    def predict_photometry(self, params):
        return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])


class TestMockAttributeAndKeyAccess:
    """generate_mock() (mapping) and SEDModel.mock() (object) support both idioms.

    Regression for the recurring "dict vs object" footgun (fresh-user audit
    2026-07): the quickstart used ``generate_mock(...).flux_obs`` (attribute)
    while the README used dict-style access on the ``.mock()`` object — code
    written for one return type broke on the other. Now both types accept both
    idioms, and generate_mock keeps full dict semantics.
    """

    def test_generate_mock_supports_attribute_access(self):
        from tengri.analysis.mock import generate_mock

        mock = generate_mock(_FakeModel(), {"a": 1.0}, key=jax.random.PRNGKey(0))
        assert isinstance(mock, dict)  # dict semantics preserved
        assert mock.flux_obs is mock["flux_obs"]  # attribute access added
        assert mock.flux_true is mock["flux_true"]
        assert mock.noise is mock["noise"]

    def test_generate_mock_noiseless_has_no_flux_obs(self):
        from tengri.analysis.mock import generate_mock

        mock = generate_mock(_FakeModel(), {"a": 1.0})
        assert "flux_obs" not in mock  # unchanged: no key ⇒ no noisy realization
        with pytest.raises(AttributeError):
            _ = mock.flux_obs

    def test_generate_mock_flattens_to_its_values(self):
        # Registered as a pytree so it flattens to its 4 values (sorted keys),
        # exactly like the plain dict it used to be — NOT to a single opaque
        # leaf (the default for an unregistered dict subclass).
        from tengri.analysis.mock import generate_mock

        mock = generate_mock(_FakeModel(), {"a": 1.0}, key=jax.random.PRNGKey(0))
        mock_leaves = jax.tree_util.tree_leaves(mock)
        dict_leaves = jax.tree_util.tree_leaves(dict(mock))
        assert len(mock_leaves) == len(dict_leaves) > 1  # flattened, not one leaf
        assert all(a is b for a, b in zip(mock_leaves, dict_leaves))  # same order/objects

    def test_mockdata_object_supports_mapping_access(self):
        from tengri.forward.sed_model import MockData

        md = MockData(
            flux_true=jnp.array([1.0, 2.0]),
            flux_obs=jnp.array([1.1, 2.1]),
            noise=jnp.array([0.1, 0.1]),
            params={"a": 1.0},
        )
        assert md["flux_obs"] is md.flux_obs  # dict-style access on the object
        assert md[0] is md.flux_true  # positional NamedTuple access still works
        assert "flux_obs" in md and "nope" not in md
        assert md.get("noise") is not None and md.get("nope") is None
        assert md.keys() == ["flux_true", "flux_obs", "noise", "params"]
        assert md.items()[0] == ("flux_true", md.flux_true)


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


# ── Task 1: resolve_short_names ───────────────────────────────────


class TestResolveShortNames:
    def test_tsnorm_short_names(self):
        from tengri.parameters.priors import Uniform
        from tengri.parameters.translate import resolve_short_names

        expanded = resolve_short_names(
            "tsnorm", {"log_total_mass": Uniform(-1, 2.5), "logzsol": Uniform(-2, 0.2)}
        )
        assert "sfh_tsnorm_log_total_mass" in expanded
        assert "met_logzsol" in expanded
        assert "log_total_mass" not in expanded

    def test_dpl_plus_field_compound(self):
        from tengri.parameters.priors import Uniform
        from tengri.parameters.translate import resolve_short_names

        expanded = resolve_short_names(
            "dpl+field",
            {"log_total_mass": Uniform(-1, 2.5), "psd_sigma": Uniform(0.1, 2.0)},
        )
        assert "sfh_dpl_log_total_mass" in expanded
        assert "sfh_field_psd_sigma" in expanded

    def test_full_names_pass_through(self):
        from tengri.parameters.priors import Uniform
        from tengri.parameters.translate import resolve_short_names

        expanded = resolve_short_names("tsnorm", {"sfh_tsnorm_log_total_mass": Uniform(-1, 2.5)})
        assert "sfh_tsnorm_log_total_mass" in expanded

    def test_list_input(self):
        from tengri.parameters.priors import Uniform
        from tengri.parameters.translate import resolve_short_names

        r1 = resolve_short_names("tsnorm", {"log_total_mass": Uniform(-1, 2.5)})
        r2 = resolve_short_names(["tsnorm"], {"log_total_mass": Uniform(-1, 2.5)})
        assert set(r1.keys()) == set(r2.keys())


# ── Task 2: Method-name validation ────────────────────────────────


class TestDeprecationWarnings:
    def test_canonical_names_in_docstring(self):
        from tengri.inference.fitter import Fitter

        doc = Fitter.run.__doc__
        for name in (
            "vi",
            "vi_linear",
            "vi_nonlinear_fast",
            "mcmc_raytrace",
            "mcmc_nuts",
            "nss",
            "auto",
        ):
            assert name in doc, f"'{name}' not in Fitter.run() docstring"


# ── Task 3: Posterior._fitter + refine() + validate() ─────────────


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


# ── Task 6: PriorPredictive ───────────────────────────────────────


class TestPriorPredictive:
    def test_prior_predictive_importable(self):
        import tengri

        assert hasattr(tengri, "PriorPredictive")

    def test_check_finite_with_none_flux(self):
        from tengri.forward.sed_model import PriorPredictive

        ppc = PriorPredictive(flux=None, sfh=jnp.zeros((10, 50)), params={})
        result = ppc.check_finite()
        assert result["ok"] is True
        assert result["n_nan"] == 0

    def test_check_finite_with_nan_flux(self):
        from tengri.forward.sed_model import PriorPredictive

        flux = jnp.array([[float("nan"), 1.0], [2.0, 3.0]])
        ppc = PriorPredictive(flux=flux, sfh=jnp.zeros((2, 50)), params={})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ppc.check_finite()
        assert result["n_nan"] >= 1
        assert result["ok"] is False
        assert len(w) >= 1


# ── Task 7: posteriors_to_dataframe ───────────────────────────────


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


# ── Task 8: PopulationPosterior.individual ─────────────────────────


class TestPopulationPosteriorIndividual:
    def test_individual_returns_empty_when_none(self):
        from tengri.inference.hierarchical import PopulationPosterior

        result = PopulationPosterior(
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

        from tengri.inference.hierarchical import PopulationPosterior

        individual_samples = [
            {"met_logzsol": jnp.array([-0.3, -0.2, -0.1])},
            {"met_logzsol": jnp.array([-0.5, -0.4, -0.3])},
        ]
        result = PopulationPosterior(
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


# ── Task 5: SEDModel.from_config (no-SSP: test classmethod exists) ───


class TestModelFromConfigInterface:
    def test_from_config_is_classmethod(self):
        import inspect

        from tengri.forward.sed_model import SEDModel

        assert hasattr(SEDModel, "from_config")
        assert isinstance(inspect.getattr_static(SEDModel, "from_config"), classmethod)

    def test_from_config_signature(self):
        import inspect

        from tengri.forward.sed_model import SEDModel
        from tengri.parameters.defaults import get_from_config_defaults

        sig = inspect.signature(SEDModel.from_config)
        assert "ssp" in sig.parameters
        assert "sfh" in sig.parameters
        assert "priors" in sig.parameters
        # Defaults now come from defaults.toml, not hardcoded in the signature.
        # The signature uses Ellipsis as a sentinel; verify TOML drives the value.
        assert sig.parameters["sfh"].default is ...
        defs = get_from_config_defaults()
        assert isinstance(defs["sfh"], str)
        assert len(defs["sfh"]) > 0


# ── SSP-required integration tests ────────────────────────────────


@_needs_ssp
class TestModelFitIntegration:
    @pytest.fixture(scope="class")
    def model_and_mock(self):
        import tengri

        model = tengri.SEDModel.from_config(
            ssp=str(_SSP_FILE),
            sfh="dpl",
            filters=["sdss_u", "sdss_g", "sdss_r"],
            redshift=0.1,
            priors=dict(
                alpha=tengri.Uniform(0.5, 3.0),
                beta=tengri.Uniform(0.3, 2.0),
                tau_gyr=tengri.Uniform(0.5, 10.0),
                log_total_mass=tengri.Uniform(-1, 2.5),
                logzsol=tengri.Uniform(-1.5, 0.2),
                tau_bc=tengri.Uniform(0, 3.0),
            ),
        )
        true_params = {
            "sfh_dpl_alpha": 1.2,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 4.0,
            # age is FREE in this spec — leaving it out used to silently pin
            # the mock at an internal default; predict paths now require a
            # value for every free parameter (MissingParameterError).
            "sfh_dpl_age_gyr": 8.0,
            "sfh_dpl_log_total_mass": 0.9,
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
        from tengri.forward.sed_model import PriorPredictive

        model, _ = model_and_mock
        ppc = model.prior_predictive(n=20, seed=0)
        assert isinstance(ppc, PriorPredictive)
        assert ppc.sfh is not None
        result = ppc.check_finite()
        assert isinstance(result["ok"], bool)

    def test_fit_batch_list_of_dicts(self, model_and_mock):
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
        results = model.fit_batch(
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
        valid = {"vi_nifty", "vi_nifty_linear", "laplace", "map"}
        assert method in valid, f"Unexpected method: {method}"

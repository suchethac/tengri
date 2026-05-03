"""Tests for the Posterior class."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.unit

# ── BPT line names and fluxes used across several test classes ──────────────
_BPT_NAMES = ("Halpha", "Hbeta", "NII_6584", "OIII_5007")
_BPT_WAVES = jnp.array([6564.61, 4862.68, 6583.45, 5008.24])
# Ha=10, Hb=4, NII=5, OIII=2 → Ha/Hb=2.5, NII/Ha=0.5, OIII/Hb=0.5
_BPT_FLUX_1D = jnp.array([10.0, 4.0, 5.0, 2.0])


@pytest.fixture
def map_posterior():
    return Posterior(
        samples=None,
        params={
            "sfh_dpl_alpha": jnp.array(1.2),
            "sfh_dpl_beta": jnp.array(1.0),
            "met_logzsol": jnp.array(-0.3),
        },
        method="MAP (Adam)",
        wall_time_s=1.5,
        diagnostics={"n_steps": 100},
        loss_history=jnp.array([10.0, 5.0, 2.0]),
    )


@pytest.fixture
def sampling_posterior():
    key = jax.random.PRNGKey(0)
    n = 100
    return Posterior(
        samples={
            "sfh_dpl_alpha": 1.2 + 0.3 * jax.random.normal(key, (n,)),
            "sfh_dpl_beta": 1.0 + 0.2 * jax.random.normal(jax.random.PRNGKey(1), (n,)),
            "met_logzsol": -0.3 + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (n,)),
        },
        params={
            "sfh_dpl_alpha": jnp.array(1.2),
            "sfh_dpl_beta": jnp.array(1.0),
            "met_logzsol": jnp.array(-0.3),
        },
        method="NUTS (BlackJAX)",
        wall_time_s=30.0,
        diagnostics={"n_divergent": 0, "n_samples": 100},
    )


class TestStats:
    def test_map_stats(self, map_posterior):
        s = map_posterior.stats()
        assert "sfh_dpl_alpha" in s
        assert "value" in s["sfh_dpl_alpha"]
        assert s["sfh_dpl_alpha"]["value"] == pytest.approx(1.2)

    def test_sampling_stats(self, sampling_posterior):
        s = sampling_posterior.stats()
        assert "sfh_dpl_alpha" in s
        assert "median" in s["sfh_dpl_alpha"]
        assert "lo_68" in s["sfh_dpl_alpha"]
        assert "hi_68" in s["sfh_dpl_alpha"]
        assert (
            s["sfh_dpl_alpha"]["lo_68"]
            < s["sfh_dpl_alpha"]["median"]
            < s["sfh_dpl_alpha"]["hi_68"]
        )


class TestResample:
    def test_resample_single(self, sampling_posterior):
        draw = sampling_posterior.resample(jax.random.PRNGKey(0), n=1)
        assert "sfh_dpl_alpha" in draw
        assert draw["sfh_dpl_alpha"].ndim == 0  # scalar

    def test_resample_batch(self, sampling_posterior):
        draw = sampling_posterior.resample(jax.random.PRNGKey(0), n=5)
        assert draw["sfh_dpl_alpha"].shape == (5,)

    def test_map_resample(self, map_posterior):
        draw = map_posterior.resample(jax.random.PRNGKey(0), n=1)
        assert float(draw["sfh_dpl_alpha"]) == pytest.approx(1.2)


class TestToParamSpec:
    def test_map_to_param_spec(self, map_posterior):
        spec = map_posterior.to_param_spec()
        from tengri.parameters.priors import Fixed

        d = spec.get_distribution("sfh_dpl_alpha")
        assert isinstance(d, Fixed)

    def test_sampling_to_param_spec(self, sampling_posterior):
        spec = sampling_posterior.to_param_spec()
        from tengri.parameters.priors import Gaussian

        d = spec.get_distribution("sfh_dpl_alpha")
        assert isinstance(d, Gaussian)
        assert d.mu == pytest.approx(1.2, abs=0.1)


class TestRepr:
    def test_map_repr(self, map_posterior):
        r = repr(map_posterior)
        assert "MAP" in r
        assert "None" in r  # no samples

    def test_sampling_repr(self, sampling_posterior):
        r = repr(sampling_posterior)
        assert "NUTS" in r


# ── Fixtures with emission lines ─────────────────────────────────────────────


@pytest.fixture
def map_eline_posterior():
    return Posterior(
        samples=None,
        params={"sfh_dpl_alpha": jnp.array(1.2)},
        method="MAP",
        wall_time_s=1.0,
        diagnostics={},
        eline_fluxes=_BPT_FLUX_1D,
        eline_names=_BPT_NAMES,
        eline_wavelengths=_BPT_WAVES,
    )


@pytest.fixture
def sampling_eline_posterior():
    """100-sample posterior with BPT emission lines, all positive fluxes."""
    key = jax.random.PRNGKey(7)
    n = 100
    # Shape (n_samples, n_lines): rows = samples, cols = lines
    fluxes = jnp.stack(
        [
            10.0 + jax.random.normal(key, (n,)),  # Halpha
            4.0 + 0.3 * jax.random.normal(jax.random.PRNGKey(1), (n,)),  # Hbeta
            5.0 + 0.3 * jax.random.normal(jax.random.PRNGKey(2), (n,)),  # NII
            2.0 + 0.2 * jax.random.normal(jax.random.PRNGKey(3), (n,)),  # OIII
        ],
        axis=1,
    )
    return Posterior(
        samples={"sfh_dpl_alpha": 1.2 + 0.1 * jax.random.normal(key, (n,))},
        params={"sfh_dpl_alpha": jnp.array(1.2)},
        method="mcmc_raytrace",
        wall_time_s=10.0,
        diagnostics={"accept_rate": 0.55, "ess_bulk": {"sfh_dpl_alpha": 80.0}},
        eline_fluxes=fluxes,
        eline_names=_BPT_NAMES,
        eline_wavelengths=_BPT_WAVES,
    )


@pytest.fixture
def negative_flux_eline_posterior():
    """MAP posterior with one negative-flux line (non-detection)."""
    # NII_6584 flux is negative → BPT ratios should be NaN
    fluxes_1d = jnp.array([10.0, 4.0, -1.0, 2.0])
    return Posterior(
        samples=None,
        params={"sfh_dpl_alpha": jnp.array(1.2)},
        method="MAP",
        wall_time_s=1.0,
        diagnostics={},
        eline_fluxes=fluxes_1d,
        eline_names=_BPT_NAMES,
        eline_wavelengths=_BPT_WAVES,
    )


# ── TestLineFluxes ────────────────────────────────────────────────────────────


class TestLineFluxes:
    def test_map_returns_triple_same_value(self, map_eline_posterior):
        fluxes = map_eline_posterior.line_fluxes()
        med, lo, hi = fluxes["Halpha"]
        assert med == lo == hi
        assert med == pytest.approx(10.0)

    def test_sampling_returns_percentiles(self, sampling_eline_posterior):
        fluxes = sampling_eline_posterior.line_fluxes()
        med, lo, hi = fluxes["Halpha"]
        # lo < med < hi for a non-degenerate distribution
        assert lo < med < hi
        # median should be near the injection value
        assert med == pytest.approx(10.0, abs=0.5)

    def test_all_lines_present(self, map_eline_posterior):
        fluxes = map_eline_posterior.line_fluxes()
        for name in _BPT_NAMES:
            assert name in fluxes

    def test_raises_when_no_eline_fluxes(self, map_posterior):
        with pytest.raises(ValueError, match="No emission line fluxes"):
            map_posterior.line_fluxes()


# ── TestBPT ───────────────────────────────────────────────────────────────────


class TestBPT:
    def test_map_returns_scalar_ratios(self, map_eline_posterior):
        x, y = map_eline_posterior.bpt_nii()
        # log10(NII/Ha) = log10(5/10) ≈ -0.301
        assert float(x) == pytest.approx(np.log10(5.0 / 10.0), abs=1e-4)
        # log10(OIII/Hb) = log10(2/4) ≈ -0.301
        assert float(y) == pytest.approx(np.log10(2.0 / 4.0), abs=1e-4)

    def test_sampling_returns_array(self, sampling_eline_posterior):
        x, y = sampling_eline_posterior.bpt_nii()
        assert x.shape == (100,)
        assert y.shape == (100,)
        assert jnp.all(jnp.isfinite(x))
        assert jnp.all(jnp.isfinite(y))

    def test_negative_flux_gives_nan(self, negative_flux_eline_posterior):
        x, _y = negative_flux_eline_posterior.bpt_nii()
        # NII_6584 < 0 → log_nii_ha = NaN
        assert jnp.isnan(x)

    def test_raises_missing_bpt_lines(self):
        p = Posterior(
            samples=None,
            params={},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
            eline_fluxes=jnp.array([5.0, 3.0]),
            eline_names=("Halpha", "Hbeta"),
            eline_wavelengths=jnp.array([6564.61, 4862.68]),
        )
        with pytest.raises(ValueError, match="BPT lines not in catalog"):
            p.bpt_nii()

    def test_raises_no_elines(self, map_posterior):
        with pytest.raises(ValueError, match="No emission line fluxes"):
            map_posterior.bpt_nii()


# ── TestBalmerDecrement ───────────────────────────────────────────────────────


class TestBalmerDecrement:
    def test_map_returns_triple_same(self, map_eline_posterior):
        med, lo, hi = map_eline_posterior.balmer_decrement()
        assert med == lo == hi
        assert med == pytest.approx(10.0 / 4.0, abs=1e-5)

    def test_sampling_returns_percentiles(self, sampling_eline_posterior):
        med, lo, hi = sampling_eline_posterior.balmer_decrement()
        assert lo <= med <= hi
        assert med == pytest.approx(10.0 / 4.0, abs=0.3)

    def test_raises_no_elines(self, map_posterior):
        with pytest.raises(ValueError, match="No emission line fluxes"):
            map_posterior.balmer_decrement()

    def test_raises_missing_halpha(self):
        p = Posterior(
            samples=None,
            params={},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
            eline_fluxes=jnp.array([4.0]),
            eline_names=("Hbeta",),
            eline_wavelengths=jnp.array([4862.68]),
        )
        with pytest.raises(ValueError, match="Halpha"):
            p.balmer_decrement()


# ── TestEquivalentWidths ──────────────────────────────────────────────────────


_C_AA_S = 2.99792458e18


class _FakeSED:
    """Minimal stand-in for SEDModel exposing predict_rest_sed only."""

    def __init__(self, wave, line_centers, line_amplitudes, sigma_aa=5.0, cont_f_lambda=1.0):
        self._wave = wave
        self._centers = line_centers
        self._amps = line_amplitudes
        self._sigma = sigma_aa
        self._cont = cont_f_lambda

    def predict_rest_sed(self, params, wave=None):
        from tengri.forward.result import SEDResult

        f_lambda = jnp.full_like(self._wave, self._cont)
        for c, a in zip(self._centers, self._amps):
            f_lambda = f_lambda + a * jnp.exp(
                -0.5 * ((self._wave - c) / self._sigma) ** 2
            ) / (self._sigma * jnp.sqrt(2.0 * jnp.pi))
        l_nu = f_lambda * (self._wave**2 / _C_AA_S)
        return SEDResult(wavelength=self._wave, sed=l_nu)


@pytest.fixture
def ew_wave():
    """Fine wavelength grid covering all BPT lines."""
    return jnp.linspace(4500.0, 7000.0, 8000)


@pytest.fixture
def map_ew_posterior(ew_wave):
    """MAP posterior with attached fake model that has H-alpha and H-beta emission."""
    fake = _FakeSED(
        ew_wave,
        line_centers=[4862.68, 5008.24, 6564.61, 6583.45],  # Hb, OIII, Ha, NII
        line_amplitudes=[20.0, 10.0, 50.0, 25.0],
    )
    p = Posterior(
        samples=None,
        params={"sfh_dpl_alpha": jnp.array(1.2)},
        method="MAP",
        wall_time_s=1.0,
        diagnostics={},
        eline_fluxes=_BPT_FLUX_1D,
        eline_names=_BPT_NAMES,
        eline_wavelengths=_BPT_WAVES,
    )
    p._model = fake
    return p


@pytest.fixture
def sampling_ew_posterior(ew_wave):
    """Sampling posterior with attached fake model and 30 draws."""
    fake = _FakeSED(
        ew_wave,
        line_centers=[4862.68, 5008.24, 6564.61, 6583.45],
        line_amplitudes=[20.0, 10.0, 50.0, 25.0],
    )
    n = 30
    key = jax.random.PRNGKey(11)
    fluxes = jnp.stack(
        [
            10.0 + 0.5 * jax.random.normal(key, (n,)),
            4.0 + 0.2 * jax.random.normal(jax.random.PRNGKey(12), (n,)),
            5.0 + 0.2 * jax.random.normal(jax.random.PRNGKey(13), (n,)),
            2.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(14), (n,)),
        ],
        axis=1,
    )
    p = Posterior(
        samples={"sfh_dpl_alpha": 1.2 + 0.05 * jax.random.normal(key, (n,))},
        params={"sfh_dpl_alpha": jnp.array(1.2)},
        method="mcmc_nuts",
        wall_time_s=5.0,
        diagnostics={},
        eline_fluxes=fluxes,
        eline_names=_BPT_NAMES,
        eline_wavelengths=_BPT_WAVES,
    )
    p._model = fake
    return p


class TestEquivalentWidths:
    def test_raises_no_eline_fluxes(self, map_posterior):
        with pytest.raises(ValueError, match="No emission line fluxes"):
            map_posterior.equivalent_widths()

    def test_raises_no_model(self, map_eline_posterior):
        # Fixture map_eline_posterior has eline_fluxes but no _model attached.
        with pytest.raises(ValueError, match="model"):
            map_eline_posterior.equivalent_widths()

    def test_map_returns_triple_same(self, map_ew_posterior):
        ew = map_ew_posterior.equivalent_widths()
        assert set(ew.keys()) == set(_BPT_NAMES)
        for name in _BPT_NAMES:
            med, lo, hi = ew[name]
            assert med == lo == hi

    def test_map_emission_lines_positive(self, map_ew_posterior):
        ew = map_ew_posterior.equivalent_widths()
        for name in _BPT_NAMES:
            med, _lo, _hi = ew[name]
            assert med > 0.0, f"{name} EW should be positive for emission"

    def test_map_halpha_recovers_amplitude_isolated(self, ew_wave):
        """With a single isolated line, EW recovers the injected line flux."""
        fake = _FakeSED(ew_wave, line_centers=[6564.61], line_amplitudes=[50.0])
        p = Posterior(
            samples=None,
            params={"sfh_dpl_alpha": jnp.array(1.2)},
            method="MAP",
            wall_time_s=1.0,
            diagnostics={},
            eline_fluxes=jnp.array([10.0]),
            eline_names=("Halpha",),
            eline_wavelengths=jnp.array([6564.61]),
        )
        p._model = fake
        ew = p.equivalent_widths()
        med, _lo, _hi = ew["Halpha"]
        assert med == pytest.approx(50.0, rel=0.05)

    def test_sampling_returns_percentiles(self, sampling_ew_posterior):
        ew = sampling_ew_posterior.equivalent_widths()
        for name in _BPT_NAMES:
            med, lo, hi = ew[name]
            # Continuum is identical across samples → distribution is degenerate
            # but lo ≤ med ≤ hi must still hold.
            assert lo <= med <= hi
            assert med > 0.0


# ── TestSummaryTable ──────────────────────────────────────────────────────────


class TestSummaryTable:
    def test_map_table_contains_method(self, map_posterior):
        t = map_posterior.summary_table()
        assert "MAP" in t
        assert "sfh_dpl_alpha" in t

    def test_sampling_table_contains_ess_header(self, sampling_posterior):
        t = sampling_posterior.summary_table()
        assert "ESS" in t
        assert "sfh_dpl_alpha" in t

    def test_sampling_table_shows_accept_rate(self):
        p = Posterior(
            samples={"x": jnp.ones(50)},
            params={"x": jnp.array(1.0)},
            method="mcmc_raytrace",
            wall_time_s=5.0,
            diagnostics={"accept_rate": 0.62},
        )
        t = p.summary_table()
        assert "accept=" in t

    def test_sampling_table_shows_divergences(self):
        p = Posterior(
            samples={"x": jnp.ones(50)},
            params={"x": jnp.array(1.0)},
            method="mcmc_nuts",
            wall_time_s=5.0,
            diagnostics={"n_divergent": 3},
        )
        t = p.summary_table()
        assert "divergences=3" in t

    def test_sampling_table_shows_final_loss(self):
        p = Posterior(
            samples={"x": jnp.ones(50)},
            params={"x": jnp.array(1.0)},
            method="MAP",
            wall_time_s=1.0,
            diagnostics={"final_loss": 12.34},
        )
        t = p.summary_table()
        assert "loss=" in t

    def test_log_evidence_included(self):
        p = Posterior(
            samples=None,
            params={"x": jnp.array(0.5)},
            method="nss",
            wall_time_s=60.0,
            diagnostics={"log_evidence_err": 0.05},
            log_evidence=-42.1,
        )
        t = p.summary_table()
        assert "log Z" in t
        assert "-42.1" in t


# ── TestAutocorrelation1D ─────────────────────────────────────────────────────


class TestAutocorrelation1D:
    def test_lag_0_is_one(self):
        x = np.random.default_rng(0).normal(size=200)
        acf = Posterior._autocorrelation_1d(x)
        assert acf[0] == pytest.approx(1.0, abs=1e-6)

    def test_constant_array_returns_zeros(self):
        x = np.ones(100)
        acf = Posterior._autocorrelation_1d(x)
        assert np.all(acf == 0.0)

    def test_length_equals_max_lag_plus_one(self):
        x = np.random.default_rng(1).normal(size=300)
        acf = Posterior._autocorrelation_1d(x, max_lag=50)
        assert len(acf) == 51

    def test_default_max_lag_is_half_n(self):
        x = np.random.default_rng(2).normal(size=200)
        acf = Posterior._autocorrelation_1d(x)
        assert len(acf) == 101  # 200//2 + 1

    def test_iid_acf_decays_to_near_zero(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=2000)
        acf = Posterior._autocorrelation_1d(x, max_lag=20)
        # For iid, all lags > 0 should be small
        assert np.all(np.abs(acf[1:]) < 0.1)


# ── TestAutocorrelation ───────────────────────────────────────────────────────


class TestAutocorrelation:
    def test_raises_on_map(self, map_posterior):
        with pytest.raises(ValueError, match="Autocorrelation requires samples"):
            map_posterior.autocorrelation()

    def test_returns_dict_of_arrays(self, sampling_posterior):
        acf = sampling_posterior.autocorrelation()
        assert "sfh_dpl_alpha" in acf
        assert isinstance(acf["sfh_dpl_alpha"], np.ndarray)

    def test_lag_0_is_one(self, sampling_posterior):
        acf = sampling_posterior.autocorrelation()
        for arr in acf.values():
            assert arr[0] == pytest.approx(1.0, abs=1e-6)

    def test_custom_max_lag(self, sampling_posterior):
        acf = sampling_posterior.autocorrelation(max_lag=10)
        for arr in acf.values():
            assert len(arr) == 11


# ── TestESS ───────────────────────────────────────────────────────────────────


class TestEffectiveSampleSize:
    def test_raises_on_map(self, map_posterior):
        with pytest.raises(ValueError, match="ESS requires samples"):
            map_posterior.effective_sample_size()

    def test_returns_dict_with_positive_ess(self, sampling_posterior):
        ess = sampling_posterior.effective_sample_size()
        assert "sfh_dpl_alpha" in ess
        assert ess["sfh_dpl_alpha"] > 0


class TestAutocorrelationTime:
    def test_raises_on_map(self, map_posterior):
        with pytest.raises(ValueError, match="Autocorrelation time requires samples"):
            map_posterior.autocorrelation_time()

    def test_returns_tau_keys(self, sampling_posterior):
        act = sampling_posterior.autocorrelation_time()
        # effective_sample_size returns {name: {tau_standard, tau_absolute, ...}}
        for info in act.values():
            assert "ess" in info


# ── TestDiagnosticsSummary ────────────────────────────────────────────────────


class TestDiagnosticsSummary:
    def test_map_returns_short_string(self, map_posterior):
        s = map_posterior.diagnostics_summary()
        assert "MAP" in s
        assert "no samples" in s

    def test_sampling_returns_table(self, sampling_posterior):
        s = sampling_posterior.diagnostics_summary()
        assert "Method" in s
        assert "Samples" in s
        assert "sfh_dpl_alpha" in s


# ── TestPosteriorPredictive ───────────────────────────────────────────────────


class _FakePhotModel:
    """Stand-in for SEDModel exposing only predict_photometry()."""

    def __init__(self, fluxes_for_params):
        # fluxes_for_params(params) -> array of band fluxes
        self._fn = fluxes_for_params

    def predict_photometry(self, params):
        return self._fn(params)


class TestPosteriorPredictive:
    def test_raises_no_model(self, sampling_posterior):
        # sampling_posterior has no _model attached
        with pytest.raises(ValueError, match="model"):
            sampling_posterior.posterior_predictive(
                data=jnp.array([1.0, 2.0]), noise=jnp.array([0.1, 0.1])
            )

    def test_map_returns_zero_residuals_when_prediction_matches_data(self):
        bands = 5
        truth = jnp.linspace(1.0, 5.0, bands)
        model = _FakePhotModel(lambda p: truth)
        p = Posterior(
            samples=None,
            params={"x": jnp.array(1.0)},
            method="MAP",
            wall_time_s=0.1,
            diagnostics={},
        )
        p._model = model
        out = p.posterior_predictive(
            data=truth, noise=jnp.full(bands, 0.1), n_samples=10
        )
        assert out["predictions"].shape == (1, bands)
        assert out["residuals"].shape == (1, bands)
        assert out["chi2"].shape == (1,)
        assert float(out["chi2"][0]) == pytest.approx(0.0, abs=1e-12)
        assert float(np.max(np.abs(out["residuals"]))) < 1e-12

    def test_sampling_returns_n_samples_predictions(self):
        bands = 4
        n = 30
        truth = jnp.linspace(1.0, 4.0, bands)
        # fake model: predict_photometry returns truth + a small bias driven by params
        model = _FakePhotModel(lambda p: truth + 0.01 * jnp.sum(jnp.asarray(list(p.values()))))
        rng = np.random.default_rng(8)
        p = Posterior(
            samples={"x": jnp.asarray(rng.normal(size=n))},
            params={"x": jnp.array(0.0)},
            method="mcmc_nuts",
            wall_time_s=1.0,
            diagnostics={},
        )
        p._model = model
        out = p.posterior_predictive(
            data=truth, noise=jnp.full(bands, 0.05), n_samples=n
        )
        assert out["predictions"].shape == (n, bands)
        assert out["residuals"].shape == (n, bands)
        assert out["chi2"].shape == (n,)
        # chi^2 must all be finite
        assert np.all(np.isfinite(np.asarray(out["chi2"])))

    def test_chi2_increases_with_misfit(self):
        bands = 3
        data = jnp.array([1.0, 2.0, 3.0])
        # Model returns data + 1.0 across the board → uniform misfit
        model = _FakePhotModel(lambda p: data + 1.0)
        p = Posterior(
            samples=None,
            params={"x": jnp.array(0.0)},
            method="MAP",
            wall_time_s=0.1,
            diagnostics={},
        )
        p._model = model
        out_low = p.posterior_predictive(data=data, noise=jnp.full(bands, 1.0))
        out_high = p.posterior_predictive(data=data, noise=jnp.full(bands, 0.1))
        # chi^2 = sum((data - pred)^2 / noise^2); halving noise → 100× chi^2
        assert float(out_high["chi2"][0]) > float(out_low["chi2"][0])
        # exact: 3 / 1 vs 3 / 0.01 = 3 vs 300
        assert float(out_low["chi2"][0]) == pytest.approx(3.0, rel=1e-9)
        assert float(out_high["chi2"][0]) == pytest.approx(300.0, rel=1e-9)

    def test_n_samples_subselects(self):
        bands = 2
        truth = jnp.array([1.0, 2.0])
        model = _FakePhotModel(lambda p: truth)
        rng = np.random.default_rng(9)
        n_total = 100
        p = Posterior(
            samples={"x": jnp.asarray(rng.normal(size=n_total))},
            params={"x": jnp.array(0.0)},
            method="mcmc_nuts",
            wall_time_s=1.0,
            diagnostics={},
        )
        p._model = model
        out = p.posterior_predictive(
            data=truth, noise=jnp.full(bands, 0.1), n_samples=20
        )
        assert out["predictions"].shape == (20, bands)
        assert out["chi2"].shape == (20,)


# ── TestBPTClassification ────────────────────────────────────────────────────


def _make_bpt_posterior(fluxes_1d_or_2d):
    """Build a Posterior with BPT lines pinned to the supplied fluxes."""
    return Posterior(
        samples=None if fluxes_1d_or_2d.ndim == 1 else {"x": jnp.zeros(fluxes_1d_or_2d.shape[0])},
        params={"x": jnp.array(0.0)},
        method="MAP" if fluxes_1d_or_2d.ndim == 1 else "mcmc_nuts",
        wall_time_s=0.1,
        diagnostics={},
        eline_fluxes=fluxes_1d_or_2d,
        eline_names=_BPT_NAMES,
        eline_wavelengths=_BPT_WAVES,
    )


class TestBPTClassification:
    def test_pure_sf_galaxy(self):
        """log([NII]/Hα)≈-0.7, log([OIII]/Hβ)≈-0.5 sits well below Kauffmann."""
        fluxes = jnp.array([10.0, 4.0, 2.0, 1.3])  # Hα, Hβ, NII, OIII
        p = _make_bpt_posterior(fluxes)
        cls = p.bpt_class()
        assert cls == "SF"

    def test_seyfert_agn_galaxy(self):
        """High [OIII]/Hβ and high [NII]/Hα → AGN."""
        # log NII/Hα = log(8/10) = -0.097, log OIII/Hβ = log(20/4) = 0.7
        # Kewley@(-0.097): 0.61/(-0.567)+1.19 = -1.076+1.19 = 0.114; 0.7 > 0.114 → AGN
        fluxes = jnp.array([10.0, 4.0, 8.0, 20.0])
        p = _make_bpt_posterior(fluxes)
        cls = p.bpt_class()
        assert cls == "AGN"

    def test_composite_galaxy(self):
        """Between Kauffmann and Kewley demarcation lines."""
        # log NII/Hα = log(5/10) = -0.301, log OIII/Hβ = log(2/4) = -0.301
        # Kauffmann@(-0.301): 0.61/(-0.351)+1.3 = -1.738+1.3 = -0.438; -0.301 > -0.438 → above
        # Kewley@(-0.301):    0.61/(-0.771)+1.19 = -0.791+1.19 = 0.399; -0.301 < 0.399 → below
        # → composite
        fluxes = jnp.array([10.0, 4.0, 5.0, 2.0])
        p = _make_bpt_posterior(fluxes)
        cls = p.bpt_class()
        assert cls == "composite"

    def test_sampling_returns_array_of_labels(self):
        rng = np.random.default_rng(20)
        n = 50
        fluxes = jnp.stack(
            [
                10.0 + rng.normal(size=n) * 0.1,  # Halpha
                4.0 + rng.normal(size=n) * 0.05,  # Hbeta
                2.0 + rng.normal(size=n) * 0.05,  # NII (low → SF)
                1.3 + rng.normal(size=n) * 0.05,  # OIII (low → SF)
            ],
            axis=1,
        )
        p = _make_bpt_posterior(fluxes)
        labels = p.bpt_class()
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (n,)
        # All draws should classify as SF given small spread
        assert np.all(labels == "SF")

    def test_raises_when_lines_missing(self, map_posterior):
        with pytest.raises(ValueError, match="No emission line fluxes"):
            map_posterior.bpt_class()

    def test_raises_when_bpt_lines_absent(self):
        p = Posterior(
            samples=None,
            params={},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
            eline_fluxes=jnp.array([5.0, 3.0]),
            eline_names=("Halpha", "Hbeta"),
            eline_wavelengths=jnp.array([6564.61, 4862.68]),
        )
        with pytest.raises(ValueError, match="BPT lines"):
            p.bpt_class()

    def test_nondetection_returns_unknown(self):
        # Negative NII flux → log ratio is NaN → 'unknown' label
        fluxes = jnp.array([10.0, 4.0, -1.0, 2.0])
        p = _make_bpt_posterior(fluxes)
        cls = p.bpt_class()
        assert cls == "unknown"


# ── TestBalmerAv ──────────────────────────────────────────────────────────────


class TestBalmerAv:
    """Tests for the Calzetti+2000 Balmer decrement → A(V) utility."""

    def test_no_dust_returns_zero(self):
        # Hα/Hβ = 2.86 (Case B) → A(V) = 0
        fluxes = jnp.array([2.86, 1.0])
        p = Posterior(
            samples=None,
            params={},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
            eline_fluxes=fluxes,
            eline_names=("Halpha", "Hbeta"),
            eline_wavelengths=jnp.array([6564.61, 4862.68]),
        )
        med, lo, hi = p.balmer_av()
        assert med == pytest.approx(0.0, abs=1e-6)
        assert med == lo == hi

    def test_dusty_galaxy_amplitude(self):
        # R_obs = 5.0; Calzetti+2000 R_V=4.05, k_Hα=2.53, k_Hβ=3.61.
        # E(B-V) = log10(5/2.86) / (0.4 * (3.61 - 2.53)) = 0.2425 / 0.432 = 0.5614
        # A(V) = 4.05 * 0.5614 = 2.274
        fluxes = jnp.array([5.0, 1.0])
        p = Posterior(
            samples=None,
            params={},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
            eline_fluxes=fluxes,
            eline_names=("Halpha", "Hbeta"),
            eline_wavelengths=jnp.array([6564.61, 4862.68]),
        )
        med, _lo, _hi = p.balmer_av()
        assert med == pytest.approx(2.274, rel=1e-3)

    def test_sampling_returns_percentiles(self):
        rng = np.random.default_rng(30)
        n = 200
        # Inject ratio ~ 4.0 with small spread → modest A(V)
        ha = 4.0 + 0.1 * rng.normal(size=n)
        hb = jnp.ones(n)
        fluxes = jnp.stack([jnp.asarray(ha), hb], axis=1)
        p = Posterior(
            samples={"x": jnp.zeros(n)},
            params={},
            method="mcmc_nuts",
            wall_time_s=1.0,
            diagnostics={},
            eline_fluxes=fluxes,
            eline_names=("Halpha", "Hbeta"),
            eline_wavelengths=jnp.array([6564.61, 4862.68]),
        )
        med, lo, hi = p.balmer_av()
        assert lo < med < hi
        # log10(4/2.86)/0.432*4.05 = 0.5905
        assert med == pytest.approx(1.36, rel=0.05)

    def test_raises_when_lines_missing(self, map_posterior):
        with pytest.raises(ValueError, match="No emission line fluxes"):
            map_posterior.balmer_av()

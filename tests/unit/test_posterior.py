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


class TestSummary:
    def test_map_summary(self, map_posterior):
        s = map_posterior.summary()
        assert "sfh_dpl_alpha" in s
        assert "value" in s["sfh_dpl_alpha"]
        assert s["sfh_dpl_alpha"]["value"] == pytest.approx(1.2)

    def test_sampling_summary(self, sampling_posterior):
        s = sampling_posterior.summary()
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


class TestEquivalentWidths:
    def test_raises_not_implemented(self, map_eline_posterior):
        with pytest.raises(NotImplementedError):
            map_eline_posterior.equivalent_widths()


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

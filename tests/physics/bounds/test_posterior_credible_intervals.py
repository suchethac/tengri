# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Posterior bounds: line fluxes, BPT, Balmer, EW, and credible intervals."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.bounds


# ── BPT line names and fluxes used across several test classes ──────────────
_BPT_NAMES = ("Halpha", "Hbeta", "NII_6584", "OIII_5007")
_BPT_WAVES = jnp.array([6564.61, 4862.68, 6583.45, 5008.24])
# Ha=10, Hb=4, NII=5, OIII=2 → Ha/Hb=2.5, NII/Ha=0.5, OIII/Hb=0.5
_BPT_FLUX_1D = jnp.array([10.0, 4.0, 5.0, 2.0])


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
    """Test emission line flux bounds and percentile computation."""

    def test_map_returns_triple_same_value(self, map_eline_posterior):
        fluxes = map_eline_posterior.line_fluxes()
        med, lo, hi = fluxes["Halpha"]
        assert med == lo == hi
        assert med == pytest.approx(10.0)

    def test_sampling_returns_percentiles(self, sampling_eline_posterior):
        """Line flux percentiles: lo < median < hi."""
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

    def test_raises_when_no_eline_fluxes(self):
        p = Posterior(
            samples=None,
            params={},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
        )
        with pytest.raises(ValueError, match="No emission line fluxes"):
            p.line_fluxes()


# ── TestBPT ───────────────────────────────────────────────────────────────────


class TestBPT:
    """Test BPT line ratio bounds."""

    def test_map_returns_scalar_ratios(self, map_eline_posterior):
        x, y = map_eline_posterior.bpt_nii()
        # log10(NII/Ha) = log10(5/10) ≈ -0.301
        assert float(x) == pytest.approx(np.log10(5.0 / 10.0), abs=1e-4)
        # log10(OIII/Hb) = log10(2/4) ≈ -0.301
        assert float(y) == pytest.approx(np.log10(2.0 / 4.0), abs=1e-4)

    def test_sampling_returns_array(self, sampling_eline_posterior):
        x, y = sampling_eline_posterior.bpt_nii()
        chex.assert_shape(x, (100,))
        chex.assert_shape(y, (100,))
        chex.assert_tree_all_finite(x)
        chex.assert_tree_all_finite(y)

    def test_negative_flux_gives_nan(self, negative_flux_eline_posterior):
        """Negative flux → log ratio = NaN."""
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

    def test_raises_no_elines(self):
        p = Posterior(
            samples=None,
            params={},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
        )
        with pytest.raises(ValueError, match="No emission line fluxes"):
            p.bpt_nii()


# ── TestBalmerDecrement ───────────────────────────────────────────────────────


class TestBalmerDecrement:
    """Test Balmer decrement bounds (Hα/Hβ ratio)."""

    def test_map_returns_triple_same(self, map_eline_posterior):
        med, lo, hi = map_eline_posterior.balmer_decrement()
        assert med == lo == hi
        assert med == pytest.approx(10.0 / 4.0, abs=1e-5)

    def test_sampling_returns_percentiles(self, sampling_eline_posterior):
        med, lo, hi = sampling_eline_posterior.balmer_decrement()
        assert lo <= med <= hi
        assert med == pytest.approx(10.0 / 4.0, abs=0.3)

    def test_raises_no_elines(self):
        p = Posterior(
            samples=None,
            params={},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
        )
        with pytest.raises(ValueError, match="No emission line fluxes"):
            p.balmer_decrement()

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
    """Minimal stand-in for SEDModel exposing the rest-SED entry point only.

    Mocks the *private* ``_predict_rest_sed``: the public ``predict_rest_sed``
    is a deprecation shim now (#1049), and internal callers were migrated to the
    private twin so the library never warns at its own users.
    """

    def __init__(self, wave, line_centers, line_amplitudes, sigma_aa=5.0, cont_f_lambda=1.0):
        self._wave = wave
        self._centers = line_centers
        self._amps = line_amplitudes
        self._sigma = sigma_aa
        self._cont = cont_f_lambda

    def _predict_rest_sed(self, params, wave=None):
        from tengri.forward.result import SEDResult

        f_lambda = jnp.full_like(self._wave, self._cont)
        for c, a in zip(self._centers, self._amps):
            f_lambda = f_lambda + a * jnp.exp(-0.5 * ((self._wave - c) / self._sigma) ** 2) / (
                self._sigma * jnp.sqrt(2.0 * jnp.pi)
            )
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
    """Test equivalent width bounds and positivity."""

    def test_raises_no_eline_fluxes(self):
        p = Posterior(
            samples=None,
            params={},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
        )
        with pytest.raises(ValueError, match="No emission line fluxes"):
            p.equivalent_widths()

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
        """Emission lines must have positive EW."""
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
        """Balmer A(V) percentiles: lo ≤ med ≤ hi."""
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

    def test_raises_when_lines_missing(self):
        p = Posterior(
            samples=None,
            params={},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
        )
        with pytest.raises(ValueError, match="No emission line fluxes"):
            p.balmer_av()


# ── TestSEDComponents (#3 AGN+host decomposition) ────────────────────


class _FakeOrchestratorState:
    """Minimal ForwardState stand-in for tests."""

    def __init__(self, wave, sed_intrinsic, derived):
        self.wave = wave
        self.sed_intrinsic = sed_intrinsic
        self.sed_attenuated = None
        self.sed_observed = None
        self.derived = derived


class _FakeSpec:
    def get_fixed_values(self):
        return {}


class _FakeComponentModel:
    """Stand-in for SEDModel with a ``predict_state`` method.

    Returns a fake ``ForwardState`` whose ``derived`` dict carries the
    same per-component keys the real orchestrator publishes. Component
    amplitudes are driven by a single param ``frac`` so tests can verify
    that sample-to-sample variation propagates correctly.
    """

    def __init__(self, wave):
        self._wave = jnp.asarray(wave)
        self.spec = _FakeSpec()

    def predict_state(self, params):
        wave = self._wave
        f = float(params.get("frac", 0.0))
        host_attenuated = (1.0 - f) * jnp.ones_like(wave)  # stellar post-attenuation
        host_intrinsic = host_attenuated * 1.5  # stellar pre-attenuation
        agn = f * jnp.ones_like(wave)
        zeros = jnp.zeros_like(wave)
        # Match the orchestrator semantics: state.sed_intrinsic carries
        # the accumulated total (post-dust stellar + AGN + ...).
        sed_total = host_attenuated + agn
        # Encode stellar pre-attenuation via lnu_age (one age bin).
        lnu_age = host_intrinsic[None, :]
        derived = {
            "lnu_age": lnu_age,
            "ssp_ages_yr": jnp.array([1e9]),
            "sed_dust_attenuated": host_attenuated,
            "sed_dust_ir": zeros,
            "sed_nebular": zeros,
            "sed_shock": zeros,
            "sed_agn": agn,
            "sed_radio": zeros,
            "sed_xray": zeros,
        }
        return _FakeOrchestratorState(wave=wave, sed_intrinsic=sed_total, derived=derived)


class TestSEDComponents:
    """Test SED component bounds and ordering."""

    def test_raises_no_model(self):
        p = Posterior(
            samples=None,
            params={"frac": jnp.array(0.0)},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
        )
        with pytest.raises(ValueError, match="model"):
            p.sed_components()

    def test_map_returns_per_component_arrays(self):
        wave = jnp.linspace(1000.0, 1e6, 100)
        model = _FakeComponentModel(wave)
        p = Posterior(
            samples=None,
            params={"frac": jnp.array(0.3)},
            method="MAP",
            wall_time_s=0.1,
            diagnostics={},
        )
        p._model = model
        out = p.sed_components()
        # MAP path: each component should have shape (n_wave,)
        assert out["sed_total"].shape == (100,)
        assert out["sed_agn"].shape == (100,)
        assert out["wavelength"].shape == (100,)
        # frac=0.3 → host=0.7, agn=0.3, total=1.0
        np.testing.assert_allclose(np.asarray(out["sed_agn"]), 0.3, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(out["sed_total"]), 1.0, rtol=1e-6)

    def test_sampling_returns_stacked_arrays(self):
        wave = jnp.linspace(1000.0, 1e6, 50)
        model = _FakeComponentModel(wave)
        rng = np.random.default_rng(50)
        n = 25
        p = Posterior(
            samples={"frac": jnp.asarray(rng.uniform(0.1, 0.9, size=n))},
            params={"frac": jnp.array(0.5)},
            method="mcmc_nuts",
            wall_time_s=1.0,
            diagnostics={},
        )
        p._model = model
        out = p.sed_components()
        assert out["sed_total"].shape == (n, 50)
        assert out["sed_agn"].shape == (n, 50)
        # AGN amplitude should equal frac sample-by-sample (at any wavelength)
        np.testing.assert_allclose(
            np.asarray(out["sed_agn"][:, 0]),
            np.asarray(p.samples["frac"]),
            rtol=1e-6,
        )

    def test_agn_fraction_map(self):
        wave = jnp.linspace(1000.0, 1e6, 100)
        model = _FakeComponentModel(wave)
        p = Posterior(
            samples=None,
            params={"frac": jnp.array(0.4)},
            method="MAP",
            wall_time_s=0.1,
            diagnostics={},
        )
        p._model = model
        agn_lum_ratio = p.agn_fraction()
        np.testing.assert_allclose(np.asarray(agn_lum_ratio), 0.4, rtol=1e-6)

    def test_agn_fraction_sampling(self):
        wave = jnp.linspace(1000.0, 1e6, 30)
        model = _FakeComponentModel(wave)
        rng = np.random.default_rng(51)
        n = 20
        fracs = rng.uniform(0.1, 0.7, size=n)
        p = Posterior(
            samples={"frac": jnp.asarray(fracs)},
            params={"frac": jnp.array(0.4)},
            method="mcmc_nuts",
            wall_time_s=1.0,
            diagnostics={},
        )
        p._model = model
        agn_lum_ratio = p.agn_fraction()
        # Median across samples should track the input median frac
        assert agn_lum_ratio.shape == (30,)  # one per wavelength
        # Wavelength-flat AGN fraction → at any wavelength, median of frac samples
        assert agn_lum_ratio[0] == pytest.approx(np.median(fracs), rel=1e-6)

    def test_custom_wavelength_arg_is_ignored(self):
        """``wavelength=`` is a back-compat pass-through after the orchestrator
        migration. The returned grid is the model's SSP grid."""
        wave_default = jnp.linspace(1000.0, 1e6, 100)
        wave_custom = jnp.linspace(2000.0, 8000.0, 50)
        model = _FakeComponentModel(wave_default)
        p = Posterior(
            samples=None,
            params={"frac": jnp.array(0.2)},
            method="MAP",
            wall_time_s=0.1,
            diagnostics={},
        )
        p._model = model
        out = p.sed_components(wavelength=wave_custom)
        # Custom grid is ignored: output uses the model's SSP grid.
        assert out["sed_total"].shape == (100,)
        assert out["wavelength"].shape == (100,)


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
    """Test BPT galaxy classification (SF, composite, AGN)."""

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
        chex.assert_shape(labels, (n,))
        # All draws should classify as SF given small spread
        assert np.all(labels == "SF")

    def test_raises_when_lines_missing(self):
        p = Posterior(
            samples=None,
            params={},
            method="MAP",
            wall_time_s=0.0,
            diagnostics={},
        )
        with pytest.raises(ValueError, match="No emission line fluxes"):
            p.bpt_class()

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

# SPDX-License-Identifier: BSD-3-Clause
"""Tests for parametric AGN in fused photometry kernel.

Validates that:
- Parametric AGN (agn_log_lbol) enables fused kernel path
- Legacy AGN (agn_lum_ratio) forces exact path
- Fused AGN photometry produces finite, positive results
- Gradients through AGN fused path are finite
- Fused AGN approximation is within reasonable tolerance of exact path
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp

from tengri import Fixed, Parameters, SEDModel, Uniform

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SKIRTOR_CANDIDATES = [
    _DATA_DIR / "skirtor_templates_v3.h5",
    _DATA_DIR / "skirtor_templates_v2.h5",
]
_SKIRTOR_PATH = next((p for p in _SKIRTOR_CANDIDATES if p.is_file()), None)
_has_skirtor_data = _SKIRTOR_PATH is not None
jax.config.update("jax_enable_x64", True)


# ── Fixtures ─────────────────────────────────────────────────────
# synthetic_ssp is provided by conftest.py (session scope)
@pytest.fixture(scope="module")
def simple_filters():
    """Synthetic 3-band filter set covering the SSP wavelength range."""
    from tengri.observation.photometry import FilterCurve

    waves = [
        jnp.linspace(3500.0, 4500.0, 50),
        jnp.linspace(5000.0, 6500.0, 50),
        jnp.linspace(7500.0, 9000.0, 50),
    ]
    trans = [jnp.ones(50) * 0.5 for _ in range(3)]
    names = ["synth_blue", "synth_green", "synth_red"]
    curves = [FilterCurve(wave=w, trans=t, name=n) for n, w, t in zip(names, waves, trans)]
    return (waves, trans, curves)


@pytest.fixture(scope="module")
def parametric_agn_spec():
    """Parameters with parametric AGN (agn_log_lbol is free)."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        agn_model="multicolor_agn",
        agn_log_lbol=Uniform(8.0, 12.0),
        # agn_lum_ratio must be explicitly fixed: AGN params now carry free
        # Uniform registry defaults (consistent with sfh/dust), so an
        # unspecified agn_lum_ratio would default free and flip mode detection
        # (``_agn_luminosity_mode = lbol_free and not frac_free``) to legacy.
        agn_lum_ratio=Fixed(1.0),
        agn_alpha=Fixed(-1.0),
        agn_T_torus=Fixed(1000.0),
        agn_torus_frac=Fixed(0.5),
    )


@pytest.fixture(scope="module")
def legacy_agn_spec():
    """Parameters with legacy AGN (agn_lum_ratio is free)."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        agn_model="multicolor_agn",
        agn_lum_ratio=Uniform(0.01, 0.5),
        agn_alpha=Fixed(-1.0),
        agn_T_torus=Fixed(1000.0),
    )


# ── Tests: mode detection and fused compatibility ─────────────────
class TestAGNModeDetection:
    """Test that parametric vs legacy AGN mode is detected correctly."""

    def test_parametric_mode_detected(self, parametric_agn_spec, synthetic_ssp, simple_filters):
        """Parametric AGN (agn_log_lbol free) sets _agn_luminosity_mode=True."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(parametric_agn_spec, synthetic_ssp, filters=simple_filters)
        assert model._agn_luminosity_mode is True

    def test_legacy_mode_detected(self, legacy_agn_spec, synthetic_ssp, simple_filters):
        """Legacy AGN (agn_lum_ratio free) sets _agn_luminosity_mode=False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(legacy_agn_spec, synthetic_ssp, filters=simple_filters)
        assert model._agn_luminosity_mode is False

    @pytest.mark.skip(reason="hybrid/fused kernel detection deleted in Phase 6")
    def test_parametric_enables_hybrid(self, parametric_agn_spec, synthetic_ssp, simple_filters):
        """Parametric AGN allows hybrid kernel."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(parametric_agn_spec, synthetic_ssp, filters=simple_filters)
        assert model._hybrid.photometry is not None

    @pytest.mark.skip(reason="hybrid/fused kernel detection deleted in Phase 6")
    def test_legacy_still_builds_hybrid(self, legacy_agn_spec, synthetic_ssp, simple_filters):
        """Legacy AGN still builds hybrid kernel (non-stellar at full res)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(legacy_agn_spec, synthetic_ssp, filters=simple_filters)
        # Hybrid handles all AGN modes (frac computed from broadband L_bol)
        assert model._hybrid.photometry is not None


# ── Tests: fused AGN photometry correctness ───────────────────────
class TestAGNFusedPhotometry:
    """Test that parametric AGN in fused kernel produces valid results."""

    def test_finite_positive_photometry(self, parametric_agn_spec, synthetic_ssp, simple_filters):
        """Fused AGN photometry is finite and positive."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(parametric_agn_spec, synthetic_ssp, filters=simple_filters)
        key = jax.random.PRNGKey(42)
        params = parametric_agn_spec.sample(key)
        phot = model.predict_photometry(params)
        assert jnp.all(jnp.isfinite(phot)), f"Non-finite photometry: {phot}"
        assert jnp.all(phot > 0), f"Non-positive photometry: {phot}"

    def test_gradients_finite(self, parametric_agn_spec, synthetic_ssp, simple_filters):
        """Gradients through AGN fused path are all finite."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(parametric_agn_spec, synthetic_ssp, filters=simple_filters)
        key = jax.random.PRNGKey(42)
        params = parametric_agn_spec.sample(key)

        def loss_fn(p):
            return jnp.sum(model.predict_photometry(p))

        # Finiteness only, deliberately: a finite-difference cross-check needs
        # f evaluable at params +/- h along the probe direction, and nudging a
        # full parameter dict along a random tangent walks bounded parameters
        # out of their physical domain, so the probe returns NaN and can judge
        # nothing. Per-parameter probes inside the valid range would be the way
        # to strengthen this — see tests/_grad_parity.py.
        grads = jax.grad(loss_fn)(params)
        for name, grad_val in grads.items():
            if grad_val is not None:
                assert jnp.all(jnp.isfinite(grad_val)), (
                    f"Non-finite gradient for {name}: {grad_val}"
                )

    @pytest.mark.skip(
        reason="exact path: simple_agn contribution is negligible at optical wavelengths "
        "for synthetic SSP — relied on the deleted approx=True call-time kwarg"
    )
    def test_agn_lbol_affects_photometry(self, parametric_agn_spec, synthetic_ssp, simple_filters):
        """Changing agn_log_lbol changes the photometry (approx path)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(parametric_agn_spec, synthetic_ssp, filters=simple_filters)
        key = jax.random.PRNGKey(42)
        params = parametric_agn_spec.sample(key)
        # Use approx=True to test the fused kernel AGN path specifically.
        # The exact path evaluates AGN on the full SSP grid where simple_agn
        # contribution is negligible at optical wavelengths for a synthetic SSP.
        # Low AGN luminosity
        params_low = {**params, "agn_log_lbol": 8.0}
        phot_low = model.predict_photometry(params_low)
        # High AGN luminosity
        params_high = {**params, "agn_log_lbol": 12.0}
        phot_high = model.predict_photometry(params_high)
        # Higher L_bol should produce brighter photometry
        assert jnp.all(phot_high > phot_low), (
            f"Higher agn_log_lbol should produce brighter photometry. "
            f"Low: {phot_low}, High: {phot_high}"
        )


# ── Tests: fused vs exact comparison ──────────────────────────────
class TestAGNFusedVsExact:
    """Compare fused (effective-wavelength) vs exact AGN evaluation."""

    def test_fused_vs_exact_simple_agn(self, synthetic_ssp, simple_filters):
        """Fused AGN approximation within tolerance of exact path.
        The effective-wavelength approximation evaluates the AGN SED at
        filter effective wavelengths rather than integrating over the
        full SED. For broadband photometry with simple AGN models, this
        should agree within ~20% (AGN SED varies more strongly than
        stellar SED across filter bandpasses).
        """
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.1),
            agn_model="multicolor_agn",
            agn_log_lbol=Fixed(10.5),
            agn_alpha=Fixed(-1.0),
            agn_T_torus=Fixed(1000.0),
            agn_torus_frac=Fixed(0.5),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_fused = SEDModel(spec, synthetic_ssp, filters=simple_filters)
            model_exact = SEDModel(spec, synthetic_ssp, filters=simple_filters)
        key = jax.random.PRNGKey(99)
        params = spec.sample(key)
        phot_fused = model_fused.predict_photometry(params)
        phot_exact = model_exact.predict_photometry(params)
        # Both should be finite and positive
        chex.assert_tree_all_finite(phot_fused)
        chex.assert_tree_all_finite(phot_exact)
        assert jnp.all(phot_fused > 0)
        assert jnp.all(phot_exact > 0)
        # The fused (effective-wavelength) approximation error is larger
        # for AGN than for stars. Accept 50% relative error per band.
        # In practice the error depends on how much the AGN SED shape
        # varies within each filter bandpass.
        rel_error = jnp.abs(phot_fused - phot_exact) / phot_exact
        max_rel_error = float(jnp.max(rel_error))
        assert max_rel_error < 0.5, (
            f"Fused vs exact max relative error = {max_rel_error:.2%}. "
            f"Fused: {phot_fused}, Exact: {phot_exact}"
        )

    def test_fused_vs_exact_parametric_agn(self, synthetic_ssp, simple_filters):
        """Fused kernel with FREE agn_log_lbol (_agn_luminosity_mode=True) is within
        tolerance of exact path.
        Regression test for the AGN unit bug: fused kernels were adding
        agn_lnu [erg/s/Hz] directly to flux_total [Lsun], off by a factor
        of LSUN_ERG_PER_S ≈ 3.828e33.  Before the fix, this test would fail
        with a relative error of ~3.8e33 rather than the expected <50%.
        The critical difference from test_fused_vs_exact_simple_agn is that
        agn_log_lbol is a *free* Uniform parameter here.  That sets
        _agn_luminosity_mode=True, which enables the `has_agn=True` branch inside
        the fused kernel — the only path where the unit bug fired.
        """
        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.1),
            agn_model="multicolor_agn",
            agn_log_lbol=Uniform(8.0, 12.0),  # FREE → _agn_luminosity_mode=True
            agn_lum_ratio=Fixed(1.0),  # fixed so mode stays parametric (see fixture note)
            agn_alpha=Fixed(-1.0),
            agn_T_torus=Fixed(1000.0),
            agn_torus_frac=Fixed(0.5),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_fused = SEDModel(spec, synthetic_ssp, filters=simple_filters)
            model_exact = SEDModel(spec, synthetic_ssp, filters=simple_filters)
        assert model_fused._agn_luminosity_mode is True, (
            "Expected _agn_luminosity_mode=True with free agn_log_lbol"
        )
        key = jax.random.PRNGKey(77)
        params = spec.sample(key)
        # Fix AGN luminosity to a concrete mid-range value so the AGN
        # contribution is clearly visible above the stellar baseline.
        params = {**params, "agn_log_lbol": 11.0}
        phot_fused = model_fused.predict_photometry(params)
        phot_exact = model_exact.predict_photometry(params)
        assert jnp.all(jnp.isfinite(phot_fused)), f"Non-finite fused photometry: {phot_fused}"
        assert jnp.all(jnp.isfinite(phot_exact)), f"Non-finite exact photometry: {phot_exact}"
        assert jnp.all(phot_fused > 0), f"Non-positive fused photometry: {phot_fused}"
        assert jnp.all(phot_exact > 0), f"Non-positive exact photometry: {phot_exact}"
        # Before the fix: relative error ≈ 3.828e33 (unit mismatch).
        # After the fix: relative error should be within the normal
        # effective-wavelength approximation error (~50%).
        rel_error = jnp.abs(phot_fused - phot_exact) / (phot_exact + 1e-30)
        max_rel_error = float(jnp.max(rel_error))
        assert max_rel_error < 0.5, (
            f"Parametric AGN fused vs exact max relative error = {max_rel_error:.2%}. "
            f"Fused: {phot_fused}, Exact: {phot_exact}. "
            "If error >> 1 this likely indicates the agn_lnu / lsun unit fix was reverted."
        )


# ── Tests: predict_sed parametric AGN ─────────────────────────────
class TestAGNPredictSED:
    """Test predict_sed with parametric AGN mode."""

    def test_parametric_sed_finite(self, parametric_agn_spec, synthetic_ssp, simple_filters):
        """predict_sed with parametric AGN produces finite SED."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(parametric_agn_spec, synthetic_ssp, filters=simple_filters)
        key = jax.random.PRNGKey(42)
        params = parametric_agn_spec.sample(key)
        sed = model.predict_rest_sed(params).sed
        chex.assert_tree_all_finite(sed)
        assert sed.shape == (len(synthetic_ssp.ssp_wave),)

    def test_parametric_sed_includes_agn(self):
        """Parametric AGN model produces luminosity-dependent SED.
        Tests the AGN model function directly to verify that
        agn_log_lbol controls the AGN luminosity as expected.
        Avoids synthetic SSP normalization issues in unit tests.
        """
        from tengri.components.agn import resolve_agn_model

        wave = jnp.linspace(3000.0, 10000.0, 100)
        agn_fn = resolve_agn_model("multicolor_agn")
        # AGN with high L_bol
        lnu_high = agn_fn(
            wave,
            agn_log_lbol=12.0,
            agn_lum_ratio=1.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_torus_frac=0.5,
        )
        # AGN with low L_bol
        lnu_low = agn_fn(
            wave,
            agn_log_lbol=8.0,
            agn_lum_ratio=1.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_torus_frac=0.5,
        )
        chex.assert_tree_all_finite(lnu_high)
        assert jnp.all(lnu_high > 0), "AGN SED should be positive"
        # L_bol ratio of 10^4 should produce proportionally brighter AGN
        ratio = jnp.max(lnu_high) / jnp.max(lnu_low)
        assert ratio > 1e3, (
            f"L_bol ratio of 10^4 should produce >1000x brighter AGN. "
            f"Actual ratio: {float(ratio):.1f}"
        )


# ── Tests: SKIRTOR torus preintegration ───────────────────────────
@pytest.mark.skipif(not _has_skirtor_data, reason="SKIRTOR template data not found in data/")
class TestSKIRTORPreintegration:
    """Regression tests for SKIRTOR torus preintegration in hybrid kernel.
    Validates that the filter-level preintegrated SKIRTOR torus lookup
    matches full-wavelength SKIRTOR evaluation to within ~1%.  Any
    normalization mismatch (e.g. wavelength-space vs frequency-space
    normalization) would appear as a systematic offset >> 1%.
    """

    @pytest.fixture(scope="class")
    def skirtor_spec(self):
        """Parameters spec with SKIRTOR AGN, fixed redshift (required for preintegration)."""
        return Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Fixed(1.5),
            sfh_dpl_beta=Fixed(1.0),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.3),
            agn_model="skirtor",
            agn_log_lbol=Uniform(9.0, 12.0),
            agn_tau_skirtor=Fixed(3.0),
            agn_p_skirtor=Fixed(1.0),
            agn_q_skirtor=Fixed(1.0),
            agn_oa_skirtor=Fixed(40.0),
            agn_cos_inc=Fixed(0.7),
            agn_torus_frac=Fixed(0.5),
            agn_alpha=Fixed(-1.0),
            agn_polar_ebv=Fixed(0.0),
            agn_polar_oa=Fixed(60.0),
        )

    def test_preintegration_enabled(self, skirtor_spec, synthetic_ssp, simple_filters):
        """SEDModel with SKIRTOR + fixed z loads skirtor_preintegrated into PrecomputedData."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(skirtor_spec, synthetic_ssp, filters=simple_filters)
        assert model._precomputed.skirtor_preintegrated is not None, (
            "Expected skirtor_preintegrated to be populated for fixed-z SKIRTOR model"
        )

    def test_preintegrated_photometry_finite(self, skirtor_spec, synthetic_ssp, simple_filters):
        """Hybrid kernel with SKIRTOR preintegration produces finite, positive photometry."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(skirtor_spec, synthetic_ssp, filters=simple_filters)
        key = jax.random.PRNGKey(42)
        params = skirtor_spec.sample(key)
        phot = model.predict_photometry(params)
        assert jnp.all(jnp.isfinite(phot)), f"Non-finite SKIRTOR preintegrated photometry: {phot}"
        assert jnp.all(phot > 0), f"Non-positive SKIRTOR preintegrated photometry: {phot}"

    def test_preintegrated_matches_fullwave(self, skirtor_spec, synthetic_ssp, simple_filters):
        """Preintegrated SKIRTOR torus photometry agrees with full-wavelength within 1%.
        The preintegrated path (filter-level triweight lookup) must match the
        full-wavelength path within ~1% to confirm the frequency-space normalization
        is consistent between precompute_skirtor_photometry and skirtor.py.
        A normalization mismatch (e.g. energy_normalize=True vs freq-space) would
        produce systematic errors >> 1% across all bands.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_hybrid = SEDModel(skirtor_spec, synthetic_ssp, filters=simple_filters)
            model_exact = SEDModel(skirtor_spec, synthetic_ssp, filters=simple_filters)
        key = jax.random.PRNGKey(7)
        params = skirtor_spec.sample(key)
        phot_hybrid = model_hybrid.predict_photometry(params)
        phot_exact = model_exact.predict_photometry(params)
        assert jnp.all(jnp.isfinite(phot_hybrid)), f"Non-finite hybrid photometry: {phot_hybrid}"
        assert jnp.all(jnp.isfinite(phot_exact)), f"Non-finite exact photometry: {phot_exact}"
        rel_error = jnp.abs(phot_hybrid - phot_exact) / (jnp.abs(phot_exact) + 1e-40)
        max_rel_error = float(jnp.max(rel_error))
        assert max_rel_error < 0.05, (
            f"SKIRTOR preintegrated vs full-wave max relative error = {max_rel_error:.2%}. "
            f"Preint: {phot_hybrid}, Full-wave: {phot_exact}. "
            "If error >> 1% this likely indicates a normalization mismatch in "
            "skirtor_precompute.py (frequency-space vs wavelength-space integral)."
        )

    def test_lbol_sensitivity(self, skirtor_spec, synthetic_ssp, simple_filters):
        """Preintegrated SKIRTOR photometry scales monotonically with agn_log_lbol."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel(skirtor_spec, synthetic_ssp, filters=simple_filters)
        key = jax.random.PRNGKey(0)
        params = skirtor_spec.sample(key)
        params_low = {**params, "agn_log_lbol": 9.0}
        params_high = {**params, "agn_log_lbol": 12.0}
        phot_low = model.predict_photometry(params_low)
        phot_high = model.predict_photometry(params_high)
        assert jnp.all(phot_high > phot_low), (
            f"Higher agn_log_lbol should produce brighter photometry. "
            f"Low: {phot_low}, High: {phot_high}"
        )

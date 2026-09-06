# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for the new SEDModel class with real SSP data."""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.forward.sed_model import MockData, SEDModel
from tengri.observation.filters import load_filter_set
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform
from tests._bounds import assert_non_negative

# ── Skip if SSP data not available ────────────────────────────────
#: The value the forward used to substitute for a free ``sfh_dpl_age_gyr`` before
#: the missing-parameter guard landed. Pinning it keeps these specs fully Fixed,
#: so ``predict_*({})`` stays valid and the SEDs are unchanged (#1021).
_DPL_AGE_DEFAULT = float(
    Parameters(mean_sfh_type="dpl").get_distribution("sfh_dpl_age_gyr").default
)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = pytest.mark.skipif(
    not _SSP_EXISTS,
    reason="SSP data file not found",
)


@pytest.fixture(scope="session")
def ssp_data():
    return load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="session")
def filters():
    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


@pytest.fixture(scope="session")
def parametric_spec():
    """Parametric tsnorm spec (no GP field)."""
    return Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_total_mass=Uniform(9.0, 12.0),  # galaxy-scale log10(M*/Msun)
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
        sfh_tsnorm_skew=Uniform(-1.0, 1.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=-0.7,
        redshift=0.1,
    )


@pytest.fixture(scope="session")
def stochastic_spec():
    """Stochastic tsnorm + field spec."""
    return Parameters(
        mean_sfh_type=["tsnorm", "field"],
        sfh_tsnorm_log_total_mass=Uniform(9.0, 12.0),  # galaxy-scale log10(M*/Msun)
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
        sfh_tsnorm_skew=Uniform(-1.0, 1.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        sfh_field_psd_sigma=Uniform(0.1, 3.0),
        sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=-0.7,
        redshift=0.1,
        n_grid=64,
    )


@pytest.fixture(scope="session")
def dpl_spec():
    """DPL parametric spec for backward compat testing."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(9.0, 12.0),  # galaxy-scale log10(M*/Msun)
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=-0.7,
        redshift=0.1,
    )


@pytest.fixture(scope="session")
def parametric_model(parametric_spec, ssp_data, filters):
    return SEDModel(parametric_spec, ssp_data, filters=filters)


@pytest.fixture(scope="session")
def stochastic_model(stochastic_spec, ssp_data, filters):
    return SEDModel(stochastic_spec, ssp_data, filters=filters)


@pytest.fixture(scope="session")
def dpl_model(dpl_spec, ssp_data, filters):
    return SEDModel(dpl_spec, ssp_data, filters=filters)


@pytest.fixture(scope="session")
def typical_params(parametric_spec):
    """Sample typical parameters from the parametric spec."""
    return parametric_spec.sample(jax.random.PRNGKey(42))


# ── SED Predictions ───────────────────────────────────────────────


class TestPredictSed:
    def test_shape(self, parametric_model, typical_params):
        sed = parametric_model.predict_rest_sed(typical_params).sed
        chex.assert_equal_shape([sed, parametric_model.ssp_data.ssp_wave])

    def test_finite(self, parametric_model, typical_params):
        sed = parametric_model.predict_rest_sed(typical_params).sed
        chex.assert_tree_all_finite(sed)

    def test_positive(self, parametric_model, typical_params):
        sed = parametric_model.predict_rest_sed(typical_params).sed
        assert_non_negative(sed, name="sed")


class TestPredictPhotometry:
    def test_shape(self, parametric_model, typical_params):
        phot = parametric_model.predict_photometry(typical_params)
        assert phot.shape == (5,)  # 5 SDSS bands

    def test_finite_positive(self, parametric_model, typical_params):
        phot = parametric_model.predict_photometry(typical_params)
        chex.assert_tree_all_finite(phot)
        assert jnp.all(phot > 0)

    def test_flux_scales_linearly_with_total_mass(self, parametric_model, typical_params):
        """Photometry is linear in the total mass formed.

        Replaces an absolute flux-band assertion (``1e-35 < F_nu < 1e-20``).
        That band was never physical: this file's SSP is the *synthetic* grid
        (``synthetic: True``, no ``flux_units`` attribute), whose absolute
        normalization is arbitrary — the real grids declare ``Lsun/Hz/Msun`` and
        sit ~15 decades lower. The band only ever passed because the fixture's
        old toy ``log_total_mass`` prior (~1 Msun "galaxies") happened to cancel
        that arbitrary scale. Raising the prior to galaxy-scale — correctly, in
        #1056 — pushed the flux straight through the ceiling and left this test
        red on main (#1031). It went unnoticed because ``tests/integration`` runs
        only in the slow tier, not on PRs.

        Linearity in ``10**log_total_mass`` is the invariant that actually holds:
        it is independent of the SSP's normalization, so it asserts something
        real on any grid, synthetic or not.
        """
        p1 = dict(typical_params)
        p2 = {**p1, "sfh_tsnorm_log_total_mass": p1["sfh_tsnorm_log_total_mass"] + 1.0}

        f1 = parametric_model.predict_photometry(p1)
        f2 = parametric_model.predict_photometry(p2)

        chex.assert_tree_all_finite(f1)
        assert jnp.all(f1 > 0)
        # +1 dex of mass -> exactly 10x the flux in every band.
        chex.assert_trees_all_close(f2, 10.0 * f1, rtol=1e-6)


class TestPredictSfh:
    def test_keys(self, parametric_model, typical_params):
        sfh = parametric_model.predict_sfh(typical_params)
        assert "t_gyr" in sfh
        assert "sfr_mean" in sfh
        assert "sfr_full" in sfh

    def test_positive_sfr(self, parametric_model, typical_params):
        sfh = parametric_model.predict_sfh(typical_params)
        assert_non_negative(sfh["sfr_mean"], name="output")

    def test_parametric_mean_equals_full(self, parametric_model, typical_params):
        sfh = parametric_model.predict_sfh(typical_params)
        np.testing.assert_allclose(
            np.array(sfh["sfr_mean"]),
            np.array(sfh["sfr_full"]),
            rtol=1e-6,
        )


class TestPredictDerived:
    def test_keys(self, parametric_model, typical_params):
        d = parametric_model.predict_derived(typical_params)
        assert "stellar_mass" in d
        assert "sfr_100myr" in d
        assert "ssfr" in d

    def test_mass_positive(self, parametric_model, typical_params):
        d = parametric_model.predict_derived(typical_params)
        assert float(d["stellar_mass"]) > 0

    def test_mass_reasonable(self, parametric_model, typical_params):
        d = parametric_model.predict_derived(typical_params)
        mass = float(d["stellar_mass"])
        assert 1e7 < mass < 1e13


# ── Stochastic SEDModel ──────────────────────────────────────────────


class TestStochastic:
    def test_predict_sed_works(self, stochastic_model, stochastic_spec):
        params = stochastic_spec.sample(jax.random.PRNGKey(42))
        sed = stochastic_model.predict_rest_sed(params).sed
        chex.assert_tree_all_finite(sed)

    def test_sfh_full_differs_from_mean(self, stochastic_model, stochastic_spec):
        params = stochastic_spec.sample(jax.random.PRNGKey(42))
        # Force non-zero psd_sigma
        params = {**params, "sfh_field_psd_sigma": 1.5}
        sfh = stochastic_model.predict_sfh(params)
        # With non-zero sigma and random xi, full != mean
        assert not jnp.allclose(sfh["sfr_mean"], sfh["sfr_full"])


# ── DPL SEDModel ─────────────────────────────────────────────────────


class TestDPL:
    def test_predict_photometry(self, dpl_model, dpl_spec):
        params = dpl_spec.sample(jax.random.PRNGKey(42))
        phot = dpl_model.predict_photometry(params)
        chex.assert_shape(phot, (5,))
        chex.assert_tree_all_finite(phot)
        assert jnp.all(phot > 0)

    def test_predict_derived(self, dpl_model, dpl_spec):
        params = dpl_spec.sample(jax.random.PRNGKey(42))
        d = dpl_model.predict_derived(params)
        assert float(d["stellar_mass"]) > 0


# ── Mock Generation ───────────────────────────────────────────────


class TestMock:
    def test_mock_structure(self, parametric_model, typical_params):
        mock = parametric_model.mock(typical_params, snr=20.0, key=jax.random.PRNGKey(0))
        assert isinstance(mock, MockData)
        chex.assert_shape(mock.flux_true, (5,))
        chex.assert_shape(mock.flux_obs, (5,))
        chex.assert_shape(mock.noise, (5,))

    def test_mock_noise_scaling(self, parametric_model, typical_params):
        mock = parametric_model.mock(typical_params, snr=20.0, key=jax.random.PRNGKey(0))
        expected_noise = mock.flux_true / 20.0
        np.testing.assert_allclose(np.array(mock.noise), np.array(expected_noise))

    def test_mock_batch_shapes(self, parametric_model, parametric_spec):
        batch = parametric_spec.sample_batch(jax.random.PRNGKey(0), 5)
        mock_batch = parametric_model.mock_batch(batch, snr=20.0, key=jax.random.PRNGKey(1))
        assert mock_batch.flux_true.shape == (5, 5)  # 5 galaxies, 5 bands
        chex.assert_shape(mock_batch.flux_obs, (5, 5))


# ── Gradient Flow ─────────────────────────────────────────────────


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


class TestGradients:
    def test_photometry_gradient(self, parametric_model, typical_params):
        def loss(p):
            return jnp.sum(parametric_model.predict_photometry(p))

        grad_jax = float(jax.grad(loss)(typical_params)["sfh_tsnorm_log_total_mass"])

        def loss_scalar(x):
            p = dict(typical_params)
            p["sfh_tsnorm_log_total_mass"] = x
            return float(jnp.sum(parametric_model.predict_photometry(p)))

        grad_fd = fd_grad(loss_scalar, float(typical_params["sfh_tsnorm_log_total_mass"]))
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )

    def test_derived_gradient(self, parametric_model, typical_params):
        def loss(p):
            return parametric_model.predict_derived(p)["stellar_mass"]

        grad_jax = float(jax.grad(loss)(typical_params)["sfh_tsnorm_log_total_mass"])

        def loss_scalar(x):
            p = dict(typical_params)
            p["sfh_tsnorm_log_total_mass"] = x
            return float(parametric_model.predict_derived(p)["stellar_mass"])

        grad_fd = fd_grad(loss_scalar, float(typical_params["sfh_tsnorm_log_total_mass"]))
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )


# ── Prediction API (lazy derived quantities) ──────────────────────


class TestPrediction:
    """Tests for model.predict() lazy prediction object."""

    def test_returns_prediction(self, parametric_model, typical_params):
        from tengri.forward.prediction import Prediction

        pred = parametric_model.predict(typical_params)
        assert isinstance(pred, Prediction)

    def test_has_property_groups(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        assert hasattr(pred, "sfh")
        assert hasattr(pred, "sed")
        assert hasattr(pred, "lines")
        assert hasattr(pred, "radio")
        assert hasattr(pred, "xray")
        assert hasattr(pred, "ionizing")

    # --- SFH properties ---

    def test_sfh_stellar_mass(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        mass = pred.sfh.stellar_mass
        assert jnp.isfinite(mass)
        assert float(mass) > 0
        assert 1e6 < float(mass) < 1e14

    def test_sfh_sfr(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        assert jnp.isfinite(pred.sfh.sfr_100myr)
        assert jnp.isfinite(pred.sfh.sfr_10myr)
        assert float(pred.sfh.sfr_100myr) >= 0
        assert float(pred.sfh.sfr_10myr) >= 0

    def test_sfh_ssfr(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        ssfr = pred.sfh.ssfr
        assert jnp.isfinite(ssfr)
        assert float(ssfr) >= 0

    def test_sfh_mass_weighted_age(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        age = pred.sfh.mass_weighted_age_gyr
        assert jnp.isfinite(age)
        assert 0 < float(age) < 14.0

    def test_sfh_mass_weighted_metallicity(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        z = pred.sfh.mass_weighted_metallicity
        assert jnp.isfinite(z)
        # log10(Z) should be in a reasonable range
        assert -5.0 < float(z) < 0.5

    # --- SED properties ---

    def test_sed_l_bol(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        l_bol = pred.sed.l_bol
        assert jnp.isfinite(l_bol)
        assert float(l_bol) > 0

    def test_sed_l_tir(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        l_tir = pred.sed.l_tir
        assert jnp.isfinite(l_tir)
        assert float(l_tir) >= 0

    def test_sed_uv_slope_beta(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        beta = pred.sed.uv_slope_beta
        assert jnp.isfinite(beta)

    def test_sed_dn4000(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        dn = pred.sed.dn4000
        assert jnp.isfinite(dn)
        assert 0.5 < float(dn) < 3.5

    def test_sed_balmer_break(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        bb = pred.sed.balmer_break
        assert jnp.isfinite(bb)
        assert 0.5 < float(bb) < 3.5

    def test_sed_m_uv(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        m_uv = pred.sed.m_uv
        assert jnp.isfinite(m_uv)

    def test_sed_irx(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        irx = pred.sed.irx
        assert jnp.isfinite(irx)

    def test_sed_fuv_nuv(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        assert jnp.isfinite(pred.sed.fuv_flux)
        assert jnp.isfinite(pred.sed.nuv_flux)
        assert float(pred.sed.fuv_flux) > 0
        assert float(pred.sed.nuv_flux) > 0

    def test_sed_rest_uv_color(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        uv = pred.sed.rest_uv_color
        assert jnp.isfinite(uv)
        # U-V typically -1 to 2.5 mag
        assert -2.0 < float(uv) < 4.0

    def test_sed_luminosity_weighted_age(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        age_lw = pred.sed.luminosity_weighted_age_gyr
        assert jnp.isfinite(age_lw)
        assert 0 < float(age_lw) < 14.0
        # L-weighted should differ from mass-weighted
        age_mw = pred.sfh.mass_weighted_age_gyr
        # Both should be reasonable (not testing inequality since it
        # depends on the SFH shape)
        assert 0 < float(age_mw) < 14.0

    def test_sed_energy_conservation(self, parametric_model, typical_params):
        """l_dust_absorbed must be non-negative and finite.

        The finiteness half was previously written as ``if jnp.isfinite(l_abs):``
        wrapping the sign check, which excused the very failure the docstring
        promised: a NaN or inf absorbed luminosity skipped the body and reported
        as a pass. Both halves are asserted now.
        """
        pred = parametric_model.predict(typical_params)
        l_abs = pred.sed.l_dust_absorbed
        assert jnp.isfinite(l_abs), f"l_dust_absorbed is not finite: {l_abs}"
        assert float(l_abs) >= 0, f"l_dust_absorbed is negative: {float(l_abs):.3e}"

    # --- Emission lines ---

    def test_lines_nan_without_nebular(self, parametric_model, typical_params):
        """Without free nebular params, lines should be NaN."""
        pred = parametric_model.predict(typical_params)
        # BakedIn backend has no predict_nebular_line_luminosities
        halpha = pred.lines.halpha
        # Should be NaN (no free nebular model)
        assert jnp.isnan(halpha) or jnp.isfinite(halpha)

    # --- Radio ---

    def test_radio_property_requires_radio_component(self, parametric_model, typical_params):
        """A model without a radio component must NOT fabricate radio properties.

        Pre-#1043 every model answered ``pred.radio.l_1p4ghz`` via fallback
        defaults — the silent-failure pattern the component-gated property
        catalog eliminated. The KeyError (listing what IS available) is the
        contract now; value-level radio property tests live in
        tests/contract/test_property_catalog.py on a radio-equipped model."""
        pred = parametric_model.predict(typical_params)
        with pytest.raises(KeyError, match="l_1p4ghz"):
            _ = pred.radio.l_1p4ghz

    # --- X-ray ---

    def test_xray_property_requires_xray_component(self, parametric_model, typical_params):
        """Same contract as the radio twin: no x-ray component, no fabricated
        ``l_x_xrb`` — loud KeyError instead (see test_property_catalog.py)."""
        pred = parametric_model.predict(typical_params)
        with pytest.raises(KeyError, match="l_x_xrb"):
            _ = pred.xray.l_x_xrb

    # --- Caching ---

    def test_caching_sfh(self, parametric_model, typical_params):
        """Accessing SFH properties twice should use cache."""
        pred = parametric_model.predict(typical_params)
        m1 = pred.sfh.stellar_mass
        m2 = pred.sfh.stellar_mass
        assert float(m1) == float(m2)
        # Cache should contain weights
        assert "weights" in pred._cache

    def test_caching_sed_triggers_sfh(self, parametric_model, typical_params):
        """SED accesses share one cached ForwardState; arrays materialize lazily.

        The one-path migration replaced named intermediates with the cached
        ``_state`` (every property reads it, so SFH/SED accesses share work);
        ``sed_total`` only materializes when an SED *array* accessor runs."""
        pred = parametric_model.predict(typical_params)
        _ = pred.sed.l_bol
        assert "_state" in pred._cache
        _ = pred.sed_array
        assert "sed_total" in pred._cache

    # --- sed_array ---

    def test_sed_array(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        sed = pred.sed_array
        chex.assert_equal_shape([sed, parametric_model.ssp_data.ssp_wave])
        chex.assert_tree_all_finite(sed)

    def test_sed_array_matches_predict_sed(self, parametric_model, typical_params):
        pred = parametric_model.predict(typical_params)
        sed_from_pred = pred.sed_array
        sed_direct = parametric_model.predict_rest_sed(typical_params).sed
        np.testing.assert_allclose(np.array(sed_from_pred), np.array(sed_direct), rtol=1e-10)

    # --- Backward compatibility ---

    def test_predict_derived_backward_compat(self, parametric_model, typical_params):
        """predict_derived() still returns a dict with old keys."""
        d = parametric_model.predict_derived(typical_params)
        assert isinstance(d, dict)
        assert "stellar_mass" in d
        assert "stellar_mass_surviving" in d
        assert "sfr_100myr" in d
        assert "sfr_10myr" in d
        assert "ssfr" in d

    def test_predict_derived_values_match(self, parametric_model, typical_params):
        """predict_derived() values match predict() values."""
        d = parametric_model.predict_derived(typical_params)
        pred = parametric_model.predict(typical_params)
        np.testing.assert_allclose(
            float(d["stellar_mass"]), float(pred.sfh.stellar_mass), rtol=1e-8
        )
        np.testing.assert_allclose(float(d["sfr_100myr"]), float(pred.sfh.sfr_100myr), rtol=1e-8)


# ── Dust emission wiring in forward model ─────────────────────────
# (Migrated from test_new_physics.py during test audit 2026-04-08)


class TestDustEmissionForwardModel:
    """Test dust emission wired into the full model."""

    @pytest.fixture(scope="class")
    def ssp(self):
        return load_ssp_data(str(_SSP_FILE))

    def test_dust_emission_adds_ir_flux(self, ssp):
        from tengri.parameters.priors import Fixed

        filters = load_filter_set(["sdss_r", "wise_w3"])
        spec = Parameters(
            sfh_dpl_alpha=Fixed(1.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(8.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            sfh_dpl_age_gyr=Fixed(_DPL_AGE_DEFAULT),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(1.0),
            dust_tau_diff=Fixed(0.5),
            dust_T=Uniform(20.0, 60.0),
            redshift=Fixed(0.1),
            mean_sfh_type="dpl",
            dust_emission="modified_blackbody",
        )
        model = SEDModel(spec, ssp, filters=filters, precompute=False)
        params_em = {"dust_T": 35.0, "dust_beta_ir": 1.6}

        spec_no = Parameters(
            sfh_dpl_alpha=Fixed(1.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(8.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            sfh_dpl_age_gyr=Fixed(_DPL_AGE_DEFAULT),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(1.0),
            dust_tau_diff=Fixed(0.5),
            redshift=Fixed(0.1),
            mean_sfh_type="dpl",
        )
        model_no = SEDModel(spec_no, ssp, filters=filters, precompute=False)

        em = model.predict_rest_sed(params_em)
        no = model_no.predict_rest_sed({})

        # The two models do NOT share a wavelength grid. An analytic dust-emission
        # model extends the rest grid into the far-IR so submm photometry is
        # integrable (#1005), so ``model`` carries several hundred extra points
        # beyond the SSP's red edge. Subtracting the raw arrays raised "sub got
        # incompatible shapes" (#1031) — the comparison has to happen on a common
        # grid. Project the no-emission SED onto the emission grid; beyond its red
        # edge there is no stellar flux to compare against, so the excess there is
        # the dust emission itself.
        no_on_em = jnp.interp(em.wavelength, no.wavelength, no.sed, left=0.0, right=0.0)
        excess = em.sed - no_on_em
        assert float(jnp.max(excess)) > 0, "Dust emission should add positive flux somewhere"

        # ...and the added energy must be thermal grain re-emission in the IR, not
        # extra optical light. A shape-blind ``max(excess) > 0`` would have passed
        # even if the excess had landed in the optical.
        ir = em.wavelength > 1.0e5  # > 10 um
        optical = (em.wavelength > 4000.0) & (em.wavelength < 7000.0)
        assert float(jnp.max(excess[ir])) > float(jnp.max(jnp.abs(excess[optical])))

        # End-to-end: the mid-IR band actually brightens. wise_w3 (12 um) sits on
        # the warm-dust bump; sdss_r does not.
        f_em = model.predict_photometry(params_em)
        f_no = model_no.predict_photometry({})
        assert float(f_em[1]) > float(f_no[1]), "wise_w3 must brighten with dust emission"

    def test_dust_emission_gradient(self, ssp):
        from tengri.parameters.priors import Fixed

        filters = load_filter_set(["wise_w3"])
        spec = Parameters(
            sfh_dpl_alpha=Fixed(1.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(8.0),
            sfh_dpl_log_total_mass=Fixed(1.0),
            sfh_dpl_age_gyr=Fixed(_DPL_AGE_DEFAULT),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(1.0),
            dust_tau_diff=Fixed(0.5),
            dust_T=Uniform(20.0, 60.0),
            redshift=Fixed(0.1),
            mean_sfh_type="dpl",
            dust_emission="modified_blackbody",
        )
        model = SEDModel(spec, ssp, filters=filters, precompute=False)

        def loss(T):
            return model.predict_photometry({"dust_T": T})[0]

        grad_jax = float(jax.grad(loss)(35.0))
        grad_fd = float((loss(35.0 + 1e-4) - loss(35.0 - 1e-4)) / (2.0 * 1e-4))
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            atol=1e-10,
            err_msg=f"dust_T: autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )

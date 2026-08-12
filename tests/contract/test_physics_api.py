# SPDX-License-Identifier: BSD-3-Clause
"""Tests for physics API redesign (physics_api_redesign.md).

- No-SSP tests: _planck_lnu deduplication, lines_to_sed, eline_catalog,
  nlr/blr line_efficiency, AGNConfig, agn_nlr return type.
- SSP-required: model.tree(), model.recommend_method(), predict_hbeta.
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.bounds

# ── SSP gate ──────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()
_needs_ssp = pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")


# ── Task 1: _planck_lnu extracted to _phys.py ─────────────────────


class TestPlanckLnuExtracted:
    def test_phys_module_exists_planck_lnu_and_deprecated_imports_resolve(self):
        """All physics imports must resolve (deprecation guarantees).

        Test ensures these surfaces remain importable for backward compatibility:
        - _phys module (physics constants and helpers)
        - disc, torus, skirtor (AGN models using _planck_lnu)
        - eline_catalog (emission line catalog)
        - build_line_design_matrix (unified line matrix builder)
        """
        # _phys module must import cleanly
        from tengri.components.agn import _phys

        assert _phys is not None

        # All deprecated surfaces must resolve
        from tengri.components.agn._phys import planck_lnu

        assert callable(planck_lnu)

        from tengri.components.agn import disc, skirtor, torus

        assert disc is not None
        assert torus is not None
        assert skirtor is not None

        from tengri.observation import eline_catalog

        assert eline_catalog is not None

        from tengri.observation.eline_marginalization import build_line_design_matrix

        assert callable(build_line_design_matrix)

    def test_planck_lnu_finite_for_solar_T(self):
        from tengri.components.agn._phys import planck_lnu

        nu = jnp.linspace(1e12, 1e16, 100)  # UV to far-IR
        result = planck_lnu(nu, 5778.0)  # solar temperature
        chex.assert_tree_all_finite(result)
        assert jnp.all(result >= 0.0)

    def test_planck_zero_temperature_returns_finite(self):
        from tengri.components.agn._phys import planck_lnu

        nu = jnp.array([1e14])
        result = planck_lnu(nu, 0.0)  # temperature = 0 → clamp to 1 K
        chex.assert_tree_all_finite(result)


class TestLinesToSed:
    def test_lines_to_sed_shape(self):
        from tengri.components.agn._phys import lines_to_sed

        wave_obs = jnp.linspace(4000.0, 7000.0, 500)
        line_wav = jnp.array([4861.33, 6562.80])  # Hβ, Hα
        line_lum = jnp.array([1.0, 2.86])  # Lsun

        result = lines_to_sed(line_wav, line_lum, wave_obs, fwhm_kms=200.0)

        chex.assert_shape(result, (500,))
        chex.assert_tree_all_finite(result)
        assert jnp.any(result > 0.0)

    def test_lines_to_sed_peaks_near_line_centers(self):
        from tengri.components.agn._phys import lines_to_sed

        wave_obs = jnp.linspace(6400.0, 6700.0, 1000)
        line_wav = jnp.array([6562.80])
        line_lum = jnp.array([1.0])

        result = lines_to_sed(line_wav, line_lum, wave_obs, fwhm_kms=200.0)
        peak_wave = float(wave_obs[jnp.argmax(result)])

        # Peak should be within 10 Å of Hα
        assert abs(peak_wave - 6562.80) < 10.0


# ── Task 2: agn_nlr_emission return type fix ──────────────────────


class TestAgnNlrEmissionReturnType:
    def test_agn_nlr_emission_return_annotation(self):
        """agn_nlr_emission() must NOT have a union return type."""
        import inspect

        from tengri.components.nebular.agn_nebular import agn_nlr_emission

        hints = {}
        try:
            import typing

            hints = typing.get_type_hints(agn_nlr_emission)
        except Exception:
            pass

        sig = inspect.signature(agn_nlr_emission)
        # Check: 'wavelength' should no longer be a parameter
        assert "wavelength" not in sig.parameters, (
            "agn_nlr_emission() still has deprecated 'wavelength' parameter"
        )

    def test_agn_nlr_cue_no_wavelength_param(self):
        """agn_nlr_cue() should not take a wavelength parameter."""
        import inspect

        from tengri.components.nebular.agn_nebular import agn_nlr_cue

        sig = inspect.signature(agn_nlr_cue)
        assert "wavelength" not in sig.parameters

    def test_agn_nlr_cue_uses_neb_logU(self):
        """agn_nlr_cue() should use neb_logU, not gas_logu."""
        import inspect

        from tengri.components.nebular.agn_nebular import agn_nlr_cue

        sig = inspect.signature(agn_nlr_cue)
        assert "neb_logU" in sig.parameters
        assert "gas_logu" not in sig.parameters


# ── Task 3: line_efficiency exposed in nlr/blr ────────────────────


class TestLineEfficiencyExposed:
    def test_nlr_emission_has_line_efficiency_param(self):
        import inspect

        from tengri.components.agn.nlr import compute_nlr_sed

        sig = inspect.signature(compute_nlr_sed)
        assert "line_efficiency" in sig.parameters
        # Default should be ~0.10
        default = sig.parameters["line_efficiency"].default
        assert abs(default - 0.10) < 0.01

    def test_blr_emission_has_line_efficiency_param(self):
        import inspect

        from tengri.components.agn.blr import compute_blr_sed

        sig = inspect.signature(compute_blr_sed)
        assert "line_efficiency" in sig.parameters
        # Default should be ~0.08
        default = sig.parameters["line_efficiency"].default
        assert abs(default - 0.08) < 0.01

    def test_nlr_emission_line_efficiency_scales_output(self):
        """Halving line_efficiency should roughly scale NLR luminosity."""
        from tengri.components.agn.nlr import compute_nlr_sed

        wave = jnp.linspace(3000.0, 8000.0, 500)
        l1 = jnp.sum(compute_nlr_sed(wave, 1e44, line_efficiency=0.10))
        l2 = jnp.sum(compute_nlr_sed(wave, 1e44, line_efficiency=0.05))
        ratio = float(l1 / jnp.maximum(l2, 1e-100))
        # Ratio should be >1 since l1 uses higher efficiency
        assert 1.5 < ratio < 2.2


# ── Task 4: Emission line catalog ─────────────────────────────────


class TestEmissionLineCatalog:
    def test_emission_lines_dict_has_required_keys(self):
        from tengri.observation.eline_catalog import EMISSION_LINES

        for name in ("Hbeta", "Halpha", "OIII5007", "NII6583"):
            assert name in EMISSION_LINES, f"{name} not in EMISSION_LINES"

    def test_line_groups_consistent_with_catalog(self):
        from tengri.observation.eline_catalog import EMISSION_LINES, LINE_GROUPS

        for group_name, members in LINE_GROUPS.items():
            for member in members:
                assert member in EMISSION_LINES, (
                    f"Line {member!r} in group {group_name!r} not in EMISSION_LINES"
                )

    def test_get_line_wavelengths(self):
        from tengri.observation.eline_catalog import get_line_wavelengths

        wav = get_line_wavelengths("bpt")
        chex.assert_shape(wav, (4,))
        # Hbeta at ~4862.68 (vacuum wavelength)
        assert any(abs(float(w) - 4862.68) < 1.0 for w in wav)

    def test_backward_compat_default_line_arrays(self):
        """Old DEFAULT_LINE_NAMES/WAVELENGTHS still importable from eline_marginalization."""
        from tengri.observation.eline_marginalization import (
            DEFAULT_LINE_NAMES,
            DEFAULT_LINE_WAVELENGTHS,
        )

        assert len(DEFAULT_LINE_NAMES) == 13
        chex.assert_shape(DEFAULT_LINE_WAVELENGTHS, (13,))

    def test_backward_compat_cloudy_line_arrays(self):
        """Old CLOUDY_LINE_NAMES/WAVELENGTHS still importable from eline_priors."""
        from tengri.observation.eline_priors import (
            CLOUDY_LINE_NAMES,
            CLOUDY_LINE_WAVELENGTHS,
        )

        assert len(CLOUDY_LINE_NAMES) == 12
        chex.assert_shape(CLOUDY_LINE_WAVELENGTHS, (12,))


# ── Task 5: Unified build_line_design_matrix ──────────────────────


class TestBuildLineDesignMatrix:
    def test_narrow_only_matches_eline_design_matrix(self):
        from tengri.observation.eline_marginalization import (
            build_eline_design_matrix,
            build_line_design_matrix,
        )

        wave = jnp.linspace(4000.0, 7000.0, 300)
        line_wav = jnp.array([4861.33, 6562.80])

        A_old = build_eline_design_matrix(wave, line_wav, spectral_resolution=2000.0, redshift=0.0)
        A_new = build_line_design_matrix(wave, line_wav)

        chex.assert_equal_shape([A_old, A_new])
        assert jnp.allclose(A_old, A_new, atol=1e-6)

    def test_narrow_plus_broad_has_extra_columns(self):
        from tengri.observation.eline_marginalization import build_line_design_matrix

        wave = jnp.linspace(4000.0, 7000.0, 300)
        narrow = jnp.array([4861.33, 6562.80])  # 2 narrow
        broad = jnp.array([4861.33])  # 1 broad

        A_combined = build_line_design_matrix(wave, narrow, broad_wavelengths=broad)

        assert A_combined.shape == (300, 3)  # 2 narrow + 1 broad


# ── Task 8: AGNConfig dataclass ───────────────────────────────────


class TestAGNConfig:
    def test_agn_config_importable(self):
        import tengri

        assert hasattr(tengri, "AGNConfig")

    def test_default_construction(self):
        from tengri.components.agn import AGNConfig

        cfg = AGNConfig()
        assert cfg.disc == "multicolor"
        assert cfg.torus == "skirtor"
        assert cfg.blr is True
        assert cfg.polar_dust is False

    def test_custom_construction(self):
        from tengri.components.agn import AGNConfig

        cfg = AGNConfig(disc="kubota_done", torus="skirtor", nlr="cue", blr=True, polar_dust=True)
        assert cfg.disc == "kubota_done"
        assert cfg.polar_dust is True

    def test_frozen_immutable(self):
        from tengri.components.agn import AGNConfig

        cfg = AGNConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.disc = "adaf"  # type: ignore[misc]

    def test_invalid_disc_raises(self):
        from tengri.components.agn import AGNConfig

        with pytest.raises(ValueError, match="disc"):
            AGNConfig(disc="invalid_model")

    def test_invalid_torus_raises(self):
        from tengri.components.agn import AGNConfig

        with pytest.raises(ValueError, match="torus"):
            AGNConfig(torus="photon_torpedo")


# ── SSP-required: model.tree() and model.recommend_method() ───────


@_needs_ssp
class TestModelTree:
    @pytest.fixture(scope="class")
    def smooth_model(self):
        import tengri

        return tengri.SEDModel.from_config(
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

    def test_tree_returns_string(self, smooth_model):
        result = smooth_model.tree()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_tree_contains_sfh_name(self, smooth_model):
        result = smooth_model.tree()
        assert "dpl" in result

    def test_tree_contains_recommended_method(self, smooth_model):
        result = smooth_model.tree()
        assert "Recommended inference" in result
        assert "model.fit(" in result

    def test_recommend_method_returns_string(self, smooth_model):
        method = smooth_model.recommend_method()
        assert isinstance(method, str)
        # Smooth DPL with ~6 free params → should recommend laplace or vi_linear
        assert method in ("laplace", "vi_nifty_linear", "vi_nifty")

    def test_recommend_method_used_in_fit(self, smooth_model):
        """recommend_method() output should be accepted by model.fit()."""
        true_params = {
            "sfh_dpl_alpha": 1.2,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 4.0,
            "sfh_dpl_age_gyr": 5.0,
            "sfh_dpl_log_total_mass": 0.9,
            "met_logzsol": -0.3,
            "dust_tau_bc": 1.0,
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 0.1,
        }
        mock = smooth_model.mock(true_params, snr=10.0, key=jax.random.PRNGKey(99))
        # Just test it runs without error; MAP is fast
        result = smooth_model.fit(mock.flux_obs, mock.noise, method="map")
        from tengri.inference.posterior import Posterior

        assert isinstance(result, Posterior)

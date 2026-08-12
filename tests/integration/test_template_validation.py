# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end validation of Astrodust, BOSA, and THEMIS dust emission templates.

Tests cover:
1. Template loading — correct keys, shapes, dtypes
2. Emission function creation via create_*_from_grid
3. SED evaluation — finite, non-negative, peaks in FIR
4. Energy conservation — scaled output proportional to L_absorbed
5. JIT compilation — JAX tracing succeeds
6. Approximate comparison to DL07 at similar parameters
"""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._jit_parity import assert_jit_matches_eager

jax.config.update("jax_enable_x64", True)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

ASTRODUST_PATH = DATA_DIR / "astrodust_templates.h5"
BOSA_PATH = DATA_DIR / "bosa_templates.h5"
THEMIS_PATH = DATA_DIR / "themis_templates.h5"

# Wavelength grid for SED evaluation: 1 to 1000 microns in Angstrom
WAVE_AA = jnp.linspace(1.0e4, 1.0e7, 1000)

# FIR peak expected between ~30 and ~300 microns (3e5 to 3e6 Angstrom)
FIR_PEAK_MIN_AA = 1.0e5  # 10 um
FIR_PEAK_MAX_AA = 5.0e6  # 500 um


# ── Astrodust (Hensley & Draine 2023) ─────────────────────────────


@pytest.fixture(scope="module")
def astrodust_data():
    """Load Astrodust templates, skip if file not found."""
    if not ASTRODUST_PATH.exists():
        pytest.skip("Astrodust templates not found; run download script first")
    from tengri.components.dust.emission import load_astrodust_templates

    return load_astrodust_templates(str(ASTRODUST_PATH))


@pytest.fixture(scope="module")
def astrodust_fn(astrodust_data):
    """Create Astrodust emission function."""
    from tengri.components.dust.emission import create_astrodust_from_grid

    return create_astrodust_from_grid(astrodust_data)


class TestAstrodustLoading:
    """Verify Astrodust template structure."""

    def test_keys(self, astrodust_data):
        expected = {"wavelength_aa", "umin_grid", "qpah_grid", "single_u", "powerlaw"}
        assert set(astrodust_data.keys()) == expected

    def test_wavelength_shape(self, astrodust_data):
        wav = astrodust_data["wavelength_aa"]
        assert wav.ndim == 1
        assert len(wav) > 100

    def test_wavelength_ascending(self, astrodust_data):
        wav = astrodust_data["wavelength_aa"]
        assert jnp.all(jnp.diff(wav) > 0)

    def test_grid_shapes_consistent(self, astrodust_data):
        nq = len(astrodust_data["qpah_grid"])
        nu = len(astrodust_data["umin_grid"])
        nw = len(astrodust_data["wavelength_aa"])
        assert astrodust_data["single_u"].shape == (nq, nu, nw)
        assert astrodust_data["powerlaw"].shape == (nq, nu, nw)

    def test_spectra_non_negative(self, astrodust_data):
        assert jnp.all(astrodust_data["single_u"] >= 0)
        assert jnp.all(astrodust_data["powerlaw"] >= 0)

    def test_spectra_finite(self, astrodust_data):
        chex.assert_tree_all_finite(astrodust_data["single_u"])
        chex.assert_tree_all_finite(astrodust_data["powerlaw"])


class TestAstrodustEmission:
    """Verify Astrodust emission function behavior."""

    def test_output_finite(self, astrodust_fn):
        sed = astrodust_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qpah=3.0
        )
        chex.assert_tree_all_finite(sed)

    def test_output_non_negative(self, astrodust_fn):
        sed = astrodust_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qpah=3.0
        )
        assert jnp.all(sed >= 0)

    def test_peak_in_fir(self, astrodust_fn):
        sed = astrodust_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=3.0
        )
        peak_wave = WAVE_AA[jnp.argmax(sed)]
        assert FIR_PEAK_MIN_AA < peak_wave < FIR_PEAK_MAX_AA, (
            f"Peak at {float(peak_wave):.0f} AA, expected {FIR_PEAK_MIN_AA}-{FIR_PEAK_MAX_AA}"
        )

    def test_energy_scaling(self, astrodust_fn):
        """Output should scale linearly with L_absorbed."""
        sed1 = astrodust_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qpah=3.0
        )
        sed2 = astrodust_fn(
            WAVE_AA, L_absorbed=2e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qpah=3.0
        )
        ratio = jnp.trapezoid(sed2, WAVE_AA) / jnp.trapezoid(sed1, WAVE_AA)
        assert abs(float(ratio) - 2.0) < 0.01

    def test_jit_compilation(self, astrodust_fn):
        sed_eager = astrodust_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qpah=3.0
        )
        sed_jit = assert_jit_matches_eager(
            lambda w: astrodust_fn(
                w, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qpah=3.0
            ),
            WAVE_AA,
        )
        assert jnp.allclose(sed_eager, sed_jit, rtol=1e-6)

    def test_parameter_sensitivity(self, astrodust_fn):
        """Different Umin should shift the peak."""
        sed_low_u = astrodust_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=0.5, dust_gamma_dl=0.01, dust_qpah=3.0
        )
        sed_high_u = astrodust_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=20.0, dust_gamma_dl=0.01, dust_qpah=3.0
        )
        peak_low = WAVE_AA[jnp.argmax(sed_low_u)]
        peak_high = WAVE_AA[jnp.argmax(sed_high_u)]
        # Higher U -> warmer dust -> shorter peak wavelength
        assert peak_high < peak_low


# ── BOSA (Boquien & Salim 2021) ───────────────────────────────────


@pytest.fixture(scope="module")
def bosa_data():
    """Load BOSA templates, skip if file not found."""
    if not BOSA_PATH.exists():
        pytest.skip("BOSA templates not found; run download script first")
    from tengri.components.dust.emission import load_bosa_templates

    return load_bosa_templates(str(BOSA_PATH))


@pytest.fixture(scope="module")
def bosa_fn(bosa_data):
    """Create BOSA emission function."""
    from tengri.components.dust.emission import create_bosa_from_grid

    return create_bosa_from_grid(bosa_data)


class TestBosaLoading:
    """Verify BOSA template structure."""

    def test_keys(self, bosa_data):
        expected = {"wavelength_aa", "log_ltir_grid", "log_ssfr_grid", "spectra"}
        assert set(bosa_data.keys()) == expected

    def test_wavelength_ascending(self, bosa_data):
        wav = bosa_data["wavelength_aa"]
        assert jnp.all(jnp.diff(wav) > 0)

    def test_grid_shapes_consistent(self, bosa_data):
        nl = len(bosa_data["log_ltir_grid"])
        ns = len(bosa_data["log_ssfr_grid"])
        nw = len(bosa_data["wavelength_aa"])
        assert bosa_data["spectra"].shape == (nl, ns, nw)

    def test_spectra_non_negative(self, bosa_data):
        assert jnp.all(bosa_data["spectra"] >= 0)

    def test_spectra_finite(self, bosa_data):
        chex.assert_tree_all_finite(bosa_data["spectra"])


class TestBosaEmission:
    """Verify BOSA emission function behavior."""

    def test_output_finite(self, bosa_fn):
        sed = bosa_fn(WAVE_AA, L_absorbed=1e10, dust_log_ssfr=-10.0)
        chex.assert_tree_all_finite(sed)

    def test_output_non_negative(self, bosa_fn):
        sed = bosa_fn(WAVE_AA, L_absorbed=1e10, dust_log_ssfr=-10.0)
        assert jnp.all(sed >= 0)

    def test_peak_in_fir(self, bosa_fn):
        sed = bosa_fn(WAVE_AA, L_absorbed=1e10, dust_log_ssfr=-10.0)
        peak_wave = WAVE_AA[jnp.argmax(sed)]
        assert FIR_PEAK_MIN_AA < peak_wave < FIR_PEAK_MAX_AA, (
            f"Peak at {float(peak_wave):.0f} AA, expected {FIR_PEAK_MIN_AA}-{FIR_PEAK_MAX_AA}"
        )

    def test_energy_scaling(self, bosa_fn):
        """Output should scale approximately linearly with L_absorbed.

        BOSA templates have a luminosity-dependent SED shape (the L_TIR axis
        shifts the radiation field and dust temperature distribution), so the
        ratio is not exactly 2.0. The 8% tolerance reflects this intentional
        physical non-linearity, not a normalization bug.
        """
        sed1 = bosa_fn(WAVE_AA, L_absorbed=1e10, dust_log_ssfr=-10.0)
        sed2 = bosa_fn(WAVE_AA, L_absorbed=2e10, dust_log_ssfr=-10.0)
        ratio = jnp.trapezoid(sed2, WAVE_AA) / jnp.trapezoid(sed1, WAVE_AA)
        assert abs(float(ratio) - 2.0) < 0.08

    def test_jit_compilation(self, bosa_fn):
        sed_eager = bosa_fn(WAVE_AA, L_absorbed=1e10, dust_log_ssfr=-10.0)
        sed_jit = assert_jit_matches_eager(
            lambda w: bosa_fn(w, L_absorbed=1e10, dust_log_ssfr=-10.0), WAVE_AA
        )
        assert jnp.allclose(sed_eager, sed_jit, rtol=1e-6)

    def test_ssfr_shifts_peak(self, bosa_fn):
        """Higher sSFR -> warmer dust -> shorter peak wavelength."""
        sed_low = bosa_fn(WAVE_AA, L_absorbed=1e10, dust_log_ssfr=-12.0)
        sed_high = bosa_fn(WAVE_AA, L_absorbed=1e10, dust_log_ssfr=-8.5)
        peak_low = WAVE_AA[jnp.argmax(sed_low)]
        peak_high = WAVE_AA[jnp.argmax(sed_high)]
        assert peak_high < peak_low


# ── THEMIS (Jones et al. 2017) ────────────────────────────────────


@pytest.fixture(scope="module")
def themis_data():
    """Load THEMIS templates, skip if file not found."""
    if not THEMIS_PATH.exists():
        pytest.skip("THEMIS templates not found; run download script first")
    from tengri.components.dust.emission import load_themis_templates

    return load_themis_templates(str(THEMIS_PATH))


@pytest.fixture(scope="module")
def themis_fn(themis_data):
    """Create THEMIS emission function."""
    from tengri.components.dust.emission import create_themis_from_grid

    return create_themis_from_grid(themis_data)


class TestThemisLoading:
    """Verify THEMIS template structure."""

    def test_keys(self, themis_data):
        # The core diffuse-ISM grid, plus the variable-alpha powerlaw axis added
        # by scripts/build_themis_alpha_axis.py (``alpha_grid`` +
        # ``powerlaw_alpha_ratio``, surfaced via the public ``has_alpha_grid``).
        required = {"wavelength_aa", "umin_grid", "qhac_grid", "single_u", "powerlaw"}
        assert required <= set(themis_data.keys())

    def test_wavelength_ascending(self, themis_data):
        wav = themis_data["wavelength_aa"]
        assert jnp.all(jnp.diff(wav) > 0)

    def test_grid_shapes_consistent(self, themis_data):
        nq = len(themis_data["qhac_grid"])
        nu = len(themis_data["umin_grid"])
        nw = len(themis_data["wavelength_aa"])
        assert themis_data["single_u"].shape == (nq, nu, nw)
        assert themis_data["powerlaw"].shape == (nq, nu, nw)

    def test_spectra_non_negative(self, themis_data):
        assert jnp.all(themis_data["single_u"] >= 0)
        assert jnp.all(themis_data["powerlaw"] >= 0)

    def test_spectra_finite(self, themis_data):
        chex.assert_tree_all_finite(themis_data["single_u"])
        chex.assert_tree_all_finite(themis_data["powerlaw"])


class TestThemisEmission:
    """Verify THEMIS emission function behavior."""

    def test_output_finite(self, themis_fn):
        sed = themis_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qhac=0.17
        )
        chex.assert_tree_all_finite(sed)

    def test_output_non_negative(self, themis_fn):
        sed = themis_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qhac=0.17
        )
        assert jnp.all(sed >= 0)

    def test_peak_in_fir(self, themis_fn):
        sed = themis_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=1.0, dust_gamma_dl=0.01, dust_qhac=0.17
        )
        peak_wave = WAVE_AA[jnp.argmax(sed)]
        assert FIR_PEAK_MIN_AA < peak_wave < FIR_PEAK_MAX_AA, (
            f"Peak at {float(peak_wave):.0f} AA, expected {FIR_PEAK_MIN_AA}-{FIR_PEAK_MAX_AA}"
        )

    def test_energy_scaling(self, themis_fn):
        """Output should scale linearly with L_absorbed."""
        sed1 = themis_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qhac=0.17
        )
        sed2 = themis_fn(
            WAVE_AA, L_absorbed=2e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qhac=0.17
        )
        ratio = jnp.trapezoid(sed2, WAVE_AA) / jnp.trapezoid(sed1, WAVE_AA)
        assert abs(float(ratio) - 2.0) < 0.01

    def test_jit_compilation(self, themis_fn):
        sed_eager = themis_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qhac=0.17
        )
        sed_jit = assert_jit_matches_eager(
            lambda w: themis_fn(
                w, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qhac=0.17
            ),
            WAVE_AA,
        )
        assert jnp.allclose(sed_eager, sed_jit, rtol=1e-6)

    def test_parameter_sensitivity(self, themis_fn):
        """Different Umin should shift the peak."""
        sed_low_u = themis_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=0.5, dust_gamma_dl=0.01, dust_qhac=0.17
        )
        sed_high_u = themis_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=20.0, dust_gamma_dl=0.01, dust_qhac=0.17
        )
        peak_low = WAVE_AA[jnp.argmax(sed_low_u)]
        peak_high = WAVE_AA[jnp.argmax(sed_high_u)]
        assert peak_high < peak_low


# ── Cross-model comparison ────────────────────────────────────────


class TestCrossModelComparison:
    """Compare Astrodust and THEMIS to DL07 at similar parameters.

    The synthetic templates should produce FIR SEDs in the same order
    of magnitude as DL07 (within ~2 dex), since they use similar physics.
    """

    @pytest.fixture(scope="class")
    def dl07_available(self):
        dl07_path = next(
            (
                DATA_DIR / f
                for f in ("dl07_templates_v2.h5", "dl07_templates.h5")
                if (DATA_DIR / f).exists()
            ),
            None,
        )
        if dl07_path is None:
            pytest.skip("DL07 templates not found")
        return True

    def test_astrodust_vs_dl07_same_ballpark(self, astrodust_fn, dl07_available):
        """Astrodust total flux within ~2 dex of DL07 at matched parameters."""
        from tengri.components.dust.emission import DUST_EMISSION_MODELS

        if "draine_li2007" not in DUST_EMISSION_MODELS:
            pytest.skip("DL07 model not registered")

        dl07_fn = DUST_EMISSION_MODELS["draine_li2007"]

        sed_astro = astrodust_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qpah=3.0
        )
        sed_dl07 = dl07_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qpah=3.0
        )

        flux_astro = float(jnp.trapezoid(sed_astro, WAVE_AA))
        flux_dl07 = float(jnp.trapezoid(sed_dl07, WAVE_AA))

        if flux_dl07 > 0 and flux_astro > 0:
            log_ratio = abs(np.log10(flux_astro / flux_dl07))
            assert log_ratio < 2.0, f"Astrodust/DL07 flux ratio: 10^{log_ratio:.1f}"

    def test_themis_vs_dl07_same_ballpark(self, themis_fn, dl07_available):
        """THEMIS total flux within ~2 dex of DL07 at matched parameters."""
        from tengri.components.dust.emission import DUST_EMISSION_MODELS

        if "draine_li2007" not in DUST_EMISSION_MODELS:
            pytest.skip("DL07 model not registered")

        dl07_fn = DUST_EMISSION_MODELS["draine_li2007"]

        # THEMIS qhac=0.17 ~ DL07 qpah=3%
        sed_themis = themis_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qhac=0.17
        )
        sed_dl07 = dl07_fn(
            WAVE_AA, L_absorbed=1e10, dust_umin=2.0, dust_gamma_dl=0.02, dust_qpah=3.0
        )

        flux_themis = float(jnp.trapezoid(sed_themis, WAVE_AA))
        flux_dl07 = float(jnp.trapezoid(sed_dl07, WAVE_AA))

        if flux_dl07 > 0 and flux_themis > 0:
            log_ratio = abs(np.log10(flux_themis / flux_dl07))
            assert log_ratio < 2.0, f"THEMIS/DL07 flux ratio: 10^{log_ratio:.1f}"

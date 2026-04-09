"""Unit tests for MAPPINGS III + V shock emission model."""

import jax
import jax.numpy as jnp
import pytest

from tengri.models.nebular.shock import (
    _FALLBACK_LINE_NAMES,
    _load_mappings_grids,
    shock_emission_sed,
    shock_line_ratios,
)


# Number of lines in the active backend (HDF5 when present, else fallback)
def _n_lines() -> int:
    grids = _load_mappings_grids()
    if grids is not None and "mappings5" in grids:
        return len(grids["mappings5"]["line_names"])
    return len(_FALLBACK_LINE_NAMES)


# ---------------------------------------------------------------------------
# shock_line_ratios — fallback path (no HDF5 file in test environment)
# ---------------------------------------------------------------------------


class TestShockLineRatios:
    """Tests for shock_line_ratios — uses HDF5 grid when present, else fallback."""

    def test_all_ratios_positive(self):
        """All line ratios should be strictly positive."""
        for v in [100.0, 300.0, 500.0, 1000.0]:
            ratios = shock_line_ratios(v)
            for name, val in ratios.items():
                assert float(val) > 0.0, f"{name} at v={v} is not positive"

    def test_hbeta_is_unity(self):
        """Hβ ratio should always be 1.0 (it is the reference line)."""
        ratios = shock_line_ratios(300.0)
        assert float(ratios["Hb_4861A"]) == pytest.approx(1.0)

    def test_nii_enhanced_relative_to_case_b(self):
        """[NII]/Hα should be elevated — a defining shock diagnostic.

        The real MAPPINGS V grid gives [NII]/Hα > 0.2 across shock velocities.
        The lower bound (0.2) is chosen conservatively: at 150 km/s the
        low-ionization post-shock gas drives relatively lower [NII]/Hα than at
        high velocities.  The diagnostic still distinguishes shocks from typical
        star-forming HII regions ([NII]/Hα ~ 0.05–0.15).
        """
        for v in [200.0, 300.0, 500.0]:
            ratios = shock_line_ratios(v)
            nii_ha = float(ratios["NII_6583A"]) / float(ratios["HA_6563A"])
            assert nii_ha > 0.2, f"[NII]/Hα={nii_ha:.2f} at v={v} not shock-like"

    def test_oiii_increases_with_velocity(self):
        """[OIII] should increase with shock velocity (higher ionization).

        The real MAPPINGS V grid (3MdBs, Allen2008 Solar, n=1) shows [OIII]/Hβ
        rising monotonically over 200–1000 km/s at solar abundance because the
        post-shock temperature increases with velocity, driving more O³⁺
        production.  The simple Allen+2008 Table 5 fallback showed a peak at
        ~300–500 km/s, but that was an approximation not present in the full
        MAPPINGS V calculation.
        """
        ratios_low = shock_line_ratios(200.0)
        ratios_high = shock_line_ratios(800.0)

        oiii_low = float(ratios_low["O3_5007A"])
        oiii_high = float(ratios_high["O3_5007A"])

        assert oiii_high > oiii_low, (
            f"[OIII] should increase from 200 to 800 km/s, got {oiii_low:.2f}→{oiii_high:.2f}"
        )

    def test_velocity_out_of_bounds_raises(self):
        """Velocities outside the grid range must raise ValueError immediately."""
        with pytest.raises(ValueError, match="shock_velocity"):
            shock_line_ratios(50.0)
        with pytest.raises(ValueError, match="shock_velocity"):
            shock_line_ratios(2000.0)

    def test_doublet_ratios(self):
        """Doublet ratios should be in the physically expected range.

        Atomic physics fixes [OIII] 5007/4959 ≈ 2.98 and [NII] 6583/6548 ≈ 2.94.
        MAPPINGS V computes these self-consistently from the radiative transfer,
        so values can differ from the textbook ratio by a few percent depending
        on density and ionization structure.  We accept ±10%.
        """
        ratios = shock_line_ratios(300.0)
        # [OIII] 5007/4959 — atomic physics: 2.98
        oiii_ratio = float(ratios["O3_5007A"]) / float(ratios["O3_4959A"])
        assert 2.5 <= oiii_ratio <= 3.3, (
            f"[OIII] 5007/4959={oiii_ratio:.3f} outside physically plausible range [2.5, 3.3]"
        )

        # [NII] 6583/6548 — atomic physics: 2.94
        nii_ratio = float(ratios["NII_6583A"]) / float(ratios["NII_6548A"])
        assert 2.5 <= nii_ratio <= 3.3, (
            f"[NII] 6583/6548={nii_ratio:.3f} outside physically plausible range [2.5, 3.3]"
        )

    def test_sii_total_positive(self):
        """Sum of [SII] doublet should be positive and physically plausible.

        The Allen+2008 fallback gave [SII] total ≈ 2.0 at 300 km/s; the real
        MAPPINGS V (3MdBs, Allen2008 Solar, n=1) gives ≈ 3.3 at the same
        conditions.  Both are physically reasonable — [SII]/Hβ ~ 1–5 is typical
        for shock-ionized gas.
        """
        ratios = shock_line_ratios(300.0)
        sii_total = float(ratios["SII_6716A"]) + float(ratios["SII_6731A"])
        assert 0.5 <= sii_total <= 10.0, (
            f"[SII] total={sii_total:.2f} outside physically plausible range [0.5, 10]"
        )


# ---------------------------------------------------------------------------
# shock_emission_sed
# ---------------------------------------------------------------------------


class TestShockEmissionSed:
    """Tests for shock_emission_sed."""

    @pytest.fixture()
    def wavelength(self):
        return jnp.linspace(3000.0, 8000.0, 5000)

    def test_output_shape(self, wavelength):
        """Output shape should match the input wavelength grid."""
        sed = shock_emission_sed(wavelength, 300.0, 1e6)
        assert sed.shape == wavelength.shape

    def test_zero_luminosity_gives_zero_sed(self, wavelength):
        """l_shock_halpha=0 must give a zero SED."""
        sed = shock_emission_sed(wavelength, 300.0, 0.0)
        assert jnp.allclose(sed, 0.0)

    def test_sed_non_negative(self, wavelength):
        """SED should be non-negative everywhere."""
        sed = shock_emission_sed(wavelength, 300.0, 1e6)
        assert jnp.all(sed >= 0.0)

    def test_sed_has_peaks_at_line_wavelengths(self, wavelength):
        """SED should have peaks near Hα 6563 Å."""
        sed = shock_emission_sed(wavelength, 300.0, 1e8, line_sigma_aa=2.0)
        ha_region = jnp.abs(wavelength - 6563.0) < 10.0
        assert jnp.max(sed[ha_region]) > jnp.median(sed[sed > 0]) * 10

    def test_delta_function_mode(self, wavelength):
        """Delta-function mode should produce at most N_lines non-zero pixels."""
        sed = shock_emission_sed(wavelength, 300.0, 1e6, line_sigma_aa=0.0)
        n_nonzero = int(jnp.sum(sed > 0))
        assert n_nonzero <= _n_lines()

    def test_gaussian_mode_broader(self, wavelength):
        """Gaussian mode should spread flux over more pixels than narrow mode."""
        sed_narrow = shock_emission_sed(wavelength, 300.0, 1e6, line_sigma_aa=1.0)
        sed_broad = shock_emission_sed(wavelength, 300.0, 1e6, line_sigma_aa=5.0)
        assert int(jnp.sum(sed_broad > 1e-30)) > int(jnp.sum(sed_narrow > 1e-30))

    def test_luminosity_scales_linearly(self, wavelength):
        """Doubling l_shock_halpha should double the SED."""
        sed1 = shock_emission_sed(wavelength, 300.0, 1e6)
        sed2 = shock_emission_sed(wavelength, 300.0, 2e6)
        nonzero = sed1 > 1e-30
        ratio = sed2[nonzero] / sed1[nonzero]
        assert jnp.allclose(ratio, 2.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestShockJIT:
    def test_line_ratios_jittable(self):
        @jax.jit
        def _get_halpha(v):
            ratios = shock_line_ratios(v)
            return ratios["HA_6563A"]

        val = _get_halpha(300.0)
        assert float(val) > 0.0

    def test_sed_jittable(self):
        wave = jnp.linspace(3000.0, 8000.0, 1000)

        @jax.jit
        def _compute(v, lum):
            return shock_emission_sed(wave, v, lum)

        sed = _compute(300.0, 1e6)
        assert sed.shape == wave.shape


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


class TestShockDifferentiable:
    def test_grad_wrt_velocity(self):
        wave = jnp.linspace(3000.0, 8000.0, 500)

        def _total_flux(v):
            return jnp.sum(shock_emission_sed(wave, v, 1e6, line_sigma_aa=2.0))

        g = jax.grad(_total_flux)(300.0)
        assert jnp.isfinite(g)
        assert float(g) != 0.0

    def test_grad_wrt_luminosity(self):
        wave = jnp.linspace(3000.0, 8000.0, 500)

        def _total_flux(lum):
            return jnp.sum(shock_emission_sed(wave, 300.0, lum, line_sigma_aa=2.0))

        g = jax.grad(_total_flux)(1e6)
        assert jnp.isfinite(g)
        assert float(g) != 0.0


# ---------------------------------------------------------------------------
# Integration with ParamSpec
# ---------------------------------------------------------------------------


class TestShockParamSpec:
    def test_shock_params_registered(self):
        from tengri.core.parameters import ParamSpec

        spec = ParamSpec(shock=True)
        params = spec.all_params
        assert "shock_frac" in params
        assert "shock_velocity" in params
        assert "shock_log_density" in params

    def test_shock_params_absent_by_default(self):
        from tengri.core.parameters import ParamSpec

        spec = ParamSpec()
        params = spec.all_params
        assert "shock_frac" not in params
        assert "shock_velocity" not in params

    def test_shock_frac_zero_default(self):
        from tengri.core.parameters import ParamSpec

        spec = ParamSpec(shock=True)
        dist = spec.get_distribution("shock_frac")
        assert dist.value == pytest.approx(0.0)

    def test_shock_velocity_bounds(self):
        from tengri.core.parameters import ParamSpec

        spec = ParamSpec(shock=True, shock_velocity=(100.0, 1000.0))
        assert "shock_velocity" in spec.free_params

        with pytest.raises(ValueError):
            ParamSpec(shock=True, shock_velocity=(50.0, 1000.0))

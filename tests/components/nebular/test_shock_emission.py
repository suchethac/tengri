# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for MAPPINGS III + V shock emission model."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri._data_setup import find_data
from tengri.components.nebular.shock import (
    _FALLBACK_LINE_NAMES,
    ShockBackend,
    _load_mappings_grids,
    compute_shock_sed,
    shock_line_ratios,
)
from tests._bounds import assert_non_negative

# parents[2] is tests/ from tests/components/nebular/, so this pointed at
# tests/data/ — which never exists, and the tests below never ran (#1431).
_H5_PATH = find_data("mappings_templates.h5")
h5_only = pytest.mark.skipif(
    _H5_PATH is None,
    reason="data/mappings_templates.h5 not found; build via download_mappings_templates.py",
)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# Number of lines in the active backend (HDF5 when present, else fallback)

pytestmark = pytest.mark.bounds


def _n_lines() -> int:
    grids = _load_mappings_grids()
    if grids is not None and "mappings5" in grids:
        return len(grids["mappings5"]["line_names"])
    return len(_FALLBACK_LINE_NAMES)


# ── shock_line_ratios — fallback path (no HDF5 file in test environment)


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

    @h5_only
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

    @h5_only
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

    def test_balmer_decrement_elevated_above_case_b(self):
        """Shock Hα/Hβ must exceed the Case B recombination ratio of 2.86.

        Osterbrock & Ferland 2006, Astrophysics of Gaseous Nebulae, §4.2:
        Case B recombination gives Hα/Hβ = 2.86 at T=10^4 K, n_e=100 cm^-3.
        Shocks produce ELEVATED Balmer ratios relative to Case B due to:
        (1) collisional excitation of Hα in post-shock warm gas,
        (2) partial Lya opacity trapping (Mathis 1986), and
        (3) large velocity widths broadening and merging line ratios differently.
        The MAPPINGS V grid yields Hα/Hβ typically 3.0–5.0 depending on velocity.
        We assert only the strict lower bound: ratio > 2.86 (must exceed Case B).
        """
        for v in [200.0, 300.0, 500.0]:
            ratios = shock_line_ratios(v)
            ha_hb = float(ratios["HA_6563A"]) / float(ratios["Hb_4861A"])
            assert ha_hb > 2.86, (
                f"Shock Hα/Hβ={ha_hb:.3f} at v={v} km/s — must exceed Case B ratio 2.86 "
                f"(Osterbrock & Ferland 2006 §4.2)"
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


# ── compute_shock_sed ────────────────────────────────────────────


class TestShockEmissionSed:
    """Tests for compute_shock_sed."""

    @pytest.fixture()
    def wavelength(self):
        return jnp.linspace(3000.0, 8000.0, 5000)

    def test_output_shape(self, wavelength):
        """Output shape should match the input wavelength grid."""
        sed = compute_shock_sed(wavelength, 300.0, 1e6)
        chex.assert_equal_shape([sed, wavelength])

    def test_zero_luminosity_gives_zero_sed(self, wavelength):
        """l_shock_halpha=0 must give a zero SED."""
        sed = compute_shock_sed(wavelength, 300.0, 0.0)
        assert jnp.allclose(sed, 0.0)

    def test_sed_non_negative(self, wavelength):
        """SED should be non-negative everywhere."""
        sed = compute_shock_sed(wavelength, 300.0, 1e6)
        assert_non_negative(sed, name="sed")

    def test_sed_has_peaks_at_line_wavelengths(self, wavelength):
        """SED should have peaks near Hα 6563 Å."""
        sed = compute_shock_sed(wavelength, 300.0, 1e8, line_sigma_aa=2.0)
        ha_region = jnp.abs(wavelength - 6563.0) < 10.0
        assert jnp.max(sed[ha_region]) > jnp.median(sed[sed > 0]) * 10

    def test_delta_function_mode(self, wavelength):
        """Delta-function mode should produce at most N_lines non-zero pixels."""
        sed = compute_shock_sed(wavelength, 300.0, 1e6, line_sigma_aa=0.0)
        n_nonzero = int(jnp.sum(sed > 0))
        assert n_nonzero <= _n_lines()

    def test_gaussian_mode_broader(self, wavelength):
        """Gaussian mode should spread flux over more pixels than narrow mode."""
        sed_narrow = compute_shock_sed(wavelength, 300.0, 1e6, line_sigma_aa=1.0)
        sed_broad = compute_shock_sed(wavelength, 300.0, 1e6, line_sigma_aa=5.0)
        assert int(jnp.sum(sed_broad > 1e-30)) > int(jnp.sum(sed_narrow > 1e-30))

    def test_luminosity_scales_linearly(self, wavelength):
        """Doubling l_shock_halpha should double the SED."""
        sed1 = compute_shock_sed(wavelength, 300.0, 1e6)
        sed2 = compute_shock_sed(wavelength, 300.0, 2e6)
        nonzero = sed1 > 1e-30
        ratio = sed2[nonzero] / sed1[nonzero]
        assert jnp.allclose(ratio, 2.0, rtol=1e-5)


# ── JIT compatibility ─────────────────────────────────────────────


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
            return compute_shock_sed(wave, v, lum)

        sed = _compute(300.0, 1e6)
        chex.assert_equal_shape([sed, wave])


# ── Differentiability ─────────────────────────────────────────────


class TestShockDifferentiable:
    @h5_only
    def test_grad_wrt_velocity(self):
        wave = jnp.linspace(3000.0, 8000.0, 500)

        def _total_flux(v):
            return jnp.sum(compute_shock_sed(wave, v, 1e6, line_sigma_aa=2.0))

        grad_jax = float(jax.grad(_total_flux)(300.0))
        grad_fd = fd_grad(_total_flux, 300.0)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=2e-1,
            err_msg="compute_shock_sed: FD check ∂/∂velocity",
        )
        assert grad_jax != 0.0

    def test_grad_wrt_luminosity(self):
        wave = jnp.linspace(3000.0, 8000.0, 500)

        def _total_flux(lum):
            return jnp.sum(compute_shock_sed(wave, 300.0, lum, line_sigma_aa=2.0))

        grad_jax = float(jax.grad(_total_flux)(1e6))
        grad_fd = fd_grad(_total_flux, 1e6)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=2e-1,
            err_msg="compute_shock_sed: FD check ∂/∂luminosity",
        )
        assert grad_jax != 0.0


# ── Integration with Parameters ────────────────────────────────────


class TestShockParamSpec:
    def test_shock_params_registered(self):
        from tengri.parameters.parameters import Parameters

        spec = Parameters(shock=True)
        params = spec.all_params
        assert "shock_frac" in params
        assert "shock_velocity" in params
        assert "shock_log_density" in params

    def test_shock_params_absent_by_default(self):
        from tengri.parameters.parameters import Parameters

        spec = Parameters()
        params = spec.all_params
        assert "shock_frac" not in params
        assert "shock_velocity" not in params

    def test_shock_frac_zero_default(self):
        from tengri.parameters.parameters import Parameters

        spec = Parameters(shock=True)
        dist = spec.get_distribution("shock_frac")
        assert dist.value == pytest.approx(0.0)

    def test_shock_velocity_bounds(self):
        from tengri.parameters.parameters import Parameters

        spec = Parameters(shock=True, shock_velocity=(100.0, 1000.0))
        assert "shock_velocity" in spec.free_params

        with pytest.raises(ValueError):
            Parameters(shock=True, shock_velocity=(50.0, 1000.0))


# ── ShockBackend protocol compliance and delegation ───────────────


class TestShockBackend:
    """Tests for the ShockBackend dataclass (Phase 6b)."""

    _WAVE = jnp.linspace(3000.0, 9000.0, 500)

    def test_protocol_attributes(self):
        """ShockBackend satisfies the NebularBackend Protocol attributes."""
        from tengri.components.nebular._protocol import NebularBackend

        b = ShockBackend()
        assert isinstance(b, NebularBackend)
        assert b.has_continuum is False
        assert b.has_free_params is True
        assert b.name == "shock"

    def test_default_abundance_and_component(self):
        b = ShockBackend()
        assert b.shock_abundance == "solar"
        assert b.shock_component == "combined"

    def test_custom_abundance(self):
        b = ShockBackend(shock_abundance="lmc", shock_component="shock")
        assert b.shock_abundance == "lmc"
        assert b.shock_component == "shock"

    def test_predict_nebular_sed_delegates_to_compute_shock_sed(self):
        """predict_nebular_sed and compute_shock_sed return identical arrays."""
        from tengri.components.nebular.shock import compute_shock_sed

        b = ShockBackend()
        v = 300.0
        l_ha = 1e40
        sed_backend = b.predict_nebular_sed(self._WAVE, v, l_ha)
        sed_direct = compute_shock_sed(self._WAVE, v, l_ha)
        assert jnp.allclose(sed_backend, sed_direct, atol=0.0, rtol=0.0)

    def test_predict_nebular_sed_returns_finite_sed(self):
        b = ShockBackend()
        sed = b.predict_nebular_sed(self._WAVE, 300.0, 1e40)
        chex.assert_shape(sed, (self._WAVE.shape[0],))
        chex.assert_tree_all_finite(sed)
        assert_non_negative(sed, name="sed")

    def test_predict_nebular_sed_extra_kwargs_ignored(self):
        """**_kwargs allows protocol-uniform call sites to pass extra args."""
        b = ShockBackend()
        sed = b.predict_nebular_sed(self._WAVE, 300.0, 1e40, ssp_weights=None, log_z=0.0)
        chex.assert_tree_all_finite(sed)

    def test_has_continuum_and_has_free_params_not_settable_via_init(self):
        """has_continuum and has_free_params are fixed; init=False in dataclass."""
        # These fields are fixed by the class — init=False means they cannot
        # be overridden at construction.
        b = ShockBackend(shock_abundance="smc")
        assert b.has_continuum is False
        assert b.has_free_params is True

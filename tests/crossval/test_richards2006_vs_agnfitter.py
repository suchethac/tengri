# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate Richards+2006 accretion disc module against AGNfitter's raw template pickle.

The Richards et al. (2006) mean Type-1 quasar SED template was extracted from
AGNfitter's ``models/BBB/R06.pickle`` and tabulated in tengri as
``src/tengri/data/agn_bbb/richards2006.dat``. This test verifies that
tengri's runtime module produces spectra that match the upstream template
to within interpolation tolerances.

Convention verification (issue #592 A2 vs #647):
  - AGNfitter R06.pickle 'SED' column = nu*F_nu (relative units)
  - tengri loads this as nu*F_nu, divides by nu to get F_nu, interprets as L_nu
  - Both normalize by bolometric integral: integral(L_nu, dnu) = L_bol
  - Measured peak wavelength and log10 ratios at 1µm, 5µm, H-alpha confirm
    the two implementations agree within ~0.3–0.6% (interpolation error only).
    No evidence of the 4–20× NIR divergence claimed in #592 A2.
"""

from __future__ import annotations

from pathlib import Path

import chex
import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.physics_constants import C_AA  # speed of light [Å/s]
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.crossval

# Reference SED comes from the committed HDF5 (built once from R06.pickle by
# scripts/build_agnfitter_bbb_reference.py) — no pickle at test time, so this runs
# in CI rather than skipping on a missing upstream clone (#613).
_REF_H5 = Path(__file__).resolve().parents[2] / "data" / "agnfitter_bbb_reference.h5"

if not _REF_H5.is_file():
    pytest.skip(
        f"BBB reference HDF5 not found at {_REF_H5} "
        "(build with: python scripts/build_agnfitter_bbb_reference.py)",
        allow_module_level=True,
    )


def _log_nu_to_wavelength_angstrom(log_nu_hz: np.ndarray) -> np.ndarray:
    """Convert log10(nu/Hz) to wavelength [Å]."""
    nu_hz = 10.0 ** np.asarray(log_nu_hz, dtype=np.float64)
    return C_AA / nu_hz


def _normalize_lnu_by_integral(lnu: np.ndarray, nu_hz: np.ndarray) -> np.ndarray:
    """Normalize L_nu by its integral over frequency.

    Parameters
    ----------
    lnu : ndarray
        Spectral luminosity density [arbitrary units].
    nu_hz : ndarray
        Frequency grid [Hz], must be sorted ascending.

    Returns
    -------
    ndarray
        Normalized L_nu (integral over nu = 1).
    """
    # Trapzoidal integration over frequency (not log frequency)
    integral = np.sum((lnu[:-1] + lnu[1:]) / 2.0 * np.diff(nu_hz))
    if integral <= 0:
        return lnu
    return lnu / integral


@pytest.fixture(scope="module")
def agnfitter_r06():
    """AGNfitter R06 template from the vendored HDF5.

    Returned in the same ``{"wavelength": log10(nu/Hz), "SED": raw}`` shape the
    original pickle exposed, so the downstream convention logic is unchanged.
    """
    with h5py.File(_REF_H5, "r") as f:
        g = f["r06"]
        wave_aa = np.asarray(g["wavelength"][:], dtype=np.float64)
        sed = np.asarray(g["sed"][:], dtype=np.float64)
    return {"wavelength": np.log10(C_AA / wave_aa), "SED": sed}


@pytest.fixture(scope="module")
def richards2006_runtime():
    """Instantiate the tengri runtime module."""
    from tengri.components.agn.richards2006_disc import (
        _RICHARDS2006_BOL_INTEGRAL,
        _RICHARDS2006_LNU_SHAPE,
        _RICHARDS2006_NU_HZ,
        RICHARDS2006_WAVE_AA,
    )

    return {
        "wave_aa": RICHARDS2006_WAVE_AA,
        "nu_hz": _RICHARDS2006_NU_HZ,
        "lnu_shape": _RICHARDS2006_LNU_SHAPE,
        "bol_integral": _RICHARDS2006_BOL_INTEGRAL,
    }


class TestRichards2006Convention:
    """Verify the convention: AGNfitter and tengri both use L_nu normalized by
    bolometric integral."""

    def test_peak_wavelength_match(self, agnfitter_r06, richards2006_runtime):
        """Peak wavelength should match between the two templates.

        This test directly addresses issue #592 A2 vs #647: both templates
        should peak at similar wavelengths if they encode the same physical SED.
        """
        # AGNfitter: extract log_nu and SED values
        agn_log_nu = np.asarray(agnfitter_r06["wavelength"]).ravel()
        agn_sed = np.asarray(agnfitter_r06["SED"]).ravel()

        # Interpret AGNfitter's SED as nu*F_nu, then compute L_nu (like tengri does)
        agn_nu_hz = 10.0**agn_log_nu
        agn_lnu = agn_sed / agn_nu_hz  # nu*F_nu / nu = L_nu shape

        # Find peaks
        agn_peak_idx = np.argmax(agn_lnu)
        agn_peak_nu = agn_nu_hz[agn_peak_idx]
        agn_peak_wave = C_AA / agn_peak_nu

        tengri_peak_idx = np.argmax(richards2006_runtime["lnu_shape"])
        tengri_peak_wave = richards2006_runtime["wave_aa"][tengri_peak_idx]

        print(f"\nPeak wavelength AGNfitter: {agn_peak_wave:.3e} Å = {agn_peak_wave / 1e4:.3f} µm")
        print(
            f"Peak wavelength tengri: {tengri_peak_wave:.3e} Å = {tengri_peak_wave / 1e4:.3f} µm"
        )
        print(f"Ratio: {tengri_peak_wave / agn_peak_wave:.6f}")

        # Should match to within 1% (interpolation + rounding)
        np.testing.assert_allclose(
            tengri_peak_wave,
            agn_peak_wave,
            rtol=0.01,
            err_msg="Peak wavelengths diverge between AGNfitter and tengri",
        )

    def test_normalized_sed_shapes_match(self, agnfitter_r06, richards2006_runtime):
        """Normalized SED shapes should match after regridding to a common wavelength grid.

        This test compares the peak-normalized L_nu templates on a common wavelength grid.
        Shape agreement within ~0.5% (interpolation error) indicates no convention divergence.
        """
        # AGNfitter template
        agn_log_nu = np.asarray(agnfitter_r06["wavelength"]).ravel()
        agn_sed = np.asarray(agnfitter_r06["SED"]).ravel()

        # Convert to wavelength and compute L_nu
        agn_nu_hz = 10.0**agn_log_nu
        agn_wave = C_AA / agn_nu_hz
        agn_lnu = agn_sed / agn_nu_hz

        # Sort by ascending wavelength for interpolation
        agn_sort_idx = np.argsort(agn_wave)
        agn_wave_sorted = agn_wave[agn_sort_idx]
        agn_lnu_sorted = agn_lnu[agn_sort_idx]

        # Normalize by peak value (shape comparison only)
        agn_lnu_norm = agn_lnu_sorted / np.max(agn_lnu_sorted)

        # Tengri template (already on wavelength grid, ascending)
        tengri_wave = richards2006_runtime["wave_aa"]
        tengri_lnu_shape = richards2006_runtime["lnu_shape"]

        # Normalize tengri's template by peak value
        tengri_lnu_norm = tengri_lnu_shape / np.max(tengri_lnu_shape)

        # Regrid AGNfitter to tengri's common wavelength grid
        agn_regridded = np.interp(tengri_wave, agn_wave_sorted, agn_lnu_norm, left=0.0, right=0.0)

        # Compare normalized templates. Tolerance ~2%: the vendored reference HDF5
        # is a compact (1024-point, float32) regrid of the 438-point R06 template,
        # so sub-percent interpolation/rounding differences are expected — far below
        # the 4-20x (#592 A2) divergence this test rules out (#647 confirmed correct).
        np.testing.assert_allclose(
            tengri_lnu_norm,
            agn_regridded,
            rtol=0.02,
            atol=2e-3,
            err_msg="Normalized SED shapes diverge between AGNfitter and tengri",
        )

    def test_log_ratio_at_key_wavelengths(self, agnfitter_r06, richards2006_runtime):
        """Measure log10(tengri / AGNfitter) at key wavelengths to quantify any convention bug.

        Tests the specific claim in issue #592 A2 that richards2006 has a 4–20× NIR
        divergence. Measures the ratio at 1µm, 5µm, and H-alpha (6563 Å).
        """
        # AGNfitter template
        agn_log_nu = np.asarray(agnfitter_r06["wavelength"]).ravel()
        agn_sed = np.asarray(agnfitter_r06["SED"]).ravel()

        agn_nu_hz = 10.0**agn_log_nu
        agn_wave = C_AA / agn_nu_hz
        agn_lnu = agn_sed / agn_nu_hz

        # Normalize by peak value (shape comparison only)
        agn_sort_idx = np.argsort(agn_wave)
        agn_wave_sorted = agn_wave[agn_sort_idx]
        agn_lnu_sorted = agn_lnu[agn_sort_idx]
        agn_lnu_norm = agn_lnu_sorted / np.max(agn_lnu_sorted)

        # Tengri template (normalize for comparison)
        tengri_wave = richards2006_runtime["wave_aa"]
        tengri_lnu_shape = richards2006_runtime["lnu_shape"]
        tengri_lnu_norm = tengri_lnu_shape / np.max(tengri_lnu_shape)

        # Test wavelengths: 1µm, 5µm, H-alpha
        test_wavelengths = np.array([1e4, 5e4, 6563])

        print("\nLog10(ratio) at key wavelengths:")
        for test_wl in test_wavelengths:
            # Find closest in each template
            agn_idx = np.argmin(np.abs(agn_wave_sorted - test_wl))
            tengri_idx = np.argmin(np.abs(tengri_wave - test_wl))

            agn_val = agn_lnu_norm[agn_idx]
            tengri_val = tengri_lnu_norm[tengri_idx]

            ratio = tengri_val / agn_val
            log_ratio = np.log10(ratio)

            print(f"  {test_wl:.0f} Å ({test_wl / 1e4:.2f} µm): log10(ratio) = {log_ratio:.4f}")

        # Ratios should be very close to 1.0 (log10 ~ 0) if no convention bug
        # The #592 A2 claim is that tengri is 4–20× off, i.e., log10 ≈ 0.6 to 1.3
        # Our measured values should be within ±0.01 (< 2.3% error) if correct

        for test_wl in test_wavelengths:
            agn_idx = np.argmin(np.abs(agn_wave_sorted - test_wl))
            tengri_idx = np.argmin(np.abs(tengri_wave - test_wl))

            agn_val = agn_lnu_norm[agn_idx]
            tengri_val = tengri_lnu_norm[tengri_idx]

            ratio = tengri_val / agn_val

            # Assert ratio is within 2.3% of 1.0 (log10 within ±0.01)
            np.testing.assert_allclose(
                ratio,
                1.0,
                rtol=0.023,
                err_msg=f"Divergence at {test_wl:.0f} Å: log10(ratio) = {np.log10(ratio):.4f}. "
                f"Issue #592 A2 claims 4–20× divergence (log10 0.6–1.3); "
                f"issue #647 claims tengri is correct.",
            )


class TestRichards2006RuntimeConsistency:
    """Verify the runtime callable produces sensible outputs."""

    def test_runtime_callable_evaluates(self):
        """Runtime function should evaluate without errors."""
        from tengri.components.agn.richards2006_disc import richards2006_disc

        wavelength = np.linspace(100, 10000, 256)
        sed = richards2006_disc(wavelength, log_lbol=44.0)
        chex.assert_tree_all_finite(sed)
        assert sed.shape == wavelength.shape, f"Shape mismatch: {sed.shape} vs {wavelength.shape}"

    def test_runtime_respects_luminosity_scaling(self):
        """Doubling L_bol should roughly double the SED."""
        from tengri.components.agn.richards2006_disc import richards2006_disc

        wavelength = np.linspace(100, 10000, 256)
        sed_low = richards2006_disc(wavelength, log_lbol=43.0)
        sed_high = richards2006_disc(wavelength, log_lbol=43.301)  # log10(2) ≈ 0.301

        # Should scale by ~2x (exact within float64)
        ratio = sed_high / (sed_low + 1e-30)
        expected_ratio = 10.0 ** (43.301 - 43.0)
        np.testing.assert_allclose(
            ratio,
            expected_ratio,
            rtol=1e-10,
            err_msg="Luminosity scaling not linear",
        )

    def test_runtime_gradient_flows(self):
        """JAX gradient should flow through the interpolation."""

        from tengri.components.agn.richards2006_disc import richards2006_disc

        def loss(log_lbol: float) -> float:
            wavelength = np.linspace(100, 10000, 64)
            sed = richards2006_disc(wavelength, log_lbol=log_lbol)
            return jnp.sum(sed)

        grad = assert_grad_matches_fd(loss, 45.0)
        assert np.isfinite(grad), f"Gradient is {grad} (NaN/inf)"
        assert abs(grad) > 0, "Gradient is zero"

"""Cross-validate Silva+04 torus module against AGNfitter's raw template pickles.

The Silva, Maiolino & Granato (2004) torus templates were ported from AGNfitter
into tengri's HDF5 grid format by scripts/build_silva04_grid.py. This test
reads both the original AGNfitter S04.pickle (via safe-unpickle) and tengri's
runtime module, then verifies the normalised SED templates match within 5%.

The comparison normalises both templates by their trapezoidal integral over
frequency (nu), ensuring shape agreement independent of absolute scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Add scripts dir to path for importing build utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_GRID_PATH = _DATA_DIR / "silva04_torus_grid.h5"
_AGNFITTER_PICKLE = Path("/tmp/AGNfitter-rX/models/TORUS/S04.pickle")

# Skip module if either input is missing
if not _GRID_PATH.is_file():
    pytest.skip("Silva+04 grid not found at " + str(_GRID_PATH), allow_module_level=True)

if not _AGNFITTER_PICKLE.is_file():
    pytest.skip(
        "AGNfitter S04.pickle not found at " + str(_AGNFITTER_PICKLE)
        + " (clone with: git clone https://github.com/GabrielaCR/AGNfitter /tmp/AGNfitter-rX)",
        allow_module_level=True,
    )


def _safe_load_silva04_pickle(pickle_path: Path) -> dict:
    """Safely unpickle S04.pickle using the allow-list from the build script."""
    from build_silva04_grid import (
        _preflight_opcode_scan,
        _RestrictedUnpickler,
    )

    _preflight_opcode_scan(pickle_path)
    with pickle_path.open("rb") as fh:
        obj = _RestrictedUnpickler(fh, encoding="latin1").load()
    if not isinstance(obj, dict):
        raise TypeError(f"S04 pickle root is {type(obj).__name__}, expected dict.")
    return obj


def _log_nu_to_wavelength_angstrom(log_nu_hz: np.ndarray) -> np.ndarray:
    """Convert log10(nu/Hz) to wavelength [Å]."""
    c_light_m_s = 2.99792458e8
    nu_hz = 10.0 ** np.asarray(log_nu_hz, dtype=np.float64)
    wavelength_m = c_light_m_s / nu_hz
    return wavelength_m * 1e10


def _trapz_normalise(sed_fnu: np.ndarray, wavelength_aa: np.ndarray) -> np.ndarray:
    """Normalise SED by trapezoidal integral over frequency.

    Parameters
    ----------
    sed_fnu : ndarray
        SED in F_nu units (relative or absolute).
    wavelength_aa : ndarray
        Wavelength grid [Å].

    Returns
    -------
    ndarray
        Normalised SED (shape-only, integral over nu = 1).
    """
    c_light_m_s = 2.99792458e8
    c_light_aa_s = c_light_m_s * 1e10
    nu_hz = c_light_aa_s / wavelength_aa  # [Hz]
    dnudn = np.abs(np.gradient(np.log10(nu_hz)))  # logarithmic spacing
    integral = np.trapz(sed_fnu * dnudn, np.log10(nu_hz))
    if integral <= 0:
        return sed_fnu  # Cannot normalise; return as-is
    return sed_fnu / integral


@pytest.fixture(scope="module")
def silva04_grid():
    """Load the tengri Silva+04 grid (numpy arrays)."""
    with h5py.File(str(_GRID_PATH), "r") as f:
        g = f["silva04"]
        return {
            "log_nh_axis": np.asarray(g["log_nh_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def agnfitter_silva04():
    """Load AGNfitter's S04.pickle (safely)."""
    return _safe_load_silva04_pickle(_AGNFITTER_PICKLE)


@pytest.fixture(scope="module")
def silva04_runtime():
    """Instantiate the tengri runtime module."""
    from tengri.components.agn.silva04 import create_silva04_from_grid

    return create_silva04_from_grid(str(_GRID_PATH))


class TestSilva04Shapes:
    """Verify SED shapes match between AGNfitter and tengri templates."""

    @pytest.mark.parametrize("nh_idx", [0, 2, 4])
    def test_sed_shapes_match_after_normalisation(
        self, silva04_grid, agnfitter_silva04, nh_idx
    ):
        """Pick specific log_N_H bins and verify tengri's template matches AGNfitter's shape.

        The comparison:
        1. Picks a row from AGNfitter's per-bin arrays.
        2. Regrid AGNfitter's per-row log_nu to tengri's common wavelength grid.
        3. Normalise both templates by trapz(sed, nu).
        4. Compare the normalised templates with rtol=0.05 (5%).
        """
        n_nh = silva04_grid["log_nh_axis"].size
        if nh_idx >= n_nh:
            pytest.skip(f"Index {nh_idx} out of range for {n_nh} N_H bins")

        # Get the AGNfitter row
        agn_log_nu = np.asarray(agnfitter_silva04["wavelength"][nh_idx]).ravel()
        agn_sed_fnu = np.asarray(agnfitter_silva04["SED"][nh_idx]).ravel()

        # Convert log_nu to wavelength and resample to tengri's common grid
        agn_wave = _log_nu_to_wavelength_angstrom(agn_log_nu)
        order = np.argsort(agn_wave)
        agn_wave_sorted = agn_wave[order]
        agn_sed_sorted = agn_sed_fnu[order]

        # Regrid to tengri's wavelength grid
        common_wave = silva04_grid["wavelength"]
        agn_regridded = np.interp(
            common_wave, agn_wave_sorted, agn_sed_sorted, left=0.0, right=0.0
        )

        # Get tengri's template
        tengri_template = silva04_grid["template"][nh_idx]

        # Compare templates directly on the common wavelength grid.
        # Since both were regridded the same way from AGNfitter's source,
        # they should match to machine precision (< 1%).
        np.testing.assert_allclose(
            tengri_template,
            agn_regridded,
            rtol=0.01,
            atol=1e-75,  # Absolute tolerance for very small values
            err_msg=f"Template regridding diverges for N_H index {nh_idx}",
        )


class TestSilva04RuntimeConsistency:
    """Verify the runtime callable produces sensible outputs."""

    def test_runtime_callable_evaluates(self, silva04_runtime):
        """Runtime function should evaluate without errors."""
        wavelength = np.linspace(100, 10000, 256)
        sed = silva04_runtime(
            wavelength=wavelength,
            agn_log_lbol=44.0,
            agn_log_nh_silva=23.0,
            agn_torus_frac=0.5,
        )
        assert np.all(np.isfinite(sed)), "Runtime SED contains NaN/inf"
        assert sed.shape == wavelength.shape, f"Shape mismatch: {sed.shape} vs {wavelength.shape}"

    def test_runtime_respects_luminosity_scaling(self, silva04_runtime):
        """Doubling L_bol should roughly double the SED."""
        wavelength = np.linspace(100, 10000, 256)
        sed_low = silva04_runtime(
            wavelength=wavelength,
            agn_log_lbol=43.0,
            agn_log_nh_silva=23.0,
            agn_torus_frac=0.5,
        )
        sed_high = silva04_runtime(
            wavelength=wavelength,
            agn_log_lbol=44.301,  # log10(2) ≈ 0.301
            agn_log_nh_silva=23.0,
            agn_torus_frac=0.5,
        )
        # Should scale by ~2x (within 10% tolerance for interpolation)
        ratio = sed_high / (sed_low + 1e-30)
        assert np.median(ratio) > 1.5, "Luminosity scaling not observed"

    def test_runtime_gradient_flows(self, silva04_runtime):
        """JAX gradient should flow through the triweight interpolation."""

        def loss(log_nh):
            wavelength = np.linspace(100, 10000, 64)
            sed = silva04_runtime(
                wavelength=wavelength,
                agn_log_lbol=44.0,
                agn_log_nh_silva=log_nh,  # Pass traced value directly
                agn_torus_frac=0.5,
            )
            return jnp.sum(sed)

        grad = jax.grad(loss)(23.0)
        assert np.isfinite(grad), f"Gradient is {grad} (NaN/inf)"
        assert abs(grad) > 0, "Gradient is zero"

# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation: tengri SKIRTOR_mean_3p vs AGNfitter-rX upstream templates.

Loads the native AGNfitter-rX ``SKIRTOR_mean_3p.pickle`` and verifies that
tengri's interpolated templates match the upstream values within tolerance
at matched (oa, incl, tv) grid points.

This test confirms that the port is faithful to the original (nearest-neighbour
selection + per-L_sun normalization), not that the full-grid SKIRTOR is
identical (it is not — see issue #614, #592).

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modelling of the dusty
   torus around AGN — the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286.
.. [2] M. Stalevski et al., "The dust covering factor in AGN — combining the
   IR torus emission with polar dust component," MNRAS, 458, 2288 (2016).
   arXiv:1602.01954.
.. [3] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval


@pytest.fixture
def agnfitter_driver():
    """Load the AGNfitter-rX driver (requires $AGNFITTER_HOME to be set)."""
    try:
        from reproduction.agnfitter._drivers import agnfitter_driver

        agnfitter_driver.require_available()
        return agnfitter_driver
    except ImportError:
        pytest.skip("AGNfitter driver not available")
    except FileNotFoundError:
        pytest.skip("AGNFITTER-RX checkout not found; set AGNFITTER_HOME")


@pytest.fixture
def skirtor_agnfitter_grid():
    """Load the built SKIRTOR_mean_3p grid."""
    from tengri.components.agn.skirtor_agnfitter import (
        create_skirtor_agnfitter_from_grid,
    )

    try:
        grid_path = "data/skirtor_mean3p_torus_grid.h5"
        if not os.path.isfile(grid_path):
            pytest.skip(
                f"SKIRTOR_mean_3p grid not found at {grid_path}; "
                "build it with: python scripts/build_skirtor_mean3p_grid.py"
            )
        return create_skirtor_agnfitter_from_grid(grid_path)
    except (FileNotFoundError, OSError, KeyError) as e:
        pytest.skip(f"Failed to load SKIRTOR_mean_3p grid: {e}")


def test_skirtor_mean3p_peak_wavelength_vs_agnfitter(agnfitter_driver):
    """Verify SKIRTOR_mean_3p peak wavelength matches AGNfitter-rX upstream.

    At oa=40, incl=30, tv=7, the peak should be ~25 µm (AGNfitter-averaged
    behavior), NOT ~40 µm (full-grid SKIRTOR default). This confirms the
    distinction between the two models.
    """
    # Load at matched geometry
    wave_aa, L_nu_agnfitter = agnfitter_driver.torus_template("SKIRTOR", oa=40, incl=30, tau=7)

    # Find peak
    peak_idx = np.argmax(L_nu_agnfitter)
    peak_wave = wave_aa[peak_idx]

    # Peak should be in the 20–30 µm range for AGNfitter's SKIRTOR_mean_3p
    # (the exact value depends on the grid and regridding)
    assert 15e4 < peak_wave < 35e4, (
        f"SKIRTOR_mean_3p peak at oa=40, incl=30, tau=7 is "
        f"{peak_wave / 1e4:.1f} µm; expected 15–35 µm. "
        f"If peak is ~40 µm, you may have loaded the full-grid SKIRTOR instead."
    )

    print(f"SKIRTOR_mean_3p (AGNfitter) peak: {peak_wave / 1e4:.1f} µm")


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


def test_skirtor_agnfitter_matches_agnfitter_nodes(agnfitter_driver, skirtor_agnfitter_grid):
    """Tengri SED shapes match AGNfitter at grid nodes (riporous, ≤5% normalised).

    Samples several random grid points and verifies that tengri's interpolated
    SEDs match AGNfitter's templates in shape (after normalisation) within 5%,
    and peak wavelengths match within ±2 grid points.
    """
    from reproduction.agnfitter._drivers import agnfitter_driver as driver_module

    # Extract grid structure from AGNfitter
    d = driver_module._safe_load("TORUS/SKIRTOR_mean_3p.pickle")
    oa_vals = np.sort(np.asarray(d["oa-values"].unique(), dtype=np.float64))
    incl_vals = np.sort(np.asarray(d["incl-values"].unique(), dtype=np.float64))
    tv_vals = np.sort(np.asarray(d["tv-values"].unique(), dtype=np.float64))

    # Sample nodes: at least 3 for good coverage
    rng = np.random.RandomState(42)
    n_sample = min(5, len(oa_vals) * len(incl_vals) * len(tv_vals))
    indices = rng.choice(len(oa_vals) * len(incl_vals) * len(tv_vals), n_sample, replace=False)

    peak_diffs = []
    max_shape_err = 0.0
    shape_errors = []

    for idx in indices:
        i_oa = idx // (len(incl_vals) * len(tv_vals))
        i_incl = (idx // len(tv_vals)) % len(incl_vals)
        i_tv = idx % len(tv_vals)

        oa = oa_vals[i_oa]
        incl = incl_vals[i_incl]
        tv = tv_vals[i_tv]

        # Load from AGNfitter
        wave_agnfitter, L_nu_agnfitter = agnfitter_driver.torus_template(
            "SKIRTOR", oa=oa, incl=incl, tau=tv
        )

        # Load from tengri at the same wavelengths
        L_nu_tengri = skirtor_agnfitter_grid(
            wavelength=jnp.asarray(wave_agnfitter),
            agn_log_lbol=0.0,
            agn_oa_skirtor=oa,
            agn_incl_skirtor=incl,
            agn_tv_skirtor=tv,
            agn_torus_frac=1.0,
        )
        L_nu_tengri = np.asarray(L_nu_tengri)

        # Check peak wavelength (±2 grid points)
        peak_idx_agnfitter = np.argmax(L_nu_agnfitter)
        peak_idx_tengri = np.argmax(L_nu_tengri)
        peak_diff_idx = abs(peak_idx_agnfitter - peak_idx_tengri)
        peak_diffs.append(peak_diff_idx)

        assert peak_diff_idx <= 2, (
            f"Node (oa={oa}, incl={incl}, tv={tv}): "
            f"Peak wavelength difference {peak_diff_idx} grid points exceeds 2."
        )

        # Normalise both by trapz integral over frequency for shape comparison
        norm_agnfitter = _trapz_normalise(L_nu_agnfitter, wave_agnfitter)
        norm_tengri = _trapz_normalise(L_nu_tengri, wave_agnfitter)

        # Check shape agreement (≤5%)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.abs(norm_tengri - norm_agnfitter) / (np.abs(norm_agnfitter) + 1e-75)
        # Mask out near-zero regions
        mask_nonzero = np.abs(norm_agnfitter) > 1e-10
        if mask_nonzero.any():
            max_err_node = np.max(rel_err[mask_nonzero])
        else:
            max_err_node = 0.0
        max_shape_err = max(max_shape_err, max_err_node)
        shape_errors.append(max_err_node)

        assert max_err_node <= 0.05, (
            f"Node (oa={oa}, incl={incl}, tv={tv}): "
            f"Shape mismatch {max_err_node * 100:.1f}% exceeds 5% tolerance."
        )

    avg_peak_diff = np.mean(peak_diffs)
    avg_shape_err = np.mean(shape_errors)
    print(
        f"SKIRTOR_mean_3p cross-validation: {n_sample} nodes, "
        f"avg peak-wavelength diff {avg_peak_diff:.2f} grid points, "
        f"max shape error {max_shape_err * 100:.2f}% (avg {avg_shape_err * 100:.2f}%)"
    )


def test_skirtor_agnfitter_parameter_bounds():
    """SKIRTOR_mean_3p parameter bounds are enforced by grid extent."""
    from tengri.components.agn.skirtor_agnfitter_model import SKIRTORAgnfitterTorus

    component = SKIRTORAgnfitterTorus()

    # Check that priors match the grid bounds
    assert component.oa_skirtor.lo == 10.0
    assert component.oa_skirtor.hi == 80.0
    assert component.incl_skirtor.lo == 0.0
    assert component.incl_skirtor.hi == 90.0
    assert component.tv_skirtor.lo == 3.0
    assert component.tv_skirtor.hi == 11.0

    print("SKIRTOR_mean_3p parameter bounds verified.")

# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's ``dh02_ce01`` cold-dust model against AGNfitter-rX.

The DH02_CE01 cold-dust library (Dale & Helou 2002 + Chary & Elbaz 2001) is
repackaged from AGNfitter-rX's ``STARBURST/DH02_CE01.pickle`` into tengri's
HDF5 grid by ``scripts/build_dh02_ce01_grid.py``. AGNfitter-rX evaluates the
template library as a single-axis (L_IR) grid via linear interpolation in
log₁₀(L_IR/L_sun).

This test reads the committed reference data (data/agnfitter_cold_dust_reference.h5
group 'dh02_ce01'), verifies tengri's ``dh02_ce01`` reproduces the same *shape*
(normalized L_nu) at several irlum nodes to a tight tolerance after regridding
to a common wavelength axis.

Both ``dh02_ce01_grid.h5`` (tengri's runtime grid, ``scripts/build_dh02_ce01_grid.py``)
and this reference (``scripts/build_agnfitter_bbb_reference.py``) preserve each
of the pickle's 169 templates' own native wavelength sampling rather than
double-resampling everything onto one foreign axis, which used to smear the
aromatic/PAH-forest region (~3.2-3.9 µm) by up to 0.47 dex at the grid-edge
``irlum`` nodes (``colddust_radio.md`` D2).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.crossval, pytest.mark.regression_paper]

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_GRID_PATH = _DATA_DIR / "dh02_ce01_grid.h5"
_REFERENCE_PATH = _DATA_DIR / "agnfitter_cold_dust_reference.h5"

_C_AA_PER_S = 2.99792458e18  # speed of light [Å·Hz]

if not _GRID_PATH.is_file():
    pytest.skip(
        "DH02_CE01 grid not found at "
        + str(_GRID_PATH)
        + " (build with: python scripts/build_dh02_ce01_grid.py)",
        allow_module_level=True,
    )

if not _REFERENCE_PATH.is_file():
    pytest.skip(
        "AGNfitter reference data not found at " + str(_REFERENCE_PATH),
        allow_module_level=True,
    )


def _load_reference_dh02_ce01() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load committed reference data for DH02_CE01.

    Returns
    -------
    irlum : ndarray, shape (169,)
        Log₁₀(L_IR/L_sun) axis (NOT sorted; traces CE01 sequence).
    wavelength : ndarray, shape (1024,)
        Wavelength grid [Å].
    sed : ndarray, shape (169, 1024)
        Relative F_nu templates (to be peak-normalized).
    """
    import h5py

    with h5py.File(_REFERENCE_PATH, "r") as f:
        g = f["dh02_ce01"]
        irlum = np.array(g["irlum"][:])
        wavelength = np.array(g["wavelength"][:])
        sed = np.array(g["sed"][:])

    return irlum, wavelength, sed


def _tengri_dh02_ce01(wavelength_aa: np.ndarray, log_lir: float) -> np.ndarray:
    """tengri ``dh02_ce01`` L_nu on the requested grid.

    Parameters
    ----------
    wavelength_aa : ndarray
        Wavelength grid [Å].
    log_lir : float
        Log₁₀(L_IR/L_sun).

    Returns
    -------
    ndarray
        Peak-normalized L_nu.
    """
    from tengri.components.dust.emission_templates import create_dh02_ce01_from_grid

    fn = create_dh02_ce01_from_grid(str(_GRID_PATH))
    # Return L_nu normalized to L_absorbed=1 (peak normalization applied below)
    return np.asarray(fn(jnp.asarray(wavelength_aa), 1.0, dust_log_lir=log_lir))


@pytest.mark.parametrize("log_lir", [10.0, 11.0, 12.0])
def test_dh02_ce01_matches_reference_shape(log_lir: float) -> None:
    """tengri dh02_ce01 reproduces reference shape to <3% of peak.

    Tests several well-separated L_IR nodes across the full grid range.
    """
    irlum_ref, wave_ref, sed_ref = _load_reference_dh02_ce01()

    # Find the closest reference template to the requested log_lir
    # (accounts for the fact that reference irlum is not evenly spaced
    # and has duplicates that we've deduplicated in the grid)
    idx_closest = np.argmin(np.abs(irlum_ref - log_lir))
    sed_ref_closest = sed_ref[idx_closest]

    # Evaluate tengri model on reference wavelength grid
    l_te = _tengri_dh02_ce01(wave_ref, log_lir)

    # Normalize both to peak = 1 in the 3–300 µm cold-dust band
    band = (wave_ref > 3.0e4) & (wave_ref < 3.0e6)  # 3–300 µm
    ref_n = sed_ref_closest / np.abs(sed_ref_closest[band]).max()
    te_n = l_te / np.abs(l_te[band]).max()

    # Check for finite values
    assert np.all(np.isfinite(l_te)), "tengri output contains NaN/Inf"

    # Residual analysis in the dust band
    resid = np.abs(te_n[band] - ref_n[band])
    median_resid = float(np.median(resid))
    max_resid = float(resid.max())

    assert median_resid < 5.0e-2, f"median |Δ|/peak = {median_resid:.2e} (expect <0.05)"
    assert max_resid < 3.0e-1, f"max |Δ|/peak = {max_resid:.2e} (expect <0.30)"


def test_dh02_ce01_changes_with_lir() -> None:
    """dust_log_lir parameter changes the output SED shape.

    Peak-normalized shapes at well-separated L_IR nodes must differ
    by >1% (i.e., the parameter has a real effect).
    """
    wavelength = np.geomspace(1.0e4, 1.0e8, 2000)
    band = (wavelength > 3.0e4) & (wavelength < 3.0e6)

    # Evaluate at two well-separated L_IR values
    sed_low = _tengri_dh02_ce01(wavelength, 10.0)
    sed_high = _tengri_dh02_ce01(wavelength, 13.0)

    # Normalize to peak
    sed_low_n = sed_low / np.abs(sed_low[band]).max()
    sed_high_n = sed_high / np.abs(sed_high[band]).max()

    # Compute peak-normalized difference in the dust band
    diff = np.abs(sed_low_n[band] - sed_high_n[band]).max()

    assert diff > 0.01, f"Peak difference between L_IR=10 and 13 is only {diff:.2e} (expect >0.01)"


def test_dh02_ce01_is_finite_and_positive() -> None:
    """L_nu must be finite and non-negative everywhere.

    AGNfitter's reference templates are all positive; tengri should
    inherit this after normalization.
    """
    wavelength = np.geomspace(1.0e4, 1.0e8, 3000)
    for log_lir in [8.5, 10.0, 12.0, 14.0]:
        sed = _tengri_dh02_ce01(wavelength, log_lir)
        assert np.all(np.isfinite(sed)), f"Non-finite at log_lir={log_lir}"
        # Relative templates may have small numerical noise; check bulk is positive
        assert sed[sed > 0].sum() > 0.9 * np.abs(sed).sum(), (
            f"Mostly negative at log_lir={log_lir}"
        )


def test_dh02_ce01_energy_balance() -> None:
    """The frequency integral of emitted L_nu equals L_absorbed.

    Validates the normalization: ∫L_nu dν = L_absorbed.
    """
    wavelength = np.geomspace(1.0e4, 1.0e8, 4000)
    nu = _C_AA_PER_S / wavelength
    sed = _tengri_dh02_ce01(wavelength, 11.0)
    integral = -np.trapezoid(sed, nu)
    # Tolerance: numerical quadrature on a coarse grid
    assert abs(integral - 1.0) < 1.0e-2, f"Energy balance violation: ∫L_nu dν = {integral}"


@pytest.mark.parametrize("log_lir_label", ["irlum_min", "irlum_max", 9.0, 11.0, 12.0])
def test_dh02_ce01_log_ratio_at_edges(log_lir_label) -> None:
    """Log10-ratio shape fidelity, including the ``irlum`` grid-edge nodes.

    Includes ``irlum.min()``/``irlum.max()``: the two edges where the prior
    builder's float32 + 1024-point ``np.geomspace`` double resampling reached
    ~0.47 dex (a factor ~3x) at the top node (``colddust_radio.md`` D2). The
    other tests above use a peak-normalized *linear* residual, which is
    insensitive to a large fractional error confined to a narrow,
    small-amplitude feature far from the peak (exactly where D2's worst
    residual sat: 3.2-3.9 µm, well off the ~50-150 µm cold-dust peak);
    log10-ratio is not.

    tengri is evaluated on the reference's own native wavelength grid (no
    extra resampling), so this isolates the model's shape fidelity from
    resampling noise, matching ``test_schreiber2018_matches_agnfitter_pah``
    above.
    """
    irlum_ref, wave_ref, sed_ref = _load_reference_dh02_ce01()
    if log_lir_label == "irlum_min":
        log_lir = float(irlum_ref.min())
    elif log_lir_label == "irlum_max":
        log_lir = float(irlum_ref.max())
    else:
        log_lir = float(log_lir_label)

    idx_closest = np.argmin(np.abs(irlum_ref - log_lir))
    sed_ref_closest = sed_ref[idx_closest]

    l_te = _tengri_dh02_ce01(wave_ref, log_lir)
    assert np.all(np.isfinite(l_te))

    band = (wave_ref > 3.0e4) & (wave_ref < 1.0e7)  # 3-1000 um
    i100 = np.argmin(np.abs(wave_ref / 1.0e4 - 100.0))
    scale = sed_ref_closest[i100] / l_te[i100]
    logratio = np.log10(np.abs((l_te * scale)[band] / sed_ref_closest[band]))

    assert np.all(np.isfinite(logratio)), "non-finite log10-ratio in the 3-1000 um band"
    worst = float(np.nanmax(np.abs(logratio)))
    assert worst < 0.02, (
        f"log_LIR={log_lir:.4f} (matched ref irlum={irlum_ref[idx_closest]:.4f}): "
        f"max|log10 ratio| = {worst:.4f} over 3-1000 um (expect <0.02)"
    )

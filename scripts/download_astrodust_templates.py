#!/usr/bin/env python3
"""Download Astrodust+PAH templates (Hensley & Draine 2023).

The Astrodust model replaces the classical DL07 grain model with an
improved dust composition (astrodust + PAH) that better reproduces the
observed polarization and emission properties of interstellar dust.

The template grid is parameterized by (qPAH, Umin) with both single-U
and power-law U components, identical to the DL07 mixing formula.

Source: Harvard Dataverse
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BMPNML

When the real data cannot be downloaded (network unavailable, API changed),
this script generates a physically motivated synthetic template grid for
testing and development.  The synthetic templates use modified blackbody
continua with Drude-profile PAH features — suitable for pipeline testing
but NOT for science.

Usage:
    python scripts/download_astrodust_templates.py
    python scripts/download_astrodust_templates.py --output data/astrodust_templates.npz
    python scripts/download_astrodust_templates.py --synthetic
    python scripts/download_astrodust_templates.py --dry-run

References:
    Hensley, B. S. & Draine, B. T. 2023, ApJ, 948, 55.
"""

import argparse
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------
# Dataverse dataset URL (Hensley & Draine 2023)
# -----------------------------------------------------------------------
DATAVERSE_DOI = "doi:10.7910/DVN/BMPNML"
DATAVERSE_API = (
    "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId={doi}"
)

# Grid parameters matching Astrodust publication
QPAH_GRID = np.array([0.47, 1.12, 2.50, 3.19, 3.90, 4.58])
UMIN_GRID = np.array([0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.0, 15.0, 20.0, 25.0])

# PAH feature central wavelengths (microns) and widths (Drude profiles)
PAH_CENTERS_UM = np.array([3.3, 6.2, 7.7, 8.6, 11.3, 12.7, 17.0])
PAH_WIDTHS_UM = np.array([0.03, 0.10, 0.40, 0.14, 0.10, 0.20, 0.30])
PAH_STRENGTHS = np.array([0.02, 0.15, 0.50, 0.08, 0.25, 0.07, 0.04])


def _planck_lam(wave_um: np.ndarray, T: float) -> np.ndarray:
    """Planck function B_lambda(T) in arbitrary units.

    Parameters
    ----------
    wave_um : array
        Wavelength in microns.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    array
        B_lambda in consistent but arbitrary units.
    """
    h = 6.626e-27  # erg s
    c = 3.0e10  # cm/s
    k = 1.381e-16  # erg/K
    wave_cm = wave_um * 1.0e-4
    x = h * c / (wave_cm * k * T)
    x = np.clip(x, 0.0, 500.0)
    return 2.0 * h * c**2 / wave_cm**5 / (np.exp(x) - 1.0 + 1e-300)


def _mbb(wave_um: np.ndarray, T: float, beta: float = 2.0) -> np.ndarray:
    """Modified blackbody in L_lambda convention.

    Parameters
    ----------
    wave_um : array
        Wavelength in microns.
    T : float
        Dust temperature in Kelvin.
    beta : float
        Emissivity spectral index.

    Returns
    -------
    array
        MBB spectrum (unnormalized).
    """
    kappa = (100.0 / wave_um) ** beta  # opacity ~ lambda^{-beta}
    return kappa * _planck_lam(wave_um, T)


def _drude_profiles(wave_um: np.ndarray, qpah: float) -> np.ndarray:
    """Sum of Drude profiles for PAH emission features.

    Parameters
    ----------
    wave_um : array
        Wavelength in microns.
    qpah : float
        PAH mass fraction (percent).  Scales overall PAH amplitude.

    Returns
    -------
    array
        PAH feature spectrum (unnormalized, same units as MBB).
    """
    pah = np.zeros_like(wave_um)
    for center, width, strength in zip(PAH_CENTERS_UM, PAH_WIDTHS_UM, PAH_STRENGTHS):
        gamma_sq = (width / center) ** 2
        x_sq = ((wave_um - center) / center) ** 2
        pah += strength * gamma_sq / (x_sq + gamma_sq)
    # Scale by qpah (normalized so qpah=3% gives order-unity contribution)
    return pah * (qpah / 3.0)


def _generate_single_spectrum(wave_um: np.ndarray, qpah: float, umin: float) -> np.ndarray:
    """Generate a single synthetic Astrodust spectrum.

    Uses the Mathis ISRF scaling: T_dust ~ 18 * U^(1/6) K for large
    grains (with small-grain stochastic heating approximated by a warmer
    component).

    Parameters
    ----------
    wave_um : array
        Wavelength grid in microns.
    qpah : float
        PAH mass fraction (percent).
    umin : float
        Radiation field intensity in Mathis ISRF units.

    Returns
    -------
    array
        Synthetic L_lambda spectrum (unnormalized).
    """
    # Large-grain equilibrium temperature
    T_big = 18.0 * umin ** (1.0 / 6.0)
    # Small-grain stochastic component (warmer)
    T_small = 60.0 * umin ** (1.0 / 5.0)

    spectrum = _mbb(wave_um, T_big, beta=2.0)
    spectrum += 0.05 * _mbb(wave_um, T_small, beta=1.5)
    spectrum += _drude_profiles(wave_um, qpah) * _planck_lam(wave_um, 800.0) * 0.001

    return np.maximum(spectrum, 0.0)


def _generate_pdr_spectrum(wave_um: np.ndarray, qpah: float, umin: float) -> np.ndarray:
    """Generate synthetic PDR (power-law U) spectrum.

    The PDR component has contributions from a range of U values from
    Umin to Umax ~ 10^6, making it warmer with stronger MIR features.

    Parameters
    ----------
    wave_um : array
        Wavelength grid in microns.
    qpah : float
        PAH mass fraction (percent).
    umin : float
        Minimum radiation field intensity.

    Returns
    -------
    array
        Synthetic PDR spectrum (unnormalized).
    """
    # PDR is dominated by high-U regions: effectively warmer
    T_pdr = 40.0 * umin ** (1.0 / 6.0)
    T_hot = 150.0

    spectrum = _mbb(wave_um, T_pdr, beta=1.8)
    spectrum += 0.15 * _mbb(wave_um, T_hot, beta=1.2)
    # Enhanced PAH features in PDR (stronger UV field excites PAH more)
    spectrum += 2.0 * _drude_profiles(wave_um, qpah) * _planck_lam(wave_um, 800.0) * 0.001

    return np.maximum(spectrum, 0.0)


def generate_synthetic_templates(output_path: str) -> None:
    """Generate synthetic Astrodust template grid for testing.

    Creates a physically motivated but approximate template grid using
    modified blackbody continua with Drude-profile PAH features.
    NOT suitable for science — use for pipeline testing only.

    Parameters
    ----------
    output_path : str
        Path for the output NPZ file.
    """
    wave_um = np.logspace(np.log10(0.1), np.log10(10000.0), 500)

    n_qpah = len(QPAH_GRID)
    n_umin = len(UMIN_GRID)
    n_wave = len(wave_um)

    spectra_single = np.zeros((n_qpah, n_umin, n_wave))
    spectra_pdr = np.zeros((n_qpah, n_umin, n_wave))

    for i, qpah in enumerate(QPAH_GRID):
        for j, umin in enumerate(UMIN_GRID):
            spectra_single[i, j] = _generate_single_spectrum(wave_um, qpah, umin)
            spectra_pdr[i, j] = _generate_pdr_spectrum(wave_um, qpah, umin)

    np.savez(
        output_path,
        wavelength_um=wave_um,
        qpah_grid=QPAH_GRID,
        umin_grid=UMIN_GRID,
        spectra_single=spectra_single,
        spectra_pdr=spectra_pdr,
    )

    print(f"  Saved synthetic Astrodust templates to: {output_path}")
    print(f"  wavelength_um: {wave_um.shape}")
    print(f"  qpah_grid: {QPAH_GRID.shape} — {QPAH_GRID}")
    print(f"  umin_grid: {UMIN_GRID.shape} — {UMIN_GRID}")
    print(f"  spectra_single: {spectra_single.shape}")
    print(f"  spectra_pdr: {spectra_pdr.shape}")
    print()
    print("  WARNING: Synthetic templates — NOT for science.")


def try_download_real(output_path: str) -> bool:
    """Attempt to download real Astrodust templates from Harvard Dataverse.

    Parameters
    ----------
    output_path : str
        Output path (not used directly — real download would need
        conversion from FITS).

    Returns
    -------
    bool
        True if download succeeded, False otherwise.
    """
    url = DATAVERSE_API.format(doi=DATAVERSE_DOI)
    print(f"  Attempting download from: {url}")
    try:
        req = urllib.request.Request(url, method="HEAD")
        resp = urllib.request.urlopen(req, timeout=10)
        if resp.status == 200:
            print("  Dataverse reachable, but FITS->NPZ conversion not yet automated.")
            print("  Falling back to synthetic templates.")
            return False
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  Download failed: {exc}")
        return False
    return False


def download_and_convert(
    output_path: str, dry_run: bool = False, force_synthetic: bool = False
) -> None:
    """Download or generate Astrodust templates.

    Parameters
    ----------
    output_path : str
        Path for the output NPZ file.
    dry_run : bool
        If True, only print what would be done.
    force_synthetic : bool
        If True, skip download attempt and generate synthetic templates.
    """
    print("Astrodust+PAH template downloader (Hensley & Draine 2023)")
    print(f"  Dataset DOI: {DATAVERSE_DOI}")
    print(f"  Output: {output_path}")
    print()

    if dry_run:
        print("[dry-run] Would download Astrodust templates from Harvard Dataverse.")
        print("[dry-run] Convert raw FITS to NPZ with keys:")
        print("  - wavelength_um: (n_wave,)")
        print("  - qpah_grid: (n_qpah,)")
        print("  - umin_grid: (n_umin,)")
        print("  - spectra_single: (n_qpah, n_umin, n_wave)")
        print("  - spectra_pdr: (n_qpah, n_umin, n_wave)")
        return

    if not force_synthetic and try_download_real(output_path):
        return

    print("  Generating synthetic Astrodust templates for testing...")
    print()
    generate_synthetic_templates(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Astrodust+PAH templates (Hensley & Draine 2023)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output NPZ path (default: data/astrodust_templates.npz)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without downloading",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip download attempt and generate synthetic templates",
    )
    args = parser.parse_args()

    if args.output is None:
        repo_root = Path(__file__).resolve().parent.parent
        args.output = str(repo_root / "data" / "astrodust_templates.npz")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    download_and_convert(args.output, dry_run=args.dry_run, force_synthetic=args.synthetic)


if __name__ == "__main__":
    main()

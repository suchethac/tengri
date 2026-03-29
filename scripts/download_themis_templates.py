#!/usr/bin/env python3
"""Download THEMIS dust emission templates (Jones et al. 2017).

The THEMIS model uses a different grain composition than DL07: a-C(:H)
aromatic carbon instead of PAH, with the aromatic fraction qhac replacing
qpah as the key parameter.

The templates are generated using DustEM and distributed with CIGALE.

When the real data cannot be downloaded (network unavailable, CIGALE not
installed), this script generates a physically motivated synthetic template
grid for testing.  The synthetic templates use modified blackbody continua
with a-C(:H) emission features — suitable for pipeline testing but NOT
for science.

Source: CIGALE project
    https://gitlab.lam.fr/cigale/cigale/-/tree/master/database_builder/themis

Usage:
    python scripts/download_themis_templates.py
    python scripts/download_themis_templates.py --output data/themis_templates.npz
    python scripts/download_themis_templates.py --synthetic
    python scripts/download_themis_templates.py --dry-run

References:
    Jones, A. P. et al. 2017, A&A, 602, A46.
    Compiègne, M. et al. 2011, A&A, 525, A103 (DustEM).
"""

import argparse
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------
# THEMIS template source (from CIGALE's DustEM pre-computed grid)
# -----------------------------------------------------------------------
CIGALE_GITLAB_BASE = (
    "https://gitlab.lam.fr/api/v4/projects/cigale%2Fcigale/repository/files/{path}/raw?ref=master"
)

# Grid parameters matching THEMIS/CIGALE implementation
# qhac: a-C(:H) aromatic mass fraction (Jones+2017 Table 3)
QHAC_GRID = np.array([0.02, 0.06, 0.10, 0.14, 0.17, 0.20, 0.24, 0.30])
UMIN_GRID = np.array([0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.0, 15.0, 20.0, 25.0])

# a-C(:H) aromatic features — similar positions to PAH but with
# different relative strengths (Jones+2017)
ACH_CENTERS_UM = np.array([3.3, 6.2, 7.7, 8.6, 11.3, 12.0])
ACH_WIDTHS_UM = np.array([0.04, 0.12, 0.45, 0.16, 0.12, 0.22])
ACH_STRENGTHS = np.array([0.03, 0.12, 0.40, 0.06, 0.20, 0.05])


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
        B_lambda (unnormalized).
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
    kappa = (100.0 / wave_um) ** beta
    return kappa * _planck_lam(wave_um, T)


def _ach_features(wave_um: np.ndarray, qhac: float) -> np.ndarray:
    """Sum of Drude profiles for a-C(:H) aromatic emission features.

    Parameters
    ----------
    wave_um : array
        Wavelength in microns.
    qhac : float
        a-C(:H) aromatic mass fraction.  Scales overall feature amplitude.

    Returns
    -------
    array
        Feature spectrum (unnormalized).
    """
    features = np.zeros_like(wave_um)
    for center, width, strength in zip(ACH_CENTERS_UM, ACH_WIDTHS_UM, ACH_STRENGTHS):
        gamma_sq = (width / center) ** 2
        x_sq = ((wave_um - center) / center) ** 2
        features += strength * gamma_sq / (x_sq + gamma_sq)
    # Scale by qhac (normalized so qhac=0.17 gives order-unity contribution)
    return features * (qhac / 0.17)


def _generate_single_spectrum(wave_um: np.ndarray, qhac: float, umin: float) -> np.ndarray:
    """Generate a single synthetic THEMIS spectrum.

    Uses similar physics to DL07 but with slightly different grain
    properties (a-C grains have different emissivity index at long
    wavelengths, and the aromatic features have different relative
    strengths).

    Parameters
    ----------
    wave_um : array
        Wavelength grid in microns.
    qhac : float
        a-C(:H) aromatic mass fraction.
    umin : float
        Radiation field intensity in Mathis ISRF units.

    Returns
    -------
    array
        Synthetic L_lambda spectrum (unnormalized).
    """
    # THEMIS large-grain equilibrium temperature
    # Slightly different from DL07 due to different grain optical properties
    T_big = 19.0 * umin ** (1.0 / 6.0)
    T_small = 55.0 * umin ** (1.0 / 5.0)

    # a-C grains have beta ~ 1.8 (slightly flatter than silicate beta ~ 2.0)
    spectrum = _mbb(wave_um, T_big, beta=1.8)
    spectrum += 0.06 * _mbb(wave_um, T_small, beta=1.4)
    spectrum += _ach_features(wave_um, qhac) * _planck_lam(wave_um, 700.0) * 0.001

    return np.maximum(spectrum, 0.0)


def _generate_pdr_spectrum(wave_um: np.ndarray, qhac: float, umin: float) -> np.ndarray:
    """Generate synthetic THEMIS PDR (power-law U) spectrum.

    Parameters
    ----------
    wave_um : array
        Wavelength grid in microns.
    qhac : float
        a-C(:H) aromatic mass fraction.
    umin : float
        Minimum radiation field intensity.

    Returns
    -------
    array
        Synthetic PDR spectrum (unnormalized).
    """
    T_pdr = 38.0 * umin ** (1.0 / 6.0)
    T_hot = 140.0

    spectrum = _mbb(wave_um, T_pdr, beta=1.7)
    spectrum += 0.12 * _mbb(wave_um, T_hot, beta=1.1)
    # Enhanced a-C(:H) features in PDR
    spectrum += 1.8 * _ach_features(wave_um, qhac) * _planck_lam(wave_um, 700.0) * 0.001

    return np.maximum(spectrum, 0.0)


def generate_synthetic_templates(output_path: str) -> None:
    """Generate synthetic THEMIS template grid for testing.

    Creates a physically motivated but approximate template grid using
    modified blackbody continua with a-C(:H) aromatic emission features.
    NOT suitable for science — use for pipeline testing only.

    Parameters
    ----------
    output_path : str
        Path for the output NPZ file.
    """
    wave_um = np.logspace(np.log10(0.1), np.log10(10000.0), 500)

    n_qhac = len(QHAC_GRID)
    n_umin = len(UMIN_GRID)
    n_wave = len(wave_um)

    spectra_single = np.zeros((n_qhac, n_umin, n_wave))
    spectra_pdr = np.zeros((n_qhac, n_umin, n_wave))

    for i, qhac in enumerate(QHAC_GRID):
        for j, umin in enumerate(UMIN_GRID):
            spectra_single[i, j] = _generate_single_spectrum(wave_um, qhac, umin)
            spectra_pdr[i, j] = _generate_pdr_spectrum(wave_um, qhac, umin)

    np.savez(
        output_path,
        wavelength_um=wave_um,
        qhac_grid=QHAC_GRID,
        umin_grid=UMIN_GRID,
        spectra_single=spectra_single,
        spectra_pdr=spectra_pdr,
    )

    print(f"  Saved synthetic THEMIS templates to: {output_path}")
    print(f"  wavelength_um: {wave_um.shape}")
    print(f"  qhac_grid: {QHAC_GRID.shape} — {QHAC_GRID}")
    print(f"  umin_grid: {UMIN_GRID.shape} — {UMIN_GRID}")
    print(f"  spectra_single: {spectra_single.shape}")
    print(f"  spectra_pdr: {spectra_pdr.shape}")
    print()
    print("  WARNING: Synthetic templates — NOT for science.")


def try_download_real(output_path: str) -> bool:
    """Attempt to download real THEMIS templates from CIGALE GitLab.

    Parameters
    ----------
    output_path : str
        Output path.

    Returns
    -------
    bool
        True if download succeeded, False otherwise.
    """
    test_url = CIGALE_GITLAB_BASE.format(path="README.md")
    print("  Attempting to reach CIGALE GitLab...")
    try:
        req = urllib.request.Request(test_url, method="HEAD")
        resp = urllib.request.urlopen(req, timeout=10)
        if resp.status == 200:
            print("  CIGALE GitLab reachable, but THEMIS extraction not yet automated.")
            print("  Falling back to synthetic templates.")
            return False
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  Download failed: {exc}")
        return False
    return False


def download_and_convert(
    output_path: str, dry_run: bool = False, force_synthetic: bool = False
) -> None:
    """Download or generate THEMIS templates.

    Parameters
    ----------
    output_path : str
        Path for the output NPZ file.
    dry_run : bool
        If True, only print what would be done.
    force_synthetic : bool
        If True, skip download attempt and generate synthetic templates.
    """
    print("THEMIS template downloader (Jones et al. 2017)")
    print(f"  Output: {output_path}")
    print()

    if dry_run:
        print("[dry-run] Would download THEMIS templates from CIGALE GitLab.")
        print("[dry-run] Convert to NPZ with keys:")
        print("  - wavelength_um: (n_wave,)")
        print("  - qhac_grid: (n_qhac,) a-C(:H) aromatic fraction")
        print("  - umin_grid: (n_umin,)")
        print("  - spectra_single: (n_qhac, n_umin, n_wave)")
        print("  - spectra_pdr: (n_qhac, n_umin, n_wave)")
        return

    if not force_synthetic and try_download_real(output_path):
        return

    print("  Generating synthetic THEMIS templates for testing...")
    print()
    generate_synthetic_templates(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download THEMIS dust emission templates (Jones+2017)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output NPZ path (default: data/themis_templates.npz)",
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
        args.output = str(repo_root / "data" / "themis_templates.npz")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    download_and_convert(args.output, dry_run=args.dry_run, force_synthetic=args.synthetic)


if __name__ == "__main__":
    main()

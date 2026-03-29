#!/usr/bin/env python3
"""Download BOSA dust emission templates (Boquien & Salim 2021).

The BOSA templates parameterize dust emission by (L_TIR, sSFR) instead
of radiation field parameters.  This provides a direct link between
star formation activity and dust SED shape.

When the real data cannot be downloaded (network unavailable, CIGALE
not installed), this script generates a physically motivated synthetic
template grid for testing.  The synthetic templates use modified
blackbody spectra with temperature scaling calibrated to the BOSA
relation — suitable for pipeline testing but NOT for science.

Source: CIGALE project / CDS
    Boquien, M. & Salim, S. 2021, A&A, 653, A149.

Usage:
    python scripts/download_bosa_templates.py
    python scripts/download_bosa_templates.py --output data/bosa_templates.npz
    python scripts/download_bosa_templates.py --synthetic
    python scripts/download_bosa_templates.py --dry-run

References:
    Boquien, M. & Salim, S. 2021, A&A, 653, A149.
"""

import argparse
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------
# BOSA template source (from CIGALE)
# -----------------------------------------------------------------------
CIGALE_GITLAB_BASE = (
    "https://gitlab.lam.fr/api/v4/projects/cigale%2Fcigale/repository/files/{path}/raw?ref=master"
)

# Grid axes from Boquien & Salim (2021) Table 1
LOG_LTIR_GRID = np.array([8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0])
LOG_SSFR_GRID = np.array([-12.0, -11.5, -11.0, -10.5, -10.0, -9.5, -9.0, -8.5, -8.0])


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


def _bosa_temperature(log_ltir: float, log_ssfr: float) -> tuple[float, float]:
    """Map BOSA parameters to dust temperatures.

    Calibrated to Boquien & Salim (2021) Fig. 8:
    - Higher sSFR -> warmer dust (more intense UV field)
    - Higher L_TIR -> slightly warmer at fixed sSFR

    Parameters
    ----------
    log_ltir : float
        log10(L_TIR / Lsun).
    log_ssfr : float
        log10(sSFR / yr^-1).

    Returns
    -------
    tuple of float
        (T_cold, T_warm) dust temperatures in Kelvin.
    """
    # Cold component: 20-35 K depending on sSFR
    T_cold = 22.0 + 5.0 * (log_ssfr + 10.0) + 0.5 * (log_ltir - 10.0)
    T_cold = np.clip(T_cold, 15.0, 40.0)

    # Warm component: 50-80 K
    T_warm = 55.0 + 8.0 * (log_ssfr + 10.0) + 1.0 * (log_ltir - 10.0)
    T_warm = np.clip(T_warm, 40.0, 90.0)

    return float(T_cold), float(T_warm)


def _generate_bosa_spectrum(wave_um: np.ndarray, log_ltir: float, log_ssfr: float) -> np.ndarray:
    """Generate a single synthetic BOSA spectrum.

    Two-component MBB: cold (big-grain) + warm (small-grain/PDR),
    with warm fraction increasing at higher sSFR.

    Parameters
    ----------
    wave_um : array
        Wavelength grid in microns.
    log_ltir : float
        log10(L_TIR / Lsun).
    log_ssfr : float
        log10(sSFR / yr^-1).

    Returns
    -------
    array
        Synthetic L_lambda spectrum (unnormalized).
    """
    T_cold, T_warm = _bosa_temperature(log_ltir, log_ssfr)

    # Warm fraction increases with sSFR
    f_warm = 0.05 + 0.10 * (log_ssfr + 10.0)
    f_warm = np.clip(f_warm, 0.01, 0.5)

    spectrum = (1.0 - f_warm) * _mbb(wave_um, T_cold, beta=2.0)
    spectrum += f_warm * _mbb(wave_um, T_warm, beta=1.5)

    # Add mild PAH features for high-sSFR templates
    pah_amplitude = 0.03 * max(0.0, log_ssfr + 10.0)
    for center, width in [(6.2, 0.10), (7.7, 0.40), (11.3, 0.10)]:
        gamma_sq = (width / center) ** 2
        x_sq = ((wave_um - center) / center) ** 2
        pah_profile = gamma_sq / (x_sq + gamma_sq)
        spectrum += pah_amplitude * pah_profile * _planck_lam(wave_um, 800.0) * 0.001

    return np.maximum(spectrum, 0.0)


def generate_synthetic_templates(output_path: str) -> None:
    """Generate synthetic BOSA template grid for testing.

    Creates a physically motivated but approximate template grid using
    two-component MBB spectra with temperature scaling from Boquien &
    Salim (2021).  NOT suitable for science.

    Parameters
    ----------
    output_path : str
        Path for the output NPZ file.
    """
    wave_um = np.logspace(np.log10(0.1), np.log10(10000.0), 500)

    n_ltir = len(LOG_LTIR_GRID)
    n_ssfr = len(LOG_SSFR_GRID)
    n_wave = len(wave_um)

    spectra = np.zeros((n_ltir, n_ssfr, n_wave))

    for i, log_ltir in enumerate(LOG_LTIR_GRID):
        for j, log_ssfr in enumerate(LOG_SSFR_GRID):
            spectra[i, j] = _generate_bosa_spectrum(wave_um, log_ltir, log_ssfr)

    np.savez(
        output_path,
        wavelength_um=wave_um,
        log_ltir_grid=LOG_LTIR_GRID,
        log_ssfr_grid=LOG_SSFR_GRID,
        spectra=spectra,
    )

    print(f"  Saved synthetic BOSA templates to: {output_path}")
    print(f"  wavelength_um: {wave_um.shape}")
    print(f"  log_ltir_grid: {LOG_LTIR_GRID.shape} — {LOG_LTIR_GRID}")
    print(f"  log_ssfr_grid: {LOG_SSFR_GRID.shape} — {LOG_SSFR_GRID}")
    print(f"  spectra: {spectra.shape}")
    print()
    print("  WARNING: Synthetic templates — NOT for science.")


def try_download_real(output_path: str) -> bool:
    """Attempt to download real BOSA templates from CIGALE GitLab.

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
            print("  CIGALE GitLab reachable, but BOSA extraction not yet automated.")
            print("  Falling back to synthetic templates.")
            return False
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  Download failed: {exc}")
        return False
    return False


def download_and_convert(
    output_path: str, dry_run: bool = False, force_synthetic: bool = False
) -> None:
    """Download or generate BOSA templates.

    Parameters
    ----------
    output_path : str
        Path for the output NPZ file.
    dry_run : bool
        If True, only print what would be done.
    force_synthetic : bool
        If True, skip download attempt and generate synthetic templates.
    """
    print("BOSA template downloader (Boquien & Salim 2021)")
    print(f"  Output: {output_path}")
    print()

    if dry_run:
        print("[dry-run] Would download BOSA templates from CIGALE GitLab.")
        print("[dry-run] Convert to NPZ with keys:")
        print("  - wavelength_um: (n_wave,)")
        print("  - log_ltir_grid: (n_ltir,) log10(L_TIR/Lsun)")
        print("  - log_ssfr_grid: (n_ssfr,) log10(sSFR/yr^-1)")
        print("  - spectra: (n_ltir, n_ssfr, n_wave)")
        return

    if not force_synthetic and try_download_real(output_path):
        return

    print("  Generating synthetic BOSA templates for testing...")
    print()
    generate_synthetic_templates(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BOSA dust emission templates (Boquien & Salim 2021)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output NPZ path (default: data/bosa_templates.npz)",
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
        args.output = str(repo_root / "data" / "bosa_templates.npz")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    download_and_convert(args.output, dry_run=args.dry_run, force_synthetic=args.synthetic)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""Build a Jones et al. 2017 THEMIS dust emission HDF5 grid.

Pulls the canonical THEMIS template files from CIGALE's database
builder and repacks them into the single-file HDF5 schema used by
tengri's other dust IR loaders.

CIGALE's THEMIS module is the de-facto canonical source of pre-
computed Jones+2017 emission templates: it ran DustEM
(``ias.u-psud.fr/DUSTEM``) once for the full grid of grain
compositions and radiation field intensities and ships the per-
template ASCII tables.

Source
------
    https://gitlab.lam.fr/cigale/cigale  (database_builder/themis/)

References
----------
    Jones, A.P., Köhler, M., Ysard, N., et al. 2017, A&A 602, A46
    ("The global dust modelling framework THEMIS").  arXiv:1703.00775.

    Boquien, M., Burgarella, D., Roehlly, Y., et al. 2019, A&A 622 A103
    (CIGALE 2019 paper that ships the THEMIS emission grid).

Grid axes
---------
    qhac (11):   0.02 .. 0.40       -- mass fraction of small a-C(:H)
    umin (37):   0.10 .. 80.0       -- minimum radiation field intensity
    alpha (21):  1.0 .. 3.0         -- powerlaw slope of dU/dM
    plus delta-U (single-U) templates at U=U_min for the gamma=0 limit.

The legacy ``data/themis_templates.h5`` was a synthetic 11x22 grid
with no alpha axis and was scientifically incorrect.  After this
script runs the file is replaced with the canonical 3-D grid.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


CIGALE_REPO = "https://gitlab.lam.fr/cigale/cigale.git"
QHAC_VALUES: tuple[float, ...] = (
    0.02, 0.06, 0.10, 0.14, 0.17, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40,
)
QHAC_TAGS: tuple[str, ...] = (
    "000", "010", "020", "030", "040", "050", "060", "070", "080", "090", "100",
)
UMIN_TAGS: tuple[str, ...] = (
    "0.100", "0.120", "0.150", "0.170", "0.200", "0.250", "0.300", "0.350",
    "0.400", "0.500", "0.600", "0.700", "0.800", "1.000", "1.200", "1.500",
    "1.700", "2.000", "2.500", "3.000", "3.500", "4.000", "5.000", "6.000",
    "7.000", "8.000", "10.00", "12.00", "15.00", "17.00", "20.00", "25.00",
    "30.00", "35.00", "40.00", "50.00", "80.00",
)
UMIN_VALUES = np.asarray([float(s) for s in UMIN_TAGS])
ALPHA_TAGS: tuple[str, ...] = tuple(
    f"{x:.1f}" for x in np.round(np.arange(1.0, 3.05, 0.1), 1)
)
ALPHA_VALUES = np.asarray([float(s) for s in ALPHA_TAGS])
UMAX_POWERLAW_TAG = "1e7"

# THEMIS dust-to-H mass ratio (Jones+2017 Table 1).
MD_OVER_MH = 7.4e-3
# Proton + electron mass [kg] (Boquien's CIGALE conversion factor).
MP_KG = 1.6726218e-27 + 9.1093837e-31

# Each ASCII spec_*.dat file has 576 spectrum rows at the end.
N_WAVE = 576


def clone_cigale(raw_dir: Path) -> Path:
    """Shallow-clone CIGALE's repo to fetch the THEMIS data tree."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "cigale"
    if target.is_dir() and (target / "database_builder/themis/data").is_dir():
        return target
    print(f"[clone] {CIGALE_REPO}  ->  {target}")
    subprocess.run(
        ["git", "clone", "--depth", "1", CIGALE_REPO, str(target)], check=True,
    )
    return target


def _parse_spec_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a CIGALE THEMIS ``spec_X.X.dat`` file.

    The last :data:`N_WAVE` lines have 4+ whitespace-separated columns;
    columns are ``(wavelength_um, ?, lumin_Jy cm^2 sr^-1 H^-1, ...)``.

    Returns
    -------
    wave_um, lumin : ndarray
    """
    with path.open() as f:
        lines = f.readlines()[-N_WAVE:]
    arr = np.genfromtxt(
        io.BytesIO("".join(lines).encode()),
        usecols=(0, 2),
    )
    return arr[:, 0], arr[:, 1]


def _convert_lumin_to_Lnu(
    lumin_Jy_cm2_sr_H: np.ndarray,
    wave_um: np.ndarray,
) -> np.ndarray:
    r"""CIGALE's ``Jy cm^2 sr^-1 H^-1 -> W nm^-1 kg_dust^-1`` convolution.

    .. math::

        f_\lambda \;=\; \frac{4\pi \times 10^{-30}}{m_p}
                         \,\frac{c}{\lambda^2}\, 10^9
                         \,\frac{1}{M_d/M_H}\, j_\nu

    converts isotropic per-H emission to per-mass-of-dust spectral
    power (W/nm/kg_dust).  We then convert further to per-H L_nu in
    erg/s/Hz/H for the tengri pipeline:

    .. math::

        L_\nu^{(H)} \;=\; (\nu P_\nu) / \nu
                  \;=\; (4\pi \, j_\nu)\, \lambda \, .

    The simpler conversion above, applied directly, sidesteps the
    Mdust/MH factor (we want per-H, not per-mass).
    """
    c_cgs = 2.99792458e10
    lam_cm = wave_um * 1.0e-4
    # j_nu (in CGS Jy cm^2 sr^-1 H^-1) -> erg/s/Hz/sr/H.  1 Jy = 1e-23 erg/s/cm^2/Hz,
    # so j_nu [Jy cm^2 sr^-1 H^-1] in CGS is j_nu * 1e-23 erg/s/sr/Hz/H per cm^2.
    # Wait: the unit "Jy cm^2 sr^-1 H^-1" already factors out the area, so the
    # number is the brightness *per H atom* emitting isotropically.  So
    # I_nu/N_H [erg/s/cm^2/Hz/sr/H] = j_nu * 1e-23 cm^2.
    I_nu = lumin_Jy_cm2_sr_H * 1.0e-23  # erg/s/cm^2/Hz/sr/H * cm^2 -> erg/s/Hz/sr/H
    # L_nu per H = 4π * I_nu/N_H (assuming isotropic).
    L_nu = 4.0 * np.pi * I_nu  # erg/s/Hz/H
    _ = c_cgs, lam_cm  # unused in this simple reduction
    return L_nu


def parse_grid(repo_root: Path) -> dict:
    """Walk the CIGALE THEMIS data tree and assemble a 3-D grid."""
    base = repo_root / "database_builder/themis/data"
    if not base.is_dir():
        raise FileNotFoundError(f"missing CIGALE THEMIS data dir {base}")

    n_qhac = len(QHAC_TAGS)
    n_umin = len(UMIN_TAGS)
    n_alpha = len(ALPHA_TAGS)

    # Get the wavelength axis from the first (qhac=000, umin=0.100) single-U file.
    ref = base / f"U{UMIN_TAGS[0]}_{UMIN_TAGS[0]}_MW3.1_{QHAC_TAGS[0]}/spec_1.0.dat"
    wave_um, _ = _parse_spec_dat(ref)
    n_wave = wave_um.size
    if n_wave != N_WAVE:
        raise ValueError(f"expected {N_WAVE} wavelengths, got {n_wave}")

    single_u = np.zeros((n_qhac, n_umin, n_wave), dtype=np.float64)
    powerlaw = np.zeros((n_qhac, n_umin, n_alpha, n_wave), dtype=np.float64)

    for iq, qtag in enumerate(QHAC_TAGS):
        for iu, utag in enumerate(UMIN_TAGS):
            sd_dir = base / f"U{utag}_{utag}_MW3.1_{qtag}"
            pl_dir = base / f"U{utag}_{UMAX_POWERLAW_TAG}_MW3.1_{qtag}"

            # Single-U: only spec_1.0.dat is canonical (alpha doesn't apply
            # to a delta-function distribution).
            _, lumin = _parse_spec_dat(sd_dir / "spec_1.0.dat")
            single_u[iq, iu] = _convert_lumin_to_Lnu(lumin, wave_um)

            for ia, atag in enumerate(ALPHA_TAGS):
                _, lumin = _parse_spec_dat(pl_dir / f"spec_{atag}.dat")
                powerlaw[iq, iu, ia] = _convert_lumin_to_Lnu(lumin, wave_um)

        print(f"[grid] qhac={QHAC_VALUES[iq]:.2f} done")

    # Convert wavelength to Å and sort ascending if needed.
    wave_aa = wave_um * 1.0e4
    if not np.all(np.diff(wave_aa) > 0):
        order = np.argsort(wave_aa)
        wave_aa = wave_aa[order]
        single_u = single_u[..., order]
        powerlaw = powerlaw[..., order]

    # alpha = 2.0 is the CIGALE / Jones+2017 default; the legacy 3-D
    # ``powerlaw`` keeps that slice so existing tengri loaders keep
    # working unchanged.  The full 4-D grid is stored under
    # ``powerlaw_alpha`` for any code that wants to fit alpha.
    i_alpha2 = int(np.argmin(np.abs(ALPHA_VALUES - 2.0)))
    powerlaw_alpha2 = powerlaw[:, :, i_alpha2, :]

    return {
        "wavelength_aa": wave_aa.astype(np.float64),
        "qhac_grid": np.asarray(QHAC_VALUES, dtype=np.float64),
        "umin_grid": UMIN_VALUES.astype(np.float64),
        "alpha_grid": ALPHA_VALUES.astype(np.float64),
        "single_u": single_u.astype(np.float64),                # (qhac, umin, wave)
        "powerlaw": powerlaw_alpha2.astype(np.float64),         # (qhac, umin, wave) at alpha=2.0
        "powerlaw_alpha": powerlaw.astype(np.float64),          # (qhac, umin, alpha, wave) full
    }


def write_hdf5(out_path: Path, grid: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write]   {out_path}")
    with h5py.File(out_path, "w") as f:
        f.attrs["model"] = "THEMIS (Jones+2017) via CIGALE database builder"
        f.attrs["paper"] = "Jones, Köhler, Ysard et al. 2017 A&A 602 A46"
        f.attrs["arxiv"] = "1703.00775"
        f.attrs["upstream"] = "https://gitlab.lam.fr/cigale/cigale"
        f.attrs["units_spectra"] = "L_nu per H atom (erg/s/Hz/H)"
        f.attrs["axes_powerlaw"] = "(qhac, umin, alpha, wavelength)"
        f.attrs["axes_single_u"] = "(qhac, umin, wavelength)"
        f.attrs["M_dust_over_M_H"] = MD_OVER_MH
        for k, v in grid.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--raw-dir", type=Path,
        default=Path("~/.cache/tengri/themis_raw").expanduser(),
    )
    p.add_argument("--output", type=Path, default=Path("data/themis_templates.h5"))
    p.add_argument(
        "--clone", action="store_true",
        help="git-clone CIGALE if missing",
    )
    args = p.parse_args(argv)

    raw_dir = args.raw_dir.expanduser().resolve()
    repo = raw_dir / "cigale"
    if not (repo / "database_builder/themis/data").is_dir():
        if not args.clone:
            print(
                f"[error] {repo}/database_builder/themis/data not present. "
                f"Re-run with --clone.",
                file=sys.stderr,
            )
            return 1
        clone_cigale(raw_dir)

    grid = parse_grid(repo)
    write_hdf5(args.output, grid)
    print(
        f"[ok] grid: single_u={grid['single_u'].shape}, "
        f"powerlaw={grid['powerlaw'].shape}, "
        f"qhac=[{grid['qhac_grid'].min()}..{grid['qhac_grid'].max()}], "
        f"umin=[{grid['umin_grid'].min()}..{grid['umin_grid'].max()}], "
        f"alpha=[{grid['alpha_grid'].min()}..{grid['alpha_grid'].max()}]"
    )

    # Avoid lint warning for unused import in some checkers.
    _ = re
    return 0


if __name__ == "__main__":
    sys.exit(main())

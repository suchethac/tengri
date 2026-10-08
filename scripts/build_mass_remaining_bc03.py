#!/usr/bin/env python
"""Build ``mass_remaining_bc03pdva94_chabrier.h5`` from the BC03 2003 model files.

The surviving-mass table must come from the same BC03 release, isochrones, IMF,
spectral library and metallicities as the spectra in ``data/bc03_pdva_stelib_chabrier.h5``.
Those spectra are the Bruzual & Charlot (2003) original release,
``bc03.models.padova_1994_chabrier_imf.tar.gz`` (Padova 1994 tracks, STELIB
``hr`` library, Chabrier IMF): the grid's spectra equal that release's
``bc2003_hr_m*_chab_ssp.ised`` to float32 round-off (see the PROVENANCE note).

The quantity repackaged is column (7), ``M*``, of each ``bc2003_hr_m*_chab_ssp.4color``
file. The release's own header defines it as "Total mass in stars at this age",
and column (9) ``Mgalaxy = M* + Mgas = 1`` for every row, where ``Mgas`` is the
mass returned to the ISM. Remnants are not returned, so ``M*`` is living stars
plus remnants per 1 Msun formed. This script asserts that identity.

Usage (from the repository root)::

    python scripts/build_mass_remaining_bc03.py \\
        --tarball /path/to/bc03.models.padova_1994_chabrier_imf.tar.gz

Data: Bruzual & Charlot (2003), MNRAS 344, 1000, http://www.bruzual.org/bc03/
"""

from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
GRID = REPO / "data" / "bc03_pdva_stelib_chabrier.h5"
OUT = (
    REPO
    / "src"
    / "tengri"
    / "data"
    / "ssp_mass_remaining"
    / "mass_remaining_bc03pdva94_chabrier.h5"
)
TAR_MEMBER = "./bc03/models/Padova1994/chabrier/bc2003_hr_{key}_chab_ssp.4color"
KEYS = ("m22", "m32", "m42", "m52", "m62", "m72")
SOURCE_URL = (
    "http://www.bruzual.org/bc03/Original_version_2003/bc03.models.padova_1994_chabrier_imf.tar.gz"
)
COL_AGE, COL_MSTAR, COL_MGAS, COL_MGAL = 0, 6, 7, 8
Z_PATTERN = re.compile(r"X=([0-9.]+), Y=([0-9.]+), Z=([0-9.]+)")


def sha256_of(path: Path) -> str:
    """Return the sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_4color(text: str) -> tuple[float, np.ndarray]:
    """Return the header metal mass fraction Z and the numeric 4color table."""
    z_abs = float(Z_PATTERN.search(text).group(3))
    rows = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    return z_abs, np.array(rows, dtype=np.float64)


def main() -> None:
    """Rebuild the table and write it to the package-data directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tarball", type=Path, required=True)
    parser.add_argument(
        "--grid", type=Path, default=GRID, help="tengri BC03 SSP grid (untracked data file)"
    )
    args = parser.parse_args()

    with h5py.File(args.grid, "r") as grid:
        grid_age_yr = 10.0 ** (grid["ssp_lg_age_gyr"][:] + 9.0)
        grid_lgz = grid["ssp_lgmet"][:]

    log_z, tables = [], []
    with tarfile.open(args.tarball, "r:gz") as tar:
        for key in KEYS:
            text = tar.extractfile(TAR_MEMBER.format(key=key)).read().decode("ascii")
            z_abs, tab = parse_4color(text)
            assert np.abs(tab[:, COL_MGAL] - 1.0).max() == 0.0, "Mgalaxy != 1"
            gap = np.abs(tab[:, COL_MSTAR] + tab[:, COL_MGAS] - tab[:, COL_MGAL]).max()
            assert gap < 2e-5, f"M* + Mgas != Mgalaxy ({gap}): M* would not include remnants"
            log_z.append(np.log10(z_abs))
            tables.append(tab)

    log_z = np.array(log_z)
    assert np.allclose(log_z, grid_lgz, atol=1e-5), (log_z, grid_lgz)

    ages = tables[0][:, COL_AGE]
    for tab in tables:
        assert np.array_equal(tab[:, COL_AGE], ages)
    age_yr = 10.0**ages
    node_gap = np.abs(age_yr / grid_age_yr[1:] - 1.0).max()
    assert grid_age_yr[0] == 0.0 and grid_age_yr.shape[0] - 1 == ages.shape[0]
    assert node_gap < 1e-5, f"BC03 4color ages are not the grid's ages ({node_gap})"

    mass = np.array([t[:, COL_MSTAR] for t in tables], dtype=np.float64)
    assert mass.min() > 0.0 and mass.max() <= 1.0
    # BC03's own M* is not strictly monotonic: it rises by up to 2% at ages of a few
    # Myr (m32/m62/m72) and by <= 7e-4 at ages above 1 Gyr. The values are the
    # release's, so they are repackaged unsmoothed and the rises are bounded here.
    rise = np.diff(mass, axis=1).max()
    assert rise <= 0.02, f"M* rises by {rise}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT, "w") as out:
        out.create_dataset("log10_age_yr", data=ages.astype(np.float64))
        out.create_dataset("log10_z_abs", data=grid_lgz.astype(np.float64))
        out.create_dataset("mass_remaining", data=mass)
        attrs = {
            "quantity": "living stars + remnants per unit formed mass",
            "isochrones": "Padova 1994 (Bertelli et al. 1994) + S. Charlot (1997), X=0.70 Y=0.28",
            "imf": "Chabrier (lognormal below 1 Msun + x=1.3 power law, 0.1-100 Msun)",
            "spectral_library": "STELIB (hr models)",
            "source": "BC03 original release 2003, column (7) M* of bc2003_hr_m*_chab_ssp.4color",
            "remnant_prescription": "BC03 internal (Bruzual & Charlot 2003, Sect. 2); "
            "not Renzini & Ciotti",
            "citation": "Bruzual, G. & Charlot, S. 2003, MNRAS, 344, 1000, "
            "doi:10.1046/j.1365-8711.2003.06897.x, arXiv:astro-ph/0309134",
            "terms_of_use": "BC03 files carry '(C) 1995-2003 G. Bruzual A. & S. Charlot - "
            "All Rights Reserved'; repackaged as a numeric extract with attribution; cite BC03",
            "monotonicity": "not strictly non-increasing: BC03's own M* rises by up to 0.0195 "
            "between adjacent nodes at log10 age 6.0-6.5, and by <= 7e-4 above 1 Gyr; unsmoothed",
            "generator": "scripts/build_mass_remaining_bc03.py",
            "generator_args": "--tarball bc03.models.padova_1994_chabrier_imf.tar.gz",
            "input_url": SOURCE_URL,
            "input_sha256": sha256_of(args.tarball),
        }
        for name, value in attrs.items():
            out.attrs[name] = value
    print(f"wrote {OUT} shape {mass.shape}; age node gap {node_gap:.2e}")


if __name__ == "__main__":
    main()

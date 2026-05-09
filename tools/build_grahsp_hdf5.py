"""Build the GRAHSP HDF5 data bundle from upstream raw template files.

Outputs ``data/grahsp/grahsp_templates.h5`` containing:

- ``feii_bruhweiler2008/wave_nm`` (n,) — Bruhweiler+Verner 2008 FeII forest
  template wavelengths in nm (de-redshifted from the upstream catalog
  default of z=0.004, since the raw CIGALE database stores the de-redshifted
  template; see paper §2.1.2).
- ``feii_bruhweiler2008/lumin`` (n,) — relative intensity, normalised so
  that the integrated H-beta-equivalent luminosity scaling matches upstream.
- ``netzer1990_lines/wave_nm`` (n_lines,) — central wavelengths in nm.
- ``netzer1990_lines/broad`` (n_lines,) — broad-line strengths relative to
  H-beta (broad).
- ``netzer1990_lines/narrow_sy2`` (n_lines,) — Sy2 narrow-line strengths
  relative to H-beta (narrow).
- ``netzer1990_lines/narrow_liner`` (n_lines,) — LINER narrow-line strengths.
- ``netzer1990_lines/name`` (n_lines,) — line names (UTF-8).
- ``torus/wave_nm`` (n_torus,) — fixed wavelength grid used by the upstream
  ``activategtorus`` module (see source for provenance).

Run::

    .venv/bin/python tools/build_grahsp_hdf5.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from tools.generate_grahsp_fixtures import (
    _load_full_torus_wave,
    parse_mor_netzer_lines,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "grahsp"
OUT = REPO_ROOT / "data" / "grahsp" / "grahsp_templates.h5"

FEII_RAW = RAW / "feii_bruhweiler2008_d11_m20_20p5.txt"
LINES_RAW = RAW / "mor_netzer_2012_emission_lines.txt"

# H-beta normalisation constant: upstream uses L(line)/L(5100Å) ratios where
# L(5100) = lambda*L_lambda. See database_builder/activate/agn/mor_netzer_2012/readme.
HBETA_BROAD_RATIO = 0.02  # L(Hb_broad) / L(5100)
HBETA_NARROW_RATIO = 0.002  # L(Hb_narrow) / L(5100)


def _load_feii_template():
    """Load and normalise the Bruhweiler+Verner 2008 FeII template.

    Mirrors upstream ``database_builder/__init__.py`` (FeII import block):

    1. Raw file columns: observed-frame wavelength [Å], :math:`L_\\nu`
       (arbitrary units). The catalog z = 4593.4/4575 - 1 ≈ 0.00404.
    2. Convert :math:`L_\\nu \\to L_\\lambda = L_\\nu c / \\lambda_{\\rm obs}^2`.
    3. De-redshift wavelengths to rest frame.
    4. Normalise so :math:`L_\\lambda(4575\\,\\mathrm{\\AA, rest}) = 1`.

    Returns wave_nm (rest-frame, nm) and the normalised :math:`L_\\lambda`.
    """
    from scipy import constants as cst

    arr = np.loadtxt(FEII_RAW)
    wave_obs_angstrom = arr[:, 0]
    L_nu = arr[:, 1]
    # Upstream catalog redshift (paper §2.1.2): the template's FeII 4593.4
    # peak appears at observed 4575.
    z = 4593.4 / 4575.0 - 1.0
    wave_rest_angstrom = wave_obs_angstrom / (1.0 + z)
    # Convert L_nu -> L_lambda using observed-frame wave (upstream convention).
    L_lambda = L_nu * cst.c / wave_obs_angstrom**2
    # Normalise at rest-frame 4575 Å.
    norm_idx = np.argmin(np.abs(wave_rest_angstrom - 4575.0))
    norm = L_lambda[norm_idx]
    L_lambda = L_lambda / norm
    return wave_rest_angstrom / 10.0, L_lambda


def main():
    feii_wave, feii_lumin = _load_feii_template()
    lines = parse_mor_netzer_lines(LINES_RAW)
    line_names = np.array([r[0] for r in lines], dtype="S")
    line_wave_nm = np.array([r[1] for r in lines], dtype=np.float64)
    line_broad = np.array([r[2] for r in lines], dtype=np.float64)
    line_narrow_sy2 = np.array([r[3] for r in lines], dtype=np.float64)
    line_narrow_liner = np.array([r[4] for r in lines], dtype=np.float64)
    torus_wave_nm = _load_full_torus_wave()

    with h5py.File(OUT, "w") as f:
        f.attrs["source"] = "JohannesBuchner/GRAHSP @ database_builder/activate/agn/"
        f.attrs["paper"] = "Buchner et al. 2024, arXiv:2405.19297"
        f.attrs["license"] = "CeCILL-v2 (upstream)"

        feii = f.create_group("feii_bruhweiler2008")
        feii.attrs["density"] = "n_H = 1e11 cm^-3"
        feii.attrs["microturbulence"] = "xi = 20 km/s"
        feii.attrs["ionizing_flux"] = "phi_H = 10^20.5 cm^-2 s^-1"
        feii.attrs["redshift_dereddened_from"] = 0.004
        feii.attrs["citation"] = "Bruhweiler & Verner 2008, ApJ, 675, 83"
        feii.create_dataset("wave_nm", data=feii_wave)
        feii.create_dataset("lumin", data=feii_lumin)

        lns = f.create_group("netzer1990_lines")
        lns.attrs["citation"] = "Netzer 1990; Mor & Netzer 2012; H-gamma from Rakshit+ 2020"
        lns.attrs["normalisation_broad"] = HBETA_BROAD_RATIO
        lns.attrs["normalisation_narrow"] = HBETA_NARROW_RATIO
        lns.attrs["units"] = "wave_nm in nm; broad/narrow in L(line)/L(Hbeta) ratios"
        lns.create_dataset("name", data=line_names)
        lns.create_dataset("wave_nm", data=line_wave_nm)
        lns.create_dataset("broad", data=line_broad)
        lns.create_dataset("narrow_sy2", data=line_narrow_sy2)
        lns.create_dataset("narrow_liner", data=line_narrow_liner)

        torus = f.create_group("torus")
        torus.attrs["source"] = "activategtorus.py self.wave (nm)"
        torus.create_dataset("wave_nm", data=torus_wave_nm)

    print(f"wrote {OUT}")
    print(f"  feii: {feii_wave.size} samples, {feii_wave.min():.1f}-{feii_wave.max():.1f} nm")
    print(f"  lines: {len(lines)} lines, {line_wave_nm.min():.0f}-{line_wave_nm.max():.0f} nm")
    print(f"  torus grid: {torus_wave_nm.size} points, {torus_wave_nm.min():.1f}-{torus_wave_nm.max():.0f} nm")


if __name__ == "__main__":
    main()

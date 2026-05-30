"""Regenerate ``data/dale2014_templates.h5`` from Dale's official source files.

The historical ``dale2014_templates.h5`` shipped with tengri does not match
either the official Dale+2014 templates (Wyoming) or pcigale's bundled
templates — its per-α shapes drift by ~6 — 1000× at sub-mm wavelengths
(see issue #415). This script writes the authoritative file directly
from the canonical source:

    https://physics.uwyo.edu/~ddale/research/seds/spectra.SFAGN.tar.gz

The README on that page (``README.SFAGN``) documents the format:

* ``spectra/spectra.0.00AGN.dat`` — pure star-forming (AGN fraction = 0).
* Column 1: wavelength in micrometres.
* Columns 2 — 65: :math:`\\log_{10}(\\nu f_\\nu)` (cgs-equivalent;
  absolute scaling is arbitrary per the README, only the shape matters).
* Column index ``c`` (1-based, 2 ≤ c ≤ 65) corresponds to
  :math:`\\alpha_{\\rm SF} = 0.0625 \\cdot (c-1)`, i.e. α=0.0625 in col 2,
  α=2.0000 in col 33, α=4.0000 in col 65.
* Grid: 1496 wavelengths × 64 α values.

Tengri storage convention (``data/dale2014_templates.h5``):

* ``wavelength_aa`` — wavelength in Angstrom, shape (1496,).
* ``alpha_grid``    — α values, shape (64,).
* ``templates_sf``  — :math:`L_\\nu` templates per-Hz, normalised so
  :math:`\\int L_\\nu d\\nu = 1`, shape (n_alpha, n_wave). The runtime
  loader ``create_dale2014_from_grid`` multiplies by ``L_absorbed``
  directly to get an :math:`L_\\nu` SED.

Conversion: :math:`L_\\nu(\\lambda) = 10^{col}(\\lambda) / \\nu`, where
:math:`\\nu = c/\\lambda`. Normalising by the per-template
:math:`\\int L_\\nu d\\nu` makes the absolute units of column 2-65 drop
out (the README explicitly says "the absolute scaling is arbitrary").

Run from repo root::

    python scripts/regenerate_dale2014_from_official.py

This will fetch the tarball into ``/tmp`` if the unpacked file is not
already there, and overwrite ``data/dale2014_templates.h5``.
"""

from __future__ import annotations

import os
import subprocess
import tarfile

import h5py
import numpy as np

SOURCE_URL = "https://physics.uwyo.edu/~ddale/research/seds/spectra.SFAGN.tar.gz"
SOURCE_TARBALL = "/tmp/dale2014_spectra.tar.gz"
SOURCE_UNPACKED_DIR = "/tmp/spectra"
SOURCE_FILE = os.path.join(SOURCE_UNPACKED_DIR, "spectra.0.00AGN.dat")
OUTPUT = "data/dale2014_templates.h5"

C_AA_PER_S = 2.998e18  # Å/s — matches tengri runtime


def _download_and_unpack() -> None:
    """Fetch and unpack the official Dale 2014 SFAGN tarball if needed."""
    if os.path.isfile(SOURCE_FILE):
        return
    if not os.path.isfile(SOURCE_TARBALL):
        print(f"Downloading {SOURCE_URL} ...")
        # The Wyoming physics server uses a self-signed certificate chain;
        # ``curl -k`` and ``urllib`` need help to ignore that. Use ``curl``
        # directly because it's already on every dev box.
        subprocess.run(
            ["curl", "-sk", "-o", SOURCE_TARBALL, SOURCE_URL],
            check=True,
        )
    os.makedirs(SOURCE_UNPACKED_DIR, exist_ok=True)
    with tarfile.open(SOURCE_TARBALL, "r:gz") as tar:
        tar.extractall("/tmp")
    if not os.path.isfile(SOURCE_FILE):
        raise FileNotFoundError(
            f"Expected {SOURCE_FILE} after unpacking {SOURCE_TARBALL}; "
            "the tarball layout may have changed."
        )


def regenerate(output_path: str = OUTPUT) -> None:
    _download_and_unpack()

    raw = np.loadtxt(SOURCE_FILE)
    if raw.shape[1] != 65:
        raise ValueError(
            f"Expected 65 columns in {SOURCE_FILE} (1 wavelength + 64 α), got {raw.shape[1]}."
        )
    n_wave = raw.shape[0]

    wave_aa = raw[:, 0] * 1e4  # μm → Å
    log_nu_fnu = raw[:, 1:]  # (n_wave, 64)

    # α grid is uniformly spaced 0.0625 to 4.0000 in steps of 0.0625.
    alphas = np.arange(1, 65, dtype=np.float64) * 0.0625
    if not (abs(alphas[31] - 2.0) < 1e-12 and abs(alphas[-1] - 4.0) < 1e-12):
        raise AssertionError("α grid does not match the README spec.")

    # Convert log10(ν·f_ν) → L_ν per α, then normalise to ∫L_ν dν = 1.
    nu = C_AA_PER_S / wave_aa  # Hz, descending (λ ascending)
    nu_fnu = 10.0**log_nu_fnu
    L_nu = nu_fnu / nu[:, None]  # (n_wave, n_alpha)
    templates = L_nu.T.copy()  # (n_alpha, n_wave)

    sort_idx = np.argsort(nu)
    nu_sorted = nu[sort_idx]
    for i in range(templates.shape[0]):
        integral = np.trapezoid(templates[i, sort_idx], nu_sorted)
        if integral <= 0:
            raise ValueError(
                f"α={alphas[i]} template integrated to {integral} — "
                "non-positive integral indicates malformed source data."
            )
        templates[i] /= integral

    # Sanity check: each row should now integrate to ~1 in ν.
    for i in range(templates.shape[0]):
        integral = np.trapezoid(templates[i, sort_idx], nu_sorted)
        if not (0.999 < integral < 1.001):
            raise ValueError(f"α={alphas[i]} normalisation failed (got {integral:.6f}).")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["model"] = "Dale et al. 2014"
        f.attrs["reference"] = "Dale et al. 2014, ApJ, 784, 83"
        f.attrs["wavelength_unit"] = "Angstrom"
        f.attrs["spectra_unit"] = "L_nu normalized (integral over nu = 1)"
        f.attrs["source"] = SOURCE_URL
        f.attrs["agn_fraction"] = "0.00 (pure star-forming)"
        f.attrs["issue"] = "https://github.com/suchethac/tengri/issues/415"

        f.create_dataset("wavelength_aa", data=wave_aa)
        f.create_dataset("alpha_grid", data=alphas)
        f["alpha_grid"].attrs["unit"] = "dimensionless"
        f.create_dataset("templates_sf", data=templates)
        f["templates_sf"].attrs["shape"] = "(n_alpha, n_wave)"

    size_kb = os.path.getsize(output_path) / 1e3
    print(f"  Dale2014: {output_path} (n_alpha={len(alphas)}, n_wave={n_wave}, {size_kb:.1f} kB)")


if __name__ == "__main__":
    regenerate()

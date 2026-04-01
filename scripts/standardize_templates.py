#!/usr/bin/env python3
"""Standardize ALL dust emission templates to a single HDF5 format.

Reads existing templates (NPZ or HDF5) via tengri's loaders, then
writes a single normalized HDF5 file per model with consistent units:

- Wavelength: Angstrom (ascending)
- Spectra: L_nu convention, normalized so ∫L_ν dν = 1 per template
- Grid axes: physical units with metadata

Output files: data/{model}_templates.h5 (overwrites existing)

Usage:
    python scripts/standardize_templates.py
"""

import os
import sys

import h5py
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def write_dl07(outpath: str) -> None:
    """Standardize DL07 templates."""
    from tengri.models.dust.emission import load_draine_li_templates

    # Load from NPZ (properly normalized)
    data = load_draine_li_templates("data/dl07_templates.npz")

    with h5py.File(outpath, "w") as f:
        f.attrs["model"] = "Draine & Li 2007"
        f.attrs["reference"] = "Draine & Li 2007, ApJ, 657, 810"
        f.attrs["wavelength_unit"] = "Angstrom"
        f.attrs["spectra_unit"] = "L_nu normalized (integral over nu = 1)"

        f.create_dataset("wavelength", data=np.array(data["wavelength"]))
        f.create_dataset("qpah_grid", data=np.array(data["qpah_grid"]))
        f["qpah_grid"].attrs["unit"] = "percent"
        f.create_dataset("umin_grid", data=np.array(data["umin_grid"]))
        f["umin_grid"].attrs["unit"] = "Mathis ISRF"
        f.create_dataset("single_u", data=np.array(data["single_u"]))
        f["single_u"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        f.create_dataset("powerlaw", data=np.array(data["powerlaw"]))
        f["powerlaw"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"

    print(f"  DL07: {outpath} ({os.path.getsize(outpath) / 1e6:.1f} MB)")


def write_dl14(outpath: str) -> None:
    """Standardize DL14 templates."""
    from tengri.models.dust.emission import load_dl14_templates

    data = load_dl14_templates("data/dl14_templates.h5")

    with h5py.File(outpath, "w") as f:
        f.attrs["model"] = "Draine & Li 2014 update"
        f.attrs["reference"] = "Draine & Li 2007 + 2014 updates"
        f.attrs["wavelength_unit"] = "Angstrom"
        f.attrs["spectra_unit"] = "L_nu normalized (integral over nu = 1)"

        f.create_dataset("wavelength", data=np.array(data["wavelength"]))
        f.create_dataset("qpah_grid", data=np.array(data["qpah_grid"]))
        f["qpah_grid"].attrs["unit"] = "percent"
        f.create_dataset("umin_grid", data=np.array(data["umin_grid"]))
        f["umin_grid"].attrs["unit"] = "Mathis ISRF"
        f.create_dataset("alpha_grid", data=np.array(data["alpha_grid"]))
        f["alpha_grid"].attrs["unit"] = "dimensionless"
        f.create_dataset("single_u", data=np.array(data["single_u"]))
        f["single_u"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        f.create_dataset("powerlaw", data=np.array(data["powerlaw"]))
        f["powerlaw"].attrs["shape"] = "(n_qpah, n_umin, n_alpha, n_wave)"

    print(f"  DL14: {outpath} ({os.path.getsize(outpath) / 1e6:.1f} MB)")


def write_dale2014(outpath: str) -> None:
    """Standardize Dale+2014 templates."""
    from tengri.models.dust.emission import create_dale2014_from_grid

    # Load raw — create_dale2014_from_grid normalizes internally
    import numpy as np

    data = np.load("data/dale2014_templates.npz")
    wave_aa = data["wavelength_aa"]
    alpha_grid = data["alpha_grid"]
    templates = data["templates_sf"]

    if templates.shape[0] == len(wave_aa):
        templates = templates.T

    # Normalize to L_nu convention
    wave_cm = wave_aa * 1e-8
    c_cgs = 2.99792458e10
    nu = c_cgs / wave_cm

    templates_lnu = templates * (wave_cm**2)[None, :] / c_cgs
    for i in range(templates_lnu.shape[0]):
        integral = -np.trapezoid(templates_lnu[i], nu)
        if integral > 0:
            templates_lnu[i] /= integral

    with h5py.File(outpath, "w") as f:
        f.attrs["model"] = "Dale et al. 2014"
        f.attrs["reference"] = "Dale et al. 2014, ApJ, 784, 83"
        f.attrs["wavelength_unit"] = "Angstrom"
        f.attrs["spectra_unit"] = "L_nu normalized (integral over nu = 1)"

        f.create_dataset("wavelength_aa", data=wave_aa)
        f.create_dataset("alpha_grid", data=alpha_grid)
        f["alpha_grid"].attrs["unit"] = "dimensionless"
        f.create_dataset("templates_sf", data=templates_lnu)
        f["templates_sf"].attrs["shape"] = "(n_alpha, n_wave)"

    print(f"  Dale2014: {outpath} ({os.path.getsize(outpath) / 1e6:.1f} MB)")


def write_astrodust(outpath: str) -> None:
    """Standardize Astrodust templates."""
    from tengri.models.dust.emission import load_astrodust_templates

    data = load_astrodust_templates("data/astrodust_templates.npz")

    with h5py.File(outpath, "w") as f:
        f.attrs["model"] = "Astrodust+PAH (Hensley & Draine 2023)"
        f.attrs["reference"] = "Hensley & Draine 2023, ApJ, 948, 55"
        f.attrs["wavelength_unit"] = "Angstrom"
        f.attrs["spectra_unit"] = "L_nu normalized (integral over nu = 1)"

        f.create_dataset("wavelength_aa", data=np.array(data["wavelength_aa"]))
        f.create_dataset("qpah_grid", data=np.array(data["qpah_grid"]))
        f["qpah_grid"].attrs["unit"] = "percent"
        f.create_dataset("umin_grid", data=np.array(data["umin_grid"]))
        f["umin_grid"].attrs["unit"] = "Mathis ISRF"
        f.create_dataset("single_u", data=np.array(data["single_u"]))
        f["single_u"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        f.create_dataset("powerlaw", data=np.array(data["powerlaw"]))
        f["powerlaw"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"

    print(f"  Astrodust: {outpath} ({os.path.getsize(outpath) / 1e6:.1f} MB)")


def write_themis(outpath: str) -> None:
    """Standardize THEMIS templates."""
    from tengri.models.dust.emission import load_themis_templates

    data = load_themis_templates("data/themis_templates.npz")

    with h5py.File(outpath, "w") as f:
        f.attrs["model"] = "THEMIS (Jones et al. 2017)"
        f.attrs["reference"] = "Jones et al. 2017, A&A, 602, A46"
        f.attrs["wavelength_unit"] = "Angstrom"
        f.attrs["spectra_unit"] = "L_nu normalized (integral over nu = 1)"

        f.create_dataset("wavelength_aa", data=np.array(data["wavelength_aa"]))
        f.create_dataset("qhac_grid", data=np.array(data["qhac_grid"]))
        f["qhac_grid"].attrs["unit"] = "fraction"
        f.create_dataset("umin_grid", data=np.array(data["umin_grid"]))
        f["umin_grid"].attrs["unit"] = "Mathis ISRF"
        f.create_dataset("single_u", data=np.array(data["single_u"]))
        f["single_u"].attrs["shape"] = "(n_qhac, n_umin, n_wave)"
        f.create_dataset("powerlaw", data=np.array(data["powerlaw"]))
        f["powerlaw"].attrs["shape"] = "(n_qhac, n_umin, n_wave)"

    print(f"  THEMIS: {outpath} ({os.path.getsize(outpath) / 1e6:.1f} MB)")


def write_bosa(outpath: str) -> None:
    """Standardize BOSA templates."""
    from tengri.models.dust.emission import load_bosa_templates

    data = load_bosa_templates("data/bosa_templates.npz")

    with h5py.File(outpath, "w") as f:
        f.attrs["model"] = "BOSA (Boquien & Salim 2021)"
        f.attrs["reference"] = "Boquien & Salim 2021, A&A, 653, A149"
        f.attrs["wavelength_unit"] = "Angstrom"
        f.attrs["spectra_unit"] = "L_nu normalized (integral over nu = 1)"

        f.create_dataset("wavelength_aa", data=np.array(data["wavelength_aa"]))
        f.create_dataset("log_ltir_grid", data=np.array(data["log_ltir_grid"]))
        f["log_ltir_grid"].attrs["unit"] = "log10(Lsun)"
        f.create_dataset("log_ssfr_grid", data=np.array(data["log_ssfr_grid"]))
        f["log_ssfr_grid"].attrs["unit"] = "log10(yr^-1)"
        f.create_dataset("spectra", data=np.array(data["spectra"]))
        f["spectra"].attrs["shape"] = "(n_ltir, n_ssfr, n_wave)"

    print(f"  BOSA: {outpath} ({os.path.getsize(outpath) / 1e6:.1f} MB)")


def main():
    print("Standardizing dust emission templates to HDF5...")
    print("Units: wavelength=Angstrom, spectra=L_nu normalized\n")

    write_dl07("data/dl07_templates.h5")
    write_dl14("data/dl14_templates.h5")
    write_dale2014("data/dale2014_templates.h5")
    write_astrodust("data/astrodust_templates.h5")
    write_themis("data/themis_templates.h5")
    write_bosa("data/bosa_templates.h5")

    print("\nDone. Remove _v2.h5 files if no longer needed:")
    print("  rm data/*_v2.h5")


if __name__ == "__main__":
    main()

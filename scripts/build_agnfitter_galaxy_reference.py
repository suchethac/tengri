#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Convert AGNfitter-rX GALAXY (BC03) reference pickles to a vendored HDF5.

AGNfitter-rX ships its stellar-population (GALAXY) library as two Python
pickles under ``models/GALAXY/``: ``BC03_840seds.pickle`` (single, solar-like
metallicity: 30 ages x 28 tau values) and ``BC03_seds_metal_medium.pickle``
(4 metallicities x 20 ages x 18 tau values). Both are ``astropy.units.Quantity``
dicts keyed by ``'wavelength'``, ``'age-values'``, ``'tau-values'``,
``'metallicity-values'``, ``'SED'`` and ``'SFR'`` (see
``functions/MODEL_AGNfitter.py`` ``GALAXY()``, the ``'BC03'`` /
``'BC03_metal'`` branches).

This is the build-time converter: it reads the upstream pickles ONCE (via a
restricted unpickler with a minimal numpy/astropy-units allow-list), converts
the native F_lambda [Lsun/Angstrom] SED to L_nu [erg/s/Hz] on the UNCHANGED
native 1221-point wavelength grid (no resampling: M-B in the parity audit
found that regridding narrow spectral features onto a coarser common grid
loses amplitude, so this vendoring keeps the tabulated grid node-exact), and
writes ``data/agnfitter_galaxy_reference.h5`` with two groups::

    python scripts/build_agnfitter_galaxy_reference.py
    git add -f data/agnfitter_galaxy_reference.h5   # data/*.h5 is gitignored

HDF5 schema
-----------
``/bc03_840`` (single metallicity, 30 ages x 28 tau):

===============  =================  ==========================================
Dataset          Shape              Description
===============  =================  ==========================================
``tau_axis``     ``(28,)``          e-folding timescale [Gyr], ascending
``age_axis``     ``(30,)``          stellar population age [yr], ascending
``wavelength_aa``  ``(1221,)``      wavelength [Angstrom], ascending (native)
``sed``          ``(28, 30, 1221)``  L_nu [erg/s/Hz] at (tau, age, wavelength)
``sfr``          ``(28, 30)``       instantaneous SFR [Msun/yr] at (tau, age)
===============  =================  ==========================================

``/bc03_metal`` (4 metallicities, 20 ages x 18 tau) additionally carries
``metal_axis`` ``(4,)`` [Z/Zsun] and its ``sed``/``sfr`` datasets carry a
leading metallicity axis: ``sed`` ``(4, 18, 20, 1221)``, ``sfr`` ``(4, 18, 20)``.

Both groups' attrs record the upstream source pickle's basename and its
SHA-256 (the file this HDF5 was built from), so a re-vendor is traceable.

Safety -- pickle.load on external data
---------------------------------------
Uses a restricted :class:`pickle.Unpickler` allow-listing only the classes an
opcode-level ``pickletools.dis`` scan of both pickles actually references:
NumPy array primitives, and ``astropy.units`` primitives (``Quantity``,
``Unit``, ``CompositeUnit``, ``IrreducibleUnit``, ``PrefixUnit``,
``_recreate_irreducible_unit``) -- these two pickles are plain dicts of
Quantity arrays, not pandas DataFrames, so no pandas classes are allow-listed
here (a narrower list than the disk/torus builders, which do need pandas).
Any other GLOBAL raises ``UnpicklingError``.

References
----------
.. [1] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
.. [2] G. Bruzual & S. Charlot, "Stellar population synthesis at the
   resolution of 2003," MNRAS, 344, 1000 (2003).
   doi:10.1046/j.1365-8711.2003.06897.x.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tengri.utils.physics_constants import C_AA, L_SUN

# Allow-list derived from a preflight `pickletools.dis` opcode scan of both
# GALAXY pickles (probes/task6/probe_bc03_pickle_inspect.py during
# development): every GLOBAL opcode either file references, and nothing else.
_SAFE_CLASSES: frozenset[tuple[str, str]] = frozenset(
    {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("astropy.units.quantity", "Quantity"),
        ("astropy.units.core", "Unit"),
        ("astropy.units.core", "CompositeUnit"),
        ("astropy.units.core", "_recreate_irreducible_unit"),
        ("astropy.units.core", "IrreducibleUnit"),
        ("astropy.units.core", "PrefixUnit"),
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler limited to numpy array + astropy.units primitives."""

    def find_class(self, module: str, name: str):
        if (module, name) not in _SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to import {module}.{name}: not in safe allow-list. "
                "If this is a legitimate numpy/astropy.units primitive, add it "
                "to _SAFE_CLASSES in build_agnfitter_galaxy_reference.py after "
                "re-running the pickletools.dis opcode scan."
            )
        return getattr(importlib.import_module(module), name)


def _load(pickle_path: Path) -> dict:
    """Load one GALAXY pickle through the restricted unpickler."""
    with pickle_path.open("rb") as fh:
        return _RestrictedUnpickler(fh, encoding="latin1").load()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lambda_to_nu_lnu(wave_aa: np.ndarray, sed_lsun_per_aa: np.ndarray) -> np.ndarray:
    """F_lambda [Lsun/Angstrom] -> L_nu [erg/s/Hz], broadcasting over the last axis.

    ``L_nu = L_lambda * lambda^2 / c`` (both in Angstrom / erg-s units), the
    standard flux-density Jacobian; no resampling, no AGNfitter-internal
    cosmetic rescaling is applied here (that lives only in the AGNfitter MCMC
    bookkeeping and is irrelevant to a physical L_nu vendoring).

    Parameters
    ----------
    wave_aa : ndarray, shape (n_wave,)
        Wavelength [Angstrom], ascending.
    sed_lsun_per_aa : ndarray, shape (..., n_wave)
        F_lambda [Lsun/Angstrom] on the last axis.

    Returns
    -------
    ndarray, shape (..., n_wave)
        L_nu [erg/s/Hz].
    """
    lam2_over_c = (wave_aa.astype(np.float64) ** 2) / C_AA  # [s]
    return sed_lsun_per_aa.astype(np.float64) * L_SUN * lam2_over_c


def _convert_bc03_840(pickle_path: Path) -> dict[str, np.ndarray]:
    d = _load(pickle_path)
    wave_aa = np.asarray(d["wavelength"].to_value("Angstrom"), dtype=np.float64)
    tau_axis = np.asarray(d["tau-values"].to_value("Gyr"), dtype=np.float64)
    age_axis = np.asarray(d["age-values"].to_value("yr"), dtype=np.float64)
    # SED/SFR native shape: (metal=1, age, tau, dustextinct=1, fesc=1[, wave]).
    sed = np.asarray(d["SED"].to_value("solLum / Angstrom"), dtype=np.float64)
    sed = sed[0, :, :, 0, 0, :]  # -> (age, tau, wave)
    sed = np.moveaxis(sed, 0, 1)  # -> (tau, age, wave)
    sfr = np.asarray(d["SFR"].to_value("solMass / yr"), dtype=np.float64)
    sfr = sfr[0, :, :, 0, 0]  # -> (age, tau)
    sfr = np.moveaxis(sfr, 0, 1)  # -> (tau, age)
    L_nu = _lambda_to_nu_lnu(wave_aa, sed)
    return {
        "tau_axis": tau_axis,
        "age_axis": age_axis,
        "wavelength_aa": wave_aa,
        "sed": L_nu,
        "sfr": sfr,
    }


def _convert_bc03_metal(pickle_path: Path) -> dict[str, np.ndarray]:
    d = _load(pickle_path)
    wave_aa = np.asarray(d["wavelength"].to_value("Angstrom"), dtype=np.float64)
    tau_axis = np.asarray(d["tau-values"].to_value("Gyr"), dtype=np.float64)
    age_axis = np.asarray(d["age-values"].to_value("yr"), dtype=np.float64)
    metal_axis = np.asarray(d["metallicity-values"], dtype=np.float64)
    # Native shape: (metal, age, tau, dustextinct=1, fesc=1[, wave]).
    sed = np.asarray(d["SED"].to_value("solLum / Angstrom"), dtype=np.float64)
    sed = sed[:, :, :, 0, 0, :]  # -> (metal, age, tau, wave)
    sed = np.moveaxis(sed, 1, 2)  # -> (metal, tau, age, wave)
    sfr = np.asarray(d["SFR"].to_value("solMass / yr"), dtype=np.float64)
    sfr = sfr[:, :, :, 0, 0]  # -> (metal, age, tau)
    sfr = np.moveaxis(sfr, 1, 2)  # -> (metal, tau, age)
    L_nu = _lambda_to_nu_lnu(wave_aa, sed)
    return {
        "metal_axis": metal_axis,
        "tau_axis": tau_axis,
        "age_axis": age_axis,
        "wavelength_aa": wave_aa,
        "sed": L_nu,
        "sfr": sfr,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--bc03-840",
        type=Path,
        default=Path("/tmp/AGNfitter-rX/models/GALAXY/BC03_840seds.pickle"),
        help="Path to the upstream BC03_840seds.pickle.",
    )
    ap.add_argument(
        "--bc03-metal",
        type=Path,
        default=Path("/tmp/AGNfitter-rX/models/GALAXY/BC03_seds_metal_medium.pickle"),
        help="Path to the upstream BC03_seds_metal_medium.pickle.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "agnfitter_galaxy_reference.h5",
        help="Output HDF5 path.",
    )
    args = ap.parse_args()

    with h5py.File(args.output, "w") as f:
        f.attrs["source"] = "AGNfitter-rX models/GALAXY"

        if args.bc03_840.is_file():
            data = _convert_bc03_840(args.bc03_840)
            g = f.create_group("bc03_840")
            g.attrs["upstream_file"] = args.bc03_840.name
            g.attrs["upstream_sha256"] = _sha256(args.bc03_840)
            g.attrs["sed_unit"] = "erg/s/Hz"
            g.attrs["sfr_unit"] = "Msun/yr"
            g.attrs["axes"] = "sed[tau, age, wavelength]; sfr[tau, age]"
            for k, v in data.items():
                g.create_dataset(k, data=v.astype(np.float64), compression="gzip")
            print(f"  bc03_840: sed {data['sed'].shape}, sfr {data['sfr'].shape}")
        else:
            print(f"  (skipped bc03_840: {args.bc03_840} not found)")

        if args.bc03_metal.is_file():
            data = _convert_bc03_metal(args.bc03_metal)
            g = f.create_group("bc03_metal")
            g.attrs["upstream_file"] = args.bc03_metal.name
            g.attrs["upstream_sha256"] = _sha256(args.bc03_metal)
            g.attrs["sed_unit"] = "erg/s/Hz"
            g.attrs["sfr_unit"] = "Msun/yr"
            g.attrs["axes"] = "sed[metal, tau, age, wavelength]; sfr[metal, tau, age]"
            for k, v in data.items():
                g.create_dataset(k, data=v.astype(np.float64), compression="gzip")
            print(f"  bc03_metal: sed {data['sed'].shape}, sfr {data['sfr'].shape}")
        else:
            print(f"  (skipped bc03_metal: {args.bc03_metal} not found)")

    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()

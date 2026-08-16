#!/usr/bin/env python3
"""Build ``data/nenkova08_torus_grid.h5`` from the FSPS CLUMPY ``.dat`` file.

The Nenkova et al. (2008) CLUMPY dusty-torus library is shipped inside FSPS
(Conroy & Gunn 2010) as ``$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat``
— the same file consumed by Prospector (Johnson et al. 2021) for its AGN
torus SED. This script vendors those templates into a tengri-native HDF5 grid
that can be interpolated *differentiably* at runtime by
:mod:`tengri.components.agn.torus`.

Why vendor instead of read the ``.dat`` at runtime
---------------------------------------------------
The historical ``nenkova_torus`` read the ``.dat`` with ``numpy`` and
interpolated with ``scipy.interpolate.interp1d`` on every call, casting
``agn_tau`` through ``float()``. That makes the torus unusable inside
``jax.jit`` / ``jax.grad`` / ``jax.vmap`` — i.e. ``agn_tau`` could never be a
*fitted* parameter (MAP / NUTS / VI all trace it). Vendoring the grid lets the
runtime load it once and interpolate with a pure-JAX triweight kernel, exactly
like the SKIRTOR / Silva+04 / CAT3D paths.

Source file format (FSPS ``Nenkova08_y010_torusg_n10_q2.0.dat``)
----------------------------------------------------------------
- 3 ``#`` comment lines (units + column description).
- 1 header row of the 9 equatorial optical depths:
  ``5 10 20 30 40 60 80 100 150``.
- ``n_wave`` data rows, each ``lambda[Å]  f_nu*9`` (f_nu normalised to unity
  per the FSPS header, but re-normalised at runtime anyway).

HDF5 schema
-----------
``/nenkova``

======================  ===================  =========================================
Dataset                  Shape                Description
======================  ===================  =========================================
``tau_axis``             ``(n_tau,)``         equatorial optical depth, ascending
``wavelength``           ``(n_wave,)``        wavelength grid [Å], ascending
``template``             ``(n_tau, n_wave)``  F_nu template (relative; per-L_sun
                                              normalised at runtime)
======================  ===================  =========================================

Port credit
-----------
Templates from Nenkova et al. 2008 (ApJ 685, 147 & 160), as reformatted and
distributed by FSPS (Conroy, Gunn & White 2009; Conroy & Gunn 2010) and used
by Prospector (Johnson et al. 2021).

Usage
-----
::

    python scripts/build_nenkova_grid.py \\
        --input "$SPS_HOME/dust/Nenkova08_y010_torusg_n10_q2.0.dat" \\
        --output data/nenkova08_torus_grid.h5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

# Equatorial optical depths tabulated in the FSPS CLUMPY file header.
_TAU_AXIS: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0)


def _default_input() -> Path:
    """Locate the FSPS CLUMPY ``.dat`` via ``$SPS_HOME`` (or ``~/Projects/fsps``)."""
    sps_home = os.environ.get("SPS_HOME") or os.path.expanduser("~/Projects/fsps")
    return Path(sps_home) / "dust" / "Nenkova08_y010_torusg_n10_q2.0.dat"


def build(input_dat: Path, output_h5: Path) -> None:
    """Read the CLUMPY ``.dat`` and emit tengri's ``nenkova08_torus_grid.h5``."""
    if not input_dat.is_file():
        raise FileNotFoundError(
            f"Nenkova+2008 CLUMPY file not found at {input_dat}. "
            "Set $SPS_HOME to your FSPS install or pass --input."
        )

    # Row 4 is the tau header; data rows follow. Column 0 is wavelength [Å],
    # columns 1..9 are F_nu for the 9 optical depths.
    data = np.genfromtxt(input_dat, skip_header=4)
    wavelength_aa = np.asarray(data[:, 0], dtype=np.float64)
    fnu_grid = np.asarray(data[:, 1:], dtype=np.float64)  # (n_wave, n_tau)

    tau_axis = np.asarray(_TAU_AXIS, dtype=np.float64)
    if fnu_grid.shape[1] != tau_axis.size:
        raise ValueError(
            f"Expected {tau_axis.size} optical-depth columns, "
            f"found {fnu_grid.shape[1]} in {input_dat}."
        )

    # Ascending wavelength; transpose to (n_tau, n_wave) for the runtime loader.
    order = np.argsort(wavelength_aa)
    wavelength_aa = wavelength_aa[order]
    template = fnu_grid[order].T.copy()  # (n_tau, n_wave)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group("nenkova")
        g.create_dataset("tau_axis", data=tau_axis, compression="gzip")
        g.create_dataset("wavelength", data=wavelength_aa, compression="gzip")
        g.create_dataset("template", data=template, compression="gzip")
        # Basename only: the absolute path is machine-specific and ships to the
        # public repo inside the committed grid. The filename is the part that
        # identifies the upstream template; the prefix only names someone's disk.
        g.attrs["source_file"] = Path(input_dat).name
        g.attrs["n_tau"] = tau_axis.size
        g.attrs["n_wave"] = wavelength_aa.size
        g.attrs["wavelength_unit"] = "Angstrom"
        g.attrs["template_unit"] = "F_nu (relative, per-L_sun normalised at runtime)"
        g.attrs["provenance"] = (
            "Nenkova et al. 2008 (ApJ 685, 147 & 160); reformatted by FSPS "
            "(Conroy & Gunn 2010); same file used by Prospector (Johnson+2021)."
        )

    print(
        f"wrote {output_h5} — {tau_axis.size} optical depths "
        f"× {wavelength_aa.size} wavelength points"
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--input",
        type=Path,
        default=_default_input(),
        help="Path to the FSPS Nenkova08_y010_torusg_n10_q2.0.dat file.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/nenkova08_torus_grid.h5"),
        help="Destination HDF5 path.",
    )
    args = p.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    _cli()

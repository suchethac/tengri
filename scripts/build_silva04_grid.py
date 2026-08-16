#!/usr/bin/env python3
"""Build ``data/silva04_torus_grid.h5`` from AGNfitter's ``S04.pickle``.

The Silva, Maiolino & Granato (2004) semi-empirical smooth-torus SED library
is shipped inside the AGNfitter repository as ``models/TORUS/S04.pickle``.
This script extracts those templates into a tengri-native HDF5 grid that
can be interpolated differentiably at runtime by
:mod:`tengri.components.agn.silva04`.

Source
------
The S04 pickle is structured (per ``functions/MODEL_AGNfitter.py`` in
AGNfitter v1/rX) as a dict with keys:

- ``Nh-values`` — per-bin hydrogen column density values (one float per bin).
- ``wavelength`` — per-bin array of ``log10(nu / Hz)``. Misleading key name:
  the values are log10 frequency, not wavelength. Converted here.
- ``SED`` — per-bin array of ``F_nu`` template values (relative, unnormalised).

Safety — pickle.load on external data
-------------------------------------
``pickle.load`` runs arbitrary code at deserialisation time. This script
uses a restricted :class:`pickle.Unpickler` whose ``find_class`` only returns
NumPy array constructors (``numpy.core.multiarray._reconstruct``,
``numpy.ndarray``, ``numpy.dtype``, plus the ``numpy._core.*`` aliases
introduced in NumPy 2). Any other ``GLOBAL`` opcode raises ``UnpicklingError``.

Before running, verify the pickle's opcode stream yourself::

    python -c "import pickletools; pickletools.dis(open('S04.pickle','rb'))"

and confirm no non-numpy GLOBALs appear. The script re-checks this during
load and aborts on anything unexpected.

HDF5 schema
-----------
``/silva04``

======================  ================  ==========================================
Dataset                  Shape             Description
======================  ================  ==========================================
``log_nh_axis``          ``(n_nh,)``       log10(N_H / cm^-2), ascending
``wavelength``           ``(n_wave,)``     common wavelength grid [Å], ascending
``template``             ``(n_nh, n_wave)``  F_nu template (unnormalised)
======================  ================  ==========================================

``template`` is per-L_sun-normalised in the same way tengri's SKIRTOR path
handles it — the runtime module divides the template by its trapezoidal
integral over frequency and multiplies by ``L_bol * agn_torus_frac``.

Reference
---------
Silva, Maiolino & Granato 2004, MNRAS 355, 973. Original paper ID and bib
entry should be cross-checked against `*(private paper draft)*`
before publication-time docstring references are written.

Port credit
-----------
Grid values repackaged from AGNfitter's published templates (Calistro Rivera et al. 2016, ApJ 833, 98).

Usage
-----
::

    git clone https://github.com/GabrielaCR/AGNfitter /tmp/AGNfitter
    python scripts/build_silva04_grid.py \\
        --input /tmp/AGNfitter/models/TORUS/S04.pickle \\
        --output data/silva04_torus_grid.h5
"""

from __future__ import annotations

import argparse
import io
import pickle
import pickletools
from pathlib import Path

import h5py
import numpy as np

_SAFE_CLASSES: frozenset[tuple[str, str]] = frozenset(
    {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }
)

_C_LIGHT_M_S = 2.99792458e8


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows numpy array primitives."""

    def find_class(self, module: str, name: str):
        if (module, name) not in _SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to import {module}.{name}: only numpy array constructors are allowed."
            )
        import importlib

        return getattr(importlib.import_module(module), name)


def _preflight_opcode_scan(pickle_path: Path) -> None:
    """Fail loudly if the pickle's opcode stream references anything non-numpy."""
    seen: set[tuple[str, str]] = set()
    with pickle_path.open("rb") as fh:
        out = io.StringIO()
        pickletools.dis(fh, annotate=0, out=out)
    for line in out.getvalue().splitlines():
        if "GLOBAL" not in line:
            continue
        try:
            qual = line.split("'", 1)[1].rsplit("'", 1)[0]
        except IndexError:
            continue
        parts = qual.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        mod, name = parts
        seen.add((mod, name))
    unexpected = seen - _SAFE_CLASSES
    if unexpected:
        raise RuntimeError(
            f"Unexpected GLOBAL references in {pickle_path}: {sorted(unexpected)}. "
            "Refusing to proceed."
        )


def _safe_load(pickle_path: Path) -> dict:
    """Safe-unpickle the S04 dict."""
    _preflight_opcode_scan(pickle_path)
    with pickle_path.open("rb") as fh:
        obj = _RestrictedUnpickler(fh, encoding="latin1").load()
    if not isinstance(obj, dict):
        raise TypeError(f"S04 pickle root is {type(obj).__name__}, expected dict.")
    return obj


def _log_nu_to_wavelength_angstrom(log_nu_hz: np.ndarray) -> np.ndarray:
    """Convert AGNfitter's ``log10(nu / Hz)`` axis to wavelength [Å] (ascending)."""
    nu_hz = 10.0 ** np.asarray(log_nu_hz, dtype=np.float64)
    wavelength_m = _C_LIGHT_M_S / nu_hz
    return wavelength_m * 1e10


def _regrid_templates(
    per_bin_log_nu: list[np.ndarray],
    per_bin_fnu: list[np.ndarray],
    *,
    n_wave: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample all Nh bins onto a common wavelength grid [Å], ascending."""
    per_bin_wave = [_log_nu_to_wavelength_angstrom(ln) for ln in per_bin_log_nu]
    w_min = max(float(w.min()) for w in per_bin_wave)
    w_max = min(float(w.max()) for w in per_bin_wave)
    common = np.geomspace(w_min, w_max, n_wave)
    out = np.empty((len(per_bin_fnu), n_wave), dtype=np.float64)
    for i, (w, f) in enumerate(zip(per_bin_wave, per_bin_fnu, strict=True)):
        order = np.argsort(w)
        out[i] = np.interp(common, w[order], np.asarray(f, dtype=np.float64)[order])
    return common, out


def build(input_pickle: Path, output_h5: Path, n_wave: int = 4096) -> None:
    """Read S04.pickle and emit tengri's ``silva04_torus_grid.h5``."""
    d = _safe_load(input_pickle)

    missing = {"Nh-values", "wavelength", "SED"} - set(d.keys())
    if missing:
        raise KeyError(f"S04 pickle missing expected keys: {sorted(missing)}")

    log_nh_values = np.asarray(d["Nh-values"], dtype=np.float64).ravel()
    n_nh = log_nh_values.size
    per_bin_log_nu = [np.asarray(d["wavelength"][i]).ravel() for i in range(n_nh)]
    per_bin_fnu = [np.asarray(d["SED"][i]).ravel() for i in range(n_nh)]

    order = np.argsort(log_nh_values)
    log_nh_values = log_nh_values[order]
    per_bin_log_nu = [per_bin_log_nu[i] for i in order]
    per_bin_fnu = [per_bin_fnu[i] for i in order]

    wavelength_aa, template = _regrid_templates(per_bin_log_nu, per_bin_fnu, n_wave=n_wave)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group("silva04")
        g.create_dataset("log_nh_axis", data=log_nh_values, compression="gzip")
        g.create_dataset("wavelength", data=wavelength_aa, compression="gzip")
        g.create_dataset("template", data=template, compression="gzip")
        # Basename only: the absolute path is machine-specific and ships to the
        # public repo inside the committed grid. The filename is the part that
        # identifies the upstream template; the prefix only names someone's disk.
        g.attrs["source_pickle"] = Path(input_pickle).name
        g.attrs["n_nh"] = n_nh
        g.attrs["n_wave"] = n_wave
        g.attrs["wavelength_unit"] = "Angstrom"
        g.attrs["template_unit"] = "F_nu (relative, per-L_sun normalised at runtime)"

    print(f"wrote {output_h5} — {n_nh} N_H bins × {n_wave} wavelength points")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--input",
        type=Path,
        default=Path("/tmp/AGNfitter/models/TORUS/S04.pickle"),
        help="Path to AGNfitter's S04.pickle.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/silva04_torus_grid.h5"),
        help="Destination HDF5 path.",
    )
    p.add_argument(
        "--n-wave",
        type=int,
        default=4096,
        help="Size of the common wavelength grid after regridding.",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Fetch S04.pickle from the AGNfitter GitHub repo (pinned tag) "
        "instead of reading --input. No AGNfitter install needed.",
    )
    args = p.parse_args()
    from _agnfitter_download import resolve

    input_pickle = resolve(args.input, "models/TORUS/S04.pickle", download=args.download)
    build(input_pickle, args.output, n_wave=args.n_wave)


if __name__ == "__main__":
    _cli()

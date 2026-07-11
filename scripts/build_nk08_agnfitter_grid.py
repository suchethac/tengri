#!/usr/bin/env python3
"""Build ``data/nenkova_agnfitter_torus_grid.h5`` from AGNfitter-rX's ``NK0_mean_1p.pickle``.

The Nenkova et al. (2008) CLUMPY inclination-averaged torus SED library, as
published with AGNfitter-rX (Martínez-Ramírez et al. 2024), is stored inside
the AGNfitter-rX repository as ``models/TORUS/NK0_mean_1p.pickle``. This script
extracts those templates into a tengri-native HDF5 grid that can be interpolated
differentiably at runtime by :mod:`tengri.components.agn.nenkova_agnfitter`.

Source
------
The NK0_mean_1p pickle is structured (per ``functions/MODEL_AGNfitter.py`` in
AGNfitter-rX) as a dict with keys:

- ``incl-values`` — per-inclination angle values (one float per bin) in degrees.
- ``wavelength`` — per-inclination array of ``log10(nu / Hz)``. Misleading key name:
  the values are log10 frequency, not wavelength. Converted here.
- ``SED`` — per-inclination array of ``F_nu`` template values (relative, unnormalised).

Safety — pickle.load on external data
-------------------------------------
``pickle.load`` runs arbitrary code at deserialisation time. This script
uses a restricted :class:`pickle.Unpickler` whose ``find_class`` only returns
NumPy array constructors (``numpy.core.multiarray._reconstruct``,
``numpy.ndarray``, ``numpy.dtype``, plus the ``numpy._core.*`` aliases
introduced in NumPy 2). Any other ``GLOBAL`` opcode raises ``UnpicklingError``.

Before running, verify the pickle's opcode stream yourself::

    python -c "import pickletools; pickletools.dis(open('NK0_mean_1p.pickle','rb'))"

and confirm no non-numpy GLOBALs appear. The script re-checks this during
load and aborts on anything unexpected.

HDF5 schema
-----------
``/nenkova_agnfitter``

============================  ================  ==========================================
Dataset                       Shape             Description
============================  ================  ==========================================
``incl_axis``                 ``(n_incl,)``     Inclination [deg], ascending
``wavelength``                ``(n_wave,)``     common wavelength grid [Å], ascending
``template``                  ``(n_incl, n_wave)``  F_nu template (unnormalised)
============================  ================  ==========================================

``template`` is per-L_sun-normalised at runtime by :mod:`tengri.components.agn.nenkova_agnfitter`
using the same approach as silva04 — the runtime module divides the template by its
trapezoidal integral over frequency and multiplies by ``L_bol * agn_torus_frac``.

Reference
---------
Nenkova, M. et al. 2008, ApJ 685, 160. DOI: 10.1088/0004-637X/685/1/160.
Martínez-Ramírez, L. N. et al. 2024, A&A 688, A46. DOI: 10.1051/0004-6361/202449329.

Port credit
-----------
Grid values extracted from AGNfitter-rX's published templates.

Usage
-----
::

    git clone https://github.com/GabrielaCR/AGNfitter /tmp/AGNfitter-rX
    python scripts/build_nk08_agnfitter_grid.py \\
        --input /tmp/AGNfitter-rX/models/TORUS/NK0_mean_1p.pickle \\
        --output data/nenkova_agnfitter_torus_grid.h5
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
        ("_codecs", "encode"),  # used for numpy string encoding
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
    """Safe-unpickle the NK0 dict."""
    _preflight_opcode_scan(pickle_path)
    with pickle_path.open("rb") as fh:
        obj = _RestrictedUnpickler(fh, encoding="latin1").load()
    if not isinstance(obj, dict):
        raise TypeError(f"NK0 pickle root is {type(obj).__name__}, expected dict.")
    return obj


def _log_nu_to_wavelength_angstrom(log_nu_hz: np.ndarray) -> np.ndarray:
    """Convert AGNfitter's ``log10(nu / Hz)`` axis to wavelength [Å] (ascending)."""
    nu_hz = 10.0 ** np.asarray(log_nu_hz, dtype=np.float64)
    wavelength_m = _C_LIGHT_M_S / nu_hz
    return wavelength_m * 1e10


def _regrid_templates(
    per_incl_log_nu: list[np.ndarray],
    per_incl_fnu: list[np.ndarray],
    *,
    n_wave: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample all inclination bins onto a common wavelength grid [Å], ascending."""
    per_incl_wave = [_log_nu_to_wavelength_angstrom(ln) for ln in per_incl_log_nu]
    w_min = max(float(w.min()) for w in per_incl_wave)
    w_max = min(float(w.max()) for w in per_incl_wave)
    common = np.geomspace(w_min, w_max, n_wave)
    out = np.empty((len(per_incl_fnu), n_wave), dtype=np.float64)
    for i, (w, f) in enumerate(zip(per_incl_wave, per_incl_fnu, strict=True)):
        order = np.argsort(w)
        out[i] = np.interp(common, w[order], np.asarray(f, dtype=np.float64)[order])
    return common, out


def build(input_pickle: Path, output_h5: Path, n_wave: int = 4096) -> None:
    """Read NK0_mean_1p.pickle and emit tengri's ``nenkova_agnfitter_torus_grid.h5``."""
    d = _safe_load(input_pickle)

    missing = {"incl-values", "wavelength", "SED"} - set(d.keys())
    if missing:
        raise KeyError(f"NK0 pickle missing expected keys: {sorted(missing)}")

    incl_values = np.asarray(d["incl-values"], dtype=np.float64).ravel()
    n_incl = incl_values.size
    per_incl_log_nu = [np.asarray(d["wavelength"][i]).ravel() for i in range(n_incl)]
    per_incl_fnu = [np.asarray(d["SED"][i]).ravel() for i in range(n_incl)]

    order = np.argsort(incl_values)
    incl_values = incl_values[order]
    per_incl_log_nu = [per_incl_log_nu[i] for i in order]
    per_incl_fnu = [per_incl_fnu[i] for i in order]

    wavelength_aa, template = _regrid_templates(per_incl_log_nu, per_incl_fnu, n_wave=n_wave)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group("nenkova_agnfitter")
        g.create_dataset("incl_axis", data=incl_values, compression="gzip")
        g.create_dataset("wavelength", data=wavelength_aa, compression="gzip")
        g.create_dataset("template", data=template, compression="gzip")
        g.attrs["source_pickle"] = str(input_pickle)
        g.attrs["n_incl"] = n_incl
        g.attrs["n_wave"] = n_wave
        g.attrs["wavelength_unit"] = "Angstrom"
        g.attrs["template_unit"] = "F_nu (relative, per-L_sun normalised at runtime)"

    print(f"wrote {output_h5} — {n_incl} inclination bins × {n_wave} wavelength points")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--input",
        type=Path,
        default=Path("/tmp/AGNfitter-rX/models/TORUS/NK0_mean_1p.pickle"),
        help="Path to AGNfitter-rX's NK0_mean_1p.pickle.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/nenkova_agnfitter_torus_grid.h5"),
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
        help="Fetch NK0_mean_1p.pickle from the AGNfitter-rX GitHub repo (pinned tag) "
        "instead of reading --input. No AGNfitter install needed.",
    )
    args = p.parse_args()
    from _agnfitter_download import resolve

    input_pickle = resolve(args.input, "models/TORUS/NK0_mean_1p.pickle", download=args.download)
    build(input_pickle, args.output, n_wave=args.n_wave)


if __name__ == "__main__":
    _cli()

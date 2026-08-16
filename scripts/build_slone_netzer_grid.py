#!/usr/bin/env python3
"""Build ``data/slone_netzer_disc_grid.h5`` from AGNfitter-rX's SN12 pickle.

The Slone & Netzer (2012) alpha-disc library ships inside AGNfitter-rX
(branch ``AGNfitter-rX_v0.1``) at ``models/BBB/SN12.pickle``. The pickle is a
dict of numpy arrays:

- ``logBHmass-values`` — black-hole mass axis, ``log10(M_BH / M_sun)`` (9 values,
  7.4 .. 9.8).
- ``logEddra-values``  — accretion-rate label list, ``log10(Mdot / Mdot_Edd)``.
- ``SED``              — ``(n_freq, n_mbh, n_edd)`` = ``(375, 9, 12)`` F_nu
  templates [erg/s/Hz].
- ``frequency``        — frequency axis [Hz] (375 values).

Eddington axis caveat (carried faithfully from AGNfitter-rX)
------------------------------------------------------------
The ``SED`` array stores **12** accretion-rate columns, but
``logEddra-values`` holds **259** entries. AGNfitter-rX's loader
(``MODEL_AGNfitter.BBB``) labels column ``j`` with ``logEddra-values[j]`` —
i.e. it uses the **first 12** entries, ``[-4.0 .. -1.96]`` (monotonic with the
columns' monotonically rising luminosity). We reproduce that exact labelling so
this grid matches what AGNfitter-rX feeds its fitter; the upstream 259-vs-12
mismatch is an AGNfitter-rX inconsistency, not introduced here.

Safety — pickle.load on external data
-------------------------------------
Uses a restricted :class:`pickle.Unpickler` allow-listing only numpy/pandas/
basic-builtin primitives, with a preflight opcode scan — the same hardening as
``scripts/build_cat3d_wind_grid.py``.

HDF5 schema
-----------
``/slone_netzer``

==============  ======================  ==================================
Dataset         Shape                   Description
==============  ======================  ==================================
``log_mbh``     ``(n_mbh,)``            log10(M_BH / M_sun), ascending
``log_edd``     ``(n_edd,)``            log10(Mdot / Mdot_Edd), ascending
``wavelength``  ``(n_wave,)``           common wavelength grid [Å], ascending
``template``    ``(n_mbh, n_edd, n_wave)``  F_nu template (unnormalised)
==============  ======================  ==================================

References
----------
- Slone, O. & Netzer, H., "The effect of disc winds on the optical and
  ultraviolet emission lines of active galactic nuclei," MNRAS 426, 656 (2012).
- Martinez-Ramirez et al. 2024, A&A 688, A46 (AGNfitter-rX).

Usage
-----
::

    python scripts/build_slone_netzer_grid.py \\
        --input /tmp/AGNfitter-rX/models/BBB/SN12.pickle \\
        --output data/slone_netzer_disc_grid.h5
"""

from __future__ import annotations

import argparse
import importlib
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
        ("builtins", "slice"),
        ("_codecs", "encode"),
    }
)
_PY2_MODULE_ALIASES: dict[str, str] = {"__builtin__": "builtins"}
_C_AA_PER_S = 2.99792458e18  # speed of light [Å·Hz]
_N_EDD_COLUMNS = 12  # SN12 SED stores 12 accretion-rate columns


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler allow-listed to numpy / basic builtins."""

    def find_class(self, module: str, name: str):
        module = _PY2_MODULE_ALIASES.get(module, module)
        if (module, name) not in _SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Refusing to import {module}.{name}: not in the safe allow-list."
            )
        return getattr(importlib.import_module(module), name)


def _preflight_opcode_scan(pickle_path: Path) -> None:
    """Abort if any GLOBAL reference is outside the allow-list."""
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
        mod = _PY2_MODULE_ALIASES.get(mod, mod)
        seen.add((mod, name))
    unexpected = seen - _SAFE_CLASSES
    if unexpected:
        raise RuntimeError(
            f"Unexpected GLOBAL references in {pickle_path}: {sorted(unexpected)}. "
            "Refusing to proceed."
        )


def _safe_load(pickle_path: Path):
    _preflight_opcode_scan(pickle_path)
    with pickle_path.open("rb") as fh:
        return _RestrictedUnpickler(fh, encoding="latin1").load()


def build(input_pickle: Path, output_h5: Path, n_wave: int = 2048) -> None:
    """Read SN12.pickle and emit tengri's ``slone_netzer_disc_grid.h5``."""
    d = _safe_load(input_pickle)
    for key in ("logBHmass-values", "logEddra-values", "SED", "frequency"):
        if key not in d:
            raise KeyError(f"SN12 pickle missing key '{key}'.")

    sed = np.asarray(d["SED"], dtype=np.float64)  # (n_freq, n_mbh, n_edd)
    if sed.ndim != 3 or sed.shape[2] != _N_EDD_COLUMNS:
        raise RuntimeError(
            f"SN12 SED has unexpected shape {sed.shape}; expected (n_freq, n_mbh, 12)."
        )
    log_mbh = np.asarray(d["logBHmass-values"], dtype=np.float64).ravel()
    # Faithful to AGNfitter-rX: column j is labelled logEddra-values[j].
    log_edd = np.asarray(d["logEddra-values"], dtype=np.float64).ravel()[:_N_EDD_COLUMNS]
    freq = np.asarray(d["frequency"], dtype=np.float64).ravel()
    wave_native = _C_AA_PER_S / freq  # Å (descending as freq ascends)

    if sed.shape[1] != log_mbh.size:
        raise RuntimeError(
            f"SN12 M_BH axis mismatch: SED has {sed.shape[1]}, labels {log_mbh.size}."
        )

    # Common ascending wavelength grid over the native template extent.
    w_min = float(wave_native.min())
    w_max = float(wave_native.max())
    common_wave = np.geomspace(w_min, w_max, n_wave)

    order = np.argsort(wave_native)
    wave_sorted = wave_native[order]
    template = np.zeros((log_mbh.size, log_edd.size, n_wave), dtype=np.float64)
    for i in range(log_mbh.size):
        for j in range(log_edd.size):
            fnu = sed[:, i, j][order]
            template[i, j] = np.interp(common_wave, wave_sorted, fnu)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        g = f.create_group("slone_netzer")
        g.create_dataset("log_mbh", data=log_mbh, compression="gzip")
        g.create_dataset("log_edd", data=log_edd, compression="gzip")
        g.create_dataset("wavelength", data=common_wave, compression="gzip")
        g.create_dataset("template", data=template, compression="gzip")
        # Basename only: the absolute path is machine-specific and ships to the
        # public repo inside the committed grid. The filename is the part that
        # identifies the upstream template; the prefix only names someone's disk.
        g.attrs["source_pickle"] = Path(input_pickle).name
        g.attrs["n_mbh"] = log_mbh.size
        g.attrs["n_edd"] = log_edd.size
        g.attrs["n_wave"] = n_wave
        g.attrs["wavelength_unit"] = "Angstrom"
        g.attrs["template_unit"] = "F_nu erg/s/Hz (renormalised at runtime)"
        g.attrs["edd_labelling"] = "logEddra-values[:12] (AGNfitter-rX convention)"

    print(
        f"wrote {output_h5} — {log_mbh.size} M_BH × {log_edd.size} Edd × "
        f"{n_wave} wavelengths; "
        f"log_mbh [{log_mbh.min():.2f}, {log_mbh.max():.2f}], "
        f"log_edd [{log_edd.min():.2f}, {log_edd.max():.2f}]"
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--input",
        type=Path,
        default=Path("/tmp/AGNfitter-rX/models/BBB/SN12.pickle"),
        help="Path to AGNfitter-rX's SN12.pickle.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/slone_netzer_disc_grid.h5"),
        help="Destination HDF5 path.",
    )
    p.add_argument("--n-wave", type=int, default=2048, help="Common wavelength grid size.")
    p.add_argument(
        "--download",
        action="store_true",
        help="Fetch SN12.pickle from the AGNfitter GitHub repo (pinned tag) "
        "instead of reading --input. No AGNfitter install needed.",
    )
    args = p.parse_args()
    from _agnfitter_download import resolve

    input_pickle = resolve(args.input, "models/BBB/SN12.pickle", download=args.download)
    build(input_pickle, args.output, n_wave=args.n_wave)


if __name__ == "__main__":
    _cli()

#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Convert AGNfitter-rX BBB (accretion-disc) reference pickles to a vendored HDF5.

AGNfitter-rX ships its accretion-disc (Big Blue Bump) templates as Python
pickles under ``models/BBB/`` (R06.pickle, KD18.pickle, THB21.pickle). The
crossval tests must NOT load pickle at test time — they read this committed
HDF5 instead, so the tests run in CI rather than skipping when the upstream
clone is absent (closes the data-gated-tests-mask-regressions gap, #613).

This is the build-time converter: it reads the upstream pickles ONCE (via a
restricted unpickler with a pandas/numpy allow-list), regrids every template to
a per-model common wavelength axis, and writes ``data/agnfitter_bbb_reference.h5``
with one group per model. Run after cloning AGNfitter-rX to ``/tmp/AGNfitter-rX``::

    python scripts/build_agnfitter_bbb_reference.py
    git add -f data/agnfitter_bbb_reference.h5   # data/*.h5 is gitignored

Convention (verified against MODEL_AGNfitter.py BBB()):
  * pickle ``wavelength`` column / key = log10(nu / Hz)
  * pickle ``SED`` = F_nu (relative units); reddening (BBBred_Prevot) is applied
    separately downstream, so the stored template is the unreddened EBV=0 SED.

The HDF5 stores wavelength in Angstrom (ascending) and SED in F_nu. Grid models
(KD18) additionally store their parameter axes.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

# Allow standalone invocation (``python scripts/build_agnfitter_bbb_reference.py``)
# to import the package's physics constants rather than redefining them.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tengri.utils.physics_constants import C_AA

# Allow-list: the minimal numpy/pandas container set that round-trips AGNfitter's
# legacy-pandas pickles. Deliberately EXCLUDES ``new_block`` / ``_unpickle_block``
# so pandas takes the legacy ``Block`` path that accepts a ``slice`` placement —
# the newer typed path rejects the slice these old pickles store. ``functools.partial``
# is required by THB21.
_SAFE_CLASSES = frozenset(
    {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.series", "Series"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.internals.blocks", "Block"),
        ("pandas.core.arrays.numpy_", "PandasArray"),
        ("__builtin__", "slice"),
        ("builtins", "slice"),
        ("_codecs", "encode"),
        ("functools", "partial"),
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler limited to numpy/pandas container classes (no arbitrary code)."""

    def find_class(self, module: str, name: str):
        if (module, name) not in _SAFE_CLASSES:
            raise pickle.UnpicklingError(f"Blocked global during unpickle: {module}.{name}")
        if module == "__builtin__":
            import builtins

            return getattr(builtins, name)
        return super().find_class(module, name)


def _load(pickle_path: Path):
    """Load an AGNfitter pickle.

    Tries the restricted unpickler first. AGNfitter's DataFrame pickles were
    written across several pandas versions whose internal block formats are not
    forward-compatible with the plain unpickler; for those we fall back to
    ``pandas.read_pickle`` (which carries the version-compat shims). This is a
    BUILD-TIME-ONLY read of a trusted, developer-supplied AGNfitter clone — the
    shipped artefact is HDF5 and no test ever reads a pickle.
    """
    try:
        with pickle_path.open("rb") as fh:
            return _RestrictedUnpickler(fh, encoding="latin1").load()
    except (pickle.UnpicklingError, TypeError, ModuleNotFoundError) as exc:
        import pandas as pd

        print(
            f"  (restricted unpickle of {pickle_path.name} failed: {exc}; "
            "falling back to trusted pandas.read_pickle)"
        )
        return pd.read_pickle(pickle_path)


def _log_nu_to_aa(log_nu_hz: np.ndarray) -> np.ndarray:
    """log10(nu/Hz) -> wavelength [Å]."""
    return C_AA / (10.0 ** np.asarray(log_nu_hz, dtype=np.float64))


def _regrid(log_nu: np.ndarray, sed: np.ndarray, common_aa: np.ndarray) -> np.ndarray:
    """Interpolate a single (log_nu, F_nu) template onto a common Å axis."""
    wave = _log_nu_to_aa(log_nu)
    order = np.argsort(wave)
    return np.interp(common_aa, wave[order], np.asarray(sed, dtype=np.float64)[order])


def _common_axis(all_log_nu: list[np.ndarray], n_wave: int) -> np.ndarray:
    """Log-spaced Å axis spanning the intersection of every node's coverage."""
    lo = max(_log_nu_to_aa(np.max(x)) for x in all_log_nu)  # bluest common edge
    hi = min(_log_nu_to_aa(np.min(x)) for x in all_log_nu)  # reddest common edge
    return np.logspace(np.log10(lo), np.log10(hi), n_wave)


def _convert_single(d: dict, common_n: int) -> tuple[np.ndarray, np.ndarray]:
    """R06/THB21-style single-template dict -> (wavelength_aa, sed)."""
    log_nu = np.asarray(d["wavelength"], dtype=np.float64)
    sed = np.asarray(d["SED"], dtype=np.float64)
    common = _common_axis([log_nu], common_n)
    return common, _regrid(log_nu, sed, common)


def _convert_grid(df, axis_cols: list[str], common_n: int):
    """KD18-style DataFrame (per-row wavelength/SED + axis columns) -> grid."""
    axes = {c: df[c].to_numpy(dtype=np.float64) for c in axis_cols}
    log_nu_rows = [np.asarray(v, dtype=np.float64) for v in df["wavelength"].to_numpy()]
    common = _common_axis(log_nu_rows, common_n)
    sed = np.stack(
        [
            _regrid(ln, np.asarray(s, dtype=np.float64), common)
            for ln, s in zip(log_nu_rows, df["SED"].to_numpy())
        ]
    )
    return common, axes, sed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--src", default="/tmp/AGNfitter-rX/models/BBB", type=Path)
    ap.add_argument(
        "--out",
        default=Path(__file__).resolve().parents[1] / "data" / "agnfitter_bbb_reference.h5",
        type=Path,
    )
    ap.add_argument("--n-wave", default=1024, type=int)
    ap.add_argument(
        "--download",
        action="store_true",
        help="Fetch R06/THB21/KD18 pickles from the AGNfitter GitHub repo "
        "(pinned tag) instead of reading --src. No AGNfitter install needed.",
    )
    args = ap.parse_args()

    if args.download:
        from _agnfitter_download import fetch

        for name in ("R06.pickle", "THB21.pickle", "KD18.pickle"):
            args.src = fetch(f"models/BBB/{name}").parent

    with h5py.File(args.out, "w") as f:
        f.attrs["source"] = "AGNfitter-rX models/BBB"
        f.attrs["wavelength_unit"] = "Angstrom"
        f.attrs["sed_unit"] = "F_nu (relative, EBV=0)"

        # R06 — single empirical Type-1 quasar SED (dict)
        r06 = _load(args.src / "R06.pickle")
        wave, sed = _convert_single(r06, args.n_wave)
        g = f.create_group("r06")
        g.create_dataset("wavelength", data=wave.astype(np.float32), compression="gzip")
        g.create_dataset("sed", data=sed.astype(np.float32), compression="gzip")

        # THB21 — Temple+2021 empirical composite (DataFrame, single row; 'nu' col)
        thb = _load(args.src / "THB21.pickle")
        if "nu" in thb.columns and "wavelength" not in thb.columns:
            thb = thb.rename(columns={"nu": "wavelength"})
        cols = [c for c in thb.columns if c not in ("wavelength", "SED")]
        wave, _axes, sed = _convert_grid(thb, cols, args.n_wave)
        g = f.create_group("thb21")
        g.create_dataset("wavelength", data=wave.astype(np.float32), compression="gzip")
        g.create_dataset("sed", data=sed.astype(np.float32), compression="gzip")
        for k, v in _axes.items():
            g.create_dataset(k, data=v.astype(np.float32), compression="gzip")

        # KD18 — Kubota & Done 2018 disc grid over (logBHmass, logEddra)
        kd = _load(args.src / "KD18.pickle")
        wave, axes, sed = _convert_grid(kd, ["logBHmass", "logEddra"], args.n_wave)
        g = f.create_group("kd18")
        g.create_dataset("wavelength", data=wave.astype(np.float32), compression="gzip")
        g.create_dataset("sed", data=sed.astype(np.float32), compression="gzip")
        g.create_dataset(
            "logBHmass", data=axes["logBHmass"].astype(np.float32), compression="gzip"
        )
        g.create_dataset("logEddra", data=axes["logEddra"].astype(np.float32), compression="gzip")

        # SN12 — Slone & Netzer 2012 disc grid. Pickle is a dict with a single
        # SED cube (n_freq, n_mbh, n_edd) sharing one linear frequency axis,
        # NOT a per-row DataFrame, so it needs its own conversion.
        sn = _load(args.src / "SN12.pickle")
        sn_lognu = np.log10(np.asarray(sn["frequency"], dtype=np.float64))
        sn_mbh = np.asarray(sn["logBHmass-values"], dtype=np.float64).ravel()
        sn_edd = np.asarray(sn["logEddra-values"], dtype=np.float64).ravel()
        sn_cube = np.asarray(sn["SED"], dtype=np.float64)  # (n_freq, n_mbh, n_edd)
        n_mbh, n_edd = sn_cube.shape[1], sn_cube.shape[2]  # axis lists can be longer
        sn_mbh, sn_edd = sn_mbh[:n_mbh], sn_edd[:n_edd]
        common = _common_axis([sn_lognu], args.n_wave)
        sn_grid = np.stack(
            [
                np.stack([_regrid(sn_lognu, sn_cube[:, i, j], common) for j in range(n_edd)])
                for i in range(n_mbh)
            ]
        )  # (n_mbh, n_edd, n_wave)
        g = f.create_group("sn12")
        g.create_dataset("wavelength", data=common.astype(np.float32), compression="gzip")
        g.create_dataset("sed", data=sn_grid.astype(np.float32), compression="gzip")
        g.create_dataset("logBHmass", data=sn_mbh.astype(np.float32), compression="gzip")
        g.create_dataset("logEddra", data=sn_edd.astype(np.float32), compression="gzip")

    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out} ({size_kb:.0f} KB)")

    # ── Cold-dust (starburst) reference: DH02_CE01 ──────────────────────────
    # Dale & Helou 2002 + Chary & Elbaz 2001, an IR-luminosity grid. Lives in
    # models/STARBURST; stored 'wavelength' is already log10(nu/Hz) per node.
    cold_src = args.src.parent / "STARBURST" / "DH02_CE01.pickle"
    if cold_src.is_file():
        dh = _load(cold_src)
        dh_irlum = np.asarray(dh["irlum-values"], dtype=np.float64).ravel()
        dh_lognu_rows = [np.asarray(w, dtype=np.float64) for w in dh["wavelength"]]
        common_dh = _common_axis(dh_lognu_rows, args.n_wave)
        dh_grid = np.stack(
            [
                _regrid(ln, np.asarray(s, dtype=np.float64), common_dh)
                for ln, s in zip(dh_lognu_rows, dh["SED"])
            ]
        )  # (n_irlum, n_wave)
        cold_out = args.out.parent / "agnfitter_cold_dust_reference.h5"
        with h5py.File(cold_out, "w") as fc:
            fc.attrs["source"] = "AGNfitter-rX models/STARBURST"
            fc.attrs["wavelength_unit"] = "Angstrom"
            fc.attrs["sed_unit"] = "F_nu (relative)"
            g = fc.create_group("dh02_ce01")
            g.create_dataset("wavelength", data=common_dh.astype(np.float32), compression="gzip")
            g.create_dataset("sed", data=dh_grid.astype(np.float32), compression="gzip")
            g.create_dataset("irlum", data=dh_irlum.astype(np.float32), compression="gzip")
        print(f"Wrote {cold_out} ({cold_out.stat().st_size / 1024:.0f} KB)")
    else:
        print(f"  (skipped cold dust: {cold_src} not found)")


if __name__ == "__main__":
    main()

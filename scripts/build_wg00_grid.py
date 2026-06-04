#!/usr/bin/env python3
"""Build ``data/wg00_attenuation_grid.h5`` from the FSPS Witt & Gordon (2000) tables.

The Witt & Gordon (2000) Monte-Carlo radiative-transfer attenuation curves are
shipped inside FSPS (Conroy & Gunn 2010) as the two ASCII files
``$SPS_HOME/dust/alldirty_h.dat`` (homogeneous local density) and
``alldirty_c.dat`` (clumpy) — the tables FSPS reads for ``dust_type=3``. This
script vendors them into a tengri-native HDF5 grid that can be interpolated
*differentiably* at runtime by :mod:`tengri.components.dust.wg00`.

Why vendor instead of read the ``.dat`` at runtime
---------------------------------------------------
WG00 attenuation is not a fixed ``k(λ)`` scaled by ``τ_V``: the *shape* of the
curve depends on ``τ_V`` (high-τ sightlines self-shield → greyer). So a faithful
``dust_type=3`` needs the full ``A(λ; τ_V)`` table, interpolated in ``τ_V``. A
pure-JAX triweight kernel over the vendored grid makes ``τ_V`` a fully
differentiable, JIT/vmap-safe *fitted* parameter — matching the SKIRTOR /
Nenkova / Silva+04 paths.

Source file format (FSPS ``alldirty_h.dat`` / ``alldirty_c.dat``)
-----------------------------------------------------------------
- 3 ``#`` comment lines. The first names the columns::

    # lam, tauV, MW+dusty, MW+shell, MW+cloudy, SMC+dusty, SMC+shell, SMC+cloudy

- ``18 × 25`` data rows: 18 ``τ_V`` blocks (0.25–10.0) of 25 wavelengths each
  (1000–30001 Å). Each of the 6 value columns is the effective attenuation
  optical depth ``A(λ)`` (FSPS applies it as ``exp(-A)`` directly).
- ``alldirty_h.dat`` is the *homogeneous* local-density case; ``alldirty_c.dat``
  is *clumpy* (Witt & Gordon's two local structures).

The 6 value columns map to ``dust_curve ∈ {MW, SMC}`` (outer) ×
``geometry ∈ {dusty, shell, cloudy}`` (inner).

HDF5 schema
-----------
``/wg00``

======================  ===========================  ===================================
Dataset                  Shape                        Description
======================  ===========================  ===================================
``wavelength``           ``(n_wave,)``                wavelength grid [Å], ascending
``tau_v_axis``           ``(n_tau,)``                 V-band optical depth, ascending
``a_lambda``             ``(2, 2, 3, n_tau, n_wave)`` effective attenuation A(λ; τ_V),
                                                      axes (structure, dust, geometry, τ, λ)
======================  ===========================  ===================================

Axis label order (stored as ``a_lambda`` attrs):

- ``structure`` : ``["homogeneous", "clumpy"]``  (``alldirty_h`` / ``alldirty_c``)
- ``dust``      : ``["mw", "smc"]``
- ``geometry``  : ``["dusty", "shell", "cloudy"]``

Port credit
-----------
Curves from Witt & Gordon 2000 (ApJ 528, 799), as reformatted and distributed by
FSPS (Conroy, Gunn & White 2009; Conroy & Gunn 2010).

Usage
-----
::

    # from a local FSPS install
    python scripts/build_wg00_grid.py --input-dir "$SPS_HOME/dust"

    # or fetch the two tables straight from the FSPS GitHub mirror
    python scripts/build_wg00_grid.py --download
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

# Column order in the FSPS ``.dat`` files (after lam, tauV): dust (outer) × geometry (inner).
_DUST_LABELS: tuple[str, ...] = ("mw", "smc")
_GEOM_LABELS: tuple[str, ...] = ("dusty", "shell", "cloudy")
_STRUCT_LABELS: tuple[str, ...] = ("homogeneous", "clumpy")

# FSPS GitHub raw mirror of the two tables (used by --download).
_FSPS_RAW = "https://raw.githubusercontent.com/cconroy20/fsps/master/dust"
_FILES = {"homogeneous": "alldirty_h.dat", "clumpy": "alldirty_c.dat"}


def _default_input_dir() -> Path:
    """Locate the FSPS ``dust/`` directory via ``$SPS_HOME`` (or ``~/Projects/fsps``)."""
    sps_home = os.environ.get("SPS_HOME") or os.path.expanduser("~/Projects/fsps")
    return Path(sps_home) / "dust"


def _parse_table(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse one ``alldirty_*.dat`` table.

    Returns
    -------
    wavelength : ndarray, shape (n_wave,)
        Ascending wavelength grid [Å].
    tau_v : ndarray, shape (n_tau,)
        Ascending V-band optical depths.
    a_lambda : ndarray, shape (n_tau, n_wave, 6)
        Effective attenuation optical depth for the 6 (dust × geometry) columns.
    """
    rows = [
        [float(x) for x in line.split()]
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    data = np.asarray(rows, dtype=np.float64)
    lam_col, tau_col, val_cols = data[:, 0], data[:, 1], data[:, 2:]
    if val_cols.shape[1] != 6:
        raise ValueError(f"expected 6 value columns, got {val_cols.shape[1]}")

    tau_v = np.unique(tau_col)  # np.unique returns sorted
    wavelength = np.unique(lam_col)
    n_tau, n_wave = tau_v.shape[0], wavelength.shape[0]
    if n_tau * n_wave != data.shape[0]:
        raise ValueError(f"non-rectangular table: {n_tau}×{n_wave} != {data.shape[0]} rows")

    tau_idx = np.searchsorted(tau_v, tau_col)
    wave_idx = np.searchsorted(wavelength, lam_col)
    a_lambda = np.zeros((n_tau, n_wave, 6), dtype=np.float64)
    a_lambda[tau_idx, wave_idx, :] = val_cols
    return wavelength, tau_v, a_lambda


def _read_source(structure: str, input_dir: Path | None, download: bool) -> str:
    """Return the raw text of one WG00 table, from disk or the FSPS mirror."""
    filename = _FILES[structure]
    if download:
        import urllib.request

        url = f"{_FSPS_RAW}/{filename}"
        with urllib.request.urlopen(url) as resp:
            return resp.read().decode("utf-8")
    path = (input_dir or _default_input_dir()) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Point --input-dir at $SPS_HOME/dust, or pass --download "
            "to fetch the tables from the FSPS GitHub mirror."
        )
    return path.read_text(encoding="utf-8")


def build(output: Path, input_dir: Path | None = None, download: bool = False) -> None:
    """Build the vendored WG00 attenuation HDF5 grid."""
    per_struct: list[np.ndarray] = []
    wavelength = tau_v = None
    for structure in _STRUCT_LABELS:
        text = _read_source(structure, input_dir, download)
        wave_s, tau_s, a_s = _parse_table(text)
        if wavelength is None:
            wavelength, tau_v = wave_s, tau_s
        elif not (np.allclose(wave_s, wavelength) and np.allclose(tau_s, tau_v)):
            raise ValueError("homogeneous/clumpy tables have mismatched λ or τ_V axes")
        # (n_tau, n_wave, 6) → (n_tau, n_wave, dust=2, geom=3) → (dust, geom, n_tau, n_wave)
        n_tau, n_wave, _ = a_s.shape
        a_dg = a_s.reshape(n_tau, n_wave, len(_DUST_LABELS), len(_GEOM_LABELS))
        per_struct.append(np.transpose(a_dg, (2, 3, 0, 1)))

    # (structure=2, dust=2, geom=3, n_tau, n_wave)
    a_lambda = np.stack(per_struct, axis=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        g = f.create_group("wg00")
        g.create_dataset("wavelength", data=wavelength)
        g.create_dataset("tau_v_axis", data=tau_v)
        ds = g.create_dataset("a_lambda", data=a_lambda)
        ds.attrs["structure"] = list(_STRUCT_LABELS)
        ds.attrs["dust"] = list(_DUST_LABELS)
        ds.attrs["geometry"] = list(_GEOM_LABELS)
        ds.attrs["source"] = "Witt & Gordon 2000 (ApJ 528, 799); FSPS alldirty_{h,c}.dat"
    print(
        f"Wrote {output}  a_lambda{a_lambda.shape}  "
        f"τ_V[{tau_v[0]}–{tau_v[-1]}]  λ[{wavelength[0]:.0f}–{wavelength[-1]:.0f} Å]"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=None, help="FSPS dust/ directory")
    p.add_argument("--download", action="store_true", help="fetch tables from FSPS GitHub")
    p.add_argument("--output", type=Path, default=Path("data/wg00_attenuation_grid.h5"))
    args = p.parse_args()
    build(args.output, input_dir=args.input_dir, download=args.download)


if __name__ == "__main__":
    main()

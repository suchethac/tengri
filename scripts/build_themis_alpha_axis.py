#!/usr/bin/env python3
r"""Add a radiation-field-slope (alpha) axis to the FSPS-built THEMIS grid.

tengri ships the FSPS/DustEM-built THEMIS templates (``build_themis_from_fsps``,
PR #574): a single-U component and a power-law (PDR) component at the Jones+2017
default ``alpha = 2.0`` only. CIGALE additionally exposes ``alpha`` (the slope of
``dU/dM \propto U^{-alpha}``) over [1.0, 3.0]. This script augments the shipped
FSPS grid with an ``alpha`` axis WITHOUT changing the ``alpha = 2`` behaviour:

    powerlaw_alpha[q, u, k] = FSPS_powerlaw[q, u] * R(u, alpha_k)

    R(u, alpha) = < CIG_pa[:, u, alpha] / CIG_pa[:, u, alpha2] >_qhac

where ``CIG_pa`` is CIGALE's DustEM power-law (``model_minmax``) grid pulled from
``pcigale.data.SimpleDatabase('themis')`` (``umax = 1e7``). The ratio ``R`` is
the *relative* alpha-dependence of the PDR spectrum at fixed ``U_min``,
qhac-averaged (alpha reshapes the U distribution, which is essentially separable
from the a-C(:H) grain fraction). At ``alpha = 2`` the ratio is identically 1,
so ``powerlaw_alpha[..., alpha2, :] == FSPS_powerlaw`` bit-for-bit — the default
SED, energy balance, and the gamma-warming calibration are unchanged.

This keeps the scientifically-preferred FSPS normalisation as the anchor while
making ``dust_alpha`` a faithful, CIGALE-derived free parameter.

Requirements: pcigale installed (tengri's main ``.venv`` has it). The FSPS
``data/themis_templates.h5`` must already exist (``build_themis_from_fsps.py``).
Output: rewrites ``data/themis_templates.h5`` in place, adding ``alpha_grid``
and ``powerlaw_alpha`` (idempotent — single_u/powerlaw are read back unchanged).

Usage
-----
    PYTHONPATH=. .venv/bin/python \
        scripts/build_themis_alpha_axis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ALPHA_GRID = np.round(np.arange(1.0, 3.0 + 1e-9, 0.1), 1)  # CIGALE themis alpha
UMAX_POWERLAW = 1e7  # CIGALE themis.py: model_minmax uses umax = 1e7


def main() -> int:
    try:
        from pcigale.data import SimpleDatabase as Database
    except ImportError:
        print(
            "Error: pcigale is not importable. Run with tengri's main venv.",
            file=sys.stderr,
        )
        return 1

    fsps_path = Path("data/themis_templates.h5")
    if not fsps_path.is_file():
        print(
            f"Error: {fsps_path} not found (run build_themis_from_fsps.py first).", file=sys.stderr
        )
        return 1

    with h5py.File(fsps_path, "r") as f:
        single_u = np.array(f["single_u"][:])  # (n_q, n_u, n_wave)
        powerlaw = np.array(f["powerlaw"][:])  # (n_q, n_u, n_wave) -- FSPS alpha=2
        qhac_grid = np.array(f["qhac_grid"][:])
        umin_grid = np.array(f["umin_grid"][:])
        wave_aa = np.array(f["wavelength_aa"][:])
        fsps_attrs = dict(f.attrs)

    n_u, n_wave = powerlaw.shape[1], powerlaw.shape[2]
    n_alpha = ALPHA_GRID.shape[0]

    # CIGALE power-law (PDR) grid: CIG_pa[cq, u, alpha, wave]. umin matches FSPS.
    print(f"Pulling CIGALE DustEM alpha-grid from pcigale (umin x {n_alpha} alpha)...")
    with Database("themis") as db:
        cig_qhac = sorted({float(q) for q in db.parameters["qhac"]})
        cig_pa = np.zeros((len(cig_qhac), n_u, n_alpha, n_wave))
        for ci, q in enumerate(cig_qhac):
            for ui, u in enumerate(umin_grid):
                for ki, a in enumerate(ALPHA_GRID):
                    m = db.get(qhac=float(q), umin=float(u), umax=UMAX_POWERLAW, alpha=float(a))
                    cig_pa[ci, ui, ki] = np.array(m.spec, dtype=np.float64)

    # Relative alpha-dependence at fixed U_min, anchored at alpha=2 (ratio=1),
    # averaged over CIGALE qhac (alpha reshapes the U distribution ~ independent
    # of grain composition). ratio[u, alpha, wave].
    i_a2 = int(np.argmin(np.abs(ALPHA_GRID - 2.0)))
    denom = cig_pa[:, :, i_a2, :]  # (cq, u, wave)
    denom_safe = np.where(np.abs(denom) > 0, denom, 1.0)
    ratio_per_q = cig_pa / denom_safe[:, :, None, :]  # (cq, u, alpha, wave)
    ratio = np.nanmean(ratio_per_q, axis=0)  # (u, alpha, wave), R(u, alpha)
    ratio[:, i_a2, :] = 1.0  # exact anchor

    # Store only the compact ratio R(umin, alpha, wave) — the loader
    # reconstructs powerlaw_alpha = powerlaw[:, :, None, :] * R. This keeps the
    # tracked file small (~MB) instead of materialising the full 4-D grid
    # (~40 MB). R is dimensionless and the loader unit-normalises each spectrum
    # anyway, so only the shape matters. ratio[:, i_a2, :] == 1 (anchor).
    ratio_f32 = ratio.astype(np.float32)

    with h5py.File(fsps_path, "w") as f:
        f.create_dataset("wavelength_aa", data=wave_aa, dtype=np.float64)
        f.create_dataset("qhac_grid", data=qhac_grid, dtype=np.float64)
        f.create_dataset("umin_grid", data=umin_grid, dtype=np.float64)
        f.create_dataset("alpha_grid", data=ALPHA_GRID, dtype=np.float64)
        f.create_dataset("single_u", data=single_u, dtype=np.float64, compression="gzip")
        f.create_dataset("powerlaw", data=powerlaw, dtype=np.float64, compression="gzip")
        # (n_umin, n_alpha, n_wave) alpha-reshaping ratio (float32, gzip).
        f.create_dataset(
            "powerlaw_alpha_ratio", data=ratio_f32, dtype=np.float32, compression="gzip"
        )
        for k, v in fsps_attrs.items():
            f.attrs[k] = v
        f.attrs["alpha_axis"] = (
            "powerlaw_alpha[q,u,k] = FSPS_powerlaw[q,u] * powerlaw_alpha_ratio[u,k]; "
            "ratio R from CIGALE pcigale.data SimpleDatabase('themis') (qhac-averaged, "
            "anchored at alpha=2 -> ratio 1). The loader reconstructs the 4-D PDR grid "
            "and unit-normalises each spectrum; alpha=2 reproduces the FSPS power-law."
        )
        f.attrs["alpha_axis_generated_by"] = "scripts/build_themis_alpha_axis.py"
    print(
        f"Wrote {fsps_path}: powerlaw_alpha_ratio={ratio_f32.shape}, alpha_grid={ALPHA_GRID.shape}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

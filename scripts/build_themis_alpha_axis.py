#!/usr/bin/env python3
r"""Tabulate the THEMIS radiation-field-slope (alpha) axis per q_hAC composition.

The U^-alpha (PDR) component of the THEMIS dust model (Jones et al. 2017,
A&A 602, A46) is the integral over the starlight intensity U of the single-U
emission of one grain mixture, with ``dU/dM ~ U^-alpha`` between ``U_min`` and
``U_max = 1e7`` (Draine & Li 2007, ApJ 657, 810, Eq. 23). The mixture is set by
q_hAC, so the spectrum at ``alpha != 2`` depends on q_hAC and alpha jointly.

The script reads the PDR spectra ``S(q, u, alpha)`` of the CIGALE THEMIS database
(``pcigale.data.SimpleDatabase('themis')``, ``umax = 1e7``) on the nodes of
``data/themis_templates.h5`` and stores the per-composition ratio

    R(q, u, alpha) = S(q, u, alpha) / S(q, u, alpha = 2)

as ``powerlaw_alpha_ratio`` with shape ``(n_qhac, n_umin, n_alpha, n_wave)``
(float32, gzip) beside ``alpha_grid``. The loader forms
``powerlaw_alpha[q, u, k] = powerlaw[q, u] * R[q, u, k]``, so the ``alpha = 2``
slice of ``R`` is exactly 1 and the ``alpha = 2`` template, energy balance and
gamma-warming calibration are those of the stored ``powerlaw`` dataset.

Requirements: pcigale importable (tengri's main ``.venv``) and an existing
``data/themis_templates.h5`` holding ``single_u``, ``powerlaw``, ``qhac_grid``,
``umin_grid`` and ``wavelength_aa``. The file is rewritten in place.

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/build_themis_alpha_axis.py
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
        # Verify that CIGALE q_hAC values match tengri's (unit: q_hAC = nstirling / 220)
        fsps_qhac_conv = qhac_grid * 2.2 / 100.0
        assert np.allclose(fsps_qhac_conv, cig_qhac, rtol=1e-6), (
            f"qhac mismatch: FSPS {fsps_qhac_conv} vs CIGALE {cig_qhac}"
        )
        cig_pa = np.zeros((len(cig_qhac), n_u, n_alpha, n_wave))
        for ci, q in enumerate(cig_qhac):
            for ui, u in enumerate(umin_grid):
                for ki, a in enumerate(ALPHA_GRID):
                    m = db.get(qhac=float(q), umin=float(u), umax=UMAX_POWERLAW, alpha=float(a))
                    cig_pa[ci, ui, ki] = np.array(m.spec, dtype=np.float64)

    # R(q, u, alpha) = S(q, u, alpha) / S(q, u, alpha=2): alpha reshapes the U
    # distribution of each grain composition separately.
    i_a2 = int(np.argmin(np.abs(ALPHA_GRID - 2.0)))
    denom = cig_pa[:, :, i_a2, :]  # (n_q, n_u, n_wave)
    assert np.all(denom > 0.0), (
        "CIGALE THEMIS alpha=2 spectra must be strictly positive: "
        f"{int(np.sum(denom <= 0.0))} non-positive entries"
    )
    ratio = np.divide(cig_pa, denom[:, :, None, :])
    ratio = np.where(
        (np.arange(n_alpha) == i_a2)[None, None, :, None], 1.0, ratio
    )  # exact anchor at alpha = 2
    assert np.array_equal(ratio[:, :, i_a2, :], np.ones_like(denom))

    ratio_f32 = ratio.astype(np.float32)

    with h5py.File(fsps_path, "w") as f:
        f.create_dataset("wavelength_aa", data=wave_aa, dtype=np.float64)
        f.create_dataset("qhac_grid", data=qhac_grid, dtype=np.float64)
        f.create_dataset("umin_grid", data=umin_grid, dtype=np.float64)
        f.create_dataset("alpha_grid", data=ALPHA_GRID, dtype=np.float64)
        f.create_dataset("single_u", data=single_u, dtype=np.float64, compression="gzip")
        f.create_dataset("powerlaw", data=powerlaw, dtype=np.float64, compression="gzip")
        # (n_qhac, n_umin, n_alpha, n_wave) per-q alpha-reshaping ratio (float32, gzip).
        f.create_dataset(
            "powerlaw_alpha_ratio",
            data=ratio_f32,
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        for k, v in fsps_attrs.items():
            f.attrs[k] = v
        f.attrs["alpha_axis"] = (
            "powerlaw_alpha[q,u,k] = FSPS_powerlaw[q,u] * powerlaw_alpha_ratio[q,u,k]; "
            "ratio R from CIGALE pcigale.data SimpleDatabase('themis') per-q_hAC "
            "(Jones et al. 2017; Draine & Li 2007 Eq. 23), "
            "anchored at alpha=2 -> ratio 1. The loader reconstructs the 4-D PDR grid "
            "and unit-normalizes each spectrum; alpha=2 reproduces the FSPS power-law."
        )
        f.attrs["alpha_axis_generated_by"] = "scripts/build_themis_alpha_axis.py"
    print(
        f"Wrote {fsps_path}: powerlaw_alpha_ratio={ratio_f32.shape}, alpha_grid={ALPHA_GRID.shape}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

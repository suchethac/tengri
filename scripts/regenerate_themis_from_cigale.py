#!/usr/bin/env python3
r"""Regenerate ``data/themis_templates.h5`` from CIGALE's THEMIS database.

tengri previously shipped a THEMIS grid with the power-law (PDR) component
collapsed at the Jones+2017 default ``alpha = 2.0`` — the radiation-field
slope ``dU/dM \propto U^{-alpha}`` could not be fitted, unlike CIGALE which
exposes ``alpha`` over [1.0, 3.0] (21 points). This script rebuilds the full
3-D grid so the ``powerlaw_alpha`` axis is available for inference.

Faithful to ``pcigale.sed_modules.themis._init_code``:

- ``single_u`` (the ``model_minmin`` component) is alpha-independent — it is
  ``db.get(qhac, umin, umax=umin, alpha=1.0)`` (a delta function at ``U=umin``).
- ``powerlaw_alpha[..., k, :]`` (the ``model_minmax`` component) IS the
  alpha-dependent one — ``db.get(qhac, umin, umax=1e7, alpha=alpha_k)``.
- The mixing ``(1 - gamma) * single_u + gamma * powerlaw`` and the
  energy-balance normalisation (``\int spec dlambda = 1``) are applied at
  predict time by tengri's loader, matching CIGALE's ``emissivity`` rescale.

Requirements: pcigale installed (tengri's main ``.venv`` already has it).
Output: ``data/themis_templates.h5`` (axes documented in file attrs).

Usage
-----
    PYTHONPATH=. /Users/suchethacooray/Projects/tengri/.venv/bin/python \
        scripts/regenerate_themis_from_cigale.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

# CIGALE THEMIS database axes (sorted ascending). Verified against
# ``pcigale.data.SimpleDatabase('themis').parameters``.
QHAC_GRID = np.array([0.02, 0.06, 0.1, 0.14, 0.17, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4])
UMIN_GRID = np.array(
    [
        0.1,
        0.12,
        0.15,
        0.17,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        1.0,
        1.2,
        1.5,
        1.7,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        10.0,
        12.0,
        15.0,
        17.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        50.0,
        80.0,
    ]
)
ALPHA_GRID = np.round(np.arange(1.0, 3.0 + 1e-9, 0.1), 1)
UMAX_POWERLAW = 1e7  # CIGALE themis.py: self.umax = 1e7


def main() -> int:
    try:
        from pcigale.data import SimpleDatabase as Database
    except ImportError:
        print(
            "Error: pcigale is not importable. Run with tengri's main venv "
            "(``/Users/suchethacooray/Projects/tengri/.venv/bin/python``).",
            file=sys.stderr,
        )
        return 1

    n_qhac, n_umin, n_alpha = len(QHAC_GRID), len(UMIN_GRID), len(ALPHA_GRID)
    wl_nm = None

    single_u = None  # (n_qhac, n_umin, n_wave)  -- alpha-independent
    powerlaw_alpha = None  # (n_qhac, n_umin, n_alpha, n_wave)

    print(f"Pulling THEMIS grid from CIGALE: {n_qhac} qhac x {n_umin} umin x {n_alpha} alpha ...")
    with Database("themis") as db:
        for i, qhac in enumerate(QHAC_GRID):
            for j, umin in enumerate(UMIN_GRID):
                # single-U (model_minmin): umax=umin, alpha irrelevant -> 1.0
                m_min = db.get(qhac=float(qhac), umin=float(umin), umax=float(umin), alpha=1.0)
                if wl_nm is None:
                    wl_nm = np.array(m_min.wl, dtype=np.float64)
                    n_wave = wl_nm.shape[0]
                    single_u = np.zeros((n_qhac, n_umin, n_wave))
                    powerlaw_alpha = np.zeros((n_qhac, n_umin, n_alpha, n_wave))
                single_u[i, j] = np.array(m_min.spec, dtype=np.float64)
                # power-law (model_minmax): umax=1e7, alpha varies
                for k, alpha in enumerate(ALPHA_GRID):
                    m_max = db.get(
                        qhac=float(qhac), umin=float(umin), umax=UMAX_POWERLAW, alpha=float(alpha)
                    )
                    powerlaw_alpha[i, j, k] = np.array(m_max.spec, dtype=np.float64)

    wl_aa = wl_nm * 10.0
    # alpha=2.0 slice kept as the back-compat 2-D ``powerlaw`` dataset.
    i_a2 = int(np.argmin(np.abs(ALPHA_GRID - 2.0)))
    powerlaw_a2 = powerlaw_alpha[:, :, i_a2, :]

    out_path = Path("data/themis_templates.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("wavelength_aa", data=wl_aa, dtype=np.float64)
        f.create_dataset("qhac_grid", data=QHAC_GRID, dtype=np.float64)
        f.create_dataset("umin_grid", data=UMIN_GRID, dtype=np.float64)
        f.create_dataset("alpha_grid", data=ALPHA_GRID, dtype=np.float64)
        f.create_dataset("single_u", data=single_u, dtype=np.float64, compression="gzip")
        f.create_dataset("powerlaw", data=powerlaw_a2, dtype=np.float64, compression="gzip")
        f.create_dataset(
            "powerlaw_alpha", data=powerlaw_alpha, dtype=np.float64, compression="gzip"
        )
        f.attrs["model"] = "THEMIS (Jones+2017) via CIGALE SimpleDatabase"
        f.attrs["paper"] = "Jones, Köhler, Ysard et al. 2017 A&A 602 A46"
        f.attrs["arxiv"] = "1703.00775"
        f.attrs["upstream"] = "pcigale.data.SimpleDatabase('themis')"
        f.attrs["axes_single_u"] = "(qhac, umin, wavelength)"
        f.attrs["axes_powerlaw_alpha"] = "(qhac, umin, alpha, wavelength)"
        f.attrs["units_spectra"] = "W/nm per kg dust (CIGALE raw; loader normalises)"
        f.attrs["generated_by"] = "scripts/regenerate_themis_from_cigale.py"
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(
        f"  single_u={single_u.shape}, powerlaw_alpha={powerlaw_alpha.shape}, wave={wl_aa.shape}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

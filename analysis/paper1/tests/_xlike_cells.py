# SPDX-License-Identifier: BSD-3-Clause
"""Write an X-like fit cell exactly as ``fit_one`` lays it out on disk.

The one fixture builder for every test that feeds fig10 (or any other X-like
consumer) a results directory. The NPZ keys come from ``fit_one.DERIVED_KEYS``
and are assembled by the writer's own ``build_npz_payload``, so a key renamed in
the writer breaks these tests instead of leaving a figure with zero points.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.paper1._posterior_utils import build_npz_payload
from analysis.paper1.fit_one import DERIVED_KEYS


def surviving_census(gal_id: int, keys, log_mass: float, half_width: float = 0.1) -> dict:
    """Census ``cells`` entries for ``keys`` of one galaxy, in the census JSON's schema."""
    return {
        f"{gal_id}_{key}": {
            "galaxy": int(gal_id),
            "config": key,
            "log_mass_survived_p16": log_mass - half_width,
            "log_mass_survived_p50": log_mass,
            "log_mass_survived_p84": log_mass + half_width,
        }
        for key in keys
    }


def write_xlike_cell(
    results_dir: Path,
    gal_id: int,
    xlike_key: str,
    *,
    adopted: bool = True,
    log_mass: float = 10.0,
    log_sfr: float = 0.5,
    log_sfr_10myr: float | None = None,
    scatter: float = 0.0,
    n_draws: int = 50,
) -> None:
    """Write ``<gal>_<key>.npz`` and ``.json`` for one cell.

    Parameters
    ----------
    log_mass : float
        log10 of the FORMED stellar mass [Msun] the draws are centered on.
    log_sfr, log_sfr_10myr : float
        log10 of the 100 Myr (and 10 Myr) SFR [Msun/yr]; the 10 Myr value defaults
        to ``log_sfr - 1`` so a consumer that reads the wrong timescale shows.
    scatter : float
        Gaussian scatter [dex] on every draw; 0 gives constant draws.
    """
    rng = np.random.default_rng(gal_id)

    def draws(center: float) -> np.ndarray:
        return 10 ** (center + scatter * rng.standard_normal(n_draws))

    centers = {
        "stellar_mass": log_mass,
        "sfr_100myr": log_sfr,
        "sfr_10myr": log_sfr - 1.0 if log_sfr_10myr is None else log_sfr_10myr,
        "dust_tau": 0.0,
    }
    derived = {key: draws(centers[key]) for key in DERIVED_KEYS}
    payload = build_npz_payload({"met_logzsol": rng.standard_normal(n_draws)}, derived)
    np.savez(results_dir / f"{gal_id}_{xlike_key}.npz", **payload)
    # The diagnostics fit_one records, consistent with ``adopted`` under both the
    # strict bar (adoption_pass) and the relaxed one, which reads them directly.
    diagnostics = (
        {"rhat_max": 1.003, "divergences": 0, "ess_min": 500.0}
        if adopted
        else {"rhat_max": 1.05, "divergences": 120, "ess_min": 40.0}
    )
    meta = {
        "gal_id": gal_id,
        "config": xlike_key,
        "z": 1.0,
        "adoption_pass": adopted,
        "n_samples": 600,
        "n_chains": 4,
        **diagnostics,
    }
    (results_dir / f"{gal_id}_{xlike_key}.json").write_text(json.dumps(meta))

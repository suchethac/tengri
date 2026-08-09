# SPDX-License-Identifier: BSD-3-Clause
"""Template families to check for threaded-vs-closure equivalence.

One row per AGN template family that declares a ``template_loader``. Kept
beside the threading contract test so adding a family is a one-line change.
"""

from __future__ import annotations

__all__ = ["SED_CASES"]

#: ``(name, module, sed_fn, loader_fn, kwargs)``
SED_CASES: list[tuple[str, str, str, str, dict]] = [
    (
        "silva04",
        "tengri.components.agn.silva04",
        "silva04_sed",
        "load_silva04_default_grid",
        {"agn_log_lbol": 11.0, "agn_log_nh_silva": 23.0, "agn_torus_frac": 0.4},
    ),
    (
        "cat3d_wind",
        "tengri.components.agn.cat3d_wind",
        "cat3d_wind_sed",
        "load_cat3d_wind_default_grid",
        {"agn_log_lbol": 11.0, "agn_cos_inc": 0.7, "agn_torus_frac": 0.4},
    ),
    (
        "skirtor_agnfitter",
        "tengri.components.agn.skirtor_agnfitter",
        "skirtor_agnfitter_sed",
        "load_skirtor_agnfitter_default_grid",
        {"agn_log_lbol": 11.0, "agn_oa_skirtor": 40.0, "agn_torus_frac": 0.4},
    ),
    (
        "nenkova_agnfitter",
        "tengri.components.agn.nenkova_agnfitter",
        "nenkova_agnfitter_sed",
        "load_nenkova_agnfitter_default_grid",
        {"agn_log_lbol": 11.0, "agn_cos_inc": 0.7, "agn_torus_frac": 0.4},
    ),
    (
        "fritz",
        "tengri.components.agn.fritz",
        "fritz_sed",
        "load_fritz_default_grid",
        {"agn_log_lbol": 11.0, "agn_fritz_tau": 1.0, "agn_torus_frac": 0.4},
    ),
]

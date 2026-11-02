# SPDX-License-Identifier: BSD-3-Clause
"""Generate the frozen ``agn_norm='cigale_joint'`` + fracAGN reference of #2320.

Run from a checkout of the commit the reference is frozen at (its SHA is stored in the file)::

    PYTHONPATH=src python tests/regression/data/agn_2320_cigale_joint_reference/generate.py <sha>

It writes ``reference.npz`` beside this script: the rest-frame ``sed_intrinsic`` [erg/s/Hz] and
``agn_torus_frac_effective`` for the two builds of
``tests/regression/bug/test_bug_2320_conserving_under_fracagn.py`` (``_build``) under
``'cigale_joint'``: total dust at ``log_total_mass = -5`` (``t > 1``, unclipped) and the
5-20 um window at ``log_total_mass = -6``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

from tengri import DEFAULT, Fixed, SEDModel

#: (name, ir_frac_window, log_total_mass) of the frozen builds.
CASES = (("total", None, -5.0), ("window", (5.0, 20.0), -6.0))


def build(ssp, window, log_total_mass):
    """The regression test's ``_build`` under ``'cigale_joint'`` with ``agn_ir_frac = 0.3``."""
    agn = {
        "type": "composable",
        "norm": "cigale_joint",
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "agn_ir_frac": Fixed(0.3),
        "all_params": Fixed(DEFAULT),
    }
    if window is not None:
        agn["ir_frac_window"] = window
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(log_total_mass),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
                "agn_screen": "none",
            },
            dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
            agn=agn,
            redshift=Fixed(0.05),
        )


def main(sha: str) -> None:
    root = Path.cwd()
    sys.path.insert(0, str(root))
    from tests.conftest import synthetic_ssp_wide

    ssp = synthetic_ssp_wide.__wrapped__()
    out = {"source_sha": np.asarray(sha)}
    for name, window, mass in CASES:
        state = build(ssp, window, mass).predict_state({})
        out[f"{name}_sed"] = np.asarray(state.sed_intrinsic)
        out[f"{name}_torus_frac"] = np.asarray(state.derived["agn_torus_frac_effective"])
    np.savez_compressed(Path(__file__).with_name("reference.npz"), **out)


if __name__ == "__main__":
    main(sys.argv[1])

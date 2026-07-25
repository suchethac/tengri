# SPDX-License-Identifier: BSD-3-Clause
r"""Float32 status of every composable-AGN disc block (#1206).

A durable inventory: for each registered disc block, build a composable AGN
(disc + SKIRTOR torus, CIGALE-joint norm) and compare ``sed_agn`` between
float64 and pure float32. Two failure classes exist beyond the exact discs:

* **Shape-class** (``kubota_done``, ``adaf``): the disc *shape* depends on L_bol
  (temperature), so the float32 path — which evaluates the runner at a low
  reference L_bol to keep the ~1e40 ``L_lambda`` arithmetic in range — gives the
  wrong (cold) shape. ``multicolor`` was in this class and is now fixed (it takes
  the true L_bol for its shape via ``agn_log_lbol_shape`` while normalizing to
  the reference); ``kubota_done`` / ``adaf`` need the same log-space +
  shape/normalization split threaded through their (more involved) internals.
* **Grid/other-class** (``relagn``, ``slone_netzer``, ``grahsp_sbpl``,
  ``adaf_lopez2024``): non-finite in float32 *even at the reference L_bol*, so
  the overflow is NOT the L_bol magnitude — it is an internal grid value or a
  per-wavelength term that needs its own float32 hardening.

This test pins the exact discs (regression guard) and ``xfail``\ s the rest
(progress tracker: fixing one turns its ``xfail`` into an unexpected pass). It is
the enforced record of "checked every AGN disc component".
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

# Discs whose spectral shape is invariant under L_bol (template / power-law) OR
# whose L_bol-dependent shape is now handled on the float32 path (multicolor,
# kubota_done — log-space internals + shape/normalization split).
_EXACT_DISCS = [
    "multicolor",
    "kubota_done",
    "powerlaw",
    "richards2006",
    "skirtor",
    "qsogen",
    "schartmann2005",
]

# Shape depends on L_bol; float32 reference evaluation gives the wrong shape.
_SHAPE_CLASS_XFAIL = ["adaf"]

# Non-finite in float32 even at the reference L_bol — a distinct internal overflow.
_GRID_CLASS_XFAIL = ["relagn", "slone_netzer", "grahsp_sbpl", "adaf_lopez2024"]


def _sed_agn(ssp, disc, dtype):
    obs = Observation(photometry=Photometry.from_names(["sdss_r", "wise_w3", "wise_w4"]))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_diff": 0.3,
            "tau_bc": 0.0,
        },
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": disc, "all_params": FIXED},
            "torus": {"type": "skirtor", "all_params": FIXED},
            "norm": "cigale_joint",
            "log_lbol": Uniform(9.0, 13.0),
            "fracAGN": 0.1,
        },
        redshift=Fixed(0.1),
    )
    p = {
        k: jnp.asarray(v, dtype=dtype)
        for k, v in {"sfh_delayed_log_total_mass": 10.0, "agn_log_lbol": 11.0}.items()
    }
    return np.asarray(model.predict_state(p).derived["sed_agn"])


def _f32_matches_f64(ssp, disc):
    with jax.enable_x64(True):
        ref = _sed_agn(ssp, disc, jnp.float64)
    with jax.enable_x64(False):
        f32 = _sed_agn(ssp, disc, jnp.float32)
    if not np.all(np.isfinite(f32)):
        return False, "non-finite in float32"
    peak = np.abs(ref).max()
    live = np.abs(ref) > 1e-6 * peak
    rel = np.abs(f32[live] - ref[live]) / np.abs(ref[live])
    return rel.max() < 1e-3, f"max_rel={rel.max():.2e}"


@pytest.mark.parametrize("disc", _EXACT_DISCS)
def test_disc_is_exact_in_float32(ssp_bare, disc):
    """These disc blocks must match float64 to float32 eps (regression guard)."""
    ok, detail = _f32_matches_f64(ssp_bare, disc)
    assert ok, f"disc '{disc}' regressed on the pure-float32 path: {detail}"


@pytest.mark.parametrize("disc", _SHAPE_CLASS_XFAIL + _GRID_CLASS_XFAIL)
@pytest.mark.xfail(reason="#1206 follow-up: disc not yet float32-hardened", strict=True)
def test_disc_float32_pending(ssp_bare, disc):
    """Progress tracker — fixing one of these flips its xfail to an unexpected pass."""
    ok, detail = _f32_matches_f64(ssp_bare, disc)
    assert ok, f"disc '{disc}' still float32-broken: {detail}"

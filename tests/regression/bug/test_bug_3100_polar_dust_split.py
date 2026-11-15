# SPDX-License-Identifier: BSD-3-Clause
"""#3100: the polar-dust split budgets on the torus's angle-integrated power, so energy closes.

Under ``agn_norm='conserving'`` with ``agn_attenuation_block='polar_dust'`` the runner shares
the AGN dust budget between the torus and the polar dust, ``share = P_polar / (budget + P_polar)``.
For a library torus that keeps its anisotropy ``a(mu)`` (Fritz, the NENKOVA and SKIRTOR AGNfitter
tori) the budget must be the angle-integrated torus power ``f L_bol``, not its sightline power
``f L_bol a(mu)/a_bar``: the polar re-emission is isotropic and observer-independent, so the
pieces (sphere means) add up to the accretion power only with the angle-integrated budget.

Checked here: the sphere-mean pieces plus their unemitted powers equal ``10**agn_log_lbol L_sun``
to the ledger's closure tolerance at several sightlines, and the ledger flag is 1.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks._protocol import collect_block_templates
from tengri.components.agn.blocks.runner import composable_agn_l_nu
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_bug

_LOG_LBOL = 10.5
#: The ledger's closure tolerance (``LEDGER_CLOSURE_RTOL``, the C2 tolerance of the contract).
_RTOL = 1.0e-6
_WAVE = np.geomspace(100.0, 1.0e7, 400)
_PIECES = (
    "log_L_agn_disc",
    "log_L_agn_torus",
    "log_L_agn_polar_dust",
    "log_L_agn_disc_unemitted",
    "log_L_agn_torus_unemitted",
)


def _evaluate(torus: str, cos_inc: float):
    """The composable runner, jitted, with the polar-dust recipe on ``kubota_done``."""
    selectors = {
        "agn_disc_block": "kubota_done",
        "agn_nlr_block": "none",
        "agn_blr_block": "none",
        "agn_feii_block": "none",
        "agn_torus_block": torus,
        "agn_attenuation_block": "polar_dust",
    }
    templates = collect_block_templates(
        {k[len("agn_") : -len("_block")]: v for k, v in selectors.items()}
    )
    params = {
        "agn_norm": "conserving",
        "agn_cos_inc": cos_inc,
        "agn_torus_frac": 0.4,
        "agn_polar_ebv": 0.2,
    }

    def run(wave, traced, ts):
        return composable_agn_l_nu(
            wave,
            _LOG_LBOL,
            return_components=True,
            return_ledger=True,
            template_state=ts,
            **selectors,
            **params,
            **traced,
        )

    with jax.enable_x64(True):
        wave = jnp.asarray(_WAVE, dtype=jnp.float64)
        _, comps = jax.jit(run)(wave, {}, templates)
        return {k: float(np.asarray(comps[k])) for k in (*_PIECES, "agn_ledger_closed")}


@pytest.mark.parametrize("torus", ["nenkova_agnfitter", "fritz"])
@pytest.mark.parametrize("cos_inc", [0.1, 0.5, 0.9])
def test_polar_split_conserves_energy_on_an_anisotropic_torus(torus, cos_inc):
    """Sum of the sphere-mean pieces and unemitted powers is the accretion power, to 1e-6."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comps = _evaluate(torus, cos_inc)
    except FileNotFoundError as exc:
        pytest.skip(f"library data absent: {exc}")
    budget = 10.0**_LOG_LBOL * L_SUN
    total = sum(10.0 ** comps[k] for k in _PIECES if np.isfinite(comps[k]))
    assert abs(total / budget - 1.0) <= _RTOL
    assert comps["agn_ledger_closed"] == 1.0

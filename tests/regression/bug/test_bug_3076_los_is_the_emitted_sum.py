# SPDX-License-Identifier: BSD-3-Clause
"""``log_L_agn_los`` is the emitted line-of-sight power; the old value is ``log_L_agn_direct_los``.

Until #3076 the composable AGN published, as ``log_L_agn_los``, the line-of-sight
isotropic-equivalent power of the disc block's own spectrum before the torus screen, the polar
dust and the energy debits. Under the 2026-10-13 rulings ``log_L_agn_los`` is the emitted
line-of-sight sum, the integral of ``sed_agn``; the old value moved, unchanged to the bit, to
``log_L_agn_direct_los``. Checked on a Type-2 sightline behind the torus, where the two
differ by orders of magnitude, and through ``SEDModel``.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data
from tengri.components.agn.blocks._protocol import collect_block_templates, resolve_agn_block
from tengri.components.agn.blocks.runner import _los_budget_wave, composable_agn_l_nu
from tengri.utils.physics_constants import C_AA, L_SUN

pytestmark = pytest.mark.regression_bug

_SSP_PATH = Path(__file__).resolve().parents[3] / "data" / "fsps_prsc_miles_chabrier.h5"
_LOG_LBOL = 11.0
_COS_TYPE2 = 0.3
_RECIPE = dict(
    agn_disc_block="kubota_done",
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="simple",
    agn_attenuation_block="none",
    agn_norm="conserving",
)
_PARAMS = dict(agn_cos_inc=_COS_TYPE2, agn_torus_frac=0.4, agn_log_mbh=8.0)
#: Dense grid for the emitted integral [A]: 10 000 nodes per decade.
_DENSE = np.geomspace(1.0e-3, 1.0e10, 130_001)


def _templates():
    return collect_block_templates({"disc": "kubota_done", "torus": "simple"})


def _run(wave, *, ledger):
    return composable_agn_l_nu(
        jnp.asarray(wave),
        _LOG_LBOL,
        return_components=True,
        return_ledger=ledger,
        template_state=_templates(),
        **_RECIPE,
        **_PARAMS,
    )


def _power(lnu, wave):
    nu = C_AA / np.asarray(wave)
    return float(-np.trapezoid(np.asarray(lnu) * nu, np.log(nu)))


def test_los_is_the_integral_of_the_emitted_sed():
    """``log_L_agn_los`` = log10 of the frequency integral of ``sed_agn``, 1e-6."""
    _, comps = _run(np.geomspace(100.0, 1.0e7, 200), ledger=True)
    sed, _ = _run(_DENSE, ledger=False)
    assert 10.0 ** float(comps["log_L_agn_los"]) == pytest.approx(_power(sed, _DENSE), rel=1e-6)


def test_direct_los_is_the_old_value_to_the_bit():
    """``log_L_agn_direct_los`` is the pre-#3076 ``log_L_agn_los``: the disc block's own
    spectrum at the sightline on the budget grid, before every screen and debit."""
    _, comps = _run(np.geomspace(100.0, 1.0e7, 200), ledger=True)
    _, comps_off = _run(np.geomspace(100.0, 1.0e7, 200), ledger=False)
    np.testing.assert_array_equal(comps["log_L_agn_direct_los"], comps_off["log_L_agn_direct_los"])
    wave = _los_budget_wave(True, jnp.float64)
    disc = resolve_agn_block("disc", "kubota_done")(
        wave, agn_log_lbol=_LOG_LBOL, agn_cos_inc=_COS_TYPE2, agn_log_mbh=8.0
    )
    old = np.log10(float(jnp.abs(jnp.trapezoid(disc, wave))))
    assert float(comps["log_L_agn_direct_los"]) == pytest.approx(old, abs=1e-12)
    # Behind the torus on a Type-2 sightline the emitted sum is not the direct emission.
    assert abs(float(comps["log_L_agn_los"]) - float(comps["log_L_agn_direct_los"])) > 0.1


@pytest.fixture(scope="module")
def ssp():
    try:
        return load_ssp_data(str(_SSP_PATH))
    except FileNotFoundError as exc:
        pytest.skip(f"SSP grid absent: {exc}")


def test_model_publishes_the_ledger_and_the_renamed_key(ssp):
    """Through ``SEDModel``: every ledger key is published and the two keys keep their meaning."""
    model = SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        agn={
            "type": "composable",
            "disc": {"type": "kubota_done", "all_params": Fixed(DEFAULT)},
            "torus": {
                "type": "simple",
                "agn_torus_frac": Fixed(0.4),
                "all_params": Fixed(DEFAULT),
            },
            "atten": {"type": "none"},
            "norm": "conserving",
            "agn_log_lbol": Fixed(_LOG_LBOL),
            "agn_cos_inc": Fixed(_COS_TYPE2),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.0),
    )
    derived = model.predict_state({}).derived
    from tengri.components.agn.blocks._ledger import COMPOSABLE_LEDGER_OUTPUTS

    for key in (*COMPOSABLE_LEDGER_OUTPUTS, "log_L_agn_direct_los", "L_agn_direct_los_lsun"):
        assert derived.get(key) is not None, key
    assert float(derived["agn_ledger_closed"]) == 1.0
    assert float(derived["L_agn_los_lsun"]) == pytest.approx(
        10.0 ** (float(derived["log_L_agn_los"]) - np.log10(L_SUN)), rel=1e-12
    )
    assert float(derived["log_L_agn_los"]) < float(derived["log_L_agn_direct_los"])

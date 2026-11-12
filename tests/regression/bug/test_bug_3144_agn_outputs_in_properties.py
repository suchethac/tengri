# SPDX-License-Identifier: BSD-3-Clause
"""The AGN component's published luminosities are reachable from ``pred.properties`` (#3144).

The AGN component declares ``L_agn_bol``, ``log_L_agn_bol``, ``L_agn_bol_lsun`` and the
composable and ADAF keys as outputs, and the forward pass publishes them in ``state.derived``.
Until the catalog carried them, a user could read none of them through ``pred.properties``,
``pred.<key>`` or ``predict_properties``: the names raised "Unknown property". Each key must be
in ``available_properties`` for the model that publishes it, and read back equal to the
internal value: on the default AGN (the grammar's composable AGN), on a monolithic AGN,
and on the composable AGN with the ADAF disc.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel

pytestmark = pytest.mark.regression_bug

#: Published on every AGN model (the forward pass sets them unconditionally).
_DEFAULT_KEYS = ("L_agn_bol", "log_L_agn_bol", "L_agn_bol_lsun")
#: Published by the composable AGN only.
_COMPOSABLE_KEYS = ("log_L_agn_direct_los", "L_agn_direct_los_lsun", "agn_torus_frac_effective")
#: Published by the composable AGN with the ADAF disc only.
_ADAF_KEYS = ("agn_adaf_mdot", "log_L_agn_adaf_accretion", "L_agn_adaf_accretion_lsun")


def _build(ssp, agn):
    return SEDModel.build(
        ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.5),
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        agn=agn,
        redshift=Fixed(0.05),
    )


def _assert_reachable(model, keys):
    """Each key is in the catalog and reads back equal to the component's internal value."""
    params = {k: jnp.asarray(v) for k, v in model.spec.sample(jax.random.PRNGKey(0)).items()}
    catalog = model.available_properties
    missing = [k for k in keys if k not in catalog]
    assert not missing, f"published but not in available_properties: {missing}"

    internal = model.predict_state(params).derived
    via_props = model.predict(params).properties
    via_accessor = model.predict_properties(params, names=keys)
    for key in keys:
        expected = np.asarray(internal[key])
        np.testing.assert_array_equal(np.asarray(via_props[key]), expected, err_msg=key)
        np.testing.assert_array_equal(np.asarray(via_accessor[key]), expected, err_msg=key)


def test_default_agn_outputs_are_reachable(synthetic_ssp_wide):
    """The grammar's default AGN is the composable one (``agn={'all_params': ...}``)."""
    model = _build(synthetic_ssp_wide, {"all_params": Fixed(DEFAULT)})
    _assert_reachable(model, _DEFAULT_KEYS + _COMPOSABLE_KEYS)


def test_composable_agn_outputs_are_reachable(synthetic_ssp_wide):
    agn = {
        "type": "composable",
        "all_params": Fixed(DEFAULT),
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
    }
    model = _build(synthetic_ssp_wide, agn)
    _assert_reachable(model, _DEFAULT_KEYS + _COMPOSABLE_KEYS)


def test_adaf_outputs_are_reachable(synthetic_ssp_wide):
    agn = {
        "type": "composable",
        "all_params": Fixed(DEFAULT),
        "disc": {"type": "adaf", "all_params": Fixed(DEFAULT)},
    }
    model = _build(synthetic_ssp_wide, agn)
    _assert_reachable(model, _DEFAULT_KEYS + _COMPOSABLE_KEYS + _ADAF_KEYS)


def test_composable_only_keys_absent_from_monolithic_agn(synthetic_ssp_wide):
    """A monolithic AGN does not publish the composable keys, so they are not catalog entries."""
    model = _build(synthetic_ssp_wide, {"type": "multicolor_agn", "all_params": Fixed(DEFAULT)})
    _assert_reachable(model, _DEFAULT_KEYS)
    absent = [k for k in _COMPOSABLE_KEYS + _ADAF_KEYS if k in model.available_properties]
    assert not absent

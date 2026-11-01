# SPDX-License-Identifier: BSD-3-Clause
r"""Component-level ``(tau_V, mu)`` dust parameterization (#3176, part 1).

``DustSEDComponentConfig(parameterization="tauv_mu")`` derives the two screen
depths from a total V-band optical depth and the diffuse-ISM fraction
(Charlot & Fall 2000; the BEAGLE/MAGPHYS ``tauV_eff``, ``mu`` convention):

.. math::

    \hat\tau_V^{\rm BC} = (1-\mu)\,\hat\tau_V, \qquad
    \hat\tau_V^{\rm ISM} = \mu\,\hat\tau_V.

These are pure-function component tests (no SSP data), so they run in default CI.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust import two_component
from tengri.components.dust._apply import screen_optical_depths
from tengri.components.dust._params import (
    DUST_PARAMETERIZATIONS,
    SINGLE_COMPONENT_PARAMS,
    TAUV_MU_PARAMS,
)
from tengri.components.dust.two_component import (
    DustSEDComponent,
    DustSEDComponentConfig,
)
from tengri.protocols.component import ForwardState
from tests._age_boundary import hand_age_derived

pytestmark = pytest.mark.contract

_WAVE = jnp.array([1500.0, 2700.0, 3550.0, 5500.0, 9000.0])
_AGES = jnp.array([1.0e6, 1.0e8, 1.0e10])


def _state(sed_neb: jnp.ndarray) -> ForwardState:
    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0])).at[2, :].set(1.0e29)
    return ForwardState(
        wave=_WAVE,
        sed_intrinsic=jnp.sum(lnu_age, axis=0) + sed_neb,
        derived={
            "lnu_age": lnu_age,
            "ssp_ages_yr": _AGES,
            **hand_age_derived(_AGES),
            "sed_nebular": sed_neb,
        },
    )


def _component(parameterization: str) -> DustSEDComponent:
    return DustSEDComponent(
        config=DustSEDComponentConfig(
            law_bc="power_law", law_diff="power_law", parameterization=parameterization
        )
    )


def test_screen_optical_depths_native_passthrough():
    bc, diff = screen_optical_depths({"dust_tau_bc": 1.0, "dust_tau_diff": 0.3}, "tau_bc_diff")
    assert bc == 1.0
    assert diff == 0.3


def test_screen_optical_depths_tauv_mu_formula():
    bc, diff = screen_optical_depths({"dust_tau_v": 2.0, "dust_mu": 0.25}, "tauv_mu")
    assert np.array_equal(np.asarray([bc, diff]), np.array([1.5, 0.5]))


def test_unknown_parameterization_refused():
    with pytest.raises(ValueError, match="tau_bc_diff"):
        screen_optical_depths({}, "bogus")


def test_config_refuses_unknown_parameterization():
    with pytest.raises(ValueError, match="tauv_mu"):
        DustSEDComponentConfig(parameterization="bogus")


def test_direct_component_tauv_mu_equals_native_bit_exact():
    state = _state(jnp.full(_WAVE.shape, 1.0e28))
    native = _component("tau_bc_diff").apply(
        state, {"dust_tau_bc": 1.5, "dust_tau_diff": 0.5, "dust_slope": -0.7}
    )
    tauv_mu = _component("tauv_mu").apply(
        state, {"dust_tau_v": 2.0, "dust_mu": 0.25, "dust_slope": -0.7}
    )
    assert np.array_equal(np.asarray(native.sed_intrinsic), np.asarray(tauv_mu.sed_intrinsic))


def test_declared_parameters_switch_with_parameterization():
    native = {d.name for d in _component("tau_bc_diff").declared_parameters()}
    tauv_mu = {d.name for d in _component("tauv_mu").declared_parameters()}
    assert {"dust_tau_bc", "dust_tau_diff"} <= native
    assert "dust_tau_v" not in native and "dust_mu" not in native
    assert {"dust_tau_v", "dust_mu"} <= tauv_mu
    assert "dust_tau_bc" not in tauv_mu and "dust_tau_diff" not in tauv_mu


def test_parameterization_registry_and_declarations():
    assert DUST_PARAMETERIZATIONS == ("tau_bc_diff", "tauv_mu")
    (mu_decl,) = TAUV_MU_PARAMS
    assert mu_decl.name == "dust_mu"
    assert [d.name for d in SINGLE_COMPONENT_PARAMS] == ["dust_tau_v"]


def test_no_literal_depth_reads_remain():
    source = Path(two_component.__file__).read_text(encoding="utf-8")
    assert source.count('params["dust_tau_bc"]') == 0
    assert source.count('params["dust_tau_diff"]') == 0


def test_grad_wrt_mu_finite():
    comp = _component("tauv_mu")
    younger = jnp.array([[1.0, 0.0, 0.0]])

    def total_transmission(mu):
        params = {"dust_tau_v": 1.5, "dust_mu": mu, "dust_slope": -0.7}
        return comp.compute_transmission(params, _WAVE, younger).sum()

    grad = float(jax.grad(total_transmission)(0.4))
    assert np.isfinite(grad)
    assert grad != 0.0

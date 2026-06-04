# SPDX-License-Identifier: BSD-3-Clause
"""Independent birth-cloud vs diffuse law parameters in two-component dust.

Regression for the CROSSVAL-01 under-attenuation: tengri tied the birth-cloud
and diffuse power-law indices to a single ``dust_slope`` (-0.7, original
Charlot & Fall 2000), so it could not express the FSPS Charlot & Fall default
``dust1_index = -1.0`` (steeper birth cloud). That left tengri ~0.45 mag low in
the NUV for young populations. ``two_component_dust`` now accepts independent
``bc_params`` / ``diff_params`` overlays so the two components can carry
different slopes (and bump / delta / Rv).

These are pure-function tests (no SSP data), so they run in default CI.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import V_BAND_ANGSTROM, two_component_dust

pytestmark = [pytest.mark.regression_bug]

# Young limit: age << t_birth so the sigmoid weight -> 1 (birth cloud active).
_YOUNG = jnp.array([1.0e4])
# Old limit: age >> t_birth so the weight -> 0 (diffuse only).
_OLD = jnp.array([1.0e10])
_WAVE = jnp.array([2700.0, 5500.0])


def _tau_from_trans(trans: np.ndarray) -> np.ndarray:
    return -np.log(np.asarray(trans))


class TestDefaultsUnchanged:
    """No bc/diff overlay -> identical to the shared-``law_params`` behaviour."""

    def test_shared_n_slope_default(self) -> None:
        ref = two_component_dust(_WAVE, _YOUNG, tau_v1=1.0, tau_v2=0.3, n_slope=-0.7)
        new = two_component_dust(
            _WAVE,
            _YOUNG,
            tau_v1=1.0,
            tau_v2=0.3,
            n_slope=-0.7,
            bc_params=None,
            diff_params=None,
        )
        np.testing.assert_array_equal(np.asarray(ref), np.asarray(new))


class TestIndependentSlopes:
    """bc_params / diff_params override the shared slope per component."""

    def test_birth_cloud_steeper_slope(self) -> None:
        """Young-limit birth cloud uses bc_params slope, not the shared one."""
        # Diffuse off (tau_v2=0); only the birth cloud contributes.
        trans = two_component_dust(
            _WAVE,
            _YOUNG,
            tau_v1=1.0,
            tau_v2=0.0,
            law_bc="power_law",
            law_diff="power_law",
            bc_params={"n_slope": -1.0},
            diff_params={"n_slope": -0.7},
        )
        tau = _tau_from_trans(trans)[0]  # single age row
        # Expected birth-cloud optical depth: tau_v1 * (lambda/5500)^-1.0,
        # weight ~ 1 at age 1e4 yr.
        k_expected = (np.asarray(_WAVE) / V_BAND_ANGSTROM) ** -1.0
        np.testing.assert_allclose(tau, 1.0 * k_expected, rtol=2e-3)

    def test_diffuse_keeps_its_own_slope(self) -> None:
        """Old-limit diffuse uses diff_params slope, independent of bc."""
        trans = two_component_dust(
            _WAVE,
            _OLD,
            tau_v1=0.0,
            tau_v2=1.0,
            law_bc="power_law",
            law_diff="power_law",
            bc_params={"n_slope": -1.0},
            diff_params={"n_slope": -0.7},
        )
        tau = _tau_from_trans(trans)[0]
        k_expected = (np.asarray(_WAVE) / V_BAND_ANGSTROM) ** -0.7
        np.testing.assert_allclose(tau, 1.0 * k_expected, rtol=2e-3)

    def test_fsps_parity_recovers_nuv(self) -> None:
        """BC slope -1.0 closes the NUV gap vs the FSPS Charlot & Fall curve.

        FSPS young-pop birth-cloud A(NUV) = 1.0857 * tau_bc * (2700/5500)^-1.0.
        With the shared -0.7 slope tengri is ~0.45 mag low; with bc slope -1.0
        it matches.
        """
        wave = jnp.array([2700.0])
        tau_bc = 1.0
        a_fsps = (2.5 / np.log(10.0)) * tau_bc * (2700.0 / 5500.0) ** -1.0

        trans_fixed = two_component_dust(
            wave,
            jnp.array([1.0e4]),
            tau_v1=tau_bc,
            tau_v2=0.0,
            bc_params={"n_slope": -1.0},
        )
        a_tengri = -2.5 * np.log10(np.asarray(trans_fixed)[0, 0])
        np.testing.assert_allclose(a_tengri, a_fsps, rtol=3e-3)


class TestIndependentShapeParams:
    """bump / delta carry per-component too (Noll/Kriek-Conroy laws)."""

    def test_bc_bump_independent_of_diffuse(self) -> None:
        wave = jnp.linspace(2000.0, 2400.0, 41)  # across the 2175 A bump
        # Birth cloud carries a UV bump; diffuse does not.
        trans = two_component_dust(
            wave,
            _YOUNG,
            tau_v1=1.0,
            tau_v2=0.0,
            law_bc="kriek_conroy",
            law_diff="kriek_conroy",
            bc_params={"dust_bump_strength": 3.0, "dust_delta": 0.0},
            diff_params={"dust_bump_strength": 0.0, "dust_delta": 0.0},
        )
        # With diffuse off and a strong BC bump, attenuation must peak near
        # 2175 A (minimum transmission there).
        t = np.asarray(trans)[0]
        i_bump = int(np.argmin(t))
        assert 2150.0 <= float(np.asarray(wave)[i_bump]) <= 2200.0

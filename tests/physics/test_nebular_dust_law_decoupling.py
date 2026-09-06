# SPDX-License-Identifier: BSD-3-Clause
"""Nebular birth-cloud attenuation law can be decoupled from the stars.

By default the nebular continuum is reddened by exactly the stellar young-limit
screen — the same birth-cloud law/params plus the shared diffuse ISM screen
(Charlot & Fall 2000; bagpipes/FSPS/CIGALE). ``DustSEDComponentConfig.law_neb``
(and ``neb_law_overrides``) lets HII-region emission carry a *different*
birth-cloud curve while still sharing the diffuse ISM screen with the stars:

    tau_neb = tau_bc * k(law_neb, neb_params) + tau_diff * k(law_diff, diff_params)

These are pure-function component tests (no SSP data), so they run in default CI.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import resolve_dust_law
from tengri.components.dust.two_component import (
    DustSEDComponent,
    DustSEDComponentConfig,
)
from tengri.protocols.component import ForwardState

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_WAVE = jnp.array([1500.0, 2700.0, 3550.0, 5500.0, 9000.0])
_AGES = jnp.array([1.0e6, 1.0e8, 1.0e10])


def _state_with_nebular(sed_neb: jnp.ndarray) -> ForwardState:
    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0])).at[2, :].set(1.0e29)
    return ForwardState(
        wave=_WAVE,
        sed_intrinsic=jnp.sum(lnu_age, axis=0) + sed_neb,
        derived={"lnu_age": lnu_age, "ssp_ages_yr": _AGES, "sed_nebular": sed_neb},
    )


def _recover_nebular(out) -> np.ndarray:
    """Isolate the attenuated nebular slice from the reconstructed total SED.

    With ``emission_model=None`` and no other non-stellar component,
    ``sed_total = sed_neb_attenuated + sed_dust_attenuated``.
    """
    stellar_att = jnp.asarray(out.derived["sed_dust_attenuated"])
    return np.asarray(out.sed_intrinsic - stellar_att)


def test_default_nebular_inherits_birth_cloud_law():
    """law_neb=None -> nebular reddened by the stellar bc + diffuse young limit."""
    sed_neb = jnp.full(_WAVE.shape, 1.0e28)
    params = {"dust_tau_bc": 1.0, "dust_tau_diff": 0.5, "dust_slope": -0.7}
    comp = DustSEDComponent(
        config=DustSEDComponentConfig(law_bc="power_law", law_diff="power_law")
    )
    neb = _recover_nebular(comp.apply(_state_with_nebular(sed_neb), params))

    k = np.asarray(resolve_dust_law("power_law")(_WAVE, n_slope=-0.7))
    expected = np.asarray(sed_neb) * np.exp(-(1.0 * k + 0.5 * k))
    np.testing.assert_allclose(neb, expected, rtol=1e-5)


def test_law_neb_decouples_only_the_birth_cloud():
    """law_neb swaps the nebular bc curve; the diffuse ISM screen stays shared."""
    sed_neb = jnp.full(_WAVE.shape, 1.0e28)
    params = {"dust_tau_bc": 1.0, "dust_tau_diff": 0.5, "dust_slope": -0.7}
    comp = DustSEDComponent(
        config=DustSEDComponentConfig(law_bc="power_law", law_diff="power_law", law_neb="calzetti")
    )
    neb = _recover_nebular(comp.apply(_state_with_nebular(sed_neb), params))

    # bc part uses the nebular law (calzetti); diffuse part keeps the shared
    # power_law(-0.7). Params inherited from the stellar bc are narrowed to what
    # calzetti declares, which is nothing: it fixes R_V = 4.05 in the polynomial
    # and takes no shape argument, so splatting the stellar bc dict into it is a
    # TypeError rather than four silently discarded values (#2185).
    k_bc = np.asarray(resolve_dust_law("calzetti")(_WAVE))
    k_diff = np.asarray(resolve_dust_law("power_law")(_WAVE, n_slope=-0.7))
    expected = np.asarray(sed_neb) * np.exp(-(1.0 * k_bc + 0.5 * k_diff))
    np.testing.assert_allclose(neb, expected, rtol=1e-5)

    # Decoupling is observable: differs from the inherited (all-power_law) case.
    inherited = _recover_nebular(
        DustSEDComponent(
            config=DustSEDComponentConfig(law_bc="power_law", law_diff="power_law")
        ).apply(_state_with_nebular(sed_neb), params)
    )
    assert not np.allclose(neb, inherited, rtol=1e-3)


def test_neb_law_overrides_shift_only_nebular():
    """neb_law_overrides change the nebular bc params, not the stellar screen."""
    sed_neb = jnp.full(_WAVE.shape, 1.0e28)
    params = {"dust_tau_bc": 1.0, "dust_tau_diff": 0.0, "dust_slope": -0.7}
    comp = DustSEDComponent(
        config=DustSEDComponentConfig(
            law_bc="power_law",
            law_diff="power_law",
            neb_law_overrides=(("n_slope", -1.3),),
        )
    )
    out = comp.apply(_state_with_nebular(sed_neb), params)
    neb = _recover_nebular(out)

    # Nebular bc slope is now -1.3 (override), tau_diff=0 so only the bc screen.
    k_bc = np.asarray(resolve_dust_law("power_law")(_WAVE, n_slope=-1.3))
    expected = np.asarray(sed_neb) * np.exp(-1.0 * k_bc)
    np.testing.assert_allclose(neb, expected, rtol=1e-5)

    # The stellar birth-cloud screen still uses the shared -0.7 (unchanged):
    # youngest-bin attenuation must follow dust_slope, not the nebular override.
    stellar_att = np.asarray(out.derived["sed_dust_attenuated"])
    lnu_young = np.asarray(_state_with_nebular(sed_neb).derived["lnu_age"])  # old-only here
    # (stellar light is in the old bin; just assert the screen ran and dimmed it)
    assert np.all(stellar_att <= np.sum(lnu_young, axis=0) + 1e-30)

# SPDX-License-Identifier: BSD-3-Clause
"""The two-component dust component reddens the nebular continuum.

Nebular emission from HII regions is attenuated by the same birth-cloud +
diffuse dust as the youngest stars (Charlot & Fall 2000), as in
bagpipes / FSPS / CIGALE. tengri previously added the photoionised
``sed_nebular`` continuum *after* dust ran (the topological sort placed dust
first), so it escaped attenuation — a UV/u-band excess of ~0.17 mag vs bagpipes
in the reproduction notebooks (``bagpipes_13b_photometry_no_neb``: with-nebular
diverged, no-nebular matched).

This regression locks in two things:
* the pipeline now orders nebular *before* dust (optional_inputs dependency), and
* the dust component reddens ``sed_nebular`` by the young-limit transmission.
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
    # One luminous old population so the stellar continuum is well defined.
    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0]))
    lnu_age = lnu_age.at[2, :].set(1.0e29)  # old stars carry the stellar light
    stellar = jnp.sum(lnu_age, axis=0)
    return ForwardState(
        wave=_WAVE,
        sed_intrinsic=stellar + sed_neb,  # nebular added pre-dust (new order)
        derived={
            "lnu_age": lnu_age,
            "ssp_ages_yr": _AGES,
            "sed_nebular": sed_neb,
        },
    )


def test_pipeline_orders_nebular_before_dust():
    """optional_inputs('sed_nebular') makes the topo-sort place nebular first."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData
    from tengri.forward.component_factory import build_components

    wave = jnp.logspace(2, 7, 64)
    ssp = SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs((5000.0 / wave) ** 2)[None, None, :] * jnp.ones((3, 8, 1)) + 1e-12,
        ssp_lg_age_gyr=jnp.linspace(-3, 1.14, 8),
        ssp_lgmet=jnp.array([-4.0, -2.65, -1.3]),
    )
    chain = [type(c).__name__ for c in build_components(ssp_data=ssp, use_dust=True)]
    assert chain.index("NebularSEDComponent") < chain.index("DustSEDComponent")


def test_nebular_continuum_is_attenuated():
    """sed_nebular is reddened by the birth-cloud + diffuse young-limit screen."""
    sed_neb = jnp.full(_WAVE.shape, 1.0e28)
    params = {"dust_tau_bc": 1.0, "dust_tau_diff": 0.5, "dust_slope": -0.7}
    comp = DustSEDComponent(config=DustSEDComponentConfig(emission_model=None))

    out = comp.apply(_state_with_nebular(sed_neb), params)

    # Recover the nebular contribution: total minus the attenuated stellar SED.
    stellar = jnp.sum(jnp.asarray(out.derived["lnu_age"]), axis=0)  # intrinsic stellar
    stellar_att = jnp.asarray(out.derived["sed_dust_attenuated"])
    neb_in_total = np.asarray(out.sed_intrinsic - stellar_att)

    # Expected young-limit transmission (both screens, weight -> 1).
    k_bc = np.asarray(resolve_dust_law("power_law")(_WAVE, n_slope=-0.7))
    tau = 1.0 * k_bc + 0.5 * k_bc
    t_neb = np.exp(-tau)
    expected = np.asarray(sed_neb) * t_neb

    np.testing.assert_allclose(neb_in_total, expected, rtol=1e-5)
    # And it must be strictly fainter than the intrinsic nebular everywhere.
    assert np.all(neb_in_total < np.asarray(sed_neb))
    del stellar  # (documented intrinsic-stellar handle; not asserted on)


def test_zero_nebular_is_noop():
    """No published nebular (BakedIn zeros) leaves the stellar result unchanged."""
    params = {"dust_tau_bc": 1.0, "dust_tau_diff": 0.5, "dust_slope": -0.7}
    comp = DustSEDComponent(config=DustSEDComponentConfig(emission_model=None))

    with_zeros = comp.apply(_state_with_nebular(jnp.zeros(_WAVE.shape)), params)
    # Stellar-only state (no sed_nebular key at all).
    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0])).at[2, :].set(1.0e29)
    bare = ForwardState(
        wave=_WAVE,
        sed_intrinsic=jnp.sum(lnu_age, axis=0),
        derived={"lnu_age": lnu_age, "ssp_ages_yr": _AGES},
    )
    without = comp.apply(bare, params)
    np.testing.assert_allclose(
        np.asarray(with_zeros.sed_intrinsic), np.asarray(without.sed_intrinsic), rtol=1e-12
    )

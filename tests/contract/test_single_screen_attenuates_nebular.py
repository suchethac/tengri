# SPDX-License-Identifier: BSD-3-Clause
"""Single-screen dust components redden the nebular continuum.

The two-component :class:`DustSEDComponent` was fixed in #668/#690 to redden the
photoionized ``sed_nebular`` continuum with the young-limit birth-cloud + diffuse
screen. The two *single-screen* siblings —
:class:`~tengri.components.dust.component.DustAttenuationSEDComponent`
(``dust_model="single_component"``) and
:class:`~tengri.components.dust.wg00_model.WG00AttenuationSEDComponent`
(``dust_model="wg00"``, FSPS ``dust_type=3``) — never got the analogous fix, so
the nebular continuum escaped attenuation entirely (the u-band excess of #668,
single-screen edition).

Unlike the two-component model, a single screen multiplies the *whole*
``sed_intrinsic``. The nebular component folds its continuum into
``sed_intrinsic``, so the only requirement is **ordering**: nebular must run
*before* dust. Declaring ``sed_nebular`` an ``optional_inputs`` key makes the
ADR-0006 topological sort place nebular first; the single law then attenuates
the nebular continuum together with the stellar light, matching
bagpipes/FSPS/CIGALE.

This regression locks in:
* the pipeline orders nebular *before* both single-screen dust components, and
* a non-zero nebular continuum folded into ``sed_intrinsic`` is attenuated.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.protocols.component import ForwardState

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_WAVE = jnp.array([1500.0, 2700.0, 3550.0, 5500.0, 9000.0])


def _synthetic_ssp():
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    wave = jnp.logspace(2, 7, 64)
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs((5000.0 / wave) ** 2)[None, None, :] * jnp.ones((3, 8, 1)) + 1e-12,
        ssp_lg_age_gyr=jnp.linspace(-3, 1.14, 8),
        ssp_lgmet=jnp.array([-4.0, -2.65, -1.3]),
    )


@pytest.mark.parametrize(
    ("dust_model", "dust_cls_name"),
    [
        ("single_component", "DustAttenuationSEDComponent"),
        ("wg00", "WG00AttenuationSEDComponent"),
    ],
)
def test_pipeline_orders_nebular_before_single_screen(dust_model, dust_cls_name):
    """optional_inputs('sed_nebular') makes the topo-sort place nebular first."""
    from tengri.forward.component_factory import build_components

    chain = [
        type(c).__name__
        for c in build_components(
            ssp_data=_synthetic_ssp(),
            use_dust=True,
            dust_model=dust_model,
            nebular_backend="baked_in",
        )
    ]
    assert chain.index("NebularSEDComponent") < chain.index(dust_cls_name)


def test_single_component_attenuates_folded_nebular():
    """The single screen reddens the nebular continuum carried in sed_intrinsic."""
    from tengri.components.dust.attenuation import calzetti
    from tengri.components.dust.component import (
        DustAttenuationSEDComponent,
        DustAttenuationSEDComponentConfig,
    )

    stellar = jnp.full(_WAVE.shape, 1.0e29)
    sed_neb = jnp.full(_WAVE.shape, 1.0e28)
    # Nebular has already been folded into sed_intrinsic by NebularSEDComponent
    # (which runs first thanks to the new ordering edge).
    state = ForwardState(
        wave=_WAVE,
        sed_intrinsic=stellar + sed_neb,
        derived={"sed_nebular": sed_neb},
    )
    comp = DustAttenuationSEDComponent(config=DustAttenuationSEDComponentConfig(law="calzetti"))
    out = comp.apply(state, {"dust_tau_v": 1.0})

    t = np.exp(-1.0 * np.asarray(calzetti(_WAVE)))
    expected = np.asarray(stellar + sed_neb) * t
    np.testing.assert_allclose(np.asarray(out.sed_intrinsic), expected, rtol=1e-6)
    # The nebular slice is strictly fainter than its intrinsic value everywhere.
    assert np.all(t < 1.0)

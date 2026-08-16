# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #981 — absolute α_ox default fed into a delta-offset knob.

Bug: ``AGNXRayCoronaSEDComponent`` declared ``alpha_ox = Fixed(-1.4)`` (an
*absolute* Lusso–Risaliti value) but forwarded it as
``xray_agn_corona(..., delta_alpha_ox=p["alpha_ox"])`` — a *delta* on top of
the Just+2007 empirical α_ox(L_2500) (post-#722 semantics). At defaults the
corona landed at α_ox ≈ −2.8, i.e. 2.4×10⁻⁴ of the Just+2007 prediction.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


class TestAGNXRayCoronaDefaults:
    def test_default_corona_matches_just2007_identity(self):
        """At component defaults the 2 keV output must sit on the Just+2007
        relation (delta = 0), not 3.6 dex below it."""
        from tengri.components.xray.agn_xray_model import AGNXRayCoronaSEDComponent

        comp = AGNXRayCoronaSEDComponent()
        wave = jnp.logspace(0.0, 3.0, 512)  # 1–1000 Å, covers E > 0.1 keV
        sed_in = jnp.zeros_like(wave)
        l_2500 = 3.16e29  # erg/s/Hz — inside the Just+2007 calibration window

        # Defaults come from the xray group's declarations, because that is
        # where this component's parameters live: it declares none of its own
        # since #1684 (they were duplicates of xray_gamma_agn / xray_E_cut /
        # xray_delta_alpha_ox under a prefix no group supplied, which is what
        # made it unbuildable). Read off the declaration rather than repeating
        # the numbers, per ADR-0011.
        from tengri.components.xray._params import PARAMS as XRAY_PARAMS
        from tengri.protocols.component import declared_default

        defaults = {
            "gamma_agn": jnp.asarray(declared_default(XRAY_PARAMS, "xray_gamma_agn")),
            "E_cut": jnp.asarray(declared_default(XRAY_PARAMS, "xray_E_cut")),
            "delta_alpha_ox": jnp.asarray(declared_default(XRAY_PARAMS, "xray_delta_alpha_ox")),
        }
        sed_out, published = comp.predict(
            defaults, sed_in, wave, L_2500_30deg=jnp.asarray(l_2500), L_agn_bol=0.0
        )

        alpha_ox = -0.137 * np.log10(l_2500) + 2.638  # Just+2007 Eq. 3
        l_2kev_expected = l_2500 * 10.0 ** (alpha_ox / 0.3838)

        wave_np = np.asarray(wave)
        sed_np = np.asarray(sed_out)
        l_2kev = float(np.interp(12.398 / 2.0, wave_np, sed_np))
        assert l_2kev / l_2kev_expected == pytest.approx(1.0, abs=0.05)
        # Publishes sed_xray (a real DerivedState field) rather than
        # L_xray_agn, which was not one -- see #1684.
        assert float(np.sum(np.asarray(published["sed_xray"]))) > 0

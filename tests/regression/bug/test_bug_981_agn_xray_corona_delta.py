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

        defaults = {name: jnp.asarray(dist.default) for name, dist in type(comp)._priors.items()}
        sed_out, published = comp.predict(
            defaults, sed_in, wave, L_2500_30deg=jnp.asarray(l_2500), L_agn_bol=0.0
        )

        alpha_ox = -0.137 * np.log10(l_2500) + 2.638  # Just+2007 Eq. 3
        l_2kev_expected = l_2500 * 10.0 ** (alpha_ox / 0.3838)

        wave_np = np.asarray(wave)
        sed_np = np.asarray(sed_out)
        l_2kev = float(np.interp(12.398 / 2.0, wave_np, sed_np))
        assert l_2kev / l_2kev_expected == pytest.approx(1.0, abs=0.05)
        assert float(published["L_xray_agn"]) > 0

# SPDX-License-Identifier: BSD-3-Clause
"""Regression: polar dust is ONE mechanism, not three (task13 fix-round-1, R22).

Before this fix the composable AGN path implemented polar dust three times:

1. The runner's Stage-1.5 ``_disc_ext`` -- Type-1-gated line-of-sight disc
   reddening, applied whenever ``agn_polar_ebv > 0`` regardless of the
   *selected* ``agn_attenuation_block`` (even ``"none"``) and regardless of
   ``agn_norm``, with NO re-emission credit for the removed energy.
2. ``skirtor_torus_block``'s bundled Casey (2012) polar graybody, added to
   the SKIRTOR thermal dust whenever ``agn_polar_ebv > 0`` (the CIGALE
   default, 0.03).
3. The standalone ``polar_dust`` attenuation block (Stage 5) + its Stage-6
   re-emission -- the only mechanism that credited the absorbed energy back
   as a graybody.

Under ``torus="skirtor"`` + ``atten="polar_dust"`` the disc screen applied
TWICE (mechanisms 1 and 3, plus mechanism 2's own graybody on the torus
side); under any other attenuation-block selection mechanism 1 leaked energy
with no re-emission credit at all. There is now exactly ONE mechanism: the
standalone ``polar_dust`` attenuation block owns line-of-sight reddening
(Type-1 sightlines only, via the existing smooth Type-1/2 mask) and isotropic
re-emission end to end. This is a deliberate behavior change: polar dust now
applies ONLY when ``agn_attenuation_block == "polar_dust"`` is explicitly
selected -- the ``agn_polar_ebv`` default (0.03) no longer silently reddens
every AGN SED.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks.atten import polar_dust_attenuation_block
from tengri.components.agn.blocks.runner import composable_agn_l_nu

pytestmark = pytest.mark.regression_bug

_WAVE = jnp.logspace(jnp.log10(500.0), jnp.log10(1e8), 300)

_TORUS_KWARGS = {
    "skirtor": dict(
        agn_tau_skirtor=7.0,
        agn_p_skirtor=1.0,
        agn_q_skirtor=1.0,
        agn_oa_skirtor=40.0,
    ),
    "fritz": dict(
        agn_fritz_r_ratio=60.0,
        agn_fritz_tau=1.0,
        agn_fritz_beta=-0.5,
        agn_fritz_gamma=4.0,
        agn_fritz_oa=60.0,
        agn_fritz_psy=0.001,
    ),
}


def _base_params(torus_type: str, atten_type: str) -> dict:
    return dict(
        agn_log_lbol=12.0,
        agn_lum_ratio=1.0,
        agn_disc_block="multicolor",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block=torus_type,
        agn_attenuation_block=atten_type,
        agn_norm="independent",
        agn_cos_inc=0.7,
        agn_torus_frac=0.5,
        **_TORUS_KWARGS[torus_type],
    )


@pytest.mark.parametrize("torus_type", ["skirtor", "fritz"])
@pytest.mark.parametrize("atten_type", ["none", "qsogen"])
def test_polar_dust_inert_when_atten_is_not_polar_dust(torus_type, atten_type):
    """agn_polar_{ebv,T,beta,oa} must be bit-identical no-ops whenever the
    selected attenuation block is NOT ``polar_dust`` -- for every torus.

    Previously mechanism 1 (runner Stage-1.5) reddened the disc regardless
    of the attenuation-block choice, and mechanism 2 (bundled in
    skirtor_torus_block) added a polar graybody regardless of the
    attenuation-block choice too. Both are gone: with neither ``"none"`` nor
    ``"qsogen"`` selected as the attenuation block, none of the four polar
    knobs can reach the computation graph at all.

    Each knob is changed ONE AT A TIME against a shared baseline (rather
    than all four at once). Mechanism 1's own Type-1/2 sigmoid mask read
    ``agn_polar_oa`` directly, so a joint perturbation that also swings
    ``agn_polar_oa`` across that mask's threshold can silently cancel
    mechanism 1's ``agn_polar_ebv`` effect against a fixed ``agn_cos_inc`` --
    confirmed empirically: the joint form of this test (oa 10->80, ebv
    0.0->0.3, at agn_cos_inc=0.7) passed bit-exact even when run against the
    pre-R22 815104ad1 source tree, because oa=10 lands Type-1 and oa=80
    lands Type-2 for that inclination, masking mechanism 1's screen at the
    "on" state entirely. Isolating ``agn_polar_ebv`` alone (oa fixed at 30
    deg, comfortably Type-1 for ``agn_cos_inc=0.7`` in both states) reveals
    mechanism 1 cleanly: ``sum(sed)`` moved from ``1.880088e+33`` to
    ``4.719414e+33`` (torus=skirtor, atten=none) against that same pre-R22
    tree -- a 2.5x change, not the bit-exact match this test now requires.
    """
    base = _base_params(torus_type, atten_type)
    default = {
        **base,
        "agn_polar_ebv": 0.03,
        "agn_polar_T": 100.0,
        "agn_polar_beta": 1.6,
        "agn_polar_oa": 30.0,
    }
    sed_default = np.asarray(composable_agn_l_nu(_WAVE, **default))
    for key, alt_value in (
        ("agn_polar_ebv", 0.3),
        ("agn_polar_T", 300.0),
        ("agn_polar_beta", 2.0),
        ("agn_polar_oa", 80.0),
    ):
        sed_alt = np.asarray(composable_agn_l_nu(_WAVE, **{**default, key: alt_value}))
        np.testing.assert_array_equal(
            sed_default,
            sed_alt,
            err_msg=(
                f"torus={torus_type!r}, atten={atten_type!r}: changing {key} alone "
                "changed the SED even though a non-polar_dust attenuation block was "
                "selected."
            ),
        )


def test_polar_dust_knobs_all_live_when_atten_is_polar_dust():
    """The inverse of the test above: selecting atten='polar_dust' makes all
    four polar knobs live, with nonzero gradient."""
    base = _base_params("skirtor", "polar_dust")
    base.update(agn_polar_ebv=0.3, agn_polar_T=100.0, agn_polar_beta=1.6, agn_polar_oa=45.0)

    def _grad(key, value):
        def obj(v):
            return jnp.sum(composable_agn_l_nu(_WAVE, **{**base, key: v}))

        return float(jax.grad(obj)(value))

    for key, value in (
        ("agn_polar_ebv", 0.3),
        ("agn_polar_T", 100.0),
        ("agn_polar_beta", 1.6),
        ("agn_polar_oa", 45.0),
    ):
        g = _grad(key, value)
        assert g != 0.0, f"{key} has zero gradient with atten='polar_dust' selected."


def test_no_double_polar_screen():
    """torus=skirtor + atten=polar_dust: the disc attenuation is exactly ONE
    application of the polar_dust screen.

    Before R22, skirtor_torus_block's bundled polar term did not itself
    re-attenuate the disc, but the runner's Stage-1.5 ``_disc_ext`` DID --
    on top of Stage 5's own ``polar_dust_attenuation_block`` factor -- so the
    disc was screened twice (an extra ``factor`` in the ratio below, not a
    second additive graybody). Compare the composable runner's own
    ``sed_agn_disc`` component (ebv=0.3 vs ebv=0, a ratio that cancels
    everything upstream of the screen) against ONE direct call to the
    standalone attenuation block at the same params.
    """
    base = _base_params("skirtor", "polar_dust")
    base.update(agn_polar_T=100.0, agn_polar_beta=1.6, agn_polar_oa=20.0)
    ebv = 0.3

    _, comp0 = composable_agn_l_nu(_WAVE, return_components=True, **{**base, "agn_polar_ebv": 0.0})
    _, comp1 = composable_agn_l_nu(_WAVE, return_components=True, **{**base, "agn_polar_ebv": ebv})

    disc0 = np.asarray(comp0["disc"])
    disc1 = np.asarray(comp1["disc"])
    observed_ratio = np.where(disc0 != 0.0, disc1 / disc0, 1.0)

    single_factor = np.asarray(
        polar_dust_attenuation_block(
            _WAVE,
            agn_polar_ebv=ebv,
            agn_cos_inc=base["agn_cos_inc"],
            agn_polar_oa=base["agn_polar_oa"],
            agn_polar_law="smc",
        )
    )
    np.testing.assert_allclose(
        observed_ratio,
        single_factor,
        rtol=1e-6,
        atol=1e-12,
        err_msg="disc attenuation is not exactly ONE application of the polar_dust screen.",
    )
    # A genuine double-application (the pre-R22 bug for torus=skirtor) would
    # show up as single_factor SQUARED, not single_factor -- a sanity check
    # that this test is actually discriminating, not vacuously passing.
    assert not np.allclose(observed_ratio, single_factor**2, rtol=1e-3), (
        "observed ratio matches single_factor**2 -- the assertion above would "
        "not have caught a reintroduced double screen."
    )

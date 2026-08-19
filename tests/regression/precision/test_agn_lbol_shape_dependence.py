# SPDX-License-Identifier: BSD-3-Clause
r"""The AGN float32 factoring must not corrupt the L_bol-dependent disc shape (#1206).

The float32 boundary for the composable AGN evaluates every block at a low
reference bolometric luminosity and re-applies ``10^(agn_log_lbol −
_AGN_LBOL_REF)`` in log space (``apply_log10_scale``) — the ``L_agn_bol`` ~1e46
erg/s scale overflows float32 otherwise. That output factoring is only valid for
blocks whose *spectral shape* is invariant under L_bol (the SKIRTOR torus
template, the power-law disc). The **multicolor disc** temperature rises with
L_bol, so its 2500/4400 Å monochromatic luminosities are sub-linear in L_bol
(``L ∝ L_bol^{~0.7}``, not ``∝ L_bol``); factoring them from a cold reference
disc gives values ~200 orders of magnitude too small (observed
``L_4400_intrinsic = 6.9e-171`` where the physical value is ~2e28 erg/s/Hz).

Those two intrinsic luminosities drive X-ray ``alpha_ox`` (via
``L_2500_intrinsic``) and the radio-loudness reference (via
``L_4400_intrinsic``), so the corruption silently propagated. It went unnoticed
because ``L_2500``/``L_4400`` ~1e28 are *already* float32-representable and were
never needed in factored form — the fix is to evaluate the AGN at the true
``agn_log_lbol`` in float64 (the reference path, bit-identical to pre-#1206
main), reserving the reference factoring for float32 only.

These are float64 correctness guards: they fail on the broken uniform-factoring
and pass on the true-``log_lbol`` evaluation.
"""

import jax
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug


def _intrinsic_luminosities(ssp, disc, lbol):
    """Return ``(L_4400_intrinsic, L_2500_intrinsic)`` [erg/s/Hz] in float64."""
    with jax.enable_x64(True):
        obs = Observation(photometry=Photometry.from_names(["sdss_r", "wise_w3", "wise_w4"]))
        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": "delayed",
                "all_params": FIXED,
                "log_total_mass": Uniform(9.0, 11.0),
                "tau_gyr": 1.0,
                "age_gyr": 5.0,
            },
            dust={"law_diff": 'calzetti', 
                "type": "two_component",
                "law_bc": "calzetti",
                "all_params": FIXED,
                "tau_diff": 0.3,
                "tau_bc": 0.0,
            },
            agn={
                "type": "composable",
                "all_params": FIXED,
                "disc": {"type": disc, "all_params": FIXED},
                "torus": {"type": "skirtor", "all_params": FIXED},
                "norm": "cigale_joint",
                "log_lbol": Uniform(9.0, 13.0),
                "fracAGN": 0.1,
            },
            redshift=Fixed(0.1),
        )
        d = model.predict_state(
            {"sfh_delayed_log_total_mass": 10.0, "agn_log_lbol": float(lbol)}
        ).derived
        return float(np.asarray(d["L_4400_intrinsic"])), float(np.asarray(d["L_2500_intrinsic"]))


def test_multicolor_disc_intrinsic_luminosities_are_physical(ssp_bare):
    """``L_4400``/``L_2500`` must be ~1e27–1e30 erg/s/Hz, not underflow-tiny (f64)."""
    l4400, l2500 = _intrinsic_luminosities(ssp_bare, "multicolor", 11.0)
    # The uniform-factoring bug produced ~6.9e-171 / ~3.8e-170; the physical disc
    # gives ~2e28 / ~2.3e28. A 1e10 floor catches the corruption with vast margin.
    assert l4400 > 1.0e10, (
        f"L_4400_intrinsic = {l4400:.3e} erg/s/Hz — the multicolor disc's optical "
        "luminosity was factored from a cold reference disc (float32 hack leaked "
        "into float64), losing ~200 decades"
    )
    assert l2500 > 1.0e10, f"L_2500_intrinsic = {l2500:.3e} erg/s/Hz — same corruption"


def test_multicolor_disc_L4400_is_sublinear_in_lbol(ssp_bare):
    """The disc optical luminosity scales sub-linearly (shape changes with L_bol).

    The broken factoring forced an exact ``×10`` per dex (linear); the physical
    multicolor disc gives ``~×4.8`` (``L ∝ L_bol^{~0.7}``). Asserting the ratio
    is well below 10 both proves the shape dependence is captured and locks out
    a regression to uniform factoring.
    """
    l4400_11, _ = _intrinsic_luminosities(ssp_bare, "multicolor", 11.0)
    l4400_12, _ = _intrinsic_luminosities(ssp_bare, "multicolor", 12.0)
    ratio = l4400_12 / l4400_11
    assert 2.0 < ratio < 8.0, (
        f"L_4400(lbol=12)/L_4400(lbol=11) = {ratio:.3f}; expected ~4.8 "
        "(sub-linear disc). A value of ~10 means uniform 10^log_lbol factoring "
        "leaked back in and flattened the L_bol-dependent disc shape"
    )


def test_powerlaw_disc_is_shape_invariant(ssp_bare):
    """Control: the power-law disc IS linear in L_bol, so factoring is exact there.

    This pins that the fix is disc-shape-specific — the power-law disc must keep
    its clean ``×10`` per dex, confirming the multicolor assertion above is not
    just loose tolerance.
    """
    l4400_11, _ = _intrinsic_luminosities(ssp_bare, "powerlaw", 11.0)
    l4400_12, _ = _intrinsic_luminosities(ssp_bare, "powerlaw", 12.0)
    ratio = l4400_12 / l4400_11
    assert abs(ratio - 10.0) < 0.5, (
        f"power-law disc L_4400 ratio = {ratio:.3f}; expected ~10 (shape-invariant, "
        "linear in L_bol)"
    )

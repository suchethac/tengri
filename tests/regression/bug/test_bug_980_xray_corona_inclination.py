# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #980 — yang20 corona anisotropy never saw the inclination.

Bug: ``XraySEDComponent._emit`` called ``xray_total`` without ``cos_inc``, so
``xray_anisotropy`` (Yang et al. 2022; X-CIGALE ``yang20.py:231-235``) always
evaluated face-on (μ = 1), a flat ×1.0718 at the default a1 = 0.5, a2 = 0 —
and varying ``agn_cos_inc`` changed the disc and torus but never the corona.

The anisotropy normalization anchors the α_ox-derived L_2keV at the Yang+2020
30° reference: the denominator ``1 − 0.13397 a1 − 0.25 a2`` is the numerator
at μ = cos 30° (0.13397 = 1 − cos 30°, 0.25 = 1 − cos² 30°), so f(cos 30°) = 1.

Isolated against CIGALE 2025.1 run live: at matched disc L_2500 (ratio 1.0009)
CIGALE's corona satisfies L_2keV = L_2500·10^(α_ox/0.3838) to 1.0000 while
tengri's own emitted L_2keV was 1.0707× its own α_ox prediction — entirely
this factor.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.xray.xray import (
    COS_INC_REF_30DEG,
    xray_anisotropy,
)

pytestmark = pytest.mark.regression_bug

_COS30 = float(np.cos(np.radians(30.0)))
_COS60 = float(np.cos(np.radians(60.0)))


def _anisotropy_factor(cos_inc: float, a1: float = 0.5, a2: float = 0.0) -> float:
    """Independent numpy reference for the Yang+2022 factor (yang20.py:231-235)."""
    return (a1 * cos_inc + a2 * cos_inc**2 + (1.0 - a1 - a2)) / (1.0 - 0.13397 * a1 - 0.25 * a2)


class TestAnisotropyAnchor:
    """The α_ox-derived L_2keV is the 30° value — f(cos 30°) ≡ 1."""

    def test_reference_constant_is_cos_30(self):
        assert pytest.approx(_COS30, rel=1e-12) == COS_INC_REF_30DEG

    def test_factor_is_unity_at_30deg(self):
        factor = float(xray_anisotropy(jnp.asarray(1.0), _COS30))
        assert factor == pytest.approx(1.0, rel=1e-4)

    def test_factor_matches_xcigale_formula_off_reference(self):
        for mu in (1.0, _COS60, 0.2):
            factor = float(xray_anisotropy(jnp.asarray(1.0), mu))
            assert factor == pytest.approx(_anisotropy_factor(mu), rel=1e-12)


class TestCoronaSeesInclination:
    """e2e: ``agn_cos_inc`` must reach the corona through the xray component."""

    @pytest.fixture(scope="class")
    def build_state(self, synthetic_ssp_wide):
        from tengri import FIXED, Fixed, SEDModel

        def _build(cos_inc: float):
            model = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                sfh={"type": "delayed", "all_params": FIXED},
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "all_params": FIXED,
                },
                agn={
                    "type": "composable",
                    "disc": {"type": "schartmann2005", "all_params": FIXED},
                    "agn_log_lbol": Fixed(11.5),
                    "agn_cos_inc": Fixed(cos_inc),
                    "all_params": FIXED,
                },
                xray={"type": "yang20", "all_params": FIXED},
                redshift=Fixed(0.0),
            )
            return model.predict_state({})

        return _build

    def test_agn_cos_inc_changes_corona(self, build_state):
        """Bug symptom: the corona was identical at every inclination."""
        state_30 = build_state(_COS30)
        state_60 = build_state(_COS60)
        wave = np.asarray(state_30.wave)
        xband = wave < 124.0  # E > 0.1 keV
        sed_30 = np.asarray(state_30.derived["sed_xray"])[xband]
        sed_60 = np.asarray(state_60.derived["sed_xray"])[xband]
        good = sed_30 > 0
        assert good.any(), "no corona flux in the X-ray band"
        ratio = np.median(sed_60[good] / sed_30[good])
        expected = _anisotropy_factor(_COS60) / _anisotropy_factor(_COS30)
        assert ratio == pytest.approx(expected, rel=1e-3)

    def test_corona_matches_alpha_ox_identity_at_reference(self, build_state):
        """At the 30° anchor, L_2keV must equal L_2500·10^(α_ox/0.3838) up to
        the documented line-of-sight terms (tbabs·cabs + 1% scattered floor,
        exp(−2/300) cutoff — a net ×0.999 at 2 keV, not ×1.07)."""
        state = build_state(_COS30)
        wave = np.asarray(state.wave)
        sed = np.asarray(state.derived["sed_xray"])
        l_2500 = float(np.asarray(state.derived["L_2500_intrinsic"]))
        assert l_2500 > 0

        # Independent identity: Just+2007 Eq. 3 with the paper coefficients.
        alpha_ox = -0.137 * np.log10(l_2500) + 2.638
        l_2kev_expected = l_2500 * 10.0 ** (alpha_ox / 0.3838)

        w_2kev = 12.398 / 2.0
        order = np.argsort(wave)
        l_2kev = float(np.interp(w_2kev, wave[order], sed[order]))
        assert l_2kev / l_2kev_expected == pytest.approx(1.0, abs=0.02)

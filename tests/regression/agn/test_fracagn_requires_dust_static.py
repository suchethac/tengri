# SPDX-License-Identifier: BSD-3-Clause
"""fracAGN (``agn_ir_frac``) refused at construction when no dust publishes L_absorbed.

The CIGALE skirtor2016 coupling ``agn_power = L_absorbed * f / (1 - f)`` is
defined only when a dust screen supplies ``L_absorbed``. Without one the forward
path used to fall back to ``1e-30`` erg/s and the torus fraction collapsed to ~0,
silently ignoring the user's ``agn_ir_frac``.

The refusal is static: it reads the resolved spec, so it fires whenever the
coupling can be active (``agn_ir_frac`` FREE, or Fixed to a value > 0), and it
fires on every construction path, not only :meth:`SEDModel.build`. A
``Fixed(0)`` coupling (the default) stays silent and the forward path is
unchanged.

Tests:
a) build, no dust, fracAGN Fixed(0.3) or FREE -> ValueError naming agn_ir_frac
b) build, dust present, fracAGN Fixed(0.3) -> builds
c) direct construction from a parsed spec, no dust, fracAGN Fixed(0.3) ->
   ValueError (the build-only guard of #944 was bypassed here)
d) no dust, fracAGN at its default -> builds silently, photometry unchanged
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from tengri import FREE, Fixed, SEDModel
from tengri.parameters.groups import parse_groups

pytestmark = pytest.mark.regression_bug

# Pre-change photometry of the dust-free, fracAGN-free composable AGN below
# (PRNGKey(0) sample), recorded on origin/main 7b2a7fbdf.
_REFERENCE_SUM = 1.2075885176529603e-11
_REFERENCE_FIRST = (5.455126164444879e-12, 2.9004212662948855e-12, 1.7384032925551905e-12)


def _agn(frac):
    return {
        "type": "composable",
        "norm": "cigale_joint",
        "torus": {"type": "skirtor"},
        "fracAGN": frac,
    }


def _kwargs(*, frac, dust):
    kw = dict(
        sfh={"type": "const"},
        agn=_agn(frac),
        redshift=Fixed(0.1),
    )
    if dust:
        kw["dust_attenuation"] = {"type": "two_component", "law": "calzetti"}
    else:
        kw["dust_attenuation"] = {"type": "none"}
    return kw


class TestFracAGNRequiresDustStatic:
    """Build-time refusal of an active fracAGN coupling with no dust component."""

    def test_build_no_dust_fixed_positive_raises(self, synthetic_ssp_wide, synthetic_tophat_obs):
        with pytest.raises(ValueError, match="agn_ir_frac"):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                **_kwargs(frac=Fixed(0.3), dust=False),
            )

    def test_build_no_dust_free_raises(self, synthetic_ssp_wide, synthetic_tophat_obs):
        with pytest.raises(ValueError, match="agn_ir_frac"):
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                **_kwargs(frac=FREE, dust=False),
            )

    def test_build_with_dust_builds(self, synthetic_ssp_wide, synthetic_tophat_obs):
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            **_kwargs(frac=Fixed(0.3), dust=True),
        )
        assert model.spec is not None

    def test_direct_construction_no_dust_positive_raises(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        spec = parse_groups(ssp_data=synthetic_ssp_wide, **_kwargs(frac=Fixed(0.3), dust=False))
        with pytest.raises(ValueError, match="agn_ir_frac"):
            SEDModel(spec, synthetic_ssp_wide, observation=synthetic_tophat_obs)

    def test_default_fracagn_no_dust_builds_silently_and_is_unchanged(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "const"},
                dust_attenuation={"type": "none"},
                agn={
                    "type": "composable",
                    "norm": "cigale_joint",
                    "torus": {"type": "skirtor"},
                },
                redshift=Fixed(0.1),
            )
        assert not [w for w in caught if "fracAGN" in str(w.message)]

        params = model.spec.sample(jax.random.PRNGKey(0))
        phot = np.asarray(model.predict_photometry(params))
        np.testing.assert_allclose(np.sum(phot), _REFERENCE_SUM, rtol=1e-10)
        np.testing.assert_allclose(phot[:3], _REFERENCE_FIRST, rtol=1e-10)

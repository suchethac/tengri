# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end wiring for per-component dust law parameters and law inheritance.

Guards that ``dust={'slope_bc': ...}`` (and friends) actually reach the forward
model — i.e. the builder -> spec -> SEDModel -> factory -> DustSEDComponent
chain is not a silent no-op — and that defaults reproduce the original
single-slope Charlot & Fall behavior. Also checks the symmetric law-inheritance
rule (set one law -> both components share it).

Uses the synthetic SSP + top-hat fixtures so it runs in default CI.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tengri

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _build(ssp, obs, **dust_extra):
    dust = {
        "law": "power_law",
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_bc": 1.0,
        "tau_diff": 0.0,
    }
    dust.update(dust_extra)
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "tsnorm", "*": tengri.FIXED},
        dust_attenuation=dust,
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=tengri.Fixed(0.05),
    )


def _phot(model):
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    return np.asarray(model.predict_photometry(p))


class TestPerComponentSlopeWiring:
    def test_slope_bc_changes_prediction(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """slope_bc must alter the SED — proves the override is not dropped.

        The synthetic SSP is old-light-dominated so the birth-cloud effect is
        small, but it must be strictly non-zero: a steeper birth-cloud slope
        reddens the young-star light that does exist. Non-identity is the
        no-op guard; the physics magnitude is covered by the curve harness and
        ``tests/physics/test_two_component_bc_diff_params.py``.
        """
        base = _phot(_build(synthetic_ssp_wide, synthetic_tophat_obs))
        steep = _phot(_build(synthetic_ssp_wide, synthetic_tophat_obs, slope_bc=-1.0))
        assert not np.array_equal(steep, base), (
            "slope_bc=-1.0 produced no change in photometry — wiring is a no-op"
        )

    def test_default_equals_explicit_minus_07(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """No override == explicitly setting both components to the -0.7 default."""
        base = _phot(_build(synthetic_ssp_wide, synthetic_tophat_obs))
        explicit = _phot(
            _build(
                synthetic_ssp_wide,
                synthetic_tophat_obs,
                slope_bc=-0.7,
                slope_diff=-0.7,
            )
        )
        np.testing.assert_allclose(explicit, base, rtol=1e-6)

    def test_bluest_band_fainter_with_steeper_bc(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Steeper birth-cloud slope -> more UV/blue attenuation -> fainter blue."""
        base = _phot(_build(synthetic_ssp_wide, synthetic_tophat_obs))
        steep = _phot(_build(synthetic_ssp_wide, synthetic_tophat_obs, slope_bc=-1.0))
        # Band 0 is the bluest (3500 A). It must not brighten.
        assert steep[0] <= base[0] + 1e-12


class TestLawInheritance:
    """Attenuation laws are explicit and required (no silent inheritance or
    power_law default) — see tests/contract/test_dust_law_grammar.py for the
    full grammar spec. These tests pin the two_component-specific angle:
    _build's base dust dict supplies its own law, so law_diff/law_bc must be
    passed WITHOUT it to actually exercise the incomplete-spec error paths.
    """

    def _dust_no_law(self, **extra):
        dust = {
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_bc": 1.0,
            "tau_diff": 0.0,
        }
        dust.update(extra)
        return dust

    def test_diff_only_raises_requires_both(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Setting only law_diff (no law_bc, no law) must raise."""
        with pytest.raises(ValueError, match="requires BOTH"):
            tengri.SEDModel.build(
                synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "tsnorm", "*": tengri.FIXED},
                dust_attenuation=self._dust_no_law(law_diff="calzetti"),
                neb={"type": "none"},
                redshift=tengri.Fixed(0.05),
            )

    def test_bc_only_raises_requires_both(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Setting only law_bc (no law_diff, no law) must raise."""
        with pytest.raises(ValueError, match="requires BOTH"):
            tengri.SEDModel.build(
                synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "tsnorm", "*": tengri.FIXED},
                dust_attenuation=self._dust_no_law(law_bc="calzetti"),
                neb={"type": "none"},
                redshift=tengri.Fixed(0.05),
            )

    def test_neither_raises_requires_law(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """No law key at all must raise — there is no silent default."""
        with pytest.raises(ValueError, match="requires either 'law'"):
            tengri.SEDModel.build(
                synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": "tsnorm", "*": tengri.FIXED},
                dust_attenuation=self._dust_no_law(),
                neb={"type": "none"},
                redshift=tengri.Fixed(0.05),
            )

    def test_shared_law_key_sets_both_screens(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """The explicit replacement for the old inheritance: law= sets both."""
        model = _build(synthetic_ssp_wide, synthetic_tophat_obs, law="cardelli")
        assert model.spec.dust_law_bc == "cardelli"
        assert model.spec.dust_law_diff == "cardelli"


class TestRoundTrip:
    def test_overrides_survive_to_groups(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """slope_bc / delta_diff round-trip through spec.to_groups()."""
        model = _build(
            synthetic_ssp_wide,
            synthetic_tophat_obs,
            slope_bc=-1.0,
            delta_diff=0.1,
        )
        groups = model.spec.to_groups()
        assert groups["dust_attenuation"]["slope_bc"] == -1.0
        assert groups["dust_attenuation"]["delta_diff"] == 0.1

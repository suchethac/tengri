# SPDX-License-Identifier: BSD-3-Clause
"""A key the grammar accepts must change something, or it must be refused.

The dust group used to accept every ``dust.emission`` parameter written at the
dust level. The comment said it was "for legacy code that flattens emission
params at the dust level ... still resolved via the dust.emission group path".
The acceptance shipped; the resolution never did. Measured against a baseline
that omits the key, **22 of 22** emission parameters were accepted and silently
discarded::

    dust_qpah   baseline 2.500  ->  flattened 2.500   DROPPED, 0 warnings
    dust_umin   ...             ->  ...               DROPPED, 0 warnings

With a prior it is worse than a wrong number — it is a missing dimension::

    dust={'emission': {...}, 'alpha': Uniform(1, 3)}   ->  dust_alpha free = False

The author believes they are fitting the dust-emission slope. They are not, and
nothing says so. For SED fitting that is a published quantity that was never
varied.

Refusing it cannot break working code: a form that silently does nothing can
have no caller that depends on the effect.

The same union exists for ``agn``, and there it genuinely resolves — 14 of 14
cross-level parameters applied — so this file pins the property ("accepted
implies effective") rather than deleting cross-level acceptance everywhere.
That is the distinction a blanket rule would have destroyed.
"""

from __future__ import annotations

import warnings

import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_SFH = {"type": "dpl", "all_params": FIXED}
_EMIS = {"type": "dale2014", "all_params": FIXED}


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "wise_w4"]))


def _build(ssp, obs, dust, dust_emission=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs = {
            "ssp_data": ssp,
            "observation": obs,
            "sfh": _SFH,
            "dust_attenuation": dust,
            "neb": {"type": "none"},
            "redshift": Fixed(0.1),
        }
        if dust_emission is not None:
            kwargs["dust_emission"] = dust_emission
        return SEDModel.build(**kwargs)


def _emission_params(ssp, obs) -> list[str]:
    """Parameters the dust_emission group contributes, by difference.

    Discovered rather than listed, so a new emission template is covered the
    day it registers.
    """
    base = {"type": "two_component", "law": "calzetti", "all_params": FIXED}
    without = set(_build(ssp, obs, dict(base)).spec.all_params)
    with_em = _build(ssp, obs, dict(base), dust_emission=_EMIS)
    return sorted(set(with_em.spec.all_params) - without)


class TestTheCensus:
    def test_the_emission_block_contributes_parameters(self, ssp_data_fsps, obs):
        """Without them this file would assert over an empty set."""
        params = _emission_params(ssp_data_fsps, obs)
        assert len(params) >= 10, f"only {len(params)} emission params found: {params}"
        assert "dust_alpha" in params


class TestFlattenedKeysAreRefusedNotDropped:
    def test_every_emission_param_is_refused_at_the_dust_level(self, ssp_data_fsps, obs):
        """Accepted-and-ignored is the failure; refusal is the fix."""
        base = {
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
        }
        accepted: list[str] = []
        for full in _emission_params(ssp_data_fsps, obs):
            short = full.removeprefix("dust_")
            baseline = float(
                _build(
                    ssp_data_fsps, obs, dict(base), dust_emission=_EMIS
                ).spec.get_fixed_values()[full]
            )
            try:
                # Try to set emission param at dust_attenuation level (should be refused)
                model = _build(
                    ssp_data_fsps, obs, {**base, short: baseline + 0.5}, dust_emission=_EMIS
                )
            except ValueError:
                continue  # refused — the contract
            got = float(model.spec.get_fixed_values()[full])
            if got == baseline:
                accepted.append(f"{short} (stayed {baseline})")
        assert not accepted, (
            f"these keys were accepted at the dust_attenuation level and silently discarded: "
            f"{accepted}. A key the grammar accepts must change something."
        )

    def test_the_refusal_names_the_nesting_that_works(self, ssp_data_fsps, obs):
        """'Unknown key alpha ... Did you mean: alpha?' is not a fix."""
        base = {
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "emission": _EMIS,
        }
        with pytest.raises(ValueError, match=r"dust_emission") as excinfo:
            _build(ssp_data_fsps, obs, {**base, "alpha": 2.5})
        message = str(excinfo.value)
        assert "emission" in message, message
        assert "Did you mean: alpha?" not in message, (
            f"the message suggests the key it just refused: {message}"
        )

    def test_a_prior_written_flat_no_longer_loses_a_fit_dimension(self, ssp_data_fsps, obs):
        """The silent version cost a free parameter, not just a value."""
        base = {
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
        }
        with pytest.raises(ValueError):
            # Try to set emission param at dust_attenuation level with a prior (should be refused)
            _build(ssp_data_fsps, obs, {**base, "alpha": Uniform(1.0, 3.0)}, dust_emission=_EMIS)

    def test_the_flat_form_works(self, ssp_data_fsps, obs):
        """The refusal is only useful if the recommended flat spelling succeeds."""
        model = _build(
            ssp_data_fsps,
            obs,
            {
                "type": "two_component",
                "law": "calzetti",
                "all_params": FIXED,
            },
            dust_emission={**_EMIS, "alpha": Uniform(1.0, 3.0)},
        )
        assert "dust_alpha" in model.spec.free_params


class TestCrossLevelAcceptanceThatWorksIsKept:
    def test_agn_still_accepts_and_applies_its_cross_level_names(self, ssp_data_fsps, obs):
        """agn resolves what it accepts (14/14 measured); do not break it.

        A blanket "no cross-level keys anywhere" rule would have removed a
        working affordance to fix a broken one.
        """
        agn = {
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "multicolor", "all_params": FIXED},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh=_SFH,
                dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
                neb={"type": "none"},
                redshift=Fixed(0.1),
                agn=agn,
            )
        name = next((p for p in sorted(base.spec.all_params) if p.startswith("agn_")), None)
        assert name, "no agn params declared — fixture cannot test this"
        baseline = float(base.spec.get_fixed_values()[name])
        short = name.removeprefix("agn_")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh=_SFH,
                dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
                neb={"type": "none"},
                redshift=Fixed(0.1),
                agn={**agn, short: baseline + 0.25},
            )
        assert float(model.spec.get_fixed_values()[name]) != baseline, (
            f"agn cross-level key {short!r} stopped taking effect"
        )

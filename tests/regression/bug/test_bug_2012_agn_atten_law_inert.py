# SPDX-License-Identifier: BSD-3-Clause
"""#2012: agn.atten validated 22 law names and applied one curve to all of them.

`agn={'atten': {'law': ...}}` checked its argument against every entry in
``DUST_LAWS`` and then mapped all of them to the single ``smc_prevot`` block, so
a user who selected Calzetti for their AGN attenuation silently got Prevot SMC.
Measured at the time: five distinct law names produced **bit-identical** SEDs at
E(B-V)=0.4, while the same comparison saw E(B-V) itself change the SED.

The validation is what made it a trap rather than a documented limit. Passing a
typo got a careful correction naming all 22 options, which told the user the
choice was real; passing a real name got silent substitution.

Fixed by validating against the law the block implements, which is the policy
``foreground`` has always used for its own single-curve limitation.

Why this file and not ``test_registry_choice_is_distinct.py``, which exists to
prove registry choices are load-bearing: that census models two selector
spellings, ``type`` and dust's ``law_bc``. ``agn.atten``'s ``law`` is a third,
introduced without extending the census -- so the new selection surface shipped
outside the coverage of the test whose whole job is catching this. Extending
that census is the better long-term home; this file pins the behavior now.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

import tengri
from tengri import FIXED, Fixed, Observation, Photometry, SEDModel

pytestmark = [pytest.mark.regression_bug, pytest.mark.contract]


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))


@pytest.fixture(scope="module")
def ssp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tengri.load_ssp("fsps_prsc_miles_chabrier")


def _build(ssp, obs, law: str, ebv: float = 0.4):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FIXED},
            agn={
                "type": "composable",
                "all_params": FIXED,
                "disc": {"type": "powerlaw", "all_params": FIXED},
                "atten": {"law": law, "attenuation_ebv": ebv},
            },
            redshift=Fixed(1.0),
        )


def _sed(ssp, obs, law: str, ebv: float = 0.4) -> np.ndarray:
    model = _build(ssp, obs, law, ebv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        return np.asarray(model.predict(params).rest_sed())


class TestOnlyImplementedLawsAreAccepted:
    def test_the_implemented_law_is_accepted(self, ssp, obs):
        assert _build(ssp, obs, "prevot_smc") is not None

    @pytest.mark.parametrize("law", ["calzetti", "smc", "cardelli", "power_law"])
    def test_a_real_but_unwired_law_is_refused(self, ssp, obs, law):
        """Refusal, not substitution. These all used to be accepted silently.

        ``smc`` is included deliberately: it is the nearest neighbor by name
        and still a different curve (~20% apart over 1000-20000 A), so
        accepting it would be the same bug at smaller magnitude.
        """
        with pytest.raises(ValueError) as excinfo:
            _build(ssp, obs, law)
        msg = str(excinfo.value)
        assert "does not implement it" in msg, (
            f"the message must say the law is real but unwired, not merely unknown: {msg}"
        )
        assert "prevot_smc" in msg, "the message must name what IS accepted"

    def test_a_typo_is_refused_as_unknown(self, ssp, obs):
        """A misspelling and an unwired law are different mistakes.

        Telling someone their typo "is a real attenuation law" would be false.
        """
        with pytest.raises(ValueError, match="Unknown dust law"):
            _build(ssp, obs, "cardelli89")


class TestTheLawIsLoadBearing:
    """The assertion whose absence let #2012 ship.

    With one law wired the distinctness check is trivially satisfied. That is
    the point: it is written against the *set* of accepted laws, so it starts
    doing real work the moment a second curve is wired, instead of needing
    someone to remember to add a test then.
    """

    def test_every_accepted_law_gives_a_distinct_sed(self, ssp, obs):
        from tengri.parameters.groups import _VALID_AGN_ATTEN_LAWS

        laws = sorted(_VALID_AGN_ATTEN_LAWS)
        seds = [_sed(ssp, obs, law) for law in laws]
        distinct = {s.tobytes() for s in seds}
        assert len(distinct) == len(laws), (
            f"{len(laws)} accepted laws produced {len(distinct)} distinct SEDs — "
            f"at least two are mapped to the same curve, which is #2012 again"
        )

    def test_ebv_is_load_bearing(self, ssp, obs):
        """The control without which the test above passes vacuously.

        If AGN attenuation never reached the SED, every law would agree and the
        distinctness assertion would be satisfied for the wrong reason. This is
        the check that made #2012's measurement trustworthy.
        """
        off = _sed(ssp, obs, "prevot_smc", ebv=0.0)
        on = _sed(ssp, obs, "prevot_smc", ebv=0.4)
        assert not np.array_equal(off, on), (
            "E(B-V) does not change the SED, so this module cannot tell an "
            "inert law from a working one"
        )


def test_the_retired_type_spelling_does_not_advertise_laws_it_ignores(ssp, obs):
    """The migration message used to name four laws, three of them ignored.

    An error that teaches the wrong thing is how the misleading form spread.
    """
    with pytest.raises(ValueError) as excinfo, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FIXED},
            agn={
                "type": "composable",
                "all_params": FIXED,
                "atten": {"type": "smc_prevot"},
            },
            redshift=Fixed(1.0),
        )
    msg = str(excinfo.value)
    assert "prevot_smc" in msg
    for ignored in ("calzetti", "power_law", "cardelli"):
        assert ignored not in msg, (
            f"the migration message still advertises {ignored!r}, which this "
            f"block does not implement"
        )

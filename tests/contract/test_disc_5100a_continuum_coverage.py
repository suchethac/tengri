# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``_DISCS_WITH_5100A_CONTINUUM`` covers every disc with a real one.

Rule 4 of :func:`~tengri.components.agn.blocks.runner.validate_block_recipe`
warns when a GRAHSP-normalized downstream block (nlr/blr='analytic' or
'grahsp', feii='grahsp'/'boroson_green', torus='grahsp') is paired with a
disc NOT in ``_DISCS_WITH_5100A_CONTINUUM``, because those downstream blocks
scale by the disc's :math:`\\lambda L_\\lambda(5100\\,\\mathrm{\\AA})`.

``slone_netzer``, ``kd18_agnfitter``, and ``kd18_agnfitter_warmindex`` are all
grid-tabulated accretion-disc templates whose own crossval tests
(``test_slone_netzer_vs_agnfitter.py`` / ``test_kd18_grid_vs_agnfitter.py``)
confirm the standard UV/optical accretion-disc peak (< 1 um in L_nu). Omitting
them from ``_DISCS_WITH_5100A_CONTINUUM`` produced an unwarranted Rule 4
``RecipeWarning`` whenever composed with a GRAHSP-family NLR/BLR/FeII/torus
block, even though the crossval shows a real UV/optical continuum.
"""

from __future__ import annotations

import warnings

import pytest

from tengri.components.agn.blocks.runner import RecipeWarning, validate_block_recipe

pytestmark = pytest.mark.contract

#: Every non-torus/nlr slot off; only the one pairing under test is active.
_QUIET_TORUS_NLR = {
    "agn_blr_block": "none",
    "agn_feii_block": "none",
    "agn_attenuation_block": "none",
}

#: Substring unique to Rule 4's message (distinguishes it from other rules).
_RULE4 = "is not in the set of impls known to produce"


def _rule4_issues(disc_block: str, *, nlr_block: str = "analytic") -> list[str]:
    """Rule 4 issues for one (disc, nlr) pairing, with everything else quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        issues = validate_block_recipe(
            agn_disc_block=disc_block,
            agn_nlr_block=nlr_block,
            agn_torus_block="none",
            **_QUIET_TORUS_NLR,
        )
    return [i for i in issues if _RULE4 in i]


@pytest.mark.parametrize(
    "disc_block", ["slone_netzer", "kd18_agnfitter", "kd18_agnfitter_warmindex"]
)
def test_grid_tabulated_discs_do_not_warn(disc_block):
    """A disc with a confirmed UV/optical continuum must not trip Rule 4."""
    assert _rule4_issues(disc_block) == [], (
        f"disc_block={disc_block!r} unexpectedly triggered the Rule 4 5100A "
        "continuum warning; check _DISCS_WITH_5100A_CONTINUUM in runner.py"
    )


def test_adaf_still_warns():
    """Negative control: ADAF's inner flow is X-ray dominated, genuinely no
    5100A continuum (see the ``_DISCS_WITH_5100A_CONTINUUM`` module comment
    in runner.py) -- Rule 4 must still fire for it, proving the test above is
    not vacuous (silenced for every disc regardless of physics)."""
    issues = _rule4_issues("adaf")
    assert len(issues) == 1, issues
    assert "adaf" in issues[0]


def test_grid_tabulated_discs_warn_free_when_built_via_parameters():
    """End-to-end: the seam a user actually touches, not just the validator."""
    from tengri.parameters import FREE, parse_groups

    for disc_block in ("slone_netzer", "kd18_agnfitter", "kd18_agnfitter_warmindex"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parse_groups(
                sfh={"type": "dpl"},
                agn={
                    "type": "composable",
                    "disc": {"type": disc_block},
                    "nlr": {"type": "analytic"},
                    "norm": "independent",
                    "all_params": FREE,
                },
                redshift=0.1,
            )
        rule4 = [
            str(w.message)
            for w in caught
            if issubclass(w.category, RecipeWarning) and _RULE4 in str(w.message)
        ]
        assert rule4 == [], f"disc={disc_block!r}: unexpected Rule 4 warning(s): {rule4}"

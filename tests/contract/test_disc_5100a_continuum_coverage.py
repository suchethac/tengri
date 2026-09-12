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


def _rule4_issues(disc_block: str, *, nlr_block: str = "grahsp") -> list[str]:
    r"""Rule 4 issues for one (disc, nlr) pairing, with everything else quiet.

    The probe block is ``nlr='grahsp'``, not ``'analytic'``: R48's table lists
    only blocks that really do normalize off the disc's 5100 A luminosity, and
    ``nlr_analytic_block`` does not (it ``del``\ s ``l5100_disc`` and scales
    off ``agn_log_lbol`` instead). Measured with every other slot off,
    marginal ``sum|sed_agn|`` over 500 A - 1 mm:

    ==================  ===============  ==================
    block               disc='none'      disc='multicolor'
    ==================  ===============  ==================
    ``nlr='analytic'``  **2.022386e+31**  2.022386e+31
    ``nlr='grahsp'``    0.000000e+00     1.642265e+31
    ``blr='analytic'``  0.000000e+00     4.686986e+30
    ``blr='grahsp'``    0.000000e+00     7.246841e+31
    ``feii='grahsp'``   0.000000e+00     5.363636e+30
    ``torus='grahsp'``  0.000000e+00     2.493180e+34
    ==================  ===============  ==================

    ``analytic`` is the one row that emits identically with no disc, so Rule 4
    no longer names it -- which is exactly why this file cannot keep probing
    with it. A probe Rule 4 can never fire for would make every "does not
    warn" assertion here pass vacuously, the negative control included.
    """
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


def test_nlr_analytic_does_not_trip_rule4_on_any_disc():
    """``nlr='analytic'`` is not disc-anchored, so Rule 4 must never name it.

    ``nlr_analytic_block`` is illuminated by the intrinsic bolometric
    ``10**agn_log_lbol``, and its body opens with ``del l5100_disc``. Measured:
    it emits 2.022386e+31 with ``agn_disc_block='none'`` -- bit-identical to
    its value with a full ``multicolor`` disc -- so warning that it "scales by
    the disc's 5100 A luminosity (zero)" is a false advisory (R48's own rule:
    naming a block that does not anchor that way is the defect).

    ADAF is the disc used here because it is the one Rule 4 fires hardest for.
    """
    assert _rule4_issues("adaf", nlr_block="analytic") == [], (
        "Rule 4 named nlr='analytic', which does not read l5100_disc"
    )


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
                    # Disc-anchored on purpose: 'analytic' is not (see
                    # _rule4_issues), so Rule 4 could not fire for it and this
                    # assertion would hold for the wrong reason.
                    "nlr": {"type": "grahsp"},
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

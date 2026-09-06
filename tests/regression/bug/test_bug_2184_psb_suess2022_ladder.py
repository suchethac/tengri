# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #2184: the ``psb_suess2022`` ladder never ascended.

https://github.com/suchethac/tengri/issues/2184

The registry entry wired the post-starburst SFH to a shape function that
splices ``tflex_gyr`` in ahead of a fixed edge array starting at 0.3 Gyr, while
the entry's own prior on ``tflex_gyr`` is Uniform(0.5, 5.0). Every draw from
that prior produced a non-ascending ladder ``[0, tlast, tflex, 0.3, 1.0, 3.0,
6.0, 13.7]``: bins of negative width, ``jnp.searchsorted`` evaluated on an
unsorted array, an integrated mass 1.1 to 3.7 % above the declared total, and a
``log_mstar_formed`` that stopped responding to ``tflex_gyr`` above the first
crossing (9.97314 / 9.98304 / 9.98304 at ``tflex_gyr`` = 0.6 / 2.0 / 4.9 Gyr).
Five fixed old bins were declared against three ``ratio_old_*`` parameters, so
one step of the ladder was pinned at zero with nothing saying so.

The entry now lays three fixed old bins out equal-width from ``tflex_gyr`` to
13.7 Gyr, and declares the two adjacent-step ratios three bins take. Three
things keep the ladder ordered everywhere, and it takes all three: the layout
derives the fixed section FROM ``tflex_gyr``; ``tflex_gyr``'s prior floor was
raised to ``tlast_gyr``'s ceiling (1.0 Gyr) so the joint prior has no crossing
draw; and pinned values, which bypass a prior, are checked against each other
at build time.

Model: Suess et al. 2022, ApJ 935, 146 (arXiv:2207.02883), Section 3.1.4, a
three-part history (a youngest bin of variable length, a flexible zone, and
three fixed old bins) built on the continuity machinery of Johnson et al. 2021,
ApJS 254, 22 (arXiv:2012.01426). The equal-width old-bin layout is tengri's,
an approximation of the paper's template edges; this entry keeps the flexible
zone as a single bin, which is the ``nflex = 1`` special case of ``psb_flex``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.stellar.sfh.nonparametric import (
    PSB_FLEX_DEFAULT_MAX_AGE_GYR,
    PSB_FLEX_DEFAULT_N_FIXED,
)
from tengri.components.stellar.sfh.registry import SFH_REGISTRY, validate_bin_edges_gyr

pytestmark = pytest.mark.regression_bug


#: Uniform lookback grid [yr]: dt = 0.69 Myr, so each bin edge falls inside one
#: cell and a Riemann sum over a piecewise-constant history is good to 1.33e-4
#: of the total (measured, worst corner of the box below).
T_YR = jnp.linspace(0.0, 13.8e9, 20_001)
DT_YR = float(T_YR[1] - T_YR[0])

#: Non-zero ratios from the issue's own reproduction, so no step of the ladder
#: is silently flat while the test claims to see it.
RATIOS = {"ratio_young": 0.5, "ratio_old_0": 0.3, "ratio_old_1": -0.2}

#: Every corner of the joint (tlast_gyr, tflex_gyr) prior box: ``tlast_gyr`` at
#: each end of Uniform(0.01, 1.0) against ``tflex_gyr`` at the floor, the
#: default and the ceiling of Uniform(1.0, 5.0). ``(1.0, 1.0)`` is the one point
#: the two priors share and the tightest case in the box: the flexible zone has
#: zero width there.
PRIOR_BOX_GYR = [
    (0.01, 1.0),  # tlast floor, tflex floor
    (1.0, 1.0),  # tlast ceiling meets the tflex floor: zero-width flexible bin
    (0.01, 2.0),  # tflex default
    (1.0, 2.0),  # tlast ceiling
    (0.01, 5.0),  # tflex ceiling
    (1.0, 5.0),  # both ceilings
]


def _declared_old_ratios() -> list[str]:
    """The entry's ``ratio_old_*`` parameters, in declaration order."""
    return [name for name in SFH_REGISTRY["psb_suess2022"].params if "ratio_old_" in name]


def _shape_kwargs(**overrides) -> dict:
    """Shape-function kwargs built through the entry's own parameter map.

    Defaults come from each parameter's declared prior, so the dict follows the
    registry rather than restating it; ``overrides`` are keyed by the internal
    (prefix-stripped) name the shape function reads.
    """
    spec = SFH_REGISTRY["psb_suess2022"]
    kwargs = {}
    for public, (internal, scale, offset) in spec.internal_param_map.items():
        kwargs[internal] = spec.params[public].default.default * scale + offset
    kwargs.update(overrides)
    return kwargs


def _sfr(**overrides) -> np.ndarray:
    """The registered shape function's history on :data:`T_YR` [Msun/yr]."""
    return np.asarray(SFH_REGISTRY["psb_suess2022"].fn(T_YR, **_shape_kwargs(**overrides)))


def _step_ages_gyr(sfr: np.ndarray) -> np.ndarray:
    """Lookback ages [Gyr] at which a piecewise-constant history changes value.

    These are the ladder's edges as the model actually applies them, read off
    the output instead of re-derived from the layout under test.
    """
    steps = np.flatnonzero(np.abs(np.diff(sfr)) > 0.0)
    return np.asarray(T_YR)[steps + 1] / 1e9


def _expected_ladder_gyr(tlast_gyr: float, tflex_gyr: float) -> np.ndarray:
    """The ladder the entry promises: [0, tlast, tflex, three equal old bins]."""
    fixed = np.linspace(tflex_gyr, PSB_FLEX_DEFAULT_MAX_AGE_GYR, PSB_FLEX_DEFAULT_N_FIXED + 1)
    return np.concatenate([[0.0, tlast_gyr], fixed])


def test_the_entry_declares_one_ratio_for_every_step_of_its_fixed_section():
    """``n_fixed`` old bins take ``n_fixed - 1`` adjacent-step ratios.

    The step between the oldest flexible bin and the youngest fixed bin is
    pinned at zero (the two share an SFR), so a surplus ratio reaches no bin
    and a missing one leaves a step stuck at zero. #2184 shipped five fixed
    bins against three ratios, which is the missing-one case.
    """
    assert len(_declared_old_ratios()) == PSB_FLEX_DEFAULT_N_FIXED - 1, (
        f"{_declared_old_ratios()} does not match the {PSB_FLEX_DEFAULT_N_FIXED} fixed "
        "old bins the entry lays down (#2184)"
    )


@pytest.mark.parametrize(("tlast_gyr", "tflex_gyr"), PRIOR_BOX_GYR)
def test_the_ladder_the_history_reveals_is_the_one_the_entry_promises(tlast_gyr, tflex_gyr):
    """Every bin lands where the layout says, everywhere in the joint prior box.

    The edges are read back off the returned history rather than recomputed: a
    piecewise-constant SFH steps exactly at its bin edges, so those step ages
    are the ladder as the model applied it. They must be strictly increasing
    (that is the ascent, observed rather than assumed) and they must sit at
    ``tlast_gyr``, the two interior fixed edges and the oldest edge where star
    formation stops.

    The ``tflex_gyr`` edge is deliberately absent from that list, because the
    flex-to-fixed step is pinned at zero: the two bins share an SFR and there is
    nothing to see at their boundary. At ``(1.0, 1.0)`` the flexible bin has
    zero width, so the promised ladder is non-decreasing rather than strictly
    ascending, while the *visible* steps stay strictly increasing: the youngest
    bin simply hands straight over to the fixed section.
    """
    ladder = _expected_ladder_gyr(tlast_gyr, tflex_gyr)
    assert np.all(np.diff(ladder) >= 0.0), f"ladder {ladder} is not ordered (#2184)"

    sfr = _sfr(tlast_gyr=tlast_gyr, tflex_gyr=tflex_gyr, **RATIOS)
    assert np.all(sfr >= 0.0)
    assert np.all(np.isfinite(sfr))

    got = _step_ages_gyr(sfr)
    assert np.all(np.diff(got) > 0.0), f"the history steps out of order: {got} (#2184)"

    want = np.array([tlast_gyr, ladder[3], ladder[4], PSB_FLEX_DEFAULT_MAX_AGE_GYR])
    # One grid cell (0.69 Myr = 6.9e-4 Gyr) of read-off slack per edge.
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1.5e-3)


@pytest.mark.parametrize(("tlast_gyr", "tflex_gyr"), PRIOR_BOX_GYR)
def test_the_history_closes_on_the_declared_total_mass(tlast_gyr, tflex_gyr):
    """``sum(SFR_i dt_i)`` is ``10**log_total_mass`` for any draw from the prior.

    A crossed ladder puts negative widths in the normalization sum, so the
    history it hands back integrates to something else: 1.1 to 3.7 % high across
    this box before the fix. The 2e-3 here is a margin, not the achieved
    accuracy: the Riemann sum on :data:`T_YR` closes to 1.33e-4 at its worst
    corner, 15x inside the tolerance, which is in turn 5x inside the smallest
    defect it has to catch.
    """
    log_total_mass = 10.25
    sfr = _sfr(log_total_mass=log_total_mass, tlast_gyr=tlast_gyr, tflex_gyr=tflex_gyr, **RATIOS)
    mass = float(np.sum(sfr) * DT_YR)
    assert mass == pytest.approx(10.0**log_total_mass, rel=2e-3, abs=0.0)


def test_every_declared_old_ratio_changes_the_history():
    """No ``ratio_old_*`` may sample a prior that reaches no bin.

    A parameter the shape function never reads is swallowed by
    ``**ratio_kwargs``: it costs a sampling dimension and changes nothing.
    """
    base = _sfr(**RATIOS)
    for name in _declared_old_ratios():
        internal = SFH_REGISTRY["psb_suess2022"].internal_param_map[name][0]
        moved = _sfr(**{**RATIOS, internal: RATIOS.get(internal, 0.0) + 0.5})
        live = base > 0.0
        assert np.max(np.abs(moved[live] / base[live] - 1.0)) > 1e-3, (
            f"{name} left the history unchanged: it reaches no bin (#2184)"
        )


def test_tflex_is_live_across_its_whole_prior(synthetic_ssp_wide):
    """Moving ``tflex_gyr`` must move the history and the SED it produces.

    The public signature of #2184: once ``tflex_gyr`` passed the first fixed
    edge at 0.3 Gyr, the crossed ladder returned the *same history* for every
    larger value. Not approximately: measured on this build at 1.1, 2.0 and 4.9
    Gyr, the SFH arrays agreed to 2.2e-16 and the SEDs to 1.2e-8, so the
    parameter was inert over more than 90 % of its prior while still costing a
    sampling dimension. Corrected, the same three builds differ by 85 % or more
    in the SFH and 1.3e-3 to 5.9e-3 in the SED.

    ``log_mstar_formed`` is deliberately not the probe here. The formed mass is
    normalized to ``log_total_mass`` by construction, so with the exact bin-edge
    knots of #765 it closes to 1e-6 dex for every ``tflex_gyr``; the spread the
    issue reported was quadrature error, which is a symptom of the defect but
    not a quantity that should respond.
    """

    def history_and_sed(tflex_gyr):
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh={
                "type": "psb_suess2022",
                "tlast_gyr": Fixed(0.2),
                "tflex_gyr": Fixed(tflex_gyr),
                "ratio_young": Fixed(RATIOS["ratio_young"]),
                "ratio_old_0": Fixed(RATIOS["ratio_old_0"]),
                "ratio_old_1": Fixed(RATIOS["ratio_old_1"]),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.05),
        )
        state = model.predict_state({})
        return np.asarray(state.derived["sfr_history"]), np.asarray(state.sed_intrinsic)

    runs = {tflex: history_and_sed(tflex) for tflex in (1.1, 2.0, 4.9)}
    for a, b in ((1.1, 2.0), (1.1, 4.9), (2.0, 4.9)):
        sfh_a, sed_a = runs[a]
        sfh_b, sed_b = runs[b]
        live = (sfh_a > 0.0) & (sfh_b > 0.0)
        assert np.max(np.abs(sfh_b[live] / sfh_a[live] - 1.0)) > 0.1, (
            f"the SFH is unchanged between tflex_gyr={a} and {b}: the parameter is "
            "inert over its prior (#2184)"
        )
        lit = (sed_a > 0.0) & (sed_b > 0.0)
        assert np.max(np.abs(sed_b[lit] / sed_a[lit] - 1.0)) > 1e-4, (
            f"the SED is unchanged between tflex_gyr={a} and {b} (#2184)"
        )


def test_bin_edge_validation_reaches_this_entry():
    """A custom ladder whose bin count the ratios cannot fill must be refused.

    Before the fix the entry's shape function matched neither branch of
    :func:`validate_bin_edges_gyr`, so a mismatched ``bin_edges_gyr`` passed
    validation and its surplus ratios were swallowed silently (#1975's failure,
    one entry later).
    """
    n_edges = PSB_FLEX_DEFAULT_N_FIXED + 1
    validate_bin_edges_gyr("psb_suess2022", np.linspace(2.0, 13.7, n_edges))
    with pytest.raises(ValueError, match="ratio_old parameters"):
        validate_bin_edges_gyr("psb_suess2022", np.linspace(2.0, 13.7, n_edges + 2))


@pytest.mark.parametrize("sfh_type", ["psb_suess2022", "psb_flex"])
def test_the_quench_epoch_priors_cannot_cross(sfh_type):
    """``tflex_gyr``'s prior floor is ``tlast_gyr``'s ceiling, on both entries.

    The flexible zone is ``[tlast_gyr, tflex_gyr]``. While the two priors
    overlapped (tlast on [0.01, 1.0], tflex from 0.5), their joint support
    contained draws with a negative-width zone: measured at
    ``tlast_gyr = 1.0, tflex_gyr = 0.5``, the ladder came out
    ``[0, 1.0, 0.5, 4.9, 9.3, 13.7]``, the flexible bin was annihilated, the
    youngest bin overran it, and the total mass still closed to 1.3e-4, so no
    mass or positivity check could see it. Removing the overlap is the only
    thing that constrains a *sampled* pair. Suess+2022 lose nothing by it: they
    fix tflex at 2 Gyr, which is still the default here.
    """
    params = SFH_REGISTRY[sfh_type].params
    prefix = "sfh_psb2022_" if sfh_type == "psb_suess2022" else "sfh_psb_flex_"
    tlast = params[f"{prefix}tlast_gyr"].default
    tflex = params[f"{prefix}tflex_gyr"].default
    assert tflex.bounds[0] >= tlast.bounds[1], (
        f"{sfh_type}: tflex_gyr prior {tflex.bounds} dips below the tlast_gyr "
        f"ceiling {tlast.bounds[1]}, so the joint prior can draw a "
        "negative-width flexible zone (#2184)"
    )
    assert tflex.default == 2.0  # the paper's own value, unchanged by the floor


def _build_psb(ssp, **sfh_extra):
    """A ``psb_suess2022`` build with the given quench-epoch dispositions."""
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={"type": "psb_suess2022", **sfh_extra},
        redshift=Fixed(0.05),
    )


def test_a_pinned_crossing_is_refused_at_build(synthetic_ssp_wide):
    """Two pinned values that cross must fail at build, not silently at predict.

    A ``Fixed`` bypasses the prior that the floor above constrains, so this is
    the other half of the same rule. The failure it prevents is silent in every
    other check: finite, non-negative, and mass-closed.
    """
    with pytest.raises(ValueError, match="negative width"):
        _build_psb(
            synthetic_ssp_wide,
            tlast_gyr=Fixed(1.0),
            tflex_gyr=Fixed(0.5),
            all_params=Fixed(DEFAULT),
        )


def test_a_pinned_value_a_free_prior_can_cross_is_refused_at_build(synthetic_ssp_wide):
    """Pinning one epoch inside the other's prior support is the same crossing.

    ``tflex_gyr`` pinned at 0.7 Gyr with ``tlast_gyr`` free on [0.01, 1.0] draws
    a crossed ladder for every ``tlast_gyr`` above 0.7, which is 30 % of that
    prior. Refused for the pair, not for either value on its own.
    """
    with pytest.raises(ValueError, match="negative width"):
        _build_psb(synthetic_ssp_wide, tflex_gyr=Fixed(0.7), all_params=FREE)


def test_a_pinned_value_the_free_prior_cannot_reach_is_accepted(synthetic_ssp_wide):
    """The check refuses crossings, not mixed dispositions.

    ``tflex_gyr`` pinned at 1.5 Gyr sits above everything ``tlast_gyr``'s prior
    can draw, so this build is well posed and must go through; so must the
    default, where both epochs are pinned at their registry values.
    """
    _build_psb(synthetic_ssp_wide, tflex_gyr=Fixed(1.5), all_params=FREE)
    _build_psb(synthetic_ssp_wide, all_params=FREE)
    _build_psb(synthetic_ssp_wide, all_params=Fixed(DEFAULT))

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
13.7 Gyr, ascending for every ``tflex_gyr`` the prior can draw, and declares
the two adjacent-step ratios three bins take.

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

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.sfh.nonparametric import (
    PSB_FLEX_DEFAULT_MAX_AGE_GYR,
    PSB_FLEX_DEFAULT_N_FIXED,
)
from tengri.components.stellar.sfh.registry import SFH_REGISTRY, validate_bin_edges_gyr

pytestmark = pytest.mark.regression_bug


#: Uniform lookback grid [yr], dense enough that each bin edge falls inside one
#: cell (dt = 0.69 Myr) and a Riemann sum over a piecewise-constant history is
#: exact to better than 1e-4 of the total.
T_YR = jnp.linspace(0.0, 13.8e9, 20_001)
DT_YR = float(T_YR[1] - T_YR[0])

#: Non-zero ratios from the issue's own reproduction, so no step of the ladder
#: is silently flat while the test claims to see it.
RATIOS = {"ratio_young": 0.5, "ratio_old_0": 0.3, "ratio_old_1": -0.2}

#: Corners of the (tlast_gyr, tflex_gyr) prior box, plus the case the defect
#: turned on: ``tlast_gyr`` pressed right up against ``tflex_gyr``.
PRIOR_BOX_GYR = [
    (0.01, 0.5),  # both floors
    (0.49, 0.5),  # tlast just below tflex
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
def test_the_ladder_ascends_everywhere_in_the_prior_box(tlast_gyr, tflex_gyr):
    """Every bin the entry lays down has positive width, and lands where promised.

    The edges are read off the returned history: a piecewise-constant SFH steps
    exactly at its bin edges. The ``tflex_gyr`` edge is deliberately absent from
    that list, because the flex-to-fixed step is pinned at zero: the two share
    an SFR and there is nothing to see at their boundary. The remaining steps
    (``tlast_gyr``, the two interior fixed edges, and the oldest edge where star
    formation stops) pin the whole layout.
    """
    ladder = _expected_ladder_gyr(tlast_gyr, tflex_gyr)
    assert np.all(np.diff(ladder) > 0.0), f"ladder {ladder} is not ascending (#2184)"

    sfr = _sfr(tlast_gyr=tlast_gyr, tflex_gyr=tflex_gyr, **RATIOS)
    assert np.all(sfr >= 0.0)

    want = np.array([tlast_gyr, ladder[3], ladder[4], PSB_FLEX_DEFAULT_MAX_AGE_GYR])
    got = _step_ages_gyr(sfr)
    # One grid cell (0.69 Myr = 6.9e-4 Gyr) of read-off slack per edge.
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1.5e-3)


@pytest.mark.parametrize(("tlast_gyr", "tflex_gyr"), PRIOR_BOX_GYR)
def test_the_history_closes_on_the_declared_total_mass(tlast_gyr, tflex_gyr):
    """``sum(SFR_i dt_i)`` is ``10**log_total_mass`` for any draw from the prior.

    A crossed ladder puts negative widths in the normalization sum, so the
    history it hands back integrates to something else: 1.1 to 3.7 % high
    across this box before the fix. The tolerance here is the Riemann sum's own
    discretization on :data:`T_YR`, the same 2e-3 the ``psb_flex`` sibling test
    uses, not a margin for a modeling error.
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


def test_formed_mass_still_responds_to_tflex_above_the_first_crossing(synthetic_ssp_wide):
    """``log_mstar_formed`` must move with ``tflex_gyr`` across the whole prior.

    The public signature of #2184: once ``tflex_gyr`` passed the first fixed
    edge at 0.3 Gyr, the crossed ladder returned the same history for every
    larger value, so the model reported 9.98304 at both 2.0 and 4.9 Gyr and the
    parameter was inert over 90 % of its prior.
    """

    def formed(tflex_gyr):
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
        return float(model.predict_state({}).derived["log_mstar_formed"])

    values = [formed(tflex_gyr) for tflex_gyr in (0.6, 2.0, 4.9)]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert abs(values[i] - values[j]) > 1e-4, (
            f"log_mstar_formed is flat in tflex_gyr: {values} (#2184)"
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

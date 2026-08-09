# SPDX-License-Identifier: BSD-3-Clause
r"""Regression: a declared prior may be wider than the grid consuming it (#1586).

A parameter carries two supports and only one is written down. The declaration
records a prior; a template-backed block also has an *implicit* support — the
axes of the grid it interpolates. ``create_slone_netzer_from_grid`` clips both
of its parameters onto those axes, so any value outside them collapses onto the
edge node.

The failure is invisible from the forward pass. There is no NaN, no warning and
no error: the SED is **bit-identical** across the excess, and because
``jnp.clip`` is flat outside its bounds the gradient there is **exactly** zero,
not merely small. A gradient-based fit initialized in that region gets no
signal and cannot move the parameter at all.

Measured against the shipped ``data/slone_netzer_disc_grid.h5``:

============= ======================= ====================== ======
parameter     declared prior          SN12 grid axis         live
============= ======================= ====================== ======
agn_log_mbh   ``Uniform(6.0, 10.0)``  ``[7.4, 9.8]``         60%
agn_log_ledd  ``Uniform(-2.0, 0.5)``  ``[-4.0, -1.9586]``    1.7%
============= ======================= ====================== ======

Both declared *defaults* — ``(7.0, -1.0)`` — are outside the grid, so both are
gradient-dead.

Why the fix is a composition-time check and not a narrower declaration:
``agn_log_mbh`` / ``agn_log_ledd`` are shared with the analytic disc models
(``kd18_disc_model``, ``adaf``, ``unified``, ``disc``), which have no grid and
legitimately want the wide physical support. Narrowing the shared declaration
would wrongly constrain them, and a ``ParamDeclaration.bound_check`` is global
to the declaration, so it cannot say "only when this block is selected". The
constraint belongs to the ``(block, parameter)`` pair, so ``Rule 9`` of
:func:`~tengri.components.agn.blocks.runner.validate_block_recipe` checks it.

Two existing guards miss this by construction, and neither is broken — the
question they answer is simply a different one:

* ``tools/check_param_defaults.py`` (#1564) compares a *default* against its
  *declared prior*. Here the default sits comfortably inside ``[6, 10]``; it is
  the prior itself that overruns the grid.
* ``tests/contract/test_agn_block_consumes.py`` perturbs every parameter across
  its prior to find inert ones, but it exempts ``slone_netzer`` and detects only
  *fully* inert parameters, not 98%-dead ones. (That exemption is justified in
  ``_consumes.py`` as "requires a data grid absent from CI" — but
  ``data/slone_netzer_disc_grid.h5`` is git-tracked, as are the ``cat3d_wind``
  and ``grahsp`` grids, so the stated reason no longer holds.)

The grid *is* shipped, so every test below runs in CI. The ``_sn12_grid`` skip
is a safety net for a stripped install, not the expected path — if these start
skipping, the data file has gone missing rather than the check being untested.
Rule 9's own logic is additionally driven with a synthetic support table so it
stays testable independently of any data file.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from tengri.components.agn._params import PARAMS
from tengri.components.agn.blocks import _grid_support
from tengri.components.agn.blocks._grid_support import (
    block_grid_support,
    is_contained,
    live_fraction,
)
from tengri.components.agn.blocks.runner import validate_block_recipe

pytestmark = pytest.mark.regression_bug

#: Every non-disc selector off, so a recipe exercises the disc block alone.
_QUIET = {
    "agn_nlr_block": "none",
    "agn_blr_block": "none",
    "agn_feii_block": "none",
    "agn_torus_block": "none",
    "agn_attenuation_block": "none",
}

#: Substring unique to a Rule 9 message, used to separate it from Rules 2-7.
_RULE9 = "bit-identical to the edge node"


def _declared_prior(name: str) -> tuple[float, float]:
    """Read a declared prior's bounds from the registry, never hand-copied."""
    for decl in PARAMS:
        if decl.name == name:
            return decl.prior.bounds
    raise AssertionError(f"{name} is no longer declared in agn PARAMS")


def _rule9_issues(**kwargs) -> list[str]:
    """Return only the Rule 9 issues from one recipe validation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        issues = validate_block_recipe(**{**_QUIET, **kwargs})
    return [i for i in issues if _RULE9 in i]


def _sn12_grid() -> dict[str, tuple[float, float]]:
    """The shipped grid's support, or skip if the data file is absent."""
    support = block_grid_support("disc", "slone_netzer")
    if not support:
        pytest.skip("slone_netzer grid not installed")
    return support


# --------------------------------------------------------------------------
# The containment arithmetic (no data file needed).
# --------------------------------------------------------------------------


def test_live_fraction_measures_the_overlap():
    """A half-overhanging prior is half live; a contained one is fully live."""
    assert live_fraction((0.0, 10.0), (0.0, 10.0)) == pytest.approx(1.0)
    assert live_fraction((0.0, 10.0), (5.0, 10.0)) == pytest.approx(0.5)
    assert live_fraction((2.0, 4.0), (0.0, 10.0)) == pytest.approx(1.0)
    # Entirely outside: every reachable value clips to one edge node.
    assert live_fraction((20.0, 30.0), (0.0, 10.0)) == 0.0
    # A fixed value is live only if it sits on the grid.
    assert live_fraction((5.0, 5.0), (0.0, 10.0)) == 1.0
    assert live_fraction((50.0, 50.0), (0.0, 10.0)) == 0.0
    # No finite grid can contain an unbounded prior.
    assert live_fraction((-jnp.inf, jnp.inf), (0.0, 10.0)) == 0.0


def test_containment_tolerates_a_transcribed_bound():
    """A prior written to match the axis must not be reported as overrunning.

    The SN12 log_edd axis ends at -1.958607314841775. A prior transcribed as
    -1.95860731 overhangs by ~5e-9 — unreachable by any fit. Comparing exactly
    flagged it and then printed a self-contradictory "0% of its range lies
    outside", which is what this guards against.
    """
    grid = (-4.0, -1.958607314841775)
    assert is_contained((-4.0, -1.95860731), grid)
    assert is_contained(grid, grid)
    # A real overrun is still caught: 0.5 dex past the top of a 2 dex axis.
    assert not is_contained((-4.0, -1.5), grid)


# --------------------------------------------------------------------------
# Rule 9, driven with a synthetic block so it gates without the data file.
# --------------------------------------------------------------------------


@pytest.fixture
def synthetic_grid_block(monkeypatch):
    """Give the analytic 'multicolor' disc a pretend grid support."""
    monkeypatch.setitem(
        _grid_support.AGN_BLOCK_GRID_SUPPORT,
        ("disc", "multicolor"),
        lambda: {"agn_log_mbh": (7.4, 9.8)},
    )
    return ("disc", "multicolor")


def test_rule9_fires_when_a_prior_overruns_the_grid(synthetic_grid_block):
    """The whole point: an overrunning prior must stop being silent."""
    issues = _rule9_issues(
        agn_disc_block="multicolor",
        param_support={"agn_log_mbh": (6.0, 10.0)},
    )
    assert len(issues) == 1, issues
    message = issues[0]
    assert "agn_log_mbh" in message
    # The message must be actionable: it names the extent to narrow to...
    assert "[7.4, 9.8]" in message
    # ...and how much of the declared range is unreachable (40% here).
    assert "40%" in message


def test_rule9_warns_rather_than_only_returning(synthetic_grid_block):
    """Issues are also emitted as warnings, like every other recipe rule."""
    from tengri.components.agn.blocks.runner import RecipeWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_block_recipe(
            agn_disc_block="multicolor",
            **_QUIET,
            param_support={"agn_log_mbh": (6.0, 10.0)},
        )
    assert any(
        issubclass(w.category, RecipeWarning) and _RULE9 in str(w.message) for w in caught
    ), [str(w.message) for w in caught]


def test_rule9_is_silent_when_the_prior_fits(synthetic_grid_block):
    """No warning when nothing can be clipped — or the rule is just noise."""
    assert (
        _rule9_issues(
            agn_disc_block="multicolor",
            param_support={"agn_log_mbh": (7.4, 9.8)},
        )
        == []
    )
    assert (
        _rule9_issues(
            agn_disc_block="multicolor",
            param_support={"agn_log_mbh": (8.0, 9.0)},
        )
        == []
    )


def test_rule9_reports_a_fully_inert_parameter_distinctly(synthetic_grid_block):
    """A prior wholly off-grid is inert, not merely truncated."""
    issues = _rule9_issues(
        agn_disc_block="multicolor",
        param_support={"agn_log_mbh": (10.5, 11.0)},
    )
    assert len(issues) == 1, issues
    assert "entirely inert" in issues[0]


def test_rule9_does_not_call_an_unbounded_prior_inert(synthetic_grid_block):
    """An untruncated Gaussian reports (-inf, inf) — its tails clip, but most
    of its mass may sit on the grid, so "entirely inert" would be false."""
    issues = _rule9_issues(
        agn_disc_block="multicolor",
        param_support={"agn_log_mbh": (-float("inf"), float("inf"))},
    )
    assert len(issues) == 1, issues
    assert "unbounded" in issues[0]
    assert "entirely inert" not in issues[0]
    # A percentage of an infinite support is not informative; do not quote one.
    assert "%" not in issues[0]


def test_rule9_catches_an_off_grid_fixed_value(synthetic_grid_block):
    """A pinned value outside the grid is clipped just as silently."""
    issues = _rule9_issues(
        agn_disc_block="multicolor",
        param_support={"agn_log_mbh": (7.0, 7.0)},
    )
    assert len(issues) == 1, issues
    assert "fixed value 7" in issues[0]


def test_a_block_with_no_grid_is_never_constrained():
    """The constraint from #1586: analytic discs keep the wide support.

    ``agn_log_mbh`` / ``agn_log_ledd`` are shared with grid-free models, so a
    wide prior on them is correct there and must not warn. This is why the
    check is scoped per block rather than narrowed on the declaration.
    """
    assert block_grid_support("disc", "multicolor") == {}
    assert (
        _rule9_issues(
            agn_disc_block="multicolor",
            param_support={
                "agn_log_mbh": _declared_prior("agn_log_mbh"),
                "agn_log_ledd": _declared_prior("agn_log_ledd"),
            },
        )
        == []
    )


def test_rule9_is_skipped_without_param_support(synthetic_grid_block):
    """Callers that pass no support get the old behavior, not a crash."""
    assert _rule9_issues(agn_disc_block="multicolor") == []
    assert _rule9_issues(agn_disc_block="multicolor", param_support={}) == []


# --------------------------------------------------------------------------
# The shipped grid (skipped when the data file is absent, e.g. in CI).
# --------------------------------------------------------------------------


def test_the_grid_support_is_read_from_the_file():
    """Bounds come from the axes, so they cannot go stale if it is rebuilt."""
    support = _sn12_grid()
    assert set(support) == {"agn_log_mbh", "agn_log_ledd"}
    for name, (lo, hi) in support.items():
        assert lo < hi, f"{name} axis is not ascending: {(lo, hi)}"


def test_the_declared_priors_overrun_the_shipped_grid():
    """The defect itself, stated against the live registry and live grid."""
    support = _sn12_grid()
    for name in ("agn_log_mbh", "agn_log_ledd"):
        declared = _declared_prior(name)
        assert not is_contained(declared, support[name]), (
            f"{name} is now contained in the grid — if the declaration was "
            "narrowed on purpose, this regression test should be updated"
        )
    # agn_log_ledd is the severe one: almost none of its prior is reachable.
    assert live_fraction(_declared_prior("agn_log_ledd"), support["agn_log_ledd"]) < 0.05


def test_composing_slone_netzer_with_its_declared_priors_warns():
    """End-to-end: the user-facing guarantee, with nothing hand-copied."""
    _sn12_grid()
    issues = _rule9_issues(
        agn_disc_block="slone_netzer",
        param_support={
            "agn_log_mbh": _declared_prior("agn_log_mbh"),
            "agn_log_ledd": _declared_prior("agn_log_ledd"),
        },
    )
    assert len(issues) == 2, issues
    assert any("agn_log_mbh" in i for i in issues)
    assert any("agn_log_ledd" in i for i in issues)


def test_the_block_pins_stay_on_grid():
    """#1578/#1586 pinned the block's own defaults on-grid; keep them there.

    Reverting either to the shared declared default (7.0 / -1.0) would put it
    back outside the axes, where it is clipped and gradient-dead. This fails
    the moment someone "tidies" the pins to match the declaration.
    """
    import inspect

    from tengri.components.agn.blocks.disc import slone_netzer_disc_block

    support = _sn12_grid()
    sig = inspect.signature(slone_netzer_disc_block)
    for name, (lo, hi) in support.items():
        pin = sig.parameters[name].default
        assert lo <= pin <= hi, (
            f"{name} block default {pin} is outside the SN12 grid [{lo}, {hi}] "
            "— it would be silently clipped and gradient-dead"
        )


def _rule9_warnings_from_parameters(**kwargs) -> list[str]:
    """Build a composable AGN Parameters and return its Rule 9 warnings."""
    from tengri.components.agn.blocks.runner import RecipeWarning
    from tengri.parameters.parameters import Parameters

    base = {"agn_model": "composable", **_QUIET}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Parameters(**base, **kwargs)
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, RecipeWarning) and _RULE9 in str(w.message)
    ]


def test_building_parameters_with_slone_netzer_warns():
    """The seam a user actually touches, not just the validator underneath.

    ``Parameters`` is what has to pass the active support down; a rule that
    only fires when called directly would be invisible in practice.
    """
    _sn12_grid()
    found = _rule9_warnings_from_parameters(agn_disc_block="slone_netzer")
    assert len(found) == 2, found
    assert any("agn_log_mbh" in m for m in found)
    assert any("agn_log_ledd" in m for m in found)


def test_building_parameters_with_on_grid_priors_is_silent():
    """Narrowing to the grid, as the warning advises, must silence it."""
    from tengri.parameters.priors import Fixed, Uniform

    support = _sn12_grid()
    assert (
        _rule9_warnings_from_parameters(
            agn_disc_block="slone_netzer",
            agn_log_mbh=Uniform(*support["agn_log_mbh"]),
            agn_log_ledd=Uniform(*support["agn_log_ledd"]),
        )
        == []
    )
    assert (
        _rule9_warnings_from_parameters(
            agn_disc_block="slone_netzer",
            agn_log_mbh=Fixed(8.6),
            agn_log_ledd=Fixed(-2.0),
        )
        == []
    )


def test_building_parameters_with_an_analytic_disc_is_silent():
    """No data file needed: an analytic disc has no grid to overrun."""
    assert _rule9_warnings_from_parameters(agn_disc_block="multicolor") == []


def test_the_forward_pass_is_bit_identical_across_the_dead_zone():
    """Why the warning is warranted: the excess does literally nothing.

    Bit-identical is a stronger probe than a tolerance comparison, and it
    *derives* the zero gradient rather than needing to measure it separately.
    """
    support = _sn12_grid()
    from tengri.components.agn.slone_netzer import slone_netzer_sed

    wave = jnp.logspace(2.5, 5.0, 400)
    edd_hi = support["agn_log_ledd"][1]
    reference = slone_netzer_sed(wave, agn_log_mbh=8.6, agn_log_ledd=edd_hi)
    for off_grid in (-1.5, -1.0, 0.0, 0.5):
        probe = slone_netzer_sed(wave, agn_log_mbh=8.6, agn_log_ledd=off_grid)
        assert jnp.array_equal(probe, reference), (
            f"agn_log_ledd={off_grid} is off-grid but no longer clips to the "
            "edge node — the premise of #1586 has changed"
        )


def test_the_gradient_is_exactly_zero_at_the_declared_defaults():
    """jnp.clip is flat outside its bounds: not small, exactly 0.0."""
    _sn12_grid()
    from tengri.components.agn.slone_netzer import slone_netzer_sed

    wave = jnp.logspace(2.5, 5.0, 400)

    def total(log_mbh, log_ledd):
        return jnp.sum(slone_netzer_sed(wave, agn_log_mbh=log_mbh, agn_log_ledd=log_ledd))

    grad = jax.grad(total, argnums=(0, 1))

    d_mbh, d_ledd = grad(7.0, -1.0)  # the declared defaults
    assert float(d_mbh) == 0.0, f"expected an exactly dead gradient, got {d_mbh}"
    assert float(d_ledd) == 0.0, f"expected an exactly dead gradient, got {d_ledd}"

    # Anti-vacuity: on-grid, the same gradient is emphatically alive, so the
    # zeros above are the clip and not a broken derivative.
    live_mbh, live_ledd = grad(8.6, -2.5)
    assert float(live_mbh) != 0.0
    assert float(live_ledd) != 0.0

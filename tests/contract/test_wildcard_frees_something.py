# SPDX-License-Identifier: BSD-3-Clause
"""``all_params: FREE`` must free something, or say so.

``FREE`` resolves to each parameter's *registry default*, and most registry
defaults are ``Fixed`` scalars. A wildcard could therefore resolve cleanly while
leaving every parameter in the group pinned — the fit then ran to completion
with that physics frozen, indistinguishable from success at the call site.

Measured before the guard landed: ``all_params: FREE`` freed 6 params in ``sfh``
and 2 in top-level ``agn``, but **zero** in ``neb``, ``xray``, ``radio``,
``shock``, ``igm`` and in the AGN sub-blocks. The bug survived because every
worked example in the docs uses ``sfh={'all_params': FREE}`` — the one group
whose registry defaults are already free.

These tests pin the contract: a wildcard that frees nothing raises rather than
passing silently.
"""

from __future__ import annotations

import pytest

import tengri
from tengri import FIXED, FREE, Uniform
from tengri.config.exceptions import ParameterError

pytestmark = pytest.mark.contract


def _free(**groups) -> set[str]:
    return set(tengri.parse_groups(**groups).free_params)


# ── Groups whose wildcard frees nothing must raise ────────────────────


@pytest.mark.parametrize(
    "group,spec",
    [
        ("neb", {"type": "cue"}),
        ("xray", {"type": "simple"}),
    ],
)
def test_wildcard_free_that_frees_nothing_raises(group, spec):
    """A FREE that cannot free anything is refused, not silently ignored."""
    with pytest.raises(ParameterError, match=r"freed 0 of \d+ parameters"):
        tengri.parse_groups(sfh={"type": "dpl"}, **{group: {**spec, "all_params": FREE}})


def test_nested_subblock_wildcard_is_guarded():
    """Sub-blocks are checked at their own nesting level.

    ``dust.emission`` is the widest no-op found (22 Dale+2014 params, none
    freeable), and it sits under a ``dust`` wildcard that *does* free two —
    so the check cannot be per-top-level-group.
    """
    with pytest.raises(ParameterError, match=r"'dust\.emission'"):
        tengri.parse_groups(
            sfh={"type": "dpl"},
            dust={
                "type": "two_component",
                "emission": {"type": "dale2014", "all_params": FREE},
            },
        )


def test_dla_wildcard_raises_once_the_only_freeable_param_is_overridden():
    """Explicit overrides shrink what the wildcard covers — possibly to nothing.

    ``dla_log_n_hi`` is the one DLA param with a free (Uniform) registry
    default, so a bare wildcard legitimately frees it. Give ``log_n_hi`` an
    explicit prior and the wildcard is left covering only Fixed-default params,
    which is the shape that silently freed nothing.
    """
    with pytest.raises(ParameterError, match=r"'igm\.dla'"):
        tengri.parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            igm={
                "type": "inoue14",
                "dla": {"log_n_hi": Uniform(19, 22), "all_params": FREE},
            },
            redshift=tengri.Fixed(2.0),
        )


def test_dla_bare_wildcard_is_allowed_because_it_frees_one():
    """The guard is about outcome, not intent: freeing 1 of 4 is still freeing."""
    freed = _free(
        sfh={"type": "dpl", "*": FIXED},
        igm={"type": "inoue14", "dla": {"all_params": FREE}},
        redshift=tengri.Fixed(2.0),
    )
    assert "dla_log_n_hi" in freed


def test_error_names_the_group_and_the_stuck_params():
    """The message must be actionable: which group, which params, what to do."""
    with pytest.raises(ParameterError) as exc:
        tengri.parse_groups(sfh={"type": "dpl"}, neb={"type": "cue", "all_params": FREE})
    msg = str(exc.value)
    assert "'neb'" in msg
    assert "neb_logU" in msg
    assert "Uniform(lo, hi)" in msg


def test_star_synonym_is_guarded_identically():
    """``'*'`` is a synonym for ``all_params`` and must not bypass the guard."""
    with pytest.raises(ParameterError, match=r"freed 0 of \d+ parameters"):
        tengri.parse_groups(sfh={"type": "dpl"}, neb={"type": "cue", "*": FREE})


def test_builders_path_is_guarded_identically():
    """``builders.*(defaults=FREE)`` lowers to the same wildcard key."""
    with pytest.raises(ParameterError, match=r"freed 0 of \d+ parameters"):
        tengri.parse_groups(sfh={"type": "dpl"}, neb=tengri.builders.neb.cue(defaults=FREE))


# ── Everything that legitimately works must keep working ──────────────


def test_sfh_wildcard_still_frees():
    assert len({p for p in _free(sfh={"type": "dpl", "all_params": FREE}) if "sfh" in p}) > 0


def test_agn_wildcard_still_frees():
    freed = _free(sfh={"type": "dpl"}, agn={"type": "composable", "all_params": FREE})
    assert any(p.startswith("agn_") for p in freed)


def test_fixed_wildcard_never_raises():
    """FIXED is imperative and always honourable — it must never trip the guard."""
    freed = _free(sfh={"type": "dpl"}, neb={"type": "cue", "all_params": FIXED})
    assert not any(p.startswith("neb_") for p in freed)


def test_explicit_priors_are_the_documented_workaround():
    freed = _free(sfh={"type": "dpl"}, neb={"type": "cue", "logU": Uniform(-4, -1)})
    assert "neb_logU" in freed


def test_absent_wildcard_is_not_a_wildcard_failure():
    """No wildcard at all is not an empty wildcard — it must not raise."""
    _free(sfh={"type": "dpl"}, neb={"type": "cue"})


# ── The shipped recipes must all survive the guard ────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "star_forming_photometry",
        "quiescent_z0",
        "agn_panchromatic",
        "stochastic_sfh_jwst",
        "mock_recovery_minimal",
        "composable_agn",
        "dust_demo",
        "high_z",
        "photoz",
        "unified_agn",
    ],
)
def test_every_shipped_recipe_parses(name):
    """Three recipes carried a no-op FREE; none may reintroduce one."""
    groups = getattr(tengri.recipes, name)()
    groups.pop("approx", None)
    tengri.parse_groups(**groups)


def test_introspection_path_is_exempt():
    """``recipe_parameters`` enumerates; it does not build a model to fit.

    Builder factories introspect each variant with ``all_params: FREE`` purely to
    surface parameter names, then read ``all_params`` regardless of free/fixed.
    That must not trip a guard aimed at user model construction.
    """
    recs = tengri.recipe_parameters(
        {"sfh": {"type": "dpl"}, "neb": {"type": "cue", "all_params": FREE}},
        free_only=False,
    )
    assert any(r.name.startswith("neb_") for r in recs)

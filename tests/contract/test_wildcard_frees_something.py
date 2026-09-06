# SPDX-License-Identifier: BSD-3-Clause
"""``all_params: FREE`` must free something, or say so.

``FREE`` used to resolve to each parameter's *registry default*, and most
registry defaults are ``Fixed`` scalars. A wildcard therefore resolved cleanly
while leaving every parameter in the group pinned — the fit ran to completion
with that physics frozen, indistinguishable from success at the call site.

Measured before the fix: ``all_params: FREE`` freed 6 params in ``sfh`` and 2 in
top-level ``agn``, but **zero** in ``neb``, ``xray``, ``radio``, ``shock``,
``igm`` and in the AGN sub-blocks. It survived because every worked example in
the docs uses ``sfh={'all_params': FREE}`` — the one group whose registry
defaults were already free.

Two halves, both pinned here:

* **The guard** — a wildcard that frees nothing raises instead of passing
  silently. Asserted against the resolver's decision function directly, so the
  test does not go stale as blocks gain declared ranges.
* **The fix** — a parameter may declare ``free_prior``, the admissible range
  ``FREE`` opens up (normally the range its own ``bound_check`` enforces, which
  a drift test cross-checks). Where no range is declared, the guard still fires
  and the documented workaround is an explicit prior.

A third case was left open by the original guard and closed in #1474: freeing a
strict *subset*. The guard skipped a group as soon as one parameter freed, so a
group holding both kinds resolved in silence — measured xray 3/9, dust 2/7,
shock 3/5, neb 7/8. That now emits a filterable
:class:`WildcardPartialFreeWarning`. It warns rather than raises because a
partial free is sometimes correct (``dust_Rv`` is fixed by definition under a
Calzetti law) and because six of the ten shipped recipes free a strict subset
today — refusing would break all six.
"""

from __future__ import annotations

import re
import warnings

import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed, Uniform
from tengri.config.exceptions import (
    ParameterError,
    WildcardNoOpWarning,
    WildcardPartialFreeWarning,
)

pytestmark = pytest.mark.contract


def _free(**groups) -> set[str]:
    # redshift FIRST so a caller passing its own wins; the reverse order would
    # silently overwrite the value under test.
    return set(tengri.parse_groups(**{"redshift": tengri.Fixed(0.1), **groups}).free_params)


# ── Groups whose wildcard frees nothing must raise ────────────────────


# The guard's *logic* is asserted directly. An integration test that pinned a
# named group ("neb frees nothing") would be asserting a physics fact that this
# very PR is in the business of changing — it would go stale the moment another
# group gains declared ranges, which is the intended direction of travel.


def test_guard_rejects_an_outcome_that_freed_nothing():
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.raises(ParameterError, match=r"freed 0 of 2 parameters"):
        _check_wildcard_freed_something({"neb": [("neb_logU", False), ("neb_fesc", False)]})


def test_guard_accepts_an_outcome_that_freed_even_one():
    """Outcome-based, not intent-based: freeing 1 of N is still freeing.

    It does not *raise* — but since #1474 it does warn, because the caller wrote
    ``all_params`` and got a strict subset. Both halves asserted here so that
    removing either is a test failure.
    """
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.warns(WildcardPartialFreeWarning):
        _check_wildcard_freed_something({"neb": [("neb_logU", True), ("neb_fesc", False)]})


def test_guard_is_per_nesting_level():
    """A sub-block that frees nothing is caught even when its parent frees.

    ``dust_attenuation`` frees ``tau_bc``/``tau_diff`` while ``dust_emission`` may free
    nothing, so the check cannot be per-top-level-group.
    """
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.raises(ParameterError, match=r"'dust_emission'"):
        _check_wildcard_freed_something(
            {
                "dust_attenuation": [("dust_tau_bc", True)],
                "dust_emission": [("dust_qpah", False)],
            }
        )


def test_a_group_with_no_declared_ranges_still_raises_end_to_end():
    """At least one real group must still exercise the guard through parse_groups.

    Found dynamically rather than hard-coded: as blocks gain ``free_prior``
    declarations the set shrinks, and this test should follow rather than break.
    """
    candidates = [
        ("radio", {}),
        ("xray", {"type": "simple"}),
        ("neb", {"type": "cue"}),
    ]
    for group, spec in candidates:
        try:
            tengri.parse_groups(
                sfh={"type": "dpl"},
                redshift=tengri.Fixed(0.1),
                **{group: {**spec, "all_params": FREE}},
            )
        except ParameterError as exc:
            assert "freed 0 of" in str(exc)
            return
    pytest.skip("every candidate group now declares free ranges — guard covered by unit tests")


def test_dla_wildcard_raises_once_every_freeable_param_is_overridden():
    """Explicit overrides shrink what the wildcard covers — possibly to nothing.

    Three DLA params are freeable: ``dla_log_n_hi`` (a free ``Uniform``
    registry default) plus ``dla_temp`` and ``dla_b_turb``, which gained
    declared ``free_prior`` ranges in #887. Override all three and the wildcard
    is left covering only ``dla_z`` — which deliberately has no ``free_prior``,
    because its 0 is a sentinel meaning "use the source redshift". That is the
    shape that used to silently free nothing, so the guard must fire.

    Before #887 overriding ``log_n_hi`` alone was enough, since it was the only
    one with a range. Naming all three keeps the test asserting the guard rather
    than asserting how few parameters happened to be declared.
    """
    with pytest.raises(ParameterError, match=r"'igm\.dla'"):
        tengri.parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            igm={
                "type": "inoue14",
                "dla": {
                    "log_n_hi": Uniform(19, 22),
                    "temp": tengri.Fixed(1e4),
                    "b_turb": tengri.Fixed(0.0),
                    "all_params": FREE,
                },
            },
            redshift=tengri.Fixed(2.0),
        )


def test_dla_bare_wildcard_is_allowed_because_it_frees_one():
    """The guard is about outcome, not intent: freeing 1 of 4 is still freeing."""
    freed = _free(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        igm={"type": "inoue14", "dla": {"all_params": FREE}},
        redshift=tengri.Fixed(2.0),
    )
    assert "dla_log_n_hi" in freed


def test_error_names_the_group_and_the_stuck_params():
    """The message must be actionable: which group, which params, what to do."""
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.raises(ParameterError) as exc:
        _check_wildcard_freed_something({"neb": [("neb_logU", False), ("neb_fesc", False)]})
    msg = str(exc.value)
    assert "'neb'" in msg
    assert "neb_logU" in msg
    assert "Uniform(lo, hi)" in msg


def test_builders_path_resolves_identically_to_the_dict_form():
    """``builders.*(all_params=FREE)`` lowers to the same wildcard key."""
    built = _free(sfh={"type": "dpl"}, neb=tengri.builders.neb.cue(all_params=FREE))
    dictform = _free(sfh={"type": "dpl"}, neb={"type": "cue", "all_params": FREE})
    assert built == dictform


# ── A partial free must be reported, not swallowed (#1474) ────────────

# The guard used to skip a group the moment *one* parameter freed, so the
# common case — some declare `free_prior`, some do not — passed in silence.
# Measured then: xray 3/9, dust 2/7, shock 3/5, neb 7/8, all silent.


def test_guard_warns_when_it_frees_only_a_subset():
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.warns(WildcardPartialFreeWarning, match=r"freed 1 of 3 parameters"):
        _check_wildcard_freed_something(
            {"xray": [("xray_gamma_agn", True), ("xray_E_cut", False), ("xray_alpha_irx", False)]}
        )


def test_the_partial_warning_names_every_stuck_param():
    """Naming them is the whole point — a bare count is not actionable."""
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.warns(WildcardPartialFreeWarning) as rec:
        _check_wildcard_freed_something(
            {"xray": [("xray_gamma_agn", True), ("xray_E_cut", False), ("xray_alpha_irx", False)]}
        )
    message = str(rec[0].message)
    assert "xray_E_cut" in message
    assert "xray_alpha_irx" in message
    # The one that DID free must not be listed as stuck.
    assert "xray_gamma_agn" not in message.split("stay pinned:")[1]


def test_a_full_free_stays_silent():
    """The warning must be conditional, or it is noise rather than signal."""
    from tengri.parameters.groups import _check_wildcard_freed_something

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _check_wildcard_freed_something({"agn.torus": [("agn_oa", True), ("agn_t_torus", True)]})
    assert not [w for w in caught if issubclass(w.category, WildcardPartialFreeWarning)]


def test_freeing_nothing_still_raises_rather_than_warning():
    """Zero keeps the harder response — a warning there would be a regression."""
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.raises(ParameterError):
        _check_wildcard_freed_something({"neb": [("neb_logU", False), ("neb_fesc", False)]})


def test_partial_warning_is_filterable_by_category():
    """A deliberate partial free must be silenceable without silencing all."""
    from tengri.parameters.groups import _check_wildcard_freed_something

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", WildcardPartialFreeWarning)
        _check_wildcard_freed_something({"neb": [("neb_logU", True), ("neb_fesc", False)]})
    assert not caught


def test_partial_free_warns_end_to_end_through_parse_groups():
    """Not just the decision function — the real user-facing path."""
    with pytest.warns(WildcardPartialFreeWarning, match=r"group 'xray'"):
        tengri.parse_groups(
            sfh={"type": "dpl"}, xray={"type": "simple", "all_params": FREE}, redshift=Fixed(0.1)
        )


@pytest.mark.parametrize(
    "groups",
    [
        {"sfh": {"type": "dpl"}, "xray": {"type": "simple", "all_params": FREE}},
        {"shock": {"all_params": FREE}},
        # NOTE: The "sfh-with-met-params" edge case was removed after #1796.
        # Before the fix, sfh would incorrectly carry met_* params when there
        # was no met block, making the wildcard advice tricky. The fix excludes
        # met_* from sfh's wildcard scope, so this is no longer an edge case.
    ],
    ids=["xray", "shock"],
)
def test_the_advice_the_partial_warning_gives_actually_works(groups):
    """Execute the remedy the message prints, rather than a hand-written twin.

    The example is built by stripping a group prefix that does not always match
    the parameter (``sfh`` holds ``met_*``), so the printed key is the thing
    under test. Parse it back out and run it.
    """
    with pytest.warns(WildcardPartialFreeWarning) as rec:
        tengri.parse_groups(**{"redshift": Fixed(0.1), **groups})

    match = re.search(r"e\.g\. (\w+)=\{'([^']+)': Uniform", str(rec[0].message))
    assert match, f"the warning no longer prints a runnable example: {rec[0].message}"
    group_kwarg, param_key = match.group(1), match.group(2)

    # Bracket each parameter's own Fixed default rather than reusing one span:
    # `xray_E_cut` must stay > 0, so a blanket Uniform(0, 1) is inadmissible.
    full = param_key if param_key.startswith(f"{group_kwarg}_") else f"{group_kwarg}_{param_key}"
    try:
        pinned = tengri.describe_parameter(full).prior.value
    except (KeyError, AttributeError):
        pinned = tengri.describe_parameter(param_key).prior.value
    lo, hi = sorted((pinned * 0.5, pinned * 2.0)) if pinned else (-0.5, 0.5)

    patched = {k: dict(v) for k, v in groups.items()}
    patched.setdefault(group_kwarg, {})[param_key] = Uniform(lo, hi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WildcardPartialFreeWarning)
        freed = set(tengri.parse_groups(**{"redshift": Fixed(0.1), **patched}).free_params)

    assert any(param_key in name for name in freed), (
        f"the advice {group_kwarg}={{{param_key!r}: ...}} did not free anything"
    )


# ── Everything that legitimately works must keep working ──────────────


def test_sfh_wildcard_still_frees():
    assert len({p for p in _free(sfh={"type": "dpl", "all_params": FREE}) if "sfh" in p}) > 0


def test_agn_wildcard_still_frees():
    freed = _free(sfh={"type": "dpl"}, agn={"type": "composable", "all_params": FREE})
    assert any(p.startswith("agn_") for p in freed)


def test_fixed_wildcard_never_raises():
    """Fixed(DEFAULT) is imperative and always honourable — it must never trip the guard."""
    freed = _free(sfh={"type": "dpl"}, neb={"type": "cue", "all_params": Fixed(DEFAULT)})
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


# ── FREE must actually free where a range is declared ─────────────────


@pytest.mark.parametrize(
    "group,spec,prefix",
    [
        ("neb", {"type": "cue"}, "neb_"),
        ("shock", {}, "shock_"),
    ],
)
def test_wildcard_free_now_frees_declared_params(group, spec, prefix):
    """The point of ``free_prior``: FREE opens the declared admissible range.

    Before it existed these groups freed exactly zero parameters and said
    nothing about it.
    """
    freed = _free(sfh={"type": "dpl"}, **{group: {**spec, "all_params": FREE}})
    assert [p for p in freed if p.startswith(prefix)]


def test_free_uses_the_declared_range_not_the_fixed_default():
    freed = tengri.parse_groups(
        sfh={"type": "dpl"}, neb={"type": "cue", "all_params": FREE}, redshift=Fixed(0.1)
    )
    logu = freed.get_distribution("neb_logU")
    assert not logu.is_fixed
    assert logu.bounds == (-5.0, 0.0)  # the range neb_logU's bound_check enforces


def test_free_prior_never_contradicts_its_own_bound_check():
    """Drift guard: a declared range must satisfy the bound the same
    declaration enforces, or the two have diverged."""
    import importlib
    import pkgutil

    import tengri.components as components_pkg
    from tengri.protocols.component import ParamDeclaration

    modules = [
        mi.name
        for mi in pkgutil.walk_packages(
            components_pkg.__path__, prefix=components_pkg.__name__ + "."
        )
        if mi.name.endswith("._params")
    ]
    modules += ["tengri.observation._params", "tengri.parameters._shared"]

    violations = []
    for module_name in modules:
        module = importlib.import_module(module_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not isinstance(attr, tuple) or not attr:
                continue
            if not all(isinstance(x, ParamDeclaration) for x in attr):
                continue
            for decl in attr:
                if decl.free_prior is None or decl.bound_check is None:
                    continue
                lo, hi = decl.free_prior.bounds
                if not decl.bound_check(lo, hi):
                    violations.append(
                        f"{decl.name}: free_prior {(lo, hi)} vs {decl.bound_error!r}"
                    )
    assert not violations, "free_prior contradicts bound_check for: " + "; ".join(violations)


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


# ── A wildcard that covers ZERO parameters must raise, not stay mute ──

# The guard above (#1474) fires only for a group that already appears in
# ``wildcard_free_outcome`` -- built by the resolve loop *appending* an
# outcome for every parameter it actually visits. When the selected
# structural choice declares no parameters for a group at all, the loop
# never visits a single one, so the group never becomes a key in that dict:
# not even an empty list. ``igm={'type': 'inoue14', 'all_params': FREE}``
# freed nothing and warned nothing, indistinguishable from success. Same
# failure for ``radio``/``shock`` when every sub-model they can select is
# switched to ``'none'`` (no component is built at all).
#
# This is the fourth outcome :func:`_check_wildcard_freed_something`
# adjudicates, fed by :func:`_seed_zero_declaration_wildcards` giving such a
# group an explicit empty entry before the check runs. It used to warn
# (:class:`WildcardNoOpWarning`); #2187 escalated it to
# :class:`ParameterError` -- see the regression class below for the bug that
# motivated the escalation: a warning here is exactly as swallowable as the
# silence it replaced, and every group covered by
# :func:`_seed_zero_declaration_wildcards`'s old hand-maintained census
# (``igm``/``radio``/``shock``) was itself an incomplete accounting of every
# way a group can legitimately cover zero parameters.


def test_guard_raises_on_an_outcome_with_zero_candidates():
    """Unit level: an empty entry raises rather than being silently skipped.

    This used to warn (:class:`WildcardNoOpWarning`); #2187 escalated it to
    :class:`ParameterError` because a warning is exactly as swallowable as
    the silence it replaced, and an empty wildcard is never useful.
    """
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.raises(ParameterError, match=r"group 'igm' covers no parameters"):
        _check_wildcard_freed_something({"igm": []})


def test_seed_zero_declaration_wildcards_is_selective():
    """The seeding helper only adds an entry for a FREE-disposed, absent group."""
    from tengri.parameters.groups import _seed_zero_declaration_wildcards

    # FREE on a zero-declaration group: seeded with an empty entry.
    assert _seed_zero_declaration_wildcards({}, {"igm": {"*": FREE}}) == {"igm": []}

    # Fixed(DEFAULT) is the documented suppression idiom -- never seeded.
    assert _seed_zero_declaration_wildcards({}, {"igm": {"*": Fixed(DEFAULT)}}) == {}

    # No disposition stated at all -- never seeded.
    assert _seed_zero_declaration_wildcards({}, {"igm": {"type": "inoue14"}}) == {}

    # A group the resolve loop already populated is left exactly as-is, not
    # overwritten with an empty list.
    existing = {"igm": [("igm_bubble_mpc", True)]}
    assert _seed_zero_declaration_wildcards(existing, {"igm": {"*": FREE}}) == existing


def test_igm_inoue14_wildcard_free_raises_no_op():
    """``inoue14`` names no top-level knob without ``patchy`` -- FREE frees nothing.

    Was a warning (:class:`WildcardNoOpWarning`); #2187 escalated it to raise.
    """
    with pytest.raises(ParameterError, match=r"group 'igm' covers no parameters"):
        tengri.parse_groups(
            sfh={"type": "dpl"}, igm={"type": "inoue14", "all_params": FREE}, redshift=Fixed(2.0)
        )


def test_igm_inoue14_wildcard_free_raises_no_op_other_params_spelling():
    """Same repro under the ``other_params`` synonym."""
    with pytest.raises(ParameterError, match=r"group 'igm' covers no parameters"):
        tengri.parse_groups(
            sfh={"type": "dpl"},
            igm={"type": "inoue14", "other_params": FREE},
            redshift=Fixed(2.0),
        )


def test_igm_inoue14_fixed_default_wildcard_does_not_warn():
    """``Fixed(DEFAULT)`` is the documented suppression idiom; never warns here."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tengri.parse_groups(
            sfh={"type": "dpl"},
            igm={"type": "inoue14", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(2.0),
        )
    assert not [w for w in caught if issubclass(w.category, WildcardNoOpWarning)]


def test_igm_patchy_wildcard_free_does_not_get_the_no_op_warning():
    """With ``patchy=True`` the wildcard genuinely frees ``igm_bubble_mpc``/``igm_x_HI``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        freed = set(
            tengri.parse_groups(
                sfh={"type": "dpl"},
                igm={"type": "inoue14", "patchy": True, "all_params": FREE},
                redshift=Fixed(2.0),
            ).free_params
        )
    assert not [w for w in caught if issubclass(w.category, WildcardNoOpWarning)]
    assert "igm_bubble_mpc" in freed
    assert "igm_x_HI" in freed


def test_radio_wildcard_free_raises_when_both_submodels_disabled():
    """``radio``'s component is never built once ``sf``/``agn`` are both ``'none'``."""
    with pytest.raises(ParameterError, match=r"group 'radio' covers no parameters"):
        tengri.parse_groups(
            sfh={"type": "dpl"},
            radio={"sf": {"type": "none"}, "agn": {"type": "none"}, "all_params": FREE},
            redshift=Fixed(0.5),
        )


def test_shock_wildcard_free_raises_when_disabled():
    """``shock={'type': 'none'}`` builds no component; FREE has nothing to free."""
    with pytest.raises(ParameterError, match=r"group 'shock' covers no parameters"):
        tengri.parse_groups(
            sfh={"type": "dpl"}, shock={"type": "none", "all_params": FREE}, redshift=Fixed(0.5)
        )


def test_sfh_dpl_wildcard_free_does_not_get_the_no_op_warning():
    """A wildcard that frees something must never trip the zero-candidate warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        freed = _free(sfh={"type": "dpl", "all_params": FREE})
    assert not [w for w in caught if issubclass(w.category, WildcardNoOpWarning)]
    assert any(p.startswith("sfh_") for p in freed)


# ── Regression: the zero-declaration seeding used to be a hand census ─
#
# #2187: ``met`` was the one group with a hand-written zero-declaration seed
# path (its own dedicated ``freed 0 of N`` message), so ``met={'type':
# 'table', 'all_params': FREE}`` already raised. Every OTHER way a group can
# structurally cover zero parameters was invisible to
# :func:`_check_wildcard_freed_something`, because
# :func:`_seed_zero_declaration_wildcards` only ever seeded the three groups
# named in its old hand-maintained census (``igm``, ``radio``, ``shock``).
# Three more mechanisms reach zero and were silent before the fix:
#
# * a structural *type* that declares no parameters at all for its group
#   (``sfh={'type': 'table'}`` -- an externally-supplied SFH has no free
#   knobs of its own);
# * a component whose grid-support scope is the empty frozenset, so every one
#   of its parameters is tagged ``wildcard_fixed_inactive`` before the
#   resolve loop ever records anything (``dust_emission`` types
#   ``dh02_ce01`` and ``pah_drude``);
# * an AGN sub-block whose parameters are excluded from the shared AGN scope
#   under the selected variant (``agn.feii`` with ``qsogen_balmer``).
#
# The fix derives the seed set from the kwargs the caller actually passed
# instead of a census, so all three (and any future case) are covered by
# construction rather than by memory.


class TestZeroDeclarationWildcardsRaise2187:
    """A wildcard covering zero parameters must raise, however it gets there."""

    def test_sfh_table_wildcard_covers_no_parameters(self):
        """``type='table'`` names an externally-supplied SFH with no free knobs."""
        with pytest.raises(ParameterError, match=r"group 'sfh' covers no parameters"):
            tengri.parse_groups(sfh={"type": "table", "all_params": FREE}, redshift=Fixed(0.1))

    @pytest.mark.parametrize("emission_type", ["dh02_ce01", "pah_drude"])
    def test_dust_emission_empty_scope_wildcard_covers_no_parameters(self, emission_type):
        """A component whose grid-support scope is empty tags every param inactive."""
        with pytest.raises(ParameterError, match=r"group 'dust_emission' covers no parameters"):
            tengri.parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": Fixed(DEFAULT),
                },
                dust_emission={"type": emission_type, "all_params": FREE},
                redshift=Fixed(0.1),
            )

    def test_agn_feii_qsogen_balmer_wildcard_covers_no_parameters(self):
        """The shared AGN scope excludes ``qsogen_balmer``'s own parameters."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ParameterError, match=r"group 'agn\.feii' covers no parameters"):
                tengri.parse_groups(
                    sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                    agn={
                        "type": "composable",
                        "feii": {"type": "qsogen_balmer", "all_params": FREE},
                    },
                    redshift=Fixed(0.1),
                )

    def test_met_table_still_raises_the_freed_0_of_form(self):
        """The one group with a pre-existing dedicated check keeps its own message.

        ``met={'type': 'table'}`` declares ``met_alpha_fe``, so this is the
        *freed 0 of N* outcome, not the *covers no parameters* one -- distinct
        code paths that must both keep working.
        """
        with pytest.raises(ParameterError, match=r"freed 0 of"):
            tengri.parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                met={"type": "table", "all_params": FREE},
                redshift=Fixed(0.1),
            )

    def test_the_error_names_the_group_the_user_actually_wrote(self):
        """Actionable: the message must name the dotted sub-block, not just 'agn'."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ParameterError) as exc:
                tengri.parse_groups(
                    sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                    agn={
                        "type": "composable",
                        "feii": {"type": "qsogen_balmer", "all_params": FREE},
                    },
                    redshift=Fixed(0.1),
                )
        assert "agn.feii" in str(exc.value)


# ── Regression: an explicit per-parameter FREE must be honored or refused ─
#
# Follow-up to #2187, same seam: ``FREE`` named for one specific parameter
# used to resolve exactly like the wildcard's per-parameter fallback -- if
# the parameter has no declared ``free_prior``, it silently stayed Fixed at
# its registry default. Measured: ``sfh={'type': 'snorm_burst', 'burst_sfr':
# FREE, 'all_params': Fixed(DEFAULT)}`` built with zero free parameters,
# ``burst_sfr`` pinned at ``Fixed(0.0)``, and no warning at all -- worse than
# the wildcard case, because the user named this exact parameter and asked
# for it by name.
#
# The wildcard's per-parameter fallback (``'*': FREE`` reaching a parameter
# with no declared range) must keep resolving to Fixed silently: that is the
# input the group-level adjudicator (:func:`_check_wildcard_freed_something`)
# needs in order to count outcomes and report a partial free or a freed-0-of-N
# refusal with the *group's* context. Only the explicitly-named path raises
# here, tagged ``"user_free"`` at the point of resolution -- never the
# wildcard's ``"wildcard_free"`` tag.


class TestExplicitPerParamFreeMustBeHonoredOrRefused2187:
    """A parameter named ``FREE`` by the user must actually free, or refuse."""

    def test_explicit_free_on_a_param_with_no_free_prior_raises(self):
        """``burst_sfr`` declares no free prior: 'no galaxy-independent interval exists'."""
        with pytest.raises(ParameterError, match=r"no declared free prior") as exc:
            tengri.parse_groups(
                sfh={
                    "type": "snorm_burst",
                    "burst_sfr": FREE,
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.1),
            )
        assert "burst_sfr" in str(exc.value)

    def test_wildcard_free_on_the_same_group_still_takes_the_adjudication_path(self):
        """The wildcard must never trip the new per-parameter error.

        ``snorm_burst`` mixes params with and without a declared free prior,
        so its wildcard is the pre-existing partial-free outcome (#1474), not
        the new per-parameter refusal.
        """
        with pytest.warns(WildcardPartialFreeWarning, match=r"group 'sfh'"):
            freed = _free(sfh={"type": "snorm_burst", "all_params": FREE})
        assert any(p.startswith("sfh_snorm_burst_") for p in freed)

    def test_explicit_free_on_a_param_with_a_declared_free_prior_still_works(self):
        """The documented-working case: naming a freeable parameter by hand."""
        freed = _free(sfh={"type": "dpl", "alpha": FREE, "all_params": Fixed(DEFAULT)})
        assert "sfh_dpl_alpha" in freed

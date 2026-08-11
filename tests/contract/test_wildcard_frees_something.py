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
from tengri import FIXED, FREE, Uniform
from tengri.config.exceptions import ParameterError, WildcardPartialFreeWarning

pytestmark = pytest.mark.contract


def _free(**groups) -> set[str]:
    return set(tengri.parse_groups(**groups).free_params)


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

    ``dust`` frees ``tau_bc``/``tau_diff`` while ``dust.emission`` may free
    nothing, so the check cannot be per-top-level-group.
    """
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.raises(ParameterError, match=r"'dust\.emission'"):
        _check_wildcard_freed_something(
            {
                "dust": [("dust_tau_bc", True)],
                "dust.emission": [("dust_qpah", False)],
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
            tengri.parse_groups(sfh={"type": "dpl"}, **{group: {**spec, "all_params": FREE}})
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
            sfh={"type": "dpl", "*": FIXED},
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
        sfh={"type": "dpl", "*": FIXED},
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


def test_star_synonym_resolves_identically_to_all_params():
    """``'*'`` is a synonym: it must reach the same resolution, not a bypass."""
    star = _free(sfh={"type": "dpl"}, neb={"type": "cue", "*": FREE})
    alias = _free(sfh={"type": "dpl"}, neb={"type": "cue", "all_params": FREE})
    assert star == alias
    assert any(p.startswith("neb_") for p in star)


def test_builders_path_resolves_identically_to_the_dict_form():
    """``builders.*(defaults=FREE)`` lowers to the same wildcard key."""
    built = _free(sfh={"type": "dpl"}, neb=tengri.builders.neb.cue(defaults=FREE))
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
        tengri.parse_groups(sfh={"type": "dpl"}, xray={"type": "simple", "all_params": FREE})


@pytest.mark.parametrize(
    "groups",
    [
        {"sfh": {"type": "dpl"}, "xray": {"type": "simple", "all_params": FREE}},
        {"shock": {"all_params": FREE}},
        # `sfh` carries met_* params, so the short-form prefix strip does not
        # apply and the message must fall back to the full name. Advice that
        # only works when the prefixes happen to line up is advice that raises.
        {"sfh": {"type": "dpl", "all_params": FREE}},
    ],
    ids=["xray", "shock", "sfh-with-met-params"],
)
def test_the_advice_the_partial_warning_gives_actually_works(groups):
    """Execute the remedy the message prints, rather than a hand-written twin.

    The example is built by stripping a group prefix that does not always match
    the parameter (``sfh`` holds ``met_*``), so the printed key is the thing
    under test. Parse it back out and run it.
    """
    with pytest.warns(WildcardPartialFreeWarning) as rec:
        tengri.parse_groups(**groups)

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
        freed = set(tengri.parse_groups(**patched).free_params)

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
    freed = tengri.parse_groups(sfh={"type": "dpl"}, neb={"type": "cue", "all_params": FREE})
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

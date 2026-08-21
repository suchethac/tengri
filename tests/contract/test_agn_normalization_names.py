# SPDX-License-Identifier: BSD-3-Clause
"""The AGN normalizations have names that say which one they are (#1296).

Four parameters carried four *different* physical definitions behind
near-identical names, two of them distinguishable only by capitalization:

    agn_fracAGN    Uniform(0, 0.99)  AGN fraction of the total dust IR
    agn_frac_agn   Uniform(0, 1.0)   L_AGN / L_total in a configurable band
    agn_frac       Uniform(0, 5.0)   L_AGN / L_stellar_bol
    dust_frac_agn  Fixed(0.0)        Dale 2014 additive AGN-heated dust

A user reaching for "the AGN fraction" had four wrong answers available and no
error to tell them apart.

The renames, each via ``_LEGACY_PARAM_ALIASES`` + ``DeprecationWarning``:

    agn_fracAGN  -> agn_ir_frac     the AGN share of the dust IR
    agn_frac_agn -> agn_band_frac   the AGN share in a band
    agn_frac     -> agn_lum_ratio   a *ratio* (0-5), not a fraction

``dust_frac_agn`` keeps its name: the ``dust_`` prefix already says which
component owns it.

**The trap this avoided.** The obvious camelCase fix, ``agn_fracAGN`` ->
``agn_frac_agn``, targets a name that already existed as a different quantity
with a different prior in the same module. Applying it would have silently
merged two AGN normalizations — in the code path #556 already had to
disentangle once. ``test_no_rename_collided`` pins that they stayed distinct.
"""

from __future__ import annotations

import warnings

import pytest

import tengri
from tengri.parameters._aliases import _LEGACY_PARAM_ALIASES, resolve_param_name

pytestmark = pytest.mark.contract

RENAMES = {
    "agn_fracAGN": "agn_ir_frac",
    "agn_frac_agn": "agn_band_frac",
    "agn_frac": "agn_lum_ratio",
}

#: Must NOT have moved — they were never ambiguous.
UNTOUCHED = ["agn_frac_hot", "agn_torus_frac", "dust_frac_agn", "noise_frac_cal"]

REGISTERED = {r["name"] for r in tengri.list_parameters()}


@pytest.mark.parametrize("old,new", sorted(RENAMES.items()))
def test_the_new_name_is_registered(old, new):
    assert new in REGISTERED, f"{new} is not in the registry"


@pytest.mark.parametrize("old", sorted(RENAMES))
def test_the_old_name_is_gone_from_the_registry(old):
    """Deprecated, not dual-registered: one canonical name per quantity."""
    assert old not in REGISTERED, (
        f"{old} is still registered. The alias should resolve it, not "
        "duplicate it — two live names for one quantity is the defect."
    )


@pytest.mark.parametrize("old,new", sorted(RENAMES.items()))
def test_the_old_name_still_resolves_with_one_warning(old, new):
    """Breaking silently is worse than breaking loudly; do neither."""
    from tengri.parameters import _aliases

    _aliases._WARNED_ALIASES.discard(old)  # the warning is once-per-process
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = resolve_param_name(old)
    assert got == new, f"{old} resolved to {got!r}, expected {new!r}"
    deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deps) == 1, f"{old} emitted {len(deps)} DeprecationWarnings, expected 1"
    assert new in str(deps[0].message), "the warning must name the replacement"


@pytest.mark.parametrize("name", UNTOUCHED)
def test_the_unambiguous_siblings_did_not_move(name):
    assert name in REGISTERED, f"{name} was renamed; it should not have been"
    assert name not in _LEGACY_PARAM_ALIASES, f"{name} gained a spurious alias"


def test_no_rename_collided():
    """The renames must not have merged two parameters into one.

    ``agn_fracAGN -> agn_frac_agn`` (the plan's proposal) would have done
    exactly that. Distinct priors are the observable proof they stayed separate.
    """
    priors = {}
    for new in RENAMES.values():
        priors[new] = tengri.describe_parameter(new).prior
    assert len(set(map(str, priors.values()))) == len(priors), (
        f"two renamed parameters ended up identical: {priors}. A collision "
        "would silently merge two AGN normalizations."
    )
    # The specific values, pinned — a merge would change one of them.
    assert str(priors["agn_ir_frac"]).startswith("Uniform(0.0, 0.99")
    assert str(priors["agn_band_frac"]).startswith("Uniform(0.0, 1.0")
    assert str(priors["agn_lum_ratio"]).startswith("Uniform(0.0, 5.0")


def test_agn_lum_ratio_no_longer_calls_itself_a_fraction():
    """Its range runs to 5.0; "fraction" was wrong in the prose as well."""
    desc = (tengri.describe_parameter("agn_lum_ratio").description or "").lower()
    assert "ratio" in desc
    _lo, hi = tengri.describe_parameter("agn_lum_ratio").prior.bounds
    assert hi > 1.0, "if the bound is now <= 1 this really is a fraction; rename back"


def test_each_new_name_says_which_normalization_it_is():
    """The point of the rename: the name must disambiguate, not just differ."""
    expectations = {
        "agn_ir_frac": "ir",
        "agn_band_frac": "band",
        "agn_lum_ratio": "ratio",
    }
    for name, token in expectations.items():
        desc = (tengri.describe_parameter(name).description or "").lower()
        assert token in desc, (
            f"{name}'s description never mentions {token!r}, so the name and "
            "the documentation disagree about what it is."
        )


#: The grammar's *short* per-parameter key, before -> after.
SHORT_FORMS = {"frac": "lum_ratio", "fracAGN": "ir_frac"}


@pytest.mark.parametrize("legacy,current", sorted(SHORT_FORMS.items()))
def test_the_legacy_short_form_key_still_works(legacy, current):
    """Renaming a parameter also invalidates its short key — it must not.

    ``agn={'frac': 0.5}`` is the nested-dict grammar's short form, derived by
    stripping the ``agn_`` prefix. The rename turned it into
    ``ValueError: Unknown key 'frac' in group 'agn'`` in 34 places before the
    alias covered short keys too.
    """
    from tengri import Fixed
    from tengri.parameters import _aliases
    from tengri.parameters.groups import parse_groups

    _aliases._WARNED_ALIASES.discard(legacy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_spec = parse_groups(
            agn={"type": "composable", legacy: Fixed(0.37)}, redshift=Fixed(0.1)
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        f"agn={{'{legacy}': ...}} must warn, not be silently accepted"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        current_spec = parse_groups(
            agn={"type": "composable", current: Fixed(0.37)}, redshift=Fixed(0.1)
        )

    assert legacy_spec.get_fixed_values() == current_spec.get_fixed_values(), (
        f"agn={{'{legacy}'}} and agn={{'{current}'}} resolve differently — the "
        "alias accepted the key but did not carry its value through."
    )


def test_the_short_form_probe_can_detect_a_difference():
    """Guard the guard: the equality above must not be trivially true."""
    from tengri import Fixed
    from tengri.parameters.groups import parse_groups

    a = parse_groups(agn={"type": "composable", "lum_ratio": Fixed(0.37)}, redshift=Fixed(0.1))
    b = parse_groups(agn={"type": "composable", "lum_ratio": Fixed(0.99)}, redshift=Fixed(0.1))
    assert a.get_fixed_values() != b.get_fixed_values(), (
        "get_fixed_values() cannot distinguish two different values, so the "
        "equivalence test above proves nothing"
    )


#: Deliberate prose mentions of a legacy name, with the reason. A docstring
#: that explains *why* a parameter was renamed has to be able to say the old
#: name; what must not survive is a stale *use* of it.
DOCUMENTED_MENTIONS = {
    ("components/agn/_params.py", "agn_frac"): (
        "agn_lum_ratio's description explains that it is a ratio, not a "
        "fraction, which is why it is no longer called agn_frac"
    ),
    ("parameters/groups.py", "agn_frac"): (
        "the short-form alias comment cites agn_frac -> agn_lum_ratio as the "
        "rename that invalidated the grammar's short key `agn={'frac': ...}`"
    ),
}


def test_no_old_name_survives_in_the_shipped_package():
    """A stale *use* in src/ would resolve via alias and warn at runtime."""
    import pathlib
    import re

    src = pathlib.Path(tengri.__file__).parent
    offenders = []
    for path in src.rglob("*.py"):
        if path.name == "_aliases.py":  # owns the legacy keys
            continue
        rel = str(path.relative_to(src))
        text = path.read_text()
        for old in RENAMES:
            if not re.search(rf"\b{re.escape(old)}\b", text):
                continue
            if (rel, old) in DOCUMENTED_MENTIONS:
                continue
            offenders.append(f"{rel}:{old}")
    assert not offenders, (
        f"old parameter names still used in src/: {offenders}. They would "
        "resolve through the alias and emit a DeprecationWarning from inside "
        "the package. If the mention is deliberate prose explaining the "
        "rename, add it to DOCUMENTED_MENTIONS with the reason."
    )


def test_the_documented_mentions_are_still_real():
    """A stale exemption would hide a genuine stale reference."""
    import pathlib
    import re

    src = pathlib.Path(tengri.__file__).parent
    for (rel, old), reason in DOCUMENTED_MENTIONS.items():
        path = src / rel
        assert path.exists(), f"exempted file no longer exists: {rel} ({reason})"
        assert re.search(rf"\b{re.escape(old)}\b", path.read_text()), (
            f"{rel} no longer mentions {old!r}, so its exemption is stale"
        )

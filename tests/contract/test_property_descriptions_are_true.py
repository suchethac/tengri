# SPDX-License-Identifier: BSD-3-Clause
"""A published property description is a claim; a claim has to be true.

``list_properties()`` and ``docs/api/_property_table.md`` are how a user finds
the 50 derived quantities and decides which one to quote in a paper. Two of
those descriptions were false:

* ``stellar_mass`` was published as *"Total stellar mass currently alive"*. It
  reads ``log_mstar_formed`` and is bit-exactly the mass **formed** — 1.56x to
  1.71x the mass still alive on the two models measured. Nothing caught it,
  because the registry ``doc=`` and the function docstring are two hand-written
  copies of one sentence and *both* said "alive": a consistency check between
  duplicates is structurally blind to an error present in both.
* ``ssfr`` was published as *"SFR / stellar_mass"*. It divides by the
  **surviving** mass. Following the published formula gave 0.59-0.64x the
  reported number — and it named the one other property whose own description
  was also wrong, so the two errors compounded.

Guard A executes the published sentence. The formula is *parsed out of the
description* rather than transcribed beside it: a hand-written copy of the
right formula would keep passing while the published text said something else,
which is precisely the two-copies failure that produced the bug. Two ways to
fail, and the shipped wording hits both:

1. a name in the published formula that is not a property the reader can look
   up (``SFR`` is not one), so the formula cannot be followed at all;
2. a formula that evaluates to a different number than the property reports.

Guard B pins the two masses: the numbers must match the words.
"""

from __future__ import annotations

import ast
import operator
import re

import jax
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel
from tengri.forward.properties import PROPERTY_REGISTRY

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

# `name op name (op name)*` over ASCII identifiers. '-' is excluded on purpose:
# it is ordinary prose here ("mass-remaining", "X-ray", "1.5-1.9x"), so
# including it would flag sentences that state no arithmetic at all.
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_FORMULA = re.compile(rf"{_IDENT}(?:\s*[/*+]\s*{_IDENT})+")


def _descriptions() -> dict[str, str]:
    """Every published description, read off the live registry."""
    out = {}
    for name, entries in PROPERTY_REGISTRY.items():
        for entry in entries:
            doc = getattr(entry, "doc", "") or ""
            if doc:
                out[name] = doc
    return out


def _published_formula(name: str, description: str) -> str | None:
    """The arithmetic a description states over *other* properties, if any.

    Returns the matched expression verbatim, so what gets evaluated is the
    string the user reads — not a paraphrase of it.
    """
    known = set(PROPERTY_REGISTRY)
    for match in _FORMULA.finditer(description):
        expression = match.group(0)
        names = set(re.findall(_IDENT, expression))
        # Arithmetic that mentions no property (prose like "L_TIR / nu*L_nu")
        # is not a claim this guard can or should execute.
        if names & known - {name}:
            return expression
    return None


_BINARY_OPS = {ast.Div: operator.truediv, ast.Mult: operator.mul, ast.Add: operator.add}


def _evaluate(formula: str, bindings: dict[str, float]) -> float:
    """Evaluate a published formula over property values.

    Walks the AST rather than calling :func:`eval`: the only nodes accepted are
    the three operators the formula pattern can produce and names that are
    bound to property values, so a description can never do anything but
    arithmetic on numbers this test already holds.
    """

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            return _BINARY_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Name):
            return bindings[node.id]
        raise AssertionError(f"unexpected node {ast.dump(node)} in formula {formula!r}")

    return float(walk(ast.parse(formula, mode="eval")))


def _formula_census() -> dict[str, str]:
    """{property: the formula its published description states}."""
    return {
        name: formula
        for name, description in _descriptions().items()
        if (formula := _published_formula(name, description)) is not None
    }


@pytest.fixture(scope="module")
def real_model(ssp_data_fsps):
    """A real SSP, because the synthetic fixture has no mass-remaining table.

    Without one, ``stellar_mass_surviving`` is NaN and ``ssfr`` falls back to
    the formed mass by design — so the very distinction under test disappears.
    """
    obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i"]))
    return SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust={"type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def real_props(real_model):
    params = dict(real_model.spec.get_fixed_values())
    if real_model.spec.free_params:
        params.update(real_model.spec.sample(jax.random.PRNGKey(0)))
    return real_model.predict(params).properties


class TestFormulaDescriptions:
    """Guard A: a published formula must be followable, and must be true."""

    def test_the_census_finds_something(self):
        """A scan that finds nothing would pass every other test vacuously."""
        census = _formula_census()
        assert census, (
            "no published description states arithmetic over another property — "
            "either the scan is broken or every formula description was removed; "
            "both need a look before this file is trusted."
        )

    @pytest.mark.parametrize("name", sorted(_formula_census()))
    def test_every_name_in_a_published_formula_is_a_property(self, name):
        """A formula the reader cannot resolve is a formula they cannot follow."""
        formula = _formula_census()[name]
        unresolvable = sorted(set(re.findall(_IDENT, formula)) - set(PROPERTY_REGISTRY))
        assert not unresolvable, (
            f"{name} is published as {_descriptions()[name]!r}. Its formula "
            f"{formula!r} names {unresolvable}, which are not properties — a "
            f"reader cannot look them up or compute them. Name the actual "
            f"properties (this is how 'SFR / stellar_mass' hid the fact that "
            f"the denominator is stellar_mass_surviving)."
        )

    @pytest.mark.parametrize("name", sorted(_formula_census()))
    def test_the_published_formula_reproduces_the_value(self, name, real_props):
        formula = _formula_census()[name]
        bindings = {other: float(real_props[other]) for other in re.findall(_IDENT, formula)}
        from_description = _evaluate(formula, bindings)
        reported = float(real_props[name])
        assert np.isfinite(reported) and np.isfinite(from_description), (
            f"{name}: reported={reported}, from description={from_description} — "
            f"a non-finite value cannot check the claim; fix the fixture."
        )
        assert np.isclose(reported, from_description, rtol=1e-9, atol=0.0), (
            f"{name} does not equal what its published description says it is.\n"
            f"  description : {_descriptions()[name]!r}\n"
            f"  formula     : {formula}\n"
            f"  reported    : {reported:.9e}\n"
            f"  from formula: {from_description:.9e}\n"
            f"  ratio       : {from_description / reported:.6f}\n"
            f"A user following the published formula gets the wrong number."
        )

    def test_the_guard_catches_the_wording_that_shipped(self):
        """Neuter check, in-process, against the exact text that was published."""
        shipped = "Specific star formation rate (SFR / stellar_mass)"
        formula = _published_formula("ssfr", shipped)
        assert formula == "SFR / stellar_mass", (
            f"the guard does not read {shipped!r} as stating a formula, so it "
            f"could not have caught the bug it exists for. Got: {formula!r}"
        )
        unresolvable = sorted(set(re.findall(_IDENT, formula)) - set(PROPERTY_REGISTRY))
        assert unresolvable == ["SFR"], (
            f"expected the guard to reject 'SFR' as unresolvable; got {unresolvable}"
        )

    def test_a_passing_mention_is_not_read_as_a_formula(self):
        """The converse, so the census cannot be satisfied by flagging everything."""
        mention = "Total stellar mass formed by the SFH, see stellar_mass_surviving"
        assert _published_formula("stellar_mass", mention) is None, (
            "a cross-reference with no arithmetic was read as a formula; the "
            "census would then demand executable claims from ordinary prose."
        )


class TestTheTwoMasses:
    """Guard B: the numbers must match the words."""

    def test_stellar_mass_is_the_formed_mass(self, real_model, real_props):
        """It reads log_mstar_formed — so it must equal 10**log_mstar_formed."""
        params = dict(real_model.spec.get_fixed_values())
        state = real_model.predict_state(params)
        formed = float(10.0 ** np.asarray(state.derived["log_mstar_formed"]))
        assert np.isclose(float(real_props["stellar_mass"]), formed, rtol=1e-12, atol=0.0)

    def test_the_formed_mass_exceeds_the_surviving_mass(self, real_props):
        """The gap is the whole reason the two names must not be confused."""
        formed = float(real_props["stellar_mass"])
        surviving = float(real_props["stellar_mass_surviving"])
        assert np.isfinite(surviving), (
            "stellar_mass_surviving is not finite on a real SSP — this fixture "
            "cannot tell the two masses apart, so it cannot test them."
        )
        assert surviving < formed, (
            f"surviving ({surviving:.4e}) is not below formed ({formed:.4e}); "
            f"mass loss should put it 1.5-1.9x lower."
        )

    @pytest.mark.parametrize(
        ("name", "required_word"),
        [("stellar_mass", "formed"), ("stellar_mass_surviving", "surviving")],
    )
    def test_each_mass_description_carries_the_word_that_distinguishes_it(
        self, name, required_word
    ):
        """ "Currently alive" describes the other one, and shipped for both."""
        description = _descriptions()[name].lower()
        assert required_word in description, (
            f"{name} is published as {_descriptions()[name]!r}, which never says "
            f"{required_word!r} — the one word that separates it from its "
            f"sibling. A reader cannot tell which of the two masses they got."
        )

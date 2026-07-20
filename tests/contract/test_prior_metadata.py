# SPDX-License-Identifier: BSD-3-Clause
"""Every distribution can carry its own units and description (#1286).

Three of the seven priors accepted ``description``/``units``; four rejected
them with ``TypeError``. ``LogUniform`` is the natural prior for a luminosity,
a mass, or a density — precisely the quantities whose units most need stating —
so a bolometric luminosity's prior could not record ``erg/s``, and
``describe_parameter`` had nothing to show for any log-uniform parameter.

The asymmetry also leaked outwards: because ``LogUniform(...).description``
raised ``AttributeError`` rather than returning ``""``, consumers read the
field as ``getattr(prior, "units", "")``. That fail-open guard existed only to
paper over an incomplete base class, so the fix is on ``Distribution`` itself —
class-level defaults — with the four constructors extended to match
``Uniform``/``Gaussian``/``Fixed``.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.parameters.priors import Distribution

pytestmark = pytest.mark.contract

#: Every public distribution, with positional args that construct it validly.
ALL_PRIORS: dict[str, tuple] = {
    "Uniform": (0.0, 1.0),
    "Gaussian": (0.0, 1.0),
    "Fixed": (0.5,),
    "LogUniform": (1e42, 1e47),
    "LogNormal": (0.0, 1.0),
    "StudentT": (0.0, 1.0, 3.0),
    "Laplace": (0.0, 1.0),
}


@pytest.mark.parametrize("name,args", sorted(ALL_PRIORS.items()))
def test_every_prior_accepts_units_and_description(name, args):
    """A luminosity prior must be able to say it is in erg/s.

    Passed by keyword: ``description`` sits after ``lo``/``hi`` on the
    truncated priors and after ``hi`` on ``Uniform``, so its *position*
    differs between classes. Only the keyword is contractual.
    """
    cls = getattr(tengri, name)
    prior = cls(*args, description="bolometric luminosity", units="erg/s")
    assert prior.description == "bolometric luminosity"
    assert prior.units == "erg/s"


@pytest.mark.parametrize("name,args", sorted(ALL_PRIORS.items()))
def test_metadata_defaults_to_empty_not_attribute_error(name, args):
    """Reading ``.units`` off a bare prior must not raise.

    This is what let six call sites decay into ``getattr(p, "units", "")``.
    """
    prior = getattr(tengri, name)(*args)
    assert prior.description == ""
    assert prior.units == ""


@pytest.mark.parametrize("name,args", sorted(ALL_PRIORS.items()))
def test_metadata_does_not_affect_equality(name, args):
    """Descriptions are annotation, not identity — round-trips must not break."""
    cls = getattr(tengri, name)
    assert cls(*args, description="a label", units="erg/s") == cls(*args)


#: For each distribution, keyword perturbations that must change its identity.
#: Every entry is a *different* prior, so ``==`` against the base must be False.
PERTURBATIONS: dict[str, list[dict]] = {
    "Uniform": [{"lo": -1.0}, {"hi": 2.0}],
    "Gaussian": [{"mu": 1.0}, {"sigma": 2.0}, {"lo": -1.0}, {"hi": 1.0}],
    "Fixed": [{"value": 0.75}],
    "LogUniform": [{"lo": 1e43}, {"hi": 1e48}],
    "LogNormal": [{"mu": 1.0}, {"sigma": 2.0}, {"lo": 1e-3}, {"hi": 10.0}],
    "StudentT": [
        {"mu": 1.0},
        {"sigma": 2.0},
        {"df": 30.0},
        {"lo": -5.0},
        {"hi": 5.0},
    ],
    "Laplace": [{"mu": 1.0}, {"b": 2.0}, {"lo": -5.0}, {"hi": 5.0}],
}


@pytest.mark.parametrize("name", sorted(PERTURBATIONS))
def test_equality_distinguishes_every_constructor_argument(name):
    """Two priors that differ in any argument must not compare equal (#1292).

    ``LogNormal.__eq__`` compared only ``(mu, sigma)``, so
    ``LogNormal(0, 1, hi=10) == LogNormal(0, 1, hi=1e9)`` was True — a 1e8x
    difference in truncation reported as "same prior". ``StudentT`` had no
    ``__eq__`` at all and fell back to identity.

    Prior equality is what the builder-vs-dict and ``to_groups`` round-trip
    contracts use to prove two construction paths agree, so a blind spot here
    does not merely give a wrong answer — it makes those contracts vacuous.
    """
    cls = getattr(tengri, name)
    kwargs = _base_kwargs(name)
    base = cls(**kwargs)

    assert base == cls(**kwargs), f"{name} is not equal to an identical instance"

    for delta in PERTURBATIONS[name]:
        other = cls(**{**kwargs, **delta})
        field = next(iter(delta))
        assert base != other, (
            f"{name}.__eq__ ignores {field!r}: {base!r} == {other!r}. "
            "Every constructor argument is part of the prior's identity."
        )


def _base_kwargs(name: str) -> dict:
    """Explicit, fully-specified constructor kwargs — perturbable by key."""
    return {
        "Uniform": {"lo": 0.0, "hi": 1.0},
        "Gaussian": {"mu": 0.0, "sigma": 1.0, "lo": -10.0, "hi": 10.0},
        "Fixed": {"value": 0.5},
        "LogUniform": {"lo": 1e42, "hi": 1e47},
        "LogNormal": {"mu": 0.0, "sigma": 1.0, "lo": 1e-6, "hi": 1e6},
        "StudentT": {"mu": 0.0, "sigma": 1.0, "df": 3.0, "lo": -10.0, "hi": 10.0},
        "Laplace": {"mu": 0.0, "b": 1.0, "lo": -10.0, "hi": 10.0},
    }[name]


def test_the_perturbation_table_covers_every_argument():
    """Guard the guard: a missing key would silently stop testing a field."""
    import inspect

    for name, deltas in PERTURBATIONS.items():
        cls = getattr(tengri, name)
        ctor = {
            p
            for p in inspect.signature(cls).parameters
            if p not in ("description", "units", "default")
        }
        covered = {k for d in deltas for k in d}
        assert covered == ctor, (
            f"{name}: perturbation table covers {sorted(covered)} but the "
            f"constructor takes {sorted(ctor)}. An uncovered argument is an "
            "argument __eq__ is free to ignore."
        )


def test_the_base_class_supplies_the_defaults():
    """Guard the guard: a new subclass inherits the contract for free.

    If someone re-adds the attributes per-subclass instead of on the base,
    this test still passes — but a *new* distribution would silently lose
    them again. Asserting on ``Distribution`` itself is what makes the fix
    structural.
    """
    assert Distribution.description == ""
    assert Distribution.units == ""

    class _Novel(Distribution):
        """A distribution that never thinks about metadata."""

    assert _Novel().description == ""
    assert _Novel().units == ""


def test_the_parametrization_covers_the_public_surface():
    """A shrinking ALL_PRIORS would make every test above vacuous."""
    exported = {
        n
        for n in tengri.__all__
        if isinstance(getattr(tengri, n, None), type)
        and issubclass(getattr(tengri, n), Distribution)
    }
    missing = exported - set(ALL_PRIORS)
    assert not missing, (
        f"public distributions not covered by this contract: {sorted(missing)}. "
        "Add them to ALL_PRIORS with valid positional args."
    )

# SPDX-License-Identifier: BSD-3-Clause
"""Asking for a property your model lacks must not be reported as a typo.

``list_properties()`` advertises all 50 registered properties regardless of
what any one model contains. Seven of them — the four ``radio`` and three
``xray`` ones — are not in a stellar+dust model's catalog, and asking for one
raised::

    KeyError: "Unknown property 'l_x_total'. Available: [...]"

``l_x_total`` is not unknown. The reader copied it off the menu; what they
needed was an ``xray`` group in their build. The message sent them hunting for
a misspelling instead, and buried the real answer in a 43-name list.

Both cases are real and they want opposite responses, so the message has to
tell them apart:

* not registered anywhere -> a typo; suggest near misses.
* registered, absent here -> name the component that declares it, and how to
  add it.

Every consumer is covered, because a fix applied to one accessor leaves the
others contradicting it: ``pred.properties[...]``, ``predict_properties``,
``Posterior.properties[...]`` and ``CatalogPosterior`` all route through the
one message.
"""

from __future__ import annotations

import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel
from tengri.forward.properties import PROPERTY_REGISTRY, missing_property_message

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


@pytest.fixture(scope="module")
def stellar_only_model(ssp_data_fsps):
    """Stellar + dust: no radio component, no X-ray component."""
    obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i"]))
    return SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust={
            "law_diff": "calzetti",
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
        },
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


def _absent(model) -> list[str]:
    """Registered properties this model cannot compute — the census."""
    return sorted(set(PROPERTY_REGISTRY) - set(model.available_properties))


class TestTheCensus:
    def test_some_registered_property_is_absent_from_this_model(self, stellar_only_model):
        """Without an absent property the rest of this file tests nothing."""
        absent = _absent(stellar_only_model)
        assert absent, (
            "every registered property is available on a stellar+dust model, so "
            "the absent-component branch is unreachable and untested. Either the "
            "fixture grew components or the registry shrank."
        )

    def test_the_absent_ones_are_still_advertised(self, stellar_only_model):
        """The trap only exists because the menu offers what the model lacks."""
        import tengri

        advertised = {row["name"] for row in tengri.list_properties()}
        absent = set(_absent(stellar_only_model))
        assert absent <= advertised, (
            f"{sorted(absent - advertised)} are absent from the model and also "
            f"not advertised, so no reader could have asked for them."
        )


class TestTheMessage:
    @pytest.mark.parametrize("consumer", ["prediction", "predict_properties"])
    def test_an_absent_property_is_not_called_unknown(self, stellar_only_model, consumer):
        absent = _absent(stellar_only_model)[0]
        params = dict(stellar_only_model.spec.get_fixed_values())
        with pytest.raises(KeyError) as excinfo:
            if consumer == "prediction":
                stellar_only_model.predict(params).properties[absent]
            else:
                stellar_only_model.predict_properties(params, names=(absent,))
        message = str(excinfo.value)
        assert "Unknown property" not in message, (
            f"{absent!r} is a registered property that this model cannot "
            f"compute, but the error calls it unknown: {message[:200]}"
        )

    @pytest.mark.parametrize("name", sorted(PROPERTY_REGISTRY))
    def test_the_message_names_the_component_and_how_to_add_it(self, name):
        component = PROPERTY_REGISTRY[name][0].component_name
        message = missing_property_message(name, available={"stellar_mass"})
        assert component in message, (
            f"the message for {name!r} never names {component!r}, the component "
            f"the reader has to add: {message[:200]}"
        )
        assert "describe_property" in message

    def test_a_misspelling_is_still_reported_as_unknown_with_near_misses(self):
        message = missing_property_message("steller_mass", available={"stellar_mass", "sfr_10myr"})
        assert "Unknown property" in message
        assert "stellar_mass" in message, f"a one-character typo got no suggestion: {message}"

    def test_a_name_nothing_resembles_gets_no_invented_suggestion(self):
        message = missing_property_message("qqqqqqqq", available={"stellar_mass"})
        assert "Unknown property" in message
        assert "Did you mean" not in message

    def test_several_bad_names_list_what_is_available_once(self):
        """Repeating 43 names per bad name turned two mistakes into a wall."""
        available = {f"prop_{i}" for i in range(43)}
        message = missing_property_message("l_x_total", "qqqqqqqq", available=available)
        assert message.count("Available on this model:") == 1, (
            f"the available list is repeated per bad name:\n{message}"
        )
        assert "l_x_total" in message and "qqqqqqqq" in message, (
            "one of the two bad names was dropped from the message"
        )
        # Each name gets its own line, so the two diagnoses stay legible.
        assert message.count("\n") >= 2, message


class TestTheHintOnlyNamesThingsThatExist:
    """Advice that does not resolve is worse than no advice."""

    def test_no_hint_when_the_component_is_not_a_grammar_group(self):
        """`nebular` declares the line properties; the grammar group is `neb`.

        Deriving the snippet from the component name would advertise a
        ``nebular={...}`` group that ``SEDModel.build`` rejects.
        """
        from tengri.forward.properties import _grammar_hint
        from tengri.parameters.groups import _GROUP_STRUCTURAL_KEYS

        assert "nebular" not in _GROUP_STRUCTURAL_KEYS, (
            "'nebular' became a grammar group — re-check this test's premise."
        )
        assert _grammar_hint("nebular") == ""

    @pytest.mark.parametrize(
        "component", sorted({e.component_name for es in PROPERTY_REGISTRY.values() for e in es})
    )
    def test_every_hint_names_only_a_real_group_and_a_real_verb(self, component):
        import tengri
        from tengri.forward.properties import _grammar_hint
        from tengri.parameters.groups import _GROUP_STRUCTURAL_KEYS

        hint = _grammar_hint(component)
        if not hint:
            return
        assert component in _GROUP_STRUCTURAL_KEYS, (
            f"the hint offers a {component!r} group that the grammar does not accept: {hint}"
        )
        if "tengri." in hint:
            verb = hint.split("tengri.")[1].split("(")[0]
            assert hasattr(tengri, verb), (
                f"the hint tells the reader to call tengri.{verb}(), which is not exported: {hint}"
            )
            getattr(tengri, verb)()  # named and callable, not merely present

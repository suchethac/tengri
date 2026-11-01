# SPDX-License-Identifier: BSD-3-Clause
"""Posterior properties share a bundle without assuming the model is an ``SEDModel`` (#3076).

The composable AGN's energy ledger is one bundle of properties, so ``PosteriorProperties``
evaluates every key of it in one call. The first version read ``model._ensure_property_catalog``
directly; a catalog posterior carries a ``ForwardModel``, which has no such method, and every
property access raised ``AttributeError`` (the catalog summary block then dropped its derived
properties). A model without an inspectable catalog evaluates each name on its own.
"""

from __future__ import annotations

import pytest

from tengri.forward.properties import Property, PropertyEntry
from tengri.inference.posterior import _bundle_siblings

pytestmark = pytest.mark.regression_bug


class _NoCatalog:
    """A model with properties but no ``_ensure_property_catalog`` (a ``ForwardModel``)."""

    available_properties = ("stellar_mass",)


def _bundle(state, params, model, template_data):
    return {}


def _entry(name, bundle=None, fn=None):
    return PropertyEntry(
        name=name, units="", group="g", doc="", component_name="agn", fn=fn, bundle=bundle
    )


class _WithCatalog:
    def __init__(self):
        self._catalog = {
            "stellar_mass": _entry("stellar_mass", fn=lambda s, p: 0.0),
            "log_L_agn_disc": _entry("log_L_agn_disc", bundle=_bundle),
            "log_L_agn_los": _entry("log_L_agn_los", bundle=_bundle),
        }

    def _ensure_property_catalog(self):
        return self._catalog


def test_a_model_without_a_catalog_evaluates_the_name_alone():
    assert _bundle_siblings(_NoCatalog(), "stellar_mass") == ("stellar_mass",)


def test_bundle_keys_are_evaluated_together_and_plain_keys_alone():
    model = _WithCatalog()
    assert _bundle_siblings(model, "stellar_mass") == ("stellar_mass",)
    assert set(_bundle_siblings(model, "log_L_agn_disc")) == {"log_L_agn_disc", "log_L_agn_los"}
    assert _bundle_siblings(model, "log_L_agn_disc", exclude={"log_L_agn_los"}) == (
        "log_L_agn_disc",
    )
    assert Property  # the public declaration type the bundle field belongs to

# SPDX-License-Identifier: BSD-3-Clause
"""Every template dust adapter axis names a parameter its component declares.

``AXIS_PARAMS`` in ``dust_emission_precompute`` is matched against the component's full
parameter names (``parameter_prefix`` + the declared attribute). A label that names no declared
parameter cannot be collapsed or looked up, and the failure is silent: the axis is never
selected. ``dale2014`` was labeled ``dust_alpha`` while its parameter is ``dust_alpha_dale``.
"""

from __future__ import annotations

import pytest

from tengri.components.dust.dust_emission_precompute import AXIS_PARAMS
from tengri.components.sed_model_component import _REGISTRY
from tengri.parameters.priors import Distribution

pytestmark = pytest.mark.contract

#: Adapter key -> the registry name of the component it tabulates.
_COMPONENT_OF = {
    "dl07": "draine_li2007",
    "draine_li2007": "draine_li2007",
    "dale2014": "dale2014",
    "draine_li2014": "draine_li2014",
    "astrodust": "astrodust",
    "themis": "themis",
    "bosa": "bosa",
    "draine2021_pah": "draine2021_pah_ir",
}


def _declared_names(cls) -> set[str]:
    prefix = getattr(cls, "parameter_prefix", "dust_")
    names: set[str] = set()
    for klass in cls.__mro__:
        for attr, value in vars(klass).items():
            if isinstance(value, Distribution):
                names.add(prefix + attr)
    return names


@pytest.mark.parametrize("key", sorted(AXIS_PARAMS))
def test_axis_names_are_declared_by_the_component(key):
    cls = _REGISTRY[_COMPONENT_OF[key]]
    declared = _declared_names(cls)
    undeclared = [name for name in AXIS_PARAMS[key] if name not in declared]
    assert not undeclared, f"{key} axes {undeclared} are not declared by {cls.__name__}"

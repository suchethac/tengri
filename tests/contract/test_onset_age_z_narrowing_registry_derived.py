# SPDX-License-Identifier: BSD-3-Clause
"""Every timing fraction in the registry carries the fraction contract (#2748).

The pairs are derived from ``ParamDef.gyr_twin``, so a new SFH family cannot
drop out of the check by omission. For every pair the Gyr name is a ``Fixed``
shim (never freed by the wildcard) and the fraction carries a Uniform prior
bounded by 1, so ``frac * age(z)`` cannot exceed the age of the universe.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from tengri.components.stellar.sfh.registry import SFH_REGISTRY
from tengri.parameters.priors import Fixed, Uniform
from tengri.parameters.timing import timing_frac_params

_PAIRS = sorted(timing_frac_params().items())


def _owner(frac_name):
    for spec in SFH_REGISTRY.values():
        if frac_name in spec.params:
            return spec
    raise AssertionError(f"{frac_name!r} is in no SFH family")


def test_there_are_23_fraction_gyr_pairs():
    assert len(_PAIRS) == 23


@pytest.mark.parametrize(("frac_name", "gyr_name"), _PAIRS, ids=[f for f, _ in _PAIRS])
def test_gyr_twin_is_a_fixed_shim(frac_name, gyr_name):
    spec = _owner(frac_name)
    assert gyr_name in spec.params, f"{gyr_name!r} missing beside {frac_name!r}"
    assert isinstance(spec.params[gyr_name].default, Fixed)


@pytest.mark.parametrize(("frac_name", "gyr_name"), _PAIRS, ids=[f for f, _ in _PAIRS])
def test_fraction_prior_is_bounded_by_the_age_of_the_universe(frac_name, gyr_name):
    default = _owner(frac_name).params[frac_name].default
    assert isinstance(default, Uniform)
    assert 0.0 < default.lo < default.hi <= 1.0 + 1e-12

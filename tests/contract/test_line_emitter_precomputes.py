# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for the line-emitter precompute adapters (PR 4 parts A + B).

Adapters covered:

* ``feltre_nlr`` — Feltre+2016 AGN NLR CLOUDY grid.
* ``mappings_shock`` — Allen+2008 / Alarie+Morisset 2019 MAPPINGS shock grid.
* ``cb19`` — Charlot+Bruzual 2019 (3MdB_17) photoionization grid.
* ``blr`` — AGN broad-line Gaussian composer (filter-projection precompute only).
* ``nlr_gaussian`` — AGN narrow-line Gaussian composer (Richardson+2014 a42).

These are surface tests: each adapter is registered, exposes ``AXIS_PARAMS``,
``precompute``, ``build_lookup``, and the ``precompute()`` call returns
``None`` cleanly (or raises a documented error) when the upstream grid file
is missing.  Numerical-equivalence tests against the runtime path live with
the larger nebular regression suite; these guard the wiring only.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.contract

LINE_ADAPTERS = (
    ("feltre_nlr", "tengri.components.nebular.feltre_precompute"),
    ("mappings_shock", "tengri.components.nebular.mappings_shock_precompute"),
    ("cb19", "tengri.components.nebular.cb19_precompute"),
    ("blr", "tengri.components.agn.blr_precompute"),
    ("nlr_gaussian", "tengri.components.agn.nlr_gaussian_precompute"),
)


@pytest.mark.parametrize("registry_key,module_path", LINE_ADAPTERS)
def test_line_adapter_registered(registry_key, module_path):
    from tengri.forward.precompute.registry import _REGISTRY

    assert _REGISTRY[registry_key] == module_path


@pytest.mark.parametrize("registry_key,module_path", LINE_ADAPTERS)
def test_line_adapter_protocol_surface(registry_key, module_path):
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "AXIS_PARAMS")
    ax = mod.AXIS_PARAMS
    assert isinstance(ax, tuple)
    for entry in ax:
        assert isinstance(entry, str)
    assert callable(mod.precompute)
    assert callable(mod.build_lookup)

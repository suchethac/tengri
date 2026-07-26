# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the taught catalog surface must be ``Catalog``, not ``CatalogFitter`` (#1369).

#1317 shipped ``Catalog`` as the one catalog noun and promised ``CatalogFitter``
becomes a *deprecated alias* — but the shipped name was still bound straight to the
original engine class: no ``DeprecationWarning``, and ``tutorial("use_cases")`` kept
teaching it, with a construction call that does not even match the constructor
(``CatalogFitter(model, data_array, noise_array)`` passes an array as ``data_type``)
and a "vmap'd — one compile" claim that is false for ``.run("map")``. The same
tutorial still taught ``tengri.Fitter`` in three other recipes, which the spec
retires from taught surfaces (spec #1320, decision 9).

Guards here:

* constructing ``tengri.CatalogFitter`` warns, and the warning names the
  replacement noun;
* the internal engine (``_CatalogFitterOriginal`` — what ``Catalog`` itself
  constructs) stays warning-free, so ``Catalog.fit`` does not warn at itself;
* the tutorial teaches only current surface: no ``CatalogFitter``, no bare
  ``tengri.Fitter(`` constructions, and its catalog recipe's calls **bind** against
  the real ``Catalog`` signatures (advice must run — the #1364 rule).
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re
import warnings

import pytest

pytestmark = pytest.mark.regression_bug


def _tutorial_text(name: str) -> str:
    import tengri

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tengri.tutorial(name)
    return buf.getvalue()


def test_catalog_fitter_construction_warns_and_names_catalog():
    import tengri

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cf = tengri.CatalogFitter(object(), [], data_type="photometry")
    dep = [w for w in rec if issubclass(w.category, DeprecationWarning)]
    assert dep, "CatalogFitter must emit a DeprecationWarning (#1317's promised alias)"
    msg = str(dep[0].message)
    assert "Catalog" in msg, f"the warning must name the replacement noun: {msg}"
    # The old surface keeps working — deprecation, not removal (spec §13).
    assert hasattr(cf, "run")
    assert cf.n_galaxies == 0


def test_internal_engine_does_not_warn():
    """``Catalog`` constructs the engine internally — that path must stay quiet."""
    from tengri.inference.catalog_fitter import _CatalogFitterOriginal

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _CatalogFitterOriginal(object(), [], data_type="photometry")
    dep = [w for w in rec if issubclass(w.category, DeprecationWarning)]
    assert not dep, "the internal engine must not warn — Catalog.fit would warn at itself"


def test_deprecated_alias_is_the_engine():
    """isinstance checks and the ``run`` surface survive the shim."""
    import tengri
    from tengri.inference.catalog_fitter import _CatalogFitterOriginal

    assert issubclass(tengri.CatalogFitter, _CatalogFitterOriginal)


def test_use_cases_tutorial_teaches_current_surface():
    text = _tutorial_text("use_cases")

    assert "CatalogFitter" not in text, "tutorial still teaches the retired noun"
    assert "tengri.Fitter(" not in text, "Fitter is internal-only (spec decision 9)"
    assert "Catalog(" in text, "the catalog recipe should teach the Catalog noun"


def test_use_cases_catalog_advice_binds():
    """Every taught catalog call must bind against the real signature (#1364 rule)."""
    from tengri.inference.catalog import Catalog

    text = _tutorial_text("use_cases")
    assert "Use case 3" in text and "Use case 4" in text
    block = text.split("Use case 3")[1].split("Use case 4")[0]

    ctor = re.search(r"Catalog\(([^)]*)\)", block)
    assert ctor, "no Catalog(...) construction taught in the catalog recipe"
    ctor_kwargs = re.findall(r"(\w+)\s*=", ctor.group(1))
    params = inspect.signature(Catalog.__init__).parameters
    for k in ctor_kwargs:
        assert k in params, f"taught Catalog kwarg {k!r} does not exist"
    assert "flux_unit" in ctor_kwargs, "flux_unit is required — teaching without it raises"

    fit_call = re.search(r"\.fit\(([^)]*)\)", block)
    assert fit_call, "no .fit(...) call taught in the catalog recipe"
    fit_kwargs = re.findall(r"(\w+)\s*=", fit_call.group(1))
    fit_params = inspect.signature(Catalog.fit).parameters
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in fit_params.values())
    for k in fit_kwargs:
        assert k in fit_params or has_var_kw, f"taught .fit kwarg {k!r} not on Catalog.fit"
    assert "key" in fit_kwargs, "Catalog.fit requires key= — teaching it without one raises"

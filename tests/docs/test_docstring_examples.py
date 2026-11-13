"""Docstring examples in touched modules import names that exist (#3452).

Also guards the removed-API stand-in: doctest collection of the package must
not raise, and probing a stand-in for ``__wrapped__`` must raise AttributeError
(not ImportError), since ``inspect`` and ``doctest`` probe dunders.
"""

from __future__ import annotations

import ast
import doctest
import importlib

import pytest

import tengri

TOUCHED_MODULES = (
    "tengri.analysis.mock",
    "tengri.profiling",
    "tengri.cosmology",
    "tengri.units",
    "tengri.observation.noise",
    "tengri.observation.observables",
    "tengri.config.settings",
    "tengri._data_setup",
    "tengri.components.agn.blocks._protocol",
    "tengri.inference.catalog",
    "tengri.citations.bibliography",
    "tengri.parameters.parameters",
    "tengri.components.stellar.sps.dsps_wrapper",
)


def _resolves(module_name: str, name: str) -> bool:
    module = importlib.import_module(module_name)
    if hasattr(module, name):
        return True
    try:
        importlib.import_module(f"{module_name}.{name}")
    except ImportError:
        return False
    return True


def _broken_tengri_imports(module_name: str) -> list[str]:
    broken = []
    module = importlib.import_module(module_name)
    for test in doctest.DocTestFinder().find(module):
        for example in test.examples:
            try:
                tree = ast.parse(example.source)
            except SyntaxError:
                continue  # fragments are checked by the doctest runner, not here
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if node.module.split(".")[0] != "tengri":
                    continue
                for alias in node.names:
                    if not _resolves(node.module, alias.name):
                        broken.append(f"{test.name}: from {node.module} import {alias.name}")
    return broken


@pytest.mark.parametrize("module_name", TOUCHED_MODULES)
def test_docstring_tengri_imports_resolve(module_name):
    assert _broken_tengri_imports(module_name) == []


@pytest.mark.parametrize("module_name", TOUCHED_MODULES)
def test_docstring_examples_pass(module_name):
    module = importlib.import_module(module_name)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    for test in doctest.DocTestFinder().find(module):
        runner.run(test, out=lambda _text: None)
    assert runner.failures == 0, f"{runner.failures} failing docstring example(s) in {module_name}"


def test_package_doctest_collection_does_not_raise():
    doctest.DocTestFinder().find(tengri)


def test_removed_api_standin_probe_for_wrapped_is_attribute_error():
    with pytest.raises(AttributeError):
        _ = tengri.KernelStrategy.__wrapped__

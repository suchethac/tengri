# SPDX-License-Identifier: BSD-3-Clause
"""#1394 — the catalog engine must not default to a ``tier="broken"`` backend.

``_CatalogFitterOriginal.run`` shipped with ``method="native_vi_linear"``,
which is registered ``tier="broken"`` (segfaults on DPL/dense_basis photometry
mocks, #231). The two ``NotImplementedError`` branches in ``run`` already told
callers to use ``mcmc_nuts`` instead, so the default contradicted the class's
own error messages.

These are contract assertions on the *declared* default and on the advice
strings — deliberately not a fit. Running the broken backend to prove it is
broken would segfault the worker, and running NUTS to prove it works would put
a multi-GB warmup in the regression shard, which is the shard least able to
absorb it (#1346).
"""

from __future__ import annotations

import inspect

import pytest

from tengri.inference._backend_registry import _BACKENDS
from tengri.inference.catalog_fitter import _CatalogFitterOriginal

pytestmark = pytest.mark.regression_bug


def _tier(name):
    entry = _BACKENDS[name]
    return getattr(entry, "tier", None) or (entry.get("tier") if isinstance(entry, dict) else None)


def test_catalog_run_default_is_not_a_broken_backend():
    """The declared default must be a runnable tier, whatever its name."""
    default = inspect.signature(_CatalogFitterOriginal.run).parameters["method"].default
    assert default in _BACKENDS, f"default method {default!r} is not a registered backend"
    assert _tier(default) != "broken", (
        f"catalog run() defaults to {default!r}, which is registered "
        f'tier="broken" — it cannot be run as documented (#1394)'
    )


def test_catalog_run_default_is_nuts():
    """Pin the specific choice, so a silent revert is a test failure."""
    default = inspect.signature(_CatalogFitterOriginal.run).parameters["method"].default
    assert default == "mcmc_nuts"


def test_the_default_supports_the_features_the_native_path_rejects():
    """NUTS must be on the chunkable path, or the default loses functionality.

    ``forward_chunk_size`` / ``n_pad`` / ``devices`` are routed by membership in
    these frozensets; a default outside both would silently fall through to the
    sequential path and warn on every one of those kwargs.
    """
    default = inspect.signature(_CatalogFitterOriginal.run).parameters["method"].default
    chunkable = _CatalogFitterOriginal._MCMC_VMAPPABLE | _CatalogFitterOriginal._NATIVE_VMAPPABLE
    assert default in chunkable
    # devices= is honoured for the MCMC set only.
    assert default in _CatalogFitterOriginal._MCMC_VMAPPABLE


def test_no_docstring_in_the_module_teaches_a_broken_backend_as_a_call():
    """Examples must not show ``run("<broken>")`` — that is what #1394 fixed."""
    import tengri.inference.catalog_fitter as mod

    broken = {n for n in _BACKENDS if _tier(n) == "broken"}
    offenders = []
    for name in dir(mod):
        obj = getattr(mod, name)
        for doc in (
            (inspect.getdoc(obj) or "",)
            if not inspect.isclass(obj)
            else (
                inspect.getdoc(obj) or "",
                *(inspect.getdoc(getattr(obj, m, None)) or "" for m in dir(obj)),
            )
        ):
            for b in broken:
                if f'run("{b}"' in doc or f"run('{b}'" in doc:
                    offenders.append((name, b))
    assert not offenders, f"docstrings teach a tier='broken' backend as a call: {offenders}"


def test_chunk_size_warning_names_every_chunkable_method():
    """The advice string is derived, not hand-written (#1394 secondary).

    The literal it replaced said "only native_vi_linear and native_vi_nonlinear"
    long after ``_MCMC_VMAPPABLE`` gave NUTS/HMC the same capability, steering
    callers off the working path onto the broken one.
    """
    src = inspect.getsource(_CatalogFitterOriginal.run)
    assert "_MCMC_VMAPPABLE | self._NATIVE_VMAPPABLE" in src, (
        "the forward_chunk_size warning must derive its method list from the "
        "dispatch sets so it cannot go stale again"
    )
    assert "Only native_vi_linear and" not in src

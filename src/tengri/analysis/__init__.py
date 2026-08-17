# SPDX-License-Identifier: BSD-3-Clause
"""Analysis layer: diagnostics, plotting, simulation utilities.

``tengri.analysis.plotting`` resolves on first attribute access rather than at
import time. The attribute used to exist only as a side effect: ``tengri``'s own
``__init__`` did ``from tengri.analysis.plotting import (...)``, which binds the
submodule on its parent package as a by-product of importing it. Making that
import lazy — so ``import tengri`` no longer drags in matplotlib — removed the
side effect along with the cost, and ``tengri.analysis.plotting.setup_style()``
began raising ``AttributeError``.

That idiom is not incidental: ``examples/_STYLE.md`` prescribes it and 280 of
the gallery scripts use it. The break was also **order-dependent**, which is
why CI did not catch it — the gallery runner executes every example in one
process, so any earlier example that touched ``tengri.plot_sed_fit`` imported the
submodule and bound the attribute for the rest of the run. Running the affected
example alone is what surfaced it.

Resolving it here keeps both properties: the attribute works after a bare
``import tengri``, and nothing imports matplotlib until something asks for it.
"""

from __future__ import annotations

import importlib
from types import ModuleType

#: Submodules reachable as attributes of this package without importing them
#: first. Only ``plotting`` was ever bound this way, so only it is restored;
#: the rest of ``analysis/`` is reached by explicit import, as before.
_LAZY_SUBMODULES = frozenset({"plotting"})


def __getattr__(name: str) -> ModuleType:
    """Import a lazily-bound submodule on first access, then cache it."""
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# No ``__dir__`` here, deliberately. #1431 consolidated fourteen hand-rolled
# copies onto ``tengri._completion.curated_dir`` and guards the result, so a
# new one would have to curate through that helper and join the census in
# ``tests/contract/test_curated_dir_mechanism.py``. Curating this namespace
# would also be wrong: ``_CURATED_DIR`` has to equal ``dir()`` exactly, so a
# tuple naming ``plotting`` would *hide* the six real siblings beside it
# (diagnostics, mock, simulate, sbc, feature_strengths, population_mocks).
# Restoring the attribute never needed a completion menu — ``plotting`` simply
# appears in ``dir()`` once something has touched it, as for any lazy submodule.

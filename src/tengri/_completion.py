# SPDX-License-Identifier: BSD-3-Clause
"""One implementation of the curated module ``__dir__``.

``__all__`` governs ``from ... import *`` but not :func:`dir`, so a namespace
that does nothing shows its own imports -- ``Any``, ``Callable``, the
``__future__`` ``annotations`` object, internal helpers -- as completions
beside the physics (#1288). Fourteen modules each answered that with their own
nine-line ``def __dir__`` (#1431).

They had already drifted into two spellings, ``sorted(__all__)`` and
``list(__all__)``, which no caller can tell apart: :func:`dir` sorts whatever
``__dir__`` returns. The divergence was drift rather than design, so this
module keeps the sorted spelling and both groups collapse onto it.

``tests/contract/test_curated_dir_mechanism.py`` fails on any new
module-level ``def __dir__`` under ``src/tengri``, so the boilerplate cannot
grow back.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

__all__ = ["curated_dir"]


def curated_dir(names: Sequence[str]) -> Callable[[], list[str]]:
    """Build a module-level ``__dir__`` offering exactly ``names``.

    Parameters
    ----------
    names: sequence of str
        The names to offer. Usually the module's own ``__all__``; two modules
        pass a wider ``_CURATED_DIR`` tuple because they re-export more than
        they advertise for ``import *``. Held by reference, so a module that
        builds its ``__all__`` in place stays consistent.

    Returns
    -------
    callable
        A zero-argument ``__dir__``, ready to bind at module level::

            __all__ = ["alpha", "beta"]
            __dir__ = curated_dir(__all__)

    Notes
    -----
    Filtering only -- every name stays reachable by attribute access. This
    trims the completion surface, it does not make anything private.
    """

    def __dir__() -> list[str]:
        return sorted(names)

    return __dir__

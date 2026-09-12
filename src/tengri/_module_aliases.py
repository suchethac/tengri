# SPDX-License-Identifier: BSD-3-Clause
"""Import finder for aliased submodules — resolves alias spellings to canonical modules.

Fixes #2256: sys.modules aliases for top-level package shortcuts (e.g.,
sys.modules["tengri.sps"] = the real tengri.components.stellar.sps package) only
alias the TOP-LEVEL name. Submodule imports through the alias (e.g.,
`from tengri.sps.dsps_wrapper import X`) create a NEW module via re-execution,
leaving two module objects for one file in sys.modules with independent
module-level state.

Solution: custom MetaPathFinder that intercepts aliased submodule imports and
routes them through the canonical name, binding the EXISTING canonical module
object under the aliased name.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from typing import Any


class _AliasFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder that resolves aliased submodule imports to canonical modules.

    For any name matching an alias prefix (e.g., "tengri.sps" or
    "tengri.sps.dsps_wrapper"), computes the canonical equivalent and imports it
    via importlib.import_module. Returns a spec that binds the canonical module
    object under the aliased name, avoiding re-execution.

    Parameters
    ----------
    alias_map : dict[str, str]
        Mapping from alias name (e.g., "tengri.sps") to canonical name
        (e.g., "tengri.components.stellar.sps").

    Attributes
    ----------
    alias_map : dict[str, str]
        Frozen at construction time.
    """

    def __init__(self, alias_map: dict[str, str]) -> None:
        """Initialize the finder with an alias map.

        Parameters
        ----------
        alias_map : dict[str, str]
            Mapping from alias name to canonical name. Every canonical value
            must not start with any alias key (checked at installation).
        """
        self.alias_map = alias_map

    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: Any = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Resolve spec for aliased or canonical module names.

        If fullname exactly matches an alias OR starts with an alias + ".",
        compute the canonical equivalent, import it, and return a spec that
        binds the canonical module object under the aliased name.

        Parameters
        ----------
        fullname : str
            The full import path (e.g., "tengri.sps" or "tengri.sps.dsps_wrapper").
        path
            Ignored; only fullname is inspected.
        target
            Ignored.

        Returns
        -------
        importlib.machinery.ModuleSpec | None
            A spec that loads the canonical module, or None if fullname does
            not match an alias.
        """
        # Check if fullname is an exact alias or a submodule under an alias
        for alias, canonical in self.alias_map.items():
            if fullname == alias or fullname.startswith(alias + "."):
                # Compute canonical name by replacing the alias prefix
                canonical_name = canonical + fullname[len(alias) :]
                # Import the canonical module (may recursively call this finder,
                # but canonical names never match an alias prefix by design)
                importlib.import_module(canonical_name)
                # Return a spec that yields the already-imported canonical module
                # from sys.modules, avoiding re-execution
                loader = _AliasLoader(canonical_name)
                return importlib.util.spec_from_loader(fullname, loader)
        return None


class _AliasLoader(importlib.abc.Loader):
    """Loader that binds an already-imported canonical module under an alias.

    Creates no module; returns the canonical module from sys.modules.

    Parameters
    ----------
    canonical_name : str
        The fully qualified name of the canonical module in sys.modules.
    """

    def __init__(self, canonical_name: str) -> None:
        """Initialize with the canonical module name.

        Parameters
        ----------
        canonical_name : str
            Name of the canonical module to retrieve from sys.modules.
        """
        self.canonical_name = canonical_name

    def create_module(self, spec: Any) -> Any:
        """Return the canonical module without creating a new one.

        Returns
        -------
        module object
            The canonical module retrieved from sys.modules.
        """
        return sys.modules[self.canonical_name]

    def exec_module(self, module: Any) -> None:
        """No-op: canonical module is already fully initialized.

        Parameters
        ----------
        module
            Ignored; the returned module is already in its final state.
        """
        pass


def install_alias_finder(alias_map: dict[str, str]) -> None:
    """Install the alias finder into sys.meta_path.

    Inserts a _AliasFinder instance at position 0 of sys.meta_path if one is
    not already present. Replaces any existing _AliasFinder to handle
    re-initialization (idempotent).

    Verifies that no canonical value is a prefix of any alias key (which would
    create ambiguity and cause infinite recursion).

    Parameters
    ----------
    alias_map : dict[str, str]
        Mapping from alias name (e.g., "tengri.sps") to canonical name
        (e.g., "tengri.components.stellar.sps"). Canonical names must not
        start with any alias key (checked at installation).

    Raises
    ------
    ValueError
        If a canonical value starts with an alias key, indicating a
        circular or ambiguous mapping.
    """
    # Validate: no canonical value can start with any alias key
    for alias in alias_map:
        for canonical in alias_map.values():
            if canonical.startswith(alias):
                raise ValueError(
                    f"Canonical '{canonical}' starts with alias '{alias}': "
                    "would create ambiguous or circular import."
                )

    # Remove any existing _AliasFinder to avoid stacking
    sys.meta_path[:] = [finder for finder in sys.meta_path if not isinstance(finder, _AliasFinder)]

    # Install at position 0 (highest priority)
    sys.meta_path.insert(0, _AliasFinder(alias_map))

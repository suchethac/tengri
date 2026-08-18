# SPDX-License-Identifier: BSD-3-Clause
"""Deprecation helpers for the tengri public API.

Centralizes the shims used to keep old import paths working while the API
is reorganized toward the structure described in
``docs/dev/api_migration_v0.x.md``. Every entry here MUST have a matching
row in that migration document.

The three patterns provided:

- :func:`deprecated_alias`: wraps a callable so calling it emits one
  ``DeprecationWarning`` and then forwards to the new implementation.
- :func:`deprecated_attribute`: for module-level attribute access via
  ``__getattr__`` (PEP 562). Use this when the old name is a class or
  constant rather than a function.
- :func:`resolve_renamed_flag`: for a boolean *keyword argument* that was
  renamed. The other two replace a whole symbol; this one lets a single
  parameter change spelling without breaking the call.

All three helpers are intentionally tiny; deprecation should never become
infrastructure.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = [
    "UNSET",
    "deprecated_alias",
    "deprecated_attribute",
    "resolve_renamed_flag",
]

F = TypeVar("F", bound=Callable[..., Any])


class _Unset:
    """Sentinel for "this keyword was not supplied at all"."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()
"""Sentinel distinguishing "not passed" from "passed a falsy value"."""


def resolve_renamed_flag(
    new_value: Any,
    old_value: Any,
    *,
    old_name: str,
    new_name: str,
    caller: str,
    default: bool = False,
    drop_version: str = "1.0",
) -> bool:
    """Resolve a boolean keyword that was renamed, still honoring the old spelling.

    Parameters
    ----------
    new_value : Any
        Value supplied under the *new* keyword, or its default.
    old_value : Any
        Value supplied under the *old* keyword, or :data:`UNSET` when the
        caller did not pass it.
    old_name, new_name : str
        The deprecated and current keyword names.
    caller : str
        Qualified name of the calling function, used in both messages
        (e.g. ``"Prediction.photometry"``).
    default : bool, optional
        The new keyword's default. Used only to tell an explicit value
        apart from an unspecified one when checking for a contradiction.
        Default False.
    drop_version : str, optional
        Version in which the old spelling stops working. Default ``"1.0"``.

    Returns
    -------
    bool
        The value to act on.

    Raises
    ------
    TypeError
        If both spellings were supplied with contradictory values. Silently
        picking one would make the call mean something the caller did not
        write, which this package treats as a bug rather than a convenience
        (NAMING_CONTRACT §4b.3a).

    Notes
    -----
    **JIT-compatible**: no — argument handling, runs at trace time.

    A contradiction is only detectable when the new keyword was given a
    non-default value; ``f(new=False, old=True)`` is indistinguishable from
    ``f(old=True)`` and resolves to the one affirmative instruction present.
    """
    if old_value is UNSET:
        return bool(new_value)

    warnings.warn(
        f"`{caller}({old_name}=...)` is deprecated and will be removed in "
        f"tengri v{drop_version}; use `{new_name}=` instead.",
        DeprecationWarning,
        stacklevel=3,
    )

    if bool(new_value) != bool(default) and bool(new_value) != bool(old_value):
        raise TypeError(
            f"{caller}() got both `{new_name}={new_value!r}` and "
            f"`{old_name}={old_value!r}`, which contradict each other. "
            f"`{old_name}` is the deprecated spelling of `{new_name}` — pass "
            f"only `{new_name}=`."
        )
    return bool(old_value)


def deprecated_alias(
    new: F,
    *,
    old_name: str,
    new_name: str | None = None,
    drop_version: str = "1.0",
) -> F:
    """Wrap *new* so calls under the old name emit a DeprecationWarning.

    Parameters
    ----------
    new : callable
        The replacement implementation. Forwarded to verbatim.
    old_name : str
        The deprecated name (e.g. ``"get_dust_law"``).
    new_name : str, optional
        The replacement name. Defaults to ``new.__name__``.
    drop_version : str, optional
        Version in which the old name will be removed. Default ``"1.0"``.

    Returns
    -------
    callable
        A wrapper with the same signature as *new* that warns on first
        invocation per call site (Python's default DeprecationWarning
        deduplication applies).

    Notes
    -----
    Use at module top level::

        from tengri.components.dust.registry import resolve_dust_law
        from tengri._deprecated import deprecated_alias

        get_dust_law = deprecated_alias(resolve_dust_law, old_name="get_dust_law")
    """
    target_name = new_name or getattr(new, "__name__", "<callable>")

    @functools.wraps(new)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"`{old_name}` is deprecated and will be removed in tengri "
            f"v{drop_version}; use `{target_name}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new(*args, **kwargs)

    _wrapper.__name__ = old_name
    _wrapper.__qualname__ = old_name
    _wrapper.__doc__ = (
        f"Deprecated alias for :func:`{target_name}`. Will be removed in tengri v{drop_version}."
    )
    return _wrapper  # type: ignore[return-value]


def deprecated_attribute(
    value: Any,
    *,
    old_name: str,
    new_name: str,
    drop_version: str = "1.0",
) -> Any:
    """Emit a DeprecationWarning and return *value*.

    Designed for use inside a module-level ``__getattr__`` (PEP 562)::

        def __getattr__(name):
            if name == "OldClass":
                from tengri.subpackage import NewClass

                return deprecated_attribute(
                    NewClass,
                    old_name="tengri.OldClass",
                    new_name="tengri.subpackage.NewClass",
                )
            raise AttributeError(name)

    Parameters
    ----------
    value : object
        The replacement object (class, constant, module, ...).
    old_name : str
        Fully qualified deprecated name.
    new_name : str
        Fully qualified replacement name.
    drop_version : str, optional
        Version in which the old name will be removed. Default ``"1.0"``.
    """
    warnings.warn(
        f"`{old_name}` is deprecated and will be removed in tengri "
        f"v{drop_version}; use `{new_name}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return value

"""Deprecation helpers for the naming contract transition.

Provides reusable decorators and class factories so that deprecated
aliases emit ``DeprecationWarning`` without duplicating boilerplate.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated_alias(canonical_name: str, remove_in: str = "1.0") -> Callable[[F], F]:
    """Decorator: emits DeprecationWarning when the old-name function is called."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{func.__name__} is deprecated, use {canonical_name} instead. "
                f"Will be removed in tengri v{remove_in}.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_class_alias(
    old_name: str,
    new_cls: type,
    remove_in: str = "1.0",
) -> type:
    """Return a proxy class that warns on instantiation but delegates to *new_cls*.

    ``isinstance`` and ``issubclass`` checks against the alias still work.
    """

    class _AliasType(type):
        def __call__(cls, *args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{old_name} is deprecated, use {new_cls.__name__} instead. "
                f"Will be removed in tengri v{remove_in}.",
                DeprecationWarning,
                stacklevel=2,
            )
            return new_cls(*args, **kwargs)

        def __instancecheck__(cls, instance: Any) -> bool:
            return isinstance(instance, new_cls)

        def __subclasscheck__(cls, subclass: type) -> bool:
            return issubclass(subclass, new_cls)

    return _AliasType(old_name, (new_cls,), {})

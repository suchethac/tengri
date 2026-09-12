# SPDX-License-Identifier: BSD-3-Clause
"""Cache key generation from object attributes under an exclusion-based policy.

A cache key is derived from an object's attributes under a policy ledger: each
attribute is classified as content (hash its value), shape (hash only shape/dtype),
or excluded (ignore it). An attribute the policy does not classify is included by
default (fail-safe: over-keying recomputes, under-keying is silently wrong) and
reported so a test can fail on it.

Three cache key modes:
- ``"content"``: hash the attribute's full value (scalars, arrays, nested structures)
- ``"shape"``: hash only shape and dtype (for arrays); ``None`` keys as ``None``
  (an optional array attribute that is legitimately absent, e.g.
  ``Spectroscopy.covariance``); any other non-array raises TypeError
- ``"exclude"``: omit the attribute from the key

Assumptions:
- Memoized arrays (JAX arrays used in caches) are immutable: a JAX array aliased
  into two objects will have the same id() for both, enabling O(1) memo hits via
  id-based weakref dictionaries. NumPy arrays are mutable and are hashed on every call.
"""

import contextlib
import dataclasses
import enum
import functools
import hashlib
import types
import weakref
from collections.abc import Hashable, Mapping

import numpy as np

# Type aliases
PolicyRow = tuple[str, str]
"""(mode, reason) tuple: mode in MODES, reason is a string explaining the choice."""

KeyPolicy = Mapping[str, PolicyRow]
"""Dict from attribute name to PolicyRow."""

MODES = ("content", "shape", "exclude")
"""Allowed policy modes."""

# Module-level memoization for JAX arrays (id → (weakref, key))
_ARRAY_KEY_CACHE = {}


def content(reason: str) -> PolicyRow:
    """Helper to create a content-mode policy row.

    Parameters
    ----------
    reason : str
        Explanation for why this attribute should be included by content.

    Returns
    -------
    PolicyRow
        ``("content", reason)`` tuple.
    """
    return ("content", reason)


def shape(reason: str) -> PolicyRow:
    """Helper to create a shape-mode policy row.

    Parameters
    ----------
    reason : str
        Explanation for why this attribute should be keyed by shape only.

    Returns
    -------
    PolicyRow
        ``("shape", reason)`` tuple.
    """
    return ("shape", reason)


def exclude(reason: str) -> PolicyRow:
    """Helper to create an exclude-mode policy row.

    Parameters
    ----------
    reason : str
        Explanation for why this attribute should be omitted from the key.

    Returns
    -------
    PolicyRow
        ``("exclude", reason)`` tuple.
    """
    return ("exclude", reason)


def stable_digest(data: bytes) -> str:
    """Return a process-stable digest of bytes.

    Uses BLAKE2b (deterministic across Python processes, unlike Python's
    built-in hash which is salted per process).

    Parameters
    ----------
    data : bytes
        The data to digest.

    Returns
    -------
    str
        Hexadecimal digest (32 characters for 16-byte digest).
    """
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def array_key(value) -> tuple:
    """Return a hashable key for an array's content and shape.

    Memoizes JAX arrays (immutable, identified by id()) via weakref to keep
    the cache bounded. NumPy arrays are mutable and are hashed on every call
    without memoization.

    Parameters
    ----------
    value : array_like
        A NumPy, JAX, or other array-like object with shape and dtype.

    Returns
    -------
    tuple
        ``("array", tuple(shape), str(dtype), digest)`` where digest is a
        stable hash of the array's bytes.

    Notes
    -----
    JAX arrays: memo hit is valid only if the weakref is alive and the
    memoized object is identical (``is value``). When a JAX array is garbage-collected,
    its cache entry is dead.

    NumPy arrays: mutable, so hashing on every call. A NumPy array modified
    in place between two calls yields different keys (no stale memo).
    """
    is_jax = False
    try:
        import jax

        # Try to detect if this is a JAX array
        if hasattr(jax, "Array"):
            is_jax = isinstance(value, jax.Array)
        else:
            # Fallback: check module and type name
            is_jax = type(value).__module__.startswith("jax")
    except ImportError:
        pass

    if is_jax:
        # Try to use memo via id() and weakref
        key = id(value)
        hit = _ARRAY_KEY_CACHE.get(key)
        if hit is not None:
            ref, cached_key = hit
            if ref() is value:
                return cached_key

        # Compute and cache
        arr = np.asarray(value)
        digest = stable_digest(np.ascontiguousarray(arr).tobytes())
        result = ("array", tuple(arr.shape), str(arr.dtype), digest)
        # A miss is the cheap moment to drop entries whose array is gone, so
        # the memo stays bounded by the number of LIVE keyed arrays.
        for dead in [k for k, (ref, _) in _ARRAY_KEY_CACHE.items() if ref() is None]:
            del _ARRAY_KEY_CACHE[dead]
        # Some array-like objects don't support weakref
        with contextlib.suppress(TypeError):
            _ARRAY_KEY_CACHE[key] = (weakref.ref(value), result)
        return result
    else:
        # NumPy or other array-like: no memoization
        arr = np.asarray(value)
        digest = stable_digest(np.ascontiguousarray(arr).tobytes())
        return ("array", tuple(arr.shape), str(arr.dtype), digest)


def baked(value) -> Hashable:
    """Return a hashable, equality-stable stand-in for a value.

    Reduces values to immutable, hashable forms suitable for cache keying.
    Delegates to object-specific methods (cache_key, jit_cache_key) when available.

    Parameters
    ----------
    value : object
        Any Python object.

    Returns
    -------
    Hashable
        A hashable, equality-stable representation.

    Raises
    ------
    TypeError
        If the value is a JAX tracer (inside a jit boundary) or an unrecognized
        type with no cache_key, jit_cache_key, or __repr__ method.

    Notes
    -----
    Floats are preserved at full precision (not rounded).

    Containers (list, tuple, set, dict, MappingProxyType) are recursively baked.

    Objects are keyed in this order:
    1. None, bool, int, float, str, bytes → unchanged
    2. tuple, list → tuple of baked items
    3. set, frozenset → tuple of sorted, baked items
    4. Mapping (dict, MappingProxyType) → tuple of sorted (key_str, baked_value) pairs
    5. Objects with cache_key() → ("cache_key", type_qualname, cache_key())
    6. Objects with jit_cache_key() → ("jit_cache_key", type_qualname, jit_cache_key())
    7. Dataclass instances → frozen_dataclass_key(value)
    8. functools.partial → ("partial", baked(func), baked(args), baked(keywords))
    9. Callables with __closure__ → ("callable", "module.qualname", tuple of captured)
    10. Bound methods (non-class __self__) → ("callable", "module.qualname", instance)
    11. Plain callables (function, class, builtin) → ("callable", "module.qualname")
    12. Array-like (has shape and dtype) → array_key(value) or raises on tracer
    13. Enum members → ("enum", type_qualname, member_name)
    14. Objects with custom __repr__ → ("repr", type_qualname, repr(value))
    15. Otherwise → TypeError
    """
    # Phase 1: Scalars unchanged
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value

    # Phase 2: Containers
    if isinstance(value, (tuple, list)):
        return tuple(baked(v) for v in value)

    if isinstance(value, (set, frozenset)):
        return tuple(sorted((baked(v) for v in value), key=repr))

    if isinstance(value, (Mapping, types.MappingProxyType)):
        return tuple(sorted((str(k), baked(v)) for k, v in value.items()))

    # Phase 2b: a class object is keyed by its qualified name. This runs before
    # the delegation phase because getattr(cls, "cache_key") on a class that
    # DEFINES cache_key is the unbound function, and calling it would raise.
    if isinstance(value, type):
        return ("callable", f"{value.__module__}.{value.__qualname__}")

    # Phase 3: Delegation to cache_key or jit_cache_key
    if callable(getattr(value, "cache_key", None)):
        return ("cache_key", type(value).__qualname__, value.cache_key())

    if callable(getattr(value, "jit_cache_key", None)):
        return ("jit_cache_key", type(value).__qualname__, value.jit_cache_key())

    # Phase 4: Dataclass
    if dataclasses.is_dataclass(value):
        return frozen_dataclass_key(value)

    # Phase 5: functools.partial
    if isinstance(value, functools.partial):
        return (
            "partial",
            baked(value.func),
            baked(value.args),
            baked(value.keywords),
        )

    # Phase 5b: Callable (function, method, class)
    if callable(value):
        qualname = f"{value.__module__}.{value.__qualname__}"

        # Check for closure: non-empty __closure__ attribute
        closure = getattr(value, "__closure__", None)
        if closure is not None:
            # Bake each cell's contents. Empty cells raise ValueError on cell_contents access.
            captured = []
            for cell in closure:
                try:
                    cell_value = cell.cell_contents
                    captured.append(baked(cell_value))
                except ValueError:
                    # Empty cell
                    captured.append(("empty_cell",))
            return ("callable", qualname, tuple(captured))

        # Check for bound method: __self__ present and not a class
        self_obj = getattr(value, "__self__", None)
        if self_obj is not None and not isinstance(self_obj, type):
            return ("callable", qualname, baked(self_obj))

        # Plain function, class, or builtin
        return ("callable", qualname)

    # Phase 6: Array-like
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        # Check for JAX tracer
        try:
            import jax

            if isinstance(value, jax.core.Tracer):
                raise TypeError(
                    f"a traced value cannot key a cache: {type(value).__name__}"
                ) from None
        except (ImportError, AttributeError):
            # jax not available or Tracer attribute doesn't exist
            if type(value).__name__.endswith("Tracer"):
                raise TypeError(
                    f"a traced value cannot key a cache: {type(value).__name__}"
                ) from None
        return array_key(value)

    # Phase 7: Enum members
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__qualname__, value.name)

    # Phase 8: Custom __repr__
    if type(value).__repr__ is not object.__repr__:
        return ("repr", type(value).__qualname__, repr(value))

    # Phase 9: Unrecognized
    raise TypeError(
        f"cannot bake {type(value).__qualname__}: "
        f"define cache_key() on it or exclude the attribute in the policy"
    )


def frozen_dataclass_key(cfg) -> tuple:
    """Return a hashable key for a dataclass instance.

    Parameters
    ----------
    cfg : object
        A dataclass instance.

    Returns
    -------
    tuple
        ``(type_qualname, ((field_name, baked_value), ...))``

    Raises
    ------
    TypeError
        If ``cfg`` is not a dataclass instance.
    """
    if not dataclasses.is_dataclass(cfg):
        raise TypeError(f"{type(cfg).__qualname__} is not a dataclass instance")

    fields_tuple = tuple((f.name, baked(getattr(cfg, f.name))) for f in dataclasses.fields(cfg))
    return (type(cfg).__qualname__, fields_tuple)


def classify(obj, policy: KeyPolicy) -> tuple[dict[str, str], tuple[str, ...]]:
    """Classify object attributes against a policy.

    For each attribute in ``vars(obj)``, determine its mode from the policy
    (or default to "content" if absent).

    Parameters
    ----------
    obj : object
        The object to classify.
    policy : KeyPolicy
        Mapping from attribute name to PolicyRow.

    Returns
    -------
    modes : dict[str, str]
        Mapping from attribute name to its mode.
    unclassified : tuple[str, ...]
        Attribute names not present in the policy (sorted).

    Raises
    ------
    ValueError
        If a policy row's mode is not in MODES.
    """
    modes = {}
    unclassified_set = set()

    for name in sorted(vars(obj)):
        if name in policy:
            mode, _ = policy[name]
            if mode not in MODES:
                raise ValueError(f"policy mode {mode!r} for {name!r} not in MODES {MODES}")
            if mode != "exclude":
                modes[name] = mode
        else:
            modes[name] = "content"
            unclassified_set.add(name)

    return modes, tuple(sorted(unclassified_set))


def derive_key(
    obj,
    policy: KeyPolicy,
    *,
    version: int | str | None = None,
    tail: tuple = (),
) -> tuple:
    """Derive a cache key from an object and policy.

    Parameters
    ----------
    obj : object
        The object to key.
    policy : KeyPolicy
        Mapping from attribute name to PolicyRow.
    version : int, str, or None, optional
        Optional version identifier (e.g., API version, algorithm version).
        Included in the key; different versions produce different keys.
        Default: None.
    tail : tuple, optional
        Additional session state not part of any object (e.g., jax_enable_x64,
        backend name). Each element is baked and included in the key.
        Default: empty tuple.

    Returns
    -------
    tuple
        ``(type_qualname, version, entries, baked_tail)`` where ``entries`` is
        a tuple of (name, keyed_value) for each non-excluded attribute.

    Raises
    ------
    TypeError
        If an attribute with mode "shape" is not array-like.
    ValueError
        If a policy row's mode is not in MODES.

    Notes
    -----
    Attributes classified as "content" are baked (full value hashing).
    Attributes classified as "shape" are reduced to ("shape", shape_tuple, dtype_str),
    except ``None``, which stays ``None``: a "shape"-mode attribute is not
    guaranteed to be an array on every instance (e.g. ``Spectroscopy.covariance``
    defaults to ``None``, meaning "no covariance supplied", not "an array of
    unknown shape"), and ``None`` can never collide with a real ``("shape", ...)``
    tuple.
    Attributes not in the policy are classified as "content" by default.
    Excluded attributes are omitted.
    """
    modes, _ = classify(obj, policy)

    entries = []
    for name in sorted(modes):
        mode = modes[name]
        attr_value = getattr(obj, name)

        if mode == "content":
            keyed = baked(attr_value)
        elif mode == "shape":
            if attr_value is None:
                keyed = None
            elif not (hasattr(attr_value, "shape") and hasattr(attr_value, "dtype")):
                raise TypeError(
                    f"attribute {name!r} has mode 'shape' "
                    f"but is not array-like (type: {type(attr_value).__qualname__})"
                )
            else:
                arr = np.asarray(attr_value)
                keyed = ("shape", tuple(arr.shape), str(arr.dtype))
        # exclude is filtered out by classify

        entries.append((name, keyed))

    return (type(obj).__qualname__, version, tuple(entries), tuple(baked(t) for t in tail))


def assert_policy_complete(objs, policy: KeyPolicy) -> None:
    """Assert that a policy covers all attributes on the given objects.

    Over an iterable of objects, the union of ``vars()`` names must equal
    the set of policy keys.

    Parameters
    ----------
    objs : Iterable[object]
        Objects whose attributes to check.
    policy : KeyPolicy
        Mapping from attribute name to PolicyRow.

    Raises
    ------
    AssertionError
        If there are unclassified attributes (present on some object but absent
        from policy) or stale policy entries (in policy but absent from all objects).
        Lists both categories sorted in a single message.
    """
    all_attrs = set()
    for obj in objs:
        all_attrs.update(vars(obj))

    policy_attrs = set(policy.keys())

    unclassified = sorted(all_attrs - policy_attrs)
    stale = sorted(policy_attrs - all_attrs)

    if unclassified or stale:
        msg_parts = []
        if unclassified:
            msg_parts.append(f"unclassified attributes: {unclassified}")
        if stale:
            msg_parts.append(f"stale policy entries: {stale}")
        raise AssertionError("; ".join(msg_parts))

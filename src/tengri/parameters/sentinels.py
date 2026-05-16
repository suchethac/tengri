"""Sentinel objects for nested-dict model builder API.

Provides two module-level singleton sentinels:

- ``FREE`` — marks a parameter to use the registry's default prior
- ``FIXED`` — marks a parameter to pin to the registry's default value

Both sentinels preserve singleton identity across copy, pickle, and deepcopy operations.

Examples
--------
>>> from tengri import FREE, FIXED
>>> config = {"sfh_field_psd_sigma": FREE, "dust_slope": FIXED}
>>> import copy
>>> copied = copy.deepcopy(config)
>>> copied["sfh_field_psd_sigma"] is FREE
True
>>> copied["dust_slope"] is FIXED
True
"""

from __future__ import annotations

from typing import ClassVar


class _Sentinel:
    """Base class for singleton sentinel objects.

    Implements singleton pattern with pickling support to ensure that
    even after pickle/unpickle or deepcopy, identity is preserved
    (``sentinel is sentinel`` always remains True).

    Parameters
    ----------
    name : str
        Human-readable name returned by repr().

    Attributes
    ----------
    name : str
        Name of the sentinel, used in repr().
    """

    _instances: ClassVar[dict[str, _Sentinel]] = {}

    def __new__(cls, name: str) -> _Sentinel:
        """Implement singleton: return cached instance if it exists.

        Parameters
        ----------
        name : str
            Unique name for this sentinel.

        Returns
        -------
        _Sentinel
            The unique instance for this name.
        """
        if name not in cls._instances:
            instance = super().__new__(cls)
            instance.name = name
            cls._instances[name] = instance
        return cls._instances[name]

    def __repr__(self) -> str:
        """Return the sentinel's name as its representation.

        Returns
        -------
        str
            The name passed to __new__.
        """
        return self.name

    def __reduce__(self) -> tuple:
        """Support pickling while preserving singleton identity.

        Returns
        -------
        tuple
            (callable, args) pair that recreates the singleton.

        Notes
        -----
        When unpickling, calling ``_Sentinel(self.name)`` invokes __new__,
        which returns the cached instance (singleton behavior preserved).
        """
        return (_Sentinel, (self.name,))

    def __hash__(self) -> int:
        """Return a stable hash based on the sentinel's name.

        Returns
        -------
        int
            Hash of the sentinel's name.
        """
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Compare equality: only equal to itself.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        bool
            True if other is this exact sentinel instance, False otherwise.
        """
        return self is other


# Module-level singletons
FREE = _Sentinel("FREE")
"""Sentinel marking a parameter to use the registry's default prior.

Use in the nested-dict model builder API to defer prior specification to the
registry. Example::

    from tengri import FREE, FIXED
    model = tengri.SEDModel({
        "sfh_field_psd_sigma": FREE,  # use default prior
        "dust_slope": FIXED,           # use default value
    })

Identity is preserved across pickle, copy, and deepcopy operations.
"""

FIXED = _Sentinel("FIXED")
"""Sentinel marking a parameter to pin to the registry's default value.

Use in the nested-dict model builder API, especially in wildcard slots like
``'*': FIXED`` to fix all parameters not explicitly mentioned. Example::

    from tengri import FREE, FIXED
    model = tengri.SEDModel({
        "*": FIXED,                    # fix all parameters
        "sfh_field_psd_sigma": FREE,   # except this one: use default prior
    })

Identity is preserved across pickle, copy, and deepcopy operations.
"""

__all__ = ["FIXED", "FREE"]

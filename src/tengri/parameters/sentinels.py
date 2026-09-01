# SPDX-License-Identifier: BSD-3-Clause
"""Sentinel objects for nested-dict model builder API.

Provides two module-level singleton sentinels:

- ``FREE``: marks a parameter to use the registry's default prior
- ``DEFAULT``: legal only as ``Fixed(DEFAULT)``, an explicit spelling of
  "pin at the registry default" for one named parameter

Both sentinels preserve singleton identity across copy, pickle, and deepcopy operations.

Examples
--------
>>> from tengri import FREE, DEFAULT
>>> config = {"sfh_field_psd_sigma": FREE, "dust_slope": DEFAULT}
>>> import copy
>>> copied = copy.deepcopy(config)
>>> copied["sfh_field_psd_sigma"] is FREE
True
>>> copied["dust_slope"] is DEFAULT
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

Place it inside a group dict passed to :meth:`tengri.SEDModel.build`, either
against one parameter or against the ``'all_params'`` wildcard, to defer the
choice of prior to the registry::

    from tengri import SEDModel, FREE, Fixed, DEFAULT, Uniform

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE, "beta": Uniform(1.0, 3.0)},
        dust={"type": "two_component", "all_params": Fixed(DEFAULT)},
    )

Here the double-power-law SFH parameters become free on their registry
default priors, with ``beta`` free on the explicit prior given instead.

Identity is preserved across pickle, copy, and deepcopy operations.

Notes
-----
``'all_params': FREE`` frees only the parameters a group declares with a
default prior. Groups whose parameters default to :class:`tengri.Fixed`
values (radio and shock) are unaffected by the wildcard; give those an
explicit prior instead (e.g. ``shock={"frac": Uniform(0.0, 1.0)}``).
"""

DEFAULT = _Sentinel("DEFAULT")
"""Sentinel marking the registry default as the value of a ``Fixed(...)`` pin.

Only legal as the argument of :class:`tengri.Fixed`, e.g. ``Fixed(DEFAULT)``,
inside a group dict passed to :meth:`tengri.SEDModel.build` /
:func:`tengri.parameters.parse_groups`::

    from tengri import SEDModel, Fixed, DEFAULT

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        met={"type": "delta", "logzsol": Fixed(DEFAULT)},
    )

``Fixed(0.3)`` pins the parameter at your own value, ``0.3``; ``Fixed(DEFAULT)``
pins it at the registry default instead -- the same value the ``'all_params'``
wildcard set to ``Fixed(DEFAULT)`` would have used for that parameter. It
resolves through the identical canonical-default resolver, never a second
path.

Bare ``DEFAULT`` (not wrapped in ``Fixed(...)``) is not a legal value
anywhere in a group dict; it raises :class:`tengri.config.exceptions.ParameterError`
pointing at the ``Fixed(DEFAULT)`` spelling.

Identity is preserved across pickle, copy, and deepcopy operations.
"""

#: Internal wildcard key in the nested-dict grammar. Sets ``FREE`` or
#: ``Fixed(DEFAULT)`` for every parameter in a group. The normalizer rewrites user-facing
#: ``WILDCARD_ALIAS`` ('all_params') or its synonym ``WILDCARD_ALIAS_OTHER``
#: ('other_params') to this key internally. '*' is NOT a user input synonym;
#: it is the internal representation after normalization.
WILDCARD_KEY = "*"

#: Preferred, self-explanatory spelling of the wildcard key. Equivalent to
#: ``WILDCARD_KEY`` in every group dict (e.g. ``sfh={'all_params': FREE}``).
#: Emitted by :meth:`Parameters.to_groups` and shown in ``summary()`` tags.
#: ``WILDCARD_ALIAS_OTHER`` ('other_params') is an exact synonym -- see its
#: docstring for when each spelling reads best. Only one of the two may
#: appear in a given group dict.
WILDCARD_ALIAS = "all_params"

#: Exact synonym of :data:`WILDCARD_ALIAS`. Both spellings normalize to the
#: same internal :data:`WILDCARD_KEY` ('*') and are otherwise fully
#: interchangeable everywhere the grammar accepts a wildcard -- top-level
#: groups, nested sub-blocks, and the builder factories' ``all_params=`` /
#: ``other_params=`` keyword. A dict (or factory call) carrying both raises,
#: since they set the same policy twice.
#:
#: The two spellings exist for the two shapes a group dict takes. ``all_params``
#: reads best when the wildcard is the group's only directive -- the
#: everything-free (or everything-fixed) case::
#:
#:     sfh={'type': 'dpl', 'all_params': FREE}
#:
#: ``other_params`` reads best written LAST, after explicit per-parameter
#: entries, where it means "the others"::
#:
#:     sfh={'type': 'dpl', 'alpha': Uniform(0.5, 3.0), 'other_params': Fixed(DEFAULT)}
#:
#: Pick whichever reads better at the call site; the parser and every
#: downstream consumer treat them identically.
WILDCARD_ALIAS_OTHER = "other_params"

__all__ = ["DEFAULT", "FREE", "WILDCARD_ALIAS", "WILDCARD_ALIAS_OTHER", "WILDCARD_KEY"]

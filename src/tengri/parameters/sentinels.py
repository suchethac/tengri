# SPDX-License-Identifier: BSD-3-Clause
"""Sentinel objects for nested-dict model builder API.

Provides two module-level singleton sentinels:

- ``FREE``: marks a parameter to use the registry's default prior
- ``FIXED``: marks a parameter to pin to the registry's default value

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

Place it inside a group dict passed to :meth:`tengri.SEDModel.build`, either
against one parameter or against the ``'all_params'`` wildcard, to defer the
choice of prior to the registry::

    from tengri import SEDModel, FREE, FIXED, Uniform

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE, "beta": Uniform(1.0, 3.0)},
        dust={"type": "two_component", "all_params": FIXED},
    )

Here the double-power-law SFH parameters become free on their registry
default priors, with ``beta`` free on the explicit prior given instead.

Identity is preserved across pickle, copy, and deepcopy operations.

Notes
-----
``'all_params': FREE`` frees only the parameters a group declares with a
default prior. Groups whose parameters default to :class:`tengri.Fixed`
values: radio and shock; are unaffected by the wildcard; give those an
explicit prior instead (e.g. ``shock={"frac": Uniform(0.0, 1.0)}``).
"""

FIXED = _Sentinel("FIXED")
"""Sentinel marking a parameter to pin to the registry's default value.

Place it inside a group dict passed to :meth:`tengri.SEDModel.build`. Its
most common use is the ``'all_params'`` wildcard, which pins every parameter
in the group that is not named explicitly::

    from tengri import SEDModel, FREE, FIXED

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": FREE},
    )

Here the double-power-law shape parameters are held at their default values
and ``sfh_dpl_log_total_mass`` is the one SFH parameter left free.
``'all_params': FIXED`` is also what a group gets when it sets no wildcard at
all, so naming it is a way to be explicit rather than a change in behavior.

Identity is preserved across pickle, copy, and deepcopy operations.

Notes
-----
The wildcard reaches only the group it appears in. Groups you do not mention
are still built from their own defaults and may contribute free parameters of
their own; the spec above leaves ``dust_tau_bc`` and ``dust_tau_diff`` free,
because no ``dust`` group was given. Read
:meth:`tengri.Parameters.summary` on the built spec rather than assuming the
wildcard fixed everything.
"""

#: Internal wildcard key in the nested-dict grammar. Sets ``FREE``/``FIXED``
#: for every parameter in a group. The normalizer rewrites user-facing
#: ``WILDCARD_ALIAS`` ('all_params') to this key internally. '*' is NOT a
#: user input synonym; it is the internal representation after normalization.
WILDCARD_KEY = "*"

#: Preferred, self-explanatory spelling of the wildcard key. Equivalent to
#: ``WILDCARD_KEY`` in every group dict (e.g. ``sfh={'all_params': FREE}``).
#: Emitted by :meth:`Parameters.to_groups` and shown in ``summary()`` tags.
WILDCARD_ALIAS = "all_params"

__all__ = ["FIXED", "FREE", "WILDCARD_ALIAS", "WILDCARD_KEY"]

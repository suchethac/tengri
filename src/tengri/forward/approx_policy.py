# SPDX-License-Identifier: BSD-3-Clause
"""The resolved approximation policy for a built :class:`~tengri.SEDModel`.

A model's ``approx=`` argument names *intent* — :class:`~tengri.WavePrecomp`,
:class:`~tengri.SpectrumPrecomp`, :class:`~tengri.FeaturePrecomp`, or nothing.
:class:`ApproxPolicy` is what that intent resolves to: the flat, settled set of
switches the forward pass and the component precomputes actually read.

Why this is a class and not a dict
----------------------------------
It was a dict, read at 43 sites across 9 modules, almost always as
``approx.get("some_key", <default>)``. That call cannot fail on a dict — a
typo, a renamed key, or a key the caller never populated all quietly return
the default.

Here the defaults are not neutral. The default beside ``n_subbands`` at those
call sites is ``0``, which is the sentinel that *disables* the dust
quadrature, so ``approx.get("n_subbnads", 0)`` silently drops the band
projection onto the effective-wavelength form — measured up to 42 % wrong in
the rest-UV at z = 1. No exception, no warning, just different numbers.

That shape shipped once already: ``approx=SpectrumPrecomp()`` on a joint
observation reached the projector with a live photometry LUT and no
``n_subbands`` field, and the ``getattr(cfg, "n_subbands", 0)`` beside it
returned the disabling sentinel.

So this type keeps the ``Mapping`` interface the existing call sites use, but
validates the key on every read and raises on anything unknown. There is no
longer a spelling of "read a key that does not exist" that returns a default.

Notes
-----
Frozen, per the project's immutability rule: build a changed policy with
:meth:`ApproxPolicy.replace`, which validates field names, rather than
assigning into it.

This module holds no physics and imports nothing from the component tree, so
it is safe to import from anywhere in ``forward/`` or ``components/``.
"""

from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Iterator, Mapping
from typing import Any

__all__ = ["BAND_PROJECTION_KEYS", "ApproxPolicy"]

#: The band-projection knobs — the subset :class:`~tengri.WavePrecomp` owns.
#:
#: Both the default set and the copy that carries a user's ``WavePrecomp`` onto
#: the policy key off this tuple, rather than naming the fields twice. Copying
#: field-by-field is how one gets forgotten at one end and silently keeps a
#: default that contradicts the others — which is what happened when
#: ``_DEFAULT_APPROX`` and ``WavePrecomp`` each carried their own copy and
#: disagreed on two of them.
BAND_PROJECTION_KEYS: tuple[str, ...] = (
    "band_integration",
    "taylor_correction",
    "n_subbands",
    "fast_dust_emission",
)


@dataclasses.dataclass(frozen=True)
class ApproxPolicy(Mapping):
    """Resolved approximation switches for one built model.

    Parameters
    ----------
    wave_precomp : bool, default False
        Route photometry through the SSP x filter LUT rather than the exact
        wavelength grid.
    ztable : bool, default False
        Interpolate the LUT over a redshift table. An internal extension of
        ``wave_precomp`` for free or catalog-ranged redshift, not a user flag.
    spectrum_precomp : bool, default False
        Route spectroscopy through the per-pixel LUT.
    igm : bool, default True
        Precompute IGM transmission at filter effective wavelengths for the
        fixed-redshift hybrid kernel.
    band_integration : {'quadrature', 'taylor', 'effective_wavelength'}
        How the multiplicative dust screen is integrated through a bandpass.
        See :class:`~tengri.WavePrecomp` for the accuracy of each.
    taylor_correction : bool, default False
        Whether the first-order spectral-moment tensors are built. Implied by
        ``band_integration``; kept as a separate switch because the stellar
        precompute consumes it directly.
    n_subbands : int, default 5
        Quadrature nodes per filter. Implied by ``band_integration`` as above.
    fast_dust_emission : bool, default False
        Sample the dust IR template at the filter effective wavelength when
        its exact constant-response form cannot be built.

    Notes
    -----
    ``band_integration``, ``taylor_correction`` and ``n_subbands`` are resolved
    into mutual agreement by ``tengri.WavePrecomp.__post_init__`` before
    they reach here; this class stores the settled values. It does not
    re-derive them, so that there is one resolution site rather than two that
    can disagree — which is the defect that motivated this type.

    Examples
    --------
    >>> policy = ApproxPolicy()
    >>> policy.band_integration
    'quadrature'
    >>> policy["n_subbands"]
    5
    >>> policy.get("n_subbnads", 0)
    Traceback (most recent call last):
        ...
    KeyError: ...
    >>> policy.replace(wave_precomp=True).wave_precomp
    True
    """

    wave_precomp: bool = False
    ztable: bool = False
    spectrum_precomp: bool = False
    igm: bool = True

    band_integration: str = "quadrature"
    taylor_correction: bool = False
    n_subbands: int = 5
    fast_dust_emission: bool = False

    # ── Mapping interface ────────────────────────────────────────────────
    # Kept so the 43 pre-existing ``approx[...]`` / ``approx.get(...)`` call
    # sites need no edit. Both spellings validate; neither can fall through
    # to a default for a key that does not exist.

    @classmethod
    def _field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(cls))

    def _check_key(self, key: Any) -> str:
        """Raise ``KeyError`` naming the near-miss if ``key`` is not a field."""
        names = self._field_names()
        if isinstance(key, str) and key in names:
            return key
        suggestions = (
            difflib.get_close_matches(key, names, n=2, cutoff=0.6) if isinstance(key, str) else []
        )
        hint = f" Did you mean {' or '.join(map(repr, suggestions))}?" if suggestions else ""
        raise KeyError(
            f"{key!r} is not an ApproxPolicy field.{hint} "
            f"Valid keys: {', '.join(names)}. "
            "This raises rather than returning a default because the defaults "
            "here disable approximations rather than being neutral — a typo "
            "would silently change the numbers."
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, self._check_key(key))

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``self[key]``, raising on an unknown key.

        ``default`` is accepted for call-site compatibility and is never
        consulted: every field always holds a value, so the only way the
        default could be reached is a key that does not exist — precisely the
        case that must not pass silently.
        """
        return getattr(self, self._check_key(key))

    def __iter__(self) -> Iterator[str]:
        return iter(self._field_names())

    def __len__(self) -> int:
        return len(self._field_names())

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError(
            "ApproxPolicy is immutable; it replaced a mutable dict whose "
            "in-place edits made the resolved state hard to follow. Use "
            "policy.replace(**changes), which returns a new policy and "
            "validates the field names."
        )

    # ── immutable update ─────────────────────────────────────────────────

    def replace(self, **changes: Any) -> ApproxPolicy:
        """Return a new policy with ``changes`` applied.

        Parameters
        ----------
        **changes
            Field names and their new values. An unknown name raises
            ``TypeError`` from :func:`dataclasses.replace`.

        Returns
        -------
        ApproxPolicy
            A new instance; the receiver is unchanged.
        """
        return dataclasses.replace(self, **changes)

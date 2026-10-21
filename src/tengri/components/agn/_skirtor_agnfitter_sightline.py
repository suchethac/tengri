# SPDX-License-Identifier: BSD-3-Clause
r"""Sightline of the SKIRTOR-AGNfitter tori, read from the model's one inclination.

The AGNfitter-rX ``SKIRTOR_mean_1p``, ``SKIRTOR_mean_2p`` and ``SKIRTOR_mean_3p``
reductions of the SKIRTOR library (Stalevski et al. 2016 [1]_; packaged by
AGNfitter-rX [2]_) tabulate the torus on an inclination axis :math:`i` [deg], measured
from the polar axis: :math:`i = 0^\circ` is face-on, :math:`i = 90^\circ` edge-on. That
is the same angle ``agn_cos_inc`` is the cosine of, so the library is keyed at

.. math::

    i = \frac{180^\circ}{\pi}\,\arccos(\cos i),

where :math:`\cos i` is ``agn_cos_inc`` [dimensionless] and :math:`i` is the library's
inclination [deg]. The model has one inclination (owner ruling, #2605): the torus
sightline, the disc's visibility mask and the X-ray corona all read ``agn_cos_inc``.

The library's own inclination parameter ``agn_incl_skirtor`` is retired (#3085).
:func:`resolve_retired_incl_skirtor` translates it, with a ``DeprecationWarning``, to
``agn_cos_inc = cos(radians(agn_incl_skirtor))``, and refuses a call that gives both
names with different sightlines.

References
----------
.. [1] M. Stalevski, C. Ricci, Y. Ueda, P. Lira, J. Fritz, and M. Baes, "The dust
   covering factor in active galactic nuclei," MNRAS, 458, 2288 (2016).
   arXiv:1602.06954. https://doi.org/10.1093/mnras/stw444
.. [2] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the radio-to-X-ray
   spectral energy distributions of AGNs," A&A 688, A46 (2024). arXiv:2405.12111.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from tengri.components.agn.fritz import _COS_INC_FACE_ON_CAP

__all__ = [
    "AXIS_TRANSFORMS",
    "RETIRED_INCL_KEY",
    "accepts_retired_incl_skirtor",
    "free_incl_position",
    "resolve_retired_incl_skirtor",
    "skirtor_incl_from_cos_inc",
]

F = TypeVar("F", bound=Callable[..., Any])

#: The retired inclination parameter of the SKIRTOR-AGNfitter tori [deg].
RETIRED_INCL_KEY = "agn_incl_skirtor"

#: Largest difference in ``cos i`` at which a retired angle and an ``agn_cos_inc`` given in
#: the same call count as one sightline: ``cos(radians(30))`` and the declared default
#: ``agn_cos_inc`` differ by 1.1e-16.
_AGREEMENT_ATOL = 1.0e-9


def skirtor_incl_from_cos_inc(cos_inc) -> jnp.ndarray:
    r"""The SKIRTOR-AGNfitter library inclination for the model's one inclination.

    Parameters
    ----------
    cos_inc : float or array_like
        :math:`\cos i`, with :math:`i` the inclination from the polar axis
        (1 = face-on, 0 = edge-on) [dimensionless].

    Returns
    -------
    ndarray
        :math:`i` [deg], the coordinate of the library's ``incl_axis`` (0 face-on,
        90 edge-on).

    Notes
    -----
    .. math::

        i = \frac{180^\circ}{\pi}\,\arccos(\cos i).

    The round trip ``cos(radians(i))`` then ``arccos`` is not exact in float64: the
    declared default ``agn_cos_inc`` (``cos 30 deg``) maps to
    :math:`i = 30^\circ + 1.4\times 10^{-14}`.

    **JIT-compatible**: yes. **Gradient-safe**: yes; finite and nonzero on the whole
    :math:`\cos i` prior. Above :math:`\cos i = 1 - 10^{-6}` (0.08 deg from the pole,
    where the slope of ``arccos`` is infinite) the line tangent at that cap continues,
    as for the Fritz library's elevation
    (:func:`~tengri.components.agn.fritz.fritz_psy_from_cos_inc`).

    References
    ----------
    .. [1] M. Stalevski et al., MNRAS, 458, 2288 (2016). arXiv:1602.06954.
    """
    c = jnp.asarray(cos_inc)
    cap = _COS_INC_FACE_ON_CAP
    incl_cap = jnp.degrees(jnp.arccos(cap))
    slope_cap = -jnp.degrees(1.0 / jnp.sqrt(1.0 - cap * cap))
    inside = jnp.degrees(jnp.arccos(jnp.where(c <= cap, c, cap)))
    beyond = incl_cap + slope_cap * (c - cap)
    return jnp.where(c <= cap, inside, beyond)


def free_incl_position(incl_axis: int, collapsed: Mapping[int, float]) -> int | None:
    """Position of the inclination among a collapsed lookup's free axis values.

    Parameters
    ----------
    incl_axis : int
        Index of the inclination axis in the full grid.
    collapsed : mapping of int to float
        Collapsed axis index -> the coordinate it was collapsed at.

    Returns
    -------
    int or None
        Index of the inclination in the free values a collapsed lookup receives, or
        ``None`` when the inclination axis was itself collapsed.
    """
    if incl_axis in collapsed:
        return None
    return incl_axis - sum(1 for i in collapsed if i < incl_axis)


#: Grid-axis coordinate of the parameter governing each SKIRTOR-AGNfitter axis that is not
#: tabulated in that parameter: the inclination axis is in degrees, governed by
#: ``agn_cos_inc``. Passed to ``collapse_fixed_axes`` so a Fixed ``agn_cos_inc`` collapses
#: at its degree coordinate.
AXIS_TRANSFORMS: dict[str, Callable[[float], float]] = {
    "agn_cos_inc": lambda cos_inc: float(skirtor_incl_from_cos_inc(cos_inc)),
}


def _is_concrete(value) -> bool:
    """Whether ``value`` can be read on the host (not a JAX tracer)."""
    return not isinstance(value, jax.core.Tracer)


def _show(value) -> str:
    """A concrete scalar as a plain float, anything else as its repr."""
    return repr(float(value)) if jnp.ndim(value) == 0 else repr(value)


def _retired_message(caller: str, incl, cos_inc) -> str:
    """The deprecation text, with the translated value when it is concrete."""
    value = (
        f" ({_show(incl)} deg -> agn_cos_inc = {_show(cos_inc)})"
        if _is_concrete(incl) and jnp.ndim(incl) == 0
        else ""
    )
    return (
        f"{caller}: 'agn_incl_skirtor' is deprecated and will be removed in tengri v1.0 "
        f"(#3085). The SKIRTOR-AGNfitter tori take their sightline from the model's one "
        f"inclination, 'agn_cos_inc' (cos i, i from the polar axis), which the disc and the "
        f"X-ray corona also read. Translated to agn_cos_inc = cos(radians(agn_incl_skirtor))"
        f"{value}; pass 'agn_cos_inc' instead."
    )


def resolve_retired_incl_skirtor(kwargs: Mapping[str, Any], *, caller: str) -> Mapping[str, Any]:
    """Translate a retired ``agn_incl_skirtor`` in ``kwargs`` to ``agn_cos_inc``.

    Parameters
    ----------
    kwargs : mapping
        Keyword arguments of an AGN call.
    caller : str
        Name of the function receiving ``kwargs``, for the messages.

    Returns
    -------
    mapping
        ``kwargs`` itself when it carries no ``agn_incl_skirtor``. Otherwise a new dict
        without it and with ``agn_cos_inc``: the given one when both names are present and
        agree, else ``cos(radians(agn_incl_skirtor))``.

    Raises
    ------
    ValueError
        When ``kwargs`` carries both ``agn_incl_skirtor`` and ``agn_cos_inc`` and they
        name different sightlines (by more than ``1e-9`` in ``cos i``), or when either
        is traced, so agreement cannot be checked.

    Warns
    -----
    DeprecationWarning
        Whenever ``agn_incl_skirtor`` is present.
    """
    if RETIRED_INCL_KEY not in kwargs:
        return kwargs
    incl = kwargs[RETIRED_INCL_KEY]
    translated = jnp.cos(jnp.deg2rad(jnp.asarray(incl)))
    rest = {k: v for k, v in kwargs.items() if k != RETIRED_INCL_KEY}
    if "agn_cos_inc" in kwargs:
        given = kwargs["agn_cos_inc"]
        if not (_is_concrete(incl) and _is_concrete(given)):
            raise ValueError(
                f"{caller}: got both 'agn_incl_skirtor' (retired, #3085) and 'agn_cos_inc' "
                f"under a JAX transform, where their agreement cannot be checked. Pass "
                f"'agn_cos_inc' only: it is the one inclination of the AGN."
            )
        gap = jnp.max(jnp.abs(translated - jnp.asarray(given)))
        if not float(gap) <= _AGREEMENT_ATOL:
            raise ValueError(
                f"{caller}: 'agn_incl_skirtor' = {_show(incl)} deg (cos i = {_show(translated)}) "
                f"and 'agn_cos_inc' = {_show(given)} name two different sightlines. The AGN has "
                f"one inclination, 'agn_cos_inc' (#2605); 'agn_incl_skirtor' is retired (#3085). "
                f"Pass 'agn_cos_inc' only."
            )
        warnings.warn(_retired_message(caller, incl, translated), DeprecationWarning, stacklevel=3)
        return rest
    warnings.warn(_retired_message(caller, incl, translated), DeprecationWarning, stacklevel=3)
    return {**rest, "agn_cos_inc": translated}


def accepts_retired_incl_skirtor(fn: F) -> F:
    """Decorate a SKIRTOR-AGNfitter entry point to translate a retired ``agn_incl_skirtor``.

    Parameters
    ----------
    fn : callable
        A function taking ``agn_cos_inc`` as a keyword argument.

    Returns
    -------
    callable
        ``fn`` behind :func:`resolve_retired_incl_skirtor`; ``inspect.signature`` reports
        ``fn``'s own signature.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if RETIRED_INCL_KEY in kwargs:
            kwargs = dict(resolve_retired_incl_skirtor(kwargs, caller=f"{fn.__name__}()"))
        return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]

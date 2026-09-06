# SPDX-License-Identifier: BSD-3-Clause
"""Dust attenuation-law registry and shared curve helpers.

Leaf module for the ``DUST_LAWS`` catalog: the ``register_dust_law`` decorator,
``resolve_dust_law`` / ``list_laws`` lookups, the ``DustLawRegistryEntry``
metadata record, and the shared curve helpers (``_drude_profile``,
``_calzetti_l02_kprime``) used across law families. Imports only ``jnp`` and
physical constants, so law modules and the ``attenuation`` facade can import it
without an import cycle (#843).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # annotation only; see the local import in list_laws (#843)
    from tengri.registry import _RegistryTable

# ── Attenuation law catalog ─────────────────────────────────────


@dataclass(frozen=True)
class DustLawRegistryEntry:
    """Registry entry for a dust attenuation law with optional metadata.

    Attributes
    ----------
    callable : Callable
        The dust attenuation law function.
    citation : str
        Optional academic citation. Default empty string.
    status : str
        Model status: "production", "experimental", "demo", or "deprecated".
        Default "production".
    short_doc : str
        Optional one-line description. Default empty string.

    Notes
    -----
    **JIT-compatible**: no, dataclass for registry initialization.

    """

    callable: Callable
    citation: str = ""
    status: str = "production"
    short_doc: str = ""

    def __call__(self, *args, **kwargs):
        """Forward calls to the wrapped callable."""
        return object.__getattribute__(self, "callable")(*args, **kwargs)

    def __getattr__(self, name: str):
        """Forward attribute access to wrapped callable."""
        callable_obj = object.__getattribute__(self, "callable")
        return getattr(callable_obj, name)


DUST_LAWS: dict[str, Callable] = {}


def register_dust_law(
    name: str,
    citation: str = "",
    status: str = "production",
    short_doc: str = "",
) -> Callable:
    """Decorator factory that registers a dust attenuation curve under a name.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"calzetti"``, ``"power_law"``).
    citation : str, optional
        Academic citation for the model. Default empty string.
    status : str, optional
        Model status ("production", "experimental", "demo", "deprecated").
        Default "production".
    short_doc : str, optional
        One-line description. Default empty string.

    Returns
    -------
    Callable
        Decorator that registers the decorated function and returns it unchanged.

    Notes
    -----
    **JIT-compatible**: no, registration happens at factory time before JIT.

    Decorated functions must implement the ``DustAttenuationLaw`` protocol:
    accept a wavelength array and keyword arguments, returning an attenuation
    curve ``k(λ)`` [dimensionless].
    """

    def decorator(fn: Callable) -> Callable:
        """Inner decorator that registers function in DUST_LAWS dict.

        Parameters
        ----------
        fn : Callable
            Dust attenuation law function matching ``DustAttenuationLaw`` protocol.

        Returns
        -------
        Callable
            The input function unchanged (enables use as a decorator).

        Notes
        -----
        **JIT-compatible**: depends on the decorated function.
        """
        entry = DustLawRegistryEntry(
            callable=fn,
            citation=citation,
            status=status,
            short_doc=short_doc,
        )
        DUST_LAWS[name] = entry
        law_kwarg_names.cache_clear()
        return fn

    return decorator


def resolve_dust_law(name: str) -> Callable:
    """Return a registered dust attenuation law by name.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"calzetti"``, ``"power_law"``).

    Returns
    -------
    Callable
        The registered dust law function.

    Raises
    ------
    ValueError
        If ``name`` is not in the registry.

    Notes
    -----
    **JIT-compatible**: no, registry lookup happens at factory time.

    The returned function matches the ``DustAttenuationLaw`` protocol and can
    be called with wavelengths and law-specific parameters.
    """
    if name not in DUST_LAWS:
        raise ValueError(f"Unknown dust law '{name}'. Available: {list(DUST_LAWS.keys())}")
    return DUST_LAWS[name]


def _law_callable(law: str | Callable) -> Callable:
    """Resolve a law given either its registry key or the function itself.

    Parameters
    ----------
    law : str or callable
        Registry key, or an already-resolved law function.

    Returns
    -------
    Callable
        The law function. A :class:`DustLawRegistryEntry` is unwrapped to the
        function it holds so :func:`inspect.signature` sees the real signature.
    """
    fn = resolve_dust_law(law) if isinstance(law, str) else law
    if isinstance(fn, DustLawRegistryEntry):
        fn = object.__getattribute__(fn, "callable")
    return fn


@cache
def law_kwarg_names(law: str | Callable) -> frozenset[str]:
    """Keyword arguments the attenuation law declares, beyond ``wavelength``.

    Parameters
    ----------
    law : str or callable
        Registry key (e.g. ``"noll09"``) or the law function itself.

    Returns
    -------
    frozenset of str
        Law-function keyword names (``n_slope``, ``dust_delta``, ...). Empty
        for a curve that reads only wavelength (``calzetti``, ``smc``, the
        grain-model tables).

    Notes
    -----
    **JIT-compatible**: no; signature introspection, cached because every
    evaluation of a two-screen model asks twice and the registry is populated
    at import time. :func:`register_dust_law` clears the cache.

    The signature is the only honest source. A law that also declared
    ``**kwargs`` would *accept* every parameter and read none of them, which is
    why ``tools/check_dust_law_kwargs.py`` refuses one: "does the call
    succeed?" cannot answer "does this law read this parameter?".
    """
    try:
        sig = inspect.signature(_law_callable(law))
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        return frozenset()
    return frozenset(
        p.name
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL) and p.name != "wavelength"
    )


def select_law_kwargs(law: str | Callable, law_params: Mapping) -> dict:
    """Narrow a shared law-parameter dict to what one law reads.

    Parameters
    ----------
    law : str or callable
        Registry key or law function.
    law_params : Mapping
        Law-function keyword arguments, possibly the union across two screens.

    Returns
    -------
    dict
        The subset of ``law_params`` whose keys ``law`` declares.

    Notes
    -----
    **JIT-compatible**: yes; dict construction only, values pass through
    untouched (traced arrays stay traced).

    A two-screen model holds ONE parameter dict and two laws, so a key can
    legitimately belong to the other screen. Narrowing here is what lets the
    laws themselves drop ``**kwargs``: the caller decides what each law is
    offered, and the law refuses anything it cannot use. Callers must first
    check that every key is read by *some* law in play
    (:func:`reject_unread_law_kwargs`), otherwise the narrowing silently
    absorbs a key nothing reads, which is the defect this pair replaces
    (#2185).
    """
    declared = law_kwarg_names(law)
    return {k: v for k, v in law_params.items() if k in declared}


def reject_unread_law_kwargs(law_params: Mapping, laws: tuple, context: str) -> None:
    """Raise when a law-parameter key is read by none of the laws in play.

    Parameters
    ----------
    law_params : Mapping
        Law-function keyword arguments the caller assembled.
    laws : tuple
        Registry keys or law functions the parameters will be offered to.
    context : str
        Caller name for the message, e.g. ``"two_component_dust"``.

    Raises
    ------
    ValueError
        Naming the unread keys, the laws in play, and what they do read.

    Notes
    -----
    **JIT-compatible**: no; a build-time / trace-time key check on Python
    strings. Runs once per trace, not per array element.
    """
    read: set[str] = set()
    for law in laws:
        if law is None:
            continue
        read |= set(law_kwarg_names(law))
    unread = sorted(k for k in law_params if k not in read)
    if not unread:
        return
    named = ", ".join(repr(k) for k in laws if k is not None)
    accepts = ", ".join(sorted(read)) if read else "no shape parameters at all"
    plural = "s" if len(unread) > 1 else ""
    raise ValueError(
        f"{context}: {', '.join(repr(k) for k in unread)} "
        f"{'are' if len(unread) > 1 else 'is'} not read by any law in play "
        f"({named}), so the value{plural} would be silently discarded. "
        f"These laws read: {accepts}."
    )


# Curated headline subset for gallery comparisons. Each entry's value is the
# canonical kwargs to evaluate the law at "nominal" tuning; callers pass
# wavelengths positionally and get k(lambda) at tau_V=1.
_HEADLINE_LAWS: dict[str, tuple[str, dict]] = {
    "Calzetti+2000": ("calzetti", {}),
    "Charlot & Fall (slope=-0.7)": ("power_law", {"n_slope": -0.7}),
    "Cardelli+1989 (MW, Rv=3.1)": ("cardelli", {"dust_Rv": 3.1}),
    "SMC (Gordon+2003)": ("smc", {}),
    "Kriek & Conroy 2013": ("kriek_conroy", {"dust_bump_strength": 1.0, "dust_delta": 0.0}),
    "Salim+2018": ("salim", {}),
}


def list_laws(headline: bool = True) -> _RegistryTable:
    """List the attenuation laws, each row carrying a one-arg callable.

    The ``fn`` column holds ``fn(wave_aa) -> k(wave)`` at the law's
    canonical parameters with ``tau_V = 1``. Use for plotting k(lambda)
    comparisons without restating each law's argument signature.

    Parameters
    ----------
    headline : bool, optional
        If True (default) list the 6 textbook laws, named by display
        label with citation. If False list every registered law by
        registry name with no kwargs baked in.

    Returns
    -------
    _RegistryTable
        One row per law, with columns ``name`` and ``fn``. ``fn`` holds a
        live callable and is hidden from the printed table.

    Notes
    -----
    Returned ``{label: callable}`` before #1574; every discovery verb
    returns a table (#1285). ``.to_dict("fn")`` reproduces that mapping
    exactly, so ``for label, fn in list_laws().items()`` becomes
    ``for label, fn in list_laws().to_dict("fn").items()``.

    Examples
    --------
    >>> from tengri.dust import list_laws
    >>> for label, fn in list_laws().to_dict("fn").items():
    ...     plt.plot(wave, fn(wave), label=label)
    """
    # Imported here, not at module scope: this module is a leaf by design
    # so the law modules and the attenuation facade can import it without
    # a cycle (#843, see the module docstring).
    from tengri.registry import _RegistryTable

    if headline:
        rows = [
            {
                "name": label,
                "kind": "dust_law",
                "law": name,
                "fn": (lambda w, _n=name, _kw=kwargs: DUST_LAWS[_n](w, **_kw)),
            }
            for label, (name, kwargs) in _HEADLINE_LAWS.items()
        ]
    else:
        rows = [
            {"name": name, "kind": "dust_law", "law": name, "fn": fn}
            for name, fn in DUST_LAWS.items()
        ]
    return _RegistryTable(rows)


# ── Utility: Drude profile for the 2175 Angstrom UV bump ──────────


def _drude_profile(
    wave_um: jnp.ndarray,
    x0: float = 0.2175,
    gamma: float = 0.035,
) -> jnp.ndarray:
    r"""Drude profile for the 2175 Å UV absorption bump.

    Parameters
    ----------
    wave_um : array_like, shape (n_wave,)
        Wavelength grid. [μm]
    x0 : float, optional
        Central wavelength of the bump. [μm] Default: 0.2175 (2175 Å).
    gamma : float, optional
        FWHM of the profile. [μm] Default: 0.035 (350 Å).

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized Drude profile (dimensionless, in [0, 1]).

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The Drude profile is:

    .. math::

        D(\lambda; \lambda_0, \gamma) = \frac{(\lambda \, \gamma)^2}{(\lambda^2 - \lambda_0^2)^2 + (\lambda \, \gamma)^2}

    where :math:`\lambda` is wavelength [μm], :math:`\lambda_0` is the central wavelength [μm],
    and :math:`\gamma` is the FWHM [μm]. This is a standard resonance profile used to model
    the 2175 Å silicate bump in interstellar dust attenuation.

    **Upstream**: Following Kriek & Conroy (2013) [1]_ and standard dust attenuation conventions.

    References
    ----------
    .. [1] M. Kriek and C. Conroy, "The Dust Attenuation Law in Distant Galaxies: Evidence
       for Variation with Spectral Type," ApJL, 775, L16 (2013).
       https://doi.org/10.1088/2041-8205/775/1/L16
    """
    return (wave_um * gamma) ** 2 / ((wave_um**2 - x0**2) ** 2 + (wave_um * gamma) ** 2)


# ── Internal helpers ──────────────────────────────────────────────


def _calzetti_l02_kprime(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Compute k'(lambda) = A(lambda)/E(B-V) using Leitherer (2002) + Calzetti (2000).

    Switches between the two extinction laws depending on wavelength: Leitherer et al. (2002)
    for the far-UV, Calzetti et al. (2000) for longer wavelengths. Returns the raw
    reddening curve (NOT normalized by R_V).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Reddening curve k'(λ) = A(λ)/E(B-V), unnormalized. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    Uses piecewise polynomials following Leitherer et al. (2002) and Calzetti et al. (2000).
    The transition occurs at 0.18 μm (1800 Å), matching the standalone ``dust_attenuation.averages.L02`` model.

    References
    ----------
    .. [1] C. Leitherer et al., "Global Far-Ultraviolet (912-1800 Å) Properties of
       Star-forming Galaxies," ApJS, 140, 303 (2002).
       https://doi.org/10.1086/342486

    .. [2] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
       Dust in Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250
    """
    wave_um = wavelength / 1e4
    x = 1.0 / wave_um
    rv = 4.05

    # L02 polynomial (valid 0.097-0.18 um)
    k_l02 = 5.472 + 0.671 * x - 9.218e-3 * x**2 + 2.620e-3 * x**3

    # Calzetti UV polynomial (valid 0.12-0.63 um)
    k_uv = 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + rv

    # Calzetti IR polynomial (valid 0.63-2.2 um)
    k_ir = 2.659 * (-1.857 + 1.040 * x) + rv

    # L02 valid range: 970-1800 A (Leitherer+2002 ApJS 140 303 Eq. 14).  Use 0.18 um cutoff,
    # matching the standalone leitherer02 function; 0.15 um was too conservative.
    k_calz = jnp.where(wave_um >= 0.63, k_ir, k_uv)
    return jnp.where(wave_um <= 0.18, k_l02, k_calz)


def _calzetti_kprime_unnormalized(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Compute k'(lambda) = A(lambda)/E(B-V) using Calzetti (2000) only, unnormalized.

    Returns the raw reddening curve (NOT normalized by R_V or k(5500)).
    Used internally by kriek_conroy to apply its own normalization logic.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid. [Å]

    Returns
    -------
    ndarray, shape (n_wave,)
        Raw reddening curve k'(λ) = A(λ)/E(B-V), unnormalized by R_V. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    Uses piecewise polynomials following Calzetti et al. (2000).
    This is a pure polynomial evaluation with no normalization applied.

    References
    ----------
    .. [1] S. Calzetti et al., "The Dust Content and Opacity of Star-Forming
       Galaxies," ApJ, 533, 682 (2000).
       https://doi.org/10.1086/308692
    """
    wave_um = wavelength / 1e4
    x = 1.0 / wave_um
    rv = 4.05

    # Calzetti UV polynomial (valid 0.12-0.63 um)
    k_uv = 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + rv

    # Calzetti IR polynomial (valid 0.63-2.2 um)
    k_ir = 2.659 * (-1.857 + 1.040 * x) + rv

    # Piecewise selection. Polynomial is extrapolated through the FUV to keep the
    # dust attenuation defined across the full SED range.
    return jnp.where(wave_um >= 0.63, k_ir, k_uv)

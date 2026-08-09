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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # annotation only — see the local import in list_laws (#843)
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
    **JIT-compatible**: no — dataclass for registry initialization.

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
    **JIT-compatible**: no — registration happens at factory time before JIT.

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
    **JIT-compatible**: no — registry lookup happens at factory time.

    The returned function matches the ``DustAttenuationLaw`` protocol and can
    be called with wavelengths and law-specific parameters.
    """
    if name not in DUST_LAWS:
        raise ValueError(f"Unknown dust law '{name}'. Available: {list(DUST_LAWS.keys())}")
    return DUST_LAWS[name]


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
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

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
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

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

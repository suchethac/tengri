# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories for the dust_emission top-level group.

The grammar now separates dust attenuation and IR emission into two peer
top-level groups:

>>> dust_attenuation = {"type": "two_component", "law": "calzetti"}
>>> dust_emission = {"type": "dale2014", "all_params": FIXED, "alpha_dale": Fixed(2.0)}

Each emission variant returned by
``tengri.parameters.groups._valid_dust_emission_types`` gets a
factory in this module. That helper derives directly from the live
``DUST_EMISSION_MODELS`` registry (plus a closed set of lazy-loadable
names like ``dl07_tabulated``) so the validator path and the factory
namespace share a single source of truth (ADR-0005 / ADR-0008). The
parser activates a single superset of dust-emission params regardless
of which model is chosen, so all factories share an identical
signature; the variant string selects the physics.

Examples
--------
>>> from tengri import builders, FIXED, Fixed
>>> builders.dust.emission.dale2014(all_params=FIXED, alpha_dale=Fixed(2.0))  # doctest: +SKIP
{'type': 'dale2014', 'all_params': FIXED, 'alpha_dale': Fixed(2.0)}
"""

from __future__ import annotations

from collections.abc import Callable

from tengri.builders._factory import make_factory, short_form
from tengri.parameters.groups import _valid_dust_emission_types
from tengri.parameters.registry import recipe_parameters
from tengri.parameters.sentinels import FREE, WILDCARD_ALIAS

_PREFIXES = ("dust_",)
# Param names that belong to dust *emission* rather than attenuation.
# The parser activates these only when an emission model is selected;
# they're the natural set for the emission factories.
_EMISSION_PREFIXES = (
    "dust_T",
    "dust_beta_ir",
    "dust_alpha_dale",
    "dust_umin",
    "dust_umax",
    "dust_gamma_dl",
    "dust_qpah",
    "dust_pah",
    "dust_lgU",
    "dust_log_",
    # Energy-balance relaxation factor (η): L_IR = η · L_absorbed. Default
    # Fixed(1.0) = strict balance; free it (e.g. eta_balance=LogNormal(0, 0.2))
    # to fit galaxies whose UV/optical and FIR are spatially decoupled and so
    # violate strict energy balance (high-z sources). See ``_params.py``.
    "dust_eta_balance",
    # Two-temperature ``energy_balance_split`` knobs — warm/cold split with an
    # optional AGN-IR term. Threaded through ``two_component`` so the model is
    # fully free-able through the grammar (no silent-dropped params).
    "dust_f_cold",
    "dust_L_agn_ir",
    "dust_beta_warm",
    "dust_beta_cold",
)


def _discover_params(variant: str) -> list[str]:
    """Return short-form names for the emission-side dust params.

    Each emission variant activates the same superset (the parser is
    conservative); we still introspect per variant so that future
    variants with extra params surface automatically.
    """
    recipe = {
        "sfh": {"type": "dpl"},
        "dust_attenuation": {
            "type": "two_component",
            "law": "calzetti",  # Shared law for both BC and diffuse
            WILDCARD_ALIAS: FREE,
        },
        "dust_emission": {"type": variant, WILDCARD_ALIAS: FREE},
    }
    records = recipe_parameters(recipe, free_only=False)
    out: list[str] = []
    for rec in records:
        if not rec.name.startswith(_EMISSION_PREFIXES):
            continue
        out.append(short_form(rec.name, prefixes=_PREFIXES))
    return out


def _populate_factories() -> dict[str, Callable[..., dict]]:
    factories: dict[str, Callable[..., dict]] = {}
    for variant in sorted(_valid_dust_emission_types()):
        factories[variant] = make_factory(
            variant=variant,
            short_params=_discover_params(variant),
            qualname_prefix="tengri.builders.dust.emission",
            module_name="tengri.builders.dust.emission",
            short_doc=f"Dust IR emission model: {variant!r}.",
        )
    return factories


_FACTORIES = _populate_factories()
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of emission-model variants exposed by this module."""
    return sorted(_FACTORIES)


def relaxed_energy_balance(model: str = "dale2014", *, sigma: float = 0.2) -> dict:
    """Dust IR emission with a *relaxed* (opt-in) energy balance.

    Returns a ``dust_emission`` group dict that frees the energy-balance factor
    ``dust_eta_balance`` (``L_IR = eta * L_absorbed``) under a soft
    ``LogNormal(mu=0, sigma)`` prior centered on strict balance (median
    ``eta = 1``). Use it for galaxies whose UV/optical and FIR are spatially
    decoupled and so violate strict energy balance (e.g. high-z sources) — the
    way AGNfitter offers an *optional* energy-balance prior, in contrast to
    CIGALE/MAGPHYS which enforce it. The IR template shape stays fixed; only the
    overall IR luminosity is allowed to float around the absorbed energy.

    Parameters
    ----------
    model : str
        Emission model variant carrying the IR shape (default ``'dale2014'``).
        Any name in :func:`available` works.
    sigma : float
        Standard deviation (in natural-log space) of the ``LogNormal`` prior on
        ``eta``. ``0.2`` allows ~+/-20% deviation; widen for looser balance.

    Returns
    -------
    dict
        A ``dust_emission`` group dict, e.g. ``{'type': 'dale2014', 'all_params': FIXED,
        'eta_balance': LogNormal(mu=0.0, sigma=0.2)}``.

    Examples
    --------
    >>> from tengri import SEDModel, builders
    >>> model = SEDModel.build(  # doctest: +SKIP
    ...     ssp_data=ssp,
    ...     observation=obs,
    ...     dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
    ...     dust_emission=builders.dust.emission.relaxed_energy_balance(),
    ... )
    """
    from tengri.parameters.priors import LogNormal
    from tengri.parameters.sentinels import FIXED

    if model not in _FACTORIES:
        raise ValueError(f"Unknown dust emission model {model!r}. Available: {available()}")
    return {"type": model, WILDCARD_ALIAS: FIXED, "eta_balance": LogNormal(mu=0.0, sigma=sigma)}


__all__ = ["available", "relaxed_energy_balance", *sorted(_FACTORIES)]

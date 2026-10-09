# SPDX-License-Identifier: BSD-3-Clause
"""Lyu & Rieke 2018 polar-dust AGN templates.

The three model names select a fixed public template family. ``agn_log_lbol``
sets the luminosity of the finite 0.01–1000 μm tau=0 reference template, and
``agn_lyu2018_tau_v`` interpolates the family along its published optical-depth
axis.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

import h5py
import jax.numpy as jnp

from tengri._data_setup import find_data_str
from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.utils.scale import apply_log10_scale
from tengri.utils.sed_quantities import LOG10_L_SUN

LYU2018_AGN_FAMILIES = {
    "lyu2018": "norm",
    "lyu2018_wdd": "wdd",
    "lyu2018_hdd": "hdd",
}


@functools.cache
def load_lyu2018_templates(family: str) -> dict[str, jnp.ndarray]:
    """Load one public Lyu AGN family as arrays for the forward model.

    Parameters
    ----------
    family : {"norm", "wdd", "hdd"}
        Static family selection corresponding to NORMAL, WDD, or HDD.

    Returns
    -------
    dict[str, ndarray]
        Wavelengths in Å, the dimensionless tau_V grid, relative F_nu template
        rows, and the fixed tau=0 native-frequency integral ``c0_reference``.

    Notes
    -----
    **JIT-compatible**: no. This cached loader performs file I/O before the
    arrays are passed to the compiled forward calculation.
    """
    if family not in {"norm", "wdd", "hdd"}:
        raise ValueError(f"Unknown Lyu AGN family {family!r}.")
    path = find_data_str("lyu2018_agn.h5")
    if path is None:
        raise FileNotFoundError("Packaged Lyu2018 AGN templates were not found.")
    with h5py.File(path, "r") as h5:
        group = h5[f"families/{family}"]
        return {
            "wavelength_aa": jnp.asarray(group["wavelength_aa"][:]),
            "tau_v": jnp.asarray(group["tau_v"][:]),
            "flux_nu_relative": jnp.asarray(group["flux_nu_relative"][:]),
            "c0_reference": jnp.asarray(group["c0_reference"][()]),
        }


def lyu2018_spectrum(
    wavelength_aa: jnp.ndarray,
    agn_log_lbol: jnp.ndarray,
    agn_lyu2018_tau_v: jnp.ndarray,
    templates: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    r"""Interpolate a Lyu family and scale it to its finite-band reference.

    The selected template is linearly interpolated in :math:`\tau_V`, then
    linearly interpolated in :math:`F_\nu` along the native wavelength axis.
    Values outside the template's 0.01–1000 μm support are zero.

    .. math::

       L_\nu(\lambda,\tau_V,\ell) =
       10^\ell L_\odot\,T(\lambda,\tau_V)/C_{0,f}.

    Here :math:`C_{0,f}` is one native-frequency integral of the selected
    family's tau=0 row and is shared by every tau row. Thus ``agn_log_lbol``
    is the tau=0 luminosity reference over the finite template range; it is not
    a bolometric luminosity of the central engine.

    Parameters
    ----------
    wavelength_aa : ndarray
        Rest-frame wavelength coordinates in Å.
    agn_log_lbol : ndarray
        Base-10 logarithm of the finite-template reference luminosity in solar
        luminosities.
    agn_lyu2018_tau_v : ndarray
        Interpolated polar-dust optical depth, between 0 and 10.
    templates : mapping[str, ndarray]
        Runtime arrays from :func:`load_lyu2018_templates`.

    Returns
    -------
    ndarray
        AGN spectral luminosity density in erg s⁻¹ Hz⁻¹ on the requested grid.

    Notes
    -----
    **JIT-compatible**: yes, pure JAX interpolation and scaling. The template
    mapping is a runtime input so exact, wave-precomputed, and spectrum-
    precomputed predictions share the same calculation.
    """
    tau_grid = templates["tau_v"]
    tau = jnp.clip(agn_lyu2018_tau_v, tau_grid[0], tau_grid[-1])
    lower = jnp.clip(jnp.searchsorted(tau_grid, tau, side="right") - 1, 0, tau_grid.size - 2)
    fraction = (tau - tau_grid[lower]) / (tau_grid[lower + 1] - tau_grid[lower])
    family_flux = templates["flux_nu_relative"]
    relative_flux = family_flux[lower] * (1.0 - fraction) + family_flux[lower + 1] * fraction
    flux_on_wave = jnp.interp(
        wavelength_aa,
        templates["wavelength_aa"],
        relative_flux,
        left=0.0,
        right=0.0,
    )
    # Keep the relative template near its tabulated scale before applying the
    # luminosity factor. Dividing by C0 first makes the array tiny and can
    # overflow the reverse-mode intermediate in float32.
    return apply_log10_scale(
        flux_on_wave,
        jnp.asarray(agn_log_lbol) + LOG10_L_SUN - jnp.log10(templates["c0_reference"]),
    )


def _family_spectrum(
    family: str,
    wavelength_aa: jnp.ndarray,
    agn_log_lbol: jnp.ndarray = DEFAULT_AGN_LOG_LBOL,
    agn_lyu2018_tau_v: jnp.ndarray = 0.0,
    *,
    templates: Mapping[str, jnp.ndarray] | None = None,
    **kwargs,
) -> jnp.ndarray:
    """Call the shared kernel for a statically selected family."""
    del kwargs
    if templates is None:
        templates = load_lyu2018_templates(family)
    return lyu2018_spectrum(wavelength_aa, agn_log_lbol, agn_lyu2018_tau_v, templates)


def lyu2018_agn(
    wavelength_aa: jnp.ndarray,
    agn_log_lbol: jnp.ndarray = DEFAULT_AGN_LOG_LBOL,
    agn_lyu2018_tau_v: jnp.ndarray = 0.0,
    *,
    templates: Mapping[str, jnp.ndarray] | None = None,
    **kwargs,
) -> jnp.ndarray:
    """NORMAL-family wrapper for the public Lyu2018 AGN template."""
    return _family_spectrum(
        "norm", wavelength_aa, agn_log_lbol, agn_lyu2018_tau_v, templates=templates, **kwargs
    )


def lyu2018_wdd_agn(
    wavelength_aa: jnp.ndarray,
    agn_log_lbol: jnp.ndarray = DEFAULT_AGN_LOG_LBOL,
    agn_lyu2018_tau_v: jnp.ndarray = 0.0,
    *,
    templates: Mapping[str, jnp.ndarray] | None = None,
    **kwargs,
) -> jnp.ndarray:
    """WDD-family wrapper for the public Lyu2018 AGN template."""
    return _family_spectrum(
        "wdd", wavelength_aa, agn_log_lbol, agn_lyu2018_tau_v, templates=templates, **kwargs
    )


def lyu2018_hdd_agn(
    wavelength_aa: jnp.ndarray,
    agn_log_lbol: jnp.ndarray = DEFAULT_AGN_LOG_LBOL,
    agn_lyu2018_tau_v: jnp.ndarray = 0.0,
    *,
    templates: Mapping[str, jnp.ndarray] | None = None,
    **kwargs,
) -> jnp.ndarray:
    """HDD-family wrapper for the public Lyu2018 AGN template."""
    return _family_spectrum(
        "hdd", wavelength_aa, agn_log_lbol, agn_lyu2018_tau_v, templates=templates, **kwargs
    )

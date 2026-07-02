# SPDX-License-Identifier: BSD-3-Clause
"""Hensley & Draine Astrodust+PAH emission as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._port_base import EmissionPort
from tengri.parameters.priors import Fixed

__all__ = ["AstrodustIRSEDComponent"]


class AstrodustIRSEDComponent(EmissionPort):
    """Hensley & Draine (2023) Astrodust+PAH dust IR emission template.

    Wraps the pure closure from the tabulated Astrodust+PAH template library,
    parameterized identically to Draine & Li (2007): minimum radiation field
    (umin), power-law mixing fraction (gamma), and PAH mass fraction (qpah).

    The model uses the Astrodust grain composition and PAH population from
    Hensley & Draine (2023), updated from the classic Draine & Li (2007)
    models.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    **Template auto-loading**: the closure lazy-loads HDF5 templates on
    first call (at trace time). After lazy loading, all subsequent calls
    are pure JAX.

    References
    ----------
    .. [1] Hensley, B. S. & Draine, B. T., 2023,
       "Astrodust: Optical properties of dust grain mixtures",
       ApJ, 948, 55. https://doi.org/10.3847/1538-4357/acc270

    """

    name: str = "astrodust"

    # Free parameters (user-facing names, prefix-stripped)
    umin = Fixed(1.0)
    gamma_dl = Fixed(0.01)
    qpah = Fixed(3.0)

    _citations_tuple: ClassVar[tuple[str, ...]] = ("hensley_draine2023",)

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute Astrodust+PAH dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "umin", "gamma_dl", "qpah"
            (or subset if some are Fixed).
        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (typically zeros for a dust emission component).
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        L_ir : float
            Total absorbed luminosity in erg/s.

        Returns
        -------
        tuple[ndarray, dict]
            (sed_out, published) where sed_out is the updated SED and published
            contains {"sed_dust_ir": emission SED in erg/s/Hz}.

        """
        from tengri.components.dust.emission import astrodust as ast_fn

        sed = ast_fn(
            wave,
            L_ir,
            dust_umin=p["umin"],
            dust_gamma_dl=p["gamma_dl"],
            dust_qpah=p["qpah"],
        )
        return sed_in + sed, {"sed_dust_ir": sed}

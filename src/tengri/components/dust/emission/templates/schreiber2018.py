# SPDX-License-Identifier: BSD-3-Clause
"""Schreiber et al. (2018) dust emission template as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._port_base import EmissionPort
from tengri.parameters.priors import Fixed

__all__ = ["Schreiber2018IRSEDComponent"]


class Schreiber2018IRSEDComponent(EmissionPort):
    """Schreiber et al. (2018) dust IR emission template.

    Wraps the pure closure from the tabulated Schreiber et al. (2018) template
    library, parameterized by dust temperature (T_dust) and PAH fraction (f_pah).

    The model interpolates linearly in the T_dust grid and linearly mixes
    continuum and PAH components.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    **Template auto-loading**: the closure lazy-loads HDF5 templates on
    first call (at trace time). After lazy loading, all subsequent calls
    are pure JAX.

    References
    ----------
    .. [1] Schreiber, C., Pannella, M., Elbaz, D., et al., 2018,
       "The ALMA Spectroscopic Survey in the Hubble Ultra Deep Field:
       The molecular gas content of galaxies and tension with
       IllustrisTNG and the Santa Cruz Simulations",
       A&A, 609, A30. https://doi.org/10.1051/0004-6361/201731506

    """

    name: str = "schreiber2018"

    # Free parameters (user-facing names, prefix-stripped)
    tdust = Fixed(25.0)
    fpah = Fixed(0.05)

    _citations_tuple: ClassVar[tuple[str, ...]] = ("schreiber2018",)

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute Schreiber et al. (2018) dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "tdust", "fpah"
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
        # schreiber2018 has no top-level closure wrapper (only a lazy-loader entry),
        # so resolve it via the loader dict — same pattern as dale2014_cigale.
        from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS

        schreiber_fn = DUST_EMISSION_MODELS["schreiber2018"]
        sed = schreiber_fn(
            wave,
            L_ir,
            dust_tdust=p["tdust"],
            dust_fpah=p["fpah"],
        )
        return sed_in + sed, {"sed_dust_ir": sed}

# SPDX-License-Identifier: BSD-3-Clause
"""Jones et al. THEMIS dust emission template as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Fixed

__all__ = ["ThemisIRSEDComponent"]


class ThemisIRSEDComponent(SEDModelComponent):
    """Jones et al. (2017) THEMIS/DustEM dust IR emission template.

    Wraps the pure closure from the tabulated THEMIS/DustEM template library.
    Uses the same Draine & Li (2007) mixing formula but with the THEMIS grain
    composition.

    The model mixes single-U (diffuse) and power-law (PDR) components via
    the power-law fraction, with aromatic carbon fraction (qhac) controlling
    PAH-like features. Supports both bilinear interpolation (2D: qhac, umin)
    and trilinear interpolation (3D: qhac, umin, alpha) depending on template
    availability.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    **Template auto-loading**: the closure lazy-loads HDF5 templates on
    first call (at trace time). After lazy loading, all subsequent calls
    are pure JAX.

    References
    ----------
    .. [1] Jones, A. P., Ysard, N., Köhler, M., et al., 2017,
       "The THEMIS model: A review", A&A, 602, A46.
       https://doi.org/10.1051/0004-6361/201628997

    """

    name: str = "themis"
    parameter_prefix: str = "dust_"

    # Free parameters (user-facing names, prefix-stripped)
    umin = Fixed(1.0)
    gamma_dl = Fixed(0.01)
    qhac = Fixed(0.17)
    alpha = Fixed(2.0)

    # Cross-component contract
    optional_inputs: ClassVar[dict[str, str]] = {"L_ir": "erg/s"}
    outputs: ClassVar[dict[str, str]] = {"sed_dust_ir": "erg/s/Hz"}

    _citations_tuple: ClassVar[tuple[str, ...]] = ("jones2017",)

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute THEMIS dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "umin", "gamma_dl", "qhac",
            "alpha" (or subset if some are Fixed).
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
        from tengri.components.dust.emission import themis as themis_fn

        sed = themis_fn(
            wave,
            L_ir,
            dust_umin=p["umin"],
            dust_gamma_dl=p["gamma_dl"],
            dust_qhac=p["qhac"],
            dust_alpha=p["alpha"],
        )
        return sed_in + sed, {"sed_dust_ir": sed}

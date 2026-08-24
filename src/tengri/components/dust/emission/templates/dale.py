# SPDX-License-Identifier: BSD-3-Clause
"""Dale et al. (2014) dust emission template as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.priors import Fixed

__all__ = ["Dale2014CigaleIRSEDComponent", "Dale2014IRSEDComponent"]


class Dale2014IRSEDComponent(EmissionComponent):
    """Dale et al. (2014) dust IR emission template.

    Wraps the pure closure from the tabulated Dale et al. (2014) template
    library, parameterized by dust-to-stellar mass ratio (alpha) and AGN
    power contribution (frac_agn).

    The model interpolates in the 1D alpha grid and optionally mixes a
    pure-AGN QSO template when frac_agn > 0.

    **Embedded star-forming radio continuum**: The Dale+2014 templates include
    a rising star-forming radio synchrotron continuum that extends to 2.2459e9 Å
    (1.335 GHz). When this variant is combined with an active SF radio block
    (``radio.sf.type != 'none'``), the synchrotron is double-counted in rest_sed
    between ~1.34 and ~10 GHz (3–22 cm), and the composed SED steps down ~2x at
    the 1.335 GHz template edge where the embedded tail ends. **Combining
    dale2014 with SF radio raises ConfigError at build time by design; use
    dale2014_cigale instead** (see issue #1970).

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    **Template auto-loading**: the closure lazy-loads HDF5 templates on
    first call (at trace time). After lazy loading, all subsequent calls
    are pure JAX.

    References
    ----------
    .. [1] Dale, D. A., Helou, G., Magdis, G. E., et al., 2014,
       "The calibration of the IRAS and wise infrared luminosity distance
       indicators at z <= 0.3", ApJ, 784, 83.
       https://doi.org/10.1088/0004-637X/784/1/83

    """

    name: str = "dale2014"

    # Free parameters (user-facing names, prefix-stripped)
    alpha_dale = Fixed(2.0)
    frac_agn = Fixed(0.0)

    _citations_tuple: ClassVar[tuple[str, ...]] = ("dale2014",)

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute Dale et al. (2014) dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "alpha_dale", "frac_agn"
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
        from tengri.components.dust.emission import dale2014 as dale_fn

        sed = dale_fn(
            wave,
            L_ir,
            dust_alpha_dale=p["alpha_dale"],
            dust_frac_agn=p["frac_agn"],
        )
        return sed_in + sed, {"sed_dust_ir": sed}


class Dale2014CigaleIRSEDComponent(EmissionComponent):
    """Dale et al. (2014) dust IR emission template (CIGALE variant).

    Wraps the pure closure from the tabulated Dale et al. (2014) template
    library, using the CIGALE variant parameterization (dust-to-stellar mass
    ratio alpha and AGN power contribution frac_agn).

    **Radio tail stripped per CIGALE convention**: Unlike the standard dale2014
    variant, this CIGALE-adapted version has the star-forming radio synchrotron
    continuum removed beyond 7.727e7 Å. This makes it safe to combine with
    an active SF radio block (``radio.sf.type != 'none'``) without double-counting
    the radio continuum. This is the recommended variant when using the radio
    component.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    **Template auto-loading**: the closure lazy-loads HDF5 templates on
    first call (at trace time). After lazy loading, all subsequent calls
    are pure JAX.

    References
    ----------
    .. [1] Dale, D. A., Helou, G., Magdis, G. E., et al., 2014,
       "The calibration of the IRAS and wise infrared luminosity distance
       indicators at z <= 0.3", ApJ, 784, 83.
       https://doi.org/10.1088/0004-637X/784/1/83

    """

    name: str = "dale2014_cigale"

    # Free parameters (user-facing names, prefix-stripped)
    alpha_dale = Fixed(2.0)
    frac_agn = Fixed(0.0)

    _citations_tuple: ClassVar[tuple[str, ...]] = ("dale2014",)

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute Dale et al. (2014) dust emission (CIGALE variant).

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "alpha_dale", "frac_agn"
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
        # Import the lazy-loading wrapper from the registry, which will resolve
        # to the appropriate CIGALE-variant closure on first call.
        from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS

        dale_cigale_fn = DUST_EMISSION_MODELS["dale2014_cigale"]
        sed = dale_cigale_fn(
            wave,
            L_ir,
            dust_alpha_dale=p["alpha_dale"],
            dust_frac_agn=p["frac_agn"],
        )
        return sed_in + sed, {"sed_dust_ir": sed}

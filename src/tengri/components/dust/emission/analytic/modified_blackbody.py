# SPDX-License-Identifier: BSD-3-Clause
"""Modified blackbody dust emission as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.priors import Fixed
from tengri.parameters.resolve import require_redshift

__all__ = ["ModifiedBlackbodyIRSEDComponent"]


class ModifiedBlackbodyIRSEDComponent(EmissionComponent):
    """Optically-thin modified blackbody dust IR emission.

    Wraps the pure closure :func:`~tengri.components.dust.emission.modified_blackbody`,
    which provides a 2-3 parameter emission model (dust temperature, emissivity index,
    and optional energy-balance relaxation factor).

    The unnormalized spectrum is::

        S_nu ~ nu^beta * B_nu(T_dust)

    which is then normalized so that the frequency integral equals ``L_ir``.

    When ``redshift > 0``, the dust temperature is corrected for CMB heating
    (da Cunha et al. 2013) and the observed flux is reduced by the CMB contrast factor.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    References
    ----------
    .. [1] B. T. Draine, "Physics of the Interstellar and Intergalactic Medium"
       (Princeton University Press, 2011). Chapter 22: thermal continuum
       emission from dust grains in the optically-thin limit.
       https://ui.adsabs.harvard.edu/abs/2011piim.book.....D

    .. [2] da Cunha, E., Emerson, D. J., & Ivison, R. J., et al. 2013,
       "On the effect of the cosmic microwave background in high-redshift
       (sub-)millimeter observations", ApJ, 766, 13. arXiv:1302.0844.

    """

    name: str = "modified_blackbody"

    # Free parameters (user-facing names, prefix-stripped)
    T = Fixed(30.0)
    beta_ir = Fixed(1.8)
    epsilon_mbb = Fixed(1.0)

    _citations_tuple: ClassVar[tuple[str, ...]] = (
        "draine2011",
        "da_cunha2013",
    )

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute modified blackbody dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "T", "beta_ir",
            "epsilon_mbb" (or subset if some are Fixed).
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
        from tengri.components.dust.emission import modified_blackbody as mbb_fn

        z = jnp.asarray(
            require_redshift(p, "components.dust.emission.analytic.modified_blackbody.predict")
        )
        sed = mbb_fn(
            wave,
            L_ir,
            dust_T=p["T"],
            dust_beta_ir=p["beta_ir"],
            dust_epsilon_mbb=p["epsilon_mbb"],
            redshift=z,
        )
        return sed_in + sed, {"sed_dust_ir": sed}

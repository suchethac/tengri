# SPDX-License-Identifier: BSD-3-Clause
"""Casey (2012) modified blackbody + mid-IR power law as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._port_base import EmissionPort
from tengri.parameters.priors import Fixed

__all__ = ["Casey2012IRSEDComponent"]


class Casey2012IRSEDComponent(EmissionPort):
    """Casey (2012) modified blackbody + mid-IR power law dust emission.

    Wraps the pure closure :func:`~tengri.components.dust.emission.casey2012`,
    which combines a modified blackbody (FIR peak from cold/warm dust) with
    a mid-IR power law (Wien-side excess from warm dust continuum), joined
    by a smooth sigmoid transition function.

    The implemented model uses the following convention::

        S(ν) = N_pl * ν^α_mid * f(λ)         [mid-IR power law, f→1 at short λ]
             + N_bb * ν^(3+β) / (exp(hν/kT) - 1) * (1 - f(λ))   [FIR MBB, 1-f→1 at long λ]

    where the transition function f(λ) = 1 / (1 + (λ / λ_0)^2) smoothly joins them.

    When ``optically_thin=True`` (static knob), the mid-IR power-law component
    is zeroed, leaving only the modified blackbody.

    When ``redshift > 0``, the dust temperature is corrected for CMB heating
    (da Cunha et al. 2013) and the observed flux is reduced by the CMB contrast
    factor.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    References
    ----------
    .. [1] Casey, C. M., 2012, "Revised Constraints on Dust Emissivity and
       Temperature from Planck and IRAS: a Resolved-source Perspective",
       MNRAS, 425, 3094. arXiv:1206.2926.

    .. [2] da Cunha, E., Emerson, D. J., & Ivison, R. J., et al. 2013,
       "On the effect of the cosmic microwave background in high-redshift
       (sub-)millimeter observations", ApJ, 766, 13. arXiv:1302.0844.

    """

    name: str = "casey2012"

    # Free parameters (user-facing names, prefix-stripped)
    T = Fixed(35.0)
    beta_ir = Fixed(1.8)
    alpha_mir = Fixed(2.0)

    _citations_tuple: ClassVar[tuple[str, ...]] = (
        "casey2012",
        "da_cunha2013",
    )

    # Static knob: zero the mid-IR power-law component if True
    _optically_thin: bool = False

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute Casey (2012) modified blackbody + mid-IR power law emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "T", "beta_ir", "alpha_mir"
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
        from tengri.components.dust.emission import casey2012 as casey_fn

        z = jnp.asarray(p.get("redshift", 0.0))
        sed = casey_fn(
            wave,
            L_ir,
            dust_T=p["T"],
            dust_beta_ir=p["beta_ir"],
            dust_alpha_mir=p["alpha_mir"],
            optically_thin=self._optically_thin,
            redshift=z,
        )
        return sed_in + sed, {"sed_dust_ir": sed}

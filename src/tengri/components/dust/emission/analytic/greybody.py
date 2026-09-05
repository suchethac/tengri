# SPDX-License-Identifier: BSD-3-Clause
"""General-opacity greybody dust emission as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.priors import Fixed
from tengri.parameters.resolve import require_redshift

__all__ = ["GreybodyIRSEDComponent"]


class GreybodyIRSEDComponent(EmissionComponent):
    """General-opacity graybody dust IR emission.

    Wraps the pure closure :func:`~tengri.components.dust.emission.greybody`,
    which provides a parametric graybody model with variable opacity pivot.

    The unnormalized spectrum is::

        S_nu ~ (1 - exp(-(lam_0/lam)^beta)) * B_nu(T_dust)

    which is then normalized so that the frequency integral equals ``L_ir``.

    This is the general-opacity graybody form used by Synthesizer
    (``Greybody(..., optically_thin=False)``) and CIGALE's ``mbb`` module
    (Boquien et al. 2019). For the optionally-thin form, use
    ``modified_blackbody``; for the graybody plus mid-IR power law, use
    ``casey2012``.

    When ``redshift > 0``, the dust temperature is corrected for CMB
    heating (da Cunha et al. 2013) and the observed flux is reduced by
    the CMB contrast factor.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, differentiable everywhere.

    References
    ----------
    .. [1] Boquien, M., Burgarella, D., Roehlly, Y., et al. 2019,
       A&A 622, A103. CIGALE: Code Investigating GALaxy Emission.
       https://doi.org/10.1051/0004-6361/201834156

    .. [2] Casey, C. M., 2012, MNRAS, 425, 3094, Eqs. 1-2, 11-12.
       doi:10.1111/j.1365-2966.2012.21455.x, arXiv:1206.1595.

    .. [3] da Cunha, E., Emerson, D. J., & Ivison, R. J., et al. 2013,
       "On the effect of the cosmic microwave background in high-redshift
       (sub-)millimeter observations", ApJ, 766, 13. arXiv:1302.0844.

    """

    name: str = "greybody"

    # Free parameters (user-facing names, prefix-stripped)
    T = Fixed(35.0)
    beta_ir = Fixed(1.8)
    lambda_0_um = Fixed(200.0)
    epsilon_mbb = Fixed(1.0)

    _citations_tuple: ClassVar[tuple[str, ...]] = (
        "boquien2019",
        "casey2012",
        "dacunha2013",
    )

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute greybody dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "T", "beta_ir",
            "lambda_0_um", "epsilon_mbb" (or subset if some are Fixed).
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
        from tengri.components.dust.emission import greybody as gb_fn

        z = jnp.asarray(require_redshift(p, "components.dust.emission.analytic.greybody.predict"))
        sed = gb_fn(
            wave,
            L_ir,
            dust_T=p["T"],
            dust_beta_ir=p["beta_ir"],
            dust_lambda_0_um=p["lambda_0_um"],
            dust_epsilon_mbb=p["epsilon_mbb"],
            redshift=z,
        )
        return sed_in + sed, {"sed_dust_ir": sed}

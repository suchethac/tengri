# SPDX-License-Identifier: BSD-3-Clause
"""Schreiber et al. (2016) dust emission as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.priors import Fixed
from tengri.parameters.resolve import require_redshift

__all__ = ["Schreiber2016AnalyticIRSEDComponent"]


class Schreiber2016AnalyticIRSEDComponent(EmissionComponent):
    r"""Schreiber et al. (2016) 2-parameter dust emission model (analytic).

    Wraps the pure closure :func:`~tengri.components.dust.emission.schreiber2016`,
    which mixes dust continuum and PAH emission by a fractional parameter.

    The dust continuum is a modified blackbody (modified_blackbody with β=1.5).
    The PAH component is **approximated** as a sum of Drude profiles at standard
    wavelengths (not the full Schreiber+ mid-IR aromatic forest).

    For the CIGALE-faithful tabulated version with the real PAH feature forest,
    select ``schreiber2018`` (``data/schreiber2018_templates.h5``) instead —
    this analytic model is the lightweight, grid-free approximation.

    The model composition is:

    .. math::

        L_\nu = (1 - f_{\rm PAH}) L_\nu^{\rm continuum} + f_{\rm PAH} L_\nu^{\rm PAH}

    where:

    - Dust continuum: modified blackbody with temperature T_dust and emissivity
      index β = 1.5, using the same normalization as ``modified_blackbody``.
    - PAH: sum of Drude profiles at standard rest wavelengths (3.3, 6.2, 7.7,
      8.6, 11.3, 12.7 μm) with relative strengths from Smith et al. (2007).

    The total integral over frequency is normalized to ``L_ir``.

    When ``redshift > 0``, the dust temperature is corrected for CMB heating
    (da Cunha et al. 2013) and the observed flux is reduced by the CMB contrast
    factor.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    **Naming note** (#849): the user-facing parameters are the canonical
    ``dust_T`` and ``dust_f_pah`` (the closure arg is likewise ``dust_f_pah``,
    so the mapping is now the identity). The old spellings ``dust_tdust`` /
    ``dust_fpah`` resolve via ``_LEGACY_PARAM_ALIASES`` with a warning.

    References
    ----------
    .. [1] Schreiber, C., Elbaz, D., Sparre, M., et al., 2016,
       "Universal dust attenuation laws", A&A, 589, A35.
       https://doi.org/10.1051/0004-6361/201527923

    .. [2] Smith, J. D. T., Draine, B. T., Dale, D. A., et al., 2007,
       "The mid-infrared emission of ultraluminous infrared galaxies," ApJ, 656, 770.
       arXiv:astro-ph/0701042. https://doi.org/10.1086/510378

    .. [3] da Cunha, E., Emerson, D. J., & Ivison, R. J., et al. 2013,
       "On the effect of the cosmic microwave background in high-redshift
       (sub-)millimeter observations", ApJ, 766, 13. arXiv:1302.0844.

    """

    name: str = "schreiber2016"

    # Free parameters (user-facing names, prefix-stripped). Canonical (#849):
    # ``dust_T`` + ``dust_f_pah`` (the old ``dust_fpah`` spelling is an alias).
    T = Fixed(30.0)
    f_pah = Fixed(0.05)

    _citations_tuple: ClassVar[tuple[str, ...]] = (
        "schreiber2016",
        "smith2007",
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
        """Compute Schreiber (2016) dust continuum + PAH emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped: keys are "T", "f_pah"
            (or subset if some are Fixed). Note: "f_pah" is mapped to the
            closure's "dust_f_pah" argument.
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
        from tengri.components.dust.emission import schreiber2016 as sch_fn

        z = jnp.asarray(
            require_redshift(p, "components.dust.emission.analytic.schreiber2016.predict")
        )
        # Canonical f_pah maps straight to the closure's dust_f_pah arg (#849)
        sed = sch_fn(
            wave,
            L_ir,
            dust_T=p["T"],
            dust_f_pah=p["f_pah"],
            redshift=z,
        )
        return sed_in + sed, {"sed_dust_ir": sed}

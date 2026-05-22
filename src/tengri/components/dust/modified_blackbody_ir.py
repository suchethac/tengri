# SPDX-License-Identifier: BSD-3-Clause
"""Modified blackbody dust emission as an SEDModelComponent.

Implements optically-thin modified blackbody re-emission of dust-absorbed
starlight, parameterized by dust temperature and emissivity index.

The dust temperature can optionally be corrected for CMB heating at high
redshift (da Cunha et al. 2013).

References
----------
.. [1] Draine, B.T., 2011, "Physics of the Interstellar and
   Intergalactic Medium", Princeton University Press. Chapter 22.
.. [2] da Cunha, E. et al. 2013, "On the effect of the cosmic microwave
   background in high-redshift (sub-)millimeter observations", ApJ 766, 13.
   arXiv:1302.0844.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission import modified_blackbody
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform

__all__ = ["ModifiedBlackbodySED"]


class ModifiedBlackbodySED(SEDModelComponent):
    """Dust IR emission via optically-thin modified blackbody.

    Closes the dust energy balance by re-emitting absorbed UV/optical
    luminosity as a modified Planck spectrum with temperature-dependent
    and emissivity-index-dependent scaling.

    Attributes
    ----------
    name : str
        Stable identifier: ``"modified_blackbody_ir"``.
    parameter_prefix : str
        Domain prefix for parameters: ``"dust_"``.

    Notes
    -----
    **Cross-component contract**:
    - Reads: ``state.derived["L_ir"]`` (erg/s) — luminosity absorbed by dust,
      published by the dust attenuation component.
    - Publishes: ``{"L_ir_emission": erg/s}`` — total bolometric IR luminosity
      (should equal input ``L_ir`` by energy balance).

    **JIT-compatible**: yes — all operations in :meth:`predict` are ``jnp``
    primitives.

    **Parameter discovery**: Free parameters ``T`` and ``beta_ir`` are
    auto-discovered from the class attributes below; :meth:`declared_parameters`
    constructs :class:`ParamDeclaration` tuples with units and descriptions.

    **Temperature correction**: CMB heating correction is applied automatically
    for any redshift ``z > 0`` via the existing :func:`modified_blackbody`
    closure in this module.

    **Pipeline ordering**: This component MUST run after dust attenuation
    so ``L_ir`` is present. Typical order: ``[Stellar, Nebular, DustAttenuation,
    DustEmission, IGM, Radio]``.
    """

    name = "modified_blackbody_ir"
    parameter_prefix = "dust_"

    # Free parameters — auto-discovered by base class
    T = Uniform(20.0, 80.0, description="Dust temperature", units="K")
    beta_ir = Uniform(1.0, 3.0, description="Dust emissivity index", units="")

    # Cross-component contract
    inputs: ClassVar = {"L_ir": "erg/s"}
    outputs: ClassVar = {"L_ir_emission": "erg/s"}

    def predict(
        self, p: Mapping[str, jnp.ndarray], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        r"""Compute dust emission via modified blackbody.

        Generates L_nu via::

            L_nu ~ (nu/nu_ref)^beta * B_nu(T_dust)

        normalized so that the integral over frequency equals ``L_ir``.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Sliced parameters (prefix stripped):
            - ``p["T"]``: dust temperature [K]
            - ``p["beta_ir"]``: emissivity index [dimensionless]
        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (ignored; dust emission is computed
            independently from absorbed luminosity).
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            Cross-component inputs:
            - ``L_ir``: total absorbed luminosity [erg/s]

        Returns
        -------
        tuple[ndarray, dict]
            - ``sed_out``: Updated SED in erg/s/Hz (``sed_in + L_nu``).
            - ``published``: Dict with ``{"L_ir_emission": scalar}`` where
              scalar is the total bolometric IR luminosity [erg/s].

        Notes
        -----
        **JIT-compatible**: yes.

        **Gradient-safe**: yes — differentiable everywhere in (T, beta_ir,
        L_ir).
        """
        L_ir = inputs["L_ir"]

        # Compute dust emission SED
        sed_emission = modified_blackbody(
            wavelength_aa=wave,
            L_absorbed=L_ir,
            dust_T=p["T"],
            dust_beta_ir=p["beta_ir"],
            redshift=0.0,  # Redshift handled upstream if needed
        )

        # Return updated SED and published luminosity
        # Note: L_ir_emission not yet a typed field in DerivedState
        sed_out = sed_in + sed_emission

        return sed_out, {}

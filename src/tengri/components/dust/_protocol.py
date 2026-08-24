# SPDX-License-Identifier: BSD-3-Clause
"""Protocol interfaces for dust models in tengri.

This module defines the structural types (Protocols) that dust attenuation
laws and dust emission templates must implement. Contributors adding new
dust models should ensure their implementations match these signatures.
"""

from typing import Protocol

import jax.numpy as jnp

__all__ = ["DustAttenuationLaw", "DustEmissionTemplate"]


class DustAttenuationLaw(Protocol):
    """Protocol for dust attenuation curves.

    Any dust law function must accept a wavelength array and optional
    parameters, returning the normalized attenuation curve k(λ).

    All attenuation curves are normalized such that k(5500 Angstrom) = 1.0
    (or for some parameterized laws, k(V) is determined by the parameters).

    Examples
    --------
    Implementing a simple power-law dust law::

        def my_powerlaw_dust(wavelength, n_slope=-0.7, **kwargs):
            '''My custom dust law.'''
            return (wavelength / 5500.0) ** n_slope

    Registering it in the dust law registry::

        from tengri.components.dust.attenuation import register_dust_law


        @register_dust_law("my_powerlaw")
        def my_powerlaw_dust(wavelength, n_slope=-0.7, **kwargs):
            return (wavelength / 5500.0) ** n_slope

    Using it in a model::

        from tengri.components.dust.attenuation import resolve_dust_law

        law = resolve_dust_law("my_powerlaw")
        k = law(wavelengths, n_slope=-0.9)

    """

    def __call__(
        self,
        wavelength: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """Compute the dust attenuation curve.

        Parameters
        ----------
        wavelength: jax.Array, shape (n_wave,)
            Rest-frame wavelength grid [Å].
        **kwargs
            Law-specific parameters (e.g., dust_bump_strength, dust_delta,
            dust_Rv, n_slope, redshift) [dimensionless]. Unknown kwargs must
            be accepted but may be ignored via **_kwargs.

        Returns
        -------
        jax.Array, shape (n_wave,)
            Normalized attenuation curve k(λ) [dimensionless], typically with
            k(5500 Å) = 1.0. Values in [0, 1] for physical attenuation curves.
            Returned array dtype matches input wavelength dtype.

        References
        ----------
        .. [1] See :mod:`tengri.components.dust.attenuation` for specific
               dust law citations (Calzetti, Gordon, SMC, etc.).

        Notes
        -----
        **JIT-compatible**: yes (required for all dust laws).

        **Gradient-safe**: yes (required for likelihood evaluation and inference).

        """
        ...


class DustEmissionTemplate(Protocol):
    """Protocol for dust emission templates.

    Dust emission models take the absorbed bolometric luminosity and
    wavelength grid, then return the IR SED in erg/s/Hz.

    The emission must satisfy energy balance: the integrated emission
    over all wavelengths equals the absorbed energy from the attenuation
    step, ensuring flux conservation.

    Examples
    --------
    Implementing a modified blackbody emission model::

        def my_mbb_emission(wavelength, L_absorbed, T_dust=25.0, **kwargs):
            '''Optically-thin modified blackbody.'''
            from tengri.utils.physics_constants import H_PLANCK, K_BOLTZ, C_CGS

            # Planck function, scale by L_absorbed
            ...

    Registering it::

        from tengri.components.dust.emission import register_emission_model


        @register_emission_model("my_mbb")
        def my_mbb_emission(wavelength, L_absorbed, T_dust=25.0, **kwargs): ...

    Using it::

        from tengri.components.dust.emission import the DUST_EMISSION_MODELS loader cache

        model = DUST_EMISSION_MODELS["my_mbb"]
        l_nu = model(wavelengths, L_absorbed=1e10, T_dust=30.0)

    """

    def __call__(
        self,
        wavelength: jnp.ndarray,
        L_absorbed: float,
        **kwargs,
    ) -> jnp.ndarray:
        """Compute the dust emission SED.

        Parameters
        ----------
        wavelength: jax.Array, shape (n_wave,)
            Rest-frame wavelength grid [Å].
        L_absorbed: float
            Total bolometric luminosity absorbed by dust [L_sun].
            This is computed from the attenuation step as the integral
            of (1 - transmission) × stellar_luminosity over wavelength.
        **kwargs
            Model-specific parameters (e.g., T_dust, dust_umin, dust_gamma,
            dust_qpah) [various units]. Unknown kwargs must be accepted but
            may be ignored.

        Returns
        -------
        jax.Array, shape (n_wave,)
            Dust emission SED L_ν [erg/s/Hz] (rest-frame).
            Must satisfy energy conservation: ∫ L_ν dν ≈ L_absorbed [L_sun]
            when integrated over the full spectral range.

        References
        ----------
        .. [1] See :mod:`tengri.components.dust.emission` for specific
               dust emission model citations (Dale, Draine & Li, Casey, etc.).

        Notes
        -----
        **JIT-compatible**: yes (required for all dust emission models).

        **Gradient-safe**: yes (required for likelihood evaluation and inference).

        """
        ...

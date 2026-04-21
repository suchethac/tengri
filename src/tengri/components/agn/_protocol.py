"""Protocol interfaces for AGN models in tengri.

This module defines the structural types (Protocols) that AGN emission
models must implement. Contributors adding new AGN physics (accretion
discs, tori, BLR, NLR) should ensure their implementations match these
signatures.
"""

from typing import Protocol

import jax.numpy as jnp

__all__ = ["AGNModel"]


class AGNModel(Protocol):
    """Protocol for AGN emission models.

    Any AGN model function must accept a wavelength array and bolometric
    luminosity, returning the rest-frame SED in erg/s/Hz.

    AGN models are composed of physical components (disc, torus, BLR, NLR,
    X-ray) that may be added, and each component follows this interface.

    The AGN bolometric luminosity is always specified as log10(L_bol / L_sun)
    at the model input level; the implementation converts to cgs (erg/s)
    internally as needed.

    Examples
    --------
    Implementing a power-law accretion disc emission model::

        def my_powerlaw_disc(wavelength, agn_log_lbol, alpha=-0.5, **kwargs):
            '''Simple power-law disc SED.'''
            L_bol_ergs = 10.0**agn_log_lbol * L_SUN_ERG  # convert to erg/s
            # f_ν ∝ ν^α, convert wavelength to frequency, normalize
            ...

    Registering it::

        from tengri.components.agn.unified import register_agn_model


        @register_agn_model("my_powerlaw_disc")
        def my_powerlaw_disc(wavelength, agn_log_lbol, alpha=-0.5, **kwargs): ...

    Using it::

        from tengri.components.agn.unified import resolve_agn_model

        model = resolve_agn_model("my_powerlaw_disc")
        sed = model(wavelengths, agn_log_lbol=44.0, alpha=-0.4)
    """

    def __call__(
        self,
        wavelength: jnp.ndarray,
        agn_log_lbol: float,
        **kwargs,
    ) -> jnp.ndarray:
        """Compute the AGN emission SED.

        Parameters
        ----------
        wavelength : jax.Array, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        agn_log_lbol : float
            AGN bolometric luminosity as log10(L_bol / L_sun).
            Always in solar luminosities at the API level.
            Implementation must convert to erg/s (1 L_sun ≈ 3.839e33 erg/s)
            if needed for physical calculations.
        **kwargs
            Model-specific parameters passed through from the unified AGN
            combiner or intermediate models. Examples include:
            - ``agn_torus_frac``: covering factor [0, 1]
            - ``agn_torus_angle``: viewing angle (degrees)
            - ``agn_bh_spin``: black hole spin [0, 1]
            - Temperature parameters, opacity indices, etc.
            Unknown kwargs must be accepted but may be ignored.

        Returns
        -------
        jax.Array, shape (n_wave,)
            AGN emission SED L_ν in erg/s/Hz (rest-frame).
            All returned SEDs are absolute luminosities (not normalized).
            Multiple model components (disc, torus, etc.) can be added
            directly at the wavelength array level.
        """
        ...

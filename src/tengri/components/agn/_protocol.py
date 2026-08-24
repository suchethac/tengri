# SPDX-License-Identifier: BSD-3-Clause
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

    Defines the interface that all AGN emission models must implement. A model
    accepts a wavelength grid and bolometric luminosity, and returns the
    rest-frame SED in erg/s/Hz. Models can be composed of physical components
    (accretion disc, torus, broad-line region, narrow-line region, X-ray
    corona) that are computed independently and added.

    The AGN bolometric luminosity is specified as log10(L_bol / L_sun) at the
    API level; implementations must convert to CGS (erg/s) as needed for
    physical calculations.

    Notes
    -----
    See :mod:`tengri.components.agn.unified` for model registration and lookup
    utilities.

    Examples
    --------
    Implementing a custom power-law accretion disc model::

        def my_powerlaw_disc(wavelength, agn_log_lbol, alpha=-0.5, **kwargs):
            \"\"\"Simple power-law accretion disc SED.\"\"\"
            L_bol_erg_s = 10.0**agn_log_lbol * L_SUN_ERG  # convert to erg/s
            # Compute f_ν ∝ ν^α on the wavelength grid
            ...
            return sed  # shape (n_wave,), [erg/s/Hz]

    Registering the model::

        from tengri.components.agn.unified import register_agn_model


        @register_agn_model("my_powerlaw_disc")
        def my_powerlaw_disc(wavelength, agn_log_lbol, alpha=-0.5, **kwargs): ...

    Using it in SEDModel::

        from tengri.components.agn.unified import resolve_agn_model

        model = resolve_agn_model("my_powerlaw_disc")
        sed = model(wavelengths, agn_log_lbol=11.0, alpha=-0.4)
        # ``agn_log_lbol`` is log10(L_bol / L_sun); 11 ⇒ L_bol ≈ 4e44 erg/s.
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
        wavelength : array_like, shape (n_wave,)
            Rest-frame wavelength grid. [Angstrom]
        agn_log_lbol : float
            AGN bolometric luminosity. [log10(L_sun)]
        **kwargs
            Model-specific keyword arguments passed from the unified AGN
            interface or parent components. Common parameters include:

            - ``agn_torus_frac``: Torus covering factor. [dimensionless, 0–1]
            - ``agn_torus_angle``: Viewing angle to torus axis. [degrees, 0–90]
            - ``agn_bh_spin``: Black hole spin parameter. [dimensionless, 0–1]
            - Temperature, opacity, radial structure parameters (model-specific)

            Unknown kwargs must be accepted but may be silently ignored.

        Returns
        -------
        ndarray, shape (n_wave,)
            AGN emission SED L_ν. [erg/s/Hz]

        References
        ----------
        .. [1] See :mod:`tengri.components.agn` module documentation for citations
               to specific AGN physics implementations (disc, torus, NLR, BLR, X-ray).

        Notes
        -----
        **Absolute luminosities**: The returned SED is in absolute units
        [erg/s/Hz], not normalized or flux-density units. Multiple components
        (disc, torus, BLR, NLR, X-ray) can be summed at the wavelength array
        level.

        **JIT-compatible**: All implementations must use ``jnp`` primitives and
        avoid Python-level branching on traced values to ensure
        ``jax.jit`` compatibility.
        """
        ...

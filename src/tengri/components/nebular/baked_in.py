# SPDX-License-Identifier: BSD-3-Clause
"""Baked-in nebular emission backend.

When SSP files already include nebular emission (wNE files),
this backend is a no-op: it returns zero additional contribution.
The nebular emission is already part of the SSP flux arrays.
"""

import warnings

import jax.numpy as jnp


class BakedInNebularWarning(UserWarning):
    """Warning raised when BakedInBackend is used with SSPs.

    Indicates that nebular emission is baked into the SSP file at fixed logU
    and escape fraction. These are NOT free parameters; fitting nebular
    properties requires switching to a different backend (CueBackend or
    CloudyGridBackend).

    Notes
    -----
    The SSP file's nebular assumptions (typically logU = -3) are determined
    when the SSP grid was generated and cannot be changed by the model.

    """


class BakedInBackend:
    """Nebular backend for SSP files with pre-included emission.

    A no-op backend that returns zero additional nebular flux because the SSP
    templates already contain nebular emission at fixed ionization parameter
    and escape fraction. This is the default backend when no CLOUDY grid is
    specified.

    Parameters
    ----------
    ionizing_source_warning: str, optional
        Verbosity control for the fixed-nebular limitation warning. One of
        'raise' (raise ValueError), 'warn' (emit UserWarning), or 'suppress'
        (silent). Default: 'warn'.

    Attributes
    ----------
    has_continuum: bool
        True: continuum is baked into the SSP.
    has_free_params: bool
        False: ionization parameter and escape fraction are fixed.
    name: str
        Identifier "baked_in".

    Notes
    -----
    **JIT-compatible**: yes, predict_nebular_sed and
    predict_nebular_line_fluxes return zero arrays.

    To fit nebular properties, switch to CloudyGridBackend or CueBackend
    which provide free parameters for ionization parameter, escape fraction,
    and metallicity.

    """

    def __init__(self, ionizing_source_warning: str = "warn") -> None:
        """Initialize BakedInBackend.

        Parameters
        ----------
        ionizing_source_warning: str, optional
            Verbosity control. Default: "warn".

        """
        self.name = "baked_in"
        self.has_free_params = False
        self.has_continuum = True
        if ionizing_source_warning not in ("raise", "warn", "suppress"):
            raise ValueError("ionizing_source_warning must be 'raise', 'warn', or 'suppress'")
        if ionizing_source_warning != "suppress":
            msg = (
                "BakedInBackend: nebular emission is baked into the SSP file at a "
                "FIXED logU and FIXED escape fraction determined when the SSP grid "
                "was generated (commonly logU = −3, but depends on the SSP file). "
                "The ionization parameter and escape fraction are NOT free parameters "
                ": varying neb_logU or neb_fesc in your Parameters will have no "
                "effect. Check your SSP file's nebular assumptions. Switch to "
                "CloudyGridBackend or CueBackend to vary nebular properties. "
                "To suppress when building via SEDModel.build: "
                "warnings.filterwarnings('ignore', "
                "message='BakedInBackend: nebular emission is baked'). "
                "(ionizing_source_warning='suppress' also works, but it is a "
                "BakedInBackend(...) constructor argument and the build grammar "
                "does not forward it.)"
            )
            if ionizing_source_warning == "raise":
                raise ValueError(msg)
            warnings.warn(msg, BakedInNebularWarning, stacklevel=2)

    def predict_nebular_sed(
        self,
        ssp_weights: jnp.ndarray,
        ssp_wave: jnp.ndarray,
        log_z: float,
        **neb_params,
    ) -> jnp.ndarray:
        """Return zero nebular contribution (already in SSP).

        Parameters
        ----------
        ssp_weights: array, shape (n_age,) or (n_met, n_age)
            CSP mass weights (unused).
        ssp_wave: array, shape (n_wave,)
            Wavelength grid [Angstrom].
        log_z: float
            Stellar metallicity (unused) [log10(Z)].
        **neb_params
            Additional nebular parameters (all unused).

        Returns
        -------
        array, shape (n_wave,)
            Zero array: nebular emission is baked into the SSP [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes, returns jnp.zeros_like.

        """
        return jnp.zeros_like(ssp_wave)

    def predict_nebular_line_fluxes(
        self,
        ssp_weights: jnp.ndarray,
        log_z: float,
        **neb_params,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return empty line arrays.

        Parameters
        ----------
        ssp_weights: array, shape (n_age,) or (n_met, n_age)
            CSP mass weights (unused).
        log_z: float
            Stellar metallicity (unused) [log10(Z)].
        **neb_params
            Additional nebular parameters (all unused).

        Returns
        -------
        wavelengths: array, shape (0,)
            Empty array [Angstrom].
        luminosities: array, shape (0,)
            Empty array [erg/s].

        References
        ----------
        Nebular emission is pre-calculated in the SSP templates when using
        this backend; see the SSP file documentation for assumptions about
        ionization parameter and escape fraction.

        Notes
        -----
        **JIT-compatible**: yes, returns empty jnp arrays.

        """
        return jnp.array([]), jnp.array([])

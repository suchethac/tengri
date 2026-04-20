"""Baked-in nebular emission backend.

When SSP files already include nebular emission (wNE files),
this backend is a no-op: it returns zero additional contribution.
The nebular emission is already part of the SSP flux arrays.
"""

import warnings

import jax.numpy as jnp


class BakedInNebularWarning(UserWarning):
    """Warning raised when BakedInBackend is used with fixed nebular emission.

    The ionization parameter and escape fraction are NOT free parameters.
    Fitting nebular emission properties requires switching to a different backend.
    """


class BakedInBackend:
    """Nebular backend for SSP files with pre-included emission.

    This is the default backend when no CLOUDY grid is specified.
    It adds zero additional nebular flux, since the SSP templates
    already contain nebular emission at fixed logU and logZ.

    Parameters
    ----------
    ionizing_source_warning : str
        One of ``'raise'``, ``'warn'``, or ``'suppress'``. Controls how the
        fixed-nebular-emission limitation is communicated. Default ``'warn'``.
    """

    def __init__(self, ionizing_source_warning: str = "warn") -> None:
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
                "— varying neb_logU or neb_fesc in your Parameters will have no "
                "effect. Check your SSP file's nebular assumptions. Switch to "
                "CloudyGridBackend or CueBackend to vary nebular properties. "
                "To suppress: pass ionizing_source_warning='suppress'."
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

        Returns
        -------
        array, shape (n_wave,)
            Zero array — nebular emission is baked into the SSP.
        """
        return jnp.zeros_like(ssp_wave)

    def predict_nebular_line_fluxes(
        self,
        ssp_weights: jnp.ndarray,
        log_z: float,
        **neb_params,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return empty line arrays.

        Returns
        -------
        wavelengths : array, shape (0,)
        luminosities : array, shape (0,)
        """
        return jnp.array([]), jnp.array([])

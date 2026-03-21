"""Baked-in nebular emission backend.

When SSP files already include nebular emission (wNE files),
this backend is a no-op: it returns zero additional contribution.
The nebular emission is already part of the SSP flux arrays.
"""

import jax.numpy as jnp


class BakedInBackend:
    """Nebular backend for SSP files with pre-included emission.

    This is the default backend when no CLOUDY grid is specified.
    It adds zero additional nebular flux, since the SSP templates
    already contain nebular emission at fixed logU and logZ.
    """

    def __init__(self) -> None:
        self.name = "baked_in"
        self.has_free_params = False

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

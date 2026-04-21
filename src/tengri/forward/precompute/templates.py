"""Generic template-based precompute helpers.

Thin wrappers over :mod:`tengri.forward.precompute.grid`
(:func:`~tengri.forward.precompute.grid.preintegrate_grid` +
:func:`~tengri.forward.precompute.grid.interp_nd_triweight`) that handle common
adapter boilerplate: L_λ → L_ν unit conversion, energy-normalization for
templates scaled by L_absorbed / L_bol at runtime, and a standard JIT lookup
closure.

Component-specific adapters (``components/dust/dust_emission_precompute.py``,
``components/agn/skirtor_precompute.py``, etc.) should call these functions
rather than talk to ``preintegrate_grid`` directly, so their shapes remain
consistent under the Precompute Protocol.
"""

from __future__ import annotations

import jax
import numpy as np

from tengri.forward.precompute.grid import (
    PreintegratedGrid,
    interp_nd_triweight,
    preintegrate_grid,
)
from tengri.utils.physics_constants import AA_TO_CM as _AA_TO_CM, C_CGS as _C_CGS


def precompute_template_photometry(
    templates: np.ndarray,
    wave_rest: np.ndarray,
    filter_waves: list,
    filter_trans: list,
    axes: tuple[np.ndarray, ...],
    redshift: float = 0.0,
    dl_cm: float = 1.0,
    energy_normalize: bool = True,
    units: str = "lnu",
) -> PreintegratedGrid:
    """Preintegrate any template grid through photometric filters.

    Generic entry point for template-based components. Handles unit
    conversion (L_λ → L_ν if needed) and delegates to
    :func:`~tengri.forward.precompute.grid.preintegrate_grid`.

    Parameters
    ----------
    templates : ndarray
        Shape ``(*grid_dims, n_wave)``.  Template spectra.
    wave_rest : ndarray
        Shape ``(n_wave,)``.  Rest-frame wavelengths [Ångström].
    filter_waves : list[ndarray]
        Per-filter wavelength arrays (observed frame).
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    axes : tuple[ndarray, ...]
        One array per grid dimension (for triweight interpolation).
    redshift : float
        Source redshift (0 for rest-frame templates).
    dl_cm : float
        Luminosity distance [cm] (1 for normalized templates).
    energy_normalize : bool
        Normalize each template to unit bolometric luminosity before
        integration.  Required for templates scaled by L_absorbed or
        L_bol at runtime (DL07, Dale, SKIRTOR, etc.).  Default True.
    units : str
        ``"lnu"`` if templates are in L_ν [erg/s/Hz], or ``"llam"``
        if templates are in L_λ [erg/s/Å].  When ``"llam"``, converts
        to L_ν via L_ν = L_λ × λ²/c before integration.

    Returns
    -------
    PreintegratedGrid
        Precomputed filter-integrated photometry with triweight axes/edges.
    """
    templates = np.asarray(templates)
    wave_rest = np.asarray(wave_rest)

    if units == "llam":
        wave_cm = wave_rest * _AA_TO_CM
        templates = templates * (wave_cm**2) / _C_CGS

    return preintegrate_grid(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=dl_cm,
        axes=tuple(np.asarray(ax) for ax in axes),
        energy_normalize=energy_normalize,
    )


def build_template_photometry_lookup(preint: PreintegratedGrid):
    """Build a JIT-compiled lookup from a preintegrated template grid.

    Uses triweight interpolation for C²-continuous gradients.  The
    returned function takes the grid parameters + a scalar scaling
    factor and returns photometry in (n_filters,).

    Parameters
    ----------
    preint : PreintegratedGrid
        Output of :func:`precompute_template_photometry`.

    Returns
    -------
    callable
        ``(scale, *grid_params) -> array (n_filters,)`` where *grid_params*
        are scalar query points along each axis.
    """
    phot = preint.phot
    axes = preint.axes
    edges = preint.edges

    @jax.jit
    def lookup(scale, *grid_params):
        """Interpolate template photometry at given grid parameters and scale by luminosity.

        Parameters
        ----------
        scale : float
            Luminosity scaling factor (L_absorbed, L_bol, etc.).
        *grid_params : float
            Per-axis query points for triweight interpolation.

        Returns
        -------
        array, shape (n_filters,)
            Photometry in filter bands.
        """
        normed = interp_nd_triweight(phot, axes, edges, grid_params)
        return scale * normed

    return lookup

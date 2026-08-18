# SPDX-License-Identifier: BSD-3-Clause
"""Fiber aperture-fraction utilities for spatially-resolved spectroscopy.

When a fiber spectrograph observes a galaxy at redshift z with a fiber of
angular radius :math:`\\theta_{\\rm fib}`, only the fraction of the
galaxy's total light inside the fiber footprint is captured. Classical
SED-fitting codes typically scale the spectrum by a single number to
match the photometry, implicitly assuming the galaxy is uniform across
the aperture (the flat-slab approximation). Tengri's spatial sub-model
makes this aperture fraction explicit: integrate the
:class:`tengri.protocols.ForwardState`-published surface-brightness
profile inside the fiber mask and divide by the integral over the
whole grid.

This module provides:

- :func:`arcsec_to_kpc`: convert angular size to physical size at
  redshift z via the angular diameter distance.
- :func:`circular_aperture_mask`: build a 0/1 mask on the spatial
  grid, optionally with a small sigmoidal edge softening for
  differentiability.
- :func:`aperture_fraction`: the fraction of an unnormalized 2D
  profile that lies inside a given (kpc) radius from the origin.

All functions are pure JAX. The aperture-fraction integral uses a
sigmoid-softened mask by default so the result is differentiable in
the aperture radius — useful when the aperture is a calibration
parameter, or when the user wants to marginalize over fiber-placement
uncertainty.

This module is for *forward-model* aperture fractions (Sérsic ×
fiber → spectrum scaling). For catalog-side photometry aperture
corrections (input flux preprocessing) see
:mod:`tengri.observation.aperture`.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.utils.cosmology import angular_diameter_distance

__all__ = [
    "aperture_fraction",
    "arcsec_to_kpc",
    "circular_aperture_mask",
]


_RAD_PER_ARCSEC = jnp.pi / 180.0 / 3600.0
_KPC_PER_CM = 1.0 / 3.0857e21


def arcsec_to_kpc(
    arcsec: jnp.ndarray | float,
    z: jnp.ndarray | float,
) -> jnp.ndarray:
    """Convert an angular size in arcsec to a physical size in kpc at redshift z.

    Uses the angular diameter distance:

    .. math::

        d_{\\rm physical} = \\theta \\cdot D_A(z)

    Parameters
    ----------
    arcsec : float or ndarray
        Angular size in arcseconds.
    z : float or ndarray
        Source redshift.

    Returns
    -------
    ndarray
        Physical size in kiloparsecs.

    Notes
    -----
    JIT-compatible.
    """
    da_cm = jnp.asarray(angular_diameter_distance(z))
    return jnp.asarray(arcsec) * _RAD_PER_ARCSEC * da_cm * _KPC_PER_CM


def circular_aperture_mask(
    grid_kpc: tuple[jnp.ndarray, jnp.ndarray],
    radius_kpc: jnp.ndarray | float,
    center_kpc: tuple[float, float] = (0.0, 0.0),
    softness: float = 0.01,
) -> jnp.ndarray:
    """Build a 2D mask for a circular aperture on the spatial grid.

    Parameters
    ----------
    grid_kpc : tuple of (ndarray, ndarray), each shape (ny, nx)
        ``(x_grid, y_grid)`` — physical-coordinate grids from a
        :class:`SpatialModel`.
    radius_kpc : float or ndarray
        Aperture radius in kpc.
    center_kpc : tuple of (float, float), default (0.0, 0.0)
        Aperture center in kpc — useful when the fiber is offset
        from the galaxy nucleus.
    softness : float, default 0.01
        Sigmoidal edge softening as a fraction of ``radius_kpc``.
        Set to ``0.0`` for a hard top-hat (non-differentiable at
        the edge).

    Returns
    -------
    ndarray, shape (ny, nx)
        Mask values in [0, 1]. ~1 inside the aperture, ~0 outside,
        with a soft transition spanning ``radius_kpc * softness``
        for ``softness > 0``.

    Notes
    -----
    JIT/grad-compatible (when ``softness > 0``).
    """
    import jax

    x, y = grid_kpc
    cx, cy = center_kpc
    r = jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    if softness <= 0.0:
        return (r <= radius_kpc).astype(r.dtype)
    return jax.nn.sigmoid((radius_kpc - r) / (radius_kpc * softness))


def aperture_fraction(
    profile_2d: jnp.ndarray,
    grid_kpc: tuple[jnp.ndarray, jnp.ndarray],
    radius_kpc: jnp.ndarray | float,
    center_kpc: tuple[float, float] = (0.0, 0.0),
    softness: float = 0.01,
) -> jnp.ndarray:
    """Fraction of an unnormalized 2D profile inside a circular aperture.

    .. math::

        f_{\\rm ap} = \\frac{\\int_{\\rm aperture} I(x, y) \\, dx \\, dy}
                          {\\int I(x, y) \\, dx \\, dy}

    Computes the ratio of the integrated profile inside the mask to
    the total integral. The result is dimensionless and in [0, 1].

    Parameters
    ----------
    profile_2d : ndarray, shape (ny, nx)
        Unnormalized surface-brightness profile (e.g.
        ``state.derived["spatial_profile_2d"]``).
    grid_kpc : tuple of (ndarray, ndarray), each shape (ny, nx)
        Physical-coordinate grids.
    radius_kpc : float or ndarray
        Aperture radius in kpc.
    center_kpc : tuple of (float, float), default (0.0, 0.0)
        Aperture center.
    softness : float, default 0.01
        See :func:`circular_aperture_mask`.

    Returns
    -------
    ndarray, shape ()
        Scalar aperture fraction.

    Notes
    -----
    JIT/grad/vmap-compatible. Pixel area cancels in the ratio so this
    function does not need an explicit `dx, dy` argument.
    """
    mask = circular_aperture_mask(grid_kpc, radius_kpc, center_kpc, softness)
    return (profile_2d * mask).sum() / profile_2d.sum()

# SPDX-License-Identifier: BSD-3-Clause
"""Publication grid for AGN published luminosities (#2745).

A published bolometric luminosity is the integral of the SED the component emits.
Integrating on the caller's ``wave`` makes the value depend on how the caller samples
wavelength, so the AGN classes evaluate the emitted SED on this fixed grid instead and
integrate there. The grid is log-uniform in wavelength over the whole emission range of
the AGN templates and disc models, and contains any diagnostic wavelength passed in
``extra_aa`` exactly, so a node value is the emitted value at that wavelength.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tengri.components.agn._phys import wavelength_to_nu

__all__ = [
    "PUBLICATION_HI_AA",
    "PUBLICATION_LO_AA",
    "PUBLICATION_NODES",
    "emitted_bolometric",
    "publication_wave",
]

#: Lower edge of the publication grid [Angstrom] (X-ray corona of KD18 reaches below 10 A).
PUBLICATION_LO_AA: float = 1.0e-3

#: Upper edge of the publication grid [Angstrom] (far infrared, 1e10 A = 1 m).
PUBLICATION_HI_AA: float = 1.0e10

#: Number of log-uniform nodes. The trapezoid in ln(nu) of the emitted SED agrees with a
#: 200000-node reference to better than 3e-7 for every AGN template class at this count;
#: at 4000 nodes the template classes miss 1e-6.
PUBLICATION_NODES: int = 8000


def publication_wave(dtype, extra_aa=()) -> jnp.ndarray:
    """Fixed publication wavelength grid, ascending, with diagnostic nodes included.

    Parameters
    ----------
    dtype : dtype
        Floating dtype of the returned array (the caller's wavelength dtype).
    extra_aa : sequence of float, optional
        Diagnostic wavelengths [Angstrom] that must be grid nodes exactly.

    Returns
    -------
    ndarray, shape (n_pub,)
        Ascending wavelengths [Angstrom], independent of any caller grid.
    """
    nodes = np.geomspace(PUBLICATION_LO_AA, PUBLICATION_HI_AA, PUBLICATION_NODES)
    nodes = np.unique(np.concatenate([nodes, np.asarray(extra_aa, dtype=np.float64)]))
    return jnp.asarray(nodes, dtype=dtype)


def emitted_bolometric(sed_pub, wave_pub) -> jnp.ndarray:
    r"""Bolometric luminosity of an emitted SED on the publication grid.

    .. math::

        L = \int L_\nu \, \mathrm{d}\nu = \int L_\nu(\nu)\,\nu \,\mathrm{d}\ln\nu

    evaluated by the trapezoid rule in :math:`\ln\nu` on the nodes of ``wave_pub``.

    Parameters
    ----------
    sed_pub : array_like, shape (n_pub,)
        Emitted spectral luminosity density on ``wave_pub`` [erg/s/Hz].
    wave_pub : array_like, shape (n_pub,)
        Ascending wavelengths of the publication grid [Angstrom].

    Returns
    -------
    ndarray, shape ()
        Integrated luminosity over the whole grid [erg/s].

    Notes
    -----
    **JIT/grad-safe**: yes. ``nu`` is descending on an ascending wavelength grid, so the
    trapezoid is negated, following the repository's convention.
    """
    nu = wavelength_to_nu(wave_pub)
    return -jnp.trapezoid(jnp.asarray(sed_pub) * nu, jnp.log(nu))

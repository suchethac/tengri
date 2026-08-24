# SPDX-License-Identifier: BSD-3-Clause
r"""Prévot+1984 SMC disc reddening, the single source for ``agn_ebv_disc``.

Both the monolithic AGN forward models (:mod:`tengri.components.agn.unified`)
and the composable runner (:mod:`tengri.components.agn.blocks.runner`) apply
disc obscuration through this one function, so ``agn_ebv_disc`` behaves
identically on every path (reddening unification). Keeping it in a low-level
module (it depends only on :func:`prevot_smc`) avoids the ``runner → unified``
import cycle.

Note this is **distinct** from the polar-dust extinction (``agn_polar_ebv``),
which uses the SMC curve at ``R_V = 2.93`` and is applied only for Type-1 lines
of sight: see ``blocks/runner.py``'s ``_disc_ext``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.dust.attenuation import prevot_smc

__all__ = ["redden_disc"]

#: Prévot et al. 1984 SMC ratio of total-to-selective extinction. The standard
#: AGN-disc obscuration prescription (AGNfitter ``BBBred_Prevot``).
_R_V_PREVOT_SMC: float = 2.72


def redden_disc(wavelength: Array, l_disc: Array, agn_ebv_disc: Array) -> Array:
    r"""Apply Prévot+1984 SMC extinction (:math:`R_V = 2.72`) to a disc SED.

    A no-op when ``agn_ebv_disc = 0`` (multiplies by :math:`10^0 = 1`).

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    l_disc: array_like, shape (n_wave,)
        Un-reddened disc SED [erg/s/Hz] or [erg/s/Å], the multiplicative
        extinction factor is dimensionless, so either convention passes through.
    agn_ebv_disc: float
        Disc color excess :math:`E(B-V)` [mag].

    Returns
    -------
    ndarray, shape (n_wave,)
        Reddened disc SED, same units as ``l_disc``.

    Notes
    -----
    JIT/grad/vmap-safe (``prevot_smc`` ramps smoothly through the X-ray region).

    .. math::

        L_{\rm red}(\lambda) = L(\lambda)\,
            10^{-0.4\, k(\lambda)\, R_V\, E(B-V)}

    with :math:`k(\lambda) = A(\lambda)/A(V)` (tengri's ``k(V) = 1`` convention)
    and :math:`R_V = 2.72`.

    References
    ----------
    .. [1] P. Prévot et al., "The galactic interstellar extinction law in the
       Small Magellanic Cloud," A&A, 132, 389 (1984).
    """
    k = prevot_smc(wavelength)
    return l_disc * jnp.power(10.0, -0.4 * k * _R_V_PREVOT_SMC * agn_ebv_disc)

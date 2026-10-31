# SPDX-License-Identifier: BSD-3-Clause
"""Milky Way foreground screen: one transmission used on every projection path.

The screen is a dust screen in front of the whole source, so it multiplies the
observed-frame SED after IGM absorption and redshifting. Every projection
(observed SED, photometry, spectroscopy) evaluates it through
:func:`foreground_transmission`, so the same E(B-V) and law reach all of them.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.dust.attenuation import cardelli, fitzpatrick99_unnormalized

#: Laws usable as a foreground screen, keyed by the ``foreground={'law': ...}`` name.
#: Each takes ``(wave_obs, dust_Rv=...)`` and returns A(lambda)/A(V). Fitzpatrick (1999)
#: enters in its paper convention (A(V) = R_V E(B-V) from the spline, no renormalization),
#: so the transmission matches the tabulated A(lambda)/E(B-V). The registered dust law
#: ``fitzpatrick99`` is the same curve normalized to k(5500 A) = 1; both come from one
#: implementation.
FOREGROUND_LAWS = {"cardelli": cardelli, "fitzpatrick99": fitzpatrick99_unnormalized}


def foreground_transmission(wave_obs, ebmv_mw, rv, law):
    r"""Observed-frame transmission of the Milky Way foreground screen.

    Parameters
    ----------
    wave_obs : array_like, shape (n_wave,)
        Observed-frame wavelength [Angstrom].
    ebmv_mw : float or array_like
        Foreground reddening E(B-V) [mag]. A traced value is allowed.
    rv : float or array_like
        Total-to-selective extinction ratio R_V [dimensionless].
    law : str
        Key of :data:`FOREGROUND_LAWS`.

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission :math:`T(\lambda) = 10^{-0.4\,A_V\,k(\lambda)}` [dimensionless],
        with :math:`A_V = R_V\,E(B-V)` and :math:`k(\lambda) = A(\lambda)/A(V)`.
        Equal to one at ``ebmv_mw = 0``.

    Raises
    ------
    ValueError
        If ``law`` is not a key of :data:`FOREGROUND_LAWS`.

    Notes
    -----
    **JIT-compatible**: yes; ``law`` is a static Python string and selects the
    curve at trace time. **Gradient-safe**: yes, with respect to ``ebmv_mw`` and
    ``rv``. **Vmap-safe**: yes over ``ebmv_mw``.
    """
    if law not in FOREGROUND_LAWS:
        raise ValueError(f"Unknown foreground law {law!r}. Valid: {sorted(FOREGROUND_LAWS)}.")
    k_lambda = FOREGROUND_LAWS[law](wave_obs, dust_Rv=rv)
    a_v = rv * ebmv_mw
    return jnp.power(10.0, -0.4 * a_v * k_lambda)


def foreground_rest_transmission(wave_rest, z, ebmv_mw, rv, law):
    r"""Foreground transmission sampled on a rest-frame grid at redshift ``z``.

    Parameters
    ----------
    wave_rest : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    z : float or array_like
        Source redshift [dimensionless].
    ebmv_mw, rv, law
        As in :func:`foreground_transmission`.

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission at ``wave_rest * (1 + z)`` [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes. The rest grid is shared with the SED it multiplies,
    as the IGM transmission is.
    """
    return foreground_transmission(jnp.asarray(wave_rest) * (1.0 + z), ebmv_mw, rv, law)


def combine_observed_transmission(igm_trans, foreground_trans):
    """Product of the IGM and foreground transmissions on one grid, or the present one.

    Parameters
    ----------
    igm_trans : ndarray, shape (n_wave,), or None
        IGM transmission on the rest grid; ``None`` when IGM is absent.
    foreground_trans : ndarray, shape (n_wave,), or None
        Foreground screen on the rest grid; ``None`` when no screen is configured.

    Returns
    -------
    ndarray, shape (n_wave,), or None
        ``None`` only when both inputs are ``None``.

    Notes
    -----
    **JIT-compatible**: yes; ``None`` is a structural check on the Python side.
    """
    if foreground_trans is None:
        return igm_trans
    if igm_trans is None:
        return foreground_trans
    return igm_trans * foreground_trans

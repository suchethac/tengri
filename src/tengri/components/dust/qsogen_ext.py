# SPDX-License-Identifier: BSD-3-Clause
r"""Temple+2021 empirical quasar extinction curve (qsogen ``pl_ext_comp_03``).

This is the reddening law the qsogen SED code (Temple, Hewett & Banerji 2021
[1]_) applies to the quasar continuum: an *empirically-derived quasar*
extinction curve, distinct from the SMC Prevot law AGNfitter uses for its
``EBVbbb``. It is stored and applied in qsogen's own convention:

.. math::

    A_\lambda = E(B-V)\,\bigl[\,E(\lambda-V)/E(B-V) + R\,\bigr],
    \qquad R = 3.1,

where the tabulated curve is the *color excess* :math:`E(\lambda-V)/E(B-V)`
(zero at V = 5500 Å) rather than :math:`A_\lambda/E(B-V)` directly. Contrast
AGNfitter's ``BBBred_Prevot``, which stores an analytic SMC fit already in the
:math:`A_\lambda/E(B-V)` convention. See
``src/tengri/data/agn_ext/PROVENANCE.md`` for the curve's origin.

References
----------
.. [1] M. J. Temple, P. C. Hewett & M. Banerji, MNRAS, 508, 737 (2021).
   arXiv:2109.04472. https://doi.org/10.1093/mnras/stab2586
"""

from __future__ import annotations

from importlib.resources import files

import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = ["QSOGEN_EXT_R", "qsogen_quasar_extinction"]

#: qsogen's default total-to-selective ratio ``R = A_V/E(B-V)``
#: (``qsosed.redden_spectrum(R=3.1)``). Applied on top of the tabulated
#: ``E(λ-V)/E(B-V)`` curve to recover ``A_λ/E(B-V) = curve + R``.
QSOGEN_EXT_R: float = 3.1


def _load_curve() -> tuple[np.ndarray, np.ndarray]:
    """Load the Temple+2021 quasar extinction curve at import time.

    Returns ``(wavelength[Å], E(λ-V)/E(B-V))`` ascending in wavelength.
    """
    path = files("tengri.data.agn_ext") / "qsogen_quasar_ext.dat"
    with path.open("r") as fh:
        arr = np.loadtxt(fh)
    wave_aa = np.asarray(arr[:, 0], dtype=np.float64)
    excess = np.asarray(arr[:, 1], dtype=np.float64)
    order = np.argsort(wave_aa)
    return wave_aa[order], excess[order]


_WAVE_AA, _EXCESS = _load_curve()
_WAVE_JNP = jnp.asarray(_WAVE_AA)
_EXCESS_JNP = jnp.asarray(_EXCESS)
_WAVE_MIN = float(_WAVE_AA[0])
_WAVE_MAX = float(_WAVE_AA[-1])


def qsogen_quasar_extinction(wavelength: Array, R: float = QSOGEN_EXT_R) -> Array:
    r"""``A_λ/E(B-V)`` for the Temple+2021 quasar extinction curve.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    R : float, optional
        Total-to-selective ratio :math:`R = A_V/E(B-V)`. Default ``3.1``
        (qsogen's ``redden_spectrum`` default).

    Returns
    -------
    ndarray, shape (n_wave,)
        :math:`A_\lambda/E(B-V) = E(\lambda-V)/E(B-V) + R` [mag/mag] inside the
        tabulated domain (500–60000 Å), and ``0`` outside it: the curve is
        undefined there, so no extinction is applied (this protects a full
        radio-to-X-ray AGN grid from unphysical extrapolated attenuation;
        qsogen's own model grid stays within the domain).

    Notes
    -----
    **JIT/grad/vmap-safe**: yes, ``jnp.interp`` plus a ``jnp.where`` mask.

    The tabulated curve is :math:`E(\lambda-V)/E(B-V)` (qsogen's convention,
    zero at V), so ``R`` is added to obtain :math:`A_\lambda/E(B-V)`. Reddening
    a disc SED then multiplies by :math:`10^{-0.4\,A_\lambda}` at the desired
    :math:`E(B-V)`.
    """
    wave = jnp.asarray(wavelength)
    excess = jnp.interp(wave, _WAVE_JNP, _EXCESS_JNP)
    a_over_ebv = excess + R
    in_domain = (wave >= _WAVE_MIN) & (wave <= _WAVE_MAX)
    return jnp.where(in_domain, a_over_ebv, 0.0)

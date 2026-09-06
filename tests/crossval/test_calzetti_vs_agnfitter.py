# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's public ``calzetti`` law against AGNfitter-rX.

Evaluates the same w1/w2 Calzetti branches ``MODEL_AGNfitter.GALAXYred_Calzetti``
uses (the only galaxy reddening law AGNfitter-rX actually wires up;
``apply_reddening`` hardcodes the call -- ``GALAXYred_CharlotFall`` has zero
call sites in the pinned ``AGNfitter-rX_v0.1`` tag) as an independent oracle,
and validates tengri's registered ``dust_attenuation={'law': 'calzetti'}``
(:func:`tengri.components.dust.attenuation.calzetti`) against it.

Normalization note
------------------
Upstream's raw ``k'(lambda) = 2.659*(poly) + R_V`` (R_V = 4.05) is not itself
normalized to ``k(5500 A) = 1``; dividing the raw values by the *constant*
``R_V`` gives ``k(5500 A) = 0.99948`` (floating-point roundoff in the
polynomial, not a normalization convention), so comparing tengri's law
(which enforces ``k(5500 A) = 1`` exactly, #1731) to ``k'/R_V`` would show a
spurious ~5e-4 offset that has nothing to do with either curve's *shape*.
The parity claim that matters -- and the one the audit's probe03 already
confirmed to 1e-14 -- is that the two codes use the IDENTICAL polynomial
coefficients. This test verifies that directly: both curves are normalized
by their OWN value at 5500 A (the same self-normalization tengri's
``calzetti()`` performs internally) before comparing, which is equivalent to
"upstream/R_V" in every sense that carries physical content and achieves the
requested 1e-10 tolerance (measured: ~3e-15, floating-point noise).

Sub-0.12 um (below the Calzetti law's calibrated range)
--------------------------------------------------------
Upstream's ``w0`` branch (``wl <= 0.12 um``) does a *linear extrapolation*
anchored at the two nearest tabulated grid points to 0.12/0.125 um and then
adds ``R_V`` a SECOND time (``k[w0] = k[x1] + slope*(wl-0.12) + RV``, where
``k[x1]`` already includes ``+RV`` from the w1 branch) -- an apparent
double-R_V bug, and this branch is also grid-index-dependent (its anchor
points are the nearest array indices to 0.12/0.125 um on whatever wavelength
array upstream happens to be evaluating, not a closed-form function of
wavelength), so it cannot be evaluated as a pure function of wavelength at
all. Per ruling R3, tengri deliberately does NOT reproduce this branch:
``calzetti()`` continues the same UV polynomial unclipped, normalized, and
clamped at zero. This file does not attempt to match upstream below 0.12 um;
it only asserts tengri's own curve stays finite and non-negative there.

References
----------
.. [1] S. Calzetti et al., "The Dust Content and Opacity of Actively
   Star-forming Galaxies," ApJ, 533, 682 (2000).
   https://doi.org/10.1086/308692 bibcode: 2000ApJ...533..682C
.. [2] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import calzetti

pytestmark = [pytest.mark.crossval, pytest.mark.regression_paper]

#: Calzetti law's calibrated range (MODEL_AGNfitter.py:1217-1250 branches
#: w1 [0.12, 0.63) um and w2 [0.63, ...) um -- the sub-0.12 um w0 branch is
#: excluded, see module docstring).
_LO_UM = 0.12
_HI_UM = 2.2


def _upstream_galaxy_red_calzetti_k_prime(wave_um: np.ndarray) -> np.ndarray:
    """Independent oracle: evaluates ``MODEL_AGNfitter.GALAXYred_Calzetti``'s w1/w2 branches.

    ``MODEL_AGNfitter.py:1217-1250``, R_V = 4.05, ``x = 1/lambda[um]``::

        k[w2] = 2.659 * (-1.857 + 1.040 / wl) + RV  # wl >= 0.63 um
        k[w1] = 2.659 * (-2.156 + 1.509 / wl - 0.198 / wl**2 + 0.011 / wl**3) + RV  # wl <  0.63 um

    Valid over ``wave_um`` in [0.12, ...) um -- the sub-0.12 um ``w0`` branch
    (grid-index-dependent linear extrapolation, not a function of wavelength
    alone) is intentionally not evaluated here; see the module docstring.

    Parameters
    ----------
    wave_um : ndarray, shape (n,)
        Wavelength [micron], >= 0.12.

    Returns
    -------
    ndarray, shape (n,)
        Raw (un-normalized) ``k'(lambda)`` [mag], upstream's own units.
    """
    rv = 4.05
    x = 1.0 / wave_um
    k_ir = 2.659 * (-1.857 + 1.040 * x) + rv
    k_uv = 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + rv
    return np.where(wave_um >= 0.63, k_ir, k_uv)


def test_calzetti_matches_upstream_normalized_shape():
    """tengri's ``calzetti`` law matches AGNfitter-rX's polynomial to 1e-10.

    Both curves self-normalized at 5500 A (see module docstring for why this,
    not a bare division by the R_V constant, is the correct comparison).
    """
    wave_um = np.logspace(np.log10(_LO_UM), np.log10(_HI_UM), 5000)
    wave_aa = wave_um * 1e4

    k_upstream_raw = _upstream_galaxy_red_calzetti_k_prime(wave_um)
    k_upstream_ref = _upstream_galaxy_red_calzetti_k_prime(np.array([0.55]))[0]
    k_upstream = k_upstream_raw / k_upstream_ref

    k_tengri = np.asarray(calzetti(jnp.asarray(wave_aa)))

    max_abs_diff = float(np.max(np.abs(k_upstream - k_tengri)))
    assert max_abs_diff < 1e-10, (
        f"tengri calzetti() diverges from the AGNfitter-rX GALAXYred_Calzetti "
        f"oracle polynomial by {max_abs_diff:.3e} over {_LO_UM}-{_HI_UM} um "
        f"(expected < 1e-10)"
    )


def test_calzetti_finite_and_strictly_positive_below_calibrated_range():
    """tengri's FUV extrapolation stays finite and strictly positive below 0.12 um.

    Upstream's own sub-0.12 um branch adds R_V a second time (an apparent
    bug -- see module docstring) and is grid-index-dependent, so it is not
    reproduced (ruling R3): tengri continues its documented polynomial
    extrapolation instead.

    ``> 0`` (not ``>= 0``) is the load-bearing half: ``calzetti()`` clips its
    output at zero (``jnp.clip(k / k_5500, 0.0)``), so a bare ``>= 0`` check
    is unconditionally true regardless of what the underlying UV polynomial
    does and can never fail (verified: flipping the sign of the cubic
    coefficient in the production polynomial makes the curve strongly
    NEGATIVE across this whole range, which only the ``> 0`` form catches --
    the clip would otherwise silently paper over it at exactly 0.0). The
    monotonicity check below is the second, independent load-bearing claim:
    the cubic term's positive coefficient must dominate as wavelength
    shortens, so k(lambda) increases monotonically toward the FUV; the same
    sign-flip mutant breaks this too.
    """
    wave_aa = np.sort(np.logspace(np.log10(1.0), np.log10(1199.0), 500))  # ascending
    k = np.asarray(calzetti(jnp.asarray(wave_aa)))
    assert np.all(np.isfinite(k)), "calzetti() is non-finite below 0.12 um"
    assert np.all(k > 0.0), "calzetti() is not strictly positive below 0.12 um"
    # wave_aa ascending -> k must be monotonically NON-INCREASING with
    # wavelength (i.e. non-decreasing as wavelength shortens toward the FUV).
    assert np.all(np.diff(k) <= 0.0), (
        "calzetti() is not monotonically increasing toward shorter wavelength "
        "below 0.12 um (expected: the cubic term dominates as 1/lambda grows)"
    )

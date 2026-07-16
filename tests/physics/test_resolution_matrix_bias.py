# SPDX-License-Identifier: BSD-3-Clause
"""A Gaussian LSF biases line shape when the true LSF is asymmetric (#1163).

The #1163 justification: DESI/PFS ship an explicit resolution matrix because the
instrument LSF is not exactly Gaussian. When the true profile is skewed, the
Gaussian ``apply_lsf`` cannot follow it — it conserves integrated flux but
misplaces the line *centroid*, and centroids are what pin kinematics and
redshift in a spectral fit. This test quantifies that bias on a synthetic line
using a deliberately skewed banded resolution matrix (no external data needed).

The banded operator subsuming the exact Gaussian is proved separately in
``tests/components/spectroscopy/test_banded_operator.py``; here we show the
*converse* — where a Gaussian is not enough.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import tengri  # noqa: F401
from tengri.observation.banded import BandedMatrix, banded_matvec, gaussian_resolution_bands
from tengri.observation.spectrum import apply_lsf

pytestmark = pytest.mark.limit

_C_KM_S = 299792.458
_HALPHA = 6564.61  # vacuum [Angstrom]


def _skewed_bands(wave, resolution, n_diag=21, skew=1.8):
    """Gaussian bands with an inflated red wing — an asymmetry a symmetric
    Gaussian LSF cannot represent, re-normalized so flux is still conserved."""
    bm = gaussian_resolution_bands(jnp.asarray(wave), resolution, n_diag)
    off = np.asarray(bm.offsets)
    data = np.asarray(bm.data).copy()
    data = data * np.where(off[:, None] > 0, skew, 1.0)  # inflate the red wing
    data /= data.sum(axis=0, keepdims=True)  # keep each output pixel normalized
    return BandedMatrix(offsets=jnp.asarray(off), data=jnp.asarray(data))


def _centroid(wave, y):
    return float(np.sum(wave * y) / np.sum(y))


def test_gaussian_lsf_biases_line_centroid_vs_true_R():
    n = 600
    wave = np.geomspace(_HALPHA - 65.0, _HALPHA + 55.0, n)
    line = np.exp(-0.5 * ((wave - _HALPHA) / 1.2) ** 2)  # narrow emission line

    bm = _skewed_bands(wave, 3000.0)
    y_true = np.asarray(banded_matvec(bm.offsets, bm.data, jnp.asarray(line)))
    y_gauss = np.asarray(
        apply_lsf(jnp.asarray(line), jnp.asarray(wave), 3000.0, sigma_lib_kms=0.0)
    )

    # Both operators conserve integrated flux (the skew is a re-weighting, not a
    # gain): the Gaussian is not "wrong" in flux — it is wrong in *shape*.
    np.testing.assert_allclose(y_true.sum(), y_gauss.sum(), rtol=2e-3)

    dcen_ang = _centroid(wave, y_true) - _centroid(wave, y_gauss)
    dcen_kms = dcen_ang / _HALPHA * _C_KM_S
    assert abs(dcen_ang) > 0.05, (
        f"skewed true-R vs Gaussian LSF centroid bias too small to be meaningful: "
        f"{dcen_ang:.4f} A ({dcen_kms:.2f} km/s)"
    )
    # A several-km/s systematic on a line centroid propagates straight into a
    # velocity-dispersion / redshift fit — the reason #1163 wants the real R.
    assert abs(dcen_kms) > 3.0, f"centroid bias {dcen_kms:.2f} km/s below the km/s scale"

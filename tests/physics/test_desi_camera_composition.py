# SPDX-License-Identifier: BSD-3-Clause
"""Each DESI camera's photon budget closes within that camera (#1183).

DESI delivers three cameras whose wavelength coverage overlaps, each with its
own resolution operator (Guy et al. 2023 [1]_). Concatenating the camera grids
in camera order makes the joint operator block diagonal, and the physical
content of "block diagonal" is a conservation statement: **a camera's line
spread function redistributes photons only within that camera**. Camera b's
reddest pixels must not borrow flux from camera r, even though the two cameras
observe the same wavelengths there.

Delivered resolution rows are normalized (including the band-truncated rows at
each camera edge), so a flat spectrum must come through unchanged. A
composition that leaked across a seam would push the seam pixels *above* unity
by collecting weight from the neighboring camera — so the conservation
statement is what detects the leak. Every assertion below is paired with a
deliberately-leaky control, because "flux is conserved" is also what a test
that measures nothing reports.

Scope: this exercises the delivered-matrix code path on a DESI-*format* file
generated at test time. It does **not** close #1183's remaining item, a physics
test on a genuine DESI spectrum, which still needs a delivered file.

References
----------
.. [1] Guy, J. et al. 2023, "The Spectroscopic Data Processing Pipeline for the
       Dark Energy Spectroscopic Instrument", AJ, 165, 144, arXiv:2209.14482,
       DOI 10.3847/1538-3881/acb212.
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri  # noqa: F401  (enables float64)
from tengri.observation.banded import BandedMatrix, banded_matvec
from tests._desi_fixture import write_desi_coadd

pytestmark = pytest.mark.conservation


@pytest.fixture
def desi_setup(tmp_path):
    """Per-camera spectra plus the block-diagonal operator over their grids."""
    pytest.importorskip("astropy")
    from tengri.io import desi_spectroscopy, read_desi_cameras

    path = tmp_path / "coadd-physics.fits"
    built = write_desi_coadd(path, n_pix=64, n_diag=7, sigma_pix=1.3)
    cameras = read_desi_cameras(path)
    return cameras, desi_spectroscopy(cameras).resolution_matrix, built


def _leaky(operator, n_total):
    """The same bands with their cross-boundary entries restored.

    The control for every conservation claim here: if the composition did not
    mask the seam, this is what it would look like.
    """
    offsets = np.asarray(operator.offsets)
    data = np.asarray(operator.data).copy()
    for k, offset in enumerate(offsets):
        columns = np.arange(n_total) + offset
        inside = (columns >= 0) & (columns < n_total)
        data[k] = np.where(inside & (data[k] == 0.0), 0.25, data[k])
    return BandedMatrix(offsets=operator.offsets, data=data)


def test_flat_spectrum_is_conserved_across_every_camera(desi_setup):
    """Normalized rows carry a constant through unchanged — seams included."""
    cameras, operator, built = desi_setup
    n_total = built["n_pix"] * len(cameras)

    out = np.asarray(banded_matvec(operator.offsets, operator.data, np.ones(n_total)))
    np.testing.assert_allclose(out, 1.0, rtol=1e-12, atol=1e-12)

    # Teeth: a seam leak over-recovers, so the assertion above is discriminating.
    leaked = np.asarray(banded_matvec(*_leaky(operator, n_total), np.ones(n_total)))
    seam = built["n_pix"] - 1
    assert leaked[seam] > 1.0 + 1e-6


def test_composition_equals_each_camera_operating_alone(desi_setup):
    """The joint operator restricted to a camera *is* that camera's operator."""
    from tengri.io.desi import desi_resolution_offsets
    from tengri.observation.banded import resolution_bands_from_desi

    cameras, operator, built = desi_setup
    n_pix = built["n_pix"]
    n_total = n_pix * len(cameras)

    rng = np.random.default_rng(4)
    x = rng.standard_normal(n_total)
    joint = np.asarray(banded_matvec(operator.offsets, operator.data, x))

    for index, camera in enumerate(cameras):
        block = resolution_bands_from_desi(
            np.asarray(camera.resolution),
            desi_resolution_offsets(built["n_diag"]),
        )
        alone = np.asarray(
            banded_matvec(block.offsets, block.data, x[index * n_pix : (index + 1) * n_pix])
        )
        segment = joint[index * n_pix : (index + 1) * n_pix]
        # float64 eps only: the joint operator sums the same bands in a
        # different offset order (desispec stores them descending).
        np.testing.assert_allclose(segment, alone, rtol=1e-12, atol=1e-12)

    # Teeth: the leaky control breaks the equality precisely at the seam.
    leaked = np.asarray(banded_matvec(*_leaky(operator, n_total), x))
    assert not np.allclose(leaked[:n_pix], joint[:n_pix], rtol=1e-12, atol=1e-12)


def test_muting_one_camera_leaves_the_others_bit_identical(desi_setup):
    """No photon crosses a camera boundary in either direction."""
    cameras, operator, built = desi_setup
    n_pix = built["n_pix"]
    n_total = n_pix * len(cameras)

    rng = np.random.default_rng(5)
    x = rng.standard_normal(n_total)
    full = np.asarray(banded_matvec(operator.offsets, operator.data, x))

    muted_input = x.copy()
    muted_input[:n_pix] = 0.0
    muted = np.asarray(banded_matvec(operator.offsets, operator.data, muted_input))

    assert np.max(np.abs(muted[:n_pix])) == 0.0
    np.testing.assert_array_equal(muted[n_pix:], full[n_pix:])


def test_operator_reaches_project_spectrum(desi_setup):
    """The loader's Spectroscopy is fit-ready: its matrix drives the projector.

    A peaked emission profile is the probe, because convolving a peak with a
    normalized kernel must *lower* the peak by a finite amount — unlike a flat
    model, where an applied and an unapplied operator agree to float64 eps and
    the check would pass without measuring anything.
    """
    import jax.numpy as jnp

    from tengri.observation.spectrum import project_spectrum

    cameras, operator, built = desi_setup
    wave_obs = np.concatenate([camera.wave for camera in cameras])
    redshift, dl_cm = 0.1, 1.0e27

    wave_rest = np.linspace(1000.0, 12000.0, 6000)
    line_center, line_sigma = 4000.0, 90.0
    sed_rest = 1e28 * (1.0 + 4.0 * np.exp(-0.5 * ((wave_rest - line_center) / line_sigma) ** 2))

    def project(matrix):
        return np.asarray(
            project_spectrum(
                jnp.asarray(sed_rest),
                jnp.asarray(wave_rest),
                jnp.asarray(wave_obs),
                redshift,
                dl_cm,
                resolution_matrix=matrix,
            )
        )

    smoothed = project(operator)
    unsmoothed = project(None)

    peak = int(np.argmax(unsmoothed))
    assert peak < built["n_pix"], "the probe line must land in camera b"
    # Convolution with a normalized kernel lowers a peak; require a real margin,
    # not the eps-level difference a no-op would leave.
    assert smoothed[peak] < unsmoothed[peak] * (1.0 - 1e-3)

    # Cameras r and z see only flat continuum, so they must come through
    # untouched — the line broadening in camera b does not reach them.
    beyond_b = slice(built["n_pix"], None)
    np.testing.assert_allclose(smoothed[beyond_b], unsmoothed[beyond_b], rtol=1e-10, atol=0.0)

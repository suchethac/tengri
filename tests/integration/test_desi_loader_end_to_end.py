# SPDX-License-Identifier: BSD-3-Clause
"""A DESI coadd becomes a fit-ready model (#1183).

The unit and physics tests pin the loader and the block-diagonal operator in
isolation. This closes the loop: a DESI-format file goes through
``read_desi_cameras`` -> ``desi_spectroscopy`` -> ``SEDModel.build`` ->
``predict_spectrum``, and the flux/error vectors ``read_desi`` returns line up
with the prediction pixel for pixel.

That alignment is the whole contract. The grid is the camera-order
concatenation, the resolution operator is block diagonal over that same
concatenation, and ``Spectroscopy.__post_init__`` refuses any operator whose
width disagrees — so if the loader ever went back to sorting the cameras, or
returned a different pixel count than it built the operator for, this fails.

Needs an SSP grid; skips without one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._desi_fixture import write_desi_coadd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ssp_path() -> Path | None:
    """Any committed SSP grid — the test is about wiring, not a specific grid."""
    candidates = sorted((REPO_ROOT / "data").glob("ssp_*.h5"))
    return candidates[0] if candidates else None


@pytest.fixture(scope="module")
def ssp_data():
    path = _ssp_path()
    if path is None:
        pytest.skip(f"no SSP grid in {REPO_ROOT / 'data'} (data/ssp_*.h5)")
    from tengri import load_ssp_data

    return load_ssp_data(str(path))


@pytest.fixture
def desi_model(tmp_path, ssp_data):
    """A model whose observation came entirely out of a DESI-format file."""
    pytest.importorskip("astropy")
    from tengri import FIXED, Fixed, Observation, SEDModel
    from tengri.io import desi_spectroscopy, read_desi, read_desi_cameras

    path = tmp_path / "coadd-e2e.fits"
    built = write_desi_coadd(path, n_pix=48, n_diag=7)

    cameras = read_desi_cameras(path)
    spectrum = read_desi(path)
    spectroscopy = desi_spectroscopy(cameras)

    model = SEDModel.build(
        ssp_data=ssp_data,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": FIXED},
        dust_attenuation={"type": "single_component", "law": "calzetti", "all_params": FIXED},
        redshift=Fixed(0.1),
    )
    return model, spectroscopy, spectrum, built


def test_predicts_on_the_loaded_grid(desi_model):
    import jax

    model, spectroscopy, _spectrum, built = desi_model

    n_total = built["n_pix"] * len(built["cameras"])
    assert spectroscopy.n_pixels == n_total
    assert spectroscopy.has_resolution_matrix

    params = model.spec.sample(jax.random.PRNGKey(0))
    predicted = np.asarray(model.predict_spectrum(params))

    assert predicted.shape == (n_total,)
    assert np.all(np.isfinite(predicted))


def test_read_desi_arrays_align_with_the_prediction(desi_model):
    """The data vector and the model vector must index the same pixels."""
    import jax

    model, _, spectrum, built = desi_model
    params = model.spec.sample(jax.random.PRNGKey(0))
    predicted = np.asarray(model.predict_spectrum(params))

    wave, flux, flux_err, meta = spectrum
    assert wave.shape == predicted.shape
    assert flux.shape == predicted.shape
    assert flux_err.shape == predicted.shape

    # And the grid really is camera order, not sorted -- the two differ,
    # because the fixture's cameras overlap the way DESI's do.
    assert not np.allclose(wave, np.sort(wave))
    assert meta["n_pix_per_camera"] == (built["n_pix"],) * len(built["cameras"])


def test_resolution_matrix_changes_the_prediction(desi_model):
    """The operator is not decorative: dropping it changes the spectrum."""
    import dataclasses

    import jax

    model, spectroscopy, _spectrum, _built = desi_model
    params = model.spec.sample(jax.random.PRNGKey(0))
    with_matrix = np.asarray(model.predict_spectrum(params))

    from tengri import FIXED, Fixed, Observation, SEDModel

    bare = dataclasses.replace(spectroscopy, resolution_matrix=None, resolution=None)
    bare_model = SEDModel.build(
        ssp_data=model.ssp_data,
        observation=Observation(spectroscopy=bare),
        sfh={"type": "dpl", "all_params": FIXED},
        dust_attenuation={"type": "single_component", "law": "calzetti", "all_params": FIXED},
        redshift=Fixed(0.1),
    )
    without = np.asarray(bare_model.predict_spectrum(params))

    assert without.shape == with_matrix.shape
    # A stellar spectrum has real structure at these pixels, so a normalized
    # LSF must move it measurably somewhere.
    assert np.max(np.abs(with_matrix - without)) > 0.0

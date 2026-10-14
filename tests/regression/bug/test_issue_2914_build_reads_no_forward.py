# SPDX-License-Identifier: BSD-3-Clause
"""#2914: build-time reads of fixed values run no forward model.

Two build-time helpers ran the whole forward model eagerly (outside ``jax.jit``, so
every primitive compiled on its own, below the persistent cache's floor) only to read
a value that never depends on the parameters:

* ``_snap_to_nebular_catalog`` read ``state.derived["line_waves"]``: the nebular
  backend's static catalog, which ``_published_line_wavelengths_static`` already
  returns bit-for-bit (#2239);
* ``precompute_nebular_grid`` read ``state.wave``: ``SEDModel._rest_wavelength``,
  which ``predict_state`` puts in ``ForwardState.wave`` and no component rewrites.

Both now read the value directly. These tests pin that the direct reads equal what the
forward model published.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
from tests._data_skip import CUE_WEIGHTS, DATA_DIR, requires_cue_weights

pytestmark = pytest.mark.regression_bug

_SSP_PATH = DATA_DIR / "fsps_prsc_miles_chabrier.h5"

#: Within the 0.5 A tolerance of vacuum H-alpha (6564.61) and [OIII] 5007 (5008.24),
#: and one far from every catalog line.
_TARGETS = np.array([6564.5, 5008.0, 4000.0])


@pytest.fixture(scope="module")
def _cue_fixture_available():
    if not _SSP_PATH.is_file() or not CUE_WEIGHTS.is_file():
        pytest.skip(f"needs {_SSP_PATH} and {CUE_WEIGHTS}")


def _build(full_catalog: bool):
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return SEDModel.build(
        ssp_data=load_ssp_data(str(_SSP_PATH)),
        observation=Observation(photometry=Photometry.from_names(["sdss_g"])),
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "full_catalog": full_catalog},
        redshift=Fixed(0.1),
    )


def _snap_against(catalog, targets, tol_aa=0.5):
    """``_snap_to_nebular_catalog``'s rule, applied to an explicit catalog."""
    diffs = np.abs(targets[:, None] - catalog[None, :])
    nearest = catalog[np.argmin(diffs, axis=1)]
    return np.where(diffs.min(axis=1) <= tol_aa, nearest, targets)


def _forward_raises(self, *args, **kwargs):
    raise AssertionError("a build-time read of a fixed value ran the forward model")


@requires_cue_weights
@pytest.mark.parametrize("full_catalog", [True, False])
def test_snap_to_nebular_catalog_runs_no_forward(_cue_fixture_available, full_catalog):
    from tengri.forward.sed_model import _snap_to_nebular_catalog

    model = _build(full_catalog)
    published = np.asarray(model.predict_state({}).derived["line_waves"])
    expected = _snap_against(published, _TARGETS)

    with mock.patch.object(SEDModel, "predict_state", _forward_raises):
        snapped = np.asarray(_snap_to_nebular_catalog(model, _TARGETS))

    np.testing.assert_array_equal(snapped, expected)
    assert snapped[0] != _TARGETS[0], "H-alpha target did not snap to the catalog"
    assert snapped[2] == _TARGETS[2], "a target far from every line must stay put"


@requires_cue_weights
def test_state_wave_is_the_rest_wavelength(_cue_fixture_available):
    model = _build(True)
    wave = model.predict_state({}).wave
    rest = model._rest_wavelength
    assert wave.dtype == rest.dtype
    np.testing.assert_array_equal(np.asarray(wave), np.asarray(rest))

# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the AGNfitter driver reads all five Task-4 torus reductions.

Task 4 vendored five additional AGNfitter-rX torus reductions (the NK08 2- and
3-parameter averages, the SKIRTOR 1- and 2-parameter averages, and the
low-wind-fraction half of CAT3D) into ``agnfitter_torus_reference.h5``, but
left the driver with no accessor for them -- the notebook could read the
four headline libraries (S04, NK08, SKIRTOR, CAT3D) only. This test pins the
Task 9 fix: :func:`torus_template` and :func:`torus_axes` dispatch to the new
groups by name, on the SAME (incl, oa, tau, a, fwd) keyword vocabulary the
four legacy libraries already use, and :func:`list_tori` enumerates all nine.

The node-exact tests below index the raw h5 ``template`` array directly (not
through the driver) and compare -- this is what would catch a transposed
axis order, the main risk on a genuinely multi-dimensional reduction
(NK08_3P has three axes, CAT3D_LOWFWD three): a swapped axis order still
returns a *some* finite positive template, so a bare finiteness/positivity
check would pass on a silently wrong answer.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.contract

_REF = Path(__file__).resolve().parents[2] / "data" / "agnfitter_torus_reference.h5"

if not _REF.is_file():
    pytest.skip(f"Reference data not found at {_REF}", allow_module_level=True)


@pytest.fixture
def driver():
    return pytest.importorskip(
        "reproduction.agnfitter._drivers.agnfitter_driver",
        reason="reproduction package not importable",
    )


_REDUCTIONS = (
    "NK08_2P",
    "NK08_3P",
    "SKIRTOR_MEAN1P",
    "SKIRTOR_MEAN2P",
    "CAT3D_LOWFWD",
)
_H5_GROUP = {
    "NK08_2P": "nk08_2p",
    "NK08_3P": "nk08_3p",
    "SKIRTOR_MEAN1P": "skirtor_mean1p",
    "SKIRTOR_MEAN2P": "skirtor_mean2p",
    "CAT3D_LOWFWD": "cat3d_lowfwd",
}
_AXES = {
    "NK08_2P": ("incl", "oa"),
    "NK08_3P": ("incl", "oa", "tau"),
    "SKIRTOR_MEAN1P": ("incl",),
    "SKIRTOR_MEAN2P": ("oa", "incl"),
    "CAT3D_LOWFWD": ("incl", "a", "fwd"),
}
_H5_AXIS_KEY = {
    "incl": "incl_axis",
    "oa": "oa_axis",
    "tau": "tv_axis",
    "a": "a_axis",
    "fwd": "fwd_axis",
}


def test_list_tori_includes_all_nine(driver):
    names = driver.list_tori()
    assert set(names) == {"S04", "NK08", "SKIRTOR", "CAT3D", *_REDUCTIONS}


@pytest.mark.parametrize("name", _REDUCTIONS)
def test_axes_match_direct_h5_read(driver, name):
    """``torus_axes`` returns exactly the h5's own axis arrays, in order."""
    with h5py.File(_REF, "r") as h:
        g = h[_H5_GROUP[name]]
        expected = {axis: np.asarray(g[_H5_AXIS_KEY[axis]][:]) for axis in _AXES[name]}
    got = driver.torus_axes(name)
    assert set(got) == set(expected)
    for axis, arr in expected.items():
        assert np.array_equal(got[axis], arr), f"{name}: axis {axis!r} does not match h5"


@pytest.mark.parametrize("name", _REDUCTIONS)
def test_template_matches_direct_h5_index_at_a_named_node(driver, name):
    """``torus_template`` at an exact grid value equals a direct h5 index.

    Selects a DIFFERENT node position on each axis (1, 2, 3, ... by axis
    order, wrapped to that axis's size) -- deliberately not the same index
    on every axis -- and confirms the (wavelength, template) pair returned
    equals ``template[i, j, ...]`` read straight from the h5, with the axis
    values passed by keyword in the SAME order every other torus panel in
    the notebook uses (incl, oa, tau, a, fwd). Using distinct per-axis
    indices is what makes this catch a transposed axis order: with the same
    index on every axis (e.g. all "node 1"), swapping two axes in the
    driver's dispatch selects ``template[1, 1]`` either way and the bug is
    invisible; here a swap selects a different, still individually
    well-formed template slice and fails the comparison.
    """
    with h5py.File(_REF, "r") as h:
        g = h[_H5_GROUP[name]]
        axis_values = {axis: np.asarray(g[_H5_AXIS_KEY[axis]][:]) for axis in _AXES[name]}
        idx = tuple((pos + 1) % axis_values[axis].size for pos, axis in enumerate(_AXES[name]))
        expected_template = np.asarray(g["template"][:])[idx]
        expected_wave = np.asarray(g["wavelength"][:])

    kwargs = {axis: float(axis_values[axis][i]) for axis, i in zip(_AXES[name], idx, strict=True)}
    wave, L_nu = driver.torus_template(name, **kwargs)

    assert np.array_equal(wave, expected_wave)
    # torus_template divides the raw template by the driver's cosmetic "TO"
    # renorm constant (1e-40) -- a fixed, code-visible factor, not a fit.
    np.testing.assert_allclose(L_nu, expected_template / 1e-40, rtol=1e-12)


@pytest.mark.parametrize("name", _REDUCTIONS)
def test_template_is_a_well_formed_sed(driver, name):
    wave, L_nu = driver.torus_template(name)
    assert wave.shape == L_nu.shape and wave.size > 1
    assert np.all(np.isfinite(wave)) and np.all(np.isfinite(L_nu))
    assert np.all(np.diff(wave) > 0), f"{name}: wavelengths not ascending"
    assert np.nanmax(L_nu) > 0.0, f"{name}: all-zero template"


def test_unknown_reduction_name_raises(driver):
    with pytest.raises(ValueError, match="Unknown torus library"):
        driver.torus_template("NOT_A_REDUCTION")
    with pytest.raises(ValueError, match="Unknown torus library"):
        driver.torus_axes("NOT_A_REDUCTION")

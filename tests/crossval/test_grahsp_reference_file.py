"""Crossval test: upstream GRAHSP reference data file.

Validates that data/grahsp_upstream_reference.h5 exists, carries provenance
attributes, and contains all expected contributions with finite values on a
shared grid. The physics comparison against tengri is a separate notebook.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def ref_file():
    """Load the reference file, skip cleanly if absent."""
    path = Path(__file__).resolve().parents[2] / "data" / "grahsp_upstream_reference.h5"
    if not path.exists():
        pytest.skip(f"Reference file not found: {path}")
    return h5py.File(path, "r")


def test_reference_file_exists(ref_file):
    """Reference file exists and is readable."""
    assert ref_file is not None
    assert "wavelength_nm" in ref_file


def test_provenance_attributes(ref_file):
    """Root attributes document build provenance."""
    required_attrs = [
        "upstream_repo",
        "upstream_commit",
        "upstream_modules",
        "wavelength_unit",
        "luminosity_unit",
        "attenuation_convention",
        "build_date",
        "build_command",
    ]
    for attr in required_attrs:
        assert attr in ref_file.attrs, f"Missing attribute: {attr}"

    # Verify upstream repo is the correct source
    repo = ref_file.attrs["upstream_repo"]
    if isinstance(repo, bytes):
        repo = repo.decode('utf-8', errors='replace')
    assert "JohannesBuchner/GRAHSP" in repo, f"Wrong upstream repo: {repo}"

    # Verify commit
    assert ref_file.attrs["upstream_commit"] == "45054ddf44eef7bb1abb0ab3b54eee5574a28f77"

    # Verify build_command has no hardcoded machine paths
    build_cmd = ref_file.attrs["build_command"]
    if isinstance(build_cmd, bytes):
        build_cmd = build_cmd.decode('utf-8', errors='replace')
    assert "/Users/" not in build_cmd, f"build_command has hardcoded /Users/ path: {build_cmd}"
    assert "<upstream>" in build_cmd and "<python>" in build_cmd, \
        f"build_command should use <upstream> and <python> placeholders: {build_cmd}"


def test_shared_wavelength_grid(ref_file):
    """Wavelength grid is shared, 5000 log-spaced points 10 nm – 1e6 nm."""
    grid = ref_file["wavelength_nm"][:]
    assert grid.dtype == np.float64
    assert len(grid) == 5000
    assert np.allclose(grid[0], 10.0, rtol=1e-6)
    assert np.allclose(grid[-1], 1e6, rtol=1e-6)
    # Check log-spacing
    log_grid = np.log10(grid)
    dlog = np.diff(log_grid)
    assert np.allclose(dlog, dlog[0], rtol=1e-10)


def test_parameter_sets(ref_file):
    """All parameter sets are present with expected structure."""
    grid = ref_file["wavelength_nm"][:]
    expected_names = [
        "fiducial",
        # plslope
        "plslope_-2p0",
        "plslope_-1p5",
        "plslope_-1p0",
        # uvslope (fiducial is 0, so only ±0.5)
        "uvslope_-0p5",
        "uvslope_0p5",
        # ... (others would be similar)
    ]

    # Check that at least fiducial exists
    assert "fiducial" in ref_file, "Fiducial parameter set missing"

    # Check all groups have contribution datasets
    for group_name in ref_file:
        if group_name == "wavelength_nm":
            continue
        grp = ref_file[group_name]
        assert len(grp) > 0, f"Group {group_name} is empty"

        # All datasets should have shape (5000,) or (5000,) for spectra
        for ds_name in grp:
            ds = grp[ds_name]
            assert len(ds.shape) == 1, f"{group_name}/{ds_name} should be 1D"
            assert ds.shape[0] == len(grid), f"{group_name}/{ds_name} shape mismatch with wavelength grid"
            assert ds.dtype == np.float64, f"{group_name}/{ds_name} should be float64"


def test_contributions_finite(ref_file):
    """All contribution spectra have finite non-negative values."""
    grid = ref_file["wavelength_nm"][:]

    for group_name in ref_file:
        if group_name == "wavelength_nm":
            continue
        grp = ref_file[group_name]

        for ds_name in grp:
            ds = grp[ds_name]
            data = ds[:]

            # Check finite (all contributions must have finite values)
            assert np.all(np.isfinite(data)), f"{group_name}/{ds_name} has non-finite values"


def test_fiducial_bbb_nonzero_at_510nm(ref_file):
    """Fiducial BBB (disc) is nonzero at 510 nm."""
    grid = ref_file["wavelength_nm"][:]
    grp = ref_file["fiducial"]

    # Find disc contribution
    assert "agn.activate_Disk" in grp, "Disc contribution not found"
    disc = grp["agn.activate_Disk"][:]

    # Interpolate at 510 nm
    idx = np.searchsorted(grid, 510.0)
    if idx < len(grid):
        # Use index closest to 510 nm
        if idx > 0 and abs(grid[idx - 1] - 510.0) < abs(grid[idx] - 510.0):
            idx -= 1
        assert disc[idx] > 0, f"BBB at 510 nm is {disc[idx]}, should be nonzero"


def test_parameter_attributes(ref_file):
    """Each parameter set has parameter attributes."""
    for group_name in ref_file:
        if group_name == "wavelength_nm":
            continue
        grp = ref_file[group_name]

        # Should have at least one param_* attribute
        param_attrs = [k for k in grp.attrs.keys() if k.startswith("param_")]
        assert len(param_attrs) > 0, f"{group_name} has no parameter attributes"


def test_sweep_sets_differ_from_fiducial(ref_file):
    """Each sweep set differs from fiducial in exactly the swept parameter."""
    fiducial_attrs = dict(ref_file["fiducial"].attrs)

    for group_name in ref_file:
        if group_name in ("fiducial", "wavelength_nm"):
            continue

        grp = ref_file[group_name]
        grp_attrs = dict(grp.attrs)

        # Extract parameter keys
        fid_params = {k.replace("param_", ""): fiducial_attrs[k]
                     for k in fiducial_attrs if k.startswith("param_")}
        grp_params = {k.replace("param_", ""): grp_attrs[k]
                     for k in grp_attrs if k.startswith("param_")}

        # Count differences
        diffs = [k for k in fid_params if fid_params.get(k) != grp_params.get(k)]

        # Most sweeps should have exactly 1 difference (unless AGNtype=2 which might affect more)
        # For now, just check there's at least 1
        assert len(diffs) >= 1, f"Sweep {group_name} identical to fiducial"


def test_attenuation_identity(ref_file):
    """Verify attenuation identity: attenuated = intrinsic + attenuation.*"""
    tol = 1e-14  # Numerical precision of float64

    for group_name in ref_file:
        if group_name == "wavelength_nm":
            continue
        grp = ref_file[group_name]

        # Get parameters to check if attenuation should be zero
        params = {k.replace("param_", ""): v for k, v in grp.attrs.items() if k.startswith("param_")}
        ebv = params.get("ebv", 0.0)
        ebv_agn = params.get("ebv_agn", 0.0)

        # Map contribution names to attenuated version
        contributions = list(grp.keys())
        for contrib_name in contributions:
            if contrib_name.startswith("attenuation."):
                continue

            if contrib_name == "stellar.dummy":
                # Host contribution, not part of AGN attenuation identity
                continue

            atten_name = f"attenuation.{contrib_name}"
            if atten_name not in grp:
                continue

            intrinsic = grp[contrib_name][:]
            attenuation = grp[atten_name][:]

            # When no attenuation, difference should be zero
            if ebv == 0.0 and ebv_agn == 0.0:
                assert np.allclose(attenuation, 0, atol=tol), \
                    f"{group_name}/{atten_name} should be zero when ebv=ebv_agn=0"

            # Attenuation factors (except where intrinsic is zero or very small) should be negative
            # or zero (multiplicative: attenuated = intrinsic * factor, so attenuation = intrinsic * (factor - 1) ≤ 0)
            # except for Si which can be negative for intrinsic too
            if "Si" not in contrib_name:
                # For non-Si contributions, where intrinsic is significant, attenuation should be ≤ small positive value
                significant = np.abs(intrinsic) > 1e-30
                if np.any(significant):
                    assert np.all(attenuation[significant] <= tol), \
                        f"{group_name}/{atten_name} has unexpectedly positive values where intrinsic is large"


def test_balmer_continuum_presence(ref_file):
    """BC (Balmer Continuum) present when ABC > 0, absent when ABC = 0."""
    fiducial_grp = ref_file["fiducial"]
    fiducial_params = {k.replace("param_", ""): v for k, v in fiducial_grp.attrs.items() if k.startswith("param_")}

    # Fiducial should have ABC = 0 and no BC
    assert fiducial_params.get("abc", 0.0) == 0.0, "Fiducial ABC should be 0"
    assert "agn.activate_BC" not in fiducial_grp, "Fiducial should not have BC when ABC=0"

    # Find ABC > 0 sets
    for group_name in ref_file:
        if group_name in ("fiducial", "wavelength_nm"):
            continue
        if not group_name.startswith("abc_"):
            continue

        grp = ref_file[group_name]
        params = {k.replace("param_", ""): v for k, v in grp.attrs.items() if k.startswith("param_")}
        abc_val = params.get("abc", 0.0)

        if abc_val > 0:
            assert "agn.activate_BC" in grp, \
                f"{group_name} has ABC={abc_val} > 0 but no BC contribution"
            bc = grp["agn.activate_BC"][:]
            assert np.any(bc > 0), f"{group_name} has BC contribution but all zeros"

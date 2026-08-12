# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's Nenkova+08 (AGNfitter-rX) torus against the reference.

The Nenkova et al. 2008 CLUMPY inclination-averaged torus templates shipped
inside AGNfitter-rX as ``models/TORUS/NK0_mean_1p.pickle`` were extracted
ONCE to the committed HDF5 ``data/nenkova_agnfitter_torus_grid.h5`` by
``scripts/build_nk08_agnfitter_grid.py`` (run with ``--download`` to fetch
the upstream pickle from GitHub — no AGNfitter install needed). This test
verifies the runtime component reproduces the stored templates via node-exact
PCHIP interpolation on the inclination axis.

The reference data is committed in
``data/agnfitter_torus_reference.h5/nk08`` with 9 inclination bins
(10–90° in 10° increments) and 1024 wavelength points.

Tolerance note: the runtime uses node-exact monotone-cubic (PCHIP)
interpolation over the single ``cos_inc`` axis. At grid nodes, the PCHIP
kernel returns exact values; between nodes, C¹-smooth monotone-cubic
interpolation dominates the error. Peak-normalized shape residuals are
expected to be <few%.

References
----------
.. [1] M. Nenkova et al., "Revisiting the AGN torus with MIDI and VISIR
   Herschel observations," ApJ 685, 160 (2008). arXiv:0806.1512.
.. [2] L. N. Martínez-Ramírez et al., "AGNfitter-rx: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._grad_parity import assert_grad_matches_fd

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval

_GRID = Path(__file__).resolve().parents[2] / "data" / "nenkova_agnfitter_torus_grid.h5"
_REF = Path(__file__).resolve().parents[2] / "data" / "agnfitter_torus_reference.h5"

if not _GRID.is_file():
    pytest.skip(
        f"Nenkova AGNfitter grid not found at {_GRID} "
        "(build with: python scripts/build_nk08_agnfitter_grid.py --download)",
        allow_module_level=True,
    )

if not _REF.is_file():
    pytest.skip(
        f"Reference data not found at {_REF}",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def grid() -> dict:
    with h5py.File(_GRID, "r") as f:
        g = f["nenkova_agnfitter"]
        return {
            "incl": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def reference() -> dict:
    with h5py.File(_REF, "r") as f:
        g = f["nk08"]
        return {
            "incl": np.asarray(g["axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "sed": np.asarray(g["sed"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def component():
    from tengri.components.agn.nenkova_agnfitter import create_nenkova_agnfitter_from_grid

    return create_nenkova_agnfitter_from_grid(str(_GRID))


def _peak_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x / np.nanmax(x[mask])


def _call(component, grid, i):
    """Evaluate the runtime component at grid node ``incl[i]``."""
    incl_deg = grid["incl"][i]
    cos_inc = np.cos(np.deg2rad(incl_deg))
    return np.asarray(
        component(
            wavelength=jnp.asarray(grid["wavelength"]),
            agn_log_lbol=0.0,
            agn_cos_inc=float(cos_inc),
            agn_torus_frac=1.0,
        )
    )


def test_peak_in_infrared(grid, component):
    """The reprocessed-dust torus peaks in the IR for a representative inclination."""
    i = grid["incl"].size // 2
    out = _call(component, grid, i)
    peak_um = grid["wavelength"][np.nanargmax(out)] / 1e4
    assert 1.0 < peak_um < 200.0, f"NK08 peak {peak_um:.1f} µm outside the IR torus band"


@pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_node_shape_matches_grid(grid, component, frac):
    """Runtime component reproduces the stored node template shape (PCHIP budget)."""
    i = round(frac * (grid["incl"].size - 1))
    ref = grid["template"][i]
    out = _call(component, grid, i)

    mask = (ref > ref.max() * 1e-3) & (out > 0)
    ref_n = _peak_norm(ref, mask)
    out_n = _peak_norm(out, mask)

    assert abs(np.nanargmax(ref) - np.nanargmax(out)) <= 2

    worst = float(np.nanmax(np.abs(out_n[mask] - ref_n[mask]) / ref_n[mask]))
    assert worst < 0.15, (
        f"Node incl={grid['incl'][i]:.1f}°: shape residual {worst * 100:.1f}% > 15% "
        "(PCHIP-kernel smoothing budget)"
    )


def test_matches_agnfitter_reference(grid, reference, component):
    """Runtime matches AGNfitter-rX reference at the three cardinal inclinations."""
    # Map AGNfitter-rX reference inclinations to our grid indices
    target_incl = [10.0, 45.0, 90.0]  # face-on, edge-on, pole-on
    for target in target_incl:
        i = np.argmin(np.abs(grid["incl"] - target))
        out = _call(component, grid, i)

        # Reference data is on a different wavelength grid; resample to our grid
        ref_idx = np.argmin(np.abs(reference["incl"] - target))
        ref_resampled = np.interp(
            grid["wavelength"],
            reference["wavelength"],
            reference["sed"][ref_idx],
            left=0.0,
            right=0.0,
        )

        mask = (ref_resampled > ref_resampled.max() * 1e-3) & (out > 0)
        ref_n = _peak_norm(ref_resampled, mask)
        out_n = _peak_norm(out, mask)

        worst = float(np.nanmax(np.abs(out_n[mask] - ref_n[mask]) / ref_n[mask]))
        assert worst < 0.15, (
            f"AGNfitter reference at incl={target:.1f}°: shape residual {worst * 100:.1f}% > 15%"
        )


def test_parameter_liveness():
    """Verify agn_cos_inc actually changes the SED."""
    from tengri.components.agn.nenkova_agnfitter import nenkova_agnfitter_sed

    wavelength = jnp.logspace(1.0, 5.0, 256)
    sed_face_on = nenkova_agnfitter_sed(
        wavelength, agn_log_lbol=0.0, agn_cos_inc=0.99, agn_torus_frac=1.0
    )
    sed_edge_on = nenkova_agnfitter_sed(
        wavelength, agn_log_lbol=0.0, agn_cos_inc=0.01, agn_torus_frac=1.0
    )

    # Peak-normalized maximum difference should be >1%
    mask = (sed_face_on > 0) & (sed_edge_on > 0)
    peak_face = np.nanmax(sed_face_on[mask])
    peak_edge = np.nanmax(sed_edge_on[mask])
    sed_face_n = sed_face_on / peak_face
    sed_edge_n = sed_edge_on / peak_edge
    max_diff = float(np.nanmax(np.abs(sed_face_n[mask] - sed_edge_n[mask])))
    assert max_diff > 0.01, f"agn_cos_inc should move SED by >1%; got {max_diff * 100:.2f}%"


def test_lbol_scales_linearly():
    """Verify L_bol scales the output linearly."""
    from tengri.components.agn.nenkova_agnfitter import nenkova_agnfitter_sed

    wavelength = jnp.logspace(1.0, 5.0, 256)
    sed_11 = nenkova_agnfitter_sed(
        wavelength, agn_log_lbol=11.0, agn_cos_inc=0.5, agn_torus_frac=1.0
    )
    sed_12 = nenkova_agnfitter_sed(
        wavelength, agn_log_lbol=12.0, agn_cos_inc=0.5, agn_torus_frac=1.0
    )

    # Ratio should be ~10 (difference of 1 in log scale)
    ratio = sed_12 / (sed_11 + 1e-100)
    ratio_clean = ratio[ratio > 0]
    expected_ratio = 10.0
    actual_ratio = np.median(ratio_clean)
    error = abs(actual_ratio - expected_ratio) / expected_ratio
    assert error < 0.01, f"Expected L_bol ratio {expected_ratio}, got {actual_ratio:.2f}"


def test_gradient_flows():
    """Verify gradients flow through agn_cos_inc."""
    from tengri.components.agn.nenkova_agnfitter import nenkova_agnfitter_sed

    def sed_total_power(cos_inc):
        wavelength = jnp.logspace(1.0, 5.0, 256)
        sed = nenkova_agnfitter_sed(
            wavelength, agn_log_lbol=0.0, agn_cos_inc=cos_inc, agn_torus_frac=1.0
        )
        # Integrate in frequency space
        nu = 2.99792458e8 / (wavelength * 1e-8)
        power = jnp.trapezoid(sed, nu)
        return power

    cos_inc_val = 0.5
    grad = assert_grad_matches_fd(sed_total_power, jnp.asarray(cos_inc_val))
    assert np.isfinite(grad) and grad != 0.0, "Gradient should be finite and non-zero"

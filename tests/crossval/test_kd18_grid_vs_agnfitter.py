# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's AGNfitter-rX KD18 discs against their vendored grids.

The Kubota & Done (2018) grid-tabulated accretion-disc library ships inside
AGNfitter-rX at ``models/BBB/KD18.pickle`` (2-D: log M_BH, log Edd) and
``models/BBB/KD18_warmInd.pickle`` (3-D: + the warm-Comptonization spectral
index, AGNfitter-rX's ``warmIndex``). Both were converted ONCE to the
committed HDF5 files (``data/kd18_agnfitter_disc_grid.h5``,
``data/kd18_agnfitter_warmindex_disc_grid.h5``) by
``scripts/build_kd18_grid.py`` (run ``--download`` to fetch the upstream
pickles from GitHub -- no AGNfitter install needed). This test reads only
those HDF5 files (no pickle, no AGNfitter driver) so it runs in CI rather
than skipping (#613). The grids' stored node templates ARE the AGNfitter
reference; the test checks that the runtime components reproduce them.

Tolerance note: like the SN12 disc (``test_slone_netzer_vs_agnfitter.py``),
both KD18 blocks use **node-exact bilinear/trilinear** interpolation over
their grid axes (the disc peak wavelength varies strongly with accretion
rate, so a smooth kernel would smear it). At a grid node this returns the
stored template exactly, so the shape residual here is at the
floating-point level, not a kernel budget.

References
----------
.. [1] A. Kubota & C. Done, "A physical model of the broad-band continuum of
   AGN and its implications for the UV/X relation and optical variability,"
   MNRAS 480, 1247 (2018). doi:10.1093/mnras/sty1890. arXiv:1804.00171.
   bibcode:2018MNRAS.480.1247K.
.. [2] L. N. Martinez-Ramirez et al., "AGNfitter-rx: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111. DOI: 10.1051/0004-6361/202449329.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_PLAIN_GRID = Path(__file__).resolve().parents[2] / "data" / "kd18_agnfitter_disc_grid.h5"
_WARMINDEX_GRID = (
    Path(__file__).resolve().parents[2] / "data" / "kd18_agnfitter_warmindex_disc_grid.h5"
)

if not _PLAIN_GRID.is_file() or not _WARMINDEX_GRID.is_file():
    pytest.skip(
        f"KD18-agnfitter grid(s) not found at {_PLAIN_GRID} / {_WARMINDEX_GRID} "
        "(build with: python scripts/build_kd18_grid.py --download)",
        allow_module_level=True,
    )


def _read_group(path: Path, group: str, keys: tuple[str, ...]) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        g = f[group]
        return {k: np.asarray(g[k][:], dtype=np.float64) for k in keys}


@pytest.fixture(scope="module")
def plain_grid() -> dict:
    return _read_group(
        _PLAIN_GRID, "kd18_agnfitter", ("log_mbh", "log_edd", "wavelength", "template")
    )


@pytest.fixture(scope="module")
def warmindex_grid() -> dict:
    return _read_group(
        _WARMINDEX_GRID,
        "kd18_agnfitter_warmindex",
        ("log_mbh", "log_edd", "gamma_warm", "wavelength", "template"),
    )


@pytest.fixture(scope="module")
def plain_component():
    from tengri.components.agn.kd18_agnfitter import create_kd18_agnfitter_from_grid

    return create_kd18_agnfitter_from_grid(str(_PLAIN_GRID))


@pytest.fixture(scope="module")
def warmindex_component():
    from tengri.components.agn.kd18_agnfitter import create_kd18_agnfitter_warmindex_from_grid

    return create_kd18_agnfitter_warmindex_from_grid(str(_WARMINDEX_GRID))


def _peak_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x / np.nanmax(x[mask])


def _call_plain(component, grid, i, j):
    """Evaluate the plain-grid runtime component at node (log_mbh[i], log_edd[j])."""
    return np.asarray(
        component(
            wavelength=jnp.asarray(grid["wavelength"]),
            agn_log_lbol=0.0,
            agn_log_mbh=float(grid["log_mbh"][i]),
            agn_log_ledd=float(grid["log_edd"][j]),
        )
    )


def _call_warmindex(component, grid, i, j, k):
    """Evaluate the warmindex-grid runtime component at node (i, j, k)."""
    return np.asarray(
        component(
            wavelength=jnp.asarray(grid["wavelength"]),
            agn_log_lbol=0.0,
            agn_log_mbh=float(grid["log_mbh"][i]),
            agn_log_ledd=float(grid["log_edd"][j]),
            agn_gamma_warm=float(grid["gamma_warm"][k]),
        )
    )


def _assert_node_exact(ref: np.ndarray, out: np.ndarray, label: str) -> None:
    mask = (ref > ref.max() * 1e-3) & (out > 0)
    ref_n = _peak_norm(ref, mask)
    out_n = _peak_norm(out, mask)

    assert int(np.nanargmax(ref)) == int(np.nanargmax(out)), (
        f"{label}: peak wavelength index moved ({np.nanargmax(ref)} -> {np.nanargmax(out)})"
    )
    worst = float(np.nanmax(np.abs(out_n[mask] - ref_n[mask]) / ref_n[mask]))
    assert worst < 1e-3, f"{label}: shape residual {worst:.2e} > 1e-3 (should be node-exact)"


def test_peak_is_uv_optical_plain(plain_grid, plain_component):
    """A standard accretion disc peaks in the UV/optical (< 1 um)."""
    out = _call_plain(
        plain_component,
        plain_grid,
        i=plain_grid["log_mbh"].size // 2,
        j=plain_grid["log_edd"].size // 2,
    )
    peak_um = plain_grid["wavelength"][np.nanargmax(out)] / 1e4
    assert peak_um < 1.0, f"KD18 disc peak {peak_um:.3f} um is not UV/optical"


#: Spanning nodes including both edges of each axis (0/last) plus an interior one.
_PLAIN_NODES = [(0, 0), (14, 14), (0, 14), (14, 0), (7, 7)]


@pytest.mark.parametrize("i,j", _PLAIN_NODES)
def test_plain_node_shape_matches_grid(plain_grid, plain_component, i, j):
    """Node-exact bilinear reproduces the stored KD18 template to float level."""
    ref = plain_grid["template"][i, j]
    out = _call_plain(plain_component, plain_grid, i, j)
    _assert_node_exact(ref, out, f"plain node (i={i}, j={j})")


#: Spanning nodes for the 3-D grid, including every axis's edges.
_WARMINDEX_NODES = [
    (0, 0, 0),
    (14, 14, 9),
    (0, 14, 0),
    (14, 0, 9),
    (0, 0, 9),
    (14, 14, 0),
    (7, 7, 4),
]


@pytest.mark.parametrize("i,j,k", _WARMINDEX_NODES)
def test_warmindex_node_shape_matches_grid(warmindex_grid, warmindex_component, i, j, k):
    """Node-exact trilinear reproduces the stored KD18_warmInd template."""
    ref = warmindex_grid["template"][i, j, k]
    out = _call_warmindex(warmindex_component, warmindex_grid, i, j, k)
    _assert_node_exact(ref, out, f"warmindex node (i={i}, j={j}, k={k})")


def test_plain_grid_axis_extent(plain_grid):
    """The committed plain grid spans the expected KD18 (M_BH, Edd) extent."""
    assert plain_grid["log_mbh"].size == 15
    assert plain_grid["log_edd"].size == 15
    assert plain_grid["log_mbh"].min() == pytest.approx(6.0)
    assert plain_grid["log_mbh"].max() == pytest.approx(10.0)
    assert plain_grid["log_edd"].min() == pytest.approx(-1.5)
    assert plain_grid["log_edd"].max() == pytest.approx(0.0)


def test_warmindex_grid_axis_extent(warmindex_grid):
    """The committed warmindex grid spans the expected KD18_warmInd extent."""
    assert warmindex_grid["log_mbh"].size == 15
    assert warmindex_grid["log_edd"].size == 15
    assert warmindex_grid["gamma_warm"].size == 10
    assert warmindex_grid["gamma_warm"].min() == pytest.approx(1.5)
    assert warmindex_grid["gamma_warm"].max() == pytest.approx(4.0)


def test_plain_and_warmindex_are_not_equivalent_at_any_fixed_gamma_warm(
    plain_grid, warmindex_grid
):
    """The Step-1 measurement, pinned as a regression: the two pickles disagree.

    Motivates vendoring two grids/blocks instead of one grid with a pinned
    ``gamma_warm`` default: even the best-matching candidate value disagrees
    by more than 10% at some wavelength, node-by-node on (log_mbh, log_edd).
    """
    plain = plain_grid["template"]  # (15, 15, 100)
    warm = warmindex_grid["template"]  # (15, 15, 10, 100)
    best_over_k = []
    for k in range(warm.shape[2]):
        mask = (plain > 0) & (warm[:, :, k, :] > 0)
        ratio = np.log10(warm[:, :, k, :][mask]) - np.log10(plain[mask])
        best_over_k.append(float(np.max(np.abs(ratio))))
    assert min(best_over_k) > 0.1, (
        "the closest-matching gamma_warm value is now within 10% of the plain "
        "KD18 grid everywhere -- if the vendored grids changed on purpose, "
        "reconsider whether a single grid + pinned default now suffices"
    )

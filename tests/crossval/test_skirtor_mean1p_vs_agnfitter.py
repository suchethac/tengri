# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's SKIRTOR_mean_1p torus against the AGNfitter-rX grid.

AGNfitter-rX ships ``models/TORUS/SKIRTOR_mean_1p.pickle`` -- the Stalevski+2016
SKIRTOR library averaged over opening angle, clumpiness (p, q), and optical
depth, leaving inclination only. It was converted ONCE to the committed HDF5
``data/skirtor_mean1p_torus_grid.h5`` by
``scripts/build_agnfitter_torus_reductions.py`` at native wavelength
resolution (every row of the upstream pickle shares one wavelength axis, so
nothing was resampled). This test compares the runtime component against an
INDEPENDENT load of the same upstream pickle
(``data/agnfitter_torus_reference.h5/skirtor_mean1p``, written by a
separate, row-by-row reader in ``scripts/build_agnfitter_bbb_reference.py``).

References
----------
.. [1] Stalevski et al. 2016, MNRAS, 458, 2288. arXiv:1602.06954.
.. [2] Martinez-Ramirez et al. 2024, A&A, 688, A46 (AGNfitter-rX). arXiv:2405.12111.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_GRID = Path(__file__).resolve().parents[2] / "data" / "skirtor_mean1p_torus_grid.h5"
_REF = Path(__file__).resolve().parents[2] / "data" / "agnfitter_torus_reference.h5"

if not _GRID.is_file():
    pytest.skip(
        f"SKIRTOR_mean_1p grid not found at {_GRID} "
        "(build with: python scripts/build_agnfitter_torus_reductions.py --which skirtor_mean1p)",
        allow_module_level=True,
    )
if not _REF.is_file():
    pytest.skip(f"Reference data not found at {_REF}", allow_module_level=True)


@pytest.fixture(scope="module")
def grid() -> dict:
    with h5py.File(_GRID, "r") as f:
        g = f["skirtor_mean1p"]
        return {
            "incl": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def reference() -> dict:
    with h5py.File(_REF, "r") as f:
        g = f["skirtor_mean1p"]
        return {
            "incl": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def component():
    from tengri.components.agn.skirtor_agnfitter_1p import create_skirtor_agnfitter_1p_from_grid

    return create_skirtor_agnfitter_1p_from_grid(str(_GRID))


def _peak_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x / np.nanmax(x[mask])


def test_reference_matches_vendored_grid_axis(grid, reference):
    assert np.array_equal(grid["incl"], reference["incl"])


@pytest.mark.parametrize("i_incl", [0, -1, 4])  # both edges + one interior node
def test_node_exact_matches_independent_reference(grid, reference, component, i_incl):
    """Runtime component reproduces the INDEPENDENTLY-loaded reference at grid
    nodes (incl. both axis edges) to < 1e-3 relative, peak-normalized."""
    incl_deg = float(grid["incl"][i_incl])
    ref_i = int(np.argmin(np.abs(reference["incl"] - incl_deg)))
    ref = reference["template"][ref_i]
    ref_wave = reference["wavelength"]

    out = np.asarray(
        component(
            wavelength=jnp.asarray(ref_wave),
            agn_log_lbol=0.0,
            agn_incl_skirtor=incl_deg,
            agn_torus_frac=1.0,
        )
    )

    mask = (ref > ref.max() * 1e-3) & (out > 0)
    ref_n = _peak_norm(ref, mask)
    out_n = _peak_norm(out, mask)
    assert int(np.nanargmax(ref)) == int(np.nanargmax(out))
    worst = float(np.nanmax(np.abs(out_n[mask] - ref_n[mask]) / ref_n[mask]))
    assert worst < 1e-3, (
        f"Node incl={incl_deg}: shape residual {worst:.2e} > 1e-3 "
        "vs the independently-loaded reference"
    )


def test_parameter_liveness(grid, component):
    """agn_incl_skirtor changes the SED by more than 1% across the grid extent."""
    wave = grid["wavelength"]
    a = np.asarray(
        component(
            wavelength=jnp.asarray(wave),
            agn_log_lbol=0.0,
            agn_incl_skirtor=float(grid["incl"].min()),
            agn_torus_frac=1.0,
        )
    )
    b = np.asarray(
        component(
            wavelength=jnp.asarray(wave),
            agn_log_lbol=0.0,
            agn_incl_skirtor=float(grid["incl"].max()),
            agn_torus_frac=1.0,
        )
    )
    mask = (a > 0) & (b > 0)
    a_n, b_n = a / np.nanmax(a[mask]), b / np.nanmax(b[mask])
    assert float(np.nanmax(np.abs(a_n[mask] - b_n[mask]))) > 0.01


def test_extents_match_h5(grid):
    """Declared prior equals the grid extent (mirrors the Task 1 guard)."""
    from tengri.components.agn._params import PARAMS

    by_name = {pd.name: pd.prior for pd in PARAMS}
    incl = by_name["agn_incl_skirtor"]
    assert (float(incl.lo), float(incl.hi)) == (
        float(grid["incl"].min()),
        float(grid["incl"].max()),
    )


def test_composable_wiring_differs_from_sibling_and_gradients_live(synthetic_ssp_wide):
    """Public dict-grammar build: sed_agn differs from a sibling torus type,
    and jax.grad of a band flux is nonzero for every declared parameter."""
    from tengri import DEFAULT, FREE, Fixed, SEDModel
    from tengri.observation import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    def _tophat(c, frac=0.16, n=40):
        w = jnp.linspace(c * (1 - frac), c * (1 + frac), n)
        return FilterCurve(
            wave=w, trans=jnp.sin(jnp.linspace(0, jnp.pi, n)) * 0.6, name=f"b{int(c)}"
        )

    obs = Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (5000.0, 2e4, 1e5, 5e5)))
    )

    def _build(torus_type, all_params):
        return SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "type": "composable",
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                "torus": {"type": torus_type, "all_params": all_params},
                "all_params": Fixed(DEFAULT),
                "agn_log_lbol": Fixed(12.0),
                "norm": "independent",
            },
            redshift=Fixed(0.05),
        )

    m_new = _build("skirtor_agnfitter_1p", Fixed(DEFAULT))
    m_sibling = _build("cat3d_wind", Fixed(DEFAULT))
    sed_new = np.asarray(m_new.predict({}).sed.components["sed_agn"])
    sed_sibling = np.asarray(m_sibling.predict({}).sed.components["sed_agn"])
    assert not np.array_equal(sed_new, sed_sibling), (
        "skirtor_agnfitter_1p torus is bit-identical to cat3d_wind at default params"
    )

    m_free = _build("skirtor_agnfitter_1p", FREE)
    free_agn = {p for p in m_free.spec.free_params if p.startswith("agn_")}
    assert {"agn_incl_skirtor", "agn_torus_frac"} <= free_agn

    p0 = dict(m_free.spec.sample(jax.random.PRNGKey(0)))

    def obj(pd):
        return jnp.log(jnp.sum(m_free.predict_photometry(pd)) + 1e-300)

    for name in ("agn_incl_skirtor", "agn_torus_frac"):
        v0 = jnp.asarray(p0[name])
        g = float(jax.grad(lambda v, name=name: obj({**p0, name: v}))(v0))
        assert g != 0.0, f"{name}: gradient of band flux is exactly zero"

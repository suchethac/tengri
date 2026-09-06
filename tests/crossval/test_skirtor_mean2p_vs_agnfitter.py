# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's SKIRTOR_mean_2p torus against the AGNfitter-rX grid.

AGNfitter-rX ships ``models/TORUS/SKIRTOR_mean_2p.pickle`` -- the Stalevski+2016
SKIRTOR library averaged over clumpiness (p, q) and optical depth, leaving
(oa, incl). It was converted ONCE to the committed HDF5
``data/skirtor_mean2p_torus_grid.h5`` by
``scripts/build_agnfitter_torus_reductions.py`` at native wavelength
resolution (every row of the upstream pickle shares one wavelength axis, so
nothing was resampled). This test compares the runtime component against an
INDEPENDENT load of the same upstream pickle
(``data/agnfitter_torus_reference.h5/skirtor_mean2p``, written by a
separate, row-by-row reader in ``scripts/build_agnfitter_bbb_reference.py``).

References
----------
.. [1] Stalevski et al. 2016, MNRAS, 458, 2288. arXiv:1602.01954.
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

_GRID = Path(__file__).resolve().parents[2] / "data" / "skirtor_mean2p_torus_grid.h5"
_REF = Path(__file__).resolve().parents[2] / "data" / "agnfitter_torus_reference.h5"

if not _GRID.is_file():
    pytest.skip(
        f"SKIRTOR_mean_2p grid not found at {_GRID} "
        "(build with: python scripts/build_agnfitter_torus_reductions.py --which skirtor_mean2p)",
        allow_module_level=True,
    )
if not _REF.is_file():
    pytest.skip(f"Reference data not found at {_REF}", allow_module_level=True)


@pytest.fixture(scope="module")
def grid() -> dict:
    with h5py.File(_GRID, "r") as f:
        g = f["skirtor_mean2p"]
        return {
            "oa": np.asarray(g["oa_axis"][:], dtype=np.float64),
            "incl": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def reference() -> dict:
    with h5py.File(_REF, "r") as f:
        g = f["skirtor_mean2p"]
        return {
            "oa": np.asarray(g["oa_axis"][:], dtype=np.float64),
            "incl": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def component():
    from tengri.components.agn.skirtor_agnfitter_2p import create_skirtor_agnfitter_2p_from_grid

    return create_skirtor_agnfitter_2p_from_grid(str(_GRID))


def _peak_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x / np.nanmax(x[mask])


def test_reference_matches_vendored_grid_axes(grid, reference):
    assert np.array_equal(grid["oa"], reference["oa"])
    assert np.array_equal(grid["incl"], reference["incl"])


@pytest.mark.parametrize("i_oa,j_incl", [(0, 0), (0, -1), (-1, 0), (-1, -1), (3, 6)])
def test_node_exact_matches_independent_reference(grid, reference, component, i_oa, j_incl):
    """Runtime component reproduces the INDEPENDENTLY-loaded reference at grid
    nodes (incl. every axis edge) to < 1e-3 relative, peak-normalized."""
    oa_deg = float(grid["oa"][i_oa])
    incl_deg = float(grid["incl"][j_incl])

    ref_i = int(np.argmin(np.abs(reference["oa"] - oa_deg)))
    ref_j = int(np.argmin(np.abs(reference["incl"] - incl_deg)))
    ref = reference["template"][ref_i, ref_j]
    ref_wave = reference["wavelength"]

    out = np.asarray(
        component(
            wavelength=jnp.asarray(ref_wave),
            agn_log_lbol=0.0,
            agn_oa_skirtor=oa_deg,
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
        f"Node (oa={oa_deg}, incl={incl_deg}): shape residual {worst:.2e} > 1e-3 "
        "vs the independently-loaded reference"
    )


def test_parameter_liveness_oa(grid, component):
    """agn_oa_skirtor changes the SED by more than 1% across the grid extent."""
    wave = grid["wavelength"]
    a = np.asarray(
        component(
            wavelength=jnp.asarray(wave),
            agn_log_lbol=0.0,
            agn_oa_skirtor=float(grid["oa"].min()),
            agn_incl_skirtor=30.0,
            agn_torus_frac=1.0,
        )
    )
    b = np.asarray(
        component(
            wavelength=jnp.asarray(wave),
            agn_log_lbol=0.0,
            agn_oa_skirtor=float(grid["oa"].max()),
            agn_incl_skirtor=30.0,
            agn_torus_frac=1.0,
        )
    )
    mask = (a > 0) & (b > 0)
    a_n, b_n = a / np.nanmax(a[mask]), b / np.nanmax(b[mask])
    assert float(np.nanmax(np.abs(a_n[mask] - b_n[mask]))) > 0.01


def test_extents_match_h5(grid):
    """Declared priors equal the grid extent (mirrors the Task 1 guard)."""
    from tengri.components.agn._params import PARAMS

    by_name = {pd.name: pd.prior for pd in PARAMS}
    oa = by_name["agn_oa_skirtor"]
    incl = by_name["agn_incl_skirtor"]
    assert (float(oa.lo), float(oa.hi)) == (float(grid["oa"].min()), float(grid["oa"].max()))
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

    m_new = _build("skirtor_agnfitter_2p", Fixed(DEFAULT))
    m_sibling = _build("cat3d_wind", Fixed(DEFAULT))
    sed_new = np.asarray(m_new.predict({}).sed.components["sed_agn"])
    sed_sibling = np.asarray(m_sibling.predict({}).sed.components["sed_agn"])
    assert not np.array_equal(sed_new, sed_sibling), (
        "skirtor_agnfitter_2p torus is bit-identical to cat3d_wind at default params"
    )

    m_free = _build("skirtor_agnfitter_2p", FREE)
    free_agn = {p for p in m_free.spec.free_params if p.startswith("agn_")}
    assert {"agn_oa_skirtor", "agn_incl_skirtor", "agn_torus_frac"} <= free_agn

    p0 = dict(m_free.spec.sample(jax.random.PRNGKey(0)))

    def obj(pd):
        return jnp.log(jnp.sum(m_free.predict_photometry(pd)) + 1e-300)

    for name in ("agn_oa_skirtor", "agn_incl_skirtor", "agn_torus_frac"):
        v0 = jnp.asarray(p0[name])
        g = float(jax.grad(lambda v, name=name: obj({**p0, name: v}))(v0))
        assert g != 0.0, f"{name}: gradient of band flux is exactly zero"

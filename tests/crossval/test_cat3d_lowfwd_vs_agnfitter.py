# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's CAT3D-Wind low-fwd torus against AGNfitter-rX.

AGNfitter-rX's ``models/TORUS/CAT3D_mean_3p.pickle`` concatenates two
disjoint three-parameter Hönig & Kishimoto (2017) sub-libraries: rows 0-209
(this module) cover the low polar-wind-fraction range (``fwd`` 0.15-0.75) at
a wider radial-index range (``a`` -3 to -0.5); rows 210-377
(:mod:`tengri.components.agn.cat3d_wind`, the reduction AGNfitter-rX's own
fitter selects by default) cover the high wind-fraction range. Rows 0-209
were converted ONCE to the committed HDF5
``data/cat3d_wind_lowfwd_torus_grid.h5`` by
``scripts/build_agnfitter_torus_reductions.py`` at native wavelength
resolution (every row shares one wavelength axis, so nothing was
resampled) -- and, being a full Cartesian product, needed no
nearest-neighbor cell filling (unlike the high-fwd rows 210+). This test
compares the runtime component against an INDEPENDENT load of the same
upstream pickle (``data/agnfitter_torus_reference.h5/cat3d_lowfwd``,
written by a separate, row-by-row reader in
``scripts/build_agnfitter_bbb_reference.py``).

References
----------
.. [1] S. F. Hönig & M. Kishimoto, ApJL 838, L20 (2017). arXiv:1702.08691.
.. [2] Martínez-Ramírez et al., A&A 688, A46 (2024) (AGNfitter-rX). arXiv:2405.12111.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_GRID = Path(__file__).resolve().parents[2] / "data" / "cat3d_wind_lowfwd_torus_grid.h5"
_REF = Path(__file__).resolve().parents[2] / "data" / "agnfitter_torus_reference.h5"

if not _GRID.is_file():
    pytest.skip(
        f"CAT3D-Wind low-fwd grid not found at {_GRID} "
        "(build with: python scripts/build_agnfitter_torus_reductions.py --which cat3d_lowfwd)",
        allow_module_level=True,
    )
if not _REF.is_file():
    pytest.skip(f"Reference data not found at {_REF}", allow_module_level=True)


@pytest.fixture(scope="module")
def grid() -> dict:
    with h5py.File(_GRID, "r") as f:
        g = f["cat3d_wind_lowfwd"]
        return {
            "incl": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "a": np.asarray(g["a_axis"][:], dtype=np.float64),
            "fwd": np.asarray(g["fwd_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def reference() -> dict:
    with h5py.File(_REF, "r") as f:
        g = f["cat3d_lowfwd"]
        return {
            "incl": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "a": np.asarray(g["a_axis"][:], dtype=np.float64),
            "fwd": np.asarray(g["fwd_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@pytest.fixture(scope="module")
def component():
    from tengri.components.agn.cat3d_wind_lowfwd import create_cat3d_wind_lowfwd_from_grid

    return create_cat3d_wind_lowfwd_from_grid(str(_GRID))


def _peak_norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return x / np.nanmax(x[mask])


def test_reference_matches_vendored_grid_axes(grid, reference):
    assert np.array_equal(grid["incl"], reference["incl"])
    assert np.array_equal(grid["a"], reference["a"])
    assert np.array_equal(grid["fwd"], reference["fwd"])


def test_full_cartesian_no_missing_cells(grid):
    """Unlike the high-fwd sub-library, the low-fwd rows are a complete
    Cartesian product -- no build-time nearest-neighbor fill was needed."""
    assert grid["template"].shape[:3] == (
        grid["incl"].size,
        grid["a"].size,
        grid["fwd"].size,
    )
    assert np.all(np.isfinite(grid["template"]))


@pytest.mark.parametrize(
    "i_incl,j_a,k_fwd",
    [(0, 0, 0), (0, -1, -1), (-1, 0, -1), (-1, -1, 0), (-1, -1, -1), (3, 3, 2)],
)
def test_node_exact_matches_independent_reference(grid, reference, component, i_incl, j_a, k_fwd):
    """Runtime component reproduces the INDEPENDENTLY-loaded reference at grid
    nodes (incl. every axis edge) to < 1e-3 relative, peak-normalized."""
    incl_deg = float(grid["incl"][i_incl])
    a = float(grid["a"][j_a])
    fwd = float(grid["fwd"][k_fwd])
    cos_inc = float(np.cos(np.deg2rad(incl_deg)))

    ref_i = int(np.argmin(np.abs(reference["incl"] - incl_deg)))
    ref_j = int(np.argmin(np.abs(reference["a"] - a)))
    ref_k = int(np.argmin(np.abs(reference["fwd"] - fwd)))
    ref = reference["template"][ref_i, ref_j, ref_k]
    ref_wave = reference["wavelength"]

    out = np.asarray(
        component(
            wavelength=jnp.asarray(ref_wave),
            agn_log_lbol=0.0,
            agn_cos_inc=cos_inc,
            agn_a_cat3d_lowfwd=a,
            agn_fwd_cat3d_lowfwd=fwd,
            agn_torus_frac=1.0,
        )
    )

    mask = (ref > ref.max() * 1e-3) & (out > 0)
    ref_n = _peak_norm(ref, mask)
    out_n = _peak_norm(out, mask)
    assert int(np.nanargmax(ref)) == int(np.nanargmax(out))
    worst = float(np.nanmax(np.abs(out_n[mask] - ref_n[mask]) / ref_n[mask]))
    assert worst < 1e-3, (
        f"Node (incl={incl_deg}, a={a}, fwd={fwd}): shape residual {worst:.2e} > 1e-3 "
        "vs the independently-loaded reference"
    )


def test_parameter_liveness_a_and_fwd(grid, component):
    """agn_a_cat3d_lowfwd and agn_fwd_cat3d_lowfwd each change the SED by
    more than 1% across the grid extent."""
    wave = grid["wavelength"]

    def _sed(a, fwd):
        return np.asarray(
            component(
                wavelength=jnp.asarray(wave),
                agn_log_lbol=0.0,
                agn_cos_inc=0.5,
                agn_a_cat3d_lowfwd=a,
                agn_fwd_cat3d_lowfwd=fwd,
                agn_torus_frac=1.0,
            )
        )

    fwd_mid = float(np.median(grid["fwd"]))
    a_lo = _sed(float(grid["a"].min()), fwd_mid)
    a_hi = _sed(float(grid["a"].max()), fwd_mid)
    mask = (a_lo > 0) & (a_hi > 0)
    lo_n, hi_n = a_lo / np.nanmax(a_lo[mask]), a_hi / np.nanmax(a_hi[mask])
    assert float(np.nanmax(np.abs(lo_n[mask] - hi_n[mask]))) > 0.01

    a_mid = float(np.median(grid["a"]))
    fwd_lo = _sed(a_mid, float(grid["fwd"].min()))
    fwd_hi = _sed(a_mid, float(grid["fwd"].max()))
    mask = (fwd_lo > 0) & (fwd_hi > 0)
    lo_n, hi_n = fwd_lo / np.nanmax(fwd_lo[mask]), fwd_hi / np.nanmax(fwd_hi[mask])
    assert float(np.nanmax(np.abs(lo_n[mask] - hi_n[mask]))) > 0.01


def test_extents_match_h5(grid):
    """Declared priors equal the grid extent (mirrors the Task 1 guard)."""
    from tengri.components.agn._params import PARAMS

    by_name = {pd.name: pd.prior for pd in PARAMS}
    a = by_name["agn_a_cat3d_lowfwd"]
    fwd = by_name["agn_fwd_cat3d_lowfwd"]
    assert (float(a.lo), float(a.hi)) == (float(grid["a"].min()), float(grid["a"].max()))
    assert (float(fwd.lo), float(fwd.hi)) == (float(grid["fwd"].min()), float(grid["fwd"].max()))


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

    m_new = _build("cat3d_wind_lowfwd", Fixed(DEFAULT))
    m_sibling = _build("skirtor_agnfitter", Fixed(DEFAULT))
    sed_new = np.asarray(m_new.predict({}).sed.components["sed_agn"])
    sed_sibling = np.asarray(m_sibling.predict({}).sed.components["sed_agn"])
    assert not np.array_equal(sed_new, sed_sibling), (
        "cat3d_wind_lowfwd torus is bit-identical to skirtor_agnfitter at default params"
    )

    m_free = _build("cat3d_wind_lowfwd", FREE)
    free_agn = {p for p in m_free.spec.free_params if p.startswith("agn_")}
    assert {"agn_a_cat3d_lowfwd", "agn_fwd_cat3d_lowfwd", "agn_torus_frac"} <= free_agn

    p0 = dict(m_free.spec.sample(jax.random.PRNGKey(0)))

    def obj(pd):
        return jnp.log(jnp.sum(m_free.predict_photometry(pd)) + 1e-300)

    for name in ("agn_a_cat3d_lowfwd", "agn_fwd_cat3d_lowfwd", "agn_torus_frac"):
        v0 = jnp.asarray(p0[name])
        g = float(jax.grad(lambda v, name=name: obj({**p0, name: v}))(v0))
        assert jnp.isfinite(g), (
            f"{name}: gradient of band flux is {g} -- `nan != 0.0` is True, so "
            "the liveness assertion below cannot see a non-finite gradient (#2178)"
        )
        assert g != 0.0, f"{name}: gradient of band flux is exactly zero"

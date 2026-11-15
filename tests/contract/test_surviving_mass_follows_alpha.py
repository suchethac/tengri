# SPDX-License-Identifier: BSD-3-Clause
"""The surviving stellar mass follows [alpha/Fe] on a 4-D grid that embeds an alpha table (#2250).

A 4-D SSP grid carries ``ssp_mass_remaining`` (n_met, n_age), the alpha = 0 plane, and may carry
``ssp_mass_remaining_alpha`` (n_met, n_alpha, n_age). With the alpha table present the surviving
mass is the contraction with the table collapsed at ``met_alpha_fe``. Without it the 2-D table is
used at every alpha, as before (documented in test c).
"""

from __future__ import annotations

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import (
    SSPData,
    load_ssp_data,
    predict_surviving_mass,
)
from tengri.config.exceptions import TengriIOError
from tengri.parameters import Fixed
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.contract

ALPHA_NODES = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
N_MET, N_ALPHA, N_AGE, N_WAVE = 3, 5, 20, 300
# Base flux-free fixture values (the loader/pytree tests need only the axes).
MR_2D = np.tile(np.linspace(0.9, 0.5, N_AGE), (N_MET, 1))  # below 1 at every alpha node
MR_ALPHA = MR_2D[:, None, :] * (1.0 + 0.05 * ALPHA_NODES[None, :, None])
LGMET = np.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN
AGES_GYR = np.logspace(-1.0, 1.14, N_AGE)


def _ssp(*, alpha_table=True, flux_alpha=True) -> SSPData:
    """A small synthetic SSP grid; 4-D (with [alpha/Fe]) or 3-D at alpha = 0."""
    wave = jnp.linspace(3000.0, 10000.0, N_WAVE)
    ages = jnp.linspace(-1.0, 1.14, N_AGE)
    feature = jnp.exp(-0.5 * ((wave - 5175.0) / 40.0) ** 2)
    base = jnp.full((N_MET, N_AGE, N_WAVE), 1e-3) * (1.0 - 0.2 * feature)[None, None, :]
    if not flux_alpha:
        return SSPData(
            ssp_wave=wave,
            ssp_flux=base,
            ssp_lg_age_gyr=ages,
            ssp_lgmet=jnp.asarray(LGMET),
            ssp_mass_remaining=jnp.asarray(MR_2D),
        )
    flux = jnp.stack(
        [base * (1.0 + 0.1 * a) for a in ALPHA_NODES],
        axis=1,
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=jnp.asarray(LGMET),
        ssp_mass_remaining=jnp.asarray(MR_2D),
        ssp_alpha_fe=jnp.asarray(ALPHA_NODES),
        ssp_mass_remaining_alpha=jnp.asarray(MR_ALPHA) if alpha_table else None,
    )


def _slab3d(ssp4: SSPData, k: int) -> SSPData:
    return SSPData(
        ssp_wave=ssp4.ssp_wave,
        ssp_flux=ssp4.ssp_flux[:, k],
        ssp_lg_age_gyr=ssp4.ssp_lg_age_gyr,
        ssp_lgmet=ssp4.ssp_lgmet,
        ssp_mass_remaining=ssp4.ssp_mass_remaining,
    )


def _model(ssp, *, alpha_free=True):
    observation = tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_g", "sdss_r"]),
    )
    met = {"logzsol": Fixed(-0.3)}
    if alpha_free:
        met["alpha_fe"] = Uniform(-0.2, 0.6)
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
        },
        met=met,
        redshift=Fixed(0.05),
        neb={"type": "ssp"},
    )


def _surviving(model, alpha: float) -> float:
    out = model.predict_properties({"met_alpha_fe": alpha}, names=("stellar_mass_surviving",))
    return float(np.asarray(out["stellar_mass_surviving"]))


@pytest.fixture(scope="module")
def ssp4():
    return _ssp()


def test_surviving_mass_scales_linearly_with_the_alpha_table(ssp4):
    m = _model(ssp4)
    base = _surviving(m, 0.0)
    assert _surviving(m, 0.4) == pytest.approx(1.02 * base, rel=1e-10)


def test_surviving_mass_between_nodes_is_linear_in_alpha(ssp4):
    m = _model(ssp4)
    base = _surviving(m, 0.0)
    assert _surviving(m, 0.3) == pytest.approx(1.015 * base, rel=1e-10)


def test_alpha_zero_on_the_4d_grid_equals_the_3d_slice(ssp4):
    m4 = _model(ssp4)
    m3 = _model(_slab3d(ssp4, 1), alpha_free=False)
    assert _surviving(m4, 0.0) == _surviving_fixed(m3)


def _surviving_fixed(model) -> float:
    out = model.predict_properties({}, names=("stellar_mass_surviving",))
    return float(np.asarray(out["stellar_mass_surviving"]))


def test_grid_without_the_alpha_table_is_alpha_blind_as_before():
    ssp = _ssp(alpha_table=False)
    m = _model(ssp)
    assert _surviving(m, 0.0) == _surviving(m, 0.4)


def test_gradient_with_respect_to_alpha_is_finite_and_matches_fd(ssp4):
    m = _model(ssp4)

    def f(a):
        out = m.predict_properties({"met_alpha_fe": a}, names=("stellar_mass_surviving",))
        return out["stellar_mass_surviving"]

    g = float(jax.grad(f)(0.25))
    h = 1e-4
    fd = float((f(0.25 + h) - f(0.25 - h)) / (2 * h))
    assert np.all(np.isfinite(g)), "a non-finite d(surviving mass)/d(met_alpha_fe)"
    assert np.any(g != 0.0), "an identically zero d(surviving mass)/d(met_alpha_fe)"
    np.testing.assert_allclose(g, fd, rtol=1e-4)
    # The table is S(alpha) = S(0) * (1 + 0.05 alpha), so dS/dalpha = 0.05 S(0): nonzero only
    # when the alpha table is used.
    np.testing.assert_allclose(g, 0.05 * float(f(0.0)), rtol=1e-6)


def test_predict_surviving_mass_takes_alpha(ssp4):
    t_lb = jnp.geomspace(1e6, 1e10, 200)
    sfr = jnp.ones_like(t_lb)
    s0 = predict_surviving_mass(sfr, t_lb, ssp4, alpha_fe=0.0)
    s4 = predict_surviving_mass(sfr, t_lb, ssp4, alpha_fe=0.4)
    assert float(s4) == pytest.approx(1.02 * float(s0), rel=1e-10)


# ---- loader -------------------------------------------------------------------------------


def _write_h5(path, *, alpha_table=None, mr_2d=MR_2D, alpha_shape=None, alpha_fe=ALPHA_NODES):
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, N_WAVE)
        f["ssp_flux"] = np.full((N_MET, N_ALPHA, N_AGE, N_WAVE), 1e-4)
        f["ssp_lg_age_gyr"] = np.log10(AGES_GYR)
        f["ssp_lgmet"] = LGMET
        f["ssp_alpha_fe"] = alpha_fe
        f["ssp_mass_remaining"] = mr_2d
        if alpha_table is not None:
            f["ssp_mass_remaining_alpha"] = alpha_table
        f.attrs["metallicity_axis"] = "total Z: synthetic"
    return path


def test_loader_keeps_both_tables(tmp_path):
    path = _write_h5(tmp_path / "synthetic_alpha_totalz.h5", alpha_table=MR_ALPHA)
    ssp = load_ssp_data(path)
    assert ssp.ssp_mass_remaining_alpha is not None
    assert ssp.ssp_mass_remaining_alpha.shape == (N_MET, N_ALPHA, N_AGE)


def test_loader_refuses_wrong_alpha_table_shape(tmp_path):
    bad = MR_ALPHA[:, :3, :]
    path = _write_h5(tmp_path / "synthetic_alpha_totalz.h5", alpha_table=bad)
    with pytest.raises(TengriIOError, match="ssp_mass_remaining_alpha"):
        load_ssp_data(path)


def test_loader_refuses_alpha_zero_plane_disagreeing_with_2d_table(tmp_path):
    bad = MR_ALPHA.copy()
    bad[:, 1, :] *= 1.01  # alpha = 0 plane is index 1
    path = _write_h5(tmp_path / "synthetic_alpha_totalz.h5", alpha_table=bad)
    with pytest.raises(TengriIOError, match="disagree"):
        load_ssp_data(path)


def test_loader_without_alpha_table_leaves_field_none(tmp_path):
    path = _write_h5(tmp_path / "synthetic_alpha_totalz.h5", alpha_table=None)
    ssp = load_ssp_data(path)
    assert ssp.ssp_mass_remaining_alpha is None
    assert ssp.ssp_alpha_fe is not None


# ---- pytree -------------------------------------------------------------------------------


def test_pytree_round_trips_the_alpha_table(ssp4):
    leaves, treedef = jax.tree_util.tree_flatten(ssp4)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(
        np.asarray(back.ssp_mass_remaining_alpha), np.asarray(ssp4.ssp_mass_remaining_alpha)
    )


def test_3d_ssp_leaf_count_is_unchanged():
    # Leaves of a 3-D grid with a 2-D table: wave, flux, age, lgmet, mass_remaining.
    # The new field is None on 3-D grids and adds no leaf.
    ssp3 = _slab3d(_ssp(), 1)
    leaves = jax.tree_util.tree_leaves(ssp3)
    assert len(leaves) == 5

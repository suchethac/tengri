# SPDX-License-Identifier: BSD-3-Clause

"""
Regression: BC03 age-0 anchor template causes NaN in Q_H tables (#2418).

The BC03 SSP has ssp_lg_age_gyr[0] = -inf. Photoionized nebular backends
build Q_H axes as ssp_lg_age_gyr + 9.0, putting -inf on the interpolation grid.
In _interp_index_weight, dx = grid[1] - (-inf) = inf and w = NaN for every age
in the first interval, spreading NaN through sed_nebular.

Fix: apply ssp_log_age_yr_axis() to floor the anchor at 0.1 Myr, reading the
same ZERO_AGE_ANCHOR_FLOOR_LG_AGE_YR (utils/ssp_anchor.py) as the stellar
surviving-mass floor (#1016), and guard interpolation weights in
_interp_index_weight with isfinite(dx).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri._data_setup import find_data_str
from tengri.components.stellar.sps.dsps_wrapper import SSPData

pytestmark = pytest.mark.regression_bug


_PRSC_GRID = find_data_str("cloudy_grid_prsc.h5")
_CB19_GRID = find_data_str("cb19_templates.h5")

_needs_prsc = pytest.mark.skipif(
    _PRSC_GRID is None,
    reason=(
        "cloudy_grid_prsc.h5 is not shipped in git; set TENGRI_DATA_DIR to a checkout that has it"
    ),
)
_needs_cb19 = pytest.mark.skipif(
    _CB19_GRID is None,
    reason=(
        "cb19_templates.h5 is not shipped in git; set TENGRI_DATA_DIR to a checkout that has it"
    ),
)


@pytest.fixture(scope="module")
def synthetic_anchor_ssp() -> SSPData:
    """Synthetic SSP with BC03-like age-0 anchor template."""
    n_wave = 900
    wave = jnp.logspace(2.0, 11.0, n_wave)
    ages_gyr = jnp.linspace(-3.0, 1.14, 25).at[0].set(-jnp.inf)  # age-0 anchor, BC03 shape
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    age_trend = 1.0 + 0.15 * (jnp.nan_to_num(ages_gyr, neginf=-3.0) + 1.0)
    flux = (
        ((5000.0 / wave) ** 2)[None, None, :]
        * age_trend[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=ages_gyr,
        ssp_lgmet=lgmet,
    )


def test_axis_helper_floors_the_anchor_and_is_a_noop_elsewhere() -> None:
    """ssp_log_age_yr_axis floors -inf anchor and leaves others alone."""
    from tengri.components.nebular._shared import ssp_log_age_yr_axis
    from tengri.utils.ssp_anchor import ZERO_AGE_ANCHOR_FLOOR_LG_AGE_YR

    # Test anchor flooring
    inp = jnp.array([-jnp.inf, -3.9, -3.85, 1.3])
    out = ssp_log_age_yr_axis(inp)

    assert np.isfinite(np.asarray(out)).all(), "All nodes must be finite"
    assert np.all(np.diff(np.asarray(out)) > 0), "Axis must be strictly ascending"
    assert float(out[0]) == 5.0, f"Anchor floor: {out[0]} != 5.0"
    assert float(out[0]) == ZERO_AGE_ANCHOR_FLOOR_LG_AGE_YR
    assert np.allclose(np.asarray(out)[1:], [5.1, 5.15, 10.3], atol=1e-12)

    # Test no-op on finite input
    inp_finite = jnp.array([-3.5, -3.0, 1.1])
    out_finite = ssp_log_age_yr_axis(inp_finite)
    assert np.array_equal(np.asarray(out_finite), np.asarray(inp_finite) + 9.0)


def test_qh_bilinear_is_finite_across_a_non_finite_first_node() -> None:
    """_qh_bilinear handles -inf grid node without NaN weights."""
    from tengri.components.nebular._shared import _qh_bilinear

    table = jnp.ones((2, 4))
    met = jnp.array([-2.0, -1.0])
    age = jnp.array([-jnp.inf, 5.1, 6.0, 7.0])

    for query_age in [-jnp.inf, 5.0, 5.05]:
        q = _qh_bilinear(table, met, age, -1.5, query_age, missing=0.0)
        assert np.isfinite(float(q)), f"Query {query_age} gave non-finite weight"
        assert float(q) == pytest.approx(1.0, abs=1e-6)


def test_mappings_precompute_floors_the_axis(synthetic_anchor_ssp: SSPData) -> None:
    """MappingsPhotoStellarBackend._precompute_qh floors the age axis."""
    from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend

    b = MappingsPhotoStellarBackend.__new__(MappingsPhotoStellarBackend)
    b._precompute_qh(synthetic_anchor_ssp)

    assert np.isfinite(np.asarray(b._qh_log_age)).all(), "Q_H axis must be finite"
    assert float(b._qh_log_age[0]) == 5.0, f"Floor: {b._qh_log_age[0]} != 5.0"
    assert np.all(np.diff(np.asarray(b._qh_log_age)) > 0), "Axis must be ascending"


def test_cb19_precompute_floors_the_axis(synthetic_anchor_ssp: SSPData) -> None:
    """CB19Backend._precompute_qh floors the age axis; the anchor query is finite."""
    from tengri.components.nebular.cloudy_cb19 import CB19Backend

    b = CB19Backend.__new__(CB19Backend)
    b._max_neb_log_age = 8.0  # what __init__ sets; _precompute_qh reads nothing else
    b._precompute_qh(synthetic_anchor_ssp)

    assert np.isfinite(np.asarray(b._qh_log_age)).all(), "Q_H axis must be finite"
    assert float(b._qh_log_age[0]) == 5.0, f"Floor: {b._qh_log_age[0]} != 5.0"
    q_anchor = float(b._get_qh_at(-2.65, -jnp.inf))
    q_floor = float(b._get_qh_at(-2.65, 5.0))
    assert np.isfinite(q_anchor), "Query at the anchor gave NaN"
    assert q_anchor == pytest.approx(q_floor, abs=1e-12)
    assert 0.0 < q_anchor <= 1.0, f"peak-normalized Q_H out of (0, 1]: {q_anchor}"


@_needs_prsc
def test_cloudy_grid_backend_on_synthetic_anchor_ssp(
    synthetic_anchor_ssp: SSPData,
) -> None:
    """CloudyGridBackend handles anchor without NaN; forward model is finite."""
    from tengri.components.nebular.cloudy_grid import CloudyGridBackend

    b = CloudyGridBackend(_PRSC_GRID, synthetic_anchor_ssp)
    assert float(b._qh_log_age[0]) == 5.0

    q_anchor = float(b._get_qh_at(-2.65, -jnp.inf))
    q_floor = float(b._get_qh_at(-2.65, 5.0))
    assert np.isfinite(q_anchor), "Query at anchor gave NaN"
    assert np.isfinite(q_floor), "Query at floor gave NaN"
    assert q_anchor == pytest.approx(q_floor, abs=1e-6)

    # Forward model
    model = tengri.SEDModel.build(
        ssp_data=synthetic_anchor_ssp,
        sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={"type": "none"},
        neb={"type": "cloudy", "grid": _PRSC_GRID, "all_params": tengri.Fixed(tengri.DEFAULT)},
        redshift=tengri.Fixed(0.0),
    )
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    state = model.predict_state(params)
    sed = np.asarray(state.derived["sed_nebular"])

    assert np.isfinite(sed).all(), "sed_nebular contains NaN"
    assert sed.max() > 0, "sed_nebular is all-zero"


@_needs_prsc
def test_cloudy_grid_on_bc03_repro(ssp_data_bc03: SSPData) -> None:
    """Original defect: BC03 + Cloudy gives NaN nebular SED."""
    model = tengri.SEDModel.build(
        ssp_data=ssp_data_bc03,
        sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        dust_emission={"type": "draine_li2014", "all_params": tengri.Fixed(tengri.DEFAULT)},
        neb={"type": "cloudy", "grid": _PRSC_GRID, "all_params": tengri.Fixed(tengri.DEFAULT)},
        radio={
            "sf": {"type": "bell2003"},
            "agn": {"type": "powerlaw"},
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        redshift=tengri.Fixed(0.0),
    )
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    state = model.predict_state(params)
    sed = np.asarray(state.derived["sed_nebular"])

    assert np.isfinite(sed).all(), "sed_nebular contains NaN"
    assert (sed[np.asarray(state.wave) > 1e8] > 0).all(), "Radio tail is zero"


@_needs_cb19
def test_cb19_on_bc03_is_finite(ssp_data_bc03: SSPData) -> None:
    """CB19 backend handles BC03 anchor without NaN."""
    from tengri.components.nebular.cloudy_cb19 import CB19DegenerateGridError

    try:
        model = tengri.SEDModel.build(
            ssp_data=ssp_data_bc03,
            sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={"type": "none"},
            neb={"type": "cb19", "grid": _CB19_GRID, "all_params": tengri.Fixed(tengri.DEFAULT)},
            redshift=tengri.Fixed(0.0),
        )
    except CB19DegenerateGridError as exc:
        assert "flat placeholder" in str(exc), (
            f"Unexpected CB19DegenerateGridError (not the placeholder): {str(exc)[:120]}"
        )
        pytest.skip(f"cb19 grid resolved to the flat placeholder: {str(exc)[:120]}")

    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    state = model.predict_state(params)
    sed = np.asarray(state.derived["sed_nebular"])

    assert np.isfinite(sed).all(), "sed_nebular contains NaN"


@_needs_prsc
def test_gradient_through_the_floored_axis_is_finite_and_nonzero(
    synthetic_anchor_ssp: SSPData,
) -> None:
    """Gradient of sed_nebular w.r.t. logU is finite and nonzero."""
    model = tengri.SEDModel.build(
        ssp_data=synthetic_anchor_ssp,
        sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={"type": "none"},
        neb={
            "type": "cloudy",
            "grid": _PRSC_GRID,
            "logU": tengri.FREE,
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        redshift=tengri.Fixed(0.0),
    )
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    assert "neb_logU" in params
    # The declared prior Uniform(-5, 0) is wider than the grid's log U axis (-4 to -1); the
    # seed-0 draw (-0.29) sits above it, where the lookup clips and the gradient is 0.
    params["neb_logU"] = jnp.asarray(-2.5)

    # Find the 5000 A node
    wave_rest = np.asarray(model._rest_wavelength)
    idx = np.argmin(np.abs(wave_rest - 5000.0))

    def loss(p):
        return model.predict_state(p).derived["sed_nebular"][idx]

    g = float(jax.grad(loss)(params)["neb_logU"])

    assert np.isfinite(g), f"Gradient is non-finite: {g}"
    assert g != 0.0, f"Gradient is exactly zero: {g}"


def test_floor_is_one_constant_shared_with_the_stellar_path() -> None:
    """The nebular Q_H floor and the #1016 surviving-mass floor are one object, not two 5.0s."""
    from tengri.components.nebular._shared import (
        ZERO_AGE_ANCHOR_FLOOR_LG_AGE_YR as nebular_floor,
    )
    from tengri.components.stellar.sps.dsps_wrapper import (
        ZERO_AGE_ANCHOR_FLOOR_LG_AGE_YR as stellar_floor,
    )
    from tengri.utils.ssp_anchor import ZERO_AGE_ANCHOR_FLOOR_LG_AGE_YR as home_floor

    assert nebular_floor is home_floor, "nebular _shared split from utils/ssp_anchor.py"
    assert stellar_floor is home_floor, "dsps_wrapper split from utils/ssp_anchor.py"
    assert home_floor == 5.0

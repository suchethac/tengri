# SPDX-License-Identifier: BSD-3-Clause
"""#2460: neb_logU prior narrowed to the Cloudy grid's log U axis.

The shared ``neb_logU`` declaration is ``Uniform(-5, 0)``; every shipped
``cloudy_grid_*.h5`` interpolates log U on seven nodes, ``[-4, -1]``, and the
lookup clips outside them (``jnp.clip`` is flat there): the SED is
bit-identical across the excess and the gradient is exactly zero (the #1586
mechanism, on a nebular backend). ``components/grid_support.py`` registers
``("neb", "cloudy")`` (and ``"cb19"``/``"mappings"``/``"mappings_agn"``, which
clip the same way) so the build narrows a free ``neb_logU`` prior to the
grid's actual axis and warns when a value cannot be narrowed away from the
dead region.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed, Observation, Photometry
from tengri._data_setup import find_data_str
from tengri.components.grid_support import GRID_SUPPORT, grid_support
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import GridSupportWarning

pytestmark = pytest.mark.regression_bug

_PRSC_GRID = find_data_str("cloudy_grid_prsc.h5")

_needs_prsc = pytest.mark.skipif(
    _PRSC_GRID is None,
    reason=(
        "cloudy_grid_prsc.h5 is not shipped in git; set TENGRI_DATA_DIR to a checkout that has it"
    ),
)


@pytest.fixture(scope="module")
def synthetic_ssp() -> SSPData:
    """Small synthetic SSP; only Q_H matters here, not the stellar continuum.

    Ages span log10(age/Gyr) in [-3, 1.14] like ``test_bug_2418``'s
    ``synthetic_anchor_ssp`` -- old enough at the young end to keep the const
    SFH within cosmic time at ``redshift=Fixed(0.0)``.
    """
    n_wave = 400
    wave = jnp.logspace(2.0, 5.0, n_wave)
    ages_gyr = jnp.linspace(-3.0, 1.14, 15)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    flux = jnp.broadcast_to((5000.0 / wave) ** 2, (3, 15, n_wave))
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=ages_gyr,
        ssp_lgmet=lgmet,
    )


def _prsc_model(ssp, **neb_overrides):
    obs = Observation(photometry=Photometry.from_names(["sdss_g"]))
    neb = {"type": "cloudy", "grid": _PRSC_GRID, "all_params": Fixed(DEFAULT)}
    neb.update(neb_overrides)
    return tengri.SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        neb=neb,
        redshift=Fixed(0.0),
    )


@_needs_prsc
def test_neb_logu_narrowed_to_grid_axis(synthetic_ssp):
    """A free neb_logU prior is narrowed from Uniform(-5, 0) to the grid axis."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _prsc_model(synthetic_ssp, logU=FREE)

    dist = model.spec.get_distribution("neb_logU")
    expected_lo, expected_hi = grid_support("neb", "cloudy")["neb_logU"]
    assert abs(dist.bounds[0] - expected_lo) < 1e-9
    assert abs(dist.bounds[1] - expected_hi) < 1e-9
    # Pinned to the actual shipped axis, not just self-consistent with
    # grid_support(): a narrowing bug that reads the wrong file would pass
    # the check above while still narrowing to the wrong numbers.
    assert abs(dist.bounds[0] - (-4.0)) < 1e-9
    assert abs(dist.bounds[1] - (-1.0)) < 1e-9


@_needs_prsc
def test_gradient_at_seed0_draw_is_finite_and_nonzero(synthetic_ssp):
    """Gradient of sed_nebular w.r.t. neb_logU at the seed-0 draw, post-fix.

    Before #2460, ``model.spec.sample(PRNGKey(0))`` drew ``neb_logU=-0.29``
    from the unnarrowed ``Uniform(-5, 0)`` -- inside the dead region above
    the grid's ``-1`` edge, where the gradient is exactly 0.0. After the fix
    the free prior is narrowed to the grid axis, so the same seed's draw
    lands inside it and needs no manual pin.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _prsc_model(synthetic_ssp, logU=FREE)

    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    assert "neb_logU" in params
    draw = float(params["neb_logU"])
    assert -4.0 <= draw <= -1.0, f"seed-0 draw {draw} is outside the narrowed grid axis"

    wave_rest = np.asarray(model._rest_wavelength)
    idx = np.argmin(np.abs(wave_rest - 5000.0))

    def loss(p):
        return model.predict_state(p).derived["sed_nebular"][idx]

    g = float(jax.grad(loss)(params)["neb_logU"])
    assert np.isfinite(g), f"Gradient is non-finite: {g}"
    assert g != 0.0, f"Gradient is exactly zero: {g}"


@_needs_prsc
def test_neb_logu_fixed_outside_grid_warns(synthetic_ssp):
    """Fixed(-0.5) outside the grid axis warns (does not silently clip mute).

    Not a raise: :class:`~tengri.config.exceptions.GridSupportWarning`'s own
    docstring states the #1586 policy deliberately -- "overhanging a grid is
    wasteful but not ill-posed" -- and that policy is unchanged by this fix.
    What #2460 fixes is that the warning previously never fired for any
    "neb" selector at all (``Parameters._selected_grid_components`` did not
    know about it); it now does, with the grid extent named in the message.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _prsc_model(synthetic_ssp, logU=Fixed(-0.5))

    grid_warnings = [w for w in caught if issubclass(w.category, GridSupportWarning)]
    assert len(grid_warnings) == 1, [str(w.message) for w in caught]
    msg = str(grid_warnings[0].message)
    assert "neb_logU" in msg
    assert "-0.5" in msg
    assert "-4" in msg and "-1" in msg  # the grid extent

    # Still builds and predicts (the clip is silent numerically, not a raise):
    # the fixed value is bit-identical to the -1.0 edge node.
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    state = model.predict_state(params)
    assert np.isfinite(np.asarray(state.derived["sed_nebular"])).all()


def test_cue_neb_backend_has_no_grid_support():
    """Cue (an emulator, not a grid) is deliberately unregistered (#2460)."""
    assert grid_support("neb", "cue") == {}


def test_neb_grid_support_census():
    """GRID_SUPPORT registers every Cloudy-grid-backed neb type that clips neb_logU.

    Pins the set rather than only checking presence: a regression that drops
    one of the four (e.g. re-introducing the previous "cb19/mappings differ,
    skip them" mistake) is caught even though each one, read in isolation,
    would still look registered.
    """
    neb_keys = {name for selector, name in GRID_SUPPORT if selector == "neb"}
    assert neb_keys == {"cloudy", "cb19", "mappings", "mappings_agn"}

# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2645, closing condition: every public SFH route ends at age(z).

``psb_flex`` and ``psb_suess2022`` (both :func:`psb_continuity_flex`) take the
age of the universe at the evaluation redshift as their oldest fixed-bin edge.
The stellar component and the model's public routes (``predict_sfh``, the SFR
of ``predict_sfh_quantities``) and :func:`sample_sfh_prior` must all receive
it, so that they agree with the component's own published history: none may
form stars older than the universe, and none may run past the component's
support by more than one grid cell. The reference is the component's
``sfr_history`` on ``sfh_grid_lbt_yr``.

Before the injection reached the public routes, ``predict_sfh`` ran to 8.80 Gyr
at z = 0.5 (age(z) = 8.59 Gyr) while the component stopped at 8.49 Gyr.

The support end of a curve is the largest lookback with SFR > 0. A route and
the reference are compared to one grid cell: a linear grid cannot place the end
more finely than its own spacing.

References: Suess et al. 2022, ApJ 935, 146; Wild et al. 2020, MNRAS 494, 529
(Eq. 5); issue #2567 (age-anchored nonparametric bin ladders).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.sfh import sample_sfh_prior
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug

PSB_FAMILIES = ["psb_flex", "psb_suess2022"]
REDSHIFTS = (0.5, 2.0, 6.0)
N_DRAWS = 8


def _build(ssp, sfh_type, redshift):
    sfh_group = {
        "type": sfh_type,
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Fixed(10.0),
        "tlast_gyr": Fixed(0.2),
        "tflex_gyr": Fixed(2.0),
    }
    return SEDModel.build(
        ssp_data=ssp,
        redshift=Fixed(redshift),
        neb={"type": "none"},
        sfh=sfh_group,
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )


def _support_end(t_gyr, sfr) -> float:
    """Largest lookback [Gyr] with SFR > 0, on the grid ``t_gyr``."""
    t_gyr = np.asarray(t_gyr, dtype=float)
    sfr = np.asarray(sfr, dtype=float)
    positive = sfr > 0.0
    assert positive.any(), "the route returned no star formation"
    return float(t_gyr[positive].max())


def _cell_gyr(t_gyr, t_at_gyr) -> float:
    """Spacing of the sorted grid ``t_gyr`` in the cell that contains ``t_at_gyr``."""
    t = np.sort(np.asarray(t_gyr, dtype=float))
    i = int(np.clip(np.searchsorted(t, t_at_gyr), 1, t.size - 1))
    return float(t[i] - t[i - 1])


@pytest.mark.parametrize("z", REDSHIFTS)
@pytest.mark.parametrize("sfh_type", PSB_FAMILIES)
def test_public_routes_end_at_age_of_universe_and_agree(synthetic_ssp_wide, sfh_type, z):
    """The three routes end at age(z), within one grid cell of the component's end."""
    model = _build(synthetic_ssp_wide, sfh_type, z)
    age_gyr = float(age_at_z(z))

    state = model.predict_state({})
    comp_t = np.asarray(state.derived["sfh_grid_lbt_yr"]) / 1e9
    comp_end = _support_end(comp_t, state.derived["sfr_history"])

    out = model.predict_sfh({})
    lin_t = np.asarray(out["t_gyr"], dtype=float)
    lin_cell = _cell_gyr(lin_t, comp_end)
    routes = {
        "predict_sfh sfr_mean": (lin_t, out["sfr_mean"], lin_cell),
        "predict_sfh sfr_full": (lin_t, out["sfr_full"], lin_cell),
    }

    key = jax.random.PRNGKey(0)
    prior_age_yr, curves = sample_sfh_prior(sfh_type, key, n=N_DRAWS, redshift=z)
    prior_t = np.asarray(prior_age_yr, dtype=float) / 1e9
    prior_cell = _cell_gyr(prior_t, comp_end)
    for i, curve in enumerate(np.asarray(curves)):
        routes[f"sample_sfh_prior draw {i}"] = (prior_t, curve, prior_cell)

    for name, (t, sfr, cell) in routes.items():
        end = _support_end(t, sfr)
        assert end <= age_gyr + 1e-6, (
            f"{sfh_type} z={z} {name}: star formation to {end:.3f} Gyr lookback, "
            f"beyond age(z) = {age_gyr:.3f} Gyr"
        )
        assert abs(end - comp_end) <= cell, (
            f"{sfh_type} z={z} {name}: support ends at {end:.3f} Gyr, component at "
            f"{comp_end:.3f} Gyr (one grid cell = {cell:.3f} Gyr)"
        )


@pytest.mark.parametrize("sfh_type", PSB_FAMILIES)
def test_registry_route_refuses_missing_age_universe_yr(synthetic_ssp_wide, sfh_type):
    """The registry composed SFH refuses to run without age(z), no 13.7 Gyr fallback."""
    model = _build(synthetic_ssp_wide, sfh_type, 2.0)
    kw = model._sfh_call_kwargs(model._get_internal_params({}), redshift=2.0)
    assert "age_universe_yr" in kw
    del kw["age_universe_yr"]

    with pytest.raises(ValueError, match="age_universe_yr"):
        model._sfh_fn(model.age_yr, **kw)

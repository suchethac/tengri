# SPDX-License-Identifier: BSD-3-Clause
"""The SFH evaluator against an independent numerical reference (#3505).

``apply``, the fast path and the prior sampler share one evaluator, so comparing them
with each other cannot catch an error in it. Here the expected SSP-age weights and
formed mass are computed in this file from the family's SFR(t) alone:

* the SFR is the registry callable evaluated on a dense lookback grid ``[1 yr, age(z)]``
  at kwargs this file builds from the declared parameters (public to internal map,
  ``*_frac`` timings times age(z), the age anchor, the bin edges), not read back from
  the evaluator;
* each parcel ``SFR dt`` (trapezoid between grid points) is split between the two
  bracketing SSP template ages linearly in ``log10`` age, the interpolation the
  component documents; parcels younger than the youngest template go to it, older
  to the oldest, none lies beyond age(z);
* the weights are the normalized sums; the formed mass is the declared
  ``10**log_total_mass`` where the family declares one, else the integral.

The reference's own accuracy is measured by halving its grid (``REFERENCE_N``); the
tolerances below are set from that convergence and from the evaluator's own quadrature.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar import component as C
from tengri.components.stellar.component import (
    StellarSEDComponent,
    StellarSEDComponentConfig,
    evaluate_sfh,
)
from tengri.components.stellar.sfh._prior_sampling import _bin_edges_for, _prior_draws
from tengri.components.stellar.sfh.registry import (
    _NONPARAM_NAMES,
    SFH_REGISTRY,
    UNVALIDATED_SFH_TYPES,
)
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug

REDSHIFTS = (0.1, 1.5)
N_GRID = 64
NOT_A_FAMILY = ("field",)
#: Additive-only modifiers have no smooth base alone.
MODIFIERS = ("burst", "recent", "recent_ssfr")


def _families():
    return sorted(
        k
        for k in SFH_REGISTRY
        if k not in (*NOT_A_FAMILY, *MODIFIERS, "table") and k not in UNVALIDATED_SFH_TYPES
    )


def _publics(name):
    """Deterministic draw inside the free prior of every declared parameter of ``name``.

    The registry default of most parameters is ``Fixed``, so a draw from the default would
    sit on the default point. The free prior (where declared) moves the point; Gyr twins of
    a ``*_frac`` timing are not drawn (a timing is its fraction or its Gyr value).
    """
    params = SFH_REGISTRY[name].params
    twins = {p.gyr_twin for p in params.values() if p.gyr_twin is not None}
    free = {
        k: p.free_prior for k, p in params.items() if p.free_prior is not None and k not in twins
    }
    draws = _prior_draws([name], jax.random.PRNGKey(11), 1, free, n_grid=N_GRID)
    return {k: v[0] for k, v in draws.items()}


def _kwargs(name, publics, age_gyr, edges):
    """Internal kwargs of the family callable, built here from the declared parameters."""
    spec = SFH_REGISTRY[name]
    kw = {}
    for public, (internal, scale, offset) in spec.internal_param_map.items():
        if public not in publics:
            continue
        twin = spec.params[public].gyr_twin
        if twin is not None:  # a *_frac timing: its Gyr twin is frac * age(z)
            g_internal, g_scale, _ = spec.internal_param_map[twin]
            kw[g_internal] = publics[public] * age_gyr * g_scale
        else:
            kw[internal] = publics[public] * scale + offset
    sig = inspect.signature(spec.fn).parameters
    if "age_universe_yr" in sig:
        kw["age_universe_yr"] = age_gyr * 1e9
    if edges is not None and name in _NONPARAM_NAMES and "bin_edges_gyr" in sig:
        kw["bin_edges_gyr"] = edges
    return spec, kw


def _lookback_grid(t_obs_yr, n):
    """Union of a linear and a log grid on ``[1 yr, t_obs]``: resolves old and young structure."""
    lin = np.linspace(1.0, t_obs_yr, n)
    log = np.geomspace(1.0, t_obs_yr, n)
    return np.unique(np.concatenate([lin, log]))


def reference_age_weights(sfr_fn, t_obs_yr, node_lg_age_yr, n):
    """Normalized SSP-age weights and integrated mass of ``sfr_fn`` on a grid of size ``n``."""
    t = _lookback_grid(t_obs_yr, n)
    s = np.asarray(sfr_fn(jnp.asarray(t)))
    dm = 0.5 * (s[1:] + s[:-1]) * np.diff(t)
    lg = np.log10(0.5 * (t[1:] + t[:-1]))
    k = np.clip(np.searchsorted(node_lg_age_yr, lg) - 1, 0, len(node_lg_age_yr) - 2)
    f = np.clip((lg - node_lg_age_yr[k]) / np.diff(node_lg_age_yr)[k], 0.0, 1.0)
    w = np.zeros(len(node_lg_age_yr))
    np.add.at(w, k, dm * (1.0 - f))
    np.add.at(w, k + 1, dm * f)
    return w / w.sum(), float(dm.sum())


#: Weight tolerance [absolute, weights sum to 1], from the convergence checks below: the
#: reference moves by at most 1.5e-5 when its step is halved (``REFERENCE_TOL``), and the
#: evaluator agrees with it to at most 6.0e-4 once its own quadrature is refined to
#: ``REFINED_FACTOR`` (a step-edge family converges at first order).
REFERENCE_TOL = 5e-5
WEIGHT_TOL = 1e-3
#: At the shipped quadrature the evaluator's own discretization (the budget documented in
#: ``_refine_sfh_table_ages``) is the larger term: measured at most 6.8e-3, on a narrow
#: ``continuity_flex`` draw; it falls to <= 6.0e-4 at the refined factor.
SHIPPED_WEIGHT_TOL = 1e-2
REFINED_FACTOR = 256
REFERENCE_N = 200_000


def _refine(monkeypatch):
    for name in ("INTEGRAND_FACTOR_PARAMETRIC", "INTEGRAND_FACTOR_TABULATED"):
        monkeypatch.setattr(C, name, REFINED_FACTOR)


def _weights(comp, params, ssp):
    w, m, _ages, _young = comp._joint_weights_impl(params, ssp)
    return np.asarray(w).sum(axis=0), float(m)


def _setup(name, z, ssp):
    age_gyr = float(age_at_z(z))
    publics = _publics(name)
    edges = _bin_edges_for([name], z)
    spec, kw = _kwargs(name, publics, age_gyr, edges)
    config = StellarSEDComponentConfig(sfh_model=name, n_grid=N_GRID, sfh_bin_edges_gyr=edges)
    comp = StellarSEDComponent(config=config, ssp_data=ssp)
    params = {**publics, "redshift": jnp.asarray(z), "met_logzsol": jnp.asarray(0.0)}
    return age_gyr, publics, edges, spec, kw, config, comp, params


@pytest.mark.parametrize("z", REDSHIFTS)
@pytest.mark.parametrize("name", _families())
def test_weights_and_mass_match_the_independent_reference(
    synthetic_ssp_wide, monkeypatch, name, z
):
    ssp = synthetic_ssp_wide
    age_gyr, publics, _edges, spec, kw, _cfg, comp, params = _setup(name, z, ssp)
    defaults = {k: float(p.default.default) for k, p in spec.params.items()}
    assert any(float(publics[k]) != defaults[k] for k in publics if k in defaults), (
        "the draw sits on the default point"
    )
    node = np.asarray(ssp.ssp_lg_age_gyr) + 9.0

    def sfr(t):
        return spec.fn(t, **kw)

    ref, _ = reference_age_weights(sfr, age_gyr * 1e9, node, REFERENCE_N)
    ref2, _ = reference_age_weights(sfr, age_gyr * 1e9, node, 2 * REFERENCE_N)
    assert np.max(np.abs(ref - ref2)) <= REFERENCE_TOL, "the reference has not converged"

    shipped, mass = _weights(comp, params, ssp)
    assert np.max(np.abs(shipped - ref2)) <= SHIPPED_WEIGHT_TOL
    _refine(monkeypatch)
    refined, _ = _weights(comp, params, ssp)
    assert np.max(np.abs(refined - ref2)) <= WEIGHT_TOL

    declared = [k for k in publics if k.endswith("log_total_mass")]
    assert len(declared) == 1, f"{name!r} declares no single formed mass: {declared}"
    np.testing.assert_allclose(mass, 10.0 ** float(publics[declared[0]]), rtol=1e-10)


@pytest.mark.parametrize("z", REDSHIFTS)
@pytest.mark.parametrize("name", _families())
def test_evaluator_uses_the_declared_edges_and_timings(synthetic_ssp_wide, name, z):
    """Bin edges are the declared ladder; each timing is its fraction of age(z)."""
    age_gyr, publics, edges, spec, _kw, config, _comp, params = _setup(name, z, synthetic_ssp_wide)
    used = evaluate_sfh(
        config, params, 10.0 ** (np.asarray(synthetic_ssp_wide.ssp_lg_age_gyr) + 9.0), N_GRID
    )
    if edges is not None and name in _NONPARAM_NAMES:
        np.testing.assert_array_equal(np.asarray(used.sfh_kwargs["bin_edges_gyr"]), edges)
        assert edges[-1] == pytest.approx(age_gyr, rel=1e-12), "the ladder ends at age(z)"
    for public, p in spec.params.items():
        if p.gyr_twin is None:
            continue
        np.testing.assert_allclose(
            float(used.sfh_timing_gyr[p.gyr_twin]), float(publics[public]) * age_gyr, rtol=1e-12
        )


@pytest.mark.parametrize("z", REDSHIFTS)
def test_table_matches_the_independent_reference(synthetic_ssp_wide, monkeypatch, z):
    """A tabulated history: SFR is the table in cosmic time, zero outside its support."""
    ssp = synthetic_ssp_wide
    age_gyr = float(age_at_z(z))
    t_nodes = np.linspace(1.0, 10.0, 19) if z < 1.0 else np.linspace(0.5, 4.0, 19)
    s_nodes = 2.0 + 3.0 * np.exp(
        -0.5 * ((t_nodes - 0.4 * t_nodes[-1]) / (0.15 * t_nodes[-1])) ** 2
    )
    node = np.asarray(ssp.ssp_lg_age_gyr) + 9.0

    def sfr(t):
        cosmic = (age_gyr * 1e9 - np.asarray(t)) / 1e9
        return jnp.asarray(np.interp(cosmic, t_nodes, s_nodes, left=0.0, right=0.0))

    ref, _ = reference_age_weights(sfr, age_gyr * 1e9, node, REFERENCE_N)
    ref2, mass_ref = reference_age_weights(sfr, age_gyr * 1e9, node, 2 * REFERENCE_N)
    assert np.max(np.abs(ref - ref2)) <= REFERENCE_TOL
    comp = StellarSEDComponent(
        config=StellarSEDComponentConfig(sfh_model="table", n_grid=N_GRID), ssp_data=ssp
    )
    params = {
        "redshift": jnp.asarray(z),
        "met_logzsol": jnp.asarray(0.0),
        "sfh_t_gyr": jnp.asarray(t_nodes),
        "sfh_sfr": jnp.asarray(s_nodes),
    }
    shipped, mass = _weights(comp, params, ssp)
    assert np.max(np.abs(shipped - ref2)) <= SHIPPED_WEIGHT_TOL
    _refine(monkeypatch)
    refined, _ = _weights(comp, params, ssp)
    assert np.max(np.abs(refined - ref2)) <= WEIGHT_TOL
    np.testing.assert_allclose(mass, float(np.trapezoid(s_nodes, t_nodes)) * 1e9, rtol=1e-10)
    # The reference's trapezoid over a piecewise-linear table: 1.2e-6 measured.
    np.testing.assert_allclose(mass, mass_ref, rtol=1e-5)

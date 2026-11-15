# SPDX-License-Identifier: BSD-3-Clause
"""Class sweep: one SFH evaluation shared by every path (#3505, mechanism M7).

Before this mechanism the stellar weights were computed three times: by
``apply``, by ``compute_joint_weights`` (which refused the binned and tabulated
families), and by ``sample_sfh_prior`` (which refused the GP field). Each copy
drifted. The sweep is parametrized from the registry, not a hand list, so a new
SFH family is covered without editing this file:

* every ``SFH_REGISTRY`` key except the ``field`` modulator, the tabulated
  ``table`` SFH, and one composite (list) SFH;
* the GP field off and on;
* the metallicity models ``delta``, ``ramp``, ``table`` (runtime ``met_history``, tabulated
  SFH only) and ``table_build`` (a Z(t) table pinned on the config, any SFH).

Checked on each combination:

* ``compute_joint_weights`` and the weights ``apply`` publishes agree
  (``joint_weights`` to rtol 1e-12, ``total_mass`` to 1e-10);
* with the field on, each draw's formed mass equals the declared
  ``10**log_total_mass`` (or the table integral) to 1e-10 (#2622, B25);
* the prior curves of :func:`sample_sfh_prior` equal the registry SFR at the same
  kwargs, with the field modulation applied draw by draw.

The contradictory combinations are listed in :data:`CONTRADICTORY` with their
reason, excluded at collection, and pinned by a test, so none is skipped silently.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar import component as C
from tengri.components.stellar.component import age_universe_kwargs
from tengri.components.stellar.sfh import sample_sfh_prior
from tengri.components.stellar.sfh._prior_sampling import _bin_edges_for, _prior_draws
from tengri.components.stellar.sfh.registry import (
    SFH_REGISTRY,
    UNVALIDATED_SFH_TYPES,
    resolve_sfh,
)
from tengri.cosmology import age_at_z
from tengri.parameters.resolve import merge_fixed_params
from tengri.parameters.timing import resolve_timing_fracs
from tengri.protocols.component import ForwardState
from tengri.utils.grid import make_log_age_grid

pytestmark = pytest.mark.regression_bug

Z_OBS = 0.1
N_GRID = 64
#: Cosmic-time nodes of the tabulated history [Gyr]: wholly inside age(z = 0.1).
T_NODES_GYR = np.linspace(1.0, 10.0, 19)
SFR_NODES = 2.0 + 3.0 * np.exp(-0.5 * ((T_NODES_GYR - 4.0) / 1.5) ** 2)
MET_NODES = np.linspace(-1.0, 0.0, 19)
COMPOSITE = ("dpl", "burst")
#: A composite with a binned (non-parametric) member: the build-time ladder must reach it
#: in apply, the fast path and the prior sampler (#3505).
COMPOSITE_BINNED = ("continuity", "burst")
#: Additive-only modifiers: each raises alone (it has no base to act on), so each is
#: swept as a composite with a smooth partner, and its standalone form is listed here.
ADDITIVE_ONLY = {
    "burst": "an additive modifier: alone it has no smooth base, so resolve_sfh refuses it",
    "recent": "an episode that splices over a smooth member: alone it has nothing to splice",
    "recent_ssfr": "an episode that splices over a smooth member: alone it has nothing to splice",
}
MODIFIER_COMPOSITES = [("delayed", "recent"), ("delayed", "recent_ssfr")]
#: ``table`` is the runtime ``met_history`` route (riding the SFH table's time axis);
#: ``table_build`` is a Z(t) table pinned on the config at build, which any SFH takes.
METALLICITY_MODES = ("delta", "ramp", "table", "table_build")
MET_BUILD_LOG_AGE_YR = np.linspace(6.0, 10.14, 16)
MET_BUILD_LOG_Z_ABS = np.linspace(-3.0, -1.6, 16)

#: Registry keys that are not SFH families: the field toggle supplies the modulator.
NOT_A_FAMILY = ("field",)

#: Contradictory input, excluded at collection: only the runtime ``met_history`` route
#: (metallicity ``table``) with an SFH other than ``table``. The library's own reason
#: (``StellarSEDComponentConfig`` build-time refusal, ``METALLICITY_TABLE_SOURCE_MESSAGE``):
#: "That array is read only with sfh_model='table', so with any other SFH the table must
#: be given here." The build-time table (``table_build``) is that other route, swept below.
CONTRADICTORY = {
    (sfh, "table"): (
        "the runtime met_history array is read only with sfh_model='table', so with any "
        "other SFH the table must be given on the config (METALLICITY_TABLE_SOURCE_MESSAGE)"
    )
    for sfh in [
        *(k for k in SFH_REGISTRY if k not in NOT_A_FAMILY),
        COMPOSITE,
        COMPOSITE_BINNED,
        *MODIFIER_COMPOSITES,
    ]
    if sfh != "table"
}


#: No prior to draw from: a tabulated history arrives as runtime arrays (sfh_t_gyr,
#: sfh_sfr), so :func:`sample_sfh_prior` has nothing to sample for it.
PRIOR_EXCLUSIONS = {
    "table": "no prior: the tabulated history arrives as runtime arrays, not draws",
}


def _sfh_cases():
    families = sorted(k for k in SFH_REGISTRY if k not in (*NOT_A_FAMILY, *ADDITIVE_ONLY))
    return [*families, COMPOSITE, COMPOSITE_BINNED, *MODIFIER_COMPOSITES]


def _sweep_cells():
    cells = []
    for sfh in _sfh_cases():
        # Registered but not forward-validated: the builder refuses them by design
        # (UNVALIDATED_SFH_TYPES). They stay in the prior sweep, which needs no builder.
        if isinstance(sfh, str) and sfh in UNVALIDATED_SFH_TYPES:
            continue
        for field in (False, True):
            for met in METALLICITY_MODES:
                if (sfh, met) in CONTRADICTORY:
                    continue
                cells.append((sfh, field, met))
    return cells


def _cell_id(cell):
    sfh, field, met = cell
    name = "+".join(sfh) if isinstance(sfh, tuple) else sfh
    return f"{name}-{'field' if field else 'nofield'}-{met}"


def _sfh_type(sfh, field):
    names = list(sfh) if isinstance(sfh, tuple) else [sfh]
    if field:
        names.append("field")
    return names if len(names) > 1 else names[0]


def _build(ssp, obs, sfh, field, met):
    # ``table_build`` builds with delta metallicity; the build-time table is pinned on
    # the stellar component's config afterwards (see ``_with_build_time_met_table``).
    met_type = "delta" if met == "table_build" else met
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": _sfh_type(sfh, field), "all_params": Fixed(DEFAULT)},
            met={"type": met_type},
            dust_attenuation=None,
            neb={"type": "none"},
            redshift=Fixed(Z_OBS),
            n_grid=N_GRID,
        )


def _params(model, sfh, field, met):
    spec = model.spec
    params = merge_fixed_params(spec, dict(spec.sample(jax.random.PRNGKey(0))))
    if sfh == "table" or met == "table":
        params["sfh_t_gyr"] = jnp.asarray(T_NODES_GYR)
        params["sfh_sfr"] = jnp.asarray(SFR_NODES)
    if met == "table":
        params["met_history"] = jnp.asarray(MET_NODES)
    if field:
        params["sfh_field_psd_sigma"] = jnp.asarray(0.3)
        params["sfh_field_psd_tau_myr"] = jnp.asarray(200.0)
        params["sfh_field_xi"] = jnp.asarray(np.random.default_rng(7).standard_normal(N_GRID))
    return params


def _declared_mass(sfh, params):
    """Declared formed mass [Msun] of this draw, or the table integral for a table.

    A draw declares exactly one ``*log_total_mass`` (the builder's family prefix
    need not match the registry name: ``dense_basis`` declares ``sfh_dbp_``).
    """
    if sfh == "table":
        lbt_yr = (float(age_at_z(Z_OBS)) - T_NODES_GYR[::-1]) * 1e9
        return float(np.trapezoid(SFR_NODES[::-1], lbt_yr))
    keys = [k for k in params if k.endswith("log_total_mass")]
    assert len(keys) == 1, (
        f"expected one declared mass for {sfh!r}, found {keys}; "
        f"params carrying 'total': {sorted(k for k in params if 'total' in k)}"
    )
    return float(10.0 ** params[keys[0]])


def _with_build_time_met_table(stellar):
    """The stellar component with a Z(t) table pinned on its config (no ``met_history``)."""
    config = dataclasses.replace(
        stellar.config,
        metallicity_model="table",
        met_table_log_age_yr=jnp.asarray(MET_BUILD_LOG_AGE_YR),
        met_table_log_z_abs=jnp.asarray(MET_BUILD_LOG_Z_ABS),
    )
    return dataclasses.replace(stellar, config=config)


@pytest.fixture(scope="module")
def built(synthetic_ssp_wide, synthetic_tophat_obs):
    """Memoized (sfh, field, met) -> both paths' outputs, one build per combination."""
    memo: dict = {}

    def get(sfh, field, met):
        key = (sfh, field, met)
        if key not in memo:
            model = _build(synthetic_ssp_wide, synthetic_tophat_obs, sfh, field, met)
            params = _params(model, sfh, field, met)
            stellar = next(
                c for c in model._build_component_chain() if isinstance(c, C.StellarSEDComponent)
            )
            if met == "table_build":
                stellar = _with_build_time_met_table(stellar)
            w_c, m_c, _ages, _yf = stellar._joint_weights_impl(params, synthetic_ssp_wide)
            state = stellar.apply(
                ForwardState(wave=jnp.logspace(3.0, 4.5, 64)), params, synthetic_ssp_wide
            )
            memo[key] = {
                "params": params,
                "w_compute": w_c,
                "m_compute": m_c,
                "w_apply": state.derived["joint_weights"],
                "m_apply": 10.0 ** state.derived["log_mstar_formed"],
            }
        return memo[key]

    return get


@pytest.mark.parametrize("cell", _sweep_cells(), ids=_cell_id)
def test_joint_weights_agree_between_compute_and_apply(built, cell):
    """compute_joint_weights and the weights apply publishes are one evaluation."""
    c = built(*cell)
    np.testing.assert_allclose(
        np.asarray(c["w_compute"]), np.asarray(c["w_apply"]), rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(float(c["m_compute"]), float(c["m_apply"]), rtol=1e-10)


@pytest.mark.parametrize("cell", [c for c in _sweep_cells() if c[1]], ids=_cell_id)
def test_field_draw_forms_the_declared_mass(built, cell):
    """With the field on, this draw forms the declared mass (#2622, owner ruling B25)."""
    c = built(*cell)
    np.testing.assert_allclose(
        float(c["m_apply"]), _declared_mass(cell[0], c["params"]), rtol=1e-10
    )


@pytest.mark.parametrize(
    "sfh", [s for s in _sfh_cases() if s not in PRIOR_EXCLUSIONS], ids=lambda s: str(s)
)
@pytest.mark.parametrize("field", [False, True], ids=["nofield", "field"])
def test_prior_curves_equal_the_registry_sfr(sfh, field):
    """sample_sfh_prior evaluates the registry SFR at the same draws, row by row."""
    grid_yr = jnp.asarray(10.0 ** make_log_age_grid(256))
    family = list(sfh) if isinstance(sfh, tuple) else [sfh]
    fam = [*family, "field"] if field else family
    key = jax.random.PRNGKey(3)
    n = 3
    _age, curves = sample_sfh_prior(fam, key, n=n, age_grid_yr=grid_yr, redshift=Z_OBS)
    draws = _prior_draws(fam, key, n, {}, n_grid=256)
    composite = len(family) > 1
    composed, _, _, _ = resolve_sfh(
        family if composite else family[0], bin_edges_gyr=_bin_edges_for(family, Z_OBS)
    )
    for i in range(n):
        row = {k: v[i] for k, v in draws.items()}
        kwargs: dict = {}
        fracs: dict = {}
        param_map: dict = {}
        for name in family:
            spec = SFH_REGISTRY[name]
            for public, (internal, scale, offset) in spec.internal_param_map.items():
                param_map[public] = (internal, scale, offset)
                if public not in row:
                    continue
                value = row[public] * scale + offset
                kwargs[internal] = value
                if composite:
                    kwargs[public] = value
                if spec.params[public].gyr_twin is not None:
                    fracs[public] = row[public]
        resolve_timing_fracs(kwargs, fracs, param_map, redshift=Z_OBS)
        reference = composed(
            grid_yr, **kwargs, **age_universe_kwargs(family if composite else family[0], Z_OBS)
        )
        if field:
            reference = C._apply_gp_field(reference, row, 256, make_log_age_grid(256))
        ref = np.asarray(reference)
        # Scale-relative: the far-past endpoint is 0 against ~1e-193 roundoff, which a
        # purely relative bound cannot absorb. The bound is 1e-12 of the curve's peak.
        np.testing.assert_allclose(
            np.asarray(curves[i]), ref, rtol=1e-12, atol=1e-12 * float(np.max(np.abs(ref)))
        )


def test_exclusions_name_their_reason():
    """Every excluded combination carries a reason, and the exclusion is still real."""
    assert CONTRADICTORY
    assert all(isinstance(reason, str) and reason for reason in CONTRADICTORY.values())
    assert isinstance(PRIOR_EXCLUSIONS.get("table"), str) and PRIOR_EXCLUSIONS["table"]
    for name, reason in ADDITIVE_ONLY.items():
        assert reason
        with pytest.raises(ValueError):
            resolve_sfh(name)


def test_fast_path_unsupported_map_is_gone():
    """The refusal table of the fast path is deleted, not kept as a second evaluator."""
    assert not hasattr(C, "_FAST_PATH_UNSUPPORTED_SFH_FNS")


def test_build_time_met_table_is_read(built):
    """The build-time Z(t) table reaches the weights: they differ from delta metallicity."""
    delta = np.asarray(built("dpl", False, "delta")["w_apply"])
    table = np.asarray(built("dpl", False, "table_build")["w_apply"])
    assert np.max(np.abs(delta - table)) > 1e-3 * np.max(delta)

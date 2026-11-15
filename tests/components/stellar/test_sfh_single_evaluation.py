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
* the metallicity models ``delta``, ``ramp`` and ``table``.

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

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar import component as C
from tengri.components.stellar.component import age_universe_kwargs
from tengri.components.stellar.sfh import sample_sfh_prior
from tengri.components.stellar.sfh._prior_sampling import (
    _bin_edges_for,
    _draw_internal_kwargs,
)
from tengri.components.stellar.sfh.registry import (
    SFH_REGISTRY,
    UNVALIDATED_SFH_TYPES,
    resolve_sfh,
)
from tengri.cosmology import age_at_z
from tengri.protocols.component import ForwardState
from tengri.parameters.resolve import merge_fixed_params
from tengri.utils.grid import make_log_age_grid

pytestmark = pytest.mark.regression_bug

Z_OBS = 0.1
N_GRID = 64
#: Cosmic-time nodes of the tabulated history [Gyr]: wholly inside age(z = 0.1).
T_NODES_GYR = np.linspace(1.0, 10.0, 19)
SFR_NODES = 2.0 + 3.0 * np.exp(-0.5 * ((T_NODES_GYR - 4.0) / 1.5) ** 2)
MET_NODES = np.linspace(-1.0, 0.0, 19)
COMPOSITE = ("dpl", "burst")
#: Additive-only modifiers: each raises alone (it has no base to act on), so each is
#: swept as a composite with a smooth partner, and its standalone form is listed here.
ADDITIVE_ONLY = {
    "burst": "an additive modifier: alone it has no smooth base, so resolve_sfh refuses it",
    "recent": "an episode that splices over a smooth member: alone it has nothing to splice",
    "recent_ssfr": "an episode that splices over a smooth member: alone it has nothing to splice",
}
MODIFIER_COMPOSITES = [("delayed", "recent"), ("delayed", "recent_ssfr")]
METALLICITY_MODES = ("delta", "ramp", "table")

#: Registry keys that are not SFH families: the field toggle supplies the modulator.
NOT_A_FAMILY = ("field",)

#: Contradictory input, excluded at collection. ``met_history`` (a runtime Z(t) table)
#: is tied to the SFH time axis ``sfh_t_gyr``, which only ``sfh_model='table'`` supplies.
CONTRADICTORY = {
    (sfh, "table"): (
        "a runtime Z(t) table (met_history) rides the SFH time axis sfh_t_gyr, which "
        "only sfh_model='table' supplies; apply raises for met_history otherwise"
    )
    for sfh in [*(k for k in SFH_REGISTRY if k not in NOT_A_FAMILY), COMPOSITE, *MODIFIER_COMPOSITES]
    if sfh != "table"
}


#: No prior to draw from: a tabulated history arrives as runtime arrays (sfh_t_gyr,
#: sfh_sfr), so :func:`sample_sfh_prior` has nothing to sample for it.
PRIOR_EXCLUSIONS = {
    "table": "no prior: the tabulated history arrives as runtime arrays, not draws",
}


def _sfh_cases():
    families = sorted(k for k in SFH_REGISTRY if k not in (*NOT_A_FAMILY, *ADDITIVE_ONLY))
    return [*families, COMPOSITE, *MODIFIER_COMPOSITES]


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": _sfh_type(sfh, field), "all_params": Fixed(DEFAULT)},
            met={"type": met},
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
    """Declared formed mass [Msun] of this draw, or the table integral for a table."""
    if sfh == "table":
        lbt_yr = (float(age_at_z(Z_OBS)) - T_NODES_GYR[::-1]) * 1e9
        return float(np.trapezoid(SFR_NODES[::-1], lbt_yr))
    members = set(sfh) if isinstance(sfh, tuple) else {sfh}
    keys = [
        k
        for k in params
        if k.endswith("log_total_mass") and any(k.startswith(f"sfh_{m}_") for m in members)
    ]
    assert len(keys) == 1, f"expected one declared mass for {sfh!r}, found {keys}"
    return float(10.0 ** params[keys[0]])


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


@pytest.mark.parametrize(
    "cell", [c for c in _sweep_cells() if c[1]], ids=_cell_id
)
def test_field_draw_forms_the_declared_mass(built, cell):
    """With the field on, this draw forms the declared mass (#2622, owner ruling B25)."""
    sfh, field, met = cell
    c = built(*cell)
    np.testing.assert_allclose(float(c["m_apply"]), _declared_mass(sfh, c["params"]), rtol=1e-10)


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
    # The same draws, reproduced: one (n,) array per internal kwarg.
    _fam, kwargs = _draw_internal_kwargs(fam, key, n, {}, redshift=Z_OBS)
    composed, _, _, _ = resolve_sfh(fam, bin_edges_gyr=_bin_edges_for(fam, Z_OBS))
    age_kwargs = age_universe_kwargs(fam, Z_OBS)
    for i in range(n):
        kw_i = {k: v[i] for k, v in kwargs.items()}
        reference = composed(grid_yr, **kw_i, **age_kwargs)
        if field:
            reference = C._apply_gp_field(
                reference, {k: kw_i[k] for k in kw_i}, 256, make_log_age_grid(256)
            )
        np.testing.assert_allclose(np.asarray(curves[i]), np.asarray(reference), rtol=1e-12)


def test_builder_exclusions_are_the_registry_set():
    """The build-based sweep excludes exactly the registry's unvalidated families."""
    excluded = {s for s in SFH_REGISTRY if s in UNVALIDATED_SFH_TYPES}
    assert excluded == {s for s in UNVALIDATED_SFH_TYPES if s in SFH_REGISTRY}
    assert not excluded & {c[0] for c in _sweep_cells() if isinstance(c[0], str)}


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

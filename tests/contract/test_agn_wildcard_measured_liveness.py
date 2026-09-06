# SPDX-License-Identifier: BSD-3-Clause
"""Task 16 (AGNfitter-rX parity plan), item 2: MEASURED (not scoping-derived)
liveness contract for the AGN sub-block wildcard.

``test_agn_subblock_wildcard_scoping.py``'s Q1 test computes BOTH sides of
its "freed == declared" check from the same function
(:func:`tengri.parameters.groups._agn_subblock_declared_params`) -- correct
for catching an accidental disagreement between the wildcard machinery and
its own scoping helper, but blind to the helper itself being wrong (a real
parameter it should own but does not declare). This module closes that gap
two ways, both measured directly on ``predict_photometry``/``sed_agn``,
never by calling the scoping function a second time:

1. ``test_owned_and_live_params_are_all_freed`` -- for every (category,
   type), sweeps every ``agn_*`` name ``_AGN_PARTITION`` assigns to that
   category (not merely the ones the scoping function already claims), and
   asserts none of the names EXCLUDED from the freed set moves the digest.
   The complementary "freed subseteq live" direction is already Q1's own
   liveness check in the other file.
2. ``test_polar_names_owned_by_atten_only_under_polar_dust`` -- R22 (task13
   fix-round-1) made the standalone ``polar_dust`` attenuation block the
   ONE polar-dust mechanism on the composable path: ``agn_polar_ebv``/
   ``agn_polar_oa``/``agn_polar_T``/``agn_polar_beta`` are read ONLY when
   ``atten='polar_dust'`` is selected, for EVERY torus type. Parametrized
   over (torus type x atten type) -- literal encoding of F2/F3 as amended
   by R22, not the pre-R22 leak those facts originally measured. Also
   encodes F3's ``agn_attenuation_ebv`` fact: read only by the two atten
   types whose own signature names it (``qsogen``, ``smc_prevot``).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.config.exceptions import TengriIOError
from tengri.parameters.groups import _AGN_PARTITION

from .test_agn_subblock_wildcard_scoping import (
    _ALL_CASES,
    _make_obs,
    _make_ssp,
    _maybe_skip_grid_gated,
    _registered_types,
)


@pytest.fixture(scope="module")
def ssp():
    return _make_ssp()


@pytest.fixture(scope="module")
def obs():
    return _make_obs()


#: Every agn_* name _AGN_PARTITION assigns to "agn.<category>", read
#: directly off the partition table -- NOT via _agn_subblock_declared_params
#: (that is the function under test in the other module; this module's job
#: is to measure independently of it).
def _owned_names(category: str) -> frozenset[str]:
    owner = f"agn.{category}"
    return frozenset(name for name, group in _AGN_PARTITION.items() if group == owner)


def _build_one_category(ssp_data, observation, category, block_type, *, all_params):
    """Mirrors test_agn_subblock_wildcard_scoping.py::_build exactly (kept
    as a local copy rather than importing a private helper twice removed --
    the fixtures/registries it depends on are re-imported above)."""
    agn = {
        "type": "composable",
        "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
        "nlr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
        "blr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
        "all_params": Fixed(DEFAULT),
        "agn_log_lbol": Fixed(12.0),
        "norm": "independent",
    }
    sub = (
        {"law": "prevot_smc", "all_params": all_params}
        if category == "atten" and block_type == "smc_prevot"
        else {"type": block_type, "all_params": all_params}
    )
    agn[category] = sub
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp_data,
            observation=observation,
            sfh={
                "type": "const",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": 10.0,
                "start_gyr": 1.0,
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            agn=agn,
            redshift=Fixed(1.0),
        )


@pytest.mark.parametrize(
    ("category", "block_type"), _ALL_CASES, ids=[f"{c}/{t}" for c, t in _ALL_CASES]
)
def test_owned_and_live_params_are_all_freed(ssp, obs, category, block_type):
    """live-and-owned subseteq freed: every agn_* name partitioned to this
    category that is NOT in the wildcard's freed set must NOT move
    predict_photometry either -- a declared-reads gap the scoping function
    cannot see by construction (it IS the function being checked)."""
    try:
        model = _build_one_category(ssp, obs, category, block_type, all_params=FREE)
    except (TengriIOError, FileNotFoundError) as exc:
        _maybe_skip_grid_gated(category, block_type, exc)

    free_agn = {p for p in model.spec.free_params if p.startswith("agn_")}
    owned = _owned_names(category)
    not_freed = owned - free_agn
    if not not_freed:
        return  # nothing excluded to check

    def obj(pd):
        return jnp.log(jnp.sum(model.predict_photometry(pd)) + 1e-300)

    # Multi-seed retry (mirrors test_agn_subblock_wildcard_scoping.py's own
    # Q1 liveness check): a real-but-tiny effect can underflow to exactly
    # 0.0 at one unlucky draw. Here the direction of concern is the
    # opposite of Q1's -- a live-but-excluded name that underflows at seed
    # 0 would silently hide a real declared-reads gap -- so "live at ANY
    # seed" is the correct, not merely a cautious, criterion.
    _SEEDS = (0, 1, 2, 3, 4)
    live_but_excluded = []
    for name in sorted(not_freed):
        live_at_any_seed = False
        for seed in _SEEDS:
            p = dict(model.spec.sample(jax.random.PRNGKey(seed)))
            if name not in p:
                break  # not a parameter of this build at all (never registered)
            v0 = jnp.asarray(p[name])
            g = float(jax.grad(lambda v, name=name, p=p: obj({**p, name: v}))(v0))
            if g != 0.0:
                live_at_any_seed = True
                break
        if live_at_any_seed:
            live_but_excluded.append(name)
    assert not live_but_excluded, (
        f"{category}/{block_type}: {live_but_excluded} are owned by agn.{category} "
        f"(per _AGN_PARTITION) and measurably move predict_photometry (nonzero at "
        f"some seed among {_SEEDS}), but the wildcard's freed set excluded them -- "
        f"a declared-reads gap."
    )


#: Every registered torus / attenuation type (excluding 'none').
_ALL_TORUS_TYPES = _registered_types("torus")
_ALL_ATTEN_TYPES = _registered_types("atten")

#: R22 (task13 fix-round-1): the ONE polar-dust mechanism's own names.
_POLAR_NAMES = frozenset({"agn_polar_ebv", "agn_polar_oa", "agn_polar_T", "agn_polar_beta"})

#: F3: agn_attenuation_ebv is read only by the atten types whose own
#: signature names it (smc_prevot_block, qsogen_quasar_ext_block).
_ATTENUATION_EBV_TYPES = frozenset({"qsogen", "smc_prevot"})


def _build_torus_atten(ssp_data, observation, torus_type: str, atten_type: str):
    """Composable build with the given torus TYPE selected (Fixed(DEFAULT)
    -- only the structural choice matters here, not its own wildcard) and
    the atten sub-block's own wildcard FREE, 'norm': 'independent' explicit.
    """
    agn = {
        "type": "composable",
        "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
        "nlr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
        "blr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
        "torus": {"type": torus_type, "all_params": Fixed(DEFAULT)},
        "atten": (
            {"law": "prevot_smc", "all_params": FREE}
            if atten_type == "smc_prevot"
            else {"type": atten_type, "all_params": FREE}
        ),
        "all_params": Fixed(DEFAULT),
        "agn_log_lbol": Fixed(12.0),
        "norm": "independent",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp_data,
            observation=observation,
            sfh={
                "type": "const",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": 10.0,
                "start_gyr": 1.0,
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            agn=agn,
            redshift=Fixed(1.0),
        )


@pytest.mark.parametrize("atten_type", _ALL_ATTEN_TYPES)
@pytest.mark.parametrize("torus_type", _ALL_TORUS_TYPES)
def test_polar_names_owned_by_atten_only_under_polar_dust(ssp, obs, torus_type, atten_type):
    """R22 encoded as a literal expectation, over EVERY (torus, atten) pair:
    agn_polar_ebv/oa/T/beta are freed AND live under atten='polar_dust' for
    every torus type; inert (never freed) under every OTHER atten type, for
    every torus type. agn_attenuation_ebv is freed only under the two atten
    types whose own signature reads it (qsogen, smc_prevot)."""
    try:
        model = _build_torus_atten(ssp, obs, torus_type, atten_type)
    except (TengriIOError, FileNotFoundError) as exc:
        if (torus_type, atten_type) and "Synthesizer AGN" in str(exc):
            pytest.skip(f"{torus_type}/{atten_type}: needs data/synthesizer_grids/: {exc}")
        raise

    free = set(model.spec.free_params)

    if atten_type == "polar_dust":
        missing = _POLAR_NAMES - free
        assert not missing, f"{torus_type}/polar_dust: polar names not freed: {sorted(missing)}"

        def obj(pd):
            return jnp.log(jnp.sum(model.predict_photometry(pd)) + 1e-300)

        # Multi-seed retry (same rationale as Q1's own check in
        # test_agn_subblock_wildcard_scoping.py): a real-but-tiny effect
        # can underflow to exactly 0.0 at one unlucky draw.
        _SEEDS = (0, 1, 2, 3, 4)
        dead = []
        for name in sorted(_POLAR_NAMES):
            live_at_any_seed = False
            for seed in _SEEDS:
                p = dict(model.spec.sample(jax.random.PRNGKey(seed)))
                v0 = jnp.asarray(p[name])
                g = float(jax.grad(lambda v, name=name, p=p: obj({**p, name: v}))(v0))
                if g != 0.0:
                    live_at_any_seed = True
                    break
            if not live_at_any_seed:
                dead.append(name)
        assert not dead, (
            f"{torus_type}/polar_dust: polar names freed but dead (grad=0 at "
            f"every one of {len(_SEEDS)} seeds): {dead}"
        )
    else:
        present = _POLAR_NAMES & free
        assert not present, (
            f"{torus_type}/{atten_type}: polar names freed under a non-polar_dust "
            f"atten type: {sorted(present)}"
        )

    if atten_type in _ATTENUATION_EBV_TYPES:
        assert "agn_attenuation_ebv" in free, (
            f"{torus_type}/{atten_type}: agn_attenuation_ebv not freed, expected live"
        )
    else:
        assert "agn_attenuation_ebv" not in free, (
            f"{torus_type}/{atten_type}: agn_attenuation_ebv unexpectedly freed"
        )

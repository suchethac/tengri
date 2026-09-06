# SPDX-License-Identifier: BSD-3-Clause
"""The Feltre NLR dust-to-metal axis has ONE name: ``agn_nlr_xi_d`` (R41, #2214).

Issue #2214 was filed against a dead entry in ``_AGN_PARTITION``. Tracing it
found the ``feltre_grid.h5`` dust-to-metal axis :math:`\\xi_d` reachable under
two unrelated names:

``agn_nlr_xi_d``
    Declared in ``components/agn/_params.py`` as ``Uniform(0.1, 0.5,
    default=0.3)`` and passed by ``blocks/nlr.py`` into
    ``nlr_cloudy.compute_nlr_sed_feltre`` as ``xi_d=`` -- the name the block
    actually reads.
``neb_xid``
    Declared through an orphan table (``parameters/_builders.py::_AGN_EXTRAS``),
    merged into the registry by a dedicated adapter loop, partitioned to
    ``agn.nlr`` where the partition is only ever consulted for ``agn_``-prefixed
    names, and read by nothing on any live path. Measured at b2a2a4d33: it was
    declared under EVERY composable AGN build (``nlr`` feltre, analytic and none
    alike) and freed by nothing, and nesting it under ``nlr`` was refused.

One quantity, one name. ``neb_xid`` is retired, and writing it anywhere gets a
single loud message naming the replacement and its placement -- the generic
key resolver would otherwise answer it in the ``neb`` group with *"Did you
mean: neb_fdust?"*, a real and entirely unrelated parameter.

The liveness half of R41 (the axis is interpolated, not snapped, so a fit can
move it) lives in the grid-gated tests at the bottom of this module.
"""

from __future__ import annotations

import os
import warnings

import pytest

pytestmark = pytest.mark.contract

from tengri.parameters import DEFAULT, FREE, Fixed, parse_groups

_GRID = os.path.join("data", "feltre_grid.h5")

#: The retired spelling and the short form the sub-block grammar would give it.
_RETIRED_SPELLINGS = ("neb_xid", "xid")


def _parse(**groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_groups(
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.5),
            **groups,
        )


def _composable(nlr_type: str = "feltre", **extra):
    agn = {
        "type": "composable",
        "norm": "independent",
        "all_params": Fixed(DEFAULT),
        "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
        "nlr": {"type": nlr_type, "all_params": Fixed(DEFAULT)},
    }
    agn.update(extra)
    return agn


# ── One name in the registry and the partition ────────────────────────────


def test_the_retired_name_is_gone_from_the_registry():
    """Two live names for one grid axis is the defect; one is the fix."""
    from tengri.parameters.registry import registry

    reg = registry()
    assert "agn_nlr_xi_d" in reg, "the surviving name must stay registered"
    assert "neb_xid" not in reg, (
        "neb_xid is still registered. The Feltre dust-to-metal axis has one "
        "name, agn_nlr_xi_d, owned by the block that reads it."
    )


def test_the_partition_gives_the_axis_one_owner():
    from tengri.parameters.groups import _AGN_PARTITION

    assert _AGN_PARTITION["agn_nlr_xi_d"] == "agn.nlr"
    assert "neb_xid" not in _AGN_PARTITION, (
        "the partition still carries the retired name -- the dead entry #2214 was filed against"
    )


def test_the_orphan_table_and_its_merge_hook_are_gone():
    """``_AGN_EXTRAS`` held exactly one entry, and it was this one.

    It was a *third* declaration shape (besides ``ParamDeclaration`` and
    ``ParamDef``), the only one with no ``free_prior`` slot, kept alive for a
    single orphan. With the orphan gone the table, the generic
    ``_LAZY_DECL_EXTRAS`` hook that merged it into the AGN bucket, and the
    registry's own adapter loop have nothing left to carry.
    """
    import inspect

    from tengri.parameters import _builders, registry

    assert not hasattr(_builders, "_AGN_EXTRAS"), "the orphan table outlived its orphan"
    assert not hasattr(_builders, "_LAZY_DECL_EXTRAS"), (
        "the bucket-extras merge hook outlived its only entry"
    )
    assert "_AGN_EXTRAS" not in inspect.getsource(registry._walk_param_modules), (
        "the registry still runs an adapter loop for the deleted orphan table"
    )


def test_the_refused_free_prior_ledger_no_longer_lists_it():
    """``tools/check_param_free_priors.py`` is an intent ledger, not a cache.

    A REFUSED entry for a parameter that no longer exists is a claim about
    nothing; the guard's own stale-entry check fails on it.
    """
    import importlib.util
    import pathlib

    here = pathlib.Path(__file__).resolve().parents[2] / "tools" / "check_param_free_priors.py"
    spec = importlib.util.spec_from_file_location("_cpfp", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "neb_xid" not in mod.REFUSED


def test_the_feltre_precompute_reads_the_owned_axis_name():
    from tengri.components.nebular import feltre_precompute

    assert "agn_nlr_xi_d" in feltre_precompute.AXIS_PARAMS
    assert "neb_xid" not in feltre_precompute.AXIS_PARAMS


# ── The retired key is refused loudly, wherever it is written ─────────────


def _message_for(**groups) -> str:
    with pytest.raises(ValueError) as exc_info:
        _parse(**groups)
    message = str(exc_info.value)
    assert "agn_nlr_xi_d" in message, message
    assert "Did you mean" not in message, message
    assert "neb_fdust" not in message, message
    return message


@pytest.mark.parametrize("key", _RETIRED_SPELLINGS)
def test_the_retired_key_in_the_neb_group_names_the_replacement(key):
    """The worst of the three: the generic resolver sent ``neb_xid`` to
    ``neb_fdust``, a real parameter of an unrelated quantity."""
    _message_for(neb={"type": "cloudy", key: Fixed(0.5)})


@pytest.mark.parametrize("key", _RETIRED_SPELLINGS)
def test_the_retired_key_at_the_agn_top_level_names_the_replacement(key):
    _message_for(agn=_composable(**{key: Fixed(0.5)}))


@pytest.mark.parametrize("key", _RETIRED_SPELLINGS)
def test_the_retired_key_under_agn_nlr_names_the_replacement(key):
    agn = _composable()
    agn["nlr"] = {"type": "feltre", key: Fixed(0.5), "all_params": Fixed(DEFAULT)}
    message = _message_for(agn=agn)
    assert "nlr" in message, message


def test_the_message_shows_the_placement_that_works():
    """A refusal that does not spell the working form is half an answer."""
    message = _message_for(neb={"type": "cloudy", "neb_xid": Fixed(0.5)})
    assert "'nlr'" in message, message
    assert "feltre" in message, message


# ── The surviving name still works, at the placement the message names ────


@pytest.mark.parametrize("key", ["agn_nlr_xi_d", "nlr_xi_d"])
def test_the_surviving_name_is_accepted_under_agn_nlr(key):
    agn = _composable()
    agn["nlr"] = {"type": "feltre", key: Fixed(0.42), "all_params": Fixed(DEFAULT)}
    spec = _parse(agn=agn)
    assert spec.get_fixed_values()["agn_nlr_xi_d"] == pytest.approx(0.42, abs=0.0, rel=1e-12)


@pytest.mark.parametrize("nlr_type", ["feltre", "analytic", "none"])
def test_no_composable_build_carries_the_retired_name(nlr_type):
    """``neb_xid`` was declared under every composable AGN, feltre or not.

    The AGN component declares its whole ``agn_*`` superset by design (so any
    registered model can run without missing keys), which is why
    ``agn_nlr_xi_d`` appears under all three. The retired name appeared beside
    it and was read by none of them.
    """
    spec = _parse(agn=_composable(nlr_type=nlr_type))
    assert "agn_nlr_xi_d" in spec.all_params
    assert "neb_xid" not in spec.all_params


@pytest.mark.skipif(not os.path.exists(_GRID), reason="data/feltre_grid.h5 absent")
def test_the_nlr_wildcard_frees_the_axis():
    """``nlr={'type': 'feltre', 'all_params': FREE}`` must reach the axis.

    R34 put the six Feltre axes in the block's own wildcard scope; the seventh
    was excluded because a nearest-neighbor lookup made it measurably dead.
    R41 replaced that lookup with the interpolation the continuous axes already
    used, so the exclusion no longer describes anything.
    """
    agn = _composable()
    agn["nlr"] = {"type": "feltre", "all_params": FREE}
    spec = _parse(agn=agn)
    assert "agn_nlr_xi_d" in spec.free_params, sorted(
        p for p in spec.free_params if p.startswith("agn_nlr")
    )


@pytest.fixture(scope="module")
def _feltre_model():
    """One composable build with ``nlr='feltre'`` and the two axes freed.

    Built once: each Feltre build loads the HDF5 grid, and the measurements
    below only need to vary parameters in the params dict.
    """
    from tengri import SEDModel

    from .test_agn_subblock_wildcard_scoping import _make_ssp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=_make_ssp(),
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "type": "composable",
                "norm": "independent",
                "disc": {"type": "multicolor"},
                "nlr": {
                    "type": "feltre",
                    "agn_nlr_xi_d": FREE,
                    "agn_nlr_logU": FREE,
                    "all_params": Fixed(DEFAULT),
                },
                "agn_log_lbol": Fixed(12.0),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.05),
        )


def _prior_quantiles(name: str, qs=(0.1, 0.3, 0.5, 0.7, 0.9)) -> list[float]:
    """The declared prior's own quantiles, never a raw 0.05-0.95 sweep.

    A sweep in raw units reads a parameter as dead whenever the endpoints sit
    outside its declared support; the quantiles of the declaration are inside
    it by construction.
    """
    from tengri.parameters.registry import registry

    prior = registry()[name].prior
    lo, hi = float(prior.lo), float(prior.hi)
    return [lo + q * (hi - lo) for q in qs]


@pytest.mark.skipif(not os.path.exists(_GRID), reason="data/feltre_grid.h5 absent")
def test_the_axis_carries_a_gradient_at_every_prior_quantile(_feltre_model):
    """R41(b): the axis is interpolated, so a fit can move it.

    Measured at b2a2a4d33 with the nearest-neighbor lookup: **exactly** 0.0 at
    all five quantiles (a piecewise-constant lookup has no gradient anywhere),
    against ~5e-18 for ``agn_nlr_logU`` on the same build. That is a dead
    dimension by construction, not by underflow, which is what kept the axis
    out of the block's wildcard scope.

    ``agn_nlr_logU`` rides along as an in-model control: a build whose whole
    NLR contribution had gone to zero would show a dead xi_d for a reason that
    has nothing to do with the axis.
    """
    import jax
    import jax.numpy as jnp

    model = _feltre_model
    base = dict(model.spec.get_fixed_values())
    base["agn_nlr_xi_d"] = 0.3
    base["agn_nlr_logU"] = -3.0

    def total(name, value):
        params = {**base, name: value}
        return jnp.sum(model.predict_state(params).derived["sed_agn"])

    dead = []
    for name in ("agn_nlr_xi_d", "agn_nlr_logU"):
        for value in _prior_quantiles(name):
            grad = float(jax.grad(lambda v, n=name: total(n, v))(jnp.asarray(value)))
            if grad == 0.0:
                dead.append((name, value))
    assert not dead, f"zero gradient (dead axis) at {dead}"


@pytest.mark.skipif(not os.path.exists(_GRID), reason="data/feltre_grid.h5 absent")
def test_the_axis_moves_the_agn_sed_above_the_consumes_threshold(_feltre_model):
    """The criterion ``AGN_BLOCK_CONSUMES`` itself records: a relative
    ``sed_agn`` change above 1e-6 across the parameter's prior.

    Measured on this build: 0.154 between the 0.3 and 0.7 prior quantiles, the
    same order as the five axes the entry already lists (``agn_nlr_logU``
    0.384, ``agn_nlr_logZ`` 0.479, ``agn_nlr_alpha_pl`` 0.317). Exactly 0.0
    before R41.
    """
    import numpy as np

    model = _feltre_model
    base = dict(model.spec.get_fixed_values())
    base["agn_nlr_logU"] = -3.0

    def sed(xi_d):
        return np.asarray(model.predict_state({**base, "agn_nlr_xi_d": xi_d}).derived["sed_agn"])

    lo_q, hi_q = _prior_quantiles("agn_nlr_xi_d", qs=(0.3, 0.7))
    norm = max(np.max(np.abs(sed(0.3))), 1e-300)
    rel = np.max(np.abs(sed(hi_q) - sed(lo_q))) / norm
    assert rel > 1e-6, f"relative sed_agn change {rel:.3e} is at or below the 1e-6 no-op floor"


def test_the_consumes_entry_lists_the_axis():
    """The measured-live axis must be in the block's declared-reads entry.

    Without it the wildcard scope for ``nlr='feltre'`` excludes a dimension the
    block reads, which is exactly the gap ``AGN_BLOCK_CONSUMES`` exists to
    close.
    """
    from tengri.components.agn.blocks._consumes import AGN_BLOCK_CONSUMES

    assert "agn_nlr_xi_d" in AGN_BLOCK_CONSUMES[("nlr", "feltre")]

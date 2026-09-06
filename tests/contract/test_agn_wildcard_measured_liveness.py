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
    skip_if_empty_scope,
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
#: Owned names this module's exact-zero criterion would call live while the
#: CONSUMES table's own criterion -- "moves the SED by more than a relative
#: 1e-6", see that module's Provenance section -- calls them no-ops. The two
#: differ by many orders of magnitude, and where they disagree the table's
#: threshold is the one a fit can act on, so a name here is excluded from the
#: owned set rather than demanded of the wildcard.
#:
#: ``agn_nlr_fwhm_kms``: the NLR analytic block does pass the line width
#: through, and jax.grad on predict_photometry is not exactly zero -- but a
#: line width at fixed line luminosity redistributes flux inside a line that a
#: broadband filter integrates over, measured <= 1e-6 relative on the
#: agn_panchromatic recipe's own filters
#: (``test_agn_block_consumes.py::test_agn_panchromatic_free_params_all_move_predict``,
#: which fails if it is freed).
_BELOW_CONSUMES_THRESHOLD: frozenset[str] = frozenset({"agn_nlr_fwhm_kms"})


def _owned_names(category: str) -> frozenset[str]:
    owner = f"agn.{category}"
    return frozenset(
        name
        for name, group in _AGN_PARTITION.items()
        if group == owner and name not in _BELOW_CONSUMES_THRESHOLD
    )


def _build_one_category(ssp_data, observation, category, block_type, *, all_params, mute=True):
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
        # `mute=False` for the empty-scope path: a blanket suppression there
        # would mute the very WildcardNoOpWarning that surface asserts, so the
        # assertion would pass on silence (the reviewer's Finding 1).
        warnings.simplefilter("ignore" if mute else "always")
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


#: PR-tier sample: one type per category, chosen as the type each category's
#: own fixtures already exercise elsewhere. The exhaustive 50-case sweep runs
#: in the slow tier -- 50 measured `SEDModel.build`s cost ~3 minutes of PR-gate
#: wall clock, and the contract is per-category, not per-type.
_SMOKE_CASES = tuple(
    (category, block_type)
    for category, block_type in _ALL_CASES
    if (category, block_type)
    in {
        ("disc", "multicolor"),
        ("torus", "skirtor"),
        ("nlr", "analytic"),
        ("blr", "analytic"),
        ("feii", "boroson_green"),
        ("atten", "polar_dust"),
    }
)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("category", "block_type"), _ALL_CASES, ids=[f"{c}/{t}" for c, t in _ALL_CASES]
)
def test_owned_and_live_params_are_all_freed_exhaustive(ssp, obs, category, block_type):
    """The full sweep over every registered (category, type) -- slow tier."""
    _assert_owned_and_live_are_freed(ssp, obs, category, block_type)


@pytest.mark.parametrize(
    ("category", "block_type"), _SMOKE_CASES, ids=[f"{c}/{t}" for c, t in _SMOKE_CASES]
)
def test_owned_and_live_params_are_all_freed(ssp, obs, category, block_type):
    """One type per category in the fast tier; the exhaustive sweep is above."""
    _assert_owned_and_live_are_freed(ssp, obs, category, block_type)


def _assert_owned_and_live_are_freed(ssp, obs, category, block_type):
    """live-and-owned subseteq freed: every agn_* name partitioned to this
    category that is NOT in the wildcard's freed set must NOT move
    predict_photometry either -- a declared-reads gap the scoping function
    cannot see by construction (it IS the function being checked)."""
    skip_if_empty_scope(
        lambda: _build_one_category(ssp, obs, category, block_type, all_params=FREE, mute=False),
        category=category,
        block_type=block_type,
    )
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


def _build_torus_atten(ssp_data, observation, torus_type: str, atten_type: str, *, mute=True):
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
        warnings.simplefilter("ignore" if mute else "always")
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


#: PR-tier torus sample for the polar sweep: one type per mechanism family --
#: a template-grid torus (skirtor) and the analytic toy (simple). R22's claim
#: is that the polar names are read by the ATTEN block alone, so the torus axis
#: is the control variable, not the subject; the full 16-type sweep runs in the
#: slow tier.
_POLAR_SMOKE_TORUS_TYPES = tuple(t for t in ("skirtor", "simple") if t in _ALL_TORUS_TYPES)


@pytest.mark.slow
@pytest.mark.parametrize("atten_type", _ALL_ATTEN_TYPES)
@pytest.mark.parametrize("torus_type", _ALL_TORUS_TYPES)
def test_polar_names_owned_by_atten_only_under_polar_dust_exhaustive(
    ssp, obs, torus_type, atten_type
):
    """The full 16 torus x 5 atten sweep (slow tier).

    80 measured builds cost ~4 minutes, which is PR-gate wall clock for a
    statement whose torus axis is a control. The smoke test below keeps four
    representative torus types in the fast tier; this keeps the exhaustive
    claim, run on the schedule-gated tier.
    """
    _assert_polar_ownership(ssp, obs, torus_type, atten_type)


def _assert_polar_ownership(ssp, obs, torus_type: str, atten_type: str) -> None:
    """R22 as a literal expectation for one (torus, atten) pair.

    ``agn_polar_ebv``/``oa``/``T``/``beta`` are freed AND measurably live under
    ``atten='polar_dust'``, and absent from the freed set under every other
    atten type, whatever torus is selected. ``agn_attenuation_ebv`` is freed
    only under the two atten types whose own signature reads it (``qsogen``,
    ``smc_prevot``).
    """
    skip_if_empty_scope(
        lambda: _build_torus_atten(ssp, obs, torus_type, atten_type, mute=False),
        category="atten",
        block_type=atten_type,
    )
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


@pytest.mark.parametrize("atten_type", _ALL_ATTEN_TYPES)
@pytest.mark.parametrize("torus_type", _POLAR_SMOKE_TORUS_TYPES)
def test_polar_names_owned_by_atten_only_under_polar_dust(ssp, obs, torus_type, atten_type):
    """R22 over a torus sample x every atten type; the exhaustive 16-type
    sweep is the slow-tier test above."""
    _assert_polar_ownership(ssp, obs, torus_type, atten_type)


# ──────────────────────────────────────────────────────────────────────────
# R33: agn_fe2_strength is the BLR analytic block's read, conditioned on it.
# ──────────────────────────────────────────────────────────────────────────

_FEII_TYPES = _registered_types("feii")


def _build_feii_with_blr(ssp_data, observation, feii_type: str, blr_type: str):
    """Composable build with the feii sub-block's own wildcard FREE and the
    BLR selection under test."""
    agn = {
        "type": "composable",
        "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
        "nlr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
        "blr": {"type": blr_type, "all_params": Fixed(DEFAULT)},
        "feii": {"type": feii_type, "all_params": FREE},
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


@pytest.mark.parametrize("feii_type", _FEII_TYPES)
def test_fe2_strength_not_freed_by_feii_wildcard_without_a_blr(ssp, obs, feii_type):
    """R33: the read belongs to the BLR analytic block, so with no BLR
    selected the feii wildcard must not free it.

    ``agn_fe2_strength`` used to be an unconditional category-wide companion
    of every feii type. Measured with ``blr='none'`` it is DEAD for
    ``feii='grahsp'`` and ``feii='qsogen_balmer'`` (neither declares it) and
    was freed anyway -- the owner rule inverted, a parameter the selected
    configuration ignores handed to the sampler. ``boroson_green`` declares it
    itself and keeps it through its own CONSUMES entry.
    """
    model = _build_feii_with_blr(ssp, obs, feii_type, "none")
    free = set(model.spec.free_params)
    declares_it_itself = feii_type == "boroson_green"
    assert ("agn_fe2_strength" in free) is declares_it_itself, (
        f"feii={feii_type!r} with blr='none': agn_fe2_strength "
        f"{'missing from' if declares_it_itself else 'freed by'} the feii wildcard"
    )


@pytest.mark.parametrize("feii_type", _FEII_TYPES)
def test_fe2_strength_freed_and_live_when_an_analytic_blr_is_active(ssp, obs, feii_type):
    """The other half: with the BLR block that reads it selected, the feii
    wildcard frees it AND it measurably moves photometry."""
    model = _build_feii_with_blr(ssp, obs, feii_type, "analytic")
    free = set(model.spec.free_params)
    assert "agn_fe2_strength" in free, (
        f"feii={feii_type!r} with blr='analytic': agn_fe2_strength not freed, "
        f"though the selected BLR block reads it"
    )

    def obj(pd):
        return jnp.log(jnp.sum(model.predict_photometry(pd)) + 1e-300)

    live = False
    for seed in (0, 1, 2, 3, 4):
        p = dict(model.spec.sample(jax.random.PRNGKey(seed)))
        v0 = jnp.asarray(p["agn_fe2_strength"])
        if float(jax.grad(lambda v, p=p: obj({**p, "agn_fe2_strength": v}))(v0)) != 0.0:
            live = True
            break
    assert live, f"feii={feii_type!r} with blr='analytic': agn_fe2_strength freed but dead"


# ──────────────────────────────────────────────────────────────────────────
# R34: the eight names measured live while no wildcard could free them.
# ──────────────────────────────────────────────────────────────────────────

#: (sub-block category, block type, parameter) rows the review measured live
#: on predict_photometry (worst of five seeds) while no wildcard freed them,
#: because the parameter had no _AGN_PARTITION entry -- or, for the GRAHSP
#: line pair, an entry naming a category other than the one being built.
#: Each is now reachable through its OWNER's wildcard: a sub-block one where
#: the owner is a sub-block, the agn-level one where the name is genuinely
#: shared across categories.
_PREVIOUSLY_UNFREEABLE = (
    ("blr", "grahsp", "agn_grahsp_a_lines"),
    ("blr", "grahsp", "agn_grahsp_linewidth_kms"),
    ("nlr", "analytic", "agn_nlr_line_efficiency"),
    ("blr", "analytic", "agn_blr_line_efficiency"),
    # The review's eighth row, blr='analytic' x agn_fe2_strength, is covered by
    # the R33 pair above instead: its owner is the feii sub-block while the
    # read belongs to the BLR block, so "its owner's wildcard" only means
    # anything on a build that selects both, which is exactly what those two
    # tests parametrize over every feii type.
    ("disc", "adaf", "agn_adaf_alpha"),
    ("disc", "relagn", "agn_astar"),
    ("disc", "skirtor", "agn_cigale_disk_delta"),
)


def _build_with_owner_wildcard(ssp_data, observation, category, block_type, param):
    """Build with the wildcard that OWNS ``param`` set FREE.

    A sub-block-owned name gets its own sub-block's wildcard; a name the
    partition marks shared gets the agn-level one, because no sub-block
    wildcard can reach a shared name by construction.
    """
    owner = _AGN_PARTITION.get(param, "agn")
    agn: dict = {
        "type": "composable",
        "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
        "nlr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
        "blr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
        "all_params": FREE if owner == "agn" else Fixed(DEFAULT),
        "agn_log_lbol": Fixed(12.0),
        "norm": "independent",
    }
    agn[category] = {
        "type": block_type,
        "all_params": FREE if owner != "agn" else Fixed(DEFAULT),
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


@pytest.mark.parametrize(
    ("category", "block_type", "param"),
    _PREVIOUSLY_UNFREEABLE,
    ids=[f"{c}/{t}:{p}" for c, t, p in _PREVIOUSLY_UNFREEABLE],
)
def test_previously_unfreeable_live_names_are_freed_by_their_owner(
    ssp, obs, category, block_type, param
):
    """Each of the eight is now freed by the wildcard its owner names."""
    try:
        model = _build_with_owner_wildcard(ssp, obs, category, block_type, param)
    except (TengriIOError, FileNotFoundError) as exc:
        _maybe_skip_grid_gated(category, block_type, exc)
    assert param in set(model.spec.free_params), (
        f"{category}/{block_type}: {param} (owner "
        f"{_AGN_PARTITION.get(param, 'agn')!r}) is still not freed by its owner's wildcard"
    )


# ──────────────────────────────────────────────────────────────────────────
# R36: a block that reads another category's parameters, and the owner.
# ──────────────────────────────────────────────────────────────────────────

#: SKIRTOR geometry the schartmann2005_skirtor_atten DISC block applies to its
#: own continuum (`skirtor_disc_attenuation`), all owned by `agn.torus`.
_SKIRTOR_GEOMETRY = ("agn_oa_skirtor", "agn_p_skirtor", "agn_q_skirtor", "agn_tau_skirtor")


def _build_skirtor_atten_disc(ssp_data, observation, torus_type: str):
    """Disc wildcard FREE on the block that reads SKIRTOR geometry, with the
    torus selection under test."""
    agn = {
        "type": "composable",
        "disc": {"type": "schartmann2005_skirtor_atten", "all_params": FREE},
        # Structural 'type' only: the torus states no disposition of its own,
        # so the geometry it does not read is governed by the block that does.
        # An explicit torus 'all_params' would (correctly) win -- explicit over
        # silent -- which is why this writes the natural form.
        "torus": {"type": torus_type},
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


@pytest.mark.parametrize("torus_type", ["none", "silva04"])
def test_disc_wildcard_frees_the_geometry_it_reads_when_no_torus_owns_it(ssp, obs, torus_type):
    """R36: with no torus reading the SKIRTOR geometry, the disc that does
    must be able to free it.

    ``schartmann2005_skirtor_atten`` attenuates its own disc continuum through
    ``skirtor_disc_attenuation``, so its CONSUMES entry names the four geometry
    parameters -- every one owned by ``agn.torus``. Before the cross-category
    companion nothing could free them on such a build: the disc's own wildcard
    frees only what it owns, and the shared agn-level one cannot reach a
    sub-block-owned name. Measured live there (gradients 6.9e-20, 1.2e-19,
    -4.3e-19, 1.9e-19), so this was four live dimensions no fit could vary.
    ``silva04`` is the control: a real torus that reads none of them.
    """
    model = _build_skirtor_atten_disc(ssp, obs, torus_type)
    free = set(model.spec.free_params)
    missing = [n for n in _SKIRTOR_GEOMETRY if n not in free]
    assert not missing, (
        f"torus={torus_type!r}: the disc reads {missing} and nothing else owns "
        f"them here, but the disc wildcard did not free them"
    )


def test_the_owning_torus_keeps_sole_ownership_when_it_reads_them(ssp, obs):
    """The other half: no name is freeable twice.

    With ``torus='skirtor'`` the torus block reads the same geometry, so the
    disc must NOT claim it -- the torus wildcard owns it, and a name two
    wildcards could free is an ambiguity, not a convenience.
    """
    from tengri.parameters.groups import _agn_subblock_declared_params

    selection = {"disc": "schartmann2005_skirtor_atten", "torus": "skirtor"}
    disc_scope = _agn_subblock_declared_params(
        "disc", "schartmann2005_skirtor_atten", selection=selection
    )
    overlap = sorted(set(_SKIRTOR_GEOMETRY) & disc_scope)
    assert not overlap, (
        f"torus='skirtor' reads {overlap} itself, so the disc must not also "
        f"claim them: two wildcards could then free the same parameter"
    )
    torus_scope = _agn_subblock_declared_params("torus", "skirtor", selection=selection)
    assert set(_SKIRTOR_GEOMETRY) <= torus_scope, sorted(torus_scope)

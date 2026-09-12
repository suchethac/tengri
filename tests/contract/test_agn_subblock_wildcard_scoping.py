# SPDX-License-Identifier: BSD-3-Clause
"""Contract: every registered composable AGN sub-block answers two questions
consistently, across every (category, type) the grammar accepts (task-12
public-API audit, D2/D3 + addenda).

Q1 FREEABLE -- ``{category}={'type': block_type, 'all_params': FREE}`` frees
EXACTLY that type's own declared parameters
(:func:`tengri.parameters.groups._agn_subblock_declared_params`, signature
introspection on the selected composable block function, filtered to the
parameters it OWNS in the partition table), no more, no less; every freed
name is live (nonzero ``jax.grad`` of ``predict_photometry``). A type that
owns no parameters of its own (every knob it reads is shared with other
blocks, or split across categories -- see the module docstring on
``_agn_subblock_declared_params``) must produce an explicit "nothing was
freed" signal, never silence. Since #2187 that signal is a ``ParameterError``
("covers no parameters"); it was a ``WildcardNoOpWarning`` before.

Q2 WIRED -- the selected type must measurably change the AGN SED relative to
some other registered type in the same slot, at otherwise-identical
parameters. Measured on ``predict_state(params).derived['sed_agn']`` (a
``(sum(|SED|), max(SED))`` digest) rather than ``predict_photometry``:
broadband photometry swamps the BLR/FeII categories entirely (measured: EVERY
blr/feii type gives bit-identical photometry across every OTHER type in its
category at these filters -- the same "photometry cannot see it, the SED can"
trap ``test_wildcard_scope_is_variant_aware.py``'s own docstring documents).
No fixed threshold -- "differs" is exact bitwise inequality of the digest, not
a magnitude cutoff (measured torus spread vs cat3d_wind: sum-digest ratios
from ~0.02 to ~1.2 depending on type; a threshold picked to pass the small end
would hide a real regression at the large end).

``params`` is every FREED parameter's own declared-prior MIDPOINT (Task 16,
item 7 refinement -- the peer harness's own rule: prior-quantile points, never
raw defaults), built via ``all_params=FREE`` on the category under test and
evaluated with :func:`_prior_median_params`, NEVER ``Fixed(DEFAULT)``/``{}``.
Task-12's original version used ``all_params=Fixed(DEFAULT)``, which made
``feii='qsogen_balmer'`` measure bit-identical to ``'boroson_green'`` (issue
#2175) and need an ``xfail`` here -- but ``agn_bcnorm``'s declared DEFAULT is
0.0, and that IS upstream's own reference default (Temple's qsogen_balmer at
bcnorm=0 legitimately reproduces boroson_green), so the "identical at
defaults" measurement was an artifact of evaluating every type at the one
point (0) where its OWN distinguishing knob is defined to be a no-op, not a
wiring defect. Evaluating at each type's own prior median instead
(``agn_bcnorm``'s median is 1.0, squarely inside its Balmer-continuum-active
regime) makes qsogen_balmer measurably differ from boroson_green for real, so
the ``xfail`` this module carried is gone.

For ``torus`` and ``feii``, compared against ONE fixed reference type
(``cat3d_wind`` / ``boroson_green``) -- mirroring the addendum's own measured
baseline table exactly: comparing against "any sibling" would let a type pass
trivially by differing from some OTHER sibling while still being a no-op
relative to the specific reference the addendum measured against.
``disc``/``nlr``/``blr``/``atten`` have no such baseline; "differs from at
least one sibling" is used instead (measured non-vacuous for all four
categories).

Skips: ``nlr``/``blr``'s ``synthesizer``/``synthesizer_spectra`` need
``data/synthesizer_grids/test_grid_agn-{nlr,blr}.hdf5``, fetched via
``synthesizer-download --agn-test-grids`` and not shipped in this checkout.
Skipped narrowly on ``TengriIOError`` naming exactly that grid -- never a
broad ``except Exception``. Any other build failure is a real test failure.

Skip breakdown (task-12 fix round 1, item 4 -- named risk (b), answered from
the code, not restated by hand): 10 total across ``test_q1_...`` +
``test_q2_...``. 8 = the four Synthesizer-grid-gated ``(category, type)``
pairs above (``nlr/synthesizer``, ``nlr/synthesizer_spectra``,
``blr/synthesizer``, ``blr/synthesizer_spectra``) x {Q1, Q2} = 8 skips, each
via :func:`_maybe_skip_grid_gated`. 2 = ``test_q2_wired_against_siblings``'s
own reference-is-itself skips: ``torus/cat3d_wind`` (the ``_Q2_REFERENCE``
torus entry) and ``feii/boroson_green`` (the ``_Q2_REFERENCE`` feii entry)
each skip when the parametrized ``block_type`` IS that category's fixed
reference type ("nothing to compare a type against itself"), which happens
exactly once per category that declares a ``_Q2_REFERENCE`` entry.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import TengriIOError
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve
from tengri.parameters.groups import _agn_subblock_declared_params

#: Grammar sub-block key -> AGN_BLOCKS/AGN_BLOCK_CONSUMES category label.
_CONSUMES_CATEGORY: dict[str, str] = {
    "disc": "disc",
    "torus": "torus",
    "nlr": "nlr",
    "blr": "blr",
    "feii": "feii",
    "atten": "attenuation",
}

#: (category, type) pairs that need an external Synthesizer AGN grid this
#: checkout does not ship.
_SYNTHESIZER_GATED = {
    ("nlr", "synthesizer"),
    ("nlr", "synthesizer_spectra"),
    ("blr", "synthesizer"),
    ("blr", "synthesizer_spectra"),
}

#: One fixed reference type per category with a measured addendum baseline;
#: Q2 compares every OTHER type in that category against it. Categories
#: absent here fall back to "differs from at least one sibling" (see
#: test_q2_wired_against_siblings).
_Q2_REFERENCE: dict[str, str] = {
    "torus": "cat3d_wind",
    "feii": "boroson_green",
}


def _registered_types(category: str) -> list[str]:
    cat = _CONSUMES_CATEGORY[category]
    return sorted(n for n in AGN_BLOCKS[cat] if n != "none")


def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


#: UV -> submm, wide enough to catch disc/BBB, torus/dust IR reprocessing,
#: and (for the Q1 grad check only -- Q2 uses the more sensitive sed_agn)
#: NLR/BLR/FeII near-UV-optical features.
_CENTERS = (1500.0, 2500.0, 5000.0, 9000.0, 2.0e4, 1.0e5, 5.0e5, 2.0e6)


def _make_ssp() -> SSPData:
    """Synthetic UV->submm SSP (no data/ssp_*.h5 dependency; mirrors
    tests/conftest.py::synthetic_ssp_wide, kept local and small so this
    module's ~45 builds stay cheap). Factored out of the ``ssp`` fixture
    (task-12 fix round 1, item 2)."""
    n_met, n_age = 3, 25
    wave = jnp.logspace(2.0, 7.0, 1600)  # 100 A - 1 mm
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


def _make_obs() -> Observation:
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in _CENTERS)))


@pytest.fixture(scope="module")
def ssp() -> SSPData:
    return _make_ssp()


@pytest.fixture(scope="module")
def obs() -> Observation:
    return _make_obs()


def _block_grid_support(category: str, block_type: str) -> dict[str, tuple[float, float]]:
    """The block's own template-grid extent per parameter, if it registers one.

    A grid-backed block clips its axes onto the shipped grid, so a gradient
    measured outside that extent is exactly zero however live the parameter is
    (#1586). This lets the liveness probe below ask the question it means to
    ask.
    """
    from tengri.components.grid_support import GRID_SUPPORT

    loader = GRID_SUPPORT.get((f"agn.{category}", block_type))
    if loader is None:
        return {}
    try:
        return dict(loader())
    except (TengriIOError, FileNotFoundError):
        return {}


def _build(ssp_data, observation, category, block_type, *, all_params):
    """One minimal composable-AGN build: disc + nlr + blr always present
    (multicolor/analytic, Fixed(DEFAULT) unless under test) so the category
    under test gets a live disc continuum / BLR-Balmer reference to act on
    -- FeII and BLR-derived quantities are otherwise dead arms, not dead
    parameters (the ``shock_velocity`` trap
    ``test_wildcard_scope_is_variant_aware.py`` documents). Dust present
    (agn_fracAGN>0 without one raises ConfigError); 'norm': 'independent'
    explicit (no cross-block energy coupling to confound the comparison).
    """
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


def _sed_agn_digest(model, params: dict | None = None) -> tuple[float, float]:
    """(sum(|sed_agn|), max(sed_agn)) -- a shape-independent summary so
    cross-type comparison survives disc types that extend the wavelength
    grid (X-ray/radio wings, CLAUDE.md 'tengri extends the wavelength
    axis') and therefore return a differently-shaped sed_agn.

    ``params`` defaults to ``{}`` (every parameter at its Fixed value) --
    Q2 passes :func:`_prior_median_params` instead (item 7 refinement:
    never evaluate at Fixed(DEFAULT))."""
    sed = np.asarray(model.predict_state(params or {}).derived["sed_agn"])
    return (float(np.sum(np.abs(sed))), float(np.max(sed)))


def _prior_median_params(model) -> dict:
    """Every FREE parameter's own declared-prior MIDPOINT (Task 16, item 7
    refinement -- the peer harness's own rule): ``0.5 * (lo + hi)`` of its
    ``get_distribution(name).bounds``, never a Fixed(DEFAULT)/registry
    default. A type whose distinguishing physics only activates away from
    its own declared default (``agn_bcnorm``'s default IS the physics-off
    value, matching upstream's own reference) must be measured somewhere
    inside its live regime to tell "wired" from "identical because both
    sides were evaluated at the off-switch"."""
    out = {}
    for name in model.spec.free_params:
        lo, hi = model.spec.get_distribution(name).bounds
        out[name] = 0.5 * (lo + hi)
    return out


def _cases(categories: tuple[str, ...] = ("disc", "torus", "nlr", "blr", "feii", "atten")):
    out = []
    for category in categories:
        for block_type in _registered_types(category):
            out.append((category, block_type))
    return out


_ALL_CASES = _cases()


def _maybe_skip_grid_gated(category: str, block_type: str, exc: Exception):
    # nlr._resolve_synthesizer_grid raises TengriIOError; blr._resolve_synthesizer_grid
    # (a separate, un-shared copy) raises the builtin FileNotFoundError instead --
    # an inconsistency between the two, not something this test corrects. Narrowed
    # to exactly the four grid-gated cases AND a message naming the missing
    # Synthesizer grid, never a broad `except Exception`.
    if (
        (category, block_type) in _SYNTHESIZER_GATED
        and isinstance(exc, (TengriIOError, FileNotFoundError))
        and "Synthesizer AGN" in str(exc)
    ):
        pytest.skip(
            f"{category}/{block_type}: needs data/synthesizer_grids/ "
            f"(fetch: synthesizer-download --agn-test-grids): {exc}"
        )
    raise exc


#: #2187 / PR #2207 has landed (merged into this branch): EVERY
#: 'all_params: FREE' group now reaches the wildcard adjudicator, and
#: covered-0 is escalated from WildcardNoOpWarning to ParameterError
#: ("covers no parameters"). That change does not touch _wildcard_scopes,
#: _agn_active_param_set, or any AGN scope mechanism -- so the ONLY
#: empty-scope cases here are genuinely parameter-free variants (a type
#: whose every declared param is shared/masking, not owned by this
#: sub-block).
def _expect_empty_scope(build_fn, *, category: str, block_type: str) -> None:
    """Build via ``build_fn()`` and assert the loud, non-silent zero-scope
    signal: ``all_params: FREE`` covering zero of this type's own declared
    parameters raises ``ParameterError`` ("covers no parameters"). Also
    narrows a Synthesizer-grid-gated build failure to a skip, same as
    :func:`_maybe_skip_grid_gated`."""
    from tengri.config.exceptions import ParameterError

    try:
        with pytest.raises(ParameterError, match=r"covers no parameters"):
            build_fn()
    except (TengriIOError, FileNotFoundError) as exc:
        _maybe_skip_grid_gated(category, block_type, exc)


def _build_selection(category: str, block_type: str) -> dict[str, str]:
    """The per-category block selection :func:`_build` actually produces.

    A sub-block's scope is conditioned on what the rest of the build selects
    (R33's feii companion reads the BLR choice; R36's cross-category one reads
    the owning category's choice), so the expected side has to be computed
    against the same selection the model was built with, not a guess.
    """
    selection = {
        "disc": "multicolor",
        "torus": "none",
        "nlr": "analytic",
        "blr": "analytic",
        "feii": "none",
        "atten": "none",
    }
    selection[category] = block_type
    return selection


def skip_if_empty_scope(build_fn, *, category: str, block_type: str) -> bool:
    """Assert the loud empty-scope signal and skip, when the scope IS empty.

    Every surface that builds an ``all_params: FREE`` wildcard over a
    (category, type) whose declared set is empty has to go through
    :func:`_expect_empty_scope`, not merely leave the no-op unasserted, and
    after PR #2207 building such a wildcard raises instead of warning.
    Routing every surface through the one helper keeps that flip a
    single-line change here.

    Which pairs are empty is **measured, not listed** -- the emptiness of a
    scope moves with the ownership partition and with the companion rules
    (R33/R36), and the selection this file pins decides some of them, so a
    written list goes stale silently and reads as a contract the code never
    checks. Measured today, for the record only: ``torus/qsogen``,
    ``nlr/grahsp``, ``blr/grahsp``, ``blr/qsogen``. (``atten/qsogen_smc``,
    listed here through round 2, has owned a parameter since R34 gave it
    ``agn_ebv``.)

    Returns ``True`` after asserting and skipping is not possible (it raises
    ``Skipped``); ``False`` when the scope is non-empty and the caller should
    build normally.
    """
    if _agn_subblock_declared_params(
        category, block_type, selection=_build_selection(category, block_type)
    ):
        return False
    _expect_empty_scope(build_fn, category=category, block_type=block_type)
    pytest.skip(
        f"{category}/{block_type}: declared scope is empty; the loud no-op "
        f"signal is the whole contract here, so there is nothing to measure."
    )
    return True  # pragma: no cover - pytest.skip raises


@pytest.mark.parametrize(
    ("category", "block_type"), _ALL_CASES, ids=[f"{c}/{t}" for c, t in _ALL_CASES]
)
def test_q1_wildcard_frees_exactly_declared_and_live(ssp, obs, category, block_type):
    """Q1: the sub-block's own wildcard frees exactly its declared params,
    all live; an empty declared set produces a loud, not silent, signal
    (:func:`_expect_empty_scope` -- coordination note, PR #2207)."""
    # _build below pins blr='analytic', and that BLR block reads
    # agn_fe2_strength, so the feii sub-block's wildcard legitimately claims it
    # here (R33: the companion is conditioned on the selected BLR block, which
    # is why the selection has to be passed rather than assumed).
    expected = _agn_subblock_declared_params(
        category, block_type, selection=_build_selection(category, block_type)
    )
    assert expected is not None, (
        f"{category}/{block_type}: _agn_subblock_declared_params returned None "
        f"(type not found in AGN_BLOCKS -- should not happen for a grammar-"
        f"validated type)."
    )

    if not expected:
        _expect_empty_scope(
            lambda: _build(ssp, obs, category, block_type, all_params=FREE),
            category=category,
            block_type=block_type,
        )
        return

    try:
        model = _build(ssp, obs, category, block_type, all_params=FREE)
    except (TengriIOError, FileNotFoundError) as exc:
        _maybe_skip_grid_gated(category, block_type, exc)

    free_agn = {p for p in model.spec.free_params if p.startswith("agn_")}
    assert free_agn == expected, (
        f"{category}/{block_type}: wildcard froze {sorted(free_agn)}, declared {sorted(expected)}"
    )

    def obj(pd):
        return jnp.log(jnp.sum(model.predict_photometry(pd)) + 1e-300)

    # Retry a name across a handful of seeds before declaring it dead: a
    # weak-but-real physical effect (e.g. kubota_done's warm-Comptonization
    # component, agn_r_warm_ratio -- measured nonzero at magnitude ~1e-16 to
    # 1e-18 across most sampled points) can underflow to exactly 0.0 at one
    # unlucky draw without being architecturally dead. A name still exactly
    # 0.0 at EVERY one of these seeds is a real, not a floating-point, no-op.
    _SEEDS = (0, 1, 2, 3, 4)
    support = _block_grid_support(category, block_type)
    dead = []
    for name in sorted(free_agn):
        live_at_any_seed = False
        for seed in _SEEDS:
            p = dict(model.spec.sample(jax.random.PRNGKey(seed)))
            v0 = jnp.asarray(p[name])
            if name in support:
                # A template-backed block clips its axes onto the grid, where
                # jnp.clip makes the gradient exactly zero however live the
                # parameter is (#1586). slone_netzer's agn_log_ledd declares
                # Uniform(-2, 0.5) against an axis ending at -1.9586, so 98% of
                # the freed prior is that flat region and five draws land there
                # with probability 0.92: a dead baseline, not a dead parameter.
                # Probe inside the axis instead.
                lo, hi = support[name]
                v0 = jnp.asarray(0.5 * (lo + hi))
            g = float(jax.grad(lambda v, name=name, p=p: obj({**p, name: v}))(v0))
            if g != 0.0:
                live_at_any_seed = True
                break
        if not live_at_any_seed:
            dead.append(name)
    assert not dead, (
        f"{category}/{block_type}: freed but dead (grad=0 on photometry at "
        f"every one of {len(_SEEDS)} seeds): {dead}"
    )


@pytest.mark.parametrize(
    ("category", "block_type"), _ALL_CASES, ids=[f"{c}/{t}" for c, t in _ALL_CASES]
)
def test_describe_agn_block_params_match_wildcard_scope(ssp, obs, category, block_type):
    """Task-12 fix round 1, item 1: ``tengri.describe_agn_block(...)['params']``
    and the ``'all_params': FREE`` wildcard must agree on EXACTLY the same set
    for every registered ``(category, type)`` -- one source
    (:func:`tengri.parameters.groups._agn_subblock_declared_params`), not two.
    Before this fix, ``describe_agn_block``'s ``params`` came from raw
    ``inspect.signature`` over the bare ``AGN_BLOCKS`` callable and listed
    names the wildcard can never free -- e.g. skirtor's shared
    ``agn_cos_inc``/``agn_log_lbol`` and the ``agn.atten``-owned
    ``agn_polar_T``/``agn_polar_beta``/``agn_polar_ebv`` triple it also
    applies. Reuses this module's Q1 build (:func:`_build`)."""
    import tengri

    skip_if_empty_scope(
        lambda: _build(ssp, obs, category, block_type, all_params=FREE),
        category=category,
        block_type=block_type,
    )
    try:
        model = _build(ssp, obs, category, block_type, all_params=FREE)
    except (TengriIOError, FileNotFoundError) as exc:
        _maybe_skip_grid_gated(category, block_type, exc)

    free_agn = {p for p in model.spec.free_params if p.startswith("agn_")}
    # describe_agn_block answers about a block in ISOLATION -- it is given a
    # name and a category, nothing about the rest of a build -- while the
    # wildcard is scoped against the whole selection. Two companion reads make
    # those differ legitimately, in opposite directions: with blr='analytic'
    # pinned by _build the feii wildcard also frees agn_fe2_strength (R33),
    # and with a blr selected that reads them, feii='boroson_green' does NOT
    # claim agn_blr_cf / agn_blr_line_efficiency, which in isolation it would
    # (R36). So the contract is one source consulted twice, not one answer:
    # describe equals the isolation-scoped set, the build equals its own
    # selection-scoped set, and both come from the same function.
    isolated = _agn_subblock_declared_params(category, block_type)
    build_scoped = _agn_subblock_declared_params(
        category, block_type, selection=_build_selection(category, block_type)
    )
    rec = tengri.describe_agn_block(block_type, category=category)
    described = set(rec.get("params", []))
    assert described == isolated, (
        f"{category}/{block_type}: describe_agn_block params {sorted(described)} "
        f"!= the isolation-scoped declared set {sorted(isolated)}"
    )
    assert free_agn == build_scoped, (
        f"{category}/{block_type}: wildcard froze {sorted(free_agn)}, "
        f"selection-scoped declared set {sorted(build_scoped)}"
    )


def _q2_case_id(category: str, block_type: str) -> str:
    return f"{category}/{block_type}"


_Q2_PARAMS = [
    pytest.param(_category, _block_type, id=_q2_case_id(_category, _block_type))
    for _category, _block_type in _ALL_CASES
]


def _q2_build_and_digest(ssp, obs, category: str, block_type: str) -> tuple[float, float]:
    """Build ``category``'s wildcard FREE and evaluate the digest at every
    freed parameter's own prior median (item 7 refinement; never
    Fixed(DEFAULT)/``{}``)."""
    skip_if_empty_scope(
        lambda: _build(ssp, obs, category, block_type, all_params=FREE),
        category=category,
        block_type=block_type,
    )
    model = _build(ssp, obs, category, block_type, all_params=FREE)
    return _sed_agn_digest(model, _prior_median_params(model))


@pytest.mark.parametrize(("category", "block_type"), _Q2_PARAMS)
def test_q2_wired_against_siblings(ssp, obs, category, block_type):
    """Q2: this type's AGN SED must differ from at least one other
    registered type in the same slot (torus/feii: from the fixed reference
    specifically, matching the addendum's measured baseline), each
    evaluated at its OWN freed parameters' prior median (item 7
    refinement)."""
    try:
        this_digest = _q2_build_and_digest(ssp, obs, category, block_type)
    except (TengriIOError, FileNotFoundError) as exc:
        _maybe_skip_grid_gated(category, block_type, exc)

    reference = _Q2_REFERENCE.get(category)
    if reference is not None:
        if block_type == reference:
            pytest.skip(f"{category}/{block_type} IS the reference type; nothing to compare.")
        try:
            ref_digest = _q2_build_and_digest(ssp, obs, category, reference)
        except (TengriIOError, FileNotFoundError) as exc:
            _maybe_skip_grid_gated(category, reference, exc)
        assert this_digest != ref_digest, (
            f"{category}/{block_type}: AGN SED digest is bit-identical to the "
            f"reference {category}/{reference} at each type's own prior-median "
            f"parameters ({this_digest}) -- this type is wired as a no-op "
            f"relative to it."
        )
        return

    siblings = [t for t in _registered_types(category) if t != block_type]
    digests = []
    for sib in siblings:
        if (category, sib) in _SYNTHESIZER_GATED:
            continue
        try:
            digests.append(_q2_build_and_digest(ssp, obs, category, sib))
        except TengriIOError:
            continue
    assert digests, f"{category}/{block_type}: no buildable sibling to compare against"
    assert any(this_digest != d for d in digests), (
        f"{category}/{block_type}: AGN SED digest ({this_digest}) is bit-identical "
        f"to EVERY other {category} type at each type's own prior-median "
        f"parameters -- wired as a no-op."
    )

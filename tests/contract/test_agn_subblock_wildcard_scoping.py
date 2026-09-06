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
freed" signal (``WildcardNoOpWarning``), never silence.

Q2 WIRED -- the selected type must measurably change the AGN SED relative to
some other registered type in the same slot, at otherwise-identical
(all-default) parameters. Measured on ``predict_state({}).derived['sed_agn']``
(a ``(sum(|SED|), max(SED))`` digest) rather than ``predict_photometry``:
broadband photometry swamps the BLR/FeII categories entirely (measured: EVERY
blr/feii type gives bit-identical photometry across every OTHER type in its
category at these filters -- the same "photometry cannot see it, the SED can"
trap ``test_wildcard_scope_is_variant_aware.py``'s own docstring documents).
No fixed threshold -- "differs" is exact bitwise inequality of the digest, not
a magnitude cutoff (measured torus spread vs cat3d_wind: sum-digest ratios
from ~0.02 to ~1.2 depending on type; a threshold picked to pass the small end
would hide a real regression at the large end).

For ``torus`` and ``feii``, compared against ONE fixed reference type
(``cat3d_wind`` / ``boroson_green``) -- mirroring the addendum's own measured
baseline table exactly, and load-bearing for the xfail below: comparing
against "any sibling" would let ``qsogen_balmer`` pass trivially (it differs
from ``grahsp``, just not from ``boroson_green``). ``disc``/``nlr``/``blr``/
``atten`` have no such baseline; "differs from at least one sibling" is used
instead (measured non-vacuous for all four categories).

Known exception: ``feii='qsogen_balmer'`` is bit-identical to
``'boroson_green'`` (issue #2175, a genuine wiring defect Task 15 fixes, NOT
this task) -- ``xfail(strict=True)`` on that ONE Q2 case only, so it flips to
a loud XPASS (and Task 15 removes the marker) the moment the physics is
fixed.

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

import re
import warnings

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
    (task-12 fix round 1, item 2) so
    :func:`test_feii_qsogen_balmer_xfail_reason_is_honest` can build one
    without a pytest fixture context."""
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


def _sed_agn_digest(model) -> tuple[float, float]:
    """(sum(|sed_agn|), max(sed_agn)) -- a shape-independent summary so
    cross-type comparison survives disc types that extend the wavelength
    grid (X-ray/radio wings, CLAUDE.md 'tengri extends the wavelength
    axis') and therefore return a differently-shaped sed_agn."""
    sed = np.asarray(model.predict_state({}).derived["sed_agn"])
    return (float(np.sum(np.abs(sed))), float(np.max(sed)))


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


@pytest.mark.parametrize(
    ("category", "block_type"), _ALL_CASES, ids=[f"{c}/{t}" for c, t in _ALL_CASES]
)
def test_q1_wildcard_frees_exactly_declared_and_live(ssp, obs, category, block_type):
    """Q1: the sub-block's own wildcard frees exactly its declared params,
    all live; an empty declared set produces a loud, not silent, signal."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = _build(ssp, obs, category, block_type, all_params=FREE)
        except (TengriIOError, FileNotFoundError) as exc:
            _maybe_skip_grid_gated(category, block_type, exc)

    expected = _agn_subblock_declared_params(category, block_type)
    assert expected is not None, (
        f"{category}/{block_type}: _agn_subblock_declared_params returned None "
        f"(type not found in AGN_BLOCKS -- should not happen for a grammar-"
        f"validated type)."
    )
    free_agn = {p for p in model.spec.free_params if p.startswith("agn_")}
    assert free_agn == expected, (
        f"{category}/{block_type}: wildcard froze {sorted(free_agn)}, declared {sorted(expected)}"
    )

    if not expected:
        loud = [w for w in caught if "Wildcard" in w.category.__name__]
        assert loud, (
            f"{category}/{block_type}: 'all_params': FREE covers zero of this "
            f"type's own declared parameters, but no WildcardNoOpWarning fired "
            f"-- a silent no-op wildcard."
        )
        return

    def obj(pd):
        return jnp.log(jnp.sum(model.predict_photometry(pd)) + 1e-300)

    # Retry a name across a handful of seeds before declaring it dead: a
    # weak-but-real physical effect (e.g. kubota_done's warm-Comptonization
    # component, agn_r_warm_ratio -- measured nonzero at magnitude ~1e-16 to
    # 1e-18 across most sampled points) can underflow to exactly 0.0 at one
    # unlucky draw without being architecturally dead. A name still exactly
    # 0.0 at EVERY one of these seeds is a real, not a floating-point, no-op.
    _SEEDS = (0, 1, 2, 3, 4)
    dead = []
    for name in sorted(free_agn):
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

    try:
        model = _build(ssp, obs, category, block_type, all_params=FREE)
    except (TengriIOError, FileNotFoundError) as exc:
        _maybe_skip_grid_gated(category, block_type, exc)

    free_agn = {p for p in model.spec.free_params if p.startswith("agn_")}
    rec = tengri.describe_agn_block(block_type, category=category)
    described = set(rec.get("params", []))
    assert described == free_agn, (
        f"{category}/{block_type}: describe_agn_block params {sorted(described)} "
        f"!= wildcard-freed params {sorted(free_agn)}"
    )


def _q2_case_id(category: str, block_type: str) -> str:
    return f"{category}/{block_type}"


_Q2_PARAMS = []
for _category, _block_type in _ALL_CASES:
    if (_category, _block_type) == ("feii", "qsogen_balmer"):
        _Q2_PARAMS.append(
            pytest.param(
                _category,
                _block_type,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "#2175: feii=qsogen_balmer photometry bit-identical to "
                        "boroson_green (max rel diff 0.0) while feii=grahsp's "
                        "sed_agn digest (this module's _sed_agn_digest -- "
                        "sum(|sed_agn|) from predict_state({}).derived['sed_agn'], "
                        "on the module's 8-band UV-submm _CENTERS tophat "
                        "filters) differs from boroson_green's by 7.0996e-03 "
                        "(0.71%) at identical params (addendum's illustrative "
                        "1.438e+00 was a DIFFERENT, photometric example -- see "
                        "test_feii_qsogen_balmer_xfail_reason_is_honest, which "
                        "recomputes and pins this number)"
                    ),
                ),
                id=_q2_case_id(_category, _block_type),
            )
        )
    else:
        _Q2_PARAMS.append(
            pytest.param(_category, _block_type, id=_q2_case_id(_category, _block_type))
        )


def test_feii_qsogen_balmer_xfail_reason_is_honest():
    """Task-12 fix round 1, item 2: the qsogen_balmer xfail reason above
    cites a feii=grahsp-vs-boroson_green sed_agn digest relative difference.
    Recompute that SAME digest with THIS module's own _build/_sed_agn_digest
    (independently of the fixture-scoped ssp/obs -- via _make_ssp/_make_obs,
    so this test does not depend on test ordering or fixture caching) and
    assert it matches the number PARSED BACK OUT of the actual xfail reason
    string -- so editing the literal without updating the measurement (or a
    code change that silently moves the measurement) fails HERE, not just
    reads wrong on inspection."""
    ssp_data = _make_ssp()
    observation = _make_obs()
    grahsp_model = _build(ssp_data, observation, "feii", "grahsp", all_params=Fixed(DEFAULT))
    ref_model = _build(
        ssp_data, observation, "feii", _Q2_REFERENCE["feii"], all_params=Fixed(DEFAULT)
    )
    grahsp_sum, _ = _sed_agn_digest(grahsp_model)
    ref_sum, _ = _sed_agn_digest(ref_model)
    measured = abs(grahsp_sum - ref_sum) / abs(ref_sum)

    xfail_param = next(p for p in _Q2_PARAMS if p.id == _q2_case_id("feii", "qsogen_balmer"))
    xfail_mark = next(m for m in xfail_param.marks if m.name == "xfail")
    reason = xfail_mark.kwargs["reason"]
    match = re.search(r"by (\d\.\d+e-\d+) \(", reason)
    assert match, f"could not find the cited grahsp-vs-reference rel-diff number in: {reason!r}"
    cited = float(match.group(1))

    assert measured == pytest.approx(cited, rel=1e-3), (
        f"measured feii=grahsp-vs-boroson_green sed_agn digest rel diff "
        f"{measured:.6e} no longer matches the {cited:.6e} cited in the "
        f"qsogen_balmer xfail reason above -- update that reason string."
    )


@pytest.mark.parametrize(("category", "block_type"), _Q2_PARAMS)
def test_q2_wired_against_siblings(ssp, obs, category, block_type):
    """Q2: this type's AGN SED must differ from at least one other
    registered type in the same slot (torus/feii: from the fixed reference
    specifically, matching the addendum's measured baseline) at identical
    (all-default) parameters."""
    try:
        this_model = _build(ssp, obs, category, block_type, all_params=Fixed(DEFAULT))
    except (TengriIOError, FileNotFoundError) as exc:
        _maybe_skip_grid_gated(category, block_type, exc)
    this_digest = _sed_agn_digest(this_model)

    reference = _Q2_REFERENCE.get(category)
    if reference is not None:
        if block_type == reference:
            pytest.skip(f"{category}/{block_type} IS the reference type; nothing to compare.")
        try:
            ref_model = _build(ssp, obs, category, reference, all_params=Fixed(DEFAULT))
        except (TengriIOError, FileNotFoundError) as exc:
            _maybe_skip_grid_gated(category, reference, exc)
        ref_digest = _sed_agn_digest(ref_model)
        assert this_digest != ref_digest, (
            f"{category}/{block_type}: AGN SED digest is bit-identical to the "
            f"reference {category}/{reference} at default parameters "
            f"({this_digest}) -- this type is wired as a no-op relative to it."
        )
        return

    siblings = [t for t in _registered_types(category) if t != block_type]
    digests = []
    for sib in siblings:
        if (category, sib) in _SYNTHESIZER_GATED:
            continue
        try:
            sib_model = _build(ssp, obs, category, sib, all_params=Fixed(DEFAULT))
        except TengriIOError:
            continue
        digests.append(_sed_agn_digest(sib_model))
    assert digests, f"{category}/{block_type}: no buildable sibling to compare against"
    assert any(this_digest != d for d in digests), (
        f"{category}/{block_type}: AGN SED digest ({this_digest}) is bit-identical "
        f"to EVERY other {category} type at default parameters -- wired as a no-op."
    )

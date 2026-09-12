# SPDX-License-Identifier: BSD-3-Clause
"""Regression: SKIRTOR torus radius_ratio is wired and polar_beta matches canonical.

Task 14: ``agn_radius_ratio`` was declared on ``SKIRTORTorus`` (the standalone
``SEDModelComponent`` class, ``skirtor_model.py``) but never passed through
``predict()`` to the interpolator -- it was hardcoded to 20.0 inside
``create_skirtor_components_from_grid``, making the declared, guard-registered
parameter a silent no-op on the class. The composable ``skirtor_torus_block``
(``blocks/torus.py``) already wired ``agn_radius_ratio`` correctly before this
task; only the class path was broken.

Also, ``SKIRTORTorus.polar_beta`` was declared at ``[1.0, 2.5]`` while the
canonical ``agn_polar_beta`` declaration is ``[1.0, 2.0]`` (a restatement that
had drifted), and the class's polar-temperature parameter was registered as
``agn_polar_temperature`` while the composable block's own parameter for the
identical physical quantity is ``agn_polar_T`` -- two different registered
names for the same physics on the two paths ("skirtor" registers both a
standalone class and a composable block under the same string; Task 12).

Fix round 2 re-review found the same restated-literal-with-drifted-default
class surviving in two more places after round 1 had already derived ten of
the twelve declared parameters from canonical: ``polar_ebv`` (default 0.1 vs
canonical ``agn_polar_ebv`` 0.03) and ``log_lbol`` (default 11.0 vs canonical
``agn_log_lbol`` 10.0). ``tools/check_param_defaults.py`` cannot see either
(in-range != equal). All twelve now derive from ``declared_prior``, and
``test_skirtor_torus_all_declared_params_match_canonical`` (below) iterates
every declared parameter live so this drift class is impossible for this
class going forward, not merely absent from the names a reviewer happened to
check this round.

This file guards:

1. ``agn_radius_ratio`` changes the SED on the CLASS path
   (``SKIRTORTorus.predict()`` called directly) -- the bug this task exists
   for -- and, separately, via the composable builder (already wired before
   this task; kept as a belt-and-suspenders regression guard).
2. ``polar_beta`` and the renamed ``polar_T`` declarations match the
   canonical ``_params.py`` declarations exactly, and (fix round 2) EVERY
   declared parameter's bounds and default match canonical, with a
   documented allow-list for any deliberate deviation.
3. The class and the composable ``skirtor_torus_block`` expose the SAME
   registered parameter names for every physics parameter they share
   (RULING R12d): a hand-written map cannot notice a future rename, so the
   comparison is computed from live introspection on both sides.
4. ``agn_radius_ratio`` is live (nonzero gradient) on BOTH the class path and
   the composable path. ``agn_polar_T``/``agn_polar_beta`` are live on the
   class path and, on the composable path, ONLY when the standalone
   ``polar_dust`` attenuation block is selected (R22, task13 fix-round-1:
   the composable torus block no longer bundles its own polar-dust term) --
   inert (zero gradient) with ``atten='none'``.
5. The vendored SKIRTOR grid's three ``radius_ratio`` nodes ([10, 20, 30])
   each produce a finite, mutually-distinct SED -- the grid is tracked in
   this repository, so its absence is a test FAILURE, never a skip.
"""

from __future__ import annotations

from inspect import signature

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.components.agn.blocks.torus import skirtor_torus_block
from tengri.components.agn.skirtor import (
    _find_skirtor_grid,
    create_skirtor_components_from_grid,
)
from tengri.components.agn.skirtor_model import SKIRTORTorus, SKIRTORTorusConfig
from tengri.protocols.component import declared_prior

pytestmark = pytest.mark.regression_bug

_SFH = {"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT), "log_total_mass": -10.0}
_DUST = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.Fixed(tengri.DEFAULT),
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}
_DISC = {
    "disc": {"type": "multicolor", "all_params": tengri.Fixed(tengri.DEFAULT)},
    "all_params": tengri.Fixed(tengri.DEFAULT),
    "log_lbol": 12.0,
    "frac": 1.0,
}

#: Prefix-stripped params for a direct ``SKIRTORTorus.predict()`` call, at the
#: canonical defaults declared on the class (Task 14: includes radius_ratio
#: and the renamed polar_T).
_CLASS_PARAMS = {
    "log_lbol": jnp.array(12.0),
    "tau_skirtor": jnp.array(7.0),
    "p_skirtor": jnp.array(1.0),
    "q_skirtor": jnp.array(1.0),
    "oa_skirtor": jnp.array(40.0),
    "radius_ratio": jnp.array(20.0),
    "cos_inc": jnp.array(0.866),
    "torus_frac": jnp.array(0.5),
    "polar_ebv": jnp.array(0.1),
    "polar_T": jnp.array(100.0),
    "polar_beta": jnp.array(1.6),
    "delta": jnp.array(0.0),
}


@pytest.fixture(scope="module")
def ssp():
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")


@pytest.fixture(scope="module")
def skirtor_grid_path() -> str:
    """Resolve the vendored SKIRTOR grid via the package's own loader.

    The grid (``data/skirtor_templates_v3.h5``, 27.5 MB) is tracked in this
    repository, so ``_find_skirtor_grid`` raising ``FileNotFoundError`` is a
    real failure, never a reason to skip.
    """
    return _find_skirtor_grid()


@pytest.fixture(scope="module")
def class_component(skirtor_grid_path):
    """A ``SKIRTORTorus`` instance with templates loaded, ready to ``predict()``."""
    wave = jnp.geomspace(1e3, 1e7, 400)
    comp = SKIRTORTorus(config=SKIRTORTorusConfig(grid_path=skirtor_grid_path))
    object.__setattr__(comp, "data", comp.load(wave))
    return comp


def _sed_with_radius_ratio(ssp, radius_ratio: float) -> np.ndarray:
    model = tengri.SEDModel.build(
        ssp,
        sfh=_SFH,
        dust_attenuation=_DUST,
        agn=dict(
            _DISC,
            torus={
                "type": "skirtor",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "agn_radius_ratio": radius_ratio,
            },
        ),
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    return np.asarray(model.predict(p).rest_sed())


def test_skirtor_radius_ratio_is_not_a_noop_composable(ssp):
    """radius_ratio parameter must change predict() via the composable builder.

    The composable skirtor_torus_block was already wired before this task;
    kept as a belt-and-suspenders regression guard alongside the class-path
    test below, which is the one that catches the bug this task fixes.
    """
    # Grid nodes are [10, 20, 30] per canonical declaration
    s1 = _sed_with_radius_ratio(ssp, 10.0)
    s2 = _sed_with_radius_ratio(ssp, 30.0)
    rel = float(np.abs(s1 - s2).max() / max(np.abs(s1).max(), 1e-99))
    assert rel > 1e-3, (
        f"skirtor radius_ratio is a no-op via the builder (rel diff {rel:.2e}); "
        "wiring is incomplete."
    )


def test_skirtor_torus_class_radius_ratio_is_not_a_noop(class_component):
    """radius_ratio must change SKIRTORTorus.predict() output DIRECTLY.

    This is the guard for the actual bug this task exists for:
    ``SKIRTORTorus.predict()`` never passed ``agn_radius_ratio`` through to
    ``create_skirtor_components_from_grid`` (hardcoded default 20.0 there),
    so the declared, guard-registered class parameter was a silent no-op --
    invisible to the composable-path test above, which exercises a completely
    separate code path (``skirtor_torus_block``, already wired).
    """
    wave = jnp.geomspace(1e3, 1e7, 400)
    sed_in = jnp.zeros_like(wave)

    def _sed(radius_ratio: float) -> np.ndarray:
        p = dict(_CLASS_PARAMS)
        p["radius_ratio"] = jnp.array(radius_ratio)
        sed_out, _ = class_component.predict(p, sed_in, wave)
        return np.asarray(sed_out)

    s1 = _sed(10.0)
    s2 = _sed(30.0)
    rel = float(np.abs(s1 - s2).max() / max(np.abs(s1).max(), 1e-99))
    assert rel > 1e-3, (
        f"SKIRTORTorus.predict() radius_ratio is a no-op (rel diff {rel:.2e}); "
        "the pass-through to create_skirtor_components_from_grid is missing."
    )


def test_skirtor_torus_class_polar_beta_matches_canonical():
    """SKIRTORTorus.polar_beta must match the canonical agn_polar_beta declaration."""
    canonical = declared_prior(_AGN_PARAMS, "agn_polar_beta")
    class_prior = SKIRTORTorus.polar_beta

    assert class_prior.lo == canonical.lo, (
        f"SKIRTORTorus.polar_beta.lo={class_prior.lo} does not match "
        f"canonical agn_polar_beta.lo={canonical.lo}"
    )
    assert class_prior.hi == canonical.hi, (
        f"SKIRTORTorus.polar_beta.hi={class_prior.hi} does not match "
        f"canonical agn_polar_beta.hi={canonical.hi}"
    )
    assert class_prior.default == canonical.default, (
        f"SKIRTORTorus.polar_beta.default={class_prior.default} does not match "
        f"canonical agn_polar_beta.default={canonical.default}"
    )


def test_skirtor_torus_class_polar_t_matches_canonical():
    """SKIRTORTorus.polar_T must match the canonical agn_polar_T declaration.

    RULING R12(a): the class's polar-temperature attribute is named ``polar_T``
    (registers ``agn_polar_T``), matching the composable ``skirtor_torus_block``'s
    own parameter name for the identical physical quantity -- NOT
    ``polar_temperature``/``agn_polar_temperature``, a distinct canonical
    declaration in ``_params.py`` still consumed by ``blocks/atten.py``
    (a different, running task; untouched here).
    """
    canonical = declared_prior(_AGN_PARAMS, "agn_polar_T")
    assert hasattr(SKIRTORTorus, "polar_T"), (
        "SKIRTORTorus must declare polar_T (not polar_temperature) so its "
        "registered name agn_polar_T matches the composable skirtor_torus_block."
    )
    class_prior = SKIRTORTorus.polar_T
    assert class_prior.lo == canonical.lo
    assert class_prior.hi == canonical.hi
    assert class_prior.default == canonical.default


#: Deliberate deviations from the canonical declaration, by registered name,
#: with the one-line reason the ruling requires. Empty: every one of
#: SKIRTORTorus's twelve declared parameters derives from its canonical
#: declaration via ``declared_prior`` (Task 14 fix round 2) -- there is
#: currently no deliberate deviation on this class. Kept as a named,
#: documented allow-list (rather than skipping the check outright) so a
#: future deliberate deviation has one place to go, with a reason, instead
#: of silently drifting back into a hardcoded literal.
_DELIBERATE_DEVIATIONS: dict[str, str] = {}


def test_skirtor_torus_all_declared_params_match_canonical():
    """Every SKIRTORTorus declared parameter's bounds AND default must equal
    the canonical ``_params.py`` declaration of its registered name.

    Task 14 fix round 2: a re-review found the restated-literal-with-drifted-
    default class surviving in TWO more places after fix round 1 had already
    derived ten of the twelve declared parameters from their canonical
    declarations: ``polar_ebv`` (default 0.1 vs canonical
    ``agn_polar_ebv`` 0.03, ``_params.py:454-462``) and ``log_lbol`` (default
    11.0 vs canonical ``agn_log_lbol`` 10.0, ``_params.py:65-69``).
    ``tools/check_param_defaults.py`` cannot see either drift -- it only
    checks that a signature default is INSIDE the declared prior, not that a
    class-level restatement EQUALS it exactly. Iterates
    ``declared_parameters()`` live (never a hand-picked subset of names) so
    this drift class is impossible for this class going forward, not merely
    absent from the names a reviewer happened to check this round.
    """
    comp = SKIRTORTorus()
    declared = comp.declared_parameters()
    assert len(declared) == 12, (
        f"expected 12 declared parameters on SKIRTORTorus, got {len(declared)}: "
        f"{sorted(d.name for d in declared)} (update this test's canonical-name "
        "loop and the allow-list docstring if a parameter was deliberately added "
        "or removed)"
    )

    mismatches = []
    for decl in declared:
        name = decl.name
        if name in _DELIBERATE_DEVIATIONS:
            continue
        canonical = declared_prior(_AGN_PARAMS, name)
        class_prior = decl.prior
        if class_prior.lo != canonical.lo:
            mismatches.append(f"{name}.lo: class={class_prior.lo} canonical={canonical.lo}")
        if class_prior.hi != canonical.hi:
            mismatches.append(f"{name}.hi: class={class_prior.hi} canonical={canonical.hi}")
        if class_prior.default != canonical.default:
            mismatches.append(
                f"{name}.default: class={class_prior.default} canonical={canonical.default}"
            )
    assert not mismatches, "SKIRTORTorus restates a canonical declaration:\n" + "\n".join(
        mismatches
    )


def test_skirtor_torus_radius_ratio_declared():
    """SKIRTORTorus must declare radius_ratio as a free parameter."""
    # Check that the class has radius_ratio attribute
    assert hasattr(SKIRTORTorus, "radius_ratio"), (
        "SKIRTORTorus must declare radius_ratio as a class attribute"
    )

    # Get the declared prior
    class_radius_ratio = SKIRTORTorus.radius_ratio
    canonical = declared_prior(_AGN_PARAMS, "agn_radius_ratio")

    # Verify bounds match
    assert class_radius_ratio.lo == canonical.lo, (
        f"SKIRTORTorus.radius_ratio.lo={class_radius_ratio.lo} does not match "
        f"canonical agn_radius_ratio.lo={canonical.lo}"
    )
    assert class_radius_ratio.hi == canonical.hi, (
        f"SKIRTORTorus.radius_ratio.hi={class_radius_ratio.hi} does not match "
        f"canonical agn_radius_ratio.hi={canonical.hi}"
    )
    assert class_radius_ratio.default == canonical.default, (
        f"SKIRTORTorus.radius_ratio.default={class_radius_ratio.default} does not match "
        f"canonical agn_radius_ratio.default={canonical.default}"
    )


#: Physics parameters legitimately excluded from the class/composable
#: shared-name equality below, with the one-line reason RULING R12 requires.
#:
#: - ``agn_delta`` (R12c): the disc power-law slope used by the CLASS's OWN
#:   bundled disc (skirtor_model.py's disc-shape selection in predict()); the
#:   composable design puts the disc in the separate disc block instead, so
#:   ``skirtor_torus_block`` (torus-only) has no equivalent parameter.
#: - ``agn_band_frac`` / ``agn_torus_frac`` (R12b, RESOLVED by Task 16's R17):
#:   both registered names used to drive the identical physics (the AGN
#:   covering-factor scaling ``l_scale = L_bol x frac`` -- confirmed by
#:   ``skirtor.py``'s own ``create_skirtor_components_from_grid``, which
#:   accepts ``agn_torus_frac`` as a deprecated fallback for its canonical
#:   ``frac_agn`` kwarg, the exact quantity the class's ``band_frac``
#:   attribute was derived from) under TWO different declared names -- the
#:   class used ``agn_band_frac`` (its only consumer in the whole codebase),
#:   while ``skirtor_torus_block`` and six OTHER composable torus blocks
#:   (cat3d_wind/fritz/nenkova/nenkova_agnfitter/silva04/skirtor_agnfitter)
#:   named their own covering-factor kwarg ``agn_torus_frac``. R17 (Task 16)
#:   renamed the class's attribute to ``torus_frac``/``agn_torus_frac`` and
#:   retired ``agn_band_frac``, so this is no longer a class/block
#:   discrepancy -- both sides now share the one canonical name, and this
#:   pair no longer needs an exclusion.
#: standalone SKIRTORTorus bundles polar dust (monolithic path); the
#: composable path applies polar dust only via atten='polar_dust' (R22, Task 13)
_CLASS_ONLY_NAMES = frozenset({"agn_delta", "agn_polar_ebv", "agn_polar_T", "agn_polar_beta"})
_BLOCK_ONLY_NAMES: frozenset[str] = frozenset()


def test_skirtor_composable_and_class_param_names_reconciled():
    """Composable block and class must register the SAME shared physics names.

    RULING R12(d): computed from LIVE introspection on both sides (the
    class's ``declared_parameters()`` and the composable block's function
    signature), never a hand-written map -- a hand-written map cannot notice
    a future rename drifting the two apart again.
    """
    class_names = {d.name for d in SKIRTORTorus().declared_parameters()}
    block_names = {
        name for name in signature(skirtor_torus_block).parameters if name.startswith("agn_")
    }

    shared_class = class_names - _CLASS_ONLY_NAMES
    shared_block = block_names - _BLOCK_ONLY_NAMES

    assert shared_class == shared_block, (
        f"SKIRTORTorus and skirtor_torus_block disagree on shared parameter "
        f"names: class-only (unexpected) {sorted(shared_class - shared_block)}; "
        f"block-only (unexpected) {sorted(shared_block - shared_class)}"
    )

    # agn_radius_ratio is registered identically on both paths.
    assert "agn_radius_ratio" in class_names, (
        "agn_radius_ratio missing from SKIRTORTorus.declared_parameters()"
    )
    assert "agn_radius_ratio" in block_names, (
        "agn_radius_ratio missing from skirtor_torus_block's signature"
    )

    # agn_polar_T / agn_polar_beta: R22 (task13 fix-round-1) removed the
    # composable torus block's bundled Casey (2012) polar-dust graybody --
    # the standalone SKIRTORTorus class keeps it (monolithic path), but the
    # composable path's polar dust is owned exclusively by the standalone
    # ``polar_dust`` attenuation block (and its
    # ``polar_dust_reemission_lnu`` companion), selected via
    # ``agn={'atten': {'type': 'polar_dust'}}`` -- not by the torus block.
    # So these two names are now class-only, by design, not an accidental
    # drift this test should paper over.
    for name in ("agn_polar_T", "agn_polar_beta"):
        assert name in class_names, f"{name} missing from SKIRTORTorus.declared_parameters()"
        assert name not in block_names, (
            f"{name} is present on skirtor_torus_block's signature, but R22 "
            f"retired the composable torus block's bundled polar dust -- this "
            f"name should live only in the polar_dust attenuation block "
            f"(and its reemission companion) now."
        )


def test_skirtor_radius_ratio_polar_t_polar_beta_live_on_class_path(class_component):
    """agn_radius_ratio, agn_polar_T, agn_polar_beta must have nonzero gradient
    through SKIRTORTorus.predict() directly (the class path)."""
    wave = jnp.geomspace(1e3, 1e7, 400)
    sed_in = jnp.zeros_like(wave)

    def _integral(params) -> jnp.ndarray:
        sed_out, _ = class_component.predict(params, sed_in, wave)
        return jnp.sum(jnp.abs(sed_out))

    for name in ("radius_ratio", "polar_T", "polar_beta"):

        def _obj(v, name=name):
            p = {**_CLASS_PARAMS, name: v}
            return _integral(p)

        g = float(jax.grad(_obj)(_CLASS_PARAMS[name]))
        assert np.isfinite(g), (
            f"SKIRTORTorus.predict(): d(sum|sed|)/d({name}) is {g}. `nan != 0.0` "
            "is True, so the liveness assertion below cannot see a non-finite "
            "gradient (#2178)."
        )
        assert g != 0.0, f"SKIRTORTorus.predict(): {name} has zero gradient (dead parameter)"


def _skirtor_composable_model(ssp, *, atten: dict):
    """One composable-AGN build with the SKIRTOR torus, radius_ratio FREE on
    the torus sub-block, and the given ``atten`` sub-block dict."""
    agn = dict(
        _DISC,
        # radius_ratio partitions as an 'agn.torus' parameter -- nest it there.
        torus={
            "type": "skirtor",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "radius_ratio": tengri.FREE,
        },
        atten=atten,
        norm="independent",
    )
    return tengri.SEDModel.build(
        ssp,
        sfh=_SFH,
        dust_attenuation=_DUST,
        agn=agn,
        redshift=tengri.Fixed(0.05),
    )


def test_skirtor_radius_ratio_live_on_composable_path(ssp):
    """agn_radius_ratio must have nonzero gradient through the composable
    skirtor_torus_block, with 'norm': 'independent' explicit (no cross-block
    energy coupling to confound the measurement).

    Built ONCE with the param FREE so it is an ordinary entry of the sampled
    ``params`` dict; differentiated by perturbing that dict entry directly
    (``model.predict_photometry``, the JIT/vmap-safe surface), never by
    rebuilding the model at a traced value (``Fixed(...)`` coerces its
    argument via ``float()`` at construction and cannot accept a tracer).
    """
    model = _skirtor_composable_model(
        ssp, atten={"type": "none", "all_params": tengri.Fixed(tengri.DEFAULT)}
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))

    def _obj(pd):
        return jnp.sum(jnp.abs(model.predict(pd).rest_sed()))

    v0 = jnp.asarray(p["agn_radius_ratio"])
    g = float(jax.grad(lambda v: _obj({**p, "agn_radius_ratio": v}))(v0))
    assert np.isfinite(g), (
        f"composable skirtor_torus_block: d(sum|sed|)/d(agn_radius_ratio) is {g}. "
        "`nan != 0.0` is True, so the liveness assertion below cannot see a "
        "non-finite gradient (#2178)."
    )
    assert g != 0.0, "composable skirtor_torus_block: agn_radius_ratio has zero gradient"


def test_skirtor_polar_t_polar_beta_live_only_with_atten_polar_dust(ssp):
    """agn_polar_T/agn_polar_beta are live on the composable path ONLY when
    ``atten='polar_dust'`` is explicitly selected, and inert under
    ``atten='none'`` -- R22 (task13 fix-round-1) retired
    ``skirtor_torus_block``'s bundled Casey (2012) polar-dust reemission;
    the composable path's ONE polar-dust mechanism now lives exclusively in
    the standalone ``polar_dust`` attenuation block (Stage 5) and its
    ``polar_dust_reemission_lnu`` companion (Stage 6), which run only when
    ``agn_attenuation_block == 'polar_dust'``.
    """
    live_model = _skirtor_composable_model(
        ssp,
        atten={
            "type": "polar_dust",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "polar_T": tengri.FREE,
            "polar_beta": tengri.FREE,
        },
    )
    p_live = dict(live_model.spec.sample(jax.random.PRNGKey(0)))

    def _obj_live(pd):
        return jnp.sum(jnp.abs(live_model.predict(pd).rest_sed()))

    for name in ("agn_polar_T", "agn_polar_beta"):
        assert name in p_live, f"{name} not free under atten='polar_dust' (wiring gap)"
        v0 = jnp.asarray(p_live[name])
        g = float(jax.grad(lambda v, name=name: _obj_live({**p_live, name: v}))(v0))
        assert g != 0.0, f"atten='polar_dust': {name} has zero gradient"

    # Explicit short-key priors under atten='none': the names are declared
    # (the short-key path is independent of block-selection scoping), but
    # nothing on the composable path reads them when polar_dust is not the
    # selected attenuation block -- inert, not merely "not offered".
    inert_model = _skirtor_composable_model(
        ssp,
        atten={
            "type": "none",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "polar_T": tengri.FREE,
            "polar_beta": tengri.FREE,
        },
    )
    p_inert = dict(inert_model.spec.sample(jax.random.PRNGKey(0)))

    def _obj_inert(pd):
        return jnp.sum(jnp.abs(inert_model.predict(pd).rest_sed()))

    for name in ("agn_polar_T", "agn_polar_beta"):
        assert name in p_inert, f"{name} not free under atten='none' (short-key wiring gap)"
        v0 = jnp.asarray(p_inert[name])
        g = float(jax.grad(lambda v, name=name: _obj_inert({**p_inert, name: v}))(v0))
        assert g == 0.0, f"atten='none': {name} has NONZERO gradient (0.0 expected -- inert)"


class TestSkirtorRadiusRatioGridNodes:
    """The grid is tracked (data/skirtor_templates_v3.h5, 27.5 MB); its absence
    is a FAILURE, never a skip. Resolves the grid path via the package's own
    loader (_find_skirtor_grid), never a relative path or ``__wrapped__``."""

    @pytest.fixture(scope="class")
    def make_components(self, skirtor_grid_path):
        return create_skirtor_components_from_grid(skirtor_grid_path)

    @pytest.mark.parametrize("radius_ratio", [10.0, 20.0, 30.0])
    def test_node_is_finite(self, make_components, radius_ratio):
        """Every declared radius_ratio grid node ([10, 20, 30]) must produce
        a finite disk/dust component."""
        wave = jnp.geomspace(1e3, 1e7, 200)
        components = make_components(
            wave,
            agn_log_lbol=12.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_radius_ratio=radius_ratio,
            agn_cos_inc=0.866,
            frac_agn=0.5,
        )
        assert jnp.all(jnp.isfinite(components.disk)), (
            f"disk component contains non-finite values at radius_ratio={radius_ratio}"
        )
        assert jnp.all(jnp.isfinite(components.dust)), (
            f"dust component contains non-finite values at radius_ratio={radius_ratio}"
        )

    def test_nodes_are_mutually_distinct(self, make_components):
        """The three radius_ratio grid nodes must give mutually DIFFERENT
        dust SEDs -- node-exact interpolation reproducing the same slice at
        every node would also pass a bare finiteness check, so this is the
        load-bearing half of the guard."""
        wave = jnp.geomspace(1e3, 1e7, 200)

        def _dust(radius_ratio: float) -> np.ndarray:
            components = make_components(
                wave,
                agn_log_lbol=12.0,
                agn_tau_skirtor=7.0,
                agn_p_skirtor=1.0,
                agn_q_skirtor=1.0,
                agn_oa_skirtor=40.0,
                agn_radius_ratio=radius_ratio,
                agn_cos_inc=0.866,
                frac_agn=0.5,
            )
            return np.asarray(components.dust)

        seds = {r: _dust(r) for r in (10.0, 20.0, 30.0)}
        pairs = [(10.0, 20.0), (20.0, 30.0), (10.0, 30.0)]
        for r1, r2 in pairs:
            rel = float(np.abs(seds[r1] - seds[r2]).max() / max(np.abs(seds[r1]).max(), 1e-99))
            assert rel > 1e-6, (
                f"radius_ratio={r1} and radius_ratio={r2} give indistinguishable "
                f"dust SEDs (rel diff {rel:.2e})"
            )

# SPDX-License-Identifier: BSD-3-Clause
"""Contract: AGN template libraries thread as JIT arguments, never as constants.

Every AGN block backed by a template library must receive that library as a
traced **argument** of ``predict_observables_impl``. A block that instead calls
its module-level ``@functools.cache`` loader at trace time freezes the whole
grid into the graph as ``Constant`` ops — 31 MB for SKIRTOR, 17 MB for Fritz.

Two properties make this guard honest, and both have failed before:

* **Measure the arg-surface.** ``predict_state`` calls
  ``_template_data_for_jit()`` *inside* its own trace, so anything published
  there becomes a closure constant by construction. That surface cannot tell
  working threading from dead threading.
* **Walk constants recursively.** See :mod:`tests.contract._jaxpr_consts`.

The parametrization is over the **registry**, not a hand-written list, so a
newly registered template-backed block is covered the day it lands rather than
the day someone remembers to extend this file.
"""

from __future__ import annotations

import pathlib
import warnings

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri import FIXED, SEDModel
from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed

from ._equivalence_cases import SED_CASES
from ._jaxpr_consts import baked_mb

pytestmark = pytest.mark.contract

# Any SSP works: the property under test is whether the AGN torus library
# reaches the block as an argument, which is independent of the SSP flavor.
# Pinning one filename here would make the whole file skip silently on any
# checkout that ships a different grid.
_SSP_DIR = pathlib.Path("data")

# Budget [MB] for constants baked into the traced graph. A bare-stellar model
# measures ~0.06 MB and a correctly-threaded Cue nebular model ~0.20 MB, so the
# floor is well under 1 MB. The smallest offender (Silva04, +2.09 MB) sits
# above it; SKIRTOR (+31.4 MB) is 30x over.
_BAKED_BUDGET_MB = 1.0

# ``none`` is the no-op selector present in every category.
_SKIP_BLOCKS = frozenset({"none"})

# Block category -> the sub-block key the build grammar uses for that stage.
_GROUP_KEY = {
    "disc": "disc",
    "nlr": "nlr",
    "blr": "blr",
    "feii": "feii",
    "torus": "torus",
    "attenuation": "atten",
}


@pytest.fixture(scope="module")
def ssp():
    candidates = sorted(_SSP_DIR.glob("ssp_*.h5"))
    if not candidates:
        pytest.skip(f"no SSP grid available under {_SSP_DIR.resolve()}")
    return load_ssp_data(str(candidates[0].resolve()))


@pytest.fixture(scope="module")
def obs():
    return Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )


def _build(ssp, obs, **groups):
    """Build a model on the recommended grammar, warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FIXED},
            dust={"type": "two_component", "all_params": FIXED},
            redshift=Fixed(0.1),
            **groups,
        )


def _traced_baked_mb(model):
    """Trace ``predict_observables_impl`` with the template data as an ARGUMENT.

    Returns the megabytes of constants baked into the resulting graph.
    """
    from tengri.inference._model_cache import _default_owner

    # Populates the structural kernel cache and the component chain that
    # ``_template_data_for_jit`` walks.
    model._get_or_build_predict_observables_jit()
    cache = _default_owner.get_structural_kernel(model.compile_signature())
    impl = cache["predict_observables_impl"]

    params = model.spec.sample(jax.random.PRNGKey(0))
    closed = jax.make_jaxpr(impl)(
        params,
        model.spec.get_fixed_values(),
        model.ssp_data,
        model._template_data_for_jit(),
    )
    return baked_mb(closed)


def test_baseline_bare_stellar_bakes_almost_nothing(ssp, obs):
    """A model with no template-backed component sets the floor.

    This is the control that keeps the budget honest: if the walker stopped
    seeing constants entirely, this number would not be a plausible floor and
    every other assertion in the file would pass vacuously.
    """
    baked = _traced_baked_mb(_build(ssp, obs))
    assert baked < _BAKED_BUDGET_MB, f"baseline already bakes {baked:.2f} MB"


# Blocks still loading their library at trace time. The torus stage was
# converted first because it was the measured one; these are the same defect
# in the disc/nlr stages and need the same treatment — a ``template_loader``
# on the registration plus a grid-taking evaluator in the family module.
#
# ``strict=True`` on purpose: when one of these is fixed, this test starts
# FAILING as XPASS, which is the signal to delete its row. A non-strict xfail
# would silently absorb the fix and let the row rot.
_KNOWN_BAKING: dict[tuple[str, str], str] = {
    ("disc", "relagn"): "27.5 MB — _load_relagn_disc_grid closure (#1383)",
    ("disc", "kubota_done"): "23.0 MB — closure-captured disc grid (#1383)",
    ("disc", "schartmann2005_skirtor_atten"): "10.0 MB — SKIRTOR grid closure (#1383)",
    ("disc", "slone_netzer"): "1.8 MB — closure-captured disc grid (#1383)",
    ("nlr", "cue"): "8.5 MB — Cue NLR weights closure (#1383)",
}

# Not a threading defect: this block raises TracerBoolConversionError under
# jit regardless of where its templates live. Tracked separately so a genuine
# JIT-safety bug is not filed away as a performance issue.
_KNOWN_NOT_JITTABLE: dict[tuple[str, str], str] = {
    ("nlr", "feltre"): "TracerBoolConversionError under jit — JIT-safety bug, not baking",
}


def _all_block_cases():
    """Every (category, name) the recipe grammar can select, from the registry."""
    cases = []
    for category in sorted(AGN_BLOCKS):
        for name in sorted(AGN_BLOCKS[category]):
            if name in _SKIP_BLOCKS:
                continue
            key = (category, name)
            marks = []
            if key in _KNOWN_BAKING:
                marks.append(pytest.mark.xfail(reason=_KNOWN_BAKING[key], strict=True))
            elif key in _KNOWN_NOT_JITTABLE:
                marks.append(pytest.mark.xfail(reason=_KNOWN_NOT_JITTABLE[key], strict=True))
            cases.append(pytest.param(category, name, marks=marks, id=f"{category}-{name}"))
    return cases


@pytest.mark.parametrize("category,block", _all_block_cases())
def test_block_template_threads_as_argument(ssp, obs, category, block):
    """No AGN block, in any stage, may bake its template library into the graph.

    Parametrized over the whole registry rather than the families that
    happened to be measured, so a block that starts loading a library later
    — or a newly registered one — is caught the day it lands.
    """
    group = {"type": "composable", "all_params": FIXED, "disc": {"type": "multicolor"}}
    group[_GROUP_KEY[category]] = {"type": block}

    try:
        model = _build(ssp, obs, agn=group)
        baked = _traced_baked_mb(model)
    except (FileNotFoundError, NotImplementedError) as exc:
        pytest.skip(f"{category}/{block} unavailable: {exc}")

    assert baked < _BAKED_BUDGET_MB, (
        f"{category} block {block!r} bakes {baked:.2f} MB of templates into "
        f"the traced graph (budget {_BAKED_BUDGET_MB} MB). The library must "
        f"reach the block as a traced argument — declare a template_loader on "
        f"register_agn_block and read the 'templates' kwarg — not via a "
        f"module-level cached loader called at trace time."
    )


# ── Equivalence: threading must not move a single number ─────────────────────


@pytest.mark.parametrize(
    "name,module,sed_fn,loader_fn,kwargs", SED_CASES, ids=[c[0] for c in SED_CASES]
)
def test_threaded_grid_matches_closure_path(name, module, sed_fn, loader_fn, kwargs):
    """Passing the grid in must give the identical SED to loading it inside.

    The fix moved each family from a closed-over grid to one passed as an
    argument. That is meant to be pure plumbing, so the two paths must agree
    **exactly** — a tolerance here would hide a real change in the science.
    """
    import importlib

    mod = importlib.import_module(module)
    loader = getattr(mod, loader_fn, None)
    if loader is None:
        pytest.fail(
            f"{module}.{loader_fn} is missing — the block cannot declare a template_loader"
        )
    try:
        grid = loader()
    except FileNotFoundError:
        pytest.skip(f"{name} grid not available on disk")

    wave = jnp.linspace(1.0e4, 5.0e5, 512)
    fn = getattr(mod, sed_fn)
    closure_result = fn(wave, **kwargs)
    threaded_result = fn(wave, _template=grid, **kwargs)

    # Guard against a vacuous comparison: a family that returned all zeros
    # would pass any equality check.
    assert float(jnp.max(jnp.abs(closure_result))) > 0.0, f"{name} produced an all-zero SED"
    chex.assert_trees_all_close(threaded_result, closure_result, rtol=0.0, atol=0.0)

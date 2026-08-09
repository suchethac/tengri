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

import jax
import pytest

from tengri import FIXED, SEDModel
from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed

from ._jaxpr_consts import baked_mb

pytestmark = pytest.mark.contract

# Any SSP works: the property under test is whether the AGN torus library
# reaches the block as an argument, which is independent of the SSP flavour.
# Pinning one filename here would make the whole file skip silently on any
# checkout that ships a different grid.
_SSP_DIR = pathlib.Path("data")

# Budget [MB] for constants baked into the traced graph. A bare-stellar model
# measures ~0.06 MB and a correctly-threaded Cue nebular model ~0.20 MB, so the
# floor is well under 1 MB. The smallest offender (Silva04, +2.09 MB) sits
# above it; SKIRTOR (+31.4 MB) is 30x over.
_BAKED_BUDGET_MB = 1.0

# Blocks whose selection is a no-op or which carry no template library.
_SKIP_TORUS = frozenset({"none"})


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


@pytest.mark.parametrize("torus", sorted(set(AGN_BLOCKS["torus"]) - _SKIP_TORUS))
def test_torus_template_threads_as_argument(ssp, obs, torus):
    """No torus block may bake its template library into the graph."""
    model = _build(
        ssp,
        obs,
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": "multicolor"},
            "torus": {"type": torus},
        },
    )
    baked = _traced_baked_mb(model)
    assert baked < _BAKED_BUDGET_MB, (
        f"torus block {torus!r} bakes {baked:.2f} MB of templates into the "
        f"traced graph (budget {_BAKED_BUDGET_MB} MB). The library must reach "
        f"the block as a traced argument via template_state, not via a "
        f"module-level cached loader called at trace time."
    )

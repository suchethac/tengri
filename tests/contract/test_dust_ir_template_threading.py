# SPDX-License-Identifier: BSD-3-Clause
"""Contract: dust IR template libraries thread as JIT arguments, not constants.

The dust-IR sibling of ``test_agn_template_threading``. Same measurement: trace
``predict_observables_impl`` with ``template_data`` as a real **argument** and
walk constants recursively through sub-jaxprs.

Dust IR is the larger surface of the two. Before #1649, ``draine_li2014`` baked
**66.6 MB** into every compile and ``themis`` 39.4 MB — against the largest AGN
offender at 29.95 MB, and a bare-stellar floor of 0.05 MB.

Parametrized over ``list_dust_emission_models()`` rather than a hand-written
list, so a backend added later is covered the day it lands.
"""

from __future__ import annotations

import pathlib
import warnings

import jax
import pytest

import tengri
from tengri import FIXED, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed

from ._jaxpr_consts import baked_mb

pytestmark = pytest.mark.contract

_SSP_DIR = pathlib.Path("data")

#: Same budget as the AGN guard. Floor is ~0.05 MB; the smallest real offender
#: measured here (astrodust, 0.41 MB) still sits under it, so this budget only
#: catches libraries that are genuinely large — see ``_KNOWN_BAKING``.
_BAKED_BUDGET_MB = 1.0

# Measured 2026-08-10 (#1649). Each entry is a backend whose library is still
# read from a module-level cache at trace time.
#
# ``strict=True``: fixing one turns this XPASS, which fails the run and is the
# signal to delete its row. A non-strict xfail would absorb the fix silently.
_KNOWN_BAKING: dict[str, str] = {
    "themis": "39.41 MB — THEMIS grid closure (#1649)",
    "bosa": "4.45 MB — BOSA grid closure (#1649)",
    "draine_li2007": "3.76 MB — DL07 grid closure (#1649)",
    "dl07": "3.76 MB — alias of draine_li2007 (#1649)",
    "dl07_tabulated": "3.76 MB — alias of draine_li2007 (#1649)",
    "schreiber2018": "2.41 MB — Schreiber+2018 grid closure (#1649)",
}


@pytest.fixture(scope="module")
def ssp():
    candidates = sorted(_SSP_DIR.glob("ssp_*.h5"))
    if not candidates:
        pytest.skip(f"no SSP grid available under {_SSP_DIR.resolve()}")
    return load_ssp_data(str(candidates[0].resolve()))


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i"]))


def _traced_baked_mb(model):
    """Megabytes of constants baked into the traced forward graph."""
    from tengri.inference._model_cache import _default_owner

    model._get_or_build_predict_observables_jit()
    impl = _default_owner.get_structural_kernel(model.compile_signature())[
        "predict_observables_impl"
    ]
    params = model.spec.sample(jax.random.PRNGKey(0))
    return baked_mb(
        jax.make_jaxpr(impl)(
            params,
            model.spec.get_fixed_values(),
            model.ssp_data,
            model._template_data_for_jit(),
        )
    )


def _emission_names():
    """Every registered dust emission backend, from the live menu."""
    cases = []
    for row in tengri.list_dust_emission_models():
        name = row if isinstance(row, str) else row["name"]
        marks = []
        if name in _KNOWN_BAKING:
            marks.append(pytest.mark.xfail(reason=_KNOWN_BAKING[name], strict=True))
        cases.append(pytest.param(name, marks=marks, id=name))
    return cases


def test_baseline_without_dust_emission_bakes_almost_nothing(ssp, obs):
    """Control: the floor this budget is measured against.

    Without it a passing row cannot be distinguished from a walker that stopped
    seeing constants at all.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FIXED},
            dust={"type": "two_component", "all_params": FIXED},
            redshift=Fixed(0.1),
        )
    baked = _traced_baked_mb(model)
    assert baked < _BAKED_BUDGET_MB, f"baseline already bakes {baked:.2f} MB"


@pytest.mark.parametrize("emission", _emission_names())
def test_dust_emission_template_threads_as_argument(ssp, obs, emission):
    """No dust IR backend may bake its template library into the graph."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel.build(
                ssp_data=ssp,
                observation=obs,
                sfh={"type": "dpl", "all_params": FIXED},
                dust={
                    "type": "two_component",
                    "all_params": FIXED,
                    "emission": {"type": emission},
                },
                redshift=Fixed(0.1),
            )
        baked = _traced_baked_mb(model)
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        pytest.skip(f"dust emission {emission!r} unavailable: {exc}")

    assert baked < _BAKED_BUDGET_MB, (
        f"dust emission {emission!r} bakes {baked:.2f} MB into the traced graph "
        f"(budget {_BAKED_BUDGET_MB} MB). The library must reach the component "
        f"as a traced argument via template_data, not from a module-level cache "
        f"called at trace time."
    )

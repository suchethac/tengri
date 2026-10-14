# SPDX-License-Identifier: BSD-3-Clause
"""``SEDModel.build`` compiles a bounded number of XLA programs (#2914).

Every ``jnp`` operation evaluated eagerly at build time is dispatched as its own
XLA program: a few milliseconds to compile, under the persistent cache's 0.05 s
floor, so no cache ever serves it, and a build that evaluates a forward model or
a vmapped composite eagerly pays hundreds to thousands of them on every process.
Measured on the ``panchromatic/lut_freez`` seams fixture (#2914): 5,939 compiles
per fixture setup, 5,855 under the floor, 75 % of its wall clock.

The guard here counts calls to ``jax._src.compiler.compile_or_get_cached``, the
funnel every ``jit`` and every eager-op dispatch compile passes through, whether
the persistent cache then serves the executable or not. A count of ``log_compiles``
lines could not do this (``tests/contract/test_sequential_fit_compile_reuse.py``
says why: a cache hit logs no compile). The caches are cleared first, so the count
is the cold one a fresh process pays.

The budget is a measurement, not a target: the model below is the seams fixture's
kitchen sink without its AGN (the AGN adds the #2069 probe, one eager forward by
design, memoized per content). Lower it when a build-time eager evaluation is
removed; a rise is the regression this file exists to catch.
"""

from __future__ import annotations

import collections
from unittest import mock

import jax
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tests._data_skip import CUE_WEIGHTS, DATA_DIR, requires_cue_weights

pytestmark = pytest.mark.contract

_SSP_PATH = DATA_DIR / "fsps_prsc_miles_chabrier.h5"

#: Cold XLA compiles of the build below. Measured 2026-10-15 (jax 0.11.1, CPU): 456
#: before the build-time eager evaluations of #2914 were traced, 206 after the first
#: six of them; the budget sits just above that count. What remains is listed in the
#: #2914 thread (the onset-ceiling warning's eager cosmology, the dale2014 loader,
#: small eager tails in the z-table, term-table and energy-balance builds).
BUILD_COMPILE_BUDGET = 220


@pytest.fixture(scope="module")
def _cue_fixture_available():
    if not _SSP_PATH.is_file() or not CUE_WEIGHTS.is_file():
        pytest.skip(f"needs {_SSP_PATH} and {CUE_WEIGHTS}")


def _build_panchromatic_no_agn():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return SEDModel.build(
        ssp_data=load_ssp_data(str(_SSP_PATH)),
        observation=Observation(
            photometry=Photometry.from_names(["sdss_g", "sdss_r", "wise_w1", "herschel_250"])
        ),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 2.0),
            "tau_diff": Uniform(0.0, 2.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
        xray={"type": "simple"},
        shock={"frac": 0.1},
        redshift=Uniform(0.05, 1.0),
        approx=WavePrecomp(n_z=48, z_min=0.05, z_max=1.0),
    )


def count_build_compiles(build):
    """Run ``build()`` on cold caches and return ``(n_compiles, Counter of module names)``."""
    from jax._src import compiler
    from jax._src.lib.mlir import ir

    names: collections.Counter = collections.Counter()
    real = compiler.compile_or_get_cached

    def counting(backend, computation, *args, **kwargs):
        names[ir.StringAttr(computation.operation.attributes["sym_name"]).value] += 1
        return real(backend, computation, *args, **kwargs)

    jax.clear_caches()
    with mock.patch.object(compiler, "compile_or_get_cached", counting):
        build()
    return sum(names.values()), names


@requires_cue_weights
def test_a_free_redshift_panchromatic_build_compiles_within_budget(_cue_fixture_available):
    n, names = count_build_compiles(_build_panchromatic_no_agn)
    assert n <= BUILD_COMPILE_BUDGET, (
        f"{n} XLA compiles during SEDModel.build against a budget of "
        f"{BUILD_COMPILE_BUDGET}; the most frequent programs (eager ops compile one "
        f"program each): {names.most_common(12)}"
    )

# SPDX-License-Identifier: BSD-3-Clause
"""A sampler run through the shared driver compiles its kernel once.

The chain families run adaptation and sampling in one ``run_chain`` program, ChEES in one
``_chees_scan``, MCLMC's scan once; no sampler compiles a second program that holds the
kernel (a stability probe), and the driver itself adds no eagerly dispatched Hessian or
log-density evaluation outside those programs. Counted from the compile log of a cold run:
a fresh log-density function per run, so the compile cache cannot hide a compile.

Measured on the two-chain linear-Gaussian fixture (warmup 30, 20 draws): each chain family
compiles ``run_chain`` once and nothing else of size (the other entries are single-primitive
programs: PRNG seeding and reductions); ChEES compiles ``_chees_scan`` once; MCLMC compiles its
sampling scan once, plus about 140 single-primitive programs from BlackJAX's eagerly run
``(L, step size)`` tuner, which the driver does not own.
"""

from __future__ import annotations

import logging
import re

import jax
import jax.numpy as jnp
import pytest

from tengri.inference.sampling.driver import run_sampler
from tengri.inference.sampling.target import FlatTarget, LatentLayout

pytestmark = pytest.mark.contract

_A = jnp.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0], [1.0, 1.0, 1.0]])

#: Sampler -> the one program that holds its kernel.
_KERNEL_PROGRAM = {
    "nuts": "run_chain",
    "hmc": "run_chain",
    "dynamic_hmc": "run_chain",
    "ghmc": "run_chain",
    "barker": "run_chain",
    "mala": "run_chain",
    "chees": "_chees_scan",
    "mclmc": "_mclmc_sample_scan",
}
_FORBIDDEN = ("hessian", "probe", "stability")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.names: list[str] = []

    def emit(self, record):
        match = re.match(r"Compiling (\S+)", record.getMessage())
        if match:
            self.names.append(match.group(1))


def _compiled_names(sampler):
    def logdensity(x, args):  # a new function each run: nothing is cached
        return -0.5 * jnp.sum(((args["y"] - _A @ x) / 0.5) ** 2) - 0.5 * jnp.sum(x**2)

    target = FlatTarget(
        logdensity, LatentLayout.vector(3), jnp.zeros(3), {"y": jnp.array([0.5, 0.2, 0.4, 0.3])}
    )
    capture = _Capture()
    loggers = [
        logging.getLogger(n)
        for n in ("jax._src.dispatch", "jax._src.interpreters.pxla", "jax._src.compiler")
    ]
    saved = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(capture)
        lg.setLevel(logging.DEBUG)
    try:
        with jax.log_compiles(True):
            run_sampler(
                sampler,
                target,
                key=jax.random.PRNGKey(0),
                n_chains=2,
                n_warmup=30,
                n_samples=20,
                health_warnings=False,
            )
    finally:
        for lg, level in zip(loggers, saved, strict=True):
            lg.removeHandler(capture)
            lg.setLevel(level)
    return capture.names


@pytest.mark.parametrize("sampler", sorted(_KERNEL_PROGRAM))
def test_a_cold_run_compiles_its_kernel_program_once(sampler):
    names = _compiled_names(sampler)
    program = _KERNEL_PROGRAM[sampler]
    kernel = [n for n in names if n == f"jit({program})"]
    assert len(kernel) == 1, (sampler, names)
    assert not [n for n in names if any(word in n.lower() for word in _FORBIDDEN)], names

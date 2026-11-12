# SPDX-License-Identifier: BSD-3-Clause
"""The batched spectroscopy drivers pass the template tables as jit arguments (#3212).

``batched_log_likelihood`` threads the SSP grid, the template data and the z-table
(``model.threaded_template_inputs()``) into the per-galaxy forward. Closing over
them embeds the arrays in the compiled program: on a DESI-scale SSP cube the
executable is 2 GB and the persistent compile cache cannot store it (#3212). These
tests pin the seam:

1. The catalog bucket program holds no constant as large as the SSP flux cube.
2. The MAP driver's program holds no constant as large as the SSP flux cube.
3. The jitted batched log-likelihood, lowered with the tables as arguments, holds no
   copy of a threaded table as a constant and takes the tables as parameters. The
   closure form of the same program is checked by the same detector as a positive
   control, so a regression that captures the tables as constants fails here.
4. The float64 jitted log-likelihood with the tables as an argument agrees with the
   jitted closure to within a few ulp.

The programs are recorded from the drivers themselves and traced with their
per-galaxy inputs as arguments, so a table that a driver closes over shows up as a
constant of the trace.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference import batched_spectro
from tengri.observation.batched import build_spectro_batches
from tests.contract.test_batched_spectro_loglik import _MASS, _TRUE_MASS, _mock_catalog, _record
from tests.contract.test_batched_spectrum_forward import _banded, _free_z_model

pytestmark = pytest.mark.contract

# Numeric agreement tolerance of the argument and closure programs, in ulp. Measured
# on macOS arm64 (jax 0.11.1, CPU): 0, 9, 6 ulp across the three galaxies. Measured on
# an emulated linux/amd64 container: 55, 1, 13 ulp. Measured on the native GitHub runner
# (run 41568044832): at most 297 ulp. The tolerance is the next power of two above 297.
_MAX_ULP = 512

# Constant regex for the f64 constants of a StableHLO module: ``dense<payload> :
# tensor<dims x f64>`` at the end of the line.
_F64_CONSTANT = re.compile(r"stablehlo\.constant dense<(.*)> : tensor<([0-9x]*)f64>\s*$")
_MAIN_SIGNATURE = re.compile(r"func\.func public @main\((.*?)\) ->")
_F64_TENSOR = re.compile(r"tensor<([0-9x]*)f64>")


def _lowered_text(fn, *args) -> str:
    """StableHLO text of ``jax.jit(fn)`` lowered at ``args``."""
    return jax.jit(fn).lower(*args).as_text()


def _dims(text: str) -> tuple[int, ...]:
    """Shape from a ``"3x20x"`` dimension string, ``""`` for a scalar."""
    return tuple(int(d) for d in text.rstrip("x").split("x")) if text.strip("x") else ()


def _decode(payload: str, shape: tuple[int, ...]) -> np.ndarray | None:
    """Values of a dense f64 payload, or None when the printed form is not decodable."""
    if payload.startswith('"0x'):
        return np.frombuffer(bytes.fromhex(payload[3:-1]), dtype=np.float64).reshape(shape)
    try:
        return np.asarray(json.loads(payload), dtype=np.float64).reshape(shape)
    except (ValueError, TypeError):
        return None


def _table_findings(text: str, tables: list[np.ndarray]) -> dict[str, list[tuple[int, ...]]]:
    """Table copies held as constants, and table shapes the parameters do not supply.

    A constant counts as a table copy when its shape equals a table's shape and its
    values equal that table, or when its shape matches and its payload cannot be
    decoded (a match on shape alone is then the conservative reading). ``missing``
    lists table shapes that the ``main`` signature has no f64 parameter for, counted
    with multiplicity.
    """
    constants = []
    for line in text.splitlines():
        match = _F64_CONSTANT.search(line)
        if match is None:
            continue
        shape = _dims(match.group(2))
        for table in tables:
            if table.shape != shape:
                continue
            values = _decode(match.group(1), shape)
            if values is None or np.array_equal(values, table):
                constants.append(shape)
    signature = _MAIN_SIGNATURE.search(text)
    params = [_dims(d) for d in _F64_TENSOR.findall(signature.group(1))] if signature else []
    needed = Counter(table.shape for table in tables)
    missing = sorted((needed - Counter(params)).elements())
    return {"constants": constants, "missing": missing}


def _threaded_tables(tables, ssp) -> list[np.ndarray]:
    """The template table leaves the batched program is meant to take as arguments.

    The SSP wavelength grid is left out. The argument program holds it as a constant
    too, because the grid is read from the model rather than from the threaded tuple:
    the grid is not a parameter of that program. The flux cube and the age and
    metallicity grids are the tables the seam passes, and those are what is checked.
    """
    wave = np.asarray(ssp.ssp_wave)
    leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tables)]
    return [leaf for leaf in leaves if leaf.shape != wave.shape or not np.array_equal(leaf, wave)]


def _all_consts(closed) -> list:
    """Every constant of ``closed`` and of the jaxprs nested in its equations."""
    found = list(closed.consts)
    for eqn in closed.jaxpr.eqns:
        for value in eqn.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(item, "jaxpr") and hasattr(item, "consts"):
                    found.extend(_all_consts(item))
                elif hasattr(item, "eqns"):
                    found.extend(_all_consts(jax.core.ClosedJaxpr(item, ())))
    return found


def _ssp_nbytes(ssp) -> int:
    return int(np.asarray(ssp.ssp_flux).nbytes)


def _big_constants(closed, threshold: int) -> list[int]:
    return [
        int(np.asarray(c).nbytes) for c in _all_consts(closed) if np.asarray(c).nbytes >= threshold
    ]


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)


@pytest.fixture(scope="module")
def template(synthetic_ssp, mock):
    return _free_z_model(synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]))


@pytest.fixture(scope="module")
def bucket(mock, template):
    [(spec, batch, index)] = build_spectro_batches(
        [_record(g) for g in mock], quantum=420, model=template
    )
    np.testing.assert_array_equal(index, [0, 1, 2])
    return spec, batch


def test_catalog_bucket_program_holds_no_table_as_a_constant(
    synthetic_ssp, template, bucket, monkeypatch
):
    """The program the bucket path hands to the sampler embeds no SSP-sized table.

    The catalog driver maps its per-galaxy sampler with ``map_with_shared``; that
    call is recorded and its program traced with the per-galaxy inputs and the
    shared pytree as arguments. A template table closed over by the log-density
    is an SSP-sized constant of that trace.
    """
    recorded = {}
    real_map = batched_spectro.map_with_shared

    def recording_map(fn, xs, shared, *, batch_size):
        recorded.update(fn=fn, xs=xs, shared=shared, batch_size=batch_size)
        return real_map(fn, xs, shared, batch_size=batch_size)

    monkeypatch.setattr(batched_spectro, "map_with_shared", recording_map)
    spec, batch = bucket
    batched_spectro.fit_spectra_catalog_mcmc(
        template,
        [(spec, batch, np.arange(3))],
        sampler="nuts",
        n_warmup=2,
        n_burnin=0,
        n_samples=2,
        init="prior",
        batch_size=None,
    )
    fn, xs, shared, batch_size = (
        recorded["fn"],
        recorded["xs"],
        recorded["shared"],
        recorded["batch_size"],
    )
    program = jax.make_jaxpr(lambda x, s: real_map(fn, x, s, batch_size=batch_size))(xs, shared)
    ssp_nbytes = _ssp_nbytes(synthetic_ssp)
    big = _big_constants(program, ssp_nbytes)
    assert big == [], f"constants at or above the SSP cube size ({ssp_nbytes} B): {big}"


def test_map_program_holds_no_table_as_a_constant(synthetic_ssp, template, bucket, monkeypatch):
    """The program of the MAP driver's chunk embeds no SSP-sized table.

    ``_run_chunk`` is recorded with its nlp (the first argument) and its array
    arguments, which are traced as inputs. A template table closed over by the
    negative log-posterior becomes an SSP-sized constant of the trace.
    """
    recorded = {}
    real_chunk = batched_spectro._run_chunk

    def recording_chunk(nlp, *arrays, **kwargs):
        recorded.update(nlp=nlp, arrays=arrays, kwargs=kwargs)
        return real_chunk(nlp, *arrays, **kwargs)

    monkeypatch.setattr(batched_spectro, "_run_chunk", recording_chunk)
    spec, batch = bucket
    batched_spectro.fit_spectra_map_vmap(template, [(spec, batch, np.arange(3))], n_steps=2)
    nlp, arrays, kwargs = recorded["nlp"], recorded["arrays"], recorded["kwargs"]
    program = jax.make_jaxpr(lambda *a: real_chunk(nlp, *a, **kwargs))(*arrays)
    ssp_nbytes = _ssp_nbytes(synthetic_ssp)
    big = _big_constants(program, ssp_nbytes)
    assert big == [], f"constants at or above the SSP cube size ({ssp_nbytes} B): {big}"


def test_batched_log_likelihood_lowered_program_takes_tables_as_arguments(
    synthetic_ssp, template, bucket
):
    """The lowered batched log-likelihood takes the tables as parameters, not constants.

    The argument form is lowered with the tables as its third argument. The closure
    form, which captures the same tables, is lowered through the same helper and read
    by the same detector: it must show table copies as constants, which proves the
    detector would catch a regression to closure capture.
    """
    spec, batch = bucket
    tables = template.threaded_template_inputs()
    threaded = _threaded_tables(tables, synthetic_ssp)
    params = {_MASS: jnp.asarray([m + 0.05 for m in _TRUE_MASS])}
    argument_fn = jax.vmap(
        batched_spectro.batched_log_likelihood(template, spec), in_axes=(0, 0, None)
    )
    closure_fn = jax.vmap(batched_spectro.batched_log_likelihood(template, spec, threaded=tables))
    argument = _table_findings(_lowered_text(argument_fn, params, batch, tables), threaded)
    closure = _table_findings(_lowered_text(closure_fn, params, batch), threaded)
    assert closure["constants"], "positive control: the closure program shows no table constant"
    assert argument["constants"] == [], f"table copies held as constants: {argument['constants']}"
    missing = argument["missing"]
    assert missing == [], f"table shapes not among the parameters: {missing}"


def test_batched_log_likelihood_jitted_argument_and_closure_agree_within_max_ulp(template, bucket):
    """The jitted float64 value with the tables as an argument matches the jitted closure.

    The tables are the ``threaded`` tuple of ``model.threaded_template_inputs()``. Both
    sides are compiled with ``jax.jit``, because the constant embedding this seam
    removes happens only in a compiled program. The closure form embeds the tables as
    compile-time constants, and XLA folds them in a different order from the argument
    form, so the two differ at the ulp level and the platform sets the size: 0, 9 and 6
    ulp across the three galaxies on macOS arm64, 55, 1 and 13 on linux/amd64. The
    tolerance is the next power of two above the largest of these. Test 3 pins that the
    lowered program holds no table as a constant; this test only checks that the two
    forms agree numerically.
    """
    spec, batch = bucket
    tables = template.threaded_template_inputs()
    masses = jnp.asarray([m + 0.05 for m in _TRUE_MASS])
    params = {_MASS: masses}
    jit_argument = jax.jit(
        jax.vmap(batched_spectro.batched_log_likelihood(template, spec), in_axes=(0, 0, None))
    )
    jit_closure = jax.jit(
        jax.vmap(batched_spectro.batched_log_likelihood(template, spec, threaded=tables))
    )
    argument_values = np.asarray(jit_argument(params, batch, tables))
    closure_values = np.asarray(jit_closure(params, batch))
    assert argument_values.dtype == np.float64
    np.testing.assert_array_max_ulp(argument_values, closure_values, maxulp=_MAX_ULP)

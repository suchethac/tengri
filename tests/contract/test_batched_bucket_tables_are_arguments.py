# SPDX-License-Identifier: BSD-3-Clause
"""The batched spectroscopy drivers pass the template tables as jit arguments (#3212).

``batched_log_likelihood`` threads the SSP grid, the template data and the z-table
(``model.threaded_template_inputs()``) into the per-galaxy forward. Closing over
them embeds the arrays in the compiled program: on a DESI-scale SSP cube the
executable is 2 GB and the persistent compile cache cannot store it (#3212). These
tests pin the seam:

1. The catalog bucket program holds no constant as large as the SSP flux cube.
2. The MAP driver's program holds no constant as large as the SSP flux cube.
3. The float64 log-likelihood at a fixed standardized point is bit-identical to the
   value recorded before the tables became arguments.

The programs are recorded from the drivers themselves and traced with their
per-galaxy inputs as arguments, so a table that a driver closes over shows up as a
constant of the trace.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference import batched_spectro
from tengri.observation.batched import build_spectro_batches
from tests.contract.test_batched_spectro_loglik import _MASS, _TRUE_MASS, _mock_catalog, _record
from tests.contract.test_batched_spectrum_forward import _banded, _free_z_model

pytestmark = pytest.mark.contract

#: Per-galaxy log-likelihood at the standardized point ``_TRUE_MASS + 0.05``, float64,
#: recorded from the closure-captured build before the tables became arguments.
_RECORDED_LOGLIK = np.array([-969143.1245560908, -1209237.2322956694, -1215310.5488341248])


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


def test_batched_log_likelihood_is_bit_identical_to_recorded_float64(template, bucket):
    spec, batch = bucket
    loglik = batched_spectro.batched_log_likelihood(template, spec)
    masses = jnp.asarray([m + 0.05 for m in _TRUE_MASS])
    values = np.asarray(jax.vmap(loglik)({_MASS: masses}, batch))
    assert values.dtype == np.float64
    np.testing.assert_array_equal(values, _RECORDED_LOGLIK)

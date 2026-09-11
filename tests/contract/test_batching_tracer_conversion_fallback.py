# SPDX-License-Identifier: BSD-3-Clause
"""`vmap_chunked` must fall back for the whole ``Tracer*ConversionError`` family.

https://github.com/suchethac/tengri/issues/2264

``utils/batching.py`` caught only ``jax.errors.ConcretizationTypeError`` under
the belief that it is the base of the ``Tracer*ConversionError`` family. On
jax 0.11.1 that belief is false: ``TracerArrayConversionError`` (raised by
``np.asarray(tracer)``) and ``TracerIntegerConversionError`` (raised by
``operator.index(tracer)``) are siblings of ``ConcretizationTypeError`` under
``JAXTypeError``, not subclasses, so a mapped function that inspects its
input that way raised straight through the jittability probe instead of
falling back to the eager per-draw loop.
"""

from __future__ import annotations

import operator
import re

import jax
import numpy as np
import pytest

from tengri.utils.batching import _NOT_TRACEABLE, vmap_chunked

pytestmark = pytest.mark.contract


def _asarray_fn(params):
    """Inspects a concrete value via ``np.asarray`` -- not traceable."""
    np.asarray(params["x"])
    return {"y": params["x"] * 2}


def _bool_fn(params):
    """Inspects a concrete value via ``bool`` -- not traceable."""
    bool(params["x"])
    return {"y": params["x"] * 2}


def _int_fn(params):
    """Inspects a concrete value via ``int`` -- not traceable."""
    int(params["x"])
    return {"y": params["x"] * 2}


def _index_fn(params):
    """Inspects a concrete value via ``operator.index`` -- not traceable."""
    operator.index(params["x"])
    return {"y": params["x"] * 2}


@pytest.mark.parametrize(
    ("fn", "x_values"),
    [
        pytest.param(_asarray_fn, np.linspace(1.0, 4.0, 4), id="np_asarray"),
        pytest.param(_bool_fn, np.linspace(1.0, 4.0, 4), id="bool"),
        pytest.param(_int_fn, np.linspace(1.0, 4.0, 4), id="int"),
        pytest.param(_index_fn, np.arange(1, 5, dtype=np.int64), id="operator_index"),
    ],
)
def test_batched_eval_falls_back_for_every_tracer_conversion_error(fn, x_values):
    """A non-traceable ``fn`` must fall back to the eager loop, never raise.

    Before the fix, only the ``np.asarray`` case (``id="np_asarray"``) was RED:
    it raises ``TracerArrayConversionError``, which the handler did not catch,
    so the jittability probe's exception propagated instead of triggering the
    fallback. ``bool(tracer)`` raises ``TracerBoolConversionError`` (already a
    ``ConcretizationTypeError`` subclass) and ``int(tracer)`` raises a bare
    ``ConcretizationTypeError``, so those two cases already passed at HEAD.
    ``operator.index(tracer)`` raises ``TracerIntegerConversionError``, a
    sibling of ``ConcretizationTypeError`` exactly like ``np.asarray``'s
    ``TracerArrayConversionError``, so this case was RED at HEAD for the same
    reason the array case was (measured: dropping
    ``jax.errors.TracerIntegerConversionError`` from ``_NOT_TRACEABLE`` turns
    this case red while the other three stay green). All four are kept here
    as a regression lock against the family being narrowed again.
    """
    batch = {"x": x_values}
    with pytest.warns(UserWarning, match="cannot be traced"):
        result = vmap_chunked(fn, chunk_size=2)(batch)
    np.testing.assert_allclose(np.asarray(result["y"]), batch["x"] * 2)


def test_not_traceable_tuple_covers_every_tracer_conversion_error_class():
    """``_NOT_TRACEABLE`` must cover every ``Tracer*ConversionError`` jax exposes.

    Discovers the family from the installed ``jax.errors`` module rather than
    hardcoding jax 0.11.1's three names (``TracerArrayConversionError``,
    ``TracerBoolConversionError``, ``TracerIntegerConversionError``), so a
    future jax release that adds another sibling under ``JAXTypeError``
    fails this test instead of silently reopening #2264.
    """
    family = [
        getattr(jax.errors, name)
        for name in dir(jax.errors)
        if re.match(r"Tracer.*ConversionError$", name)
    ]
    assert family, "no Tracer*ConversionError classes found in jax.errors"
    missing = [cls for cls in family if not issubclass(cls, _NOT_TRACEABLE)]
    assert not missing, (
        f"_NOT_TRACEABLE in utils/batching.py does not cover: {[cls.__name__ for cls in missing]}"
    )

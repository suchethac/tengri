# SPDX-License-Identifier: BSD-3-Clause
"""Contract for #3067: a log of a stored float32 table is computed in float64.

numpy evaluates ``log``/``log10`` on float32 with a SIMD kernel that is not
correctly rounded and differs between x86_64 and Apple ARM, so a table built
that way differs across platforms although the file is byte-identical. Each
loader below must equal the float64-computed, once-rounded reference bit for
bit; that reference is platform independent by construction.

Marker: contract
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from tengri.utils.table_log import log_table_f32

pytestmark = pytest.mark.contract

# The references are evaluated in extended precision (``np.longdouble``) so they do
# not share the float64 ``log`` of the code under test. Where ``longdouble`` is
# float64 (Apple ARM, Windows) this reduces to the float64 reference.


def test_log_table_helper_matches_float64_reference_bit_for_bit():
    rng = np.random.default_rng(3067)
    table = (10.0 ** rng.uniform(-40, 3, size=200_000)).astype(np.float32)
    table[:5] = 0.0
    for base, fn in (("e", np.log), ("10", np.log10)):
        got = log_table_f32(table, floor=1e-37, base=base)
        ref = fn(np.maximum(table.astype(np.longdouble), 1e-37)).astype(np.float32)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, ref)


def test_nthcomp_table_log_is_float64_rounded_once():
    from tengri.components.agn._nthcomp import (
        _DEFAULT_TEMPLATE_PATH,
        _load_nthcomp_templates_impl,
    )

    if not _DEFAULT_TEMPLATE_PATH.exists():
        pytest.skip("nthcomp_templates.h5 not available")
    table_log = _load_nthcomp_templates_impl()[4]
    with h5py.File(_DEFAULT_TEMPLATE_PATH, "r") as f:
        table = f["table"][:]
    ref = np.log(np.maximum(table.astype(np.longdouble), 1e-37)).astype(np.float32)
    assert table_log.dtype == np.float32
    np.testing.assert_array_equal(table_log, ref)


def test_cb19_log_line_ratios_are_float64_rounded_once():
    from tengri.components.nebular import cloudy_cb19 as cb

    rng = np.random.default_rng(3067)
    ratios = (10.0 ** rng.uniform(-12, 2, size=100_000)).astype(np.float32)
    ratios[:3] = [0.0, np.nan, -1.0]
    got = cb._log10_ratios_f32(ratios)
    valid = np.isfinite(ratios) & (ratios > 0)
    ref = np.log10(np.where(valid, ratios.astype(np.longdouble), cb._LOG_FLOOR)).astype(np.float32)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, ref)
    assert got[0] == got[1] == got[2] == np.float32(np.log10(cb._LOG_FLOOR))

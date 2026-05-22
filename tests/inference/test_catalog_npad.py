# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for ``CatalogFitter._resolve_n_padded``.

End-to-end correctness with real SSP data lives in
``tests/integration/test_catalog_npad_e2e.py``.
"""

from __future__ import annotations

import pytest

from tengri.inference.catalog_fitter import _resolve_n_padded

pytestmark = pytest.mark.contract


class TestResolveNPadded:
    def test_none_yields_multiple_of_K(self):
        assert _resolve_n_padded(7, 4, None) == 8

    def test_none_already_multiple(self):
        assert _resolve_n_padded(8, 4, None) == 8

    def test_none_K_one(self):
        assert _resolve_n_padded(13, 1, None) == 13

    def test_auto_pads_to_next_pow2(self):
        assert _resolve_n_padded(5, 1, "auto") == 8

    def test_auto_already_pow2(self):
        assert _resolve_n_padded(8, 1, "auto") == 8

    def test_auto_with_K_rounds_up(self):
        assert _resolve_n_padded(5, 4, "auto") == 8

    def test_auto_K_larger_than_pow2(self):
        # n_gal=3, K=8 → multiple-of-K floor (8) wins over pow2 (4)
        assert _resolve_n_padded(3, 8, "auto") == 8

    def test_explicit_int_exact(self):
        assert _resolve_n_padded(7, 4, 16) == 16

    def test_explicit_int_rounds_up_to_K(self):
        # n_pad=10, K=4 → ceil(10/4)*4 = 12
        assert _resolve_n_padded(7, 4, 10) == 12

    def test_explicit_int_below_n_gal_raises(self):
        with pytest.raises(ValueError, match="must be >= n_galaxies"):
            _resolve_n_padded(10, 4, 5)

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="must be 'auto'"):
            _resolve_n_padded(7, 4, "fancy")

    def test_n_gal_one(self):
        assert _resolve_n_padded(1, 1, None) == 1
        assert _resolve_n_padded(1, 1, "auto") == 1
        assert _resolve_n_padded(1, 4, None) == 4

    def test_compile_reuse_signature(self):
        """5, 7 with auto+K=4 collapse to 8 → same XLA cache key."""
        assert _resolve_n_padded(5, 4, "auto") == 8
        assert _resolve_n_padded(7, 4, "auto") == 8
        # 9 → pow2(9)=16 → different bucket
        assert _resolve_n_padded(9, 4, "auto") == 16

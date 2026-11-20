# SPDX-License-Identifier: BSD-3-Clause
"""Contract test: ``tengri.list_dust_laws()`` is the one dust-law lister.

It carries an ``fn`` column holding the registered callable for each law,
so ``list_dust_laws().to_dict("fn")`` is the mapping ``{name: DUST_LAWS[name]}``.
The column is hidden from the rendered table, as the other hidden columns are.
"""

import pytest

import tengri
from tengri.components.dust.laws._registry import DUST_LAWS

pytestmark = pytest.mark.contract


def test_list_dust_laws_carries_an_fn_column():
    rows = tengri.list_dust_laws()
    assert rows, "no attenuation laws"
    assert all("fn" in row for row in rows)


def test_list_dust_laws_fn_is_the_registered_callable_for_every_row():
    mapping = tengri.list_dust_laws().to_dict("fn")
    assert set(mapping) == set(DUST_LAWS)
    assert all(mapping[name] is DUST_LAWS[name] for name in DUST_LAWS)


def test_list_dust_laws_does_not_render_the_fn_column():
    header = repr(tengri.list_dust_laws()).splitlines()[0].split()
    assert "fn" not in header

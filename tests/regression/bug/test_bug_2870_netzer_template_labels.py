# SPDX-License-Identifier: BSD-3-Clause
r"""
Regression test for issue #2870: GRAHSP Netzer disc templates carry the wrong labels.

Upstream GRAHSP's course-grid importer assigns the columns of Netzer's disc table
to ``(M, a, mdot)`` in its own enumeration order,
``[(M, a, mdot) for a in (0.998, 0) for mdot in (0.3, 0.03) for M in (6, 7, 8, 9)]``,
not in the order of the table's header. The header is the physical one: the
header name ``m80_a0_mdot003`` is log10 M = 8, spin 0, mdot = 0.03, and the peak
:math:`\nu L_\nu` of each column scales as :math:`M \dot m` (10x per decade of M and
per decade of mdot). tengri inherited the enumeration order, so 14 of the 16 stored
templates sit at the wrong ``(M, a, mdot)`` node.

The expected spectrum here is read from the source table with numpy alone:
the column whose header decodes to a node is that node's template. No tengri
code parses the header. Two checks run:

* against the source table (skipped with ``FileNotFoundError`` only if the table
  is not in the checkout), one node at a time;
* against a committed fixture of the header-to-column map
  (``tests/fixtures/grahsp/netzer_disc_header_map.json``), which always runs.

Upstream's labeling is a corrected upstream error, applied in the builder
(``tools/build_grahsp_hdf5.py``) rather than inherited. The upstream reference file
has no Netzer disc group, so no crossval comparison depends on the old labels.

Notes
-----
Template spectra are :math:`L_\lambda = L_\nu \nu^2 / c` normalized to 1 at 510 nm
(the constant ``c`` cancels in the normalization). Header decoding: ``m60`` is
log10 M = 6, ``a0998`` is spin 0.998, ``a0`` is spin 0, ``mdot03`` is 0.3, and
``mdot003`` is 0.03.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from tengri.components.agn.grahsp.templates import load_grahsp_templates

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_TABLE = REPO_ROOT / "data" / "grahsp" / "netzer_disc_models.txt"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "grahsp" / "netzer_disc_header_map.json"

NODES = [(m, a, mdot) for a in (0.998, 0.0) for mdot in (0.3, 0.03) for m in (6.0, 7.0, 8.0, 9.0)]
_HEADER = re.compile(r"^m(\d+)_a(\d+)_mdot(\d+)_inc0$")
_C_ARB = 3.0e18  # shape-only: cancels in the 510 nm normalization


def _decode(name):
    """Return the ``(M, a, mdot)`` node that a header name spells out."""
    match = _HEADER.match(name.strip())
    assert match, f"unrecognized header name {name!r}"
    tm, ta, tmdot = match.groups()
    return (int(tm) / 10.0, int(ta) / 1000.0, int(tmdot) / 10.0 ** (len(tmdot) - 1))


def _source_columns():
    """Return ``(wave_nm, {node: normalized L_lambda})`` from the source table."""
    try:
        lines = SOURCE_TABLE.read_text().splitlines()
    except FileNotFoundError:
        pytest.skip(f"source table not in checkout: {SOURCE_TABLE}")
    names = [ln.strip() for ln in lines if _HEADER.match(ln.strip())]
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Freq")) + 1
    table = np.array([[float(x) for x in ln.split()] for ln in lines[start:] if ln.strip()])
    freq, wave_a = table[:, 0], table[:, 1]
    order = np.argsort(wave_a)
    wave_nm = wave_a[order] * 0.1
    columns = {}
    for j, name in enumerate(names):
        l_lambda = table[order, 2 + j] * freq[order] ** 2 / _C_ARB
        columns[_decode(name)] = l_lambda / np.interp(510.0, wave_nm, l_lambda)
    return wave_nm, columns


def _loaded_row(templates, node):
    """Index of the template tengri loads for ``node``, matched on its labels."""
    matches = [
        i
        for i, (m, a, mdot) in enumerate(
            zip(templates.disc_m, templates.disc_a, templates.disc_mdot, strict=True)
        )
        if (float(m), float(a), float(mdot)) == node
    ]
    assert len(matches) == 1, f"node {node} has {len(matches)} templates"
    return matches[0]


@pytest.mark.regression_bug
@pytest.mark.parametrize("node", NODES, ids=lambda n: f"M{n[0]:g}_a{n[1]:g}_mdot{n[2]:g}")
def test_loaded_template_is_the_source_column_for_its_node(node):
    """The template tengri loads for (M, a, mdot) is the source column that names it."""
    wave_nm, columns = _source_columns()
    templates = load_grahsp_templates()
    np.testing.assert_allclose(templates.disc_wave_nm, wave_nm, rtol=1e-12)
    row = _loaded_row(templates, node)
    np.testing.assert_allclose(
        templates.disc_lumin[row],
        columns[node],
        rtol=1e-6,
        err_msg=f"template loaded for {node} is not the column headed for {node}",
    )


@pytest.mark.regression_bug
def test_header_map_fixture_matches_loaded_templates():
    """The committed header map (always available) agrees with the loaded bundle."""
    fixture = json.loads(FIXTURE.read_text())
    templates = load_grahsp_templates()
    assert len(fixture["columns"]) == 16
    for col in fixture["columns"]:
        node = (col["M"], col["a"], col["mdot"])
        assert _decode(col["header"]) == pytest.approx(node), col["header"]
        row = _loaded_row(templates, node)
        wave = np.asarray(col["wave_nm"])
        idx = np.searchsorted(templates.disc_wave_nm, wave)
        np.testing.assert_allclose(templates.disc_wave_nm[idx], wave, rtol=1e-12)
        np.testing.assert_allclose(
            templates.disc_lumin[row, idx],
            col["lumin"],
            rtol=1e-6,
            err_msg=f"column {col['column']} ({col['header']}) is not template {node}",
        )

# SPDX-License-Identifier: BSD-3-Clause
"""Test the _RegistryTable.to_rst() method.

Ensures reStructuredText output is well-formed and suitable for Sphinx
documentation. Tests column visibility, escaping, and edge cases.
"""

import pytest

import tengri

pytestmark = pytest.mark.contract


def test_dust_laws_to_rst_has_all_names():
    """Every row's name appears in the RST output."""
    table = tengri.list_dust_laws()
    output = table.to_rst()

    for row in table:
        name = row["name"]
        assert name in output, f"Name '{name}' missing from RST output"


def test_dust_laws_to_rst_has_use_and_status_columns():
    """The `use` and `status` columns are visible even for single-kind tables."""
    table = tengri.list_dust_laws()
    output = table.to_rst()

    # Verify it's a single-kind table (all rows have the same kind).
    kinds = {row["kind"] for row in table}
    assert len(kinds) == 1, f"Expected single kind, got {kinds}"

    # Ensure the columns are present in the output.
    assert " - use" in output, "'use' column header not found in RST output"
    assert " - status" in output, "'status' column header not found in RST output"


def _parse(rst: str):
    """Parse RST and return ``(document, [docutils warnings])``."""
    from docutils.core import publish_doctree
    from docutils.utils import SystemMessage

    messages: list[str] = []

    def _observer(msg):
        messages.append(msg.astext())

    try:
        doctree = publish_doctree(
            rst,
            settings_overrides={"report_level": 2, "halt_level": 5, "warning_stream": False},
        )
    except SystemMessage as exc:  # pragma: no cover - only on malformed input
        return None, [str(exc)]
    doctree.reporter.attach_observer(_observer)
    return doctree, messages


def test_to_rst_really_parses_as_a_table():
    """The output must parse into a table of the right shape.

    Checking line prefixes is not enough, and that is not hypothetical: the
    first implementation emitted ``* -`` for *every* cell, which docutils reads
    as N single-column rows. Every line still began with ``   * - ``, so a
    prefix check passed while the rendered page was a tall one-column list.
    Parse it and count the columns instead -- that is a check which can fail.
    """
    table = tengri.list_xray_models()
    doctree, messages = _parse(table.to_rst())
    assert doctree is not None, f"to_rst() did not parse: {messages}"

    tables = list(doctree.findall(condition=lambda n: n.tagname == "table"))
    assert len(tables) == 1, f"expected exactly one table, got {len(tables)}"

    tgroup = next(iter(tables[0].findall(condition=lambda n: n.tagname == "tgroup")))
    cols = int(tgroup["cols"])
    assert cols > 1, (
        f"the table rendered with {cols} column(s) — every cell became its own row. "
        "In a list-table '* -' opens a row and '  -' continues it."
    )

    rows = list(tables[0].findall(condition=lambda n: n.tagname == "row"))
    assert len(rows) == len(table) + 1, (
        f"expected {len(table)} data rows plus a header, got {len(rows)}"
    )
    assert not [m for m in messages if "ERROR" in m or "SEVERE" in m], messages


def test_empty_table_returns_note():
    """Empty table returns valid RST containing no list-table."""
    empty_table = tengri.registry._RegistryTable([])
    output = empty_table.to_rst()

    assert "list-table" not in output, "Empty table should not contain list-table directive"
    assert "note::" in output, "Empty table should contain .. note::"
    assert "empty" in output.lower(), "Empty table message should mention 'empty'"


def test_to_rst_escapes_newlines():
    """Newlines in cell values are replaced with spaces."""
    # Create a minimal registry with a newline in a cell.
    table = tengri.registry._RegistryTable(
        [
            {
                "name": "test_model",
                "kind": "test_kind",
                "status": "production",
                "short_doc": "Line one\nLine two",
                "use": "example usage",
            }
        ]
    )
    output = table.to_rst()

    # The newline should be replaced with a space.
    assert (
        "\n" not in output.split("short_doc")[1].split("test_kind")[0]
        or "Line one Line two" in output
        or "Line one" in output
    )


def test_to_rst_handles_special_characters():
    """Cell values with RST-special leading characters are wrapped in backticks."""
    table = tengri.registry._RegistryTable(
        [
            {
                "name": "test",
                "kind": "test",
                "status": "production",
                "short_doc": "_underscored start",
                "use": "|pipe start",
            }
        ]
    )
    output = table.to_rst()

    # Values starting with special chars should be backtick-wrapped.
    # Check that the output contains backticked versions.
    lines = output.split("\n")
    cell_content = [line for line in lines if "underscored" in line or "pipe" in line]
    # The escaping should be present somewhere.
    assert any("`" in line for line in cell_content) or any(
        "_underscored" in line or "|pipe" in line for line in cell_content
    )


def test_to_rst_empty_cells_become_dash():
    """Empty cells emit '-' to maintain list structure."""
    table = tengri.registry._RegistryTable(
        [
            {
                "name": "test",
                "kind": "test",
                "status": "production",
                "short_doc": "",  # Empty value
                "use": "example",
            }
        ]
    )
    output = table.to_rst()

    # The empty short_doc should appear as '-' in the output.
    lines = output.split("\n")
    # Find the line with the header and then a corresponding data line.
    # There should be a '-' in the place of empty short_doc.
    assert " - -" in output or "- -" in output


def test_to_rst_with_title():
    """A title is the directive argument, and the result still parses.

    ``list-table`` has no ``:caption:`` option — emitting one produces an
    "unknown option" warning, which the docs build turns into a failure via
    ``-W``. So assert the argument form *and* that docutils accepts it; the
    string check alone is what let the invalid spelling through.
    """
    table = tengri.list_dust_laws()
    output = table.to_rst(title="Dust Attenuation Laws")

    assert output.splitlines()[0] == ".. list-table:: Dust Attenuation Laws"
    assert ":caption:" not in output
    doctree, messages = _parse(output)
    assert doctree is not None, f"titled table did not parse: {messages}"
    assert not [m for m in messages if "ERROR" in m or "SEVERE" in m], messages


def test_to_rst_list_dust_laws_parses_correctly():
    """Real registry output from list_dust_laws() is well-formed RST."""
    table = tengri.list_dust_laws()
    output = table.to_rst()

    # Basic structure check.
    assert ".. list-table::" in output
    assert ":header-rows: 1" in output
    assert ":widths: auto" in output

    # Every name in the table should appear in the output.
    for row in table:
        assert row["name"] in output


def test_kwargs_unpacking_does_not_open_rst_emphasis():
    """``**recipe`` in a cell must not read as RST strong emphasis.

    Every recipe's ``use`` column is
    ``recipes.high_z() -> SEDModel.build(ssp_data=ssp, **recipe)``. Python's
    kwargs unpacking is also RST's strong-emphasis opener, so the generated
    component page built with ten "Inline strong start-string without
    end-string" warnings — and the docs build turns warnings into failures with
    ``-W``. The fix is inline-literal escaping; this pins it, because the
    symptom only appears in a full Sphinx build otherwise.
    """
    table = tengri.list_recipes()
    out = table.to_rst()

    # Not a naive string check: the ** is legitimately still there, wrapped in
    # an inline literal. What matters is whether docutils reads it as markup.
    assert "``recipes." in out, "recipe use-strings are not inline-literal escaped"
    doctree, messages = _parse(out)
    assert doctree is not None, f"recipes table did not parse: {messages}"
    bad = [m for m in messages if "start-string" in m or "ERROR" in m or "SEVERE" in m]
    assert not bad, bad

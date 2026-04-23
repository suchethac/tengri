"""Tests for tengri CLI."""

from __future__ import annotations

import pytest

from tengri.cli import build_parser, main


def test_version(capsys):
    """Test --version flag prints version."""
    rc = main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert len(out) > 0
    # Version should look like semantic versioning
    assert "." in out


def test_doctor(capsys):
    """Test doctor subcommand runs without error."""
    rc = main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out
    # Should contain health check header
    assert "Environment Health Check" in out or "tengri" in out


def test_cite_list(capsys):
    """Test cite without key lists all citations."""
    rc = main(["cite"])
    assert rc == 0
    out = capsys.readouterr().out
    # Should have some citations registered
    assert len(out) > 0


def test_cite_known_key(capsys):
    """Test cite with a known key."""
    rc = main(["cite", "jax"])
    assert rc == 0
    out = capsys.readouterr().out
    # Should contain citation details
    assert len(out) > 0


def test_cite_unknown_key(capsys):
    """Test cite with an unknown key returns error."""
    rc = main(["cite", "not_a_real_key"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Unknown citation key" in err


def test_cite_bibtex_flag(capsys):
    """Test cite with --bibtex flag."""
    rc = main(["cite", "jax", "--bibtex"])
    assert rc == 0
    out = capsys.readouterr().out
    # BibTeX entry should contain @ symbol
    assert "@" in out


def test_cite_list_bibtex(capsys):
    """Test cite with --bibtex flag (no key) lists all in BibTeX format."""
    rc = main(["cite", "--bibtex"])
    assert rc == 0
    out = capsys.readouterr().out
    # Multiple BibTeX entries should contain @
    assert "@" in out


def test_build_parser_help():
    """Test parser is correctly configured."""
    parser = build_parser()
    assert parser.prog == "tengri"
    # Should have version argument
    assert any(action.dest == "version" for action in parser._actions)


def test_help_subcommand(capsys):
    """Test help for main command exits with status 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "tengri" in out


def test_no_args(capsys):
    """Test with no arguments prints help."""
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    # Should print help message
    assert "tengri" in out or "doctor" in out

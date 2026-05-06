"""Tests for the _display helper function."""

from __future__ import annotations

import os

import pytest

from tengri._display import _display


def test_display_prints_by_default(capsys):
    """_display should print text to stdout by default."""
    _display("test output")
    captured = capsys.readouterr()
    assert captured.out == "test output\n"


def test_display_silent_with_quiet_env(capsys, monkeypatch):
    """_display should suppress output when TENGRI_QUIET=1."""
    monkeypatch.setenv("TENGRI_QUIET", "1")
    _display("test output")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_display_prints_with_quiet_other_value(capsys, monkeypatch):
    """_display should print unless TENGRI_QUIET is exactly '1'."""
    monkeypatch.setenv("TENGRI_QUIET", "0")
    _display("test output")
    captured = capsys.readouterr()
    assert captured.out == "test output\n"


def test_display_empty_string(capsys):
    """_display should handle empty strings."""
    _display("")
    captured = capsys.readouterr()
    assert captured.out == "\n"


def test_display_multiline(capsys):
    """_display should handle multiline text."""
    text = "line 1\nline 2\nline 3"
    _display(text)
    captured = capsys.readouterr()
    assert captured.out == "line 1\nline 2\nline 3\n"

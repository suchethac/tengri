# SPDX-License-Identifier: BSD-3-Clause
"""R73: a ``_data_setup`` message never prints an absolute path.

``download_ssp`` and ``download_template`` print where the file is, and those
lines are captured verbatim by every rendered notebook that calls them. Run
from a worktree, ``SSP file already exists at {filepath}; skipping download.``
put this machine's absolute path into ``docs/reproduction/*.ipynb`` -- four
leaks across three renders, and a fresh one on every render from anywhere but
the repository root (``tools/check_no_local_paths.py``, lint job).

The rule: show the path relative to the working directory when the file is
under it, otherwise the file's name alone. Never ``str(absolute_path)``. One
helper, ``_display_path``, and every such message goes through it.
"""

from __future__ import annotations

import os
import pathlib

import pytest

pytestmark = pytest.mark.contract

_FORBIDDEN = ("/Users/", "/home/")


def _probe_file(directory: pathlib.Path, name: str = "probe_grid.h5") -> pathlib.Path:
    """A non-empty file, so the "already exists" branch is the one taken."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"not really hdf5, but non-empty")
    return path


class TestDisplayPath:
    """The helper itself."""

    def test_a_file_under_the_working_directory_is_relative(self, tmp_path, monkeypatch):
        from tengri._data_setup import _display_path

        monkeypatch.chdir(tmp_path)
        path = _probe_file(tmp_path / "data")
        assert _display_path(path) == os.path.join("data", "probe_grid.h5")

    def test_the_working_directory_itself_is_relative(self, tmp_path, monkeypatch):
        from tengri._data_setup import _display_path

        monkeypatch.chdir(tmp_path)
        path = _probe_file(tmp_path)
        assert _display_path(path) == "probe_grid.h5"

    def test_a_file_outside_the_working_directory_is_its_name(self, tmp_path, monkeypatch):
        from tengri._data_setup import _display_path

        outside = tmp_path / "elsewhere"
        path = _probe_file(outside)
        monkeypatch.chdir(tmp_path / "cwd" if (tmp_path / "cwd").exists() else tmp_path)
        (tmp_path / "cwd").mkdir(exist_ok=True)
        monkeypatch.chdir(tmp_path / "cwd")
        assert _display_path(path) == "probe_grid.h5"

    @pytest.mark.parametrize("under_cwd", [True, False])
    def test_no_home_directory_either_way(self, tmp_path, monkeypatch, under_cwd):
        """The property that matters, stated directly."""
        from tengri._data_setup import _display_path

        cwd = tmp_path / "cwd"
        cwd.mkdir()
        path = _probe_file(cwd / "data" if under_cwd else tmp_path / "elsewhere")
        monkeypatch.chdir(cwd)
        shown = _display_path(path)
        assert not any(tok in shown for tok in _FORBIDDEN), shown
        assert not pathlib.Path(shown).is_absolute(), shown


class TestEveryMessageIsRouted:
    """The four call sites, through the functions that print them."""

    def test_ssp_already_exists_message(self, tmp_path, monkeypatch, capsys):
        import tengri._data_setup as ds

        monkeypatch.delenv("TENGRI_QUIET", raising=False)
        monkeypatch.chdir(tmp_path)
        _probe_file(tmp_path / "data", "probe_ssp.h5")
        ds.download_ssp(name="probe_ssp.h5", dest=tmp_path / "data")
        out = capsys.readouterr().out
        assert "already exists" in out, out
        assert not any(tok in out for tok in _FORBIDDEN), out
        assert os.path.join("data", "probe_ssp.h5") in out, out
        # The point of the rule: the absolute prefix must be gone. On this
        # platform a tmp_path is not under /Users/, so _FORBIDDEN alone would
        # pass on the very string this test exists to refuse.
        assert str(tmp_path.resolve()) not in out, out
        assert not any(
            pathlib.Path(word).is_absolute() for word in out.split() if "probe_ssp.h5" in word
        ), out

    def test_template_already_exists_message(self, tmp_path, monkeypatch, capsys):
        import tengri._data_setup as ds

        monkeypatch.delenv("TENGRI_QUIET", raising=False)
        monkeypatch.chdir(tmp_path)
        _probe_file(tmp_path / "data", "probe_tpl.h5")
        ds.download_template("probe_tpl.h5", dest=tmp_path / "data")
        out = capsys.readouterr().out
        assert "already exists" in out, out
        assert not any(tok in out for tok in _FORBIDDEN), out
        assert os.path.join("data", "probe_tpl.h5") in out, out
        # The point of the rule: the absolute prefix must be gone. On this
        # platform a tmp_path is not under /Users/, so _FORBIDDEN alone would
        # pass on the very string this test exists to refuse.
        assert str(tmp_path.resolve()) not in out, out
        assert not any(
            pathlib.Path(word).is_absolute() for word in out.split() if "probe_tpl.h5" in word
        ), out

    def test_a_destination_outside_the_cwd_shows_only_the_name(
        self, tmp_path, monkeypatch, capsys
    ):
        """The other branch, end to end: a download directory elsewhere."""
        import tengri._data_setup as ds

        monkeypatch.delenv("TENGRI_QUIET", raising=False)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        elsewhere = tmp_path / "elsewhere"
        _probe_file(elsewhere, "probe_ssp.h5")
        ds.download_ssp(name="probe_ssp.h5", dest=elsewhere)
        out = capsys.readouterr().out
        assert "probe_ssp.h5" in out, out
        assert "elsewhere" not in out, out
        assert not any(tok in out for tok in _FORBIDDEN), out

    def test_no_message_in_the_module_formats_a_bare_path(self):
        """The wiring, so a fifth message cannot be added unrouted.

        The four ``_display`` calls that name a file must interpolate
        ``_display_path(...)``, never the ``Path`` itself.
        """
        import re

        source = pathlib.Path(ds_source()).read_text()
        offenders = [
            line.strip()
            for line in source.splitlines()
            if re.search(r"_display\(f\".*\{(filepath|dest|input_dir|path)\}", line)
        ]
        assert not offenders, (
            "these messages interpolate a Path directly; route them through "
            "_display_path:\n  " + "\n  ".join(offenders)
        )


def ds_source() -> str:
    import tengri._data_setup as ds

    return ds.__file__

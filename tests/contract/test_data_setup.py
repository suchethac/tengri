"""Tests for SSP data setup and download helpers."""

import pytest

from tengri._data_setup import download_ssp, list_known_ssps

pytestmark = pytest.mark.contract


@pytest.mark.unit
def test_list_known_ssps_includes_default():
    """Verify the default FSPS PARSEC+MILES Chabrier SSP is in the known list."""
    ssps = list_known_ssps()
    assert "fsps_prsc_miles_chabrier" in ssps
    assert ssps["fsps_prsc_miles_chabrier"] == "fsps_prsc_miles_chabrier.h5"


@pytest.mark.unit
def test_list_known_ssps_returns_copy():
    """Verify that list_known_ssps returns a new dict, not the internal dict."""
    ssps1 = list_known_ssps()
    ssps2 = list_known_ssps()
    assert ssps1 == ssps2
    assert ssps1 is not ssps2


@pytest.mark.unit
def test_download_ssp_skips_existing(tmp_path, monkeypatch):
    """Verify that download_ssp skips existing files without calling urlopen."""
    # Pre-create the target file with non-zero size
    target_file = tmp_path / "fsps_prsc_miles_chabrier.h5"
    target_file.write_bytes(b"existing data")
    mtime_before = target_file.stat().st_mtime

    # Monkeypatch urlopen to fail if called
    def mock_urlopen(*args, **kwargs):
        raise RuntimeError("urlopen should not be called when file exists")

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    # Call download_ssp with the existing file path
    result = download_ssp(dest=tmp_path)

    # Verify file wasn't modified
    assert result == target_file
    assert target_file.read_bytes() == b"existing data"
    assert target_file.stat().st_mtime == mtime_before


@pytest.mark.unit
def test_download_ssp_uses_env_var(tmp_path, monkeypatch):
    """Verify that TENGRI_DATA_DIR env var is respected."""
    # Set the env var
    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))

    # Mock the download to avoid network calls
    def mock_urlopen(url, timeout=None):
        class FakeResponse:
            status = 200

            def read(self, chunk_size):
                # Return 16 bytes total: 2 chunks of 8 bytes
                if not hasattr(self, "_bytes_read"):
                    self._bytes_read = 0
                if self._bytes_read < 16:
                    chunk = b"x" * min(8, 16 - self._bytes_read)
                    self._bytes_read += len(chunk)
                    return chunk
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    # Call download_ssp with force=True (to override any existing file)
    result = download_ssp(force=True)

    # Verify file was created in the env var directory
    expected_path = tmp_path / "fsps_prsc_miles_chabrier.h5"
    assert result == expected_path
    assert expected_path.exists()
    assert expected_path.stat().st_size == 16


@pytest.mark.unit
def test_download_ssp_unknown_name(tmp_path):
    """Verify that download_ssp raises KeyError for unknown SSP names."""
    with pytest.raises(KeyError, match="Unknown SSP name"):
        download_ssp(name="nonexistent_ssp", dest=tmp_path)


@pytest.mark.unit
def test_download_ssp_default_dest_uses_cwd(tmp_path, monkeypatch):
    """Verify that default dest uses 'data/' relative to current directory."""

    # Change to tmp_path and mock download
    monkeypatch.chdir(tmp_path)

    # Mock the download
    def mock_urlopen(url, timeout=None):
        class FakeResponse:
            status = 200

            def read(self, chunk_size):
                if not hasattr(self, "_bytes_read"):
                    self._bytes_read = 0
                if self._bytes_read < 8:
                    chunk = b"x" * 8
                    self._bytes_read += 8
                    return chunk
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    # Call with force=True to override any existing file
    result = download_ssp(force=True)

    # Verify file was created in data/ directory (relative to tmp_path)
    expected_file = tmp_path / "data" / "fsps_prsc_miles_chabrier.h5"
    assert expected_file.exists()
    assert result.name == "fsps_prsc_miles_chabrier.h5"
    assert result.parent.name == "data"


@pytest.mark.unit
def test_download_ssp_creates_dest_directory(tmp_path, monkeypatch):
    """Verify that download_ssp creates the destination directory if needed."""
    dest = tmp_path / "subdir" / "another"
    assert not dest.exists()

    # Mock the download
    def mock_urlopen(url, timeout=None):
        class FakeResponse:
            status = 200

            def read(self, chunk_size):
                if not hasattr(self, "_bytes_read"):
                    self._bytes_read = 0
                if self._bytes_read < 8:
                    chunk = b"x" * 8
                    self._bytes_read += 8
                    return chunk
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    # Call download_ssp
    result = download_ssp(dest=dest, force=True)

    # Verify directory was created
    assert dest.exists()
    assert result.parent == dest

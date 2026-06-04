# SPDX-License-Identifier: BSD-3-Clause
"""Convenience helpers for downloading pre-computed SSP grids."""

import os
import urllib.error
import urllib.request
from pathlib import Path

__all__ = [
    "KNOWN_SSP_FILENAMES",
    "data_path",
    "download_ssp",
    "list_available_ssps",
    "list_known_ssps",
]


def data_path(filename: str) -> Path:
    """Locate a bundled data file by walking parent dirs for ``data/<filename>``.

    Sphinx-gallery ``chdir``s into each script's directory before exec, so
    hand-coded relative paths like ``"data/foo.h5"`` resolve to
    ``examples/<section>/data/foo.h5`` — which does not exist. This helper
    walks from ``Path.cwd()`` upward until it finds a sibling ``data/``
    directory containing the requested file.

    Parameters
    ----------
    filename : str
        Basename of the data file, e.g. ``"bosa_templates.h5"``.

    Returns
    -------
    pathlib.Path
        Absolute path to the file.

    Raises
    ------
    FileNotFoundError
        If no ``data/<filename>`` exists in any ancestor directory.

    Examples
    --------
    >>> from tengri import data_path
    >>> templates = data_path("bosa_templates.h5")
    >>> import h5py
    ...
    ... h5py.File(templates).keys()
    """
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / "data" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Data file 'data/{filename}' not found in any ancestor of {Path.cwd()}. "
        f"Place it under <project_root>/data/."
    )


SSP_BASE_URL = "https://halos.as.arizona.edu/suchethacooray/ssp-spectra/"

# Short alias → filename, matches the live public catalogue. The catalogue
# only ships *bare-stellar* SSPs; the ``_wNE_*`` variants in local ``data/``
# trees are post-processed (FSPS+nebular). Cue / CloudyGrid backends require
# bare SSPs and will silently under-predict if fed wNE.
_KNOWN_SSPS = {
    "fsps_prsc_miles_chabrier": "fsps_prsc_miles_chabrier.h5",
    "fsps_mist_c3k_a_chabrier": "fsps_mist_c3k_a_chabrier.h5",
    "fsps_mist_miles_chabrier": "fsps_mist_miles_chabrier.h5",
    "fsps_pdva_miles_chabrier": "fsps_pdva_miles_chabrier.h5",
    "fsps_pdva_c3k_a_chabrier": "fsps_pdva_c3k_a_chabrier.h5",
    "fsps_bsti_miles_chabrier": "fsps_bsti_miles_chabrier.h5",
    "fsps_bsti_c3k_a_chabrier": "fsps_bsti_c3k_a_chabrier.h5",
    "fsps_bsti_basel_chabrier": "fsps_bsti_basel_chabrier.h5",
    "fsps_mist_basel_chabrier": "fsps_mist_basel_chabrier.h5",
    "fsps_pdva_basel_chabrier": "fsps_pdva_basel_chabrier.h5",
    "fsps_prsc_basel_chabrier": "fsps_prsc_basel_chabrier.h5",
    "fsps_prsc_c3k_a_chabrier": "fsps_prsc_c3k_a_chabrier.h5",
    "bc03_pdva_stelib_chabrier": "bc03_pdva_stelib_chabrier.h5",
    "bpss_stars_c3k_a_chabrier": "bpss_stars_c3k_a_chabrier.h5",
    "pgny_mist_c3k_chabrier": "pgny_mist_c3k_chabrier.h5",
    # Alternative IMFs
    "fsps_mist_c3k_a_kroupa": "fsps_mist_c3k_a_kroupa.h5",
    "fsps_mist_c3k_a_salpeter": "fsps_mist_c3k_a_salpeter.h5",
    "fsps_prsc_miles_kroupa": "fsps_prsc_miles_kroupa.h5",
    "fsps_prsc_miles_salpeter": "fsps_prsc_miles_salpeter.h5",
    "fsps_prsc_c3k_a_kroupa": "fsps_prsc_c3k_a_kroupa.h5",
    "fsps_prsc_c3k_a_salpeter": "fsps_prsc_c3k_a_salpeter.h5",
}

# Reverse lookup used by ``load_ssp_data`` to auto-fetch a missing local
# file when the basename matches a known catalogue entry.
KNOWN_SSP_FILENAMES = frozenset(_KNOWN_SSPS.values())


def list_known_ssps() -> dict[str, str]:
    """Return a dict of known SSP names and their filenames.

    Returns
    -------
    dict[str, str]
        Mapping of short SSP identifier (e.g., ``"fsps_prsc_miles_chabrier"``)
        to filename (e.g., ``"fsps_prsc_miles_chabrier.h5"``).

    Examples
    --------
    >>> import tengri
    >>> ssps = tengri.list_known_ssps()
    >>> "fsps_prsc_miles_chabrier" in ssps
    True
    """
    return _KNOWN_SSPS.copy()


def list_available_ssps() -> list[dict]:
    """Structured view of the SSP catalogue grouped by family and IMF (#307).

    Returns
    -------
    list of dict
        One row per registered SSP. Keys per row::

            {
                "name": "fsps_prsc_miles_chabrier",
                "family": "fsps_prsc_miles",
                "imf": "chabrier",
                "filename": "fsps_prsc_miles_chabrier.h5",
                "downloaded": True or False,
            }

        ``family`` is the SSP name with the IMF suffix stripped (e.g.
        ``"fsps_prsc_miles"`` from ``"fsps_prsc_miles_chabrier"``).
        ``downloaded`` is ``True`` iff the file is present under any
        ``data/`` directory walked upward from ``Path.cwd()`` — the same
        discovery rule :func:`tengri.load_ssp` uses.

    Examples
    --------
    Find every IMF available for the FSPS+MIST+MILES family::

        >>> import tengri
        >>> rows = tengri.list_available_ssps()
        >>> {r["imf"] for r in rows if r["family"] == "fsps_mist_c3k_a"}
        {'chabrier', 'kroupa', 'salpeter'}

    See also
    --------
    list_known_ssps : The flat ``name → filename`` mapping that this
        view groups and enriches.
    """
    from tengri.components.stellar.sps.dsps_wrapper import _KNOWN_IMFS

    out: list[dict] = []
    for name, filename in _KNOWN_SSPS.items():
        # Strip the trailing IMF token to recover the family stem.
        imf = "unknown"
        family = name
        for token in _KNOWN_IMFS:
            suffix = "_" + token
            if name.endswith(suffix):
                imf = token
                family = name[: -len(suffix)]
                break
        out.append(
            {
                "name": name,
                "family": family,
                "imf": imf,
                "filename": filename,
                "downloaded": _ssp_file_present(filename),
            }
        )
    return sorted(out, key=lambda r: (r["family"], r["imf"]))


def _ssp_file_present(filename: str) -> bool:
    """True iff ``filename`` is found under a ``data/`` dir walked
    upward from the current working directory.

    Mirrors :func:`tengri.load_ssp`'s ancestor-walk discovery so the
    ``downloaded`` flag on :func:`list_available_ssps` matches the
    actual ``load_ssp`` outcome users will see.
    """
    cur = Path.cwd().resolve()
    for _ in range(6):
        candidate = cur / "data" / filename
        if candidate.is_file():
            return True
        if cur.parent == cur:
            break
        cur = cur.parent
    return False


def download_ssp(
    name: str = "fsps_prsc_miles_chabrier",
    dest: str | os.PathLike | None = None,
    force: bool = False,
) -> Path:
    """Download a pre-formatted SSP HDF5 file used by tengri's stellar component.

    Parameters
    ----------
    name : str, optional
        Short SSP identifier. See ``list_known_ssps()``. Defaults to
        ``"fsps_prsc_miles_chabrier"`` (FSPS PARSEC tracks + MILES library,
        Chabrier IMF — bare-stellar, Cue/CloudyGrid-compatible).
    dest : path-like, optional
        Target directory. Defaults to ``$TENGRI_DATA_DIR`` if set, else ``data/``
        relative to the current working directory.
    force : bool, optional
        Re-download even if the file already exists. Default ``False``.

    Returns
    -------
    pathlib.Path
        Path to the downloaded file.

    Raises
    ------
    KeyError
        If ``name`` is not in ``list_known_ssps()``.
    RuntimeError
        If the HTTP download fails (network error or non-200 status).

    Examples
    --------
    >>> import tengri
    >>> tengri.download_ssp()  # FSPS v3.2 → data/
    >>> tengri.download_ssp("bc03_v3.2", dest="/scratch/ssp")
    """
    # Resolve target directory
    if dest is None:
        dest = os.environ.get("TENGRI_DATA_DIR", "data")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Resolve filename
    if name not in _KNOWN_SSPS:
        raise KeyError(
            f"Unknown SSP name: {name!r}. Known names: {list(list_known_ssps().keys())}"
        )
    filename = _KNOWN_SSPS[name]
    filepath = dest / filename

    # Skip if already exists and force=False
    if filepath.exists() and filepath.stat().st_size > 0 and not force:
        from tengri._display import _display

        _display(f"SSP file already exists at {filepath}; skipping download.")
        return filepath

    # Download with progress
    url = SSP_BASE_URL + filename
    partial_filepath = filepath.with_suffix(filepath.suffix + ".partial")

    try:
        _download_file(url, partial_filepath)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        # Clean up partial file on error
        if partial_filepath.exists():
            partial_filepath.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    except KeyboardInterrupt:
        # Clean up partial file on interrupt
        if partial_filepath.exists():
            partial_filepath.unlink()
        raise

    # Atomic rename
    partial_filepath.replace(filepath)

    from tengri._display import _display

    _display(f"Downloaded SSP to {filepath}")
    return filepath


# Pre-converted component templates (HDF5) live alongside the SSP catalogue on
# the public host. These are tengri-native conversions of upstream libraries
# whose raw form is awkward to redistribute — e.g. the Fritz 2006 torus grid,
# which upstream ships only as ~24k pcigale-pickled objects (un-loadable
# without pcigale). Hosting the converted grid lets end-users fetch it with
# zero CIGALE dependency, exactly as SSPs are fetched.
TEMPLATE_BASE_URL = "https://halos.as.arizona.edu/suchethacooray/templates/"


def download_template(
    filename: str,
    dest: str | os.PathLike | None = None,
    force: bool = False,
) -> Path:
    """Download a pre-converted component-template HDF5 file.

    Mirrors :func:`download_ssp` but for component templates (AGN torus grids,
    dust IR libraries, …) hosted under :data:`TEMPLATE_BASE_URL`. The end-user
    never needs CIGALE installed — the converted ``.h5`` is fetched directly.

    Parameters
    ----------
    filename : str
        Basename of the hosted file, e.g. ``"fritz2006_torus_grid.h5"``.
    dest : path-like, optional
        Target directory. Defaults to ``$TENGRI_DATA_DIR`` if set, else ``data/``
        relative to the current working directory.
    force : bool, optional
        Re-download even if the file already exists. Default ``False``.

    Returns
    -------
    pathlib.Path
        Path to the downloaded file.

    Raises
    ------
    RuntimeError
        If the HTTP download fails (network error or non-200 status).
    """
    if dest is None:
        dest = os.environ.get("TENGRI_DATA_DIR", "data")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    filepath = dest / filename
    if filepath.exists() and filepath.stat().st_size > 0 and not force:
        from tengri._display import _display

        _display(f"Template already exists at {filepath}; skipping download.")
        return filepath

    url = TEMPLATE_BASE_URL + filename
    partial_filepath = filepath.with_suffix(filepath.suffix + ".partial")
    try:
        _download_file(url, partial_filepath)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        if partial_filepath.exists():
            partial_filepath.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    except KeyboardInterrupt:
        if partial_filepath.exists():
            partial_filepath.unlink()
        raise

    partial_filepath.replace(filepath)
    from tengri._display import _display

    _display(f"Downloaded template to {filepath}")
    return filepath


def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file from a URL to a destination path with simple progress.

    Parameters
    ----------
    url : str
        URL to download from.
    dest : Path
        Destination file path (should end with .partial for atomic write).
    chunk_size : int, optional
        Size of each download chunk in bytes. Default 8192.

    Raises
    ------
    urllib.error.HTTPError
        If the HTTP response status is not 200.
    urllib.error.URLError
        If the network request fails.
    """
    # Try to use tqdm for progress bar if available, otherwise silent
    try:
        import tqdm

        use_progress = True
    except ImportError:
        use_progress = False

    # Perform HEAD request to get total size
    try:
        head_request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head_request, timeout=10) as response:
            total_size = int(response.headers.get("Content-Length", 0))
    except Exception:
        total_size = 0

    # Download with progress
    downloaded = 0
    with open(dest, "wb") as f, urllib.request.urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise urllib.error.HTTPError(
                url, response.status, f"HTTP {response.status}", response.headers, None
            )

        if use_progress and total_size > 0:
            pbar = tqdm.tqdm(total=total_size, unit="B", unit_scale=True)
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    pbar.close()
                    break
                f.write(chunk)
                downloaded += len(chunk)
                pbar.update(len(chunk))
        else:
            # Silent download with simple progress messages
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (1024 * 1024) == 0:  # Log every 1 MB
                    from tengri._display import _display

                    _display(f"Downloaded {downloaded / (1024 * 1024):.1f} MB...")

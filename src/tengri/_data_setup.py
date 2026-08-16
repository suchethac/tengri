# SPDX-License-Identifier: BSD-3-Clause
"""Convenience helpers for downloading pre-computed SSP grids.

Every "where is my data?" question in tengri resolves through :func:`data_dirs`
(reads) and :func:`download_dir` (writes), so a user who sets one environment
variable is found by the loaders, the downloaders, and ``tengri.doctor()``
alike.
"""

import os
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

__all__ = [
    "KNOWN_SSP_FILENAMES",
    "TENGRI_DATA_ENV",
    "data_dirs",
    "data_path",
    "download_dir",
    "download_ssp",
    "find_data_str",
    "find_ssp_files",
    "list_available_ssps",
    "list_known_ssps",
]

#: Environment variable naming tengri's data directory. Governs both where
#: downloads are written and where loaders look, so pointing it at a scratch
#: filesystem moves the whole data story at once.
TENGRI_DATA_ENV = "TENGRI_DATA_DIR"

#: Deprecated spelling. Honored with a warning so existing setups keep working.
_TENGRI_DATA_ENV_LEGACY = "TENGRI_DATA"


def _env_data_dir() -> Path | None:
    """The user's configured data directory, or ``None`` if unset.

    Prefers ``$TENGRI_DATA_DIR``; falls back to the deprecated ``$TENGRI_DATA``
    with a warning. The two spellings previously governed *different* halves of
    the data story — ``TENGRI_DATA_DIR`` where downloads were written,
    ``TENGRI_DATA`` where ``doctor()`` looked — so setting either one alone left
    the other half pointed somewhere else.
    """
    value = os.environ.get(TENGRI_DATA_ENV)
    if value:
        return Path(value).expanduser()
    legacy = os.environ.get(_TENGRI_DATA_ENV_LEGACY)
    if legacy:
        warnings.warn(
            f"${_TENGRI_DATA_ENV_LEGACY} is deprecated and will stop being read; "
            f"use ${TENGRI_DATA_ENV} instead. It is now honored for both reads "
            f"and writes, so the single variable is enough.",
            DeprecationWarning,
            stacklevel=3,
        )
        return Path(legacy).expanduser()
    return None


def data_dirs() -> list[Path]:
    """Every directory tengri looks in for data files, most specific first.

    Returns
    -------
    list of pathlib.Path
        In order: ``$TENGRI_DATA_DIR`` (or the deprecated ``$TENGRI_DATA``) if
        set; each ancestor of the working directory with a ``data/``
        subdirectory; ``~/tengri/data``; the working directory itself; and the
        source tree beside the installed package. Directories are returned
        whether or not they exist — callers test the file they want.

    Notes
    -----
    The ancestor walk exists because sphinx-gallery ``chdir``s into each
    script's directory before exec, so a hand-written ``"data/foo.h5"`` would
    otherwise resolve under ``examples/<section>/``.

    The last two groups exist so this function is a superset of the per-module
    grid locators it replaces (#1431). Those searched
    ``Path(__file__).resolve().parents[4] / "data/<name>.h5"`` and the bare
    ``<root>/<name>.h5``, plus the same two relative to the working directory —
    so the working directory *itself* and the package's own source root are both
    needed here, not only their ``data/`` subdirectories.

    Package-relative resolution is anchored on this module's location rather
    than counted in ``parents[N]`` hops. Depth counting has to be re-derived
    every time a file moves between directory levels: ``dust/emission/`` still
    carries a legacy ``parents[4]`` alongside its real ``parents[5]`` for
    exactly that reason. Anchoring here means callers at any depth get the same
    answer.
    """
    out: list[Path] = []
    env = _env_data_dir()
    if env is not None:
        out.append(env)
    out.extend(parent / "data" for parent in [Path.cwd(), *Path.cwd().parents])
    out.append(Path.home() / "tengri" / "data")
    # Bare working directory: covers a grid file sitting next to the script.
    out.append(Path.cwd())
    out.extend(package_data_dirs())
    # Deduplicate, first occurrence wins so $TENGRI_DATA_DIR keeps precedence.
    # Running from the repo root makes cwd and the package root coincide, and
    # the FileNotFoundError from data_path() lists what was searched.
    seen: set[Path] = set()
    return [d for d in out if not (d in seen or seen.add(d))]


def package_data_dirs() -> list[Path]:
    """Data directories beside the installed package, independent of the cwd.

    Returns
    -------
    list of pathlib.Path
        ``<source-root>/data`` and the bare ``<source-root>``, where the source
        root is resolved from this module's own location. For a ``src/`` layout
        checkout that is the repository root; for an installed wheel it is
        whatever sits above ``site-packages/tengri`` and simply will not
        contain the files, which is harmless — callers test the file they want.

    Notes
    -----
    This is the cwd-independent half of :func:`data_dirs`. It matters when the
    process runs from an unrelated working directory: the ancestor walk finds
    nothing, but a source checkout still has its ``data/`` beside the package.

    Anchored on ``__file__`` of this module (``src/tengri/_data_setup.py``), so
    the two hops to the source root are fixed no matter which component calls
    it. That is the property the per-module ``parents[N]`` locators lacked.
    """
    pkg_root = Path(__file__).resolve().parent  # <src>/tengri
    source_root = pkg_root.parent.parent  # <src>/tengri -> <src> -> <root>
    return [source_root / "data", source_root]


def download_dir() -> Path:
    """The directory :func:`download_ssp` and :func:`download_template` write to.

    Returns
    -------
    pathlib.Path
        ``$TENGRI_DATA_DIR`` (or the deprecated ``$TENGRI_DATA``) if set, else
        ``<cwd>/data``. Always identical to ``data_dirs()[0]``, so a downloaded
        file is always found by the loaders afterwards.

    Notes
    -----
    Returns an absolute path so it compares equal to the corresponding
    :func:`data_dirs` entry. A bare relative ``Path("data")`` names the same
    directory but is a different value, which is exactly the kind of drift that
    let the write path and the read path disagree in the first place.
    """
    return _env_data_dir() or (Path.cwd() / "data")


def data_path(filename: str) -> Path:
    """Locate a bundled data file in any directory tengri reads data from.

    Searches :func:`data_dirs`: ``$TENGRI_DATA_DIR`` first if set, then each
    ancestor of the working directory with a ``data/`` subdirectory, then
    ``~/tengri/data``. The ancestor walk matters because sphinx-gallery
    ``chdir``s into each script's directory before exec, so a hand-coded
    ``"data/foo.h5"`` would otherwise resolve under ``examples/<section>/``.

    Parameters
    ----------
    filename : str
        Basename of the data file, e.g. ``"bosa_templates.h5"``.

    Returns
    -------
    pathlib.Path
        Path to the file.

    Raises
    ------
    FileNotFoundError
        If ``filename`` exists in none of :func:`data_dirs`. The message names
        the directories searched.

    Examples
    --------
    >>> from tengri import data_path
    >>> templates = data_path("bosa_templates.h5")
    >>> import h5py
    ...
    ... h5py.File(templates).keys()
    """
    for directory in data_dirs():
        candidate = directory / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Data file {filename!r} not found. Looked in: "
        f"{', '.join(str(d) for d in data_dirs()[:4])} (and further ancestors). "
        f"Place it under <project_root>/data/, or set ${TENGRI_DATA_ENV} to the "
        f"directory holding it."
    )


def find_data(*filenames: str) -> Path | None:
    """First of ``filenames`` present in any data directory, or ``None``.

    The non-raising companion to :func:`data_path`, for the component grid
    locators that have their own not-found message or a legitimate ``None``
    result. Directory *names* are ignored: a caller may pass either a bare
    ``"grid.h5"`` or the historical ``"data/grid.h5"``.

    Parameters
    ----------
    *filenames : str
        Candidate files in preference order.

    Returns
    -------
    pathlib.Path or None
        The first candidate that exists, searching every :func:`data_dirs`
        entry for each name in turn. ``None`` if none of them exist anywhere.

    Notes
    -----
    Preference is name-major, not directory-major: the first *filename* that
    exists anywhere wins. The locators this replaces ranked their candidates by
    scientific fidelity (SKIRTOR ``_v3`` over ``_v2`` over ``.npz``), so a
    directory-major search would silently downgrade the grid whenever an older
    file happened to sit earlier on the path.
    """
    dirs = data_dirs()
    for filename in filenames:
        name = Path(filename).name
        for directory in dirs:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def find_data_str(*filenames: str) -> str | None:
    """:func:`find_data` as a ``str`` path, or ``None``.

    Parameters
    ----------
    *filenames : str
        Candidate files in preference order, as for :func:`find_data`.

    Returns
    -------
    str or None
        ``str`` of the first candidate that exists, or ``None``.

    Notes
    -----
    The template loaders want a string -- they interpolate it into messages and
    hand it to readers typed for ``str``. Three modules each carried their own
    ``_find_data_file`` doing this one conversion (#1431); they now delegate
    here, so the three cannot drift apart again.
    """
    found = find_data(*filenames)
    return str(found) if found is not None else None


def require_data(filename: str, not_found_msg: str) -> str:
    """Locate ``filename``, or raise ``FileNotFoundError`` with a curated message.

    The raising companion to :func:`find_data`, for the component grid locators
    whose absence is a hard error and whose message names the build or download
    step. Hoisted from six byte-identical bodies (#1431).

    Parameters
    ----------
    filename : str
        Basename of the grid file, e.g. ``"cat3d_wind_torus_grid.h5"``.
    not_found_msg : str
        Message to raise when the file is absent. Replaces :func:`data_path`'s
        generic text *entirely* rather than wrapping it.

    Returns
    -------
    str
        Path to the file, as a string — the shape the loaders take.

    Raises
    ------
    FileNotFoundError
        With exactly ``not_found_msg`` when the file exists in no data directory.

    Notes
    -----
    ``data_path`` names the directories it searched, which is the more useful
    message for a *misconfigured* install; ``not_found_msg`` names the grid to
    fetch, which is the more useful one for a *missing* grid. The locators are
    the missing-grid case, so the curated text wins outright — pinned by
    ``tests/contract/test_data_file_resolution.py``. The generic text is
    deliberately not appended: an astronomer who sees a build command should not
    have to read past a directory listing to find it.
    """
    try:
        return str(data_path(filename))
    except FileNotFoundError:
        raise FileNotFoundError(not_found_msg) from None


def package_or_env_data_path(filename: str) -> Path:
    """Locate ``filename`` without consulting the working directory.

    For module-level defaults, which are evaluated at import: :func:`data_path`
    raises, and :func:`data_dirs` is cwd-dependent, so neither is safe to bind
    to a constant or a default argument.

    Parameters
    ----------
    filename : str
        Basename of the data file, e.g. ``"cue_weights.npz"``.

    Returns
    -------
    pathlib.Path
        The file under ``$TENGRI_DATA_DIR`` if it is there, else under
        :func:`package_data_dirs`. Falls back to the package-anchored path when
        the file exists nowhere, so a missing grid stays a *load*-time error
        naming a sensible path rather than an import-time failure.

    Notes
    -----
    Deliberately excludes the cwd ancestor walk that :func:`data_dirs` performs.
    An import-time constant would otherwise bind to whatever directory the
    process happened to start in, making the resolved path depend on the caller's
    cwd at import — the ``parents[N]`` anchoring this replaces was at least
    cwd-independent, and that property is worth keeping.
    """
    env = _env_data_dir()
    search = ([env] if env is not None else []) + package_data_dirs()
    for directory in search:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return package_data_dirs()[0] / filename


SSP_BASE_URL = "https://halos.as.arizona.edu/suchethacooray/ssp-spectra/"

#: The default SSP identifier — one constant consumed by BOTH
#: :func:`download_ssp` and :func:`~tengri.load_ssp`, so a fresh user's
#: ``tengri.download_ssp()`` fetches exactly what a subsequent
#: ``tengri.load_ssp()`` loads. It is *bare-stellar* (no baked-in nebular
#: emission): it is present in the hosted catalog, in :data:`_KNOWN_SSPS`, and
#: is the grid the Cue/CloudyGrid nebular backends — and every
#: ``tengri.recipes.*`` config — require. The ``_wNE_*`` (with-Nebular-Emission)
#: grids are produced locally, are not shipped from the catalog, and must be
#: named explicitly (``load_ssp("prsc_miles_chabrier_wNE")``).
DEFAULT_SSP = "fsps_prsc_miles_chabrier"

# Short alias → filename, matches the live public catalog. The catalog
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

# Reverse lookup used by ``load_ssp_data`` to fetch a missing local file when
# the basename matches a known catalog entry *and* the caller passed
# ``download=True``.  It also selects which of the two FileNotFoundError
# messages that function raises: a name in here is recoverable by fetching, a
# name outside it must already be on disk.  The fetch is opt-in because the
# match is on the basename only, so an unconditional one answers a mistyped
# directory by writing the grid into it (#1553).
KNOWN_SSP_FILENAMES = frozenset(_KNOWN_SSPS.values())


def find_ssp_files() -> list[Path]:
    """Every SSP grid visible to tengri, across :func:`data_dirs`.

    Returns
    -------
    list of pathlib.Path
        Paths to SSP grids, in :func:`data_dirs` order. Empty if none are
        installed.

    Notes
    -----
    Recognizes a grid two ways: a basename in :data:`KNOWN_SSP_FILENAMES` (what
    :func:`download_ssp` writes, e.g. ``fsps_prsc_miles_chabrier.h5``), or an
    ``ssp_``-prefixed filename (locally generated grids, including the ``_wNE_``
    nebular-baked variants the catalog does not ship).

    Matching on ``*.h5`` alone would report ``dl07_templates.h5`` and other
    component libraries as SSP grids; matching only ``ssp_*.h5`` — as the
    callers previously did — cannot see a single file ``download_ssp()``
    produces. Both callers share this one answer so they cannot disagree about
    whether an install has data.
    """
    out: list[Path] = []
    for directory in data_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.h5")):
            if path.name in KNOWN_SSP_FILENAMES or path.name.startswith("ssp_"):
                out.append(path)
    return out


def list_known_ssps():
    """Return the known SSP names and their filenames.

    Returns
    -------
    _RegistryTable
        One row per SSP: ``{"name": ..., "kind": "ssp", "filename": ...}``.
        Renders as a table in a notebook.

        This used to return ``dict[str, str]``, one of only two ``list_*``
        that did (#1285). Use ``.to_dict("filename")`` for the old mapping,
        or ``.names()`` for just the identifiers. Membership tests
        (``"name" in ...``) still work, because a row's ``name`` is what
        ``.names()`` reports — but ``in`` on the table itself checks rows,
        so prefer ``in ....names()``.

    Examples
    --------
    >>> import tengri
    >>> "fsps_prsc_miles_chabrier" in tengri.list_known_ssps().names()
    True
    """
    from tengri.registry import _RegistryTable

    return _RegistryTable(
        [
            {"name": name, "kind": "ssp", "filename": filename}
            for name, filename in sorted(_KNOWN_SSPS.items())
        ]
    )


def list_available_ssps() -> list[dict]:
    """Structured view of the SSP catalog grouped by family and IMF (#307).

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
    from tengri.registry import _RegistryTable

    # Already list[dict]; wrapping only adds the table repr, so every existing
    # caller keeps working unchanged (#1285).
    return _RegistryTable(sorted(out, key=lambda r: (r["family"], r["imf"])))


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


def _resolve_ssp_filename(name: str) -> str:
    """Map a short SSP identifier or a catalog filename to a catalog filename.

    Accepts either spelling so that one function serves both the curated
    ``list_known_ssps()`` names and a filename read off the live catalog via
    ``tengri.data.list_remote_ssps()``.

    Raises
    ------
    KeyError
        If ``name`` is neither a known identifier nor a ``.h5`` filename. A bare
        unknown word is a typo, not a catalog entry, so it fails loudly rather
        than becoming an HTTP 404 later.
    ValueError
        If ``name`` is a path rather than a bare filename.
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(
            f"download_ssp(name={name!r}): name must be a bare filename or a "
            f"short identifier (see tengri.list_known_ssps()), not a path — so "
            f"it cannot write outside the destination directory."
        )
    if name in _KNOWN_SSPS:
        return _KNOWN_SSPS[name]
    if name.endswith(".h5"):
        # A filename the curated table does not carry may still be live in the
        # catalog (tengri.data.list_remote_ssps reads it directly); let the HTTP
        # layer decide rather than refusing something that exists.
        return name
    raise KeyError(
        f"Unknown SSP name: {name!r}. Known names: {list_known_ssps().names()}. "
        f"Pass a bare catalog filename ending in '.h5' to fetch something the "
        f"curated table does not list."
    )


def download_ssp(
    name: str = DEFAULT_SSP,
    dest: str | os.PathLike | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download a pre-formatted SSP HDF5 file used by tengri's stellar component.

    Parameters
    ----------
    name : str, optional
        Short SSP identifier (see ``list_known_ssps()``) or a bare catalog
        filename ending in ``.h5``. Defaults to ``"fsps_prsc_miles_chabrier"``
        (FSPS PARSEC tracks + MILES library, Chabrier IMF — bare-stellar,
        Cue/CloudyGrid-compatible).
    dest : path-like, optional
        Target directory. Defaults to :func:`download_dir` — ``$TENGRI_DATA_DIR``
        if set, else ``data/`` relative to the working directory. Either way it
        is a directory :func:`data_dirs` searches, so the loaders find the file
        afterwards.
    force : bool, optional
        Re-download even if the file already exists. Default ``False``.
    progress : bool, optional
        Print download progress. Default ``True``; pass ``False`` for clean log
        output.

    Returns
    -------
    pathlib.Path
        Path to the downloaded file.

    Raises
    ------
    KeyError
        If ``name`` is neither in ``list_known_ssps()`` nor a ``.h5`` filename.
    RuntimeError
        If the HTTP download fails (network error or non-200 status).

    Examples
    --------
    >>> import tengri
    >>> tengri.download_ssp()  # default FSPS PARSEC+MILES grid → data/
    >>> tengri.download_ssp("bc03_pdva_stelib_chabrier", dest="/scratch/ssp")
    """
    # Resolve target directory
    if dest is None:
        dest = download_dir()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    filename = _resolve_ssp_filename(name)
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
        _download_file(url, partial_filepath, progress=progress)
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


# Pre-converted component templates (HDF5) live alongside the SSP catalog on
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
        dest = download_dir()
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


def require_remote_url(url: str) -> str:
    """Return ``url`` if it is an ordinary remote URL, else raise.

    ``urllib.request.urlopen`` accepts ``file://``, and on most builds
    ``ftp://`` and other handlers too. A download helper that forwards an
    unchecked string therefore doubles as a local-file reader: a config or
    catalog entry naming ``file:///etc/passwd`` gets opened and written to
    the destination path as though it had been fetched.

    Every URL tengri fetches is an https one from the public data mirror, so
    the restriction costs nothing and closes the confusion (bandit B310).

    Parameters
    ----------
    url : str
        URL about to be passed to :func:`urllib.request.urlopen`.

    Returns
    -------
    str
        The same URL, unchanged.

    Raises
    ------
    ValueError
        If the scheme is anything other than ``http`` or ``https``.
    """
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"refusing to fetch {url!r}: only http and https URLs are "
            f"downloaded, got scheme {scheme!r}. Pass a local path directly "
            f"instead of a file:// URL."
        )
    return url


def _download_file(url: str, dest: Path, chunk_size: int = 8192, progress: bool = True) -> None:
    """Download a file from a URL to a destination path with simple progress.

    Parameters
    ----------
    url : str
        URL to download from.
    dest : Path
        Destination file path (should end with .partial for atomic write).
    chunk_size : int, optional
        Size of each download chunk in bytes. Default 8192.
    progress : bool, optional
        Show a progress bar when ``tqdm`` is installed. Default ``True``;
        ``False`` downloads silently.

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

        use_progress = progress
    except ImportError:
        use_progress = False

    require_remote_url(url)

    # Perform HEAD request to get total size
    try:
        head_request = urllib.request.Request(url, method="HEAD")
        # B310: scheme restricted to http/https by require_remote_url above.
        with urllib.request.urlopen(head_request, timeout=10) as response:  # nosec B310
            total_size = int(response.headers.get("Content-Length", 0))
    except Exception:
        total_size = 0

    # Download with progress
    downloaded = 0
    # B310: scheme restricted to http/https by require_remote_url above.
    with open(dest, "wb") as f, urllib.request.urlopen(url, timeout=30) as response:  # nosec B310
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

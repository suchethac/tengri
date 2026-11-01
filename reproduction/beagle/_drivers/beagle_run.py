# SPDX-License-Identifier: BSD-3-Clause
"""Drives the BEAGLE binary: parameter files, cached Docker runs and the output parser.

BEAGLE (Chevallard & Charlot 2016; Chevallard et al. 2019) has no Python forward
model. The driver writes a parameter file and the input FITS tables, invokes the
containerized binary, and parses its native FITS output into tengri's units
(:math:`L_\\nu` in erg/s/Hz on a rest-frame Angstrom grid). Runs are cached under
``$BEAGLE_WORKDIR/cache`` outside the repository: BEAGLE output is never committed.

Workdir layout
--------------
``$BEAGLE_WORKDIR`` is the BEAGLE working directory (``params/``, ``filters/``,
``data/``, ``results/`` and ``Docker_env.list``); its parent holds ``files/templates``.
Set it with ``export BEAGLE_WORKDIR=~/beagle/BEAGLE-general``.

Units
-----
BEAGLE writes the rest-frame flux density :math:`F_\\lambda` at the luminosity
distance :math:`d_L` in ``FULL SED``. The driver forms
:math:`L_\\lambda = 4\\pi d_L^2 F_\\lambda` and converts it with
:mod:`reproduction.beagle._drivers.units`. Wavelengths at or above 2000 Angstrom
are air in BEAGLE's output and are converted once to vacuum.

IGM
---
Every section with :math:`z > 0` carries ``IGM ABSORPTION = Inoue``, BEAGLE's default
when fitting. ``FULL SED`` is the intrinsic rest-frame spectrum without IGM; the IGM
transmission is only visible through the photometry, so the IGM section measures it
through narrow top-hat bands and compares an Inoue run with a run that omits the keyword.

References
----------
.. [1] Chevallard, J. & Charlot, S. 2016, MNRAS, 462, 1415, "Modelling and
   interpreting galaxy spectral energy distributions with BEAGLE".
.. [2] Chevallard, J., et al. 2019, MNRAS, 483, 2621, "Modelling the spectral energy
   distributions of galaxies with BEAGLE: nebular emission and attenuation".
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.table import Table

from .beagle_output import (
    ParsedRun,
    load_cache,
    parse_fit_output,
    parse_mock_output,
    save_cache,
)

DEFAULT_IMAGE = "beagletool/beagle:0.29.2"
SETUP_HINT = "export BEAGLE_WORKDIR=~/beagle/BEAGLE-general"
ZSUN = 0.01524  # BEAGLE's solar metallicity (log Z/0.01524 in the input tables)
MPC_CM = 3.0856775814913673e24
AIR_VACUUM_CUTOFF_AA = 2000.0
ID_COLUMN = "ID"
DEFAULT_MIN_REL_ERR = 0.02
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

SF_NAMES = frozenset(
    {
        "sfh_type",
        "tau",
        "metallicity",
        "mass",
        "sfr",
        "max_stellar_age",
        "ssp_age",
        "young_stars_metallicity",
        "current_sfr_timescale",
        "specific_sfr",
    }
)
STELLAR_TEMPLATES = {
    "bc03": ("bc03/bc2003_hr_xmiless_chab_ssp",),
    "cb_june16": ("cb_June16/mup100/cb2016_chab_hr_xmiless_ssp",),
}
NEBULAR_TEMPLATES = {
    "ineb_june16": (
        "ineb_June16/mup100/cb2016_n2_mup100_N015_O01_deplO70_C100_June16",
        "ineb_June16/mup100/cb2016_Jan16_line_wavelengths_may2017.dat",
    ),
}
ATTENUATION_LAWS = ("CF00", "Calzetti", "CCWW13_universal", "CCWW13_full")


@dataclass(frozen=True)
class Cosmology:
    """Flat LCDM parameters written into the parameter file.

    Parameters
    ----------
    hubble_0 : float
        Hubble constant [km/s/Mpc].
    omega_matter, omega_lambda : float
        Density parameters.
    """

    hubble_0: float
    omega_matter: float
    omega_lambda: float

    def lines(self) -> tuple[str, ...]:
        """Cosmology keyword lines for a parameter file."""
        return (
            f"HUBBLE_0 = {self.hubble_0!r}",
            f"OMEGA_MATTER = {self.omega_matter!r}",
            f"OMEGA_LAMBDA = {self.omega_lambda!r}",
        )


PLANCK18_BEAGLE = Cosmology(hubble_0=67.66, omega_matter=0.30966, omega_lambda=0.69034)


# ── Workdir and availability ────────────────────────────────────────────────
def workdir() -> Path:
    """Return the BEAGLE workdir named by ``$BEAGLE_WORKDIR``.

    Raises
    ------
    ValueError
        If the variable is unset or the directory has no ``Docker_env.list``.
    """
    raw = os.environ.get("BEAGLE_WORKDIR", "")
    if not raw:
        raise ValueError(f"BEAGLE_WORKDIR is unset. Set it with: {SETUP_HINT}")
    path = Path(raw).expanduser()
    if not (path / "Docker_env.list").is_file():
        raise ValueError(f"no Docker_env.list in {path}; check BEAGLE_WORKDIR ({SETUP_HINT})")
    return path


def templates_dir() -> Path:
    """Template directory: ``$BEAGLE_TEMPLATES_DIR``, default ``<workdir>/../files/templates``."""
    raw = os.environ.get("BEAGLE_TEMPLATES_DIR")
    return Path(raw).expanduser() if raw else workdir().parent / "files" / "templates"


def docker_image() -> str:
    """Docker image tag: ``$BEAGLE_DOCKER_IMAGE`` or the pinned 0.29.2 image."""
    return os.environ.get("BEAGLE_DOCKER_IMAGE") or DEFAULT_IMAGE


def available() -> bool:
    """True when a workdir is configured and Docker is on the PATH."""
    try:
        workdir()
    except ValueError:
        return False
    return shutil.which("docker") is not None


def require_available() -> None:
    """Raise with the setup steps when :func:`available` is False."""
    if available():
        return
    raise RuntimeError(
        "BEAGLE reproduction needs a workdir and Docker.\n"
        f"  1. Set the workdir: {SETUP_HINT}\n"
        f"  2. Pull the image: docker pull --platform linux/amd64 {docker_image()}\n"
        "  3. Run the matrix: python -m reproduction.beagle._drivers.beagle_run --run-all"
    )


# ── Parameter-file building blocks ──────────────────────────────────────────
def _fmt(value: float) -> str:
    return repr(float(value))


def _tf(flag: bool) -> str:
    return "T" if flag else "F"


@dataclass(frozen=True)
class Param:
    """One ``PARAMETER`` or ``SF_PARAMETER`` line.

    Parameters
    ----------
    name : str
        BEAGLE parameter name (``redshift``, ``tau``, ``attenuation_type``, ...).
    kind : str
        The type token, e.g. ``"type:from_file"``, ``"mock:type:fixed"``,
        ``"type:fitted"``.
    value : float, optional
        Numeric value for ``value:``.
    char_value : str, optional
        Textual value for ``char_value:``.
    prior : str, optional
        Raw prior tokens, e.g. ``"prior:distribution:uniform prior:range:[0.0,2.0]"``.
    extra : str, optional
        Any further raw tokens.
    """

    name: str
    kind: str
    value: float | None = None
    char_value: str | None = None
    prior: str | None = None
    extra: str | None = None

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(f"invalid BEAGLE parameter name {self.name!r}")
        for text in (self.kind, self.char_value, self.prior, self.extra):
            if text is not None and "\t" in text:
                raise ValueError(f"parameter {self.name!r}: tabs are not allowed in a BEAGLE file")

    def render(self) -> str:
        """Return the single ``PARAMETER = ...`` (or ``SF_PARAMETER``) line."""
        head = "SF_PARAMETER" if self.name in SF_NAMES else "PARAMETER"
        tokens = [f"name:{self.name}", self.kind]
        if self.value is not None:
            tokens.append(f"value:{_fmt(self.value)}")
        if self.char_value is not None:
            tokens.append(f"char_value:{self.char_value}")
        tokens.extend(t for t in (self.prior, self.extra) if t)
        line = f"{head} = {' '.join(tokens)}"
        if "\t" in line:
            raise ValueError(f"parameter {self.name!r} rendered with a tab")
        return line


def mock_fixed(name: str, *, value: float | None = None, char_value: str | None = None) -> Param:
    """A mock-generation parameter held at one value."""
    return Param(name, "mock:type:fixed", value=value, char_value=char_value)


def from_file(name: str) -> Param:
    """A parameter read per object from the input table."""
    return Param(name, "type:from_file")


def fit_fixed(name: str, *, value: float | None = None, char_value: str | None = None) -> Param:
    """A fitting parameter held at one value."""
    return Param(name, "type:fixed", value=value, char_value=char_value)


def fit_fitted(name: str, low: float, high: float, *, order: int = 0) -> Param:
    """A fitted parameter with a uniform prior on ``[low, high]``."""
    prior = f"order_priority:{order} prior:distribution:uniform prior:range:[{low!r},{high!r}]"
    return Param(name, "type:fitted", prior=prior)


@dataclass(frozen=True, eq=False)
class FilterSpec:
    """A photometric band: its throughput curve and, for fitting, its catalog columns.

    Parameters
    ----------
    name : str
        Filter label; becomes the ``<name>_APP`` / ``<name>_ABS`` output columns.
    wl_aa, transmission : ndarray
        Throughput curve in Angstrom (ascending) and dimensionless transmission.
    flux_col, err_col : str, optional
        Catalog columns for fluxes [microJy] and their errors (fitting only).
    min_rel_err : float
        Relative error added in quadrature to the catalog error.
    """

    name: str
    wl_aa: np.ndarray
    transmission: np.ndarray
    flux_col: str | None = None
    err_col: str | None = None
    min_rel_err: float = DEFAULT_MIN_REL_ERR


def _filter_digest(filters: Sequence[FilterSpec]) -> list[bytes]:
    return [
        f.name.encode() + np.asarray(f.wl_aa, float).tobytes() + np.asarray(f.transmission, float).tobytes()
        for f in filters
    ]


def _filter_config(run_name: str, filters: Sequence[FilterSpec], *, fitting: bool) -> str:
    """Text of the ``FILTERS CONFIGURATION`` file (per-filter ``fileName`` curves)."""
    lines = ["units:microJy"]
    if fitting:
        lines.append(f"object_ID:colName:{ID_COLUMN}")
    for filt in filters:
        tokens = [f"name:{filt.name}", f"fileName:$BEAGLE_FILTERS/{run_name}_{filt.name}.dat"]
        if fitting and filt.flux_col and filt.err_col:
            tokens += [
                f"flux:colName:{filt.flux_col}",
                f"fluxErr:colName:{filt.err_col}",
                f"min_rel_err:{_fmt(filt.min_rel_err)}",
            ]
        lines.append(" ".join(tokens))
    return "\n".join(lines) + "\n"


def _output_lines(*, photometry: bool, spectrum: bool, sfh: bool, nebular: bool) -> list[str]:
    lines = [
        f"PRINT PHOTOMETRY = {_tf(photometry)}",
        f"PRINT SPECTRUM = {_tf(spectrum)}",
        "PRINT SPECTRAL INDICES = F",
        f"PRINT SF AND ChE HISTORIES = {_tf(sfh)}",
    ]
    if nebular:
        lines += ["PRINT CONTINUUM = T", "PRINT LINE FLUXES = T"]
    return lines


def _template_lines(templates: str, nebular: bool) -> list[str]:
    if nebular:
        if templates not in NEBULAR_TEMPLATES:
            raise ValueError(f"unknown nebular template set {templates!r}")
        listing, lines_cfg = NEBULAR_TEMPLATES[templates]
        return [
            f"TEMPLATES NEBULAR = $BEAGLE_TEMPLATES/{listing}",
            f"EMISSION LINES CONFIGURATION = $BEAGLE_TEMPLATES/{lines_cfg}",
        ]
    if templates not in STELLAR_TEMPLATES:
        raise ValueError(f"unknown stellar template set {templates!r}")
    (listing,) = STELLAR_TEMPLATES[templates]
    return [f"TEMPLATES = $BEAGLE_TEMPLATES/{listing}"]


def _template_files(templates: str, nebular: bool) -> list[str]:
    listing = NEBULAR_TEMPLATES if nebular else STELLAR_TEMPLATES
    return [Path(p).name for p in listing[templates]]


def _table_digest(table: Table) -> bytes:
    h = hashlib.sha256()
    for col in table.colnames:
        arr = np.asarray(table[col])
        h.update(f"{col}:{arr.dtype.str}:{arr.shape}".encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.digest()


def _run_key(body: Sequence[str], files: Sequence[str], table: Table, filters: Sequence[FilterSpec]) -> str:
    h = hashlib.sha256()
    for part in ("\n".join(body), "\n".join(sorted(files)), docker_image()):
        h.update(part.encode() + b"\0")
    h.update(_table_digest(table))
    for chunk in _filter_digest(filters):
        h.update(chunk)
    return h.hexdigest()[:12]


# ── Run specifications ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class MockRun:
    """A BEAGLE ``--mock`` run: one parameter file and one row per simulated galaxy.

    Parameters
    ----------
    section : str
        Matrix section id (``s1_ssp_bc03``); the cache name is ``section_key``.
    rows : tuple of dict
        One mapping per object: the ``from_file`` parameter values (log units as BEAGLE
        reads them). Every row has the same keys.
    templates : str
        Key of :data:`STELLAR_TEMPLATES` or :data:`NEBULAR_TEMPLATES`.
    nebular : bool
        Use the ``TEMPLATES NEBULAR`` grid and print continuum and line fluxes.
    dust : str, optional
        Attenuation law, one of :data:`ATTENUATION_LAWS`; ``tauV_eff`` and ``mu`` go in ``params``.
    params : tuple of Param
        Fixed or random mock parameters (not ``from_file``).
    filters : tuple of FilterSpec, optional
        Bands for mock photometry (``PRINT PHOTOMETRY``).
    igm : str, optional
        IGM model written as ``IGM ABSORPTION``. Defaults to ``"Inoue"``, BEAGLE's
        default when fitting; ``None`` omits the keyword, which is the no-IGM control
        (the manual documents no ``"none"`` value; the omission is to be confirmed
        by a run). IGM acts on the photometry only; ``FULL SED`` carries none.
    print_sfh : bool
        Request the ``STAR FORMATION HISTORIES`` printout. BEAGLE 0.29.2 cannot print
        the star formation history of an ``ssp`` mock (it segfaults in
        ``star_formation::print_sf_che_history``), so the request is dropped for
        ssp sections whatever this flag says.
    """

    section: str
    rows: tuple[Mapping[str, float], ...]
    templates: str
    nebular: bool = False
    dust: str | None = None
    params: tuple[Param, ...] = ()
    print_sed: bool = True
    print_sfh: bool = True
    filters: tuple[FilterSpec, ...] | None = None
    rebin: int | None = 10
    seed: int = 12345
    cosmology: Cosmology = PLANCK18_BEAGLE
    igm: str | None = "Inoue"
    mode: Literal["mock"] = field(default="mock", init=False)

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError(f"{self.section}: a mock run needs at least one row")
        keys = tuple(self.rows[0].keys())
        if any(tuple(r.keys()) != keys for r in self.rows):
            raise ValueError(f"{self.section}: every row must carry the same keys")
        clash = {p.name for p in self.params} & set(keys)
        if clash:
            raise ValueError(f"{self.section}: {sorted(clash)} are both rows and params")
        if self.dust is not None and self.dust not in ATTENUATION_LAWS:
            raise ValueError(f"{self.section}: unknown attenuation law {self.dust!r}")

    def _is_ssp(self) -> bool:
        return any(p.name == "sfh_type" and p.char_value == "ssp" for p in self.params)

    def _params(self) -> list[Param]:
        files = [from_file(k) for k in self.rows[0].keys()]
        dust = [mock_fixed("attenuation_type", char_value=self.dust)] if self.dust else []
        return [*files, *dust, *self.params]

    def input_table(self) -> Table:
        """The per-object input table, one row per case, columns named as the ``from_file`` parameters."""
        keys = list(self.rows[0].keys())
        return Table({k: np.array([float(r[k]) for r in self.rows]) for k in keys})

    def _body_lines(self) -> list[str]:
        params = self._params()
        lines = ["[main]", "VERBOSE = 2", f"SEED = {self.seed}", *self.cosmology.lines()]
        lines += _template_lines(self.templates, self.nebular)
        if self.rebin is not None:
            lines.append(f"REBIN TEMPLATES = {self.rebin}")
        if self.igm:
            lines.append(f"IGM ABSORPTION = {self.igm}")
        lines += [p.render() for p in params if p.name in SF_NAMES]
        lines += [p.render() for p in params if p.name not in SF_NAMES]
        lines += _output_lines(
            photometry=self.filters is not None,
            spectrum=self.print_sed,
            sfh=self.print_sfh and not self._is_ssp(),
            nebular=self.nebular,
        )
        lines += [f"N_OBJECTS = {len(self.rows)}", "PDF SAMPLER FILE = $BEAGLE_PARAM_DIR/MCMC_new.param"]
        return lines

    @property
    def key(self) -> str:
        """Content key: parameter body, template files, image, input table and filter curves."""
        return _run_key(
            self._body_lines(),
            _template_files(self.templates, self.nebular),
            self.input_table(),
            self.filters or (),
        )

    @property
    def name(self) -> str:
        """Cache and output name: ``<section>_<key>``."""
        return f"{self.section}_{self.key}"

    def param_text(self) -> str:
        """The full parameter file text."""
        name = self.name
        lines = self._body_lines()
        lines += [
            f"RESULTS DIRECTORY = $BEAGLE_RESULTS/{name}",
            f"MOCK CATALOGUE NAME = {name}",
            f"MOCK INPUT PARAMETERS = fileName:$BEAGLE_DATA/{name}_inputs.fits",
        ]
        if self.filters:
            lines.append(f"FILTERS CONFIGURATION = $BEAGLE_FILTERS/{name}_config.dat")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, eq=False)
class FitRun:
    """A BEAGLE ``--fit`` run over a photometric catalog.

    Parameters
    ----------
    section : str
        Section id; the cache name is ``section_key``.
    galaxy_ids : tuple of str
        Object IDs to fit; each must appear in ``catalog_table[ID_COLUMN]``.
    catalog_table : Table
        Photometric catalog: an ``ID`` column, flux and error columns, and an
        optional redshift column named by ``redshift_column``.
    filters : tuple of FilterSpec
        Bands with their catalog columns and throughput curves.
    params : tuple of Param
        Fitted, fixed and dependent parameters (``type:fitted``, ``type:fixed``, ``type:dependent``).
    redshift_column : str, optional
        Catalog column read as a Dirac redshift prior (``PRIORS CATALOGUE``).
    """

    section: str
    galaxy_ids: tuple[str, ...]
    catalog_table: Table
    filters: tuple[FilterSpec, ...]
    params: tuple[Param, ...]
    templates: str = "bc03"
    nebular: bool = False
    redshift_column: str | None = None
    print_spectrum: bool = False
    rebin: int | None = 10
    seed: int = 12345
    cosmology: Cosmology = PLANCK18_BEAGLE
    igm: str | None = "Inoue"
    mode: Literal["fit"] = field(default="fit", init=False)

    def __post_init__(self) -> None:
        if ID_COLUMN not in self.catalog_table.colnames:
            raise ValueError(f"{self.section}: catalog has no {ID_COLUMN!r} column")
        present = {str(v) for v in self.catalog_table[ID_COLUMN]}
        missing = [g for g in self.galaxy_ids if g not in present]
        if missing:
            raise ValueError(f"{self.section}: galaxy ids {missing} not in the catalog")
        if self.redshift_column and self.redshift_column not in self.catalog_table.colnames:
            raise ValueError(f"{self.section}: no redshift column {self.redshift_column!r}")
        if self.redshift_column and any(p.name == "redshift" for p in self.params):
            raise ValueError(f"{self.section}: redshift is set by redshift_column, not by params")

    def input_table(self) -> Table:
        """The photometric catalog written to ``$BEAGLE_DATA``."""
        return self.catalog_table

    def _body_lines(self) -> list[str]:
        lines = ["[main]", "VERBOSE = 2", f"SEED = {self.seed}", *self.cosmology.lines()]
        lines += _template_lines(self.templates, self.nebular)
        if self.rebin is not None:
            lines.append(f"REBIN TEMPLATES = {self.rebin}")
        if self.igm:
            lines.append(f"IGM ABSORPTION = {self.igm}")
        lines += [p.render() for p in self.params if p.name in SF_NAMES]
        lines += [p.render() for p in self.params if p.name not in SF_NAMES]
        if self.redshift_column:
            dirac = f"prior:distribution:dirac prior:colName:{self.redshift_column}"
            lines.append(Param("redshift", "type:fitted", prior=dirac).render())
        lines += _output_lines(photometry=True, spectrum=self.print_spectrum, sfh=False, nebular=self.nebular)
        lines.append("PDF SAMPLER FILE = $BEAGLE_PARAM_DIR/MCMC_new.param")
        return lines

    @property
    def key(self) -> str:
        """Content key over the parameter body, templates, image, catalog, curves and ids."""
        body = [*self._body_lines(), "GALAXY IDS " + " ".join(self.galaxy_ids)]
        return _run_key(
            body,
            _template_files(self.templates, self.nebular),
            self.catalog_table,
            self.filters,
        )

    @property
    def name(self) -> str:
        """Cache and output name: ``<section>_<key>``."""
        return f"{self.section}_{self.key}"

    def param_text(self) -> str:
        """The full ``--fit`` parameter file text."""
        name = self.name
        lines = self._body_lines()
        lines += [
            f"RESULTS DIRECTORY = $BEAGLE_RESULTS/{name}",
            f"PHOTOMETRIC CATALOGUE = $BEAGLE_DATA/{name}_inputs.fits",
            f"FILTERS CONFIGURATION = $BEAGLE_FILTERS/{name}_config.dat",
        ]
        if self.redshift_column:
            lines.append(f"PRIORS CATALOGUE = $BEAGLE_DATA/{name}_inputs.fits")
        return "\n".join(lines) + "\n"


Run = MockRun | FitRun


# ── Materializing inputs and running the binary ─────────────────────────────
def materialize(run: Run, root: Path) -> None:
    """Write the parameter file, input table and filter curves under ``root``.

    ``root`` mirrors the BEAGLE workdir (``params/``, ``data/``, ``filters/``), so the
    same call serves a live run and a parameter-only render.
    """
    for sub in ("params", "data", "filters"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "params" / f"{run.name}.param").write_text(run.param_text())
    run.input_table().write(root / "data" / f"{run.name}_inputs.fits", format="fits", overwrite=True)
    if run.filters:
        fitting = isinstance(run, FitRun)
        (root / "filters" / f"{run.name}_config.dat").write_text(
            _filter_config(run.name, run.filters, fitting=fitting)
        )
        for filt in run.filters:
            body = np.column_stack([np.asarray(filt.wl_aa, float), np.asarray(filt.transmission, float)])
            np.savetxt(
                root / "filters" / f"{run.name}_{filt.name}.dat",
                body,
                header="wavelength_angstrom transmission",
            )


def _docker_cmd(root: Path, templates: Path, image: str, run: Run) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-v",
        f"{root}:/BEAGLE_virtual_workdir",
        "-v",
        f"{templates}:/BEAGLE_virtual_workdir/templates",
        "--env-file",
        str(root / "Docker_env.list"),
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        image,
        f"--{run.mode}",
        "--parameter-file",
        f"/BEAGLE_virtual_workdir/params/{run.name}.param",
    ]


def _binary_version(image: str) -> str:
    """The binary's ``--version`` string, or the reason it could not be read."""
    proc = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64", image, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return f"unavailable (exit {proc.returncode})"
    return proc.stdout.strip() or proc.stderr.strip()


def _provenance(run: Run, image: str, version: str) -> dict[str, object]:
    files = _template_files(run.templates, run.nebular)
    param_sha = hashlib.sha256(run.param_text().encode()).hexdigest()
    return {
        "image": image,
        "version": version,
        "param_sha256": param_sha,
        "templates": files,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "host_arch": platform.machine(),
        "section": run.section,
        "key": run.key,
    }


def _parse(run: Run, results: Path, provenance: dict[str, object]) -> ParsedRun:
    if isinstance(run, FitRun):
        return parse_fit_output(results, run.galaxy_ids, provenance=provenance)
    return parse_mock_output(results / f"{run.name}.fits", provenance=provenance)


def _execute(root: Path, run: Run) -> ParsedRun:
    templates = templates_dir()
    if not templates.is_dir():
        raise FileNotFoundError(f"BEAGLE templates not found at {templates}; set BEAGLE_TEMPLATES_DIR")
    if shutil.which("docker") is None:
        raise RuntimeError("docker is not on PATH; BEAGLE runs need Docker (see require_available())")
    image = docker_image()
    materialize(run, root)
    results = root / "results" / run.name
    results.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        _docker_cmd(root, templates, image, run), capture_output=True, text=True, check=False
    )
    log = "\n".join(line for line in proc.stderr.splitlines() if not line.startswith("profiling:"))
    (results / "docker_stderr.log").write_text(log + "\n")
    if proc.returncode != 0:
        raise RuntimeError(
            f"BEAGLE exited with {proc.returncode} for {run.name}; see {results / 'docker_stderr.log'}"
        )
    parsed = _parse(run, results, _provenance(run, image, _binary_version(image)))
    save_cache(root, run.name, parsed)
    return parsed


def run(run: Run, *, allow_docker: bool = False) -> ParsedRun:
    """Return the parsed output of ``run``: from the cache, or by running the binary.

    Parameters
    ----------
    run : MockRun or FitRun
        The run specification.
    allow_docker : bool
        Run the binary on a cache miss. When False a miss raises, so a notebook never
        starts a long fit by accident.

    Returns
    -------
    ParsedRun

    Raises
    ------
    RuntimeError
        On a cache miss with ``allow_docker=False``, or when the binary fails.
    """
    root = workdir()
    if (root / "cache" / f"{run.name}.npz").is_file():
        return load_cache(root, run.name)
    if not allow_docker:
        raise RuntimeError(
            f"no cached BEAGLE run for section {run.section} ({run.name}). Run it with:\n"
            f"  python -m reproduction.beagle._drivers.beagle_run --run {run.section}"
        )
    return _execute(root, run)


# ── Command line ────────────────────────────────────────────────────────────
def _status(runs: Sequence[Run]) -> int:
    root = workdir()
    for item in runs:
        cached = (root / "cache" / f"{item.name}.npz").is_file()
        print(f"{item.name:40s} {'cached' if cached else 'missing'}")
    return 0


def _render(runs: Sequence[Run], out: Path) -> int:
    for item in runs:
        materialize(item, out)
        print(f"{item.name}: {len(item.input_table())} objects -> {out / 'params' / (item.name + '.param')}")
    return 0


def _fit_capstone() -> int:
    print("fit capstone: no Paper I galaxy ids are committed yet; nothing was run.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point (see the module docstring for the workflow)."""
    from .beagle_matrix import RUN_MATRIX, by_section

    parser = argparse.ArgumentParser(prog="beagle_run", description=__doc__.splitlines()[0])
    parser.add_argument("--render-params", action="store_true", help="write parameter files, no Docker")
    parser.add_argument("--out", type=Path, default=None, help="directory for --render-params")
    parser.add_argument("--run", metavar="SECTION", help="run one matrix section through Docker")
    parser.add_argument("--run-all", action="store_true", help="run every matrix section")
    parser.add_argument("--fit-capstone", action="store_true", help="placeholder for the fit capstone")
    parser.add_argument("--status", action="store_true", help="show which runs are cached")
    args = parser.parse_args(argv)
    try:
        return _dispatch(args, parser, RUN_MATRIX, by_section)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"beagle_run: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser, matrix, by_section) -> int:
    if args.render_params:
        out = args.out if args.out is not None else workdir()
        return _render(matrix, out)
    if args.fit_capstone:
        return _fit_capstone()
    if args.status:
        return _status(matrix)
    if args.run or args.run_all:
        require_available()
        targets = list(matrix) if args.run_all else by_section(args.run)
        for item in targets:
            parsed = run(item, allow_docker=True)
            shape = parsed.sed_lnu.shape if parsed.sed_lnu is not None else "no SED"
            print(f"{item.name}: {shape}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

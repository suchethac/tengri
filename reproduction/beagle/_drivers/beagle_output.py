# SPDX-License-Identifier: BSD-3-Clause
"""Parses BEAGLE's native FITS output into tengri's units, and caches parsed runs.

BEAGLE writes one FITS file per mock run (``<RESULTS DIRECTORY>/<MOCK CATALOGUE
NAME>.fits``) and one ``<ID>_BEAGLE.fits.gz`` per fitted object. The parser reads
whatever extensions are present and records the names it found in
``provenance["extensions"]``; a missing optional extension leaves its field
``None`` rather than failing.

Wavelengths: BEAGLE uses vacuum below 2000 Angstrom and air at or above it, for the
continuum and the emission lines alike, so every such wavelength is converted once
with :func:`tengri.utils.air_vacuum.air_to_vac`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from tengri.utils.air_vacuum import air_to_vac

from . import units

MPC_CM = 3.0856775814913673e24
AIR_VACUUM_CUTOFF_AA = 2000.0
_LINE_RE = re.compile(r"^(?P<name>.+)_(?P<wl>\d+(?:\.\d+)?)_lum_(?P<kind>obs|em)$")
_MAGNITUDES = {"APPARENT MAGNITUDES": "_APP", "ABSOLUTE MAGNITUDES": "_ABS"}
_ARRAY_FIELDS = ("wl_rest_aa", "sed_lnu", "sed_flam_at_dl", "continuum_sed")
_DICT_FIELDS = (
    "galaxy_properties",
    "star_formation",
    "sfh",
    "magnitudes_app",
    "magnitudes_abs",
    "line_lums_lsun",
    "line_fluxes",
)


@dataclass(frozen=True)
class ParsedRun:
    """One BEAGLE run in tengri's units.

    Attributes
    ----------
    wl_rest_aa : ndarray or None
        Rest-frame vacuum wavelength grid [Angstrom], shape ``(n_wl,)``.
    sed_lnu : ndarray or None
        :math:`L_\\nu` [erg/s/Hz], shape ``(n_obj, n_wl)``, from ``FULL SED``.
    sed_flam_at_dl : ndarray or None
        BEAGLE's raw ``FULL SED`` flux density :math:`F_\\lambda` at :math:`d_L`.
    galaxy_properties, star_formation, sfh : dict or None
        Per-object columns of the matching extensions (``sfh`` is ``STAR FORMATION
        HISTORIES``, one array of bins per object).
    magnitudes_app, magnitudes_abs : dict or None
        AB magnitudes keyed by filter name.
    line_lums_lsun : dict or None
        ``HII EMISSION`` line luminosities in :math:`L_\\odot`, keyed
        ``<line>_<vacuum wl>_lum_<obs|em>``.
    line_fluxes : dict or None
        ``HII EMISSION`` ``*_flux`` columns, as BEAGLE writes them.
    continuum_sed : ndarray or None
        Continuum-only :math:`L_\\nu` on ``wl_rest_aa``, when ``CONTINUUM SED`` is on the same grid.
    posterior : dict or None
        Fits only: ``{object_id: {extension: {column: array}}}``; images are under ``"image"``.
    provenance : dict
        Image tag, version string, parameter hash, templates, extensions found, timestamp.
    """

    wl_rest_aa: np.ndarray | None = None
    sed_lnu: np.ndarray | None = None
    sed_flam_at_dl: np.ndarray | None = None
    galaxy_properties: Mapping[str, np.ndarray] | None = None
    star_formation: Mapping[str, np.ndarray] | None = None
    sfh: Mapping[str, np.ndarray] | None = None
    magnitudes_app: Mapping[str, np.ndarray] | None = None
    magnitudes_abs: Mapping[str, np.ndarray] | None = None
    line_lums_lsun: Mapping[str, np.ndarray] | None = None
    line_fluxes: Mapping[str, np.ndarray] | None = None
    continuum_sed: np.ndarray | None = None
    posterior: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]] | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)


def _hdu(hdul: fits.HDUList, name: str):
    try:
        return hdul[name]
    except KeyError:
        return None


def _columns(hdu) -> dict[str, np.ndarray]:
    """Every column of a table HDU; numeric columns as float64."""
    out: dict[str, np.ndarray] = {}
    for col in hdu.columns.names:
        arr = np.asarray(hdu.data[col])
        out[col] = arr.astype(np.float64) if arr.dtype.kind in "fiu" else arr
    return out


def _image(hdu) -> np.ndarray:
    return np.asarray(hdu.data, dtype=np.float64)


def _air_to_vacuum(wl: np.ndarray) -> np.ndarray:
    """Convert BEAGLE's air wavelengths (>= 2000 Angstrom) to vacuum; leave the rest."""
    wl = np.asarray(wl, dtype=np.float64)
    return np.where(wl >= AIR_VACUUM_CUTOFF_AA, np.asarray(air_to_vac(wl), dtype=np.float64), wl)


def _to_lnu(wl: np.ndarray, flam: np.ndarray, d_l_mpc: np.ndarray) -> np.ndarray:
    """:math:`L_\\nu` [erg/s/Hz] from :math:`F_\\lambda` at :math:`d_L`: :math:`L_\\lambda = 4\\pi d_L^2 F_\\lambda`."""
    d_l_cm = np.asarray(d_l_mpc, dtype=np.float64)[:, None] * MPC_CM
    l_lambda = 4.0 * np.pi * d_l_cm**2 * np.asarray(flam, dtype=np.float64)
    _, l_nu = units.ergs_per_aa_to_erg_per_hz(wl, l_lambda)
    return l_nu


def _magnitudes(cols: dict[str, np.ndarray] | None, suffix: str) -> dict[str, np.ndarray] | None:
    if cols is None:
        return None
    return {c[: -len(suffix)]: a for c, a in cols.items() if c.endswith(suffix)}


def _line_luminosities(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``<line>_<wl>_lum_<kind>`` columns, with air wavelengths converted to vacuum."""
    out: dict[str, np.ndarray] = {}
    for col, arr in cols.items():
        match = _LINE_RE.match(col)
        if not match:
            continue
        wl = float(match["wl"])
        if wl >= AIR_VACUUM_CUTOFF_AA:
            wl = float(air_to_vac(wl))
        out[f"{match['name']}_{wl:.2f}_lum_{match['kind']}"] = arr
    return out


def _sed_block(hdul: fits.HDUList) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Wavelength (vacuum), raw F_lambda, L_nu and the galaxy-properties columns."""
    gp_hdu, wl_hdu, sed_hdu = (_hdu(hdul, n) for n in ("GALAXY PROPERTIES", "FULL SED WL", "FULL SED"))
    if gp_hdu is None or wl_hdu is None or sed_hdu is None:
        raise ValueError("BEAGLE output needs GALAXY PROPERTIES, FULL SED WL and FULL SED")
    gp = _columns(gp_hdu)
    wl = _air_to_vacuum(np.asarray(wl_hdu.data["wl"][0], dtype=np.float64))
    flam = _image(sed_hdu)
    return wl, flam, _to_lnu(wl, flam, gp["luminosity_distance"]), gp


def parse_mock_output(path: Path | str, *, provenance: Mapping[str, object] | None = None) -> ParsedRun:
    """Parse one BEAGLE ``--mock`` FITS file into a :class:`ParsedRun`.

    Parameters
    ----------
    path : path-like
        The ``<MOCK CATALOGUE NAME>.fits`` file.
    provenance : mapping, optional
        Extra provenance to merge in (image, version, hash, ...).

    Returns
    -------
    ParsedRun
        Spectra in :math:`L_\\nu`, per-object properties and magnitudes; fields absent
        from the file are ``None``.
    """
    with fits.open(path) as hdul:
        present = [h.name for h in hdul[1:]]
        wl, flam, lnu, gp = _sed_block(hdul)
        sf = _maybe(hdul, "STAR FORMATION")
        sfh = _maybe(hdul, "STAR FORMATION HISTORIES")
        app = _maybe(hdul, "APPARENT MAGNITUDES")
        abs_ = _maybe(hdul, "ABSOLUTE MAGNITUDES")
        lines = _maybe(hdul, "HII EMISSION")
        cont = _continuum(hdul, flam.shape, wl, gp)
    prov = {**dict(provenance or {}), "extensions": present, "path": str(path)}
    return ParsedRun(
        wl_rest_aa=wl,
        sed_lnu=lnu,
        sed_flam_at_dl=flam,
        galaxy_properties=gp,
        star_formation=sf,
        sfh=sfh,
        magnitudes_app=_magnitudes(app, _MAGNITUDES["APPARENT MAGNITUDES"]),
        magnitudes_abs=_magnitudes(abs_, _MAGNITUDES["ABSOLUTE MAGNITUDES"]),
        line_lums_lsun=None if lines is None else _line_luminosities(lines),
        line_fluxes=None if lines is None else {c: a for c, a in lines.items() if c.endswith("_flux")},
        continuum_sed=cont,
        provenance=prov,
    )


def _maybe(hdul: fits.HDUList, name: str) -> dict[str, np.ndarray] | None:
    hdu = _hdu(hdul, name)
    return None if hdu is None else _columns(hdu)


def _continuum(
    hdul: fits.HDUList, shape: tuple[int, ...], wl: np.ndarray, gp: Mapping[str, np.ndarray]
) -> np.ndarray | None:
    """Continuum L_nu when ``CONTINUUM SED`` shares the FULL SED grid; ``None`` otherwise."""
    hdu = _hdu(hdul, "CONTINUUM SED")
    if hdu is None:
        return None
    flam = _image(hdu)
    if flam.shape != shape:
        return None
    return _to_lnu(wl, flam, gp["luminosity_distance"])


def parse_fit_output(
    results_dir: Path | str,
    galaxy_ids: tuple[str, ...],
    *,
    provenance: Mapping[str, object] | None = None,
) -> ParsedRun:
    """Parse the per-object ``<ID>_BEAGLE.fits.gz`` posteriors of a ``--fit`` run.

    Every table extension becomes a dict of columns and every image extension a dict
    holding ``"image"``, so ``POSTERIOR PDF`` and any ``MARGINAL`` extensions the
    keywords requested are all present without a per-extension reader.
    """
    posterior: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for gid in galaxy_ids:
        path = Path(results_dir) / f"{gid}_BEAGLE.fits.gz"
        if not path.is_file():
            raise FileNotFoundError(f"missing BEAGLE posterior for object {gid}: {path}")
        with fits.open(path) as hdul:
            per: dict[str, dict[str, np.ndarray]] = {}
            for hdu in hdul[1:]:
                if hdu.columns is not None:
                    per[hdu.name] = _columns(hdu)
                elif hdu.data is not None:
                    per[hdu.name] = {"image": _image(hdu)}
        posterior[gid] = per
    prov = {**dict(provenance or {}), "results_dir": str(results_dir)}
    return ParsedRun(posterior=posterior, provenance=prov)


# ── Cache: npz arrays plus a json sidecar, flattened with "__" separators ───
def _flatten(parsed: ParsedRun) -> tuple[dict[str, np.ndarray], list[str]]:
    arrays: dict[str, np.ndarray] = {}
    present: list[str] = []
    for name in _ARRAY_FIELDS:
        value = getattr(parsed, name)
        if value is not None:
            arrays[name] = np.asarray(value)
            present.append(name)
    for name in _DICT_FIELDS:
        value = getattr(parsed, name)
        if value is not None:
            present.append(name)
            for key, arr in value.items():
                arrays[f"{name}__{key}"] = np.asarray(arr)
    if parsed.posterior is not None:
        present.append("posterior")
        for gid, exts in parsed.posterior.items():
            for ext, cols in exts.items():
                for col, arr in cols.items():
                    arrays[f"posterior__{gid}__{ext}__{col}"] = np.asarray(arr)
    return arrays, present


def save_cache(root: Path, name: str, parsed: ParsedRun) -> None:
    """Write ``<root>/cache/<name>.npz`` and ``.json``; the json holds the provenance."""
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    arrays, present = _flatten(parsed)
    np.savez_compressed(cache / f"{name}.npz", **arrays)
    meta = {"present": present, "provenance": dict(parsed.provenance)}
    (cache / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))


def load_cache(root: Path, name: str) -> ParsedRun:
    """Read a run written by :func:`save_cache`."""
    cache = root / "cache"
    meta = json.loads((cache / f"{name}.json").read_text())
    present = set(meta["present"])
    with np.load(cache / f"{name}.npz") as z:
        data = {k: z[k] for k in z.files}
    kwargs: dict[str, object] = {"provenance": meta["provenance"]}
    for name_ in _ARRAY_FIELDS:
        kwargs[name_] = data.get(name_) if name_ in present else None
    for name_ in _DICT_FIELDS:
        prefix = f"{name_}__"
        kwargs[name_] = (
            {k[len(prefix) :]: v for k, v in data.items() if k.startswith(prefix)} if name_ in present else None
        )
    if "posterior" in present:
        post: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for k, v in data.items():
            if k.startswith("posterior__"):
                _, gid, ext, col = k.split("__", 3)
                post.setdefault(gid, {}).setdefault(ext, {})[col] = v
        kwargs["posterior"] = post
    else:
        kwargs["posterior"] = None
    return ParsedRun(**kwargs)

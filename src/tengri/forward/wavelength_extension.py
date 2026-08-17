# SPDX-License-Identifier: BSD-3-Clause
"""Native wavelength-grid declarations for SED components.

This module implements the **union-of-component-grids** design called for in
issue #463: every component that ships a tabulated SED on a fixed native
wavelength grid (dust emission templates, AGN torus libraries, AGN disc grids,
X-ray templates) advertises that grid here so the orchestrator can union it
with the SSP grid at ``SEDModel.build`` time. The resulting master grid is
static (computed once, JIT-stable) and captures the full wavelength coverage
of every attached template without per-component wiring.

The semantics mirror CIGALE's ``sed.add_contribution(name, wavelength, ...)``
behavior: each module's native grid extends the master grid; analytic
components (modified blackbody, Casey 2012 single-T MBB, IGM transmission)
contribute nothing because they evaluate exactly on whatever grid they're
handed.

Schemas vary across template files — wavelengths may live under
``wavelength_aa``, ``wavelength`` (assumed Å for SED templates), or
``wavelength_um`` (×1e4 to convert), and may be nested inside an HDF5 group
matching the model name. The lookup helpers normalize all of this to
sorted ``ndarray`` values in Å.

Adding a new template-backed component: append an entry to the relevant
catalog dict (``_DUST_EMISSION_TEMPLATES`` etc.); no orchestrator change is
required.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterable

import numpy as np

from tengri._data_setup import find_data_str

logger = logging.getLogger(__name__)


# ── Schema declarations ───────────────────────────────────────────
#
# Each entry is (filename, dataset_path, unit_to_aa). The dataset_path may be a
# nested HDF5 path like ``"silva04/wavelength"`` for grouped layouts. The
# unit_to_aa factor converts the stored values into Angstrom (1.0 for ``_aa``
# and bare ``wavelength`` SED templates, 1e4 for ``wavelength_um``).

# Dust emission templates ----------------------------------------------------
# Each list of candidate filenames is tried in order; the first that exists
# wins (matches the v2-preferred behavior in dust/emission.py).
_DUST_EMISSION_TEMPLATES: dict[str, tuple[tuple[str, str, float], ...]] = {
    "dale2014": (
        ("dale2014_templates_v2.h5", "wavelength_aa", 1.0),
        ("dale2014_templates.h5", "wavelength_aa", 1.0),
    ),
    "draine_li2007": (
        ("dl07_templates_v2.h5", "wavelength", 1.0),
        ("dl07_templates.h5", "wavelength", 1.0),
    ),
    # Alias for the deprecated registry name.
    "dl07_tabulated": (
        ("dl07_templates_v2.h5", "wavelength", 1.0),
        ("dl07_templates.h5", "wavelength", 1.0),
    ),
    "draine_li2014": (
        ("dl14_templates_v2.h5", "wavelength", 1.0),
        ("dl14_templates.h5", "wavelength", 1.0),
    ),
    "astrodust": (("astrodust_templates.h5", "wavelength_um", 1e4),),
    "bosa": (("bosa_templates.h5", "wavelength_aa", 1.0),),
    "themis": (("themis_templates.h5", "wavelength_aa", 1.0),),
}

# Analytic dust-emission models — no template file, but their emission still
# needs FIR/submm support on the master grid: without it the SED truncates at
# the SSP edge (160 µm for BC03) and submm photometry is silently zero while
# energy balance re-normalizes on the truncated grid (#1005). Same failure
# mode and fix as the Cue continuum grid below. 1 µm – 1 cm covers the PAH
# complexes through the Rayleigh–Jeans tail.
_ANALYTIC_DUST_EMISSION = frozenset(
    {
        "modified_blackbody",
        "casey2012",
        "pah_drude",
        "schreiber2016",
        "energy_balance_split",
    }
)
_ANALYTIC_DUST_WAVE_AA = np.geomspace(1.0e4, 1.0e8, 512)
# Bookkeeping pseudo-model with no emission of its own — stays grid-less.
_GRIDLESS_DUST_EMISSION = frozenset({"energy_balance_split"})

# AGN torus templates --------------------------------------------------------
_AGN_TORUS_TEMPLATES: dict[str, tuple[tuple[str, str, float], ...]] = {
    "skirtor": (
        ("skirtor_templates_v3.h5", "wavelength", 1.0),
        ("skirtor_templates_v2.h5", "wavelength", 1.0),
    ),
    "silva04": (("silva04_torus_grid.h5", "silva04/wavelength", 1.0),),
    "cat3d_wind": (("cat3d_wind_torus_grid.h5", "cat3d_wind/wavelength", 1.0),),
}

# AGN disc templates ---------------------------------------------------------
_AGN_DISC_TEMPLATES: dict[str, tuple[tuple[str, str, float], ...]] = {
    "relagn": (("relagn_disc_grid.h5", "wavelength_aa", 1.0),),
}

# Standalone AGN models that bake their own SED on a native grid (no
# disc/torus block selection).
# Nebular emulator native grids ----------------------------------------------
# Cue (Li et al. 2025) ships its continuum grid inside ``cue_weights.npz``
# as the ``cont_wav`` array (~1840 points, 915 Å – 10⁸ Å). Without this
# entry the master grid stays at the SSP edge (~160 µm for BC03), so the
# rendered Cue continuum visibly cuts off in plots even though the
# emulator's native output extends to ~10 m. CLOUDY-grid / CB19 nebular
# backends evaluate exactly on their consumer's wave grid (no native of
# their own) so they contribute nothing here.
_NEBULAR_TEMPLATES: dict[str, tuple[tuple[str, str, float], ...]] = {
    # The npz key is ``cont_wavelength``; the ``cont_wav`` field on the
    # in-memory :class:`CueWeights` dataclass is assigned from it in
    # :func:`tengri.components.nebular.cue.load_cue_weights`.
    "cue": (("cue_weights.npz", "cont_wavelength", 1.0),),
}


_AGN_MODEL_TEMPLATES: dict[str, tuple[tuple[str, str, float], ...]] = {
    # QSOgen / GRAHSP / Richards2006 are analytic — they evaluate on whatever
    # wavelength grid they're handed. Nothing to declare here yet.
}


# ── H5 wavelength loader (cached) ─────────────────────────────────


@functools.cache
def _read_wavelength(filename: str, dataset_path: str, unit_to_aa: float) -> np.ndarray | None:
    """Read a wavelength array from an HDF5 or NPZ file, in Angstrom.

    Results are cached so repeated lookups across many ``SEDModel.build``
    calls (catalog-fitting, sweeps) only hit the filesystem once.

    Returns ``None`` if the file isn't present in any data directory, or if
    the dataset path doesn't exist inside the file.

    File-format dispatch is by extension: ``.npz`` → ``numpy.load`` (used by
    Cue's ``cue_weights.npz`` which carries ``cont_wav`` alongside the NN
    parameters); everything else → HDF5 via h5py (the original code path).
    """
    path = find_data_str(filename)
    if path is None:
        logger.debug("Template file %s not found in data dirs", filename)
        return None
    try:
        if path.endswith(".npz"):
            with np.load(path) as data:
                if dataset_path not in data:
                    logger.debug("Array %s not present in %s", dataset_path, path)
                    return None
                wave = np.asarray(data[dataset_path], dtype=np.float64)
        else:
            try:
                import h5py
            except ImportError:
                logger.warning(
                    "h5py not installed; cannot read native template grid from %s",
                    filename,
                )
                return None
            with h5py.File(path, "r") as h:
                if dataset_path not in h:
                    logger.debug("Dataset %s not present in %s", dataset_path, path)
                    return None
                wave = np.asarray(h[dataset_path][:], dtype=np.float64)
    except OSError as exc:
        logger.warning("Could not open %s: %r", path, exc)
        return None
    if unit_to_aa != 1.0:
        wave = wave * unit_to_aa
    # Keep only finite, strictly positive values, and sort ascending.
    wave = wave[np.isfinite(wave) & (wave > 0.0)]
    if wave.size == 0:
        return None
    return np.sort(wave)


def _first_present(candidates: Iterable[tuple[str, str, float]]) -> np.ndarray | None:
    for filename, dataset_path, unit_to_aa in candidates:
        wave = _read_wavelength(filename, dataset_path, unit_to_aa)
        if wave is not None:
            return wave
    return None


# ── Public lookups ────────────────────────────────────────────────


def native_wave_dust_emission(name: str | None) -> np.ndarray | None:
    """Native wavelength grid [Å] for a dust-emission model.

    Template models return their file's grid; analytic emitters return the
    synthetic 1 µm – 1 cm grid so the master union grid reaches the submm
    (#1005). Unknown names return ``None`` (callers treating "no native
    grid" as a fall-back to SSP coverage degrade gracefully).
    """
    if name is None or name in _GRIDLESS_DUST_EMISSION:
        return None
    if name in _ANALYTIC_DUST_EMISSION:
        return _ANALYTIC_DUST_WAVE_AA
    candidates = _DUST_EMISSION_TEMPLATES.get(name)
    if candidates is None:
        logger.debug("No native-grid declaration for dust emission %r", name)
        return None
    return _first_present(candidates)


def native_wave_nebular(model: str | None) -> np.ndarray | None:
    """Native wavelength grid [Å] for a nebular emission backend.

    Cue (``"cue"``) ships its continuum grid (~915 Å – 10⁸ Å, 1841 points)
    inside ``cue_weights.npz`` as the ``cont_wav`` array. Without it the
    master grid stops at the SSP edge (~160 µm for BC03-from-CIGALE) and
    Cue's UV-to-mm continuum visibly truncates in plots.

    CLOUDY-grid and CB19 nebular backends evaluate on whatever wave grid
    they're handed, so they contribute nothing here.
    """
    if not model or model in ("none", "off", "ssp", "cloudy", "cb19"):
        return None
    candidates = _NEBULAR_TEMPLATES.get(model)
    if candidates is None:
        return None
    return _first_present(candidates)


def native_wave_agn_torus(block: str | None) -> np.ndarray | None:
    """Native wavelength grid [Å] for an AGN torus block selection."""
    if not block or block == "none":
        return None
    candidates = _AGN_TORUS_TEMPLATES.get(block)
    if candidates is None:
        return None
    return _first_present(candidates)


def native_wave_agn_disc(block: str | None) -> np.ndarray | None:
    """Native wavelength grid [Å] for an AGN disc block selection."""
    if not block or block == "none":
        return None
    candidates = _AGN_DISC_TEMPLATES.get(block)
    if candidates is None:
        return None
    return _first_present(candidates)


def native_wave_agn_model(model: str | None) -> np.ndarray | None:
    """Native wavelength grid [Å] for a standalone AGN model.

    Most AGN models (QSOgen, GRAHSP, Richards2006, the parametric
    ``multicolor_agn``) are analytic and evaluate on whatever wavelength
    grid is supplied — they contribute nothing here. The dispatch table is
    kept for symmetry / future extensions.
    """
    if model is None:
        return None
    candidates = _AGN_MODEL_TEMPLATES.get(model)
    if candidates is None:
        return None
    return _first_present(candidates)


def collect_native_wavelength_grids(
    *,
    dust_emission_model: str | None = None,
    nebular_model: str | None = None,
    agn_model: str | None = None,
    agn_torus_block: str | None = None,
    agn_disc_block: str | None = None,
) -> list[np.ndarray]:
    """Gather every attached component's native wavelength grid.

    Returns a list of sorted ``ndarray`` grids in Angstrom. Components that
    don't carry a native template (analytic, missing data files, unknown
    name) contribute nothing — the returned list may be empty.
    """
    grids: list[np.ndarray] = []
    for w in (
        native_wave_dust_emission(dust_emission_model),
        native_wave_nebular(nebular_model),
        native_wave_agn_model(agn_model),
        native_wave_agn_torus(agn_torus_block),
        native_wave_agn_disc(agn_disc_block),
    ):
        if w is not None and w.size > 0:
            grids.append(w)
    return grids

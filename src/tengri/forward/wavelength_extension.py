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

Schemas vary across template files, wavelengths may live under
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
from tengri.utils.host_array import host_array
from tengri.utils.wavelength import NEBULAR_CONTINUUM_WAVE_MAX

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
    "dale2014_cigale": (("dale2014_templates_cigale.h5", "wavelength_aa", 1.0),),
    "draine_li2007": (
        ("dl07_templates_v2.h5", "wavelength", 1.0),
        ("dl07_templates.h5", "wavelength", 1.0),
    ),
    # Alias for the deprecated registry name.
    "dl07_tabulated": (
        ("dl07_templates_v2.h5", "wavelength", 1.0),
        ("dl07_templates.h5", "wavelength", 1.0),
    ),
    "dl07": (
        ("dl07_templates_v2.h5", "wavelength", 1.0),
        ("dl07_templates.h5", "wavelength", 1.0),
    ),
    "draine_li2014": (
        ("dl14_templates_v2.h5", "wavelength", 1.0),
        ("dl14_templates.h5", "wavelength", 1.0),
    ),
    "dl14": (
        ("dl14_templates_v2.h5", "wavelength", 1.0),
        ("dl14_templates.h5", "wavelength", 1.0),
    ),
    "astrodust": (("astrodust_templates.h5", "wavelength_um", 1e4),),
    "bosa": (("bosa_templates.h5", "wavelength_aa", 1.0),),
    "themis": (("themis_templates.h5", "wavelength_aa", 1.0),),
    "schreiber2018": (("schreiber2018_templates.h5", "schreiber2018/wavelength", 1.0),),
    "dh02_ce01": (("dh02_ce01_grid.h5", "dh02_ce01/wavelength", 1.0),),
}

# Analytic dust-emission models, no template file, but their emission still
# needs FIR/submm support on the master grid: without it the SED truncates at
# the SSP edge (160 µm for BC03) and submm photometry is silently zero while
# energy balance re-normalizes on the truncated grid (#1005). Same failure
# mode and fix as the Cue continuum grid below. 1 µm – 1 cm covers the PAH
# complexes through the Rayleigh–Jeans tail.
_ANALYTIC_DUST_EMISSION = frozenset(
    {
        "graybody",
        "modified_blackbody",
        "casey2012",
        "pah_drude",
        "schreiber2016",
        "energy_balance_split",
        "draine2021_pah",  # Alias for pah_drude
        "mbb",  # Alias for modified_blackbody
    }
)
_ANALYTIC_DUST_WAVE_AA = host_array(np.geomspace(1.0e4, 1.0e8, 512))
# Bookkeeping pseudo-model with no emission of its own, stays grid-less.
_GRIDLESS_DUST_EMISSION = frozenset({"energy_balance_split"})

# AGN torus templates --------------------------------------------------------
_AGN_TORUS_TEMPLATES: dict[str, tuple[tuple[str, str, float], ...]] = {
    "skirtor": (
        ("skirtor_templates_v3.h5", "wavelength", 1.0),
        ("skirtor_templates_v2.h5", "wavelength", 1.0),
    ),
    "silva04": (("silva04_torus_grid.h5", "silva04/wavelength", 1.0),),
    "cat3d_wind": (("cat3d_wind_torus_grid.h5", "cat3d_wind/wavelength", 1.0),),
    "cat3d_wind_lowfwd": (
        ("cat3d_wind_lowfwd_torus_grid.h5", "cat3d_wind_lowfwd/wavelength", 1.0),
    ),
    "fritz": (("fritz2006_torus_grid.h5", "fritz2006/wavelength_aa", 1.0),),
    "nenkova": (("nenkova08_torus_grid.h5", "nenkova/wavelength", 1.0),),
    "nenkova_agnfitter": (
        ("nenkova_agnfitter_torus_grid.h5", "nenkova_agnfitter/wavelength", 1.0),
    ),
    "nenkova_agnfitter_2p": (
        ("nenkova_agnfitter_2p_torus_grid.h5", "nenkova_agnfitter_2p/wavelength", 1.0),
    ),
    "nenkova_agnfitter_3p": (
        ("nenkova_agnfitter_3p_torus_grid.h5", "nenkova_agnfitter_3p/wavelength", 1.0),
    ),
    "skirtor_agnfitter": (("skirtor_mean3p_torus_grid.h5", "skirtor_mean3p/wavelength", 1.0),),
    "skirtor_agnfitter_1p": (("skirtor_mean1p_torus_grid.h5", "skirtor_mean1p/wavelength", 1.0),),
    "skirtor_agnfitter_2p": (("skirtor_mean2p_torus_grid.h5", "skirtor_mean2p/wavelength", 1.0),),
}

# AGN disc templates ---------------------------------------------------------
_AGN_DISC_TEMPLATES: dict[str, tuple[tuple[str, str, float], ...]] = {
    "relagn": (("relagn_disc_grid.h5", "wavelength_aa", 1.0),),
    # KD18 (Kubota & Done 2018) discs reach 0.062 A (200 keV) to 12.4 um; 28 per
    # cent of the bolometric energy lies below the SSP edge (91 A). The native axis
    # keeps that energy on the master grid, so a band integral or a bolometric
    # quadrature of the SED sees all of it.
    "kd18_agnfitter": (("kd18_agnfitter_disc_grid.h5", "kd18_agnfitter/wavelength", 1.0),),
    "kd18_agnfitter_warmindex": (
        ("kd18_agnfitter_warmindex_disc_grid.h5", "kd18_agnfitter_warmindex/wavelength", 1.0),
    ),
}

# Analytic AGN disc blocks (#2564). These blocks evaluate on whatever grid they
# are handed and normalize their energy to ``L_bol`` in closed form (or on a fixed
# internal grid), independent of that grid. A master grid that stops at the SSP edges
# (91 A, 160 um) would still drop the energy outside it from every downstream quadrature
# of the SED (band fluxes, bolometric checks, the energy ledgers). Each range
# [lo, hi] Angstrom is chosen so that < 1e-3 of the block's energy (default
# parameters, measured on a 1e-3 A - 1e10 A grid) lies outside it, and is
# justified by the emission physics:
#   kubota_done: hot Comptonizing corona to ~0.01 A (1 MeV) through the
#     color-corrected disc to the outer-edge Rayleigh-Jeans tail at 100 um.
#   multicolor: bare Shakura-Sunyaev disc plus the CIGALE-like EUV power-law
#     tail (starts at 8 A) to the outer-edge Rayleigh-Jeans tail at 100 um.
#   skirtor / schartmann2005*: CIGALE piecewise power laws with breakpoints
#     at 8 nm ... 1e6 nm (80 A ... 1e7 A), the steep lambda^-4 tail beyond;
#     the short side falls as lambda^alpha, < 1e-3 of the energy below 1 A.
#   adaf: Mahadevan 1997 cyclo-synchrotron (nu^0.4 rise to the mm peak), Compton
#     power law and bremsstrahlung cut off at kT_e/h (T_e < ~1e10 K gives
#     lambda > ~5e-3 A); the grid reaches 1e-3 A and 1 mm (1e7 A).
#   adaf_lopez2024: CIGALE ADAF/thin-disc blend, 8 A - 1e8 A.
#   powerlaw: nu^alpha exp(-h nu/k T_max) has no low-frequency cut-off; its
#     energy fraction beyond a wavelength is set by where the grid ends. It
#     takes the same 1 cm end as the analytic dust and torus support grids, and
#     starts where the T_max = 1e5 K exponential cut-off has removed the energy.
# 40 points per decade keeps the disc level at every node within 7e-4 of a 200000-point
# evaluation (20 per decade leaves 2.5e-3 for the ADAF blocks). Shortward of
# ``_DISC_EUV_BELOW_AA`` the grid carries 200 points per decade instead: the multicolor
# disc peaks in the EUV, and there the model-grid trapezoid of the bolometric SED is within
# 1.3e-4 of L_bol (40 per decade leaves 1.2e-3; the error falls as ~1/density). The cut
# sits below Lyman-alpha (1216 A): a denser grid across the IGM break re-resolves the
# transmission step for every other component of the model and moves the exact-path band
# fluxes of z > 7 models by up to 0.7 % against a fixed-quadrature LUT, so the extra
# nodes are spent only where the disc needs them.
_DISC_PTS_PER_DECADE = 40
_DISC_PTS_PER_DECADE_EUV = 200
_DISC_EUV_BELOW_AA = 1000.0
# The Kubota & Done disc takes fewer EUV nodes than that. Its model-grid bolometric error
# against a 3e5-node reference (log M 7 - 10, a = 0 and 0.9, log L_bol 9 - 12.5, the disc
# axis joined with the fsps_prsc_miles SSP axis) is 4.0e-4 at 40 per decade, 1.8e-4 at 60,
# 1.4e-4 at 70, 1.2e-4 at 75, 1.05e-4 at 80 and 1.9e-5 at 200, always worst at log M 7,
# a = 0.9, log L_bol 9. 75 is the smallest tested density under the 1.3e-4 bar above, and
# the model carries 625 fewer nodes than at 200.
_DISC_PTS_PER_DECADE_EUV_BY_BLOCK = {"kubota_done": 75}
# Tabulated discs whose template axis is too coarse for that normalization
# accuracy (KD18: 100 nodes over 6.3 decades, 1.6e-3): the declared grid is
# the axis plus a log grid at the same density as the analytic discs.
_DISC_DENSIFIED = frozenset({"kd18_agnfitter", "kd18_agnfitter_warmindex"})
_ANALYTIC_DISC_RANGE_AA: dict[str, tuple[float, float]] = {
    "kubota_done": (1.0e-2, 1.0e6),
    "multicolor": (8.0, 1.0e6),
    "skirtor": (1.0, 1.0e7),
    "schartmann2005": (1.0, 1.0e7),
    "schartmann2005_skirtor_atten": (1.0, 1.0e7),
    "adaf": (1.0e-3, 1.0e7),
    "adaf_lopez2024": (8.0, 1.0e8),
    "powerlaw": (1.0e1, 1.0e8),
}

# Disc blocks that stay grid-less: nothing (or < 1e-3 of the energy) lies
# outside the SSP window, measured with default parameters (#2564).
_GRIDLESS_DISC = frozenset(
    {
        "none",  # no emission
        "grahsp_sbpl",  # 1e-4 of the energy beyond 160 um, none below 91 A
        "qsogen",  # 7e-6 beyond 160 um
        "richards2006",  # 4e-4 below 91 A, 1e-5 beyond 160 um
        "slone_netzer",  # template axis 450 A - 3e7 A, nothing outside the window
    }
)

# Analytic AGN torus blocks: single-temperature and multi-temperature graybodies
# spanning 1 µm – 1 cm (IR dust emission through submm) without a template file.
# Same semantics as _ANALYTIC_DUST_EMISSION: the blocks compute their SED on any
# wavelength grid handed to them, so this synthetic grid ensures the master grid
# covers IR/submm (#2564).
_ANALYTIC_TORUS = frozenset(
    {
        "grahsp",  # GRAHSP log-Gaussian + Si feature (analytic dust continua)
        "qsogen",  # QSOgen single-T hot-dust blackbody (analytic)
        "simple",  # Single-temperature graybody torus
        "two_temperature",  # Hot + warm graybody torus
    }
)
_ANALYTIC_TORUS_WAVE_AA = host_array(np.geomspace(1.0e4, 1.0e8, 512))

# Grid-less AGN torus blocks: no emission contribution (kept for symmetry).
_GRIDLESS_TORUS = frozenset({"none"})

# Declaration stride for template axes denser than the science needs: only every
# ``stride``-th native node (plus the last one) enters the master grid; the block
# still interpolates the full-resolution template at whatever points it is
# handed. Every extra master-grid point costs each jitted predict, so a
# 4096-point axis (0.28 % steps) adds ~4000 points where ~1000 suffice.
# ``nenkova_agnfitter``: stride 4 (1025 nodes, 1.1 % steps) leaves the 8-500 um
# band mean within 1e-4 of the dense-grid value, every sub-band within 1e-4
# and the peak on the native node (measured for #2564; stride 16 already moves
# the peak by one 4.5 % step, stride 8 is the last stride with sub-bands < 5e-4).
_TORUS_DECLARATION_STRIDE = {"nenkova_agnfitter": 4}

# Standalone AGN models that bake their own SED on a native grid (no
# disc/torus block selection).
# Nebular emulator native grids ----------------------------------------------
# Cue (Li et al. 2025) ships its continuum grid inside ``cue_weights.npz``
# as the ``cont_wav`` array (~1840 points, 915 Å – 1e8 Å). The native grid
# ends at 1e8 Å (1 cm); native_wave_nebular extends it to 1e10 Å (1 m) with
# nodes at 20 points per decade, so a model without a radio block has nodes
# there, and interpolation onto longer grids (radio wing runs to 3e11 Å)
# continues the continuum as optically thin free-free (#2346). CLOUDY-grid
# / CB19 nebular backends evaluate exactly on their consumer's wave grid (no
# native of their own) so they declare nothing here; they carry the free-free
# tail wherever another component supplies nodes.
_NEBULAR_TEMPLATES: dict[str, tuple[tuple[str, str, float], ...]] = {
    # The npz key is ``cont_wavelength``; the ``cont_wav`` field on the
    # in-memory :class:`CueWeights` dataclass is assigned from it in
    # :func:`tengri.components.nebular.cue.load_cue_weights`.
    "cue": (("cue_weights.npz", "cont_wavelength", 1.0),),
}


_AGN_MODEL_TEMPLATES: dict[str, tuple[tuple[str, str, float], ...]] = {
    # QSOgen / GRAHSP / Richards2006 are analytic, they evaluate on whatever
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
    (#1005); registered aliases (``mbb``, ``draine2021_pah``, ``dl07``, ``dl14``) are
    declared like the model they alias. Unknown names return ``None`` (callers
    treating "no native grid" as a fall-back to SSP coverage degrade gracefully).
    """
    if name is None or name in _GRIDLESS_DUST_EMISSION:
        return None
    if name in _ANALYTIC_DUST_EMISSION:
        return np.asarray(_ANALYTIC_DUST_WAVE_AA)
    candidates = _DUST_EMISSION_TEMPLATES.get(name)
    if candidates is None:
        logger.debug("No native-grid declaration for dust emission %r", name)
        return None
    return _first_present(candidates)


def native_wave_nebular(model: str | None) -> np.ndarray | None:
    """Native wavelength grid [Å] for a nebular emission backend.

    Cue (``"cue"``) ships its continuum grid (~915 Å – 10⁸ Å, 1841 points)
    inside ``cue_weights.npz`` as the ``cont_wav`` array. Cue's native grid
    ends at 1e8 Å (1 cm); this function extends it to 1e10 Å (1 m) with nodes
    at 20 points per decade using analytic free-free continuation, so a model
    without a radio block has wavelength coverage to 1 m (#2346).

    CLOUDY-grid and CB19 nebular backends evaluate on whatever wave grid
    they're handed, so they contribute nothing here; if another component
    (radio block, dust IR) supplies nodes above 1 cm, the free-free tail
    continues there too.

    Returns
    -------
    ndarray or None
        Native nebular continuum wavelength grid [Å], or None for backends
        with no native grid (``"cloudy"``, ``"cb19"``, unknown names, or
        absent data files). For Cue, the grid extends from its native 1e8 Å
        to 1 m with analytic free-free continuation nodes; interpolation onto
        longer grids (radio wing to 3e11 Å) continues the tail via
        :func:`interp_continuum_with_freefree_tail`.

    Notes
    -----
    Cue's grid extension is consistent with pcigale and bagpipes, which
    tabulate nebular continuum to 1 m and 3.6 cm respectively. The extension
    is an analytic continuation (optically thin free-free, L_nu ∝ nu^-0.1),
    NOT an emulator prediction (#2346).
    """
    if not model or model in ("none", "off", "ssp", "cloudy", "cb19"):
        return None
    candidates = _NEBULAR_TEMPLATES.get(model)
    if candidates is None:
        return None
    wave = _first_present(candidates)
    if wave is None:
        return None

    # Extend past NEBULAR_CONTINUUM_WAVE_MAX (1e10 Å = 1 m) with free-free tail
    # nodes at 20 points per decade, for models that don't extend that far.
    if wave.max() < NEBULAR_CONTINUUM_WAVE_MAX:
        n_dec = np.log10(NEBULAR_CONTINUUM_WAVE_MAX) - np.log10(wave.max())
        n_pts = round(20 * n_dec) + 1
        tail = np.geomspace(wave.max(), NEBULAR_CONTINUUM_WAVE_MAX, n_pts)[1:]
        wave = np.concatenate([wave, tail])
    return wave


def native_wave_agn_torus(block: str | None) -> np.ndarray | None:
    """Native wavelength grid [Å] for an AGN torus block selection.

    Template blocks return their file's grid; analytic torus blocks return the
    synthetic 1 µm – 1 cm grid so the master union grid reaches the submm
    (#2564). Grid-less blocks (``"none"``) return ``None``.
    """
    if not block or block in _GRIDLESS_TORUS:
        return None
    if block in _ANALYTIC_TORUS:
        return np.asarray(_ANALYTIC_TORUS_WAVE_AA)
    candidates = _AGN_TORUS_TEMPLATES.get(block)
    if candidates is None:
        logger.debug("No native-grid declaration for torus block %r", block)
        return None
    wave = _first_present(candidates)
    stride = _TORUS_DECLARATION_STRIDE.get(block, 1)
    if wave is None or stride == 1:
        return wave
    keep = np.unique(np.append(np.arange(0, wave.size, stride), wave.size - 1))
    return wave[keep]


def _disc_log_grid(lo: float, hi: float, *, euv_pts: int = _DISC_PTS_PER_DECADE_EUV) -> np.ndarray:
    """Log grid over ``[lo, hi]`` [A]: 40 per decade, ``euv_pts`` per decade below 1000 A."""
    cut = min(max(lo, _DISC_EUV_BELOW_AA), hi)
    parts = []
    if lo < cut:
        parts.append(np.geomspace(lo, cut, round(np.log10(cut / lo) * euv_pts) + 1))
    if hi > cut:
        parts.append(np.geomspace(cut, hi, round(np.log10(hi / cut) * _DISC_PTS_PER_DECADE) + 1))
    return np.unique(np.concatenate(parts))


def native_wave_agn_disc(block: str | None) -> np.ndarray | None:
    """Native wavelength grid [Å] for an AGN disc block selection.

    Template discs return their file's axis, analytic discs a log grid over
    their emission range (``_ANALYTIC_DISC_RANGE_AA``), grid-less discs ``None``.
    """
    if not block or block in _GRIDLESS_DISC:
        return None
    if block in _ANALYTIC_DISC_RANGE_AA:
        lo, hi = _ANALYTIC_DISC_RANGE_AA[block]
        return _disc_log_grid(
            lo, hi, euv_pts=_DISC_PTS_PER_DECADE_EUV_BY_BLOCK.get(block, _DISC_PTS_PER_DECADE_EUV)
        )
    candidates = _AGN_DISC_TEMPLATES.get(block)
    if candidates is None:
        return None
    wave = _first_present(candidates)
    if wave is not None and block in _DISC_DENSIFIED:
        wave = np.unique(np.concatenate([wave, _disc_log_grid(wave.min(), wave.max())]))
    return wave


def native_wave_agn_model(model: str | None) -> np.ndarray | None:
    """Native wavelength grid [Å] for a standalone AGN model.

    Most AGN models (QSOgen, GRAHSP, Richards2006, the parametric
    ``multicolor_agn``) are analytic and evaluate on whatever wavelength
    grid is supplied, they contribute nothing here. The dispatch table is
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
    name) contribute nothing, the returned list may be empty.
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

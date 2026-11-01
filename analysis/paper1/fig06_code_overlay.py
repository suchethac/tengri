"""Figure 6: Overlay tengri posteriors on published SED-fitting results.

CLI: python fig06_code_overlay.py [--results-dir DIR] [--out-dir OUT]
     [--max-samples N] [--seed SEED]

Loads posterior samples from fits/ directory, computes surviving mass
using predict_properties, and overlays contours on published code values.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import logging
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import tengri

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adoption import is_adopted
from _cell_provenance import audit, banner
from _figure_style import CONFIG_COLORS, CONFIG_ORDER
from _grid_completeness import completeness_note, load_expected_galaxy_ids, present_on_disk
from config_metadata import CONFIGS, XLIKE_KEYS

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#: The locked sample, read by the grid, by fig09, and by this figure alike.
SELECTION_20 = Path(__file__).resolve().parent / "results" / "selected_galaxies_20.json"

#: The galaxies this figure draws, one panel each: the whole locked sample.
#:
#: This was a hand-written three-tuple from the three-galaxy version of the
#: paper, and it survived the move to the twenty-galaxy grid, so the one figure
#: that sets the grid against the published codes showed three galaxies while
#: Section 7 quoted numbers over all twenty. A later fix taught the completeness
#: guard to measure these three rather than the twenty -- correct for the figure
#: as it stood, but it made the guard agree with a stale panel list instead of
#: making the panels agree with the grid. Deriving them from the selection file
#: removes the second population altogether: the guard and the figure now
#: measure the same galaxies because there is only one list.
PANEL_GALAXY_IDS = tuple(load_expected_galaxy_ids(SELECTION_20))

#: Twenty galaxies make a 4 x 5 grid at two-column width.
PANEL_COLUMNS = 4
FIGURE_WIDTH = 7.1
PANEL_ROW_HEIGHT = 1.38
#: Space under the panels for tick labels, the axis label and two legend rows,
#: in inches, so the band stays tight whatever the figure's height. At 1.62 in
#: per row with fractional positions the figure plus its caption ran 101 pt
#: past the page and LaTeX cut the caption off mid-sentence.
LEGEND_BAND_IN = 0.80

#: Every panel is centered on its own galaxy but drawn at one common scale, so a
#: spread that looks twice as wide in one panel is twice as wide. The scale is
#: set by the tengri posteriors and the published central values, minus margin.
#: Published values further than this from the galaxy's tengri median are drawn
#: as arrows at the panel edge instead of setting the scale: galaxy 25489 has a
#: published log SFR of -11.3, a code reporting essentially zero, and letting
#: that set a common axis would flatten all twenty panels.
OFF_PANEL_DEX = 2.5
PANEL_MARGIN_DEX = 0.12

MARKER_COLORS = dict(CONFIG_COLORS)

#: Secondary encoding for the configurations. The palette validator passes the
#: five hues but puts III and IV at Delta E 7.6 under deuteranopia -- inside the
#: band that is legal only when something besides hue also carries identity.
#: Each configuration's contour outline therefore has its own dash pattern.
CONFIG_DASHES = {
    "I": "solid",
    "II": (0, (4.0, 1.4)),
    "III": (0, (1.0, 1.0)),
    "IV": (0, (4.0, 1.2, 1.0, 1.2)),
    "V": (0, (2.0, 1.2)),
    # Deferred and not drawn, but covered like CONFIG_COLORS covers it: a
    # mapping that names only the demonstrated rows is the partial census
    # test_config_census_coverage exists to catch, and it caught this one.
    "VI": (0, (6.0, 1.4, 1.0, 1.4, 1.0, 1.4)),
}

#: One ink for every published code, identity carried by marker shape. The
#: previous 0.2-0.8 gray ramp put Prospector at 0.8, near-invisible on white,
#: and a ramp reads as an ordering the codes do not have.
PUBLISHED_INK = "#4a4a4a"
TYPE_TAGS = {
    "blue_star_forming": "star-forming",
    "red_quiescent": "quiescent",
    "intermediate_dusty": "dusty",
    "mir_agn_candidate": "AGN cand.",
}
#: Keyed on the code names exactly as the published catalog spells them.
#:
#: This table used to say "Dense Basis" while the catalog says "Dense_Basis",
#: so the lookup missed on every galaxy and Dense Basis was never drawn: the
#: figure showed four codes under a caption promising five, and nothing
#: complained. `main` now refuses any code in the data that has no entry here.
CODE_MARKERS = {
    "BAGPIPES": "o",
    "BEAGLE": "s",
    "CIGALE": "^",
    "Dense_Basis": "v",
    "Prospector": "D",
}
#: Display names, where the catalog's spelling is not the one to print.
CODE_LABELS = {"Dense_Basis": "Dense Basis"}


#: Bump when the cached arrays change meaning or shape.
CACHE_VERSION = 1
CACHE_DIR = Path(__file__).resolve().parent / "results" / "fig06_cache"


def _cache_key(npz_path: Path, json_path: Path, max_samples: int) -> str:
    """Content hash of the cell plus the draw count: a re-fit cell misses."""
    import hashlib

    digest = hashlib.sha256()
    for path in (npz_path, json_path):
        digest.update(path.read_bytes())
    return f"v{CACHE_VERSION}:n{max_samples}:{digest.hexdigest()}"


class GalaxyData(NamedTuple):
    gal_id: int
    z: float
    config: str
    params_dict: dict
    mass_formed: np.ndarray
    mass_survived: np.ndarray
    sfr_100myr: np.ndarray


def load_fit_results(
    gal_id: int,
    config: str,
    results_dir: Path,
    max_samples: int = 200,
) -> GalaxyData | None:
    """Load posterior samples and compute derived quantities.

    Returns None if files don't exist (fit still running).

    Acceptance criteria are `_adoption.is_adopted`, shared with fig05.
    """
    npz_path = results_dir / f"{gal_id}_{config}.npz"
    json_path = results_dir / f"{gal_id}_{config}.json"

    if not (npz_path.exists() and json_path.exists()):
        return None

    # Load metadata; a best-so-far NPZ is written after every attempt
    with open(json_path) as f:
        meta = json.load(f)
    z = meta["z"]

    verdict = is_adopted(meta, config)
    if not verdict.adopted:
        logger.info(f"Skipping {gal_id}_{config} ({verdict.reason})")
        return None

    # The recompute below rebuilds the configuration's model per cell, which is
    # what makes twenty galaxies expensive. The cache is keyed on the cell's
    # content, so a re-fit cell is recomputed rather than served stale.
    # Named cache_key, not key: two loops below iterate `for key in npz.files`,
    # and under the shorter name the write at the end stored the last array
    # name ("energy") as the key, so every read missed and every render
    # silently recomputed all 87 cells.
    cache_key = _cache_key(npz_path, json_path, max_samples)
    cache_path = CACHE_DIR / f"{gal_id}_{config}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["key"]) == cache_key:
            return GalaxyData(
                gal_id=gal_id,
                z=float(cached["z"]),
                config=config,
                params_dict={},
                mass_formed=np.asarray(cached["mass_formed"]),
                mass_survived=np.asarray(cached["mass_survived"]),
                sfr_100myr=np.asarray(cached["sfr_100myr"]),
            )

    # Load NPZ; the number of saved draws is whatever the driver thinned to
    npz = np.load(npz_path, allow_pickle=False)

    # Extract parameters only (not derived quantities)
    params_dict = {}
    derived_keys = {
        "stellar_mass",
        "sfr_100myr",
        "sfr_10myr",
        "dust_tau",
        "dust_tau_name",
        "sfh_lookback_time_yr",
        "sfh_sfr_median",
        "sfh_sfr_p16",
        "sfh_sfr_p84",
        "model_photometry_median",
        "obs_fnu",
        "obs_sigma",
        "filter_names",
    }

    # The chain length is read off the sampled parameters themselves, not off
    # any one key by name.
    #
    # This used to be ``npz["redshift"].shape[0]``, which worked only because
    # every cell predated #2296. That change made ``spec.sample()`` return the
    # FREE keys only, and redshift is ``Fixed(z)`` in every configuration, so
    # the array stopped being written -- 57 of the 100 cells in rows I-V have
    # no ``redshift`` in their NPZ (I x17, II x20, IV x20, all of them the
    # cells launched after the change). A cell records the code it imported at
    # launch, so a mid-grid API change splits the archive's schema in two and
    # the figure sees a KeyError rather than a wrong number.
    #
    # Sampled parameters carry the full chain while the derived quantities are
    # written on a subsample, so the longest non-derived 1-D array IS the chain
    # length whatever the configuration happens to free. Redshift is not needed
    # here in any case: its value is ``meta["z"]``, read above.
    sampled_lengths = [
        int(npz[key].shape[0])
        for key in npz.files
        if key not in derived_keys and getattr(npz[key], "ndim", 0) == 1
    ]
    if not sampled_lengths:
        raise KeyError(
            f"{npz_path.name} carries no sampled-parameter array: cannot "
            f"determine the chain length (keys: {sorted(npz.files)})"
        )
    n_params_full = max(sampled_lengths)

    # Subsample from the full chain using the same indices for all quantities
    idx = np.round(np.linspace(0, n_params_full - 1, max_samples)).astype(int)

    for key in npz.files:
        val = npz[key]
        if (
            key not in derived_keys
            and hasattr(val, "shape")
            and len(val.shape) == 1
            and val.shape[0] == n_params_full
        ):
            params_dict[key] = val[idx]

    # Compute all derived quantities for the same samples using predict_properties
    mass_formed, mass_survived, sfr = _compute_all_derived_quantities(
        gal_id, config, params_dict, z
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        key=np.array(cache_key),
        z=np.array(z),
        mass_formed=mass_formed,
        mass_survived=mass_survived,
        sfr_100myr=sfr,
    )

    return GalaxyData(
        gal_id=gal_id,
        z=z,
        config=config,
        params_dict=params_dict,
        mass_formed=mass_formed,
        mass_survived=mass_survived,
        sfr_100myr=sfr,
    )


def resolve_configuration(config: str, configs_module) -> tuple[Callable, Callable]:
    """Resolve a configuration key to ``(builder, ssp_loader)``, grid or X-like.

    The one place a key becomes a model builder. Grid keys (``configs.CONFIG_KEYS``)
    resolve to ``configs.config_<key>`` with ``configs.load_ssp_for``; X-like keys
    (``config_metadata.XLIKE_KEYS``) to ``xlike_configs.XLIKE_BUILDERS`` with
    ``xlike_configs.load_ssp_for_xlike``. Both builders take ``(ssp, observation, z)``,
    and ``ssp_loader()`` takes no argument, so the caller treats the two alike.
    Nothing is built or loaded here.

    Parameters
    ----------
    config : str
        Grid key (``"I"``..``"VI"``) or X-like key (``"prospector_like"``, ...).
    configs_module : module
        The loaded ``configs.py`` (it is exec'd from its path, not imported).

    Raises
    ------
    KeyError
        If ``config`` is in neither suite.
    """
    if config in configs_module.CONFIG_KEYS:
        return getattr(configs_module, f"config_{config}"), partial(
            configs_module.load_ssp_for, config
        )
    if config in XLIKE_KEYS:
        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)
        xlike = importlib.import_module("analysis.paper1.xlike_configs")
        return xlike.XLIKE_BUILDERS[config], partial(xlike.load_ssp_for_xlike, config)
    raise KeyError(
        f"configuration {config!r} is neither a grid key {configs_module.CONFIG_KEYS} "
        f"nor an X-like key {XLIKE_KEYS}; add its builder before asking a figure for it"
    )


def _compute_all_derived_quantities(
    gal_id: int, config: str, params_dict: dict, z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute derived quantities for posterior samples using predict_properties.

    Returns (mass_formed_log, mass_survived_log, sfr_log) all in log10 space.
    Uses the same samples for all quantities to ensure proper correspondence.
    """
    # configs.py and candels_io.py are this script's siblings, so anchor on
    # this file. Deriving them from results_dir instead only worked while
    # results_dir was the default one two levels below them; --results-dir
    # pointing anywhere else looked for configs.py beside that directory.
    analysis_dir = Path(__file__).resolve().parent
    configs_path = analysis_dir / "configs.py"

    # Load configs module dynamically
    spec = importlib.util.spec_from_file_location("configs", configs_path)
    configs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configs)

    # Load candels_io
    candels_io_path = analysis_dir / "candels_io.py"
    spec2 = importlib.util.spec_from_file_location("candels_io", candels_io_path)
    candels_io = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(candels_io)

    config_fn, load_ssp = resolve_configuration(config, configs)
    ssp = load_ssp()

    # Load photometry
    candels_cat = candels_io.load_candels_z1()
    idx = np.where(candels_cat["id"] == gal_id)[0][0]
    row_data = candels_cat["data"][idx]
    names, _, _ = candels_io.photometry_for_row(candels_cat["header"], row_data)
    obs = tengri.Photometry.from_names(names)

    # Build model
    model = config_fn(ssp, obs, z)

    # Keep exactly the parameters the model declares, by name.
    #
    # The caller selects them by shape -- any 1-D array in the cell whose
    # length matches the draw count, minus a hand-listed set of known
    # non-parameters. That denylist has to be exhaustive to be correct, and it
    # was not: cells carry `divergent_mask` and `energy`, which are per-draw
    # diagnostics of exactly that length, and they arrived here as parameters.
    # The model refused them by name, which is the system working, but the
    # selection rule is "looks like a parameter" where it should be "is one",
    # and the model is the authority on that.
    declared = set(model.spec.free_params)
    unknown = sorted(set(params_dict) - declared)
    params_dict = {name: values for name, values in params_dict.items() if name in declared}
    if unknown:
        logger.info(
            "%s/%s: ignoring %d cell array(s) that are not declared parameters: %s",
            gal_id,
            config,
            len(unknown),
            ", ".join(unknown),
        )
    missing = sorted(declared - set(params_dict))
    if missing:
        raise SystemExit(
            f"cell {gal_id}_{config} records no draws for {missing}, which "
            f"configuration {config} declares free. The cell and the "
            "configuration disagree about the model; re-run it rather than "
            "predicting at a partial parameter set."
        )

    # Compute all derived quantities for each sample
    mass_formed_list = []
    mass_survived_list = []
    sfr_list = []
    n_samples = len(next(iter(params_dict.values()))) if params_dict else 0

    for i in range(n_samples):
        # Build sample dict with floats (not arrays)
        sample_dict = {}
        for name, vals in params_dict.items():
            sample_dict[name] = float(vals[i])

        # Compute all three quantities together
        props = model.predict_properties(
            sample_dict, names=("stellar_mass", "stellar_mass_surviving", "sfr_100myr")
        )
        mass_formed_list.append(props["stellar_mass"])  # linear Msun
        mass_survived_list.append(props["stellar_mass_surviving"])  # linear Msun
        sfr_list.append(props["sfr_100myr"])  # linear Msun/yr

    # Convert to log10 and arrays
    mass_formed_log = np.log10(np.array(mass_formed_list))
    mass_survived_log = np.log10(np.array(mass_survived_list))
    sfr_log = np.log10(np.array(sfr_list))

    # Cross-check: recomputed medians must agree with expected values
    print(f"Sample pairing verification ({gal_id}_{config}):")
    print(f"  Recomputed mass_formed median: {np.median(mass_formed_log):.4f} dex")
    print(f"  Recomputed mass_survived median: {np.median(mass_survived_log):.4f} dex")
    print(f"  Recomputed sfr_100myr median: {np.median(sfr_log):.4f} dex")

    # Verify surviving mass is below formed mass
    diff = mass_formed_log - mass_survived_log
    print(
        f"  Mass constraint check: median diff={np.median(diff):.4f} dex "
        f"(min={np.min(diff):.4f}, max={np.max(diff):.4f})"
    )
    if not np.all(diff >= 0):
        raise ValueError(
            f"Surviving mass exceeds formed mass for some samples ({gal_id}_{config}). "
            f"This indicates an error in the model prediction."
        )

    return mass_formed_log, mass_survived_log, sfr_log


def load_published_values(csv_path: Path) -> dict:
    """Load published code values from the ingested CSV."""
    values = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gal_id = int(row["id"])
            code = row["code"]
            if gal_id not in values:
                values[gal_id] = {}
            values[gal_id][code] = {
                "logmstar": float(row["logmstar"]),
                "logmstar_lo": float(row["logmstar_lo"]),
                "logmstar_hi": float(row["logmstar_hi"]),
                "logsfr": float(row["logsfr"]),
                "logsfr_lo": float(row["logsfr_lo"]),
                "logsfr_hi": float(row["logsfr_hi"]),
                "mass_definition_note": row["mass_definition_note"],
                "sfr_timescale_note": row["sfr_timescale_note"],
            }
    return values


def _darken(color: str, amount: float = 0.28) -> tuple[float, float, float]:
    """Same hue, closer to black: outline relief for the light Okabe-Ito hues.

    The validator puts orange, pink and sky blue under 3:1 against white. The
    fills stay the shared palette so this figure matches fig05 and fig09; only
    the outline is deepened.
    """
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def _contour_68(ax, x: np.ndarray, y: np.ndarray, config: str) -> None:
    """The 68% highest-density region of one configuration's posterior."""
    pad = 0.15
    xg = np.linspace(x.min() - pad, x.max() + pad, 160)
    yg = np.linspace(y.min() - pad, y.max() + pad, 160)
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    Z = gaussian_kde(np.vstack([x, y]))(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    ranked = np.sort(Z.ravel())[::-1]
    enclosed = np.cumsum(ranked)
    enclosed /= enclosed[-1]
    level = ranked[min(np.searchsorted(enclosed, 0.68), ranked.size - 1)]

    color = MARKER_COLORS[config]
    ax.contourf(X, Y, Z, levels=[level, Z.max()], colors=[color], alpha=0.30, zorder=2)
    ax.contour(
        X,
        Y,
        Z,
        levels=[level],
        colors=[_darken(color)],
        linewidths=1.1,
        linestyles=[CONFIG_DASHES[config]],
        zorder=3,
    )


def plot_galaxy_overlay(
    ax,
    gal_id: int,
    published: dict,
    tengri_data: dict[str, GalaxyData],
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """One panel: each configuration's 68% region beside the published codes."""
    (x_lo, x_hi), (y_lo, y_hi) = limits

    for config in CONFIG_ORDER:
        if config in tengri_data:
            data = tengri_data[config]
            _contour_68(ax, data.mass_survived, data.sfr_100myr, config)

    for code in sorted(CODE_MARKERS):
        if code not in published:
            continue
        p = published[code]
        x, y = p["logmstar"], p["logsfr"]
        inside = x_lo <= x <= x_hi and y_lo <= y <= y_hi
        if inside:
            ax.errorbar(
                x,
                y,
                xerr=[[x - p["logmstar_lo"]], [p["logmstar_hi"] - x]],
                yerr=[[y - p["logsfr_lo"]], [p["logsfr_hi"] - y]],
                fmt="none",
                ecolor=PUBLISHED_INK,
                elinewidth=0.55,
                alpha=0.38,
                zorder=4,
            )
            ax.plot(
                x,
                y,
                marker=CODE_MARKERS[code],
                color=PUBLISHED_INK,
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=4.6,
                linestyle="none",
                zorder=5,
            )
        else:
            # Off the common scale. The code's own marker, hollow, pinned just
            # inside the edge, with an arrow pointing where the value went. A
            # bare triangle was used first and was indistinguishable from Dense
            # Basis, whose marker is a triangle.
            w, h = x_hi - x_lo, y_hi - y_lo
            dx = -1 if x < x_lo else 1 if x > x_hi else 0
            dy = -1 if y < y_lo else 1 if y > y_hi else 0
            cx = min(max(x, x_lo + 0.12 * w), x_hi - 0.12 * w)
            cy = min(max(y, y_lo + 0.14 * h), y_hi - 0.14 * h)
            ax.plot(
                cx,
                cy,
                marker=CODE_MARKERS[code],
                markerfacecolor="white",
                markeredgecolor=PUBLISHED_INK,
                markeredgewidth=0.8,
                markersize=4.6,
                linestyle="none",
                zorder=6,
            )
            ax.annotate(
                "",
                xy=(cx + dx * 0.10 * w, cy + dy * 0.12 * h),
                xytext=(cx, cy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=PUBLISHED_INK,
                    lw=0.7,
                    mutation_scale=6,
                    shrinkA=3.0,
                    shrinkB=0.0,
                ),
                annotation_clip=False,
                zorder=6,
            )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)


def _panel_centers_and_span(panel_ids, published_all, tengri_results):
    """Each galaxy's center, and one span in (mass, SFR) shared by every panel.

    Extents use the central 68% of each configuration's draws. The first
    version used 3-97%, and two quiescent galaxies with an unconstrained SFR
    then set the scale for all twenty -- 25489 alone needed 10.1 dex and 25206
    5.2, against 2.6 for every other galaxy and a median of 1.3 -- which pressed
    the other eighteen into a thin strip.

    The SFR span is the 90th percentile of the per-galaxy heights rather than
    their maximum, for the same reason. The galaxies above it run off the
    bottom edge of their own panel, which is what an SFR unconstrained from
    below should look like; they are returned so the caller can say which.
    """
    centers, widths, heights = {}, [], {}
    for gal_id in panel_ids:
        cells = tengri_results.get(gal_id, {})
        xs = [v for d in cells.values() for v in np.percentile(d.mass_survived, [16, 84])]
        ys = [v for d in cells.values() for v in np.percentile(d.sfr_100myr, [16, 84])]
        if xs:
            x_ref, y_ref = float(np.median(xs)), float(np.median(ys))
            for p in published_all.get(gal_id, {}).values():
                if abs(p["logmstar"] - x_ref) <= OFF_PANEL_DEX:
                    xs.append(p["logmstar"])
                if abs(p["logsfr"] - y_ref) <= OFF_PANEL_DEX:
                    ys.append(p["logsfr"])
        else:
            xs = [p["logmstar"] for p in published_all.get(gal_id, {}).values()]
            ys = [p["logsfr"] for p in published_all.get(gal_id, {}).values()]
        if not xs or not ys:
            continue
        centers[gal_id] = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        widths.append(max(xs) - min(xs))
        heights[gal_id] = max(ys) - min(ys)

    span_x = max(widths) + 2 * PANEL_MARGIN_DEX
    span_y = float(np.percentile(list(heights.values()), 90)) + 2 * PANEL_MARGIN_DEX
    overflowing = sorted(g for g, h in heights.items() if h + 2 * PANEL_MARGIN_DEX > span_y)
    return centers, (span_x, span_y), overflowing


def _draw_panels(panel_ids, galaxies, published_all, tengri_results):
    """Twenty panels on a common scale, one shared legend beneath them."""
    n_rows = -(-len(panel_ids) // PANEL_COLUMNS)
    fig, axes = plt.subplots(
        n_rows,
        PANEL_COLUMNS,
        figsize=(FIGURE_WIDTH, PANEL_ROW_HEIGHT * n_rows + LEGEND_BAND_IN),
        squeeze=False,
    )
    centers, (span_x, span_y), overflowing = _panel_centers_and_span(
        panel_ids, published_all, tengri_results
    )
    print(
        f"fig06 common scale: {span_x:.2f} dex in mass, {span_y:.2f} dex in SFR; "
        f"posteriors taller than that run off their panel for {overflowing}"
    )

    for index, ax in enumerate(axes.flat):
        if index >= len(panel_ids):
            ax.set_visible(False)
            continue
        gal_id = panel_ids[index]
        cx, cy = centers[gal_id]
        limits = ((cx - span_x / 2, cx + span_x / 2), (cy - span_y / 2, cy + span_y / 2))
        plot_galaxy_overlay(
            ax, gal_id, published_all.get(gal_id, {}), tengri_results.get(gal_id, {}), limits
        )

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)
            ax.spines[side].set_color("#8a8a8a")
        ax.tick_params(labelsize=6.5, width=0.6, length=2.5, color="#8a8a8a", pad=1.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        tag = TYPE_TAGS.get(galaxies[gal_id]["type_label"], galaxies[gal_id]["type_label"])
        ax.text(
            0.04,
            0.95,
            f"{gal_id}",
            transform=ax.transAxes,
            fontsize=7.5,
            fontweight="bold",
            color="#222222",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6),
            zorder=7,
        )
        ax.text(
            0.04,
            0.80,
            tag,
            transform=ax.transAxes,
            fontsize=6.3,
            color="#6b6b6b",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.4),
            zorder=7,
        )

    fig_h = PANEL_ROW_HEIGHT * n_rows + LEGEND_BAND_IN
    fig.supxlabel(
        r"$\log_{10}\,M_{\star,\mathrm{surv}}\,/\,M_\odot$",
        fontsize=9,
        y=(LEGEND_BAND_IN - 0.31) / fig_h,
        va="center",
    )
    fig.supylabel(
        r"$\log_{10}\,\mathrm{SFR}_{100\,\mathrm{Myr}}\,/\,M_\odot\,\mathrm{yr}^{-1}$",
        fontsize=9,
        x=0.015,
    )

    config_handles = [
        plt.matplotlib.patches.Patch(
            facecolor=(*plt.matplotlib.colors.to_rgb(MARKER_COLORS[c]), 0.30),
            edgecolor=_darken(MARKER_COLORS[c]),
            linestyle=CONFIG_DASHES[c],
            linewidth=1.1,
            label=f"Configuration {c}",
        )
        for c in CONFIG_ORDER
    ]
    code_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=CODE_MARKERS[code],
            color=PUBLISHED_INK,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=5,
            linestyle="none",
            label=CODE_LABELS.get(code, code),
        )
        for code in sorted(CODE_MARKERS)
    ]
    fig.legend(
        handles=config_handles,
        loc="lower center",
        ncol=5,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.53, 0.19 / fig_h),
        handlelength=2.2,
        columnspacing=1.2,
    )
    fig.legend(
        handles=code_handles,
        loc="lower center",
        ncol=5,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.53, 0.0),
        columnspacing=1.6,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.99,
        bottom=LEGEND_BAND_IN / fig_h,
        wspace=0.28,
        hspace=0.36,
    )
    return fig


def main(
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    max_samples: int = 200,
    seed: int = 42,
):
    """Generate Figure 6: code overlay."""

    # Determine paths relative to script location
    script_dir = Path(__file__).resolve().parent
    analysis_dir = script_dir
    paper_repo_root = analysis_dir.parent.parent

    if results_dir is None:
        results_dir = Path(__file__).parent / "results" / "fits"
    else:
        results_dir = Path(results_dir)

    if out_dir is None:
        out_dir = analysis_dir / "figures"
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Unlike fig05 and fig09, this figure rebuilds each configuration's model to
    # recompute derived quantities, then feeds it the cell's samples. If the
    # cells hold a different model the rebuild raises UnknownParameterError two
    # hundred lines in, naming a parameter rather than the cause. Refuse here,
    # with the cause.
    mismatches, notes = audit(results_dir, CONFIGS)
    text = banner(results_dir, mismatches, notes)
    if text:
        print(text, file=sys.stderr)
    if mismatches:
        raise SystemExit(
            "fig06 rebuilds each configuration's model and cannot draw cells that "
            "hold a different one. Re-run the grid, or point --results-dir at a "
            "directory whose cells match configs.py."
        )

    # The audit above asks whether the cells that ARE here hold the right model.
    # It says nothing about the ones that are not, and a directory holding none
    # of them passes it trivially: every cell was logged "not ready" at INFO and
    # this script saved a finished-looking comparison whose tengri arm was empty.
    selection_path = analysis_dir / "results" / "selected_galaxies_20.json"
    try:
        locked_ids = load_expected_galaxy_ids(selection_path)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(
            f"fig06 cannot read the locked sample at {selection_path}: {exc}. "
            "Without it there is nothing to measure completeness against."
        ) from exc

    off_sample = [gal for gal in PANEL_GALAXY_IDS if gal not in locked_ids]
    if off_sample:
        raise SystemExit(
            f"fig06 draws {off_sample}, which the locked sample at "
            f"{selection_path.name} no longer contains. The panels and the sample "
            "have drifted apart; reconcile them rather than publishing a "
            "comparison for galaxies the grid does not fit."
        )

    present = present_on_disk(results_dir, PANEL_GALAXY_IDS, CONFIG_ORDER)
    if not present:
        raise SystemExit(
            f"fig06 found no finished cells for {list(PANEL_GALAXY_IDS)} in "
            f"{results_dir}. This figure's whole claim is tengri's posteriors "
            "beside the published codes, so with an empty tengri arm there is no "
            "figure to draw -- only the published points under a caption that "
            "promises a comparison. Point --results-dir at a directory holding "
            "cells for the locked sample."
        )
    shortfall = completeness_note(present, PANEL_GALAXY_IDS, CONFIG_ORDER)

    # Load published values
    csv_path = analysis_dir / "results" / "art_sedfitting_z1.csv"
    published_all = load_published_values(csv_path)
    unmapped = sorted(
        {code for gal in PANEL_GALAXY_IDS for code in published_all.get(gal, {})}
        - set(CODE_MARKERS)
    )
    if unmapped:
        raise SystemExit(
            f"fig06 has no marker for published code(s) {unmapped}, which the catalog "
            f"reports for these galaxies. Drawing on would silently omit them from a "
            f"figure whose caption names every code; add them to CODE_MARKERS."
        )

    # Galaxy metadata comes from the same locked selection the panels do. It used
    # to come from selected_galaxies.json, the three-galaxy file, with a
    # hand-patched 24497 -> 16049 swap -- a second copy of the sample that only
    # agreed with the grid for as long as someone remembered to patch it.
    meta_path = selection_path
    with open(meta_path) as f:
        meta = json.load(f)
    galaxies = {g["id"]: g for g in meta["selected_galaxies"]}

    # Load tengri results (only those available)
    tengri_results = {}
    json_sidecar = {
        "figure": "Figure 6: tengri posteriors vs published codes",
        "published_data": {},
        "tengri_data": {},
        "pending_cells": [],
        "refused_cells": [],
        "inter_code_ranges": {},
    }

    for gal_id in PANEL_GALAXY_IDS:
        if gal_id not in galaxies:
            # Dropping it silently would publish two panels under a caption
            # promising three, with nothing in the figure or the sidecar
            # recording the third.
            raise SystemExit(
                f"fig06 draws galaxy {gal_id}, which {meta_path.name} does not "
                "describe. Without its redshift and type label there is no panel "
                "to draw, and a figure short one panel must not be saved as "
                "though it were whole."
            )

        gal_meta = galaxies[gal_id]
        tengri_results[gal_id] = {}
        json_sidecar["published_data"][gal_id] = published_all.get(gal_id, {})
        json_sidecar["tengri_data"][gal_id] = {}

        # Try to load each configuration
        for config in CONFIG_ORDER:
            data = load_fit_results(gal_id, config, results_dir, max_samples)
            if data is None:
                # Two different states, and only one of them will ever change.
                # A cell whose files are absent has not run; a cell that is on
                # disk and declined has run and been refused by the adoption
                # bar, and will stay refused. Recording both as "pending" left
                # a reader of this sidecar waiting on a verdict already given.
                on_disk = (results_dir / f"{gal_id}_{config}.npz").exists() and (
                    results_dir / f"{gal_id}_{config}.json"
                ).exists()
                bucket = "refused_cells" if on_disk else "pending_cells"
                json_sidecar[bucket].append(f"{gal_id}_{config}")
                logger.info(
                    f"Skipping {gal_id}_{config} "
                    f"({'did not pass the adoption bar' if on_disk else 'not yet run'})"
                )
                continue

            tengri_results[gal_id][config] = data

            # Store in sidecar (values are already in log10 from _compute_all_derived_quantities)
            mass_surv_log = data.mass_survived
            mass_form_log = data.mass_formed
            sfr_log = data.sfr_100myr

            json_sidecar["tengri_data"][gal_id][config] = {
                "stellar_mass_survived_p16": float(np.percentile(mass_surv_log, 16)),
                "stellar_mass_survived_p50": float(np.percentile(mass_surv_log, 50)),
                "stellar_mass_survived_p84": float(np.percentile(mass_surv_log, 84)),
                "stellar_mass_formed_p16": float(np.percentile(mass_form_log, 16)),
                "stellar_mass_formed_p50": float(np.percentile(mass_form_log, 50)),
                "stellar_mass_formed_p84": float(np.percentile(mass_form_log, 84)),
                "log_sfr_100myr_p16": float(np.percentile(sfr_log, 16)),
                "log_sfr_100myr_p50": float(np.percentile(sfr_log, 50)),
                "log_sfr_100myr_p84": float(np.percentile(sfr_log, 84)),
            }

        # Compute inter_code_ranges for this galaxy
        pub_data = published_all.get(gal_id, {})
        if pub_data and gal_id in tengri_results:
            # Published inter-code range
            logmstar_values = [v["logmstar"] for v in pub_data.values()]
            logsfr_values = [v["logsfr"] for v in pub_data.values()]

            if logmstar_values and logsfr_values:
                logmstar_min = min(logmstar_values)
                logmstar_max = max(logmstar_values)
                logsfr_min = min(logsfr_values)
                logsfr_max = max(logsfr_values)

                # Find which codes attain the extremes
                logmstar_min_code = next(
                    c for c, v in pub_data.items() if v["logmstar"] == logmstar_min
                )
                logmstar_max_code = next(
                    c for c, v in pub_data.items() if v["logmstar"] == logmstar_max
                )
                logsfr_min_code = next(c for c, v in pub_data.items() if v["logsfr"] == logsfr_min)
                logsfr_max_code = next(c for c, v in pub_data.items() if v["logsfr"] == logsfr_max)

                published_ranges = {
                    "logmstar": {
                        "min": float(logmstar_min),
                        "max": float(logmstar_max),
                        "range": float(logmstar_max - logmstar_min),
                        "min_code": logmstar_min_code,
                        "max_code": logmstar_max_code,
                    },
                    "logsfr": {
                        "min": float(logsfr_min),
                        "max": float(logsfr_max),
                        "range": float(logsfr_max - logsfr_min),
                        "min_code": logsfr_min_code,
                        "max_code": logsfr_max_code,
                    },
                }

                # Tengri inter-configuration range
                tengri_configs = tengri_results.get(gal_id, {})
                if len(tengri_configs) > 0:
                    mass_surv_medians = [
                        np.median(tengri_configs[c].mass_survived)
                        for c in sorted(tengri_configs.keys())
                    ]
                    sfr_medians = [
                        np.median(tengri_configs[c].sfr_100myr)
                        for c in sorted(tengri_configs.keys())
                    ]

                    mass_surv_range = float(max(mass_surv_medians) - min(mass_surv_medians))
                    sfr_range = float(max(sfr_medians) - min(sfr_medians))

                    # Check if medians are inside published range
                    configurations = {}
                    for config in sorted(tengri_configs.keys()):
                        mass_surv_med = float(np.median(tengri_configs[config].mass_survived))
                        sfr_med = float(np.median(tengri_configs[config].sfr_100myr))

                        configurations[config] = {
                            "stellar_mass_survived_median": mass_surv_med,
                            "log_sfr_100myr_median": sfr_med,
                            "median_inside_published_range": {
                                "logmstar": (logmstar_min <= mass_surv_med <= logmstar_max),
                                "logsfr": (logsfr_min <= sfr_med <= logsfr_max),
                            },
                        }

                    json_sidecar["inter_code_ranges"][gal_id] = {
                        "published_ranges": published_ranges,
                        "tengri_inter_configuration_range": {
                            "stellar_mass_survived": mass_surv_range,
                            "log_sfr_100myr": sfr_range,
                        },
                        "configurations": configurations,
                    }

                    # Print for verification
                    print(f"inter_code_ranges[{gal_id}]:")
                    print(json.dumps(json_sidecar["inter_code_ranges"][gal_id], indent=2))

    fig = _draw_panels(PANEL_GALAXY_IDS, galaxies, published_all, tengri_results)

    # A partial grid still draws -- fig05 and fig09 stamp rather than refuse, and
    # a reader comparing panels needs the same sentence on all three. What must
    # not happen is the stamp being absent because nobody asked.
    if shortfall:
        print(shortfall, file=sys.stderr)
        json_sidecar["completeness"] = shortfall

    # Save figure
    for fmt in ["pdf", "png"]:
        out_path = out_dir / f"fig06_code_overlay.{fmt}"
        plt.savefig(out_path, dpi=300 if fmt == "png" else None, bbox_inches="tight")
        logger.info(f"Saved {out_path}")
    plt.close()

    # Save JSON sidecar
    json_path = analysis_dir / "results" / "fig06_code_overlay_data.json"
    with open(json_path, "w") as f:
        json.dump(json_sidecar, f, indent=2)
    logger.info(f"Saved {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Path to fits/ directory (default: paper1/results/fits)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: paper1/figures)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum posterior samples to process (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="fill the per-cell cache and exit without drawing; pair with --galaxies to shard",
    )
    parser.add_argument(
        "--galaxies",
        type=str,
        default="",
        help="comma-separated galaxy IDs for --precompute-only (default: all panels)",
    )

    args = parser.parse_args()
    if args.precompute_only:
        results = args.results_dir or Path(__file__).parent / "results" / "fits"
        wanted = (
            [int(g) for g in args.galaxies.split(",") if g.strip()]
            if args.galaxies
            else list(PANEL_GALAXY_IDS)
        )
        unknown = sorted(set(wanted) - set(PANEL_GALAXY_IDS))
        if unknown:
            raise SystemExit(f"not panels of this figure: {unknown}")
        for gal in wanted:
            for cfg in CONFIG_ORDER:
                load_fit_results(gal, cfg, Path(results), args.max_samples)
        raise SystemExit(0)
    main(
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        max_samples=args.max_samples,
        seed=args.seed,
    )

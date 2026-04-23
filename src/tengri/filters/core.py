"""Filter discovery helpers for the tengri filter library.

Provides convenient functions to explore available filters by instrument,
load filters, get descriptions, and suggest filters by wavelength coverage.
"""

import numpy as np

from tengri.observation.filters import (
    FILTER_REGISTRY,
    compute_effective_wavelength,
    load_filter_set,
)


def list_filters(instrument: str | None = None) -> list[str]:
    """Return sorted list of filter names shipped with tengri.

    Parameters
    ----------
    instrument : str, optional
        Filter by instrument prefix (case-insensitive, substring match).
        E.g., "sdss", "jwst", "hst". Default: None (return all).

    Returns
    -------
    list of str
        Sorted filter names. Empty list if no matches.

    Notes
    -----
    Filtering is permissive: matches if the instrument string appears
    anywhere in the filter name (case-insensitive).

    Examples
    --------
    >>> all_filters = list_filters()
    >>> sdss_filters = list_filters(instrument="sdss")
    >>> len(sdss_filters)
    5
    """
    names = sorted(FILTER_REGISTRY.keys())

    if instrument is None:
        return names

    instrument_lower = instrument.lower()
    return [name for name in names if instrument_lower in name.lower()]


def load(names: list[str]):
    """Load multiple filters by short name.

    Thin alias for ``tengri.observation.filters.load_filter_set`` for
    discoverability and consistency with the filters namespace.

    Parameters
    ----------
    names : list of str
        Short filter names from the registry (e.g., ["sdss_r", "jwst_f200w"]).

    Returns
    -------
    filter_waves : list of jnp.ndarray
        Wavelength arrays per filter, shape (n_wave,) [Angstrom].
    filter_trans : list of jnp.ndarray
        Transmission arrays per filter, shape (n_wave,) (dimensionless [0, 1]).
    filter_curves : list of FilterCurve
        Full FilterCurve objects with wavelength, transmission, and name.

    Raises
    ------
    KeyError
        If any name is not in the filter registry.

    Examples
    --------
    >>> waves, trans, curves = load(["sdss_r", "sdss_i"])
    >>> len(curves)
    2
    >>> curves[0].name
    'sdss_r'

    Notes
    -----
    Filters are downloaded from the SVO Filter Profile Service on first use
    and cached locally under data/filters/.
    """
    return load_filter_set(names)


def describe(name: str) -> str:
    """Return a one-line description of a filter.

    Computes the transmission-weighted effective wavelength and wavelength
    range of the transmission curve.

    Parameters
    ----------
    name : str
        Filter short name from the registry.

    Returns
    -------
    str
        Human-readable description. Format:
        "<name>: lambda_eff ~ X.XXX μm (range A–B μm)" or similar on failure.

    Notes
    -----
    If the filter fails to load, returns a fallback description.
    Effective wavelength is computed as the transmission-weighted mean.
    """
    try:
        fc = load_filter_set([name])[2][0]
        wave_np = np.asarray(fc.wave)
        trans_np = np.asarray(fc.trans)

        # Compute transmission-weighted effective wavelength
        lam_eff = compute_effective_wavelength(wave_np, trans_np)

        # Compute range (wavelengths with nonzero transmission)
        nonzero = trans_np > 0
        if np.any(nonzero):
            wave_min = wave_np[nonzero].min()
            wave_max = wave_np[nonzero].max()
        else:
            wave_min, wave_max = wave_np.min(), wave_np.max()

        # Format wavelengths
        def format_wave(w):
            if w >= 1e4:
                return f"{w / 1e4:.2f}"
            else:
                return f"{w:.0f}"

        if lam_eff >= 1e4:
            unit = "μm"
            lam_eff_fmt = f"{lam_eff / 1e4:.3f}"
        else:
            unit = "Å"
            lam_eff_fmt = f"{lam_eff:.0f}"

        if wave_min >= 1e4:
            min_fmt = f"{wave_min / 1e4:.2f}"
            max_fmt = f"{wave_max / 1e4:.2f}"
            range_unit = "μm"
        else:
            min_fmt = f"{wave_min:.0f}"
            max_fmt = f"{wave_max:.0f}"
            range_unit = "Å"

        return f"{name}: λ_eff ~ {lam_eff_fmt} {unit} (range {min_fmt}–{max_fmt} {range_unit})"

    except Exception:
        return f"{name}: (filter found; no summary available)"


def suggest(
    redshift: float,
    coverage: str = "visible_to_nir",
) -> list[str]:
    """Suggest filters covering a rest-frame wavelength range at a redshift.

    Parameters
    ----------
    redshift : float
        Redshift of the source (z >= 0).
    coverage : str
        Rest-frame wavelength coverage preset. Options:
        - "visible": 3500–9000 Å (optical)
        - "visible_to_nir": 3500–25000 Å (optical + near-IR) [default]
        - "uv_to_ir": 1200–50000 Å (UV + optical + IR)
        - "jwst_cover": 6000–50000 Å (rest-frame for JWST epochs)

    Returns
    -------
    list of str
        Filter names with effective wavelength falling within the
        observed-frame span corresponding to the rest-frame coverage.
        Sorted by effective wavelength.

    Raises
    ------
    ValueError
        If coverage is not recognized.

    Notes
    -----
    Observed-frame wavelength is computed as:
    λ_obs = λ_rest * (1 + z).

    Examples
    --------
    >>> suggest(z=3.0, coverage="visible_to_nir")  # z=3 galaxies, optical→NIR
    ['jwst_f115w', 'jwst_f150w', ...]
    """
    # Coverage presets (rest-frame, Angstrom)
    coverage_map = {
        "visible": (3500, 9000),
        "visible_to_nir": (3500, 25000),
        "uv_to_ir": (1200, 50000),
        "jwst_cover": (6000, 50000),
    }

    if coverage not in coverage_map:
        raise ValueError(
            f"Unknown coverage '{coverage}'. Must be one of {list(coverage_map.keys())}."
        )

    lam_rest_min, lam_rest_max = coverage_map[coverage]

    # Convert to observed frame
    lam_obs_min = lam_rest_min * (1 + redshift)
    lam_obs_max = lam_rest_max * (1 + redshift)

    # Load all filters and compute effective wavelengths
    all_names = list_filters()
    if not all_names:
        return []

    # Load all filters; skip any that fail
    wavelengths_by_name = {}
    try:
        for name in all_names:
            try:
                fc = load_filter_set([name])[2][0]
                wave_np = np.asarray(fc.wave)
                trans_np = np.asarray(fc.trans)
                lam_eff = compute_effective_wavelength(wave_np, trans_np)
                wavelengths_by_name[name] = lam_eff
            except Exception:
                # Skip filters that fail to load
                continue
    except Exception:
        return []

    # Find filters within observed-frame span
    matches = [
        name
        for name, lam_eff in wavelengths_by_name.items()
        if lam_obs_min <= lam_eff <= lam_obs_max
    ]

    # Sort by effective wavelength
    matches.sort(key=lambda name: wavelengths_by_name[name])

    return matches

# SPDX-License-Identifier: BSD-3-Clause
"""Emission line catalog for spectroscopic fitting.

Provides a comprehensive registry of emission lines with wavelengths, species
identifiers, doublet relationships, and line classifications. The catalog
determines the static array shapes for the design matrix and amplitude vectors
used in emission line marginalization.

The default catalog includes ~40 optical/NIR lines matching FastSpecFit (DESI),
with the option to load all ~166 lines from CLOUDY grids for comprehensive
UV-to-IR coverage.

References
----------

- Moustakas+2023: FastSpecFit line list (DESI standard)
- Byler+2017: CLOUDY-FSPS nebular emission line predictions
- Storey & Zeippen 2000: [OIII] transition probabilities
- NIST Atomic Spectra Database: doublet ratios

"""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

# ── Doublet ratio constants (primary / secondary) ─────────────────
# key: (primary_name, secondary_name), value: flux ratio primary/secondary

_DOUBLET_RATIOS: dict[tuple[str, str], float] = {
    ("OIII_5007", "OIII_4959"): 2.98,  # [OIII], fixed by transition probabilities
    ("NII_6584", "NII_6548"): 2.94,  # [NII], fixed by transition probabilities
    ("NeV_3426", "NeV_3346"): 1.3,  # [NeV], from transition probabilities
    ("MgII_2803", "MgII_2796"): 1.0,  # MgII, optically thick limit
    ("SIII_9532", "SIII_9069"): 2.47,  # [SIII], fixed by transition probabilities
    # [OII] 3726/3729 and [OII] 7320/7330 are electron-density diagnostics,     # their ratios are
    # NOT fixed by atomic physics and must never be constrained.
    # [SII] 6717/6731 is similarly density-sensitive and is also left unconstrained.
}

# ── Default ~40 line catalog (rest-frame vacuum wavelengths in Angstrom)
# Tuple layout: (name, wavelength_angstrom, species, is_balmer, is_broad_candidate,
#                is_strong, plot_group)
# is_strong and plot_group match FastSpecFit emlines.ecsv (Moustakas+2023).

_DEFAULT_OPTICAL_LINES: list[tuple[str, float, str, bool, bool, bool, str]] = [
    # UV lines (1200-3000 Å)
    ("Lya", 1215.67, "H1", False, False, True, "lya_nv"),
    ("NV_1240", 1240.81, "N5", False, False, True, "lya_nv"),
    ("OI_1304", 1304.35, "O1", False, False, False, "oi_1304"),
    ("SiIV_1396", 1396.76, "Si4", False, False, False, "siliv_1396"),
    ("CIV_1549", 1549.48, "C4", False, True, True, "civ_1549_heii_1640"),
    ("HeII_1640", 1640.42, "He2", False, False, False, "civ_1549_heii_1640"),
    ("AlIII_1857", 1857.40, "Al3", False, False, False, "aliii_1857_siliii_1892_ciii_1908"),
    ("SiIII_1892", 1892.03, "Si3", False, False, True, "aliii_1857_siliii_1892_ciii_1908"),
    ("CIII_1908", 1908.73, "C3", False, True, True, "aliii_1857_siliii_1892_ciii_1908"),
    # constrained secondary of MgII_2803
    ("MgII_2796", 2796.35, "Mg2", False, False, True, "mgii_2796_2803"),
    ("MgII_2803", 2803.53, "Mg2", False, False, True, "mgii_2796_2803"),
    ("NeV_3346", 3347.78, "Ne5", False, False, False, "nev_3346"),
    ("NeV_3426", 3427.94, "Ne5", False, False, False, "nev_3426"),
    # Optical lines (3700-6800 Å)
    ("OII_3726", 3727.09, "O2", False, False, True, "oii_3726_29"),
    ("OII_3729", 3729.88, "O2", False, False, True, "oii_3726_29"),
    ("NeIII_3869", 3870.16, "Ne3", False, False, False, "neiii_3869_h6"),
    ("Hepsilon", 3971.20, "H1", True, True, False, "hepsilon"),
    ("Hdelta", 4102.89, "H1", True, True, False, "hdelta"),
    ("Hgamma", 4341.68, "H1", True, True, True, "hgamma_oiii_4363"),
    ("OIII_4363", 4364.44, "O3", False, False, False, "hgamma_oiii_4363"),
    ("HeI_4471", 4472.74, "He1", False, False, False, "hei_4471"),
    ("HeII_4686", 4687.02, "He2", False, False, False, "heii_4686"),
    ("Hbeta", 4862.68, "H1", True, True, True, "hbeta"),
    ("OIII_4959", 4960.30, "O3", False, False, True, "oiii_doublet"),
    ("OIII_5007", 5008.24, "O3", False, False, True, "oiii_doublet"),
    ("NII_5755", 5756.19, "N2", False, False, False, "nii_5755"),
    ("HeI_5876", 5877.25, "He1", False, False, False, "hei_5876"),
    ("OI_6300", 6302.05, "O1", False, False, False, "oi_6300_siii_6312"),
    ("SIII_6312", 6313.81, "S3", False, False, False, "oi_6300_siii_6312"),
    ("NII_6548", 6549.86, "N2", False, False, True, "halpha_nii_6548_48"),
    ("Halpha", 6564.61, "H1", True, True, True, "halpha_nii_6548_48"),
    ("NII_6584", 6585.28, "N2", False, False, True, "halpha_nii_6548_48"),
    # SII 6717/6731 intentionally unconstrained: ratio is density-sensitive
    ("SII_6717", 6718.29, "S2", False, False, True, "sii_6716_31"),
    ("SII_6731", 6732.67, "S2", False, False, True, "sii_6716_31"),
    # Near-IR lines (7000-10000 Å)
    ("ArIII_7135", 7137.80, "Ar3", False, False, False, "ariii_7135_oii_7320_30"),
    ("OII_7320", 7321.47, "O2", False, False, False, "ariii_7135_oii_7320_30"),
    ("OII_7330", 7332.21, "O2", False, False, False, "ariii_7135_oii_7320_30"),
    ("SIII_9069", 9071.10, "S3", False, False, False, "siii_9069"),
    ("SIII_9532", 9533.23, "S3", False, False, False, "siii_9532"),
]

# ── 13-line catalog matching eline_marginalization.py
# Same wavelengths and insertion order as DEFAULT_LINE_WAVELENGTHS there.

_DEFAULT_13_LINES: list[tuple[str, float, str, bool, bool, bool, str]] = [
    ("Lya", 1215.67, "H1", False, False, True, "lya_nv"),
    ("Hdelta", 4102.89, "H1", True, True, False, "hdelta"),
    ("Hgamma", 4341.68, "H1", True, True, True, "hgamma_oiii_4363"),
    ("Hbeta", 4862.68, "H1", True, True, True, "hbeta"),
    ("OIII_4959", 4960.30, "O3", False, False, True, "oiii_doublet"),
    ("OIII_5007", 5008.24, "O3", False, False, True, "oiii_doublet"),
    ("Halpha", 6564.61, "H1", True, True, True, "halpha_nii_6548_48"),
    ("NII_6548", 6549.86, "N2", False, False, True, "halpha_nii_6548_48"),
    ("NII_6584", 6585.28, "N2", False, False, True, "halpha_nii_6548_48"),
    ("OII_3726", 3727.09, "O2", False, False, True, "oii_3726_29"),
    ("OII_3729", 3729.88, "O2", False, False, True, "oii_3726_29"),
    ("SII_6717", 6718.29, "S2", False, False, True, "sii_6716_31"),
    ("SII_6731", 6732.67, "S2", False, False, True, "sii_6716_31"),
]


# ── Data structures ───────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class DoubletConstraint:
    """Constraint linking a secondary line's flux to a primary line.

    Parameters
    ----------
    primary_idx: int
        Index of the primary (brighter) line in the catalog.
    secondary_idx: int
        Index of the secondary (fainter) line in the catalog.
    ratio: float
        Flux ratio primary/secondary, so flux_secondary = flux_primary / ratio.

    """

    primary_idx: int
    secondary_idx: int
    ratio: float


@dataclasses.dataclass(frozen=True)
class LineList:
    """Immutable registry of emission lines for spectroscopic fitting.

    Parameters
    ----------
    names: tuple[str, ...]
        Line identifiers (e.g. ``"Halpha"``, ``"OIII_5007"``).
    wavelengths: jnp.ndarray
        Rest-frame vacuum wavelengths in Angstrom, shape ``(n_lines,)``.
    species: tuple[str, ...]
        Chemical species codes matching CLOUDY convention (e.g. ``"H1"``,
        ``"O3"``, ``"N2"``).
    doublets: tuple[DoubletConstraint, ...]
        Doublet flux-ratio constraints between paired lines.
    is_balmer: tuple[bool, ...]
        True for hydrogen Balmer/Lyman series lines.
    is_broad_candidate: tuple[bool, ...]
        True for lines that can carry a broad AGN component.
    is_strong: tuple[bool, ...]
        True for lines reliably detected in DESI spectra. Matches the
        ``isstrong`` column in FastSpecFit ``emlines.ecsv`` [1]_.
    plot_group: tuple[str, ...]
        QA plot group label. Lines sharing a group are displayed together in
        diagnostic panels. Matches the ``plotgroup`` column in FastSpecFit
        ``emlines.ecsv`` [1]_ (e.g. ``"halpha_nii_6548_48"``,
        ``"oiii_doublet"``).

    Returns
    -------
    LineList
        Immutable line catalog instance.

    Attributes
    ----------
    names: tuple[str, ...]
        Line identifiers.
    wavelengths: ndarray, shape (n_lines,)
        Rest-frame vacuum wavelengths [Angstrom].
    species: tuple[str, ...]
        CLOUDY species codes.
    doublets: tuple[DoubletConstraint, ...]
        Doublet constraints.
    is_balmer: tuple[bool, ...]
        Balmer/Lyman series flags.
    is_broad_candidate: tuple[bool, ...]
        Broad component candidate flags.
    is_strong: tuple[bool, ...]
        Strong line detection flags (DESI context).
    plot_group: tuple[str, ...]
        QA plot group labels.

    Notes
    -----
    **Immutable container**: A frozen dataclass. All fields are read-only.

    **Wavelength convention**: All wavelengths are vacuum (not air) and
    rest-frame (not observed-frame).

    References
    ----------
    .. [1] Moustakas, J., Scholte, D., Dey, B., Khederlarian, A., 2023,
           "FastSpecFit: Fast spectral synthesis and emission-line fitting
           of DESI spectra", Astrophysics Source Code Library,
           record ascl:2308.005.
           https://ui.adsabs.harvard.edu/abs/2023ascl.soft08005M

    Examples
    --------
    Build the default DESI-like catalog and select BPT lines::

        cat = LineList.default_optical()
        bpt = cat.select(names=["Halpha", "Hbeta", "OIII_5007", "NII_6584"])
        C = bpt.build_constraint_matrix()  # (n_lines, n_independent)

    Inspect strong-line detections::

        strong_names = [n for n, s in zip(cat.names, cat.is_strong) if s]
        # ['Lya', 'NV_1240', 'CIV_1549', ...]

    Find lines in the Halpha+NII panel::

        panel = [n for n, g in zip(cat.names, cat.plot_group) if g == "halpha_nii_6548_48"]
        # ['NII_6548', 'Halpha', 'NII_6584']

    Use a custom wavelength subset from a line-finding step::

        detected = cat.select(wavelengths=[6564.61, 4862.68, 5008.24])

    """

    names: tuple[str, ...]
    wavelengths: jnp.ndarray = dataclasses.field(hash=False)
    species: tuple[str, ...]
    doublets: tuple[DoubletConstraint, ...]
    is_balmer: tuple[bool, ...]
    is_broad_candidate: tuple[bool, ...]
    is_strong: tuple[bool, ...]
    plot_group: tuple[str, ...]

    # ── Properties ────────────────────────────────────────────────

    @property
    def n_lines(self) -> int:
        r"""Total number of lines in the catalog.

        Returns
        -------
        int
            Number of emission lines.

        Notes
        -----
        Computed from the length of the ``names`` tuple. Constant for
        the lifetime of the object (immutable).

        """
        return len(self.names)

    @property
    def n_independent(self) -> int:
        r"""Number of independent amplitude parameters after doublet constraints.

        Returns
        -------
        int
            Number of independent parameters (total lines minus doublet secondaries).

        Notes
        -----
        Computed as ``n_lines - len(doublets)``, since each doublet constraint
        removes one degree of freedom.

        """
        return self.n_lines - len(self.doublets)

    @property
    def independent_wavelengths(self) -> jnp.ndarray:
        r"""Wavelengths of the independent (non-constrained) amplitude parameters.

        After applying doublet constraints, the design matrix has
        ``n_independent`` columns, one per independent amplitude. This property
        returns the wavelength of each independent parameter's primary line,
        in column order matching ``build_constraint_matrix()``.

        Returns
        -------
        ndarray, shape (n_independent,)
            Rest-frame vacuum wavelengths in Angstrom for independent amplitude columns.

        Notes
        -----
        The returned wavelengths correspond to the columns of the constraint matrix
        returned by :func:`build_constraint_matrix`. Column order is the same as
        the order of non-constrained lines in the original catalog.

        """
        secondary_indices = {dc.secondary_idx for dc in self.doublets}
        kept = [i for i in range(self.n_lines) if i not in secondary_indices]
        return self.wavelengths[jnp.array(kept)]

    # ── Factory methods ───────────────────────────────────────────

    @classmethod
    def default_optical(cls) -> LineList:
        r"""FastSpecFit-equivalent ~40-line catalog for optical/NIR spectroscopy.

        Returns
        -------
        LineList
            Catalog of ~40 lines sorted by wavelength [Angstrom] with
            doublet constraints auto-detected.

        Notes
        -----
        Not JIT-compatible (uses Python sorting and class method).

        Includes UV lines (1200–3000 Å), optical (3700–6800 Å), and
        near-infrared lines (7000–10000 Å). Line properties and doublet
        constraints follow the FastSpecFit standard (Moustakas et al. 2023).

        """
        lines = sorted(_DEFAULT_OPTICAL_LINES, key=lambda t: t[1])
        return cls._from_line_tuples(lines)

    @classmethod
    def default_13(cls) -> LineList:
        r"""Backward-compatible 13-line catalog (legacy default).

        Uses the same wavelengths as ``DEFAULT_LINE_WAVELENGTHS`` in
        ``eline_marginalization.py``. Insertion order is preserved for
        index-based lookups.

        Returns
        -------
        LineList
            13-line catalog with doublet constraints where applicable.

        Notes
        -----
        Not JIT-compatible (uses Python class method). Kept for
        backward compatibility with eline_marginalization.py.

        Includes Lyman alpha, Balmer series, key optical lines, and commonly
        detected forbidden lines. Suitable for quick analyses where a compact
        line set is preferred over the full optical catalog.

        """
        return cls._from_line_tuples(_DEFAULT_13_LINES)

    @classmethod
    def from_cloudy_grid(cls, filepath: str) -> LineList:
        r"""Load all lines from a CLOUDY HDF5 grid file.

        Reads ``lines/names`` and ``lines/wavelength`` datasets, parses
        species from CLOUDY naming conventions, and auto-detects doublets
        by species + wavelength proximity (< 20 Angstrom).

        Parameters
        ----------
        filepath: str
            Path to the CLOUDY HDF5 grid file containing ``lines/names`` and
            ``lines/wavelength`` datasets.

        Returns
        -------
        LineList
            Catalog populated from the CLOUDY grid, sorted by wavelength
            [Angstrom]. Doublet constraints are auto-detected by species
            and proximity.

        Raises
        ------
        ImportError
            If ``h5py`` is not installed.
        OSError
            If the file cannot be opened or read.
        KeyError
            If required datasets (``lines/names``, ``lines/wavelength``) are missing.

        Notes
        -----
        Not JIT-compatible (uses Python file I/O and h5py).

        Parses CLOUDY species names (e.g., ``"H  1"``, ``"O  3"``) into
        compact codes (``"H1"``, ``"O3"``). Doublet detection requires
        same species and wavelength separation < 20 Å.

        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to load CLOUDY grid files. Install with: pip install h5py"
            ) from exc

        with h5py.File(filepath, "r") as f:
            raw_names = [n.decode() if isinstance(n, bytes) else n for n in f["lines/names"][:]]
            raw_waves = f["lines/wavelength"][:]

        # Sort by wavelength
        order = sorted(range(len(raw_names)), key=lambda i: raw_waves[i])
        sorted_names = [raw_names[i] for i in order]
        sorted_waves = [raw_waves[i] for i in order]

        parsed_species = [_parse_cloudy_species(n) for n in sorted_names]

        names_tuple = tuple(sorted_names)
        species_tuple = tuple(parsed_species)
        waves_arr = jnp.array(sorted_waves)

        # Detect doublets by species + wavelength proximity (< 20 Å apart)
        doublets = _detect_doublets_by_proximity(
            names_tuple, sorted_waves, species_tuple, proximity_angstrom=20.0
        )

        is_balmer = tuple(_is_balmer_line(nm, sp) for nm, sp in zip(names_tuple, species_tuple))
        is_broad = tuple(_is_broad_candidate(nm, sp) for nm, sp in zip(names_tuple, species_tuple))

        return cls(
            names=names_tuple,
            wavelengths=waves_arr,
            species=species_tuple,
            doublets=doublets,
            is_balmer=is_balmer,
            is_broad_candidate=is_broad,
            is_strong=tuple(False for _ in names_tuple),
            plot_group=tuple("" for _ in names_tuple),
        )

    @classmethod
    def from_names(cls, names: Sequence[str]) -> LineList:
        r"""Construct a LineList from a list of line names.

        Uses the default optical catalog and selects only the requested
        line names. Raises ``ValueError`` if any name is unknown.

        Parameters
        ----------
        names: sequence of str
            Line identifiers to select (e.g., ``["Halpha", "OIII_5007"]``).

        Returns
        -------
        LineList
            A new LineList containing only the requested lines, in wavelength order.

        Raises
        ------
        ValueError
            If any name in ``names`` is not found in the default optical catalog.

        Notes
        -----
        Not JIT-compatible (uses Python class methods and filtering).

        This is the recommended way to construct a LineList from line names
        for schema declarations::

            obs = Observation(photometry=..., lines=LineList.from_names(["Halpha", "OIII_5007"]))

        Examples
        --------
        Select a few lines for a BPT diagram::

            ll = LineList.from_names(["Halpha", "Hbeta", "OIII_5007", "NII_6584"])
            assert ll.n_lines == 4

        """
        return cls.default_optical().select(names=names)

    # ── Filtering ─────────────────────────────────────────────────

    def select(
        self,
        wave_min: float = 0.0,
        wave_max: float = float("inf"),
        species: Sequence[str] | None = None,
        names: Sequence[str] | None = None,
        wavelengths: Sequence[float] | None = None,
    ) -> LineList:
        """Return a filtered copy of the catalog.

        Parameters
        ----------
        wave_min: float, optional
            Minimum rest-frame wavelength [Angstrom] (inclusive). Default: 0.
        wave_max: float, optional
            Maximum rest-frame wavelength [Angstrom] (inclusive).
            Default: infinity.
        species: sequence of str, optional
            If given, retain only lines whose species code is in this list.
            Default: ``None`` (no filtering).
        names: sequence of str, optional
            If given, retain only lines whose name exactly matches one in this list.
            Default: ``None`` (no filtering).
        wavelengths: sequence of float, optional
            If given, match each wavelength to the nearest line in the catalog
            within 5 Angstrom tolerance. Default: ``None`` (no filtering).

        Returns
        -------
        LineList
            New catalog containing only lines satisfying all given criteria (AND logic).
            Doublet constraints are rebuilt with updated indices; constraints
            where either member is filtered out are dropped.

        Raises
        ------
        ValueError
            If any name in ``names`` is not found in the catalog, or if any
            wavelength in ``wavelengths`` cannot be matched within 5 Angstrom.

        Notes
        -----
        Not JIT-compatible (uses Python control flow and set operations).

        All filtering criteria are combined via AND logic. If all criteria
        filter out all lines, returns an empty LineList.

        Examples
        --------
        Select BPT diagram lines by name::

            cat = LineList.default_optical()
            bpt = cat.select(names=["Halpha", "Hbeta", "OIII_5007", "NII_6584"])

        Select by observed wavelengths (e.g. from a line-finding algorithm)::

            bpt = cat.select(wavelengths=[6564.61, 4862.68, 5008.24, 6585.28])

        Select only optical window::

            optical = cat.select(wave_min=3700.0, wave_max=7000.0)

        Select hydrogen lines only::

            hydrogen = cat.select(species=["H1"])

        """
        # Guard the positional numeric args. The primary discovery use of this
        # method, pick lines by name, sits behind two positional wavelength
        # bounds, so ``select(["Halpha", "Hbeta"])`` (the natural analogy to
        # ``Photometry.from_names([...])``) silently binds the list to
        # ``wave_min`` and later crashes deep in ``wave_min <= w`` with an
        # opaque ``'>' not supported between 'list' and 'float'``. Name the fix.
        for _arg, _val in (("wave_min", wave_min), ("wave_max", wave_max)):
            if not isinstance(_val, numbers.Real):
                raise TypeError(
                    f"LineList.select() {_arg} must be a wavelength [Angstrom], got "
                    f"{type(_val).__name__}. To keep specific lines, pass them by "
                    f"keyword: select(names=[...]) or select(species=[...]); to keep "
                    f"a wavelength window, use select(wave_min=..., wave_max=...)."
                )

        waves_np = [float(w) for w in self.wavelengths]

        # Start with all indices
        kept_indices: set[int] = set(range(self.n_lines))

        # Filter by wavelength range
        if wave_min > 0.0 or wave_max < float("inf"):
            kept_indices &= {i for i, w in enumerate(waves_np) if wave_min <= w <= wave_max}

        # Filter by species
        if species is not None:
            species_set = set(species)
            kept_indices &= {i for i, sp in enumerate(self.species) if sp in species_set}

        # Filter by exact line names
        if names is not None:
            names_set = set(names)
            # Verify all requested names exist
            catalog_names = set(self.names)
            missing_names = names_set - catalog_names
            if missing_names:
                raise ValueError(
                    f"Names not found in catalog: {sorted(missing_names)}. "
                    f"Available names: {sorted(catalog_names)}"
                )
            kept_indices &= {i for i, nm in enumerate(self.names) if nm in names_set}

        # Filter by wavelength matching (nearest within 5 Å)
        if wavelengths is not None:
            matched_indices: set[int] = set()
            for target_wave in wavelengths:
                # Find nearest line within 5 Angstrom tolerance
                distances = [abs(w - target_wave) for w in waves_np]
                nearest_idx = min(range(len(distances)), key=lambda i: distances[i])
                nearest_dist = distances[nearest_idx]

                if nearest_dist > 5.0:
                    raise ValueError(
                        f"No line within 5 Angstrom of {target_wave} Å. "
                        f"Nearest line: {self.names[nearest_idx]} at "
                        f"{waves_np[nearest_idx]} Å ({nearest_dist:.2f} Å away)."
                    )
                matched_indices.add(nearest_idx)
            kept_indices &= matched_indices

        if not kept_indices:
            return LineList(
                names=(),
                wavelengths=jnp.array([]),
                species=(),
                doublets=(),
                is_balmer=(),
                is_broad_candidate=(),
                is_strong=(),
                plot_group=(),
            )

        # Sort kept indices for consistent ordering
        kept_indices_list = sorted(kept_indices)

        # Build old -> new index mapping
        old_to_new: dict[int, int] = {old: new for new, old in enumerate(kept_indices_list)}

        new_names = tuple(self.names[i] for i in kept_indices_list)
        new_waves = jnp.array([waves_np[i] for i in kept_indices_list])
        new_species = tuple(self.species[i] for i in kept_indices_list)
        new_is_balmer = tuple(self.is_balmer[i] for i in kept_indices_list)
        new_is_broad = tuple(self.is_broad_candidate[i] for i in kept_indices_list)
        new_is_strong = tuple(self.is_strong[i] for i in kept_indices_list)
        new_plot_group = tuple(self.plot_group[i] for i in kept_indices_list)

        # Rebuild doublets, keep only if both members survived the filter
        new_doublets: list[DoubletConstraint] = []
        for dc in self.doublets:
            if dc.primary_idx in old_to_new and dc.secondary_idx in old_to_new:
                new_doublets.append(
                    DoubletConstraint(
                        primary_idx=old_to_new[dc.primary_idx],
                        secondary_idx=old_to_new[dc.secondary_idx],
                        ratio=dc.ratio,
                    )
                )

        return LineList(
            names=new_names,
            wavelengths=new_waves,
            species=new_species,
            doublets=tuple(new_doublets),
            is_balmer=new_is_balmer,
            is_broad_candidate=new_is_broad,
            is_strong=new_is_strong,
            plot_group=new_plot_group,
        )

    # ── Constraint matrix ─────────────────────────────────────────

    def build_constraint_matrix(self) -> jnp.ndarray:
        r"""Build the (n_lines, n_independent) constraint matrix C.

        The constraint matrix maps from independent amplitudes to all line
        amplitudes: ``flux = C @ a_independent``.

        Returns
        -------
        ndarray, shape (n_lines, n_independent)
            Linear constraint matrix. Each row represents one emission line;
            each column represents one independent amplitude parameter.
            For constrained secondary lines, the row encodes the constraint
            relationship; for primary lines, the row has a unit entry.

        Notes
        -----
        Not JIT-compatible (uses NumPy array mutation and slicing).

        For each doublet constraint with ``flux_secondary = flux_primary / ratio``,
        the corresponding column is removed and the secondary line's row encodes
        the constraint via an off-diagonal entry ``1 / ratio`` in the primary
        column. This ensures that fitting ``n_independent`` amplitudes
        automatically enforces all doublet ratio constraints via matrix
        multiplication: ``a_full = C @ a_independent``.

        """
        n = self.n_lines
        # Build as a regular numpy array for mutation, then convert
        mat = np.eye(n, dtype=float)

        constrained_cols: set[int] = set()
        for dc in self.doublets:
            j = dc.secondary_idx
            i = dc.primary_idx
            # Zero out column j (secondary)
            mat[:, j] = 0.0
            # Set secondary flux = primary flux / ratio
            mat[j, i] = 1.0 / dc.ratio
            constrained_cols.add(j)

        # Keep only independent columns (non-constrained secondaries)
        independent_cols = [c for c in range(n) if c not in constrained_cols]
        result = mat[:, independent_cols]
        return jnp.array(result)

    # ── Internal helpers ──────────────────────────────────────────

    @classmethod
    def _from_line_tuples(
        cls,
        lines: list[tuple[str, float, str, bool, bool, bool, str]],
    ) -> LineList:
        """Construct a LineList from a list of line property tuples.

        Doublet constraints are auto-detected from ``_DOUBLET_RATIOS`` by
        matching line names in the list.

        Parameters
        ----------
        lines: list[tuple]
            Each tuple: ``(name, wavelength [Angstrom], species, is_balmer,
            is_broad_candidate, is_strong, plot_group)``.

        Returns
        -------
        LineList
            Catalog with doublet constraints auto-detected from the global
            ``_DOUBLET_RATIOS`` dictionary.

        Notes
        -----
        Private helper. Not JIT-compatible (uses Python list operations).

        Constraints are only added if both the primary and secondary lines
        from ``_DOUBLET_RATIOS`` are present in the input list.

        """
        names_list = [t[0] for t in lines]
        name_to_idx: dict[str, int] = {n: i for i, n in enumerate(names_list)}

        doublets: list[DoubletConstraint] = []
        for (primary_name, secondary_name), ratio in _DOUBLET_RATIOS.items():
            if primary_name in name_to_idx and secondary_name in name_to_idx:
                doublets.append(
                    DoubletConstraint(
                        primary_idx=name_to_idx[primary_name],
                        secondary_idx=name_to_idx[secondary_name],
                        ratio=ratio,
                    )
                )

        return cls(
            names=tuple(names_list),
            wavelengths=jnp.array([t[1] for t in lines]),
            species=tuple(t[2] for t in lines),
            doublets=tuple(doublets),
            is_balmer=tuple(t[3] for t in lines),
            is_broad_candidate=tuple(t[4] for t in lines),
            is_strong=tuple(t[5] for t in lines),
            plot_group=tuple(t[6] for t in lines),
        )


# ── Private helpers for CLOUDY parsing ────────────────────────────


def _parse_cloudy_species(name: str) -> str:
    """Extract species code from a CLOUDY line name.

    CLOUDY names follow conventions like ``"H  1 1215.67A"``,
    ``"O  3 5006.84A"``, or simple names like ``"Halpha"``.
    Returns a compact species code such as ``"H1"``, ``"O3"``.

    Parameters
    ----------
    name: str
        CLOUDY line name string.

    Returns
    -------
    str
        Compact species code.

    """
    parts = name.strip().split()
    if len(parts) >= 2:
        element = parts[0].strip()
        ionization = parts[1].strip()
        if ionization.isdigit():
            return f"{element}{ionization}"
    # Fall back to the raw name if no standard pattern found
    return name.split()[0] if name.strip() else name


def _is_balmer_line(name: str, species: str) -> bool:
    """Heuristic: True if the line is a hydrogen Balmer/Lyman series member.

    For H1 species, checks both keyword matching (e.g. "Halpha", "Lya") and
    wavelength range for CLOUDY-style names (e.g. "H  1 4101.67A").
    Balmer series: 3646–6563 Å. Lyman series: 912–1216 Å.
    """
    if species != "H1":
        return False

    # Try simple keyword match first
    balmer_keywords = ("alpha", "beta", "gamma", "delta", "epsilon", "ha", "hb")
    name_lower = name.lower()
    if any(kw in name_lower for kw in balmer_keywords):
        return True

    # For CLOUDY-style names, extract wavelength and check range
    parts = name.strip().split()
    if len(parts) >= 3:
        try:
            # CLOUDY format: "H  1 1215.67A" -> parts[2] = "1215.67A"
            wave_str = parts[2].rstrip("A").rstrip("a")
            wave = float(wave_str)
            # Balmer series (3646–6563 Å) or Lyman series (912–1216 Å)
            return (3646 <= wave <= 6563) or (912 <= wave <= 1216)
        except (ValueError, IndexError):
            pass

    return False


def _is_broad_candidate(name: str, species: str) -> bool:
    """Heuristic: True if this line can carry an AGN broad component."""
    broad_species = {"H1", "C4", "Mg2", "C3"}
    if species in broad_species:
        return True
    broad_keywords = ("lya", "civ", "mgii", "ciii", "halpha", "hbeta", "hgamma", "hdelta")
    name_lower = name.lower()
    return any(kw in name_lower for kw in broad_keywords)


def _get_doublet_ratio_by_wavelength(primary_wave: float, secondary_wave: float) -> float:
    """Look up doublet ratio by wavelength proximity to known pairs (within 5 Å)."""
    for (pname, sname), ratio in _DOUBLET_RATIOS.items():
        pwave = next((t[1] for t in _DEFAULT_OPTICAL_LINES if t[0] == pname), None)
        swave = next((t[1] for t in _DEFAULT_OPTICAL_LINES if t[0] == sname), None)
        if (
            pwave is not None
            and swave is not None
            and abs(primary_wave - pwave) < 5.0
            and abs(secondary_wave - swave) < 5.0
        ):
            return ratio
    return 1.0


def _detect_doublets_by_proximity(
    names: tuple[str, ...],
    wavelengths: list[float],
    species: tuple[str, ...],
    proximity_angstrom: float = 20.0,
) -> tuple[DoubletConstraint, ...]:
    """Detect doublet pairs from CLOUDY grids by species + wavelength proximity.

    Pairs lines of the same species that are closer than ``proximity_angstrom``
    Angstrom apart. The brighter line (longer wavelength, heuristic) is the
    primary. Ratio is set to 1.0 (unknown) unless overridden by ``_DOUBLET_RATIOS``.

    Parameters
    ----------
    names: tuple of str
    wavelengths: list of float
    species: tuple of str
    proximity_angstrom: float, optional

    Returns
    -------
    tuple of DoubletConstraint

    """
    doublets: list[DoubletConstraint] = []
    used: set[int] = set()

    for i in range(len(names)):
        if i in used:
            continue
        for j in range(i + 1, len(names)):
            if j in used:
                continue
            if species[i] != species[j]:
                continue
            if abs(wavelengths[i] - wavelengths[j]) > proximity_angstrom:
                continue
            # Primary is the one with the longer wavelength (heuristic)
            primary_idx, secondary_idx = (i, j) if wavelengths[i] > wavelengths[j] else (j, i)
            # Check if there is a known ratio
            ratio = _get_doublet_ratio_by_wavelength(
                wavelengths[primary_idx], wavelengths[secondary_idx]
            )
            doublets.append(
                DoubletConstraint(
                    primary_idx=primary_idx,
                    secondary_idx=secondary_idx,
                    ratio=ratio,
                )
            )
            used.add(i)
            used.add(j)
            break

    return tuple(doublets)


# ── Deprecated alias, removed in tengri v1.0 ─────────────────────

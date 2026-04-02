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
from collections.abc import Sequence

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Doublet ratio constants (primary / secondary)
# key: (primary_name, secondary_name), value: flux ratio primary/secondary
# ---------------------------------------------------------------------------

_DOUBLET_RATIOS: dict[tuple[str, str], float] = {
    ("OIII_5007", "OIII_4959"): 2.98,  # [OIII] — from transition probabilities
    ("NII_6584", "NII_6548"): 2.94,  # [NII] — from transition probabilities
    ("OII_3729", "OII_3726"): 1.3,  # [OII] — density-dependent, fix to typical
    ("NeV_3426", "NeV_3346"): 1.3,  # [NeV]
    ("OII_7330", "OII_7320"): 1.0,  # [OII] NIR doublet
    ("SIII_9532", "SIII_9069"): 2.47,  # [SIII]
}

# ---------------------------------------------------------------------------
# Default ~40 line catalog (rest-frame vacuum wavelengths in Angstrom)
# Tuple layout: (name, wavelength_angstrom, species, is_balmer, is_broad_candidate)
# ---------------------------------------------------------------------------

_DEFAULT_OPTICAL_LINES: list[tuple[str, float, str, bool, bool]] = [
    # UV lines (1200-3000 Å)
    ("Lya", 1215.67, "H1", True, True),
    ("NV_1240", 1240.81, "N5", False, False),
    ("OI_1304", 1304.35, "O1", False, False),
    ("SiIV_1396", 1396.76, "Si4", False, False),
    ("CIV_1549", 1549.48, "C4", False, True),
    ("HeII_1640", 1640.42, "He2", False, False),
    ("AlIII_1857", 1857.40, "Al3", False, False),
    ("SiIII_1892", 1892.03, "Si3", False, False),
    ("CIII_1908", 1908.73, "C3", False, True),
    ("MgII_2796", 2796.35, "Mg2", False, True),
    ("MgII_2803", 2803.53, "Mg2", False, True),
    ("NeV_3346", 3346.79, "Ne5", False, False),
    ("NeV_3426", 3426.85, "Ne5", False, False),
    # Optical lines (3700-6800 Å)
    ("OII_3726", 3726.03, "O2", False, False),
    ("OII_3729", 3728.82, "O2", False, False),
    ("NeIII_3869", 3869.86, "Ne3", False, False),
    ("Hepsilon", 3970.07, "H1", True, True),
    ("Hdelta", 4101.73, "H1", True, True),
    ("Hgamma", 4340.46, "H1", True, True),
    ("OIII_4363", 4363.21, "O3", False, False),
    ("HeI_4471", 4471.48, "He1", False, False),
    ("HeII_4686", 4685.71, "He2", False, False),
    ("Hbeta", 4861.33, "H1", True, True),
    ("OIII_4959", 4958.91, "O3", False, False),
    ("OIII_5007", 5006.84, "O3", False, False),
    ("NII_5755", 5755.08, "N2", False, False),
    ("HeI_5876", 5875.66, "He1", False, False),
    ("OI_6300", 6300.30, "O1", False, False),
    ("SIII_6312", 6312.06, "S3", False, False),
    ("NII_6548", 6548.05, "N2", False, False),
    ("Halpha", 6562.80, "H1", True, True),
    ("NII_6584", 6583.45, "N2", False, False),
    ("SII_6717", 6716.44, "S2", False, False),
    ("SII_6731", 6730.81, "S2", False, False),
    # Near-IR lines (7000-10000 Å)
    ("ArIII_7135", 7135.78, "Ar3", False, False),
    ("OII_7320", 7319.99, "O2", False, False),
    ("OII_7330", 7330.73, "O2", False, False),
    ("SIII_9069", 9068.60, "S3", False, False),
    ("SIII_9532", 9531.10, "S3", False, False),
]

# ---------------------------------------------------------------------------
# 13-line backward-compatible catalog matching eline_marginalization.py
# Same wavelengths and insertion order as DEFAULT_LINE_WAVELENGTHS there.
# ---------------------------------------------------------------------------

_DEFAULT_13_LINES: list[tuple[str, float, str, bool, bool]] = [
    ("Lya", 1215.67, "H1", True, True),
    ("Hdelta", 4101.73, "H1", True, True),
    ("Hgamma", 4340.46, "H1", True, True),
    ("Hbeta", 4861.33, "H1", True, True),
    ("OIII_4959", 4958.91, "O3", False, False),
    ("OIII_5007", 5006.84, "O3", False, False),
    ("Halpha", 6562.80, "H1", True, True),
    ("NII_6548", 6548.05, "N2", False, False),
    ("NII_6584", 6583.45, "N2", False, False),
    ("OII_3726", 3726.03, "O2", False, False),
    ("OII_3729", 3728.82, "O2", False, False),
    ("SII_6717", 6716.44, "S2", False, False),
    ("SII_6731", 6730.81, "S2", False, False),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DoubletConstraint:
    """Constraint linking a secondary line's flux to a primary line.

    Parameters
    ----------
    primary_idx : int
        Index of the primary (brighter) line in the catalog.
    secondary_idx : int
        Index of the secondary (fainter) line in the catalog.
    ratio : float
        Flux ratio primary/secondary, so flux_secondary = flux_primary / ratio.
    """

    primary_idx: int
    secondary_idx: int
    ratio: float


@dataclasses.dataclass(frozen=True)
class LineCatalog:
    """Immutable registry of emission lines for spectroscopic fitting.

    Parameters
    ----------
    names : tuple[str, ...]
        Line identifiers (e.g. ``"Halpha"``, ``"OIII_5007"``).
    wavelengths : jnp.ndarray
        Rest-frame vacuum wavelengths in Angstrom, shape ``(n_lines,)``.
    species : tuple[str, ...]
        Chemical species codes matching CLOUDY convention (e.g. ``"H1"``,
        ``"O3"``, ``"N2"``).
    doublets : tuple[DoubletConstraint, ...]
        Doublet flux-ratio constraints between paired lines.
    is_balmer : tuple[bool, ...]
        True for hydrogen Balmer/Lyman series lines.
    is_broad_candidate : tuple[bool, ...]
        True for lines that can carry a broad AGN component.
    """

    names: tuple[str, ...]
    wavelengths: jnp.ndarray = dataclasses.field(hash=False)
    species: tuple[str, ...]
    doublets: tuple[DoubletConstraint, ...]
    is_balmer: tuple[bool, ...]
    is_broad_candidate: tuple[bool, ...]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_lines(self) -> int:
        """Total number of lines in the catalog."""
        return len(self.names)

    @property
    def n_independent(self) -> int:
        """Number of independent amplitude parameters after doublet constraints."""
        return self.n_lines - len(self.doublets)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def default_optical(cls) -> LineCatalog:
        """FastSpecFit-equivalent ~40-line catalog for optical/NIR spectroscopy.

        Returns
        -------
        LineCatalog
            Catalog of ~40 lines sorted by wavelength with doublet constraints
            auto-detected from ``_DOUBLET_RATIOS``.
        """
        lines = sorted(_DEFAULT_OPTICAL_LINES, key=lambda t: t[1])
        return cls._from_line_tuples(lines)

    @classmethod
    def default_13(cls) -> LineCatalog:
        """Backward-compatible 13-line catalog (existing default).

        Uses the same wavelengths as ``DEFAULT_LINE_WAVELENGTHS`` in
        ``eline_marginalization.py``. The insertion order is preserved so
        that index-based lookups remain valid.

        Lines
        -----
        Lya (1215.67), Hdelta (4101.73), Hgamma (4340.46), Hbeta (4861.33),
        OIII_4959 (4958.91), OIII_5007 (5006.84), Halpha (6562.80),
        NII_6548 (6548.05), NII_6584 (6583.45), OII_3726 (3726.03),
        OII_3729 (3728.82), SII_6717 (6716.44), SII_6731 (6730.81).

        Returns
        -------
        LineCatalog
            13-line catalog with doublet constraints where applicable.
        """
        return cls._from_line_tuples(_DEFAULT_13_LINES)

    @classmethod
    def from_cloudy_grid(cls, filepath: str) -> LineCatalog:
        """Load all lines from a CLOUDY HDF5 grid file.

        Reads ``lines/names`` and ``lines/wavelength`` datasets, parses
        species from CLOUDY naming conventions, and auto-detects doublets
        by species + wavelength proximity.

        Parameters
        ----------
        filepath : str
            Path to the CLOUDY HDF5 grid file.

        Returns
        -------
        LineCatalog
            Catalog populated from the CLOUDY grid.

        Raises
        ------
        ImportError
            If ``h5py`` is not installed.
        KeyError
            If the HDF5 file does not contain expected datasets.
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
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def select(
        self,
        wave_min: float = 0.0,
        wave_max: float = float("inf"),
        species: Sequence[str] | None = None,
    ) -> LineCatalog:
        """Return a filtered copy of the catalog.

        Parameters
        ----------
        wave_min : float, optional
            Minimum rest-frame wavelength in Angstrom (inclusive).
        wave_max : float, optional
            Maximum rest-frame wavelength in Angstrom (inclusive).
        species : sequence of str, optional
            If given, retain only lines whose species code is in this list.

        Returns
        -------
        LineCatalog
            New catalog containing only lines satisfying the given criteria.
            Doublet constraints are rebuilt with updated indices; constraints
            where either member is filtered out are dropped.
        """
        waves_np = [float(w) for w in self.wavelengths]
        keep_set = set(species) if species is not None else None

        kept_indices = [
            i
            for i, (w, sp) in enumerate(zip(waves_np, self.species))
            if wave_min <= w <= wave_max and (keep_set is None or sp in keep_set)
        ]

        if not kept_indices:
            return LineCatalog(
                names=(),
                wavelengths=jnp.array([]),
                species=(),
                doublets=(),
                is_balmer=(),
                is_broad_candidate=(),
            )

        # Build old -> new index mapping
        old_to_new: dict[int, int] = {old: new for new, old in enumerate(kept_indices)}

        new_names = tuple(self.names[i] for i in kept_indices)
        new_waves = jnp.array([waves_np[i] for i in kept_indices])
        new_species = tuple(self.species[i] for i in kept_indices)
        new_is_balmer = tuple(self.is_balmer[i] for i in kept_indices)
        new_is_broad = tuple(self.is_broad_candidate[i] for i in kept_indices)

        # Rebuild doublets — keep only if both members survived the filter
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

        return LineCatalog(
            names=new_names,
            wavelengths=new_waves,
            species=new_species,
            doublets=tuple(new_doublets),
            is_balmer=new_is_balmer,
            is_broad_candidate=new_is_broad,
        )

    # ------------------------------------------------------------------
    # Constraint matrix
    # ------------------------------------------------------------------

    def build_constraint_matrix(self) -> jnp.ndarray:
        """Build the (n_lines, n_independent) constraint matrix C.

        The constraint matrix maps from independent amplitudes to all line
        amplitudes: ``flux = C @ a_independent``.

        Algorithm
        ---------
        1. Start with identity ``(n_lines, n_lines)``.
        2. For each ``DoubletConstraint(primary_idx=i, secondary_idx=j, ratio=r)``:

           - Set column ``j`` to zeros everywhere.
           - Set row ``j``, column ``i`` to ``1.0 / r``
             (secondary = primary / ratio).

        3. Remove zero columns (constrained secondaries).
        4. Result shape: ``(n_lines, n_independent)`` where
           ``n_independent = n_lines - n_doublets``.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_lines, n_independent)``.

        Examples
        --------
        For [OIII] 5007 (primary, idx=p) / 4959 (secondary, idx=s), ratio=2.98:

        - Column s is zeroed.
        - Row s, column p is set to 1/2.98 ≈ 0.336.
        - Multiplying by independent amplitude ``a`` gives flux_5007 = a and
          flux_4959 = a / 2.98.
        """
        n = self.n_lines
        # Build as a regular numpy array for mutation, then convert
        import numpy as np

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _from_line_tuples(
        cls,
        lines: list[tuple[str, float, str, bool, bool]],
    ) -> LineCatalog:
        """Construct a LineCatalog from a list of ``(name, wave, species, balmer, broad)`` tuples.

        Doublet constraints are auto-detected from ``_DOUBLET_RATIOS`` by
        matching line names in the list.

        Parameters
        ----------
        lines : list of tuples
            Each tuple: ``(name, wavelength_angstrom, species, is_balmer, is_broad_candidate)``.

        Returns
        -------
        LineCatalog
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
        )


# ---------------------------------------------------------------------------
# Private helpers for CLOUDY parsing
# ---------------------------------------------------------------------------


def _parse_cloudy_species(name: str) -> str:
    """Extract species code from a CLOUDY line name.

    CLOUDY names follow conventions like ``"H  1 1215.67A"``,
    ``"O  3 5006.84A"``, or simple names like ``"Halpha"``.
    Returns a compact species code such as ``"H1"``, ``"O3"``.

    Parameters
    ----------
    name : str
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
    """Heuristic: True if the line is a hydrogen Balmer/Lyman series member."""
    if species != "H1":
        return False
    balmer_keywords = ("alpha", "beta", "gamma", "delta", "epsilon", "lya", "ly", "ha", "hb")
    name_lower = name.lower()
    return any(kw in name_lower for kw in balmer_keywords)


def _is_broad_candidate(name: str, species: str) -> bool:
    """Heuristic: True if this line can carry an AGN broad component."""
    broad_species = {"H1", "C4", "Mg2", "C3"}
    if species in broad_species:
        return True
    broad_keywords = ("lya", "civ", "mgii", "ciii", "halpha", "hbeta", "hgamma", "hdelta")
    name_lower = name.lower()
    return any(kw in name_lower for kw in broad_keywords)


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
    names : tuple of str
    wavelengths : list of float
    species : tuple of str
    proximity_angstrom : float, optional

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
            ratio = _DOUBLET_RATIOS.get(
                (names[primary_idx], names[secondary_idx]),
                _DOUBLET_RATIOS.get((names[secondary_idx], names[primary_idx]), 1.0),
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

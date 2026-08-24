# SPDX-License-Identifier: BSD-3-Clause
"""Cloudy 23 input-deck generator for photoionization grid regeneration.

Generates complete Cloudy 23 input files (text) given ionization parameter,
hydrogen density, metallicity, and ionizing spectrum. Useful for (a)
regenerating Cue's CLOUDY training grids and (b) sanity-checking Cue's
neural-emulator predictions against fresh Cloudy runs.

Running Cloudy itself is out-of-band via external subprocess:
``cloudy -p <prefix>``.

**Optional component:** ``pip install tengri[cloudy]``: though the module
itself has no external Cloudy dependency (text generation only).

Notes
-----
**JIT-compatible**: no, generates text, not arrays.

References
----------
.. [1] Ferland, G.J. et al. 2023, "The Cloudy 23 Release",
   Frontiers in Astronomy and Space Sciences, to appear.
.. [2] Ferland, G.J. et al. 2017, "The 2017 Release Cloudy",
   Rev. Mexicana Astron. Astrofis. 53, 385.
.. [3] Byler, N. et al. 2017, "Emission-line Diagnostics of O Stars",
   ApJ, 840, 44.
.. [4] Gutkin, J., Charlot, S., & Bruzual, G. 2016, "Impact of Photoionized
   Gas on the JWST-observable Emission Line Signatures of High-z Galaxies",
   MNRAS, 462, 1757.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp


@dataclass(frozen=True)
class Cloudy23Deck:
    """Frozen representation of a Cloudy 23 input deck.

    Constructed by :func:`build_cloudy23_deck`; written to disk by
    :meth:`write`. Running Cloudy itself is out-of-band:
    ``cloudy -p <prefix>``.

    Attributes
    ----------
    prefix: str
        Prefix for the Cloudy run (e.g., "tengri_logU-3.0_logZ-0.3").
        Output files: ``<prefix>.in``, ``<prefix>.sed`` (if tabulated),
        ``<prefix>.out`` (after Cloudy runs).
    title: str
        Human-readable deck title (appears in Cloudy `.out` file).
    commands: tuple[str, ...]
        Non-abundance input commands (ionization, density, iteration, etc.).
    abundances: tuple[str, ...]
        Cloudy abundance commands (``metals``, ``grains``, per-element scaling).
    """

    prefix: str
    title: str
    commands: tuple[str, ...]
    abundances: tuple[str, ...]

    def render(self) -> str:
        """Emit the full deck as a single string.

        Returns
        -------
        str
            Complete Cloudy 23 input deck, with lines joined by ``\\n`` and
            a trailing newline.

        """
        all_lines = [
            f"title {self.title}",
            *self.commands,
            *self.abundances,
            "",  # Cloudy requires a blank line at EOF
        ]
        return "\n".join(all_lines)

    def write(self, path: str | Path) -> Path:
        """Write the rendered deck to disk.

        Parameters
        ----------
        path: str or Path
            Directory in which to write ``<prefix>.in``. If ``sed_table``
            was passed to :func:`build_cloudy23_deck`, also writes
            ``<prefix>.sed`` (tabulated ionizing spectrum).

        Returns
        -------
        Path
            Absolute path to the written ``.in`` file.

        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        in_file = path / f"{self.prefix}.in"
        in_file.write_text(self.render())

        # Write SED table if present
        if hasattr(self, "_sed_table"):
            import numpy as np

            sed_table = object.__getattribute__(self, "_sed_table")
            wave_aa = np.asarray(sed_table["wave_aa"])
            j_lambda = np.asarray(sed_table["j_lambda"])
            sed_file = path / f"{self.prefix}.sed"
            # Write as 2-column ASCII (Cloudy format)
            np.savetxt(sed_file, np.column_stack([wave_aa, j_lambda]))

        return in_file


def build_cloudy23_deck(
    *,
    log_u: float,
    log_n_h: float,
    log_z_gas: float,
    sed_table: dict[str, jnp.ndarray] | None = None,
    sed_keyword: str | None = None,
    cloudy_iterations: int = 30,
    stop_temperature_k: float = 100.0,
    save_lines_path: str | None = None,
    save_continuum_path: str | None = None,
    abundance_set: str = "gass10",
    grain_set: str = "ism",
    extra_commands: tuple[str, ...] = (),
) -> Cloudy23Deck:
    """Build a Cloudy 23 input deck for a single (logU, n_H, log Z_gas) point.

    Parameters
    ----------
    log_u: float
        Ionization parameter log10(U). Typical range: [-5, 0].
    log_n_h: float
        Hydrogen number density log10(n_H) [cm^-3]. Typical range: [-2, 6].
    log_z_gas: float
        Gas metallicity log10(Z / Z_sun). Typical range: [-4, 1].
    sed_table: dict[str, ndarray], optional
        Tabulated ionizing spectrum. Keys: "wave_aa" (rest-frame Angstrom,
        shape ``(n_wave,)``) and "j_lambda" (``erg/s/Hz/cm^2/sr`` per pixel,
        shape ``(n_wave,)``). If provided, a sidecar ``.sed`` file is written
        in :meth:`Cloudy23Deck.write`. If None, use ``sed_keyword``.
        Default: None.
    sed_keyword: str, optional
        Cloudy ionizing-spectrum keyword (e.g., ``"blackbody, T=40000 K"``).
        Ignored if ``sed_table`` is provided. If both None, raises ValueError.
        Default: None.
    cloudy_iterations: int, optional
        Number of Cloudy iterations. Default: 30.
    stop_temperature_k: float, optional
        Stop temperature [K] for nebular gas. Default: 100.0 K.
    save_lines_path: str, optional
        Path (relative to Cloudy run directory) for ``save lines column`` output.
        If None, lines are not saved. Default: None.
    save_continuum_path: str, optional
        Path (relative to Cloudy run directory) for ``save continuum`` output.
        If None, continuum is not saved. Default: None.
    abundance_set: str, optional
        Abundance set: "gass10" (Grevesse+ 2010, Cloudy default), "ism"
        (ISM gas-phase), or "h_ii" (H II region, Gutkin+2016). Default:
        "gass10".
    grain_set: str, optional
        Grain set: "ism" (graphite + silicate mix, default), "agn"
        (smaller grains for AGN), or None (no grains). Default: "ism".
    extra_commands: tuple[str, ...], optional
        Additional Cloudy commands (e.g., ``("set nFnu = 4", "print lines")``)
        to append at the end. Default: empty tuple.

    Returns
    -------
    Cloudy23Deck
        Frozen deck object with :meth:`Cloudy23Deck.render` and
        :meth:`Cloudy23Deck.write` methods.

    Raises
    ------
    ValueError
        If log_u, log_n_h, log_z_gas are out of typical range, or if both
        ``sed_table`` and ``sed_keyword`` are None, or if ``abundance_set``
        is not recognized.

    Notes
    -----
    **Validation:** Range checks raise ``ValueError`` immediately; no JAX
    tracing.

    **Tabulated SED:** If ``sed_table`` is provided, the caller is responsible
    for ensuring wavelengths are rest-frame (not observed-frame). The SED is
    written as a 2-column ASCII table by :meth:`Cloudy23Deck.write`.

    **Abundance scales:** The ``"metals <Z/Zsun>"`` command scales all
    elements in the reference set by Z/Zsun (linear, not logarithmic).
    Abundance-set selection (Grevesse vs. ISM vs. H II) controls the
    reference abundances in Cloudy's internal lookup.

    References
    ----------
    .. [1] Ferland, G.J. et al. 2023, "The Cloudy 23 Release",
       Frontiers in Astronomy and Space Sciences, to appear.
    .. [2] Gutkin, J., Charlot, S., & Bruzual, G. 2016, "Impact of
       Photoionized Gas on the JWST-observable Emission Line Signatures
       of High-z Galaxies", MNRAS, 462, 1757.

    """
    # Validate inputs
    if not (-5 <= log_u <= 0):
        raise ValueError(f"log_u={log_u} out of typical range [-5, 0]. Cloudy may not converge.")
    if not (-2 <= log_n_h <= 6):
        raise ValueError(
            f"log_n_h={log_n_h} out of typical range [-2, 6]. Physical density may be unphysical."
        )
    if not (-4 <= log_z_gas <= 1):
        raise ValueError(
            f"log_z_gas={log_z_gas} out of typical range [-4, 1]. Metallicity may be unphysical."
        )
    if cloudy_iterations < 1:
        raise ValueError(f"cloudy_iterations={cloudy_iterations} must be >= 1.")
    if stop_temperature_k <= 0:
        raise ValueError(f"stop_temperature_k={stop_temperature_k} must be > 0.")
    if sed_table is None and sed_keyword is None:
        raise ValueError("Either sed_table or sed_keyword must be provided.")
    if abundance_set not in ("gass10", "ism", "h_ii"):
        raise ValueError(
            f"abundance_set='{abundance_set}' not recognized. Choose 'gass10', 'ism', or 'h_ii'."
        )

    # Build the deck prefix from parameters
    prefix = f"tengri_logU{log_u:.1f}_logn{log_n_h:.1f}_logZ{log_z_gas:.2f}"

    # Build title
    title = f"Photoionization grid: logU={log_u}, log(n_H)={log_n_h}, log(Z/Zsun)={log_z_gas}"

    # Build non-abundance commands
    commands = []

    # Ionizing spectrum
    if sed_table is not None:
        sed_file = f"{prefix}.sed"
        commands.append(f'table SED "{sed_file}"')
    else:
        commands.append(f"ionizing source table {sed_keyword}")

    # Ionization parameter (log U)
    commands.append(f"ionization parameter {log_u}")

    # Hydrogen density (cm^-3)
    commands.append(f"hden {log_n_h}")

    # Iteration
    commands.append(f"iterate {cloudy_iterations} times")

    # Stop temperature
    commands.append(f"stop temperature {stop_temperature_k} K")

    # Output commands
    if save_lines_path is not None:
        commands.append(f"save lines column {save_lines_path}")
    if save_continuum_path is not None:
        commands.append(f"save continuum {save_continuum_path}")

    # Extra commands
    commands.extend(extra_commands)

    # Build abundance commands
    abundances = []

    # Abundance set selection (maps to Cloudy internal reference)
    if abundance_set == "gass10":
        # Cloudy 23 default; no special command needed, but document it
        pass
    elif abundance_set == "ism":
        abundances.append("set abundances ism")
    elif abundance_set == "h_ii":
        abundances.append("set abundances H II regions")

    # Metallicity scaling (linear, not log)
    z_linear = 10**log_z_gas
    abundances.append(f"metals {z_linear:.4e}")

    # Grains (if requested)
    if grain_set is not None:
        if grain_set == "ism":
            abundances.append("grains ism")
        elif grain_set == "agn":
            abundances.append("grains agn")
        else:
            abundances.append(f"grains {grain_set}")

    deck = Cloudy23Deck(
        prefix=prefix,
        title=title,
        commands=tuple(commands),
        abundances=tuple(abundances),
    )

    # If sed_table provided, attach it to the deck (for write to emit sidecar)
    if sed_table is not None:
        # Store the table on the deck for later write
        object.__setattr__(deck, "_sed_table", sed_table)

    return deck

# SPDX-License-Identifier: BSD-3-Clause
r"""Draine, Li, Hensley et al. 2021 PAHspec template physics.

Pure-JAX physics helpers for the Draine+2021 PAHspec emission grid:
slicing the 6-D :math:`\nu P_\nu` cube to the chosen categorical
configuration, converting to :math:`L_\nu`, resampling onto the
pipeline rest-frame wavelength grid, and integrating
:math:`\int L_\nu \, d\nu` for the energy-balance rescale.

The :class:`tengri.components.dust.draine2021_pah_ir.Draine2021PAHIRSEDComponent`
emission component consumes this module (dispatched via
``dust_emission={'type': 'draine2021_pah'}``).

Reference
---------
.. [1] Draine, B.T., Li, A., Hensley, B.S., Hunt, L.K., Sandstrom, K.,
   Smith, J.-D.T., 2021, "Excitation of Polycyclic Aromatic Hydrocarbon
   Emission: Dependence on Size Distribution, Ionization, and Starlight
   Spectrum and Intensity", ApJ, 917, 3.  arXiv:2011.07046.
   DOI: 10.3847/1538-4357/abff51.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp

from tengri.components.dust.emission_templates import (
    Draine2021PAHTemplates,
    load_draine2021_pahspec_templates,
)
from tengri.utils.physics_constants import C_CGS as _C_CGS

__all__ = [
    "DRAINE2021_PAH_DEFAULT_PATH",
    "PAHSPEC_PATH_ENV",
    "STARLIGHT_PROPERTIES",
    "load_pahspec_or_raise",
    "missing_template_message",
    "resample_lnu_on_aa_grid",
    "select_pahspec_axes",
    "select_pahspec_starlight_auto",
]


PAHSPEC_PATH_ENV = "TENGRI_PAHSPEC_PATH"
DRAINE2021_PAH_DEFAULT_PATH = "data/pahspec_draine2021.h5"

# ─────────────────────────────────────────────────────────────────────
# Starlight metadata for nearest-neighbor selection
# ─────────────────────────────────────────────────────────────────────

# Each entry: (kind, sps_family, age_myr, log_z_solar)
#   kind: "ssp" (Bruzual-Charlot or BPASS SSP),
#         "diffuse_isrf" (Mathis-Mezger-Panagia local Galactic ISRF),
#         "bulge" (Groves+2012 integrated M31 bulge spectrum).
#   sps_family: "BC03" | "BPASS" | None (non-SSP).
#   age_myr: SSP age in Myr; None for non-SSP.
#   log_z_solar: log10(Z / Z_sun); None for non-SSP.
#
# Z_sun convention: Draine uses Z=0.02 in tarball names; so Z=0.0004 → log_z_solar = -1.70,
# Z=0.001 → -1.30, Z=0.02 → 0.
STARLIGHT_PROPERTIES: dict[str, dict] = {
    "mMMP": {"kind": "diffuse_isrf", "sps_family": None, "age_myr": None, "log_z_solar": None},
    "m31bulge": {"kind": "bulge", "sps_family": None, "age_myr": None, "log_z_solar": None},
    "BC03_Z0.0004_10Myr": {
        "kind": "ssp",
        "sps_family": "BC03",
        "age_myr": 10.0,
        "log_z_solar": -1.70,
    },
    "BC03_Z0.02_3Myr": {"kind": "ssp", "sps_family": "BC03", "age_myr": 3.0, "log_z_solar": 0.0},
    "BC03_Z0.02_10Myr": {"kind": "ssp", "sps_family": "BC03", "age_myr": 10.0, "log_z_solar": 0.0},
    "BC03_Z0.02_100Myr": {
        "kind": "ssp",
        "sps_family": "BC03",
        "age_myr": 100.0,
        "log_z_solar": 0.0,
    },
    "BC03_Z0.02_300Myr": {
        "kind": "ssp",
        "sps_family": "BC03",
        "age_myr": 300.0,
        "log_z_solar": 0.0,
    },
    "BC03_Z0.02_1Gyr": {
        "kind": "ssp",
        "sps_family": "BC03",
        "age_myr": 1000.0,
        "log_z_solar": 0.0,
    },
    "BPASS_Z0.001_10Myr": {
        "kind": "ssp",
        "sps_family": "BPASS",
        "age_myr": 10.0,
        "log_z_solar": -1.30,
    },
    "BPASS_Z0.02_3Myr": {"kind": "ssp", "sps_family": "BPASS", "age_myr": 3.0, "log_z_solar": 0.0},
    "BPASS_Z0.02_10Myr": {
        "kind": "ssp",
        "sps_family": "BPASS",
        "age_myr": 10.0,
        "log_z_solar": 0.0,
    },
    "BPASS_Z0.02_100Myr": {
        "kind": "ssp",
        "sps_family": "BPASS",
        "age_myr": 100.0,
        "log_z_solar": 0.0,
    },
    "BPASS_Z0.02_300Myr": {
        "kind": "ssp",
        "sps_family": "BPASS",
        "age_myr": 300.0,
        "log_z_solar": 0.0,
    },
    "BPASS_Z0.02_1Gyr": {
        "kind": "ssp",
        "sps_family": "BPASS",
        "age_myr": 1000.0,
        "log_z_solar": 0.0,
    },
}

# ─────────────────────────────────────────────────────────────────────
# Unit helpers (private)
# ─────────────────────────────────────────────────────────────────────

# 1 micron = 10000 Angstrom.
_UM_TO_AA = 1.0e4


# ─────────────────────────────────────────────────────────────────────
# Path discovery + missing-template diagnostic
# ─────────────────────────────────────────────────────────────────────


def missing_template_message(path: Path) -> str:
    """Produce the canonical FileNotFoundError message body.

    Parameters
    ----------
    path : pathlib.Path
        The expected HDF5 location that was not found.

    Returns
    -------
    str
        Multi-line message naming the exact build-script invocation,
        the upstream URL, and the env-var override knob.
    """
    return (
        f"Draine+2021 PAHspec template grid not found at {path!s}.\n"
        f"\n"
        f"This component does NOT carry an analytic fallback: the "
        f"published numerical templates are required for physically "
        f"meaningful PAH emission predictions.\n"
        f"\n"
        f"Build the grid with:\n"
        f"  python scripts/build_pahspec_hdf5.py \\\n"
        f"      --output {path!s} \\\n"
        f"      --download\n"
        f"\n"
        f"This downloads ~7 GB of ASCII tarballs from "
        f"https://www.astro.princeton.edu/~draine/PAHspec/ and packs "
        f"them into a single HDF5 (~150 MB).  Override the location "
        f"by setting {PAHSPEC_PATH_ENV} or by passing template_path= "
        f"to Draine2021PAHIRConfig."
    )


def load_pahspec_or_raise(template_path: str | None) -> Draine2021PAHTemplates:
    """Load PAHspec templates, raising a helpful error when missing.

    Parameters
    ----------
    template_path : str or None
        Path override.  When ``None``, falls back to the
        :data:`PAHSPEC_PATH_ENV` environment variable, then to
        :data:`DRAINE2021_PAH_DEFAULT_PATH`.

    Returns
    -------
    Draine2021PAHTemplates
        Frozen container of the full PAHspec grid.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist on disk.  The error
        message contains the exact build-script invocation needed to
        produce the file.
    """
    if template_path is not None:
        path = Path(template_path)
    elif os.environ.get(PAHSPEC_PATH_ENV):
        path = Path(os.environ[PAHSPEC_PATH_ENV])
    else:
        # Walk parent dirs for data/pahspec_draine2021.h5 so the bundled grid
        # is found from example subdirectories (sphinx-gallery chdir's in).
        from tengri._data_setup import data_path

        try:
            path = data_path("pahspec_draine2021.h5")
        except FileNotFoundError:
            path = Path(DRAINE2021_PAH_DEFAULT_PATH)
    if not path.is_file():
        raise FileNotFoundError(missing_template_message(path))
    return load_draine2021_pahspec_templates(str(path))


def load_pahspec_draine2021(template_path: str | None = None) -> Draine2021PAHTemplates:
    r"""Load the Draine 2021 PAH emission-spectrum templates.

    Public entry point for the bundled Draine (2021) PAHspec grid. Use it
    instead of opening ``data/pahspec_draine2021.h5`` by hand; pair it with
    :func:`select_pahspec_axes` to pick a (starlight, ionization,
    size-distribution, slab) slice.

    Parameters
    ----------
    template_path : str or None, optional
        Override the grid location. When ``None`` (default), resolves the
        :data:`PAHSPEC_PATH_ENV` env var, then the bundled
        :data:`DRAINE2021_PAH_DEFAULT_PATH` (``data/pahspec_draine2021.h5``).

    Returns
    -------
    Draine2021PAHTemplates
        Frozen container of JAX arrays.

    Raises
    ------
    FileNotFoundError
        When the resolved path does not exist on disk.

    Notes
    -----
    **JIT-compatible**: no — file I/O. Call once at setup.

    References
    ----------
    .. [1] Draine, B. T., et al. 2021, ApJ, 917, 3. Excitation of Polycyclic
       Aromatic Hydrocarbon Emission. arXiv:2106.07415.
    """
    return load_pahspec_or_raise(template_path)


# ─────────────────────────────────────────────────────────────────────
# Auto-selection of starlight from upstream stellar-population params
# ─────────────────────────────────────────────────────────────────────


# Score weights (relative units only — minimization).  Calibrated so that
# (i) within an SSP family, the nearest (age, log Z) wins;
# (ii) without a family, mMMP wins for moderate ages (~1 Gyr-ish ambient
#      Galactic ISM) and m31bulge wins only for genuinely old populations
#      (≥ ~5 Gyr) where the SSP grid's 1 Gyr template is too blue.
_AUTO_LOG_Z_WEIGHT = 4.0  # metallicity dominates over age within an SSP family
_AUTO_AMBIENT_REF_LOG_AGE = 3.0  # mMMP centered on log10(1 Gyr / Myr) = 3.0
_AUTO_AMBIENT_AGE_WEIGHT = 0.5  # weak age penalty for mMMP (it's ambient)
_AUTO_OLD_AGE_THRESHOLD_LOG = 3.7  # log10(5 Gyr / Myr); m31bulge becomes free above this
_AUTO_AMBIENT_FLOOR = 0.5  # baseline penalty for fallback to mMMP
_AUTO_BULGE_FLOOR = 0.3  # baseline penalty for fallback to m31bulge


def select_pahspec_starlight_auto(
    *,
    sps_family: str | None,
    age_myr: float,
    log_z_solar: float,
) -> str:
    r"""Pick the nearest available PAHspec starlight from upstream params.

    Searches the 13 published PAHspec starlight choices (see
    :data:`STARLIGHT_PROPERTIES`) for the best match given the
    stellar population that the SED-fit upstream is using.  This is
    nearest-neighbor selection in :math:`(\log_{10} {\rm age},
    \log_{10} Z/Z_\odot)` space within the chosen SPS family, with
    fallbacks to the two non-SSP starlights (``"mMMP"`` for diffuse
    ISM, ``"m31bulge"`` for old quiescent populations).

    This function is *not* fully self-consistent — the PAHspec
    library only tabulates 13 fixed starlight spectra.  Use this when
    you want to remove the manual config burden and accept
    "nearest of 13" approximation.  Truly arbitrary starlight would
    require running Draine's PAH thermal-fluctuation code online,
    which is out of scope for this module.

    Selection algorithm
    -------------------
    1. If ``age_myr >= 5000`` (5 Gyr threshold) and no SSP family
       matches well, prefer ``"m31bulge"``.
    2. Otherwise, restrict candidates to the requested
       ``sps_family``.  ``None`` means "any SSP family is OK".  Other
       families (FSPS / MIST / PrSc) fall back to the non-SSP
       ambient choices because Draine did not tabulate templates for
       them.
    3. Score each SSP candidate by

       .. math::
            d^2 \;=\; (\log_{10}\!{\rm age}_{\rm fit}
                       - \log_{10}\!{\rm age}_{\rm template})^2
                  +\; w_Z\,(\log Z_{\rm fit}/Z_\odot
                            - \log Z_{\rm template}/Z_\odot)^2

       with ``w_Z = 4`` (metallicity dominates within a family).
    4. Score ``"mMMP"`` and ``"m31bulge"`` with constant floors plus
       age-distance penalties (mMMP to ~5 Gyr, m31bulge to ages
       larger than 5 Gyr).
    5. Return the lowest-score name.

    Parameters
    ----------
    sps_family : {"BC03", "BPASS", None} or other str
        Stellar population synthesis family used by the upstream
        stellar component.  ``None`` allows any SSP family.  Strings
        not in ``{"BC03", "BPASS"}`` fall back to non-SSP ambient
        starlights only.
    age_myr : float
        Characteristic age of the FUV-emitting young population, in
        Myr.  Typical values: 3-10 Myr for active starbursts,
        100-300 Myr for older star-forming galaxies, >5000 Myr for
        quiescent / bulge-dominated systems. [Myr]
    log_z_solar : float
        :math:`\log_{10}(Z/Z_\odot)` for the ionizing stellar
        population. [dimensionless]

    Returns
    -------
    str
        Best-matching PAHspec starlight name (a key of
        :data:`STARLIGHT_PROPERTIES`).

    Raises
    ------
    ValueError
        If ``age_myr <= 0`` or any input is non-finite.

    Notes
    -----
    **JIT-compatible**: no — pure Python decision logic at factory
    time.
    """
    import math

    if not (age_myr > 0):
        raise ValueError(f"age_myr must be > 0, got {age_myr!r}")
    if not math.isfinite(age_myr) or not math.isfinite(log_z_solar):
        raise ValueError(f"age_myr={age_myr!r}, log_z_solar={log_z_solar!r} must be finite")

    log_age_fit = math.log10(age_myr)
    is_supported_family = sps_family in (None, "BC03", "BPASS")

    scores: list[tuple[float, str]] = []
    for name, props in STARLIGHT_PROPERTIES.items():
        kind = props["kind"]
        if kind == "ssp":
            if not is_supported_family:
                # FSPS / MIST / PrSc: skip BC03/BPASS templates entirely.
                continue
            if sps_family is not None and props["sps_family"] != sps_family:
                continue
            d_age = log_age_fit - math.log10(props["age_myr"])
            d_z = log_z_solar - props["log_z_solar"]
            scores.append((d_age * d_age + _AUTO_LOG_Z_WEIGHT * d_z * d_z, name))
        elif kind == "diffuse_isrf":
            # mMMP is the local diffuse Galactic ISRF; broadly applicable
            # ambient fallback.  Weak age penalty around ~1 Gyr keeps it
            # competitive for any non-extreme population.
            d_age = log_age_fit - _AUTO_AMBIENT_REF_LOG_AGE
            scores.append((_AUTO_AMBIENT_FLOOR + _AUTO_AMBIENT_AGE_WEIGHT * d_age * d_age, name))
        elif kind == "bulge":
            # m31bulge represents a genuinely old (>5 Gyr) integrated
            # population.  Penalize heavily for younger ages; zero
            # penalty above the 5 Gyr threshold.
            d_age = max(0.0, _AUTO_OLD_AGE_THRESHOLD_LOG - log_age_fit)
            scores.append((_AUTO_BULGE_FLOOR + d_age * d_age, name))

    return min(scores)[1]


# ─────────────────────────────────────────────────────────────────────
# Categorical-axis slicing
# ─────────────────────────────────────────────────────────────────────


def select_pahspec_axes(
    templates: Draine2021PAHTemplates,
    *,
    starlight: str,
    ionization: str,
    size_distribution: str,
    slab: bool,
) -> jnp.ndarray:
    r"""Slice the 6-D PAHspec :math:`\nu P_\nu` cube to a 1-D ``lgU`` axis.

    The PAHspec grid axes are
    ``(starlight, slab, lgU, ionization, size, wavelength)``.  This
    helper picks one ``(starlight, ionization, size, slab)`` tuple
    and returns the remaining ``(n_lgU, n_wave)`` slice.

    Parameters
    ----------
    templates : Draine2021PAHTemplates
        Loader output.  Provides the integer-index lookup tables for
        the four categorical axes.
    starlight : str
        Starlight spectrum name (one of
        ``templates.starlight_names``).
    ionization : {"lo", "st", "hi"}
        PAH ionization fraction selector.
    size_distribution : {"sma", "std", "lrg"}
        PAH size distribution selector.
    slab : bool
        ``True`` for the :math:`A_V=2` slab variant, ``False`` for
        unattenuated diffuse heating.

    Returns
    -------
    jnp.ndarray, shape ``(n_lgU=15, n_wave_um)``
        :math:`\nu P_\nu` in [erg/s/H] on the template's native
        wavelength grid (microns).

    Raises
    ------
    ValueError
        If any axis label is not present in the loaded grid.
    """
    if starlight not in templates.starlight_names:
        raise ValueError(
            f"starlight={starlight!r} not in template grid; available: {templates.starlight_names}"
        )
    if ionization not in templates.ion_names:
        raise ValueError(f"ionization={ionization!r} not in {templates.ion_names}")
    if size_distribution not in templates.size_names:
        raise ValueError(f"size_distribution={size_distribution!r} not in {templates.size_names}")

    i_sl = templates.starlight_names.index(starlight)
    i_ion = templates.ion_names.index(ionization)
    i_size = templates.size_names.index(size_distribution)
    slab_arr = jnp.asarray(templates.slab)
    matches = jnp.where(slab_arr == bool(slab))[0]
    if matches.size == 0:
        raise ValueError(f"slab={slab} not in grid (available={templates.slab})")
    i_slab = int(matches[0])

    return templates.nu_pnu_total[i_sl, i_slab, :, i_ion, i_size, :]


# ─────────────────────────────────────────────────────────────────────
# Spectral resampling and integration
# ─────────────────────────────────────────────────────────────────────


def resample_lnu_on_aa_grid(
    nu_pnu_um: jnp.ndarray,
    wave_um: jnp.ndarray,
    wave_aa: jnp.ndarray,
) -> jnp.ndarray:
    r"""Convert :math:`\nu P_\nu` to :math:`L_\nu` and resample to Å grid.

    .. math::

        L_\nu(\lambda) \;=\; \frac{\nu P_\nu(\lambda)}{\nu}
        \;=\; \frac{(\nu P_\nu)\,\lambda}{c}

    where :math:`\nu P_\nu` is in [erg/s/H], :math:`\lambda` in cm,
    and :math:`c` is the speed of light in [cm/s].  Values outside
    the template's native support are zero-padded.

    Parameters
    ----------
    nu_pnu_um : array_like, shape ``(..., n_wave_um)``
        :math:`\nu P_\nu` cubes on the template's native wavelength
        grid (microns).  Per-H units [erg/s/H]; the absolute
        normalization does not survive the energy-balance rescale.
    wave_um : array_like, shape ``(n_wave_um,)``
        Template wavelengths in microns; strictly increasing.
    wave_aa : array_like, shape ``(n_wave_aa,)``
        Pipeline rest-frame wavelength grid in Angstrom.

    Returns
    -------
    ndarray, shape ``(..., n_wave_aa)``
        :math:`L_\nu` in [erg/s/Hz/H] on the pipeline grid.

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` primitives plus
    :func:`jax.vmap`.  **Gradient-safe**: yes.
    """
    lam_cm = wave_um * 1.0e-4
    L_nu_template = nu_pnu_um * lam_cm[..., None, :] / _C_CGS  # broadcast over leading dims
    template_wave_aa = wave_um * _UM_TO_AA

    def _interp_one(L_nu_row: jnp.ndarray) -> jnp.ndarray:
        return jnp.interp(
            wave_aa,
            template_wave_aa,
            L_nu_row,
            left=0.0,
            right=0.0,
        )

    return jax.vmap(_interp_one)(L_nu_template)

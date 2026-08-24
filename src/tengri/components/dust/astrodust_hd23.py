# SPDX-License-Identifier: BSD-3-Clause
r"""Hensley & Draine 2023 Astrodust+PAH template physics.

Pure-JAX physics helpers for the Astrodust+PAH emission grid published
at Harvard Dataverse (doi:10.7910/DVN/3B6E6S):

    * 91 :math:`\log_{10} U` points in -3..6 step 0.1
    * 1000 wavelength points 0.1 - 30 000 μm
    * Single fiducial PAH size distribution and ionization fraction
      (Hensley & Draine 2022) -- there is no published (qpah, size)
      grid for this model.

Three emission components are stored separately: ``astrodust``,
``pah``, and their sum (``total``).  Per-H normalization; downstream
energy-balance rescaling against ``L_ir`` is handled by
:class:`tengri.components.dust.emission.templates.astrodust.AstrodustIRSEDComponent`.

Reference
---------
.. [1] Hensley, B.S. & Draine, B.T. 2023, "The Astrodust+PAH Model: A
   Unified Description of the Extinction, Emission, and Polarization
   from Dust in the Diffuse Interstellar Medium", ApJ 948, 55,
   arXiv:2208.12365.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from tengri.utils.grid_interp import resample_template

__all__ = [
    "ASTRODUST_HD23_DEFAULT_PATH",
    "ASTRODUST_HD23_PATH_ENV",
    "AstrodustHD23Templates",
    "load_astrodust_hd23_or_raise",
    "missing_astrodust_template_message",
    "resample_lnu_on_aa_grid",
]


ASTRODUST_HD23_PATH_ENV = "TENGRI_ASTRODUST_PATH"
ASTRODUST_HD23_DEFAULT_PATH = "data/astrodust_templates.h5"


# 1 micron = 10000 Angstrom.
_UM_TO_AA = 1.0e4


@dataclass(frozen=True)
class AstrodustHD23Templates:
    r"""Frozen container for the Hensley & Draine 2023 emission grid.

    Holds the size-distribution-integrated thermal emission (HDU 7)
    and the spinning-dust emission (HDU 9) of the published
    fiducial-configuration FITS file.  All arrays are per H atom in
    cgs.  See :mod:`tengri.components.dust.astrodust_hd23` for the
    canonical loader and unit conversion.

    Attributes
    ----------
    wavelength_um: jnp.ndarray, shape ``(1000,)``
        Wavelength axis in microns; strictly increasing from 0.1 to
        3e4 μm.
    lgU: jnp.ndarray, shape ``(91,)``
        :math:`\log_{10} U` from -3 to +6 in 0.1 steps.  ``U=1`` is
        the local Galactic ISRF (Mathis-Mezger-Panagia 1983).
    L_nu_total: jnp.ndarray, shape ``(91, 1000)``
        Total thermal :math:`L_\nu` (Astrodust + PAHs) per H atom
        [erg/s/Hz/H], converted from the FITS file's
        :math:`\lambda I_\lambda / N_H` [erg/s/sr/H] via
        :math:`L_\nu = 4\pi \lambda^2 I_\lambda / c`.
    L_nu_astrodust, L_nu_pah: jnp.ndarray, shape ``(91, 1000)``
        Astrodust-only and PAH-only thermal-emission components.
    L_nu_spdust_total: jnp.ndarray, shape ``(1000,)``
        Total spinning-dust microwave emission (HDU 9, total column)
        per H atom [erg/s/Hz/H].  Assumes :math:`f_{\rm CNM}=0.28`.
        Not :math:`U`-dependent: spinning dust is mostly insensitive
        to radiation field intensity.
    L_nu_spdust_Ad_CNM, L_nu_spdust_Ad_WNM, L_nu_spdust_PAH_CNM,
    L_nu_spdust_PAH_WNM: jnp.ndarray, shape ``(1000,)``
        Per-component / per-phase spinning dust spectra, allowing
        custom :math:`f_{\rm CNM}` mixing.
    tau_per_H: jnp.ndarray, shape ``(1000, 4)``
        Extinction cross-section :math:`\tau_\lambda/N_H` [cm²/H]
        with columns ``(lambda_um, tau_Ad, tau_PAH, tau_total)``.
    sigma_sca_per_H: jnp.ndarray, shape ``(1000, 4)``
        Scattering cross-section [cm²/H], same column layout as
        extinction.  Albedo = scattering / extinction.
    p_pol_per_H: jnp.ndarray, shape ``(1000, 2)``
        Polarized extinction cross-section
        :math:`(p_\lambda/N_H)^{\rm max}` [cm²/H] from Astrodust;
        columns ``(lambda_um, p_Ad)``.
    lambda_P_lambda_polarized: jnp.ndarray, shape ``(91, 1000)``
        Polarized emission :math:`\lambda P_\lambda/N_H`
        [erg/s/sr/H] (HDU 8) from Astrodust grains.
    M_Ad_over_M_H: float
        Astrodust mass per H atom (0.00642).
    M_PAH_over_M_H: float
        PAH mass per H atom (0.000659).
    paper, arxiv, doi: str
        Citation traceability strings.
    """

    wavelength_um: jnp.ndarray
    lgU: jnp.ndarray
    L_nu_total: jnp.ndarray
    L_nu_astrodust: jnp.ndarray
    L_nu_pah: jnp.ndarray
    L_nu_spdust_total: jnp.ndarray
    L_nu_spdust_Ad_CNM: jnp.ndarray
    L_nu_spdust_Ad_WNM: jnp.ndarray
    L_nu_spdust_PAH_CNM: jnp.ndarray
    L_nu_spdust_PAH_WNM: jnp.ndarray
    tau_per_H: jnp.ndarray
    sigma_sca_per_H: jnp.ndarray
    p_pol_per_H: jnp.ndarray
    lambda_P_lambda_polarized: jnp.ndarray
    #: Fiducial per-H grain size distribution, shape ``(n_radii, 5)``:
    #: columns are ``[a_um, dn_Ad/n_H, dn_PAH/n_H, ...]``. ``None`` for older
    #: grids that predate the dataset.
    size_distribution: jnp.ndarray | None = None
    M_Ad_over_M_H: float = 0.00642
    M_PAH_over_M_H: float = 0.000659
    paper: str = "Hensley & Draine 2023, ApJ 948, 55"
    arxiv: str = "2208.12365"
    doi: str = "10.7910/DVN/3B6E6S"


def missing_astrodust_template_message(path: Path) -> str:
    """Produce the canonical FileNotFoundError message body."""
    return (
        f"Hensley & Draine 2023 Astrodust+PAH grid not found at {path!s}.\n"
        f"\n"
        f"This component does NOT carry an analytic fallback; the\n"
        f"published numerical templates are required for physically\n"
        f"meaningful predictions.\n"
        f"\n"
        f"Build the grid with:\n"
        f"  python scripts/build_astrodust_hdf5.py \\\n"
        f"      --output {path!s} \\\n"
        f"      --download\n"
        f"\n"
        f"This downloads the canonical FITS file (~3 MB) from\n"
        f"doi:10.7910/DVN/3B6E6S and packs HDU 7 (EMISSION) into a\n"
        f"compact HDF5 (~750 KB).  Override the location by setting\n"
        f"{ASTRODUST_HD23_PATH_ENV} or by passing template_path= to\n"
        f"AstrodustIRConfig."
    )


def _resolve_astrodust_path(template_path: str | None) -> Path:
    """Resolve the grid location: explicit arg → env var → ancestor-walk.

    When neither an explicit path nor the env var is given, walk parent dirs
    for ``data/astrodust_templates.h5`` (via :func:`tengri.data_path`) so the
    bundled grid is found even when the cwd is an example subdirectory
    (sphinx-gallery ``chdir``s into each script's folder). Falls back to the
    bare relative default so the caller's FileNotFoundError message still fires.
    """
    if template_path is not None:
        return Path(template_path)
    env = os.environ.get(ASTRODUST_HD23_PATH_ENV)
    if env:
        return Path(env)
    from tengri._data_setup import data_path

    try:
        return data_path("astrodust_templates.h5")
    except FileNotFoundError:
        return Path(ASTRODUST_HD23_DEFAULT_PATH)


def load_astrodust_hd23_or_raise(template_path: str | None) -> AstrodustHD23Templates:
    r"""Load the Astrodust+PAH HDF5 grid, raising on missing file.

    Parameters
    ----------
    template_path: str or None
        Path override.  When ``None``, falls back to the
        :data:`ASTRODUST_HD23_PATH_ENV` env var, then to
        :data:`ASTRODUST_HD23_DEFAULT_PATH`.

    Returns
    -------
    AstrodustHD23Templates
        Frozen container with JAX arrays.

    Raises
    ------
    FileNotFoundError
        When the resolved path does not exist on disk.
    """
    import h5py

    path = _resolve_astrodust_path(template_path)
    if not path.is_file():
        raise FileNotFoundError(missing_astrodust_template_message(path))

    with h5py.File(path, "r") as f:
        wavelength_um = jnp.asarray(f["wavelength_um"][:])
        lgU = jnp.asarray(f["lgU"][:])
        L_nu_total = jnp.asarray(f["L_nu_total"][...])
        L_nu_astrodust = jnp.asarray(f["L_nu_astrodust"][...])
        L_nu_pah = jnp.asarray(f["L_nu_pah"][...])
        L_nu_spdust_total = jnp.asarray(f["L_nu_spdust_total"][...])
        L_nu_spdust_Ad_CNM = jnp.asarray(f["L_nu_spdust_Ad_CNM"][...])
        L_nu_spdust_Ad_WNM = jnp.asarray(f["L_nu_spdust_Ad_WNM"][...])
        L_nu_spdust_PAH_CNM = jnp.asarray(f["L_nu_spdust_PAH_CNM"][...])
        L_nu_spdust_PAH_WNM = jnp.asarray(f["L_nu_spdust_PAH_WNM"][...])
        tau_per_H = jnp.asarray(f["extinction"][...])
        sigma_sca_per_H = jnp.asarray(f["scattering"][...])
        p_pol_per_H = jnp.asarray(f["polarized_extinction"][...])
        lambda_P_lambda_polarized = jnp.asarray(f["lambda_P_lambda_polarized"][...])
        size_distribution = (
            jnp.asarray(f["size_distribution"][...]) if "size_distribution" in f else None
        )
        M_Ad_over_M_H = float(f.attrs.get("M_Ad_over_M_H", 0.00642))
        M_PAH_over_M_H = float(f.attrs.get("M_PAH_over_M_H", 0.000659))

    return AstrodustHD23Templates(
        wavelength_um=wavelength_um,
        lgU=lgU,
        L_nu_total=L_nu_total,
        L_nu_astrodust=L_nu_astrodust,
        L_nu_pah=L_nu_pah,
        L_nu_spdust_total=L_nu_spdust_total,
        L_nu_spdust_Ad_CNM=L_nu_spdust_Ad_CNM,
        L_nu_spdust_Ad_WNM=L_nu_spdust_Ad_WNM,
        L_nu_spdust_PAH_CNM=L_nu_spdust_PAH_CNM,
        L_nu_spdust_PAH_WNM=L_nu_spdust_PAH_WNM,
        tau_per_H=tau_per_H,
        sigma_sca_per_H=sigma_sca_per_H,
        p_pol_per_H=p_pol_per_H,
        lambda_P_lambda_polarized=lambda_P_lambda_polarized,
        size_distribution=size_distribution,
        M_Ad_over_M_H=M_Ad_over_M_H,
        M_PAH_over_M_H=M_PAH_over_M_H,
    )


def load_astrodust_hd23(template_path: str | None = None) -> AstrodustHD23Templates:
    r"""Load the Hensley & Draine 2023 Astrodust+PAH emission grid.

    Public entry point for the bundled Astrodust+PAH templates: emission
    (:math:`L_\nu` per H for total / astrodust / PAH / spinning-dust
    components), extinction / scattering / polarization per H, and the
    fiducial grain ``size_distribution``. Use it instead of opening
    ``data/astrodust_templates.h5`` by hand.

    Parameters
    ----------
    template_path: str or None, optional
        Override the grid location. When ``None`` (default), resolves the
        :data:`ASTRODUST_HD23_PATH_ENV` env var, then the bundled
        :data:`ASTRODUST_HD23_DEFAULT_PATH` (``data/astrodust_templates.h5``).

    Returns
    -------
    AstrodustHD23Templates
        Frozen container of JAX arrays (see the class for the full field list).

    Raises
    ------
    FileNotFoundError
        When the resolved path does not exist on disk.

    Notes
    -----
    **JIT-compatible**: no, file I/O. Call once at setup, then pass the
    returned arrays into JIT-compiled code.

    References
    ----------
    .. [1] Hensley, B. S. & Draine, B. T. 2023, ApJ, 948, 55.
       The Astrodust+PAH Model. arXiv:2208.12365. doi:10.7910/DVN/3B6E6S.

    Examples
    --------
    >>> import tengri
    >>> tpl = tengri.load_astrodust_hd23()  # doctest: +SKIP
    >>> tpl.wavelength_um.shape  # doctest: +SKIP
    (...,)
    """
    return load_astrodust_hd23_or_raise(template_path)


def resample_lnu_on_aa_grid(
    L_nu_um: jnp.ndarray,
    wave_um: jnp.ndarray,
    wave_aa: jnp.ndarray,
) -> jnp.ndarray:
    r"""Interpolate :math:`L_\nu(\lambda)` onto a pipeline Å grid.

    Parameters
    ----------
    L_nu_um: array_like, shape ``(..., n_wave_um)``
        :math:`L_\nu` (per H) on the template's microns grid
        [erg/s/Hz/H].
    wave_um: array_like, shape ``(n_wave_um,)``
        Template wavelength axis in microns; strictly increasing.
    wave_aa: array_like, shape ``(n_wave_aa,)``
        Pipeline rest-frame wavelength grid in Angstrom.

    Returns
    -------
    jnp.ndarray, shape ``(..., n_wave_aa)``
        :math:`L_\nu` on the pipeline grid.  Values outside the
        template's native support are zero-padded.

    Notes
    -----
    **JIT-compatible**: yes.  **Gradient-safe**: yes.
    """
    template_wave_aa = wave_um * _UM_TO_AA

    def _interp_one(row: jnp.ndarray) -> jnp.ndarray:
        return resample_template(wave_aa, template_wave_aa, row, left=0.0, right=0.0)

    return jax.vmap(_interp_one)(L_nu_um)

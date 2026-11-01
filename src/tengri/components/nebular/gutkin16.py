# SPDX-License-Identifier: BSD-3-Clause
r"""Gutkin+2016 NEOGAL star-forming nebular line backend.

Source
------
Gutkin, Charlot & Bruzual 2016, MNRAS 462, 1757 (arXiv:1607.06086), constant-SFR
nebular tables computed with Cloudy c13.03 for 100 Myr of continuous star formation.
The tables are repackaged, with attribution, as ``data/gutkin16_grid.h5`` by
``scripts/build_gutkin16_grid.py``. Each stored value is the line luminosity per
ionizing photon, :math:`\log_{10}(L_{\rm line}/Q_H)` [erg/photon], on a grid of
:math:`(\log Z,\ \log U_S,\ \xi_d,\ \log n_H,\ {\rm C/O},\ m_{\rm up})`.

Axes and parameter map
----------------------
``neb_logU`` is Gutkin's :math:`\log_{10} U_S`, the ionization parameter at the
Strömgren radius (Gutkin+2016 eq. 7). The volume-averaged value satisfies
:math:`\langle U\rangle \approx (9/4)\,U_S`. Tengri's Cue ``neb_logU`` is the inner-face
value, so the two differ by a fixed geometric factor.

The BEAGLE parameter map (Chevallard & Charlot 2016) is:

* ``nebular_logU`` -> ``neb_logU``
* ``nebular_Z`` (in units of BEAGLE's :math:`Z_\odot = 0.01524`) -> ``neb_logZ_gas``
  with :math:`\log_{10}(Z/Z_\odot)_{\rm tengri} = \log_{10}(Z/Z_\odot)_{\rm BEAGLE} + 0.031`,
  since :math:`\log_{10}(0.01524/0.0142) = 0.031` for tengri's :math:`Z_\odot = 0.0142`
* ``nebular_xi`` -> ``neb_xi_d``
* ``nebular_CO`` -> ``neb_co_ratio``
* ``LyC_f_esc`` -> ``neb_fesc``

``neb_logZ_gas`` is absolute :math:`\log_{10} Z` (the same convention as ``log_z``
and as :class:`~tengri.components.nebular.cloudy_cb19.CB19Backend`), because the
nebular component adds ``LOG10_ZSUN`` before calling any backend.

Normalization
-------------
The per-photon normalization is anchored to the table's own Hβ luminosity under Case B
recombination (Osterbrock & Ferland 2006, Table 4.4, :math:`T_e = 10^4` K), at
:math:`\log U = -4`, :math:`n_H = 100\,{\rm cm^{-3}}`, :math:`({\rm C/O})/({\rm C/O})_\odot
= 1` and :math:`\xi_d \to 0`. BC03-2003 emits 0.07 to 0.15 dex fewer ionizing photons
than the Gutkin stellar models, so the Q_H supplied by an SSP grid is a different
normalization from the one the table was built against.

Approximations
--------------
Approximation of the Gutkin+2016 continuum: the analytic continuum is free-free plus
two-photon emission only, with no free-bound jump at the Lyman edge. Valid for the
wavelength range and the photoionized gas that the analytic model describes.

Approximation of the Lyman-alpha channel: Lyα is Case B, Lyα/Hβ = 2.863 x 8.72
(Osterbrock & Ferland 2006, Table 4.2 at :math:`T = 10^4` K, :math:`n_e = 10^2
{\rm cm^{-3}}`), appended to the table's own Hβ and not read from the grid.

Units
-----
Line luminosities are returned in :math:`L_\odot` (``L_SUN``, IAU 2015). The SED is
returned in erg/s/Hz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from tengri._cache_keys import KeyPolicy, content, derive_key, exclude
from tengri._data_setup import package_or_env_data_path
from tengri.components.lyc import lyc_shares
from tengri.components.nebular._constants import _LSUN_ERG
from tengri.components.nebular._recombination_coeffs import lyc_dust_escape_factor
from tengri.components.nebular._shared import (
    _log10_sum_of_finite,
    _qh_bilinear,
    apply_lya_escape,
    compute_analytic_nebular_continuum,
    log_qh_from_normalized,
    render_nebular_lines,
)
from tengri.components.nebular.cloudy_cb19 import CB19Backend
from tengri.config.exceptions import BackendError, ParameterError
from tengri.utils.grid_interp import library_lookup
from tengri.utils.scale import pow10

#: Lyman-alpha vacuum rest wavelength [Angstrom], appended to the table's lines.
LYA_WAVE_AA: float = 1215.67

#: Case B Lyα/Hβ = (Hα/Hβ) x (Lyα/Hα) = 2.863 x 8.72 ≈ 24.97, from Osterbrock &
#: Ferland 2006, Table 4.2 at T = 10^4 K and n_e = 10^2 cm^-3.
LYA_OVER_HB_CASE_B: float = 2.863 * 8.72

#: Name of the Hβ line in the table. Its luminosity anchors Lyα.
_HBETA_NAME: str = "Hbeta"

#: Group inside the HDF5 file.
_GROUP: str = "gutkin16"

#: Default grid file, resolved the same way as the Cloudy CB19 grid.
_DEFAULT_PATH = package_or_env_data_path("gutkin16_grid.h5")


class Gutkin16MupError(ParameterError):
    """The requested upper mass cutoff is not a node of the table.

    Parameters
    ----------
    message : str
        Human-readable description naming the valid ``mup`` values.
    """


class Gutkin16GridData(NamedTuple):
    """Numeric grid for one upper mass cutoff, ready for JIT threading.

    Axes are ascending. The line names are not stored here: strings are not JAX
    types, and this tuple is threaded through ``jax.jit`` as a template argument.
    See :func:`gutkin16_line_names`.

    Parameters
    ----------
    logZ_axis : ndarray, shape (n_z,)
        Absolute log10(Z) of the table nodes [dex].
    logU_axis : ndarray, shape (n_u,)
        log10(U_S) of the table nodes [dex].
    xi_d_axis : ndarray, shape (n_xi,)
        Dust-to-metal ratio nodes [dimensionless].
    log_nH_axis : ndarray, shape (n_nh,)
        log10(n_H / cm^-3) nodes [dex].
    co_ratio_axis : ndarray, shape (n_co,)
        (C/O)/(C/O)_sun nodes [dimensionless].
    line_wavelengths_aa : ndarray, shape (n_line,)
        Vacuum rest wavelengths [Angstrom].
    log_line_per_qh : ndarray, shape (n_z, n_u, n_xi, n_nh, n_co, n_line)
        log10(L_line / Q_H) [dex re erg/photon], at the selected ``mup``.
    """

    logZ_axis: jnp.ndarray
    logU_axis: jnp.ndarray
    xi_d_axis: jnp.ndarray
    log_nH_axis: jnp.ndarray
    co_ratio_axis: jnp.ndarray
    line_wavelengths_aa: jnp.ndarray
    log_line_per_qh: jnp.ndarray


def _resolve_path(filepath: str | Path | None) -> Path:
    path = _DEFAULT_PATH if filepath is None else Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Gutkin+2016 grid not found: {path}")
    return path


def _mup_index(mup: float, mup_axis: np.ndarray) -> int:
    matches = np.flatnonzero(np.isclose(mup_axis, float(mup), rtol=0.0, atol=1e-6))
    if matches.size == 0:
        valid = ", ".join(f"{m:g}" for m in mup_axis)
        raise Gutkin16MupError(
            f"mup={float(mup):g} is not a node of the Gutkin+2016 grid; valid mup values: {valid}"
        )
    return int(matches[0])


def load_gutkin16_grid(filepath: str | Path | None = None, mup: float = 100.0) -> Gutkin16GridData:
    """Load the Gutkin+2016 grid at one upper mass cutoff.

    Parameters
    ----------
    filepath : str or Path, optional
        HDF5 grid file. ``None`` uses ``data/gutkin16_grid.h5``, resolved as the
        Cloudy CB19 grid is resolved.
    mup : float, optional
        Upper stellar mass cutoff [Msun]. Must equal one of the table's ``mup_axis``
        nodes. Default 100.

    Returns
    -------
    Gutkin16GridData
        Axes and the line table sliced at ``mup``.

    Raises
    ------
    FileNotFoundError
        If the grid file does not exist.
    Gutkin16MupError
        If ``mup`` is not a node of the table. The message names the valid set.
    """
    path = _resolve_path(filepath)
    with h5py.File(path, "r") as f:
        grp = f[_GROUP]
        mup_index = _mup_index(mup, np.asarray(grp["mup_axis"][()], dtype=float))
        log_line = np.asarray(grp["log_line_per_qh"][()], dtype=np.float64)
        # The mup dimension is axis 5 of (Z, U, xi, nH, CO, mup, line).
        log_line = log_line[:, :, :, :, :, mup_index, :]
        return Gutkin16GridData(
            logZ_axis=jnp.asarray(grp["logZ_axis"][()]),
            logU_axis=jnp.asarray(grp["logU_axis"][()]),
            xi_d_axis=jnp.asarray(grp["xi_d_axis"][()]),
            log_nH_axis=jnp.asarray(grp["log_nH_axis"][()]),
            co_ratio_axis=jnp.asarray(grp["co_ratio_axis"][()]),
            line_wavelengths_aa=jnp.asarray(grp["line_wavelengths_aa"][()]),
            log_line_per_qh=jnp.asarray(log_line),
        )


def gutkin16_line_names(filepath: str | Path | None = None) -> tuple[str, ...]:
    """Return the table's line names, in the order of its line axis.

    Parameters
    ----------
    filepath : str or Path, optional
        HDF5 grid file. ``None`` uses ``data/gutkin16_grid.h5``.

    Returns
    -------
    tuple of str, length n_line
        Line labels as stored in the file.
    """
    with h5py.File(_resolve_path(filepath), "r") as f:
        names = f[_GROUP]["line_names"][()]
    return tuple(n.decode() if isinstance(n, bytes) else str(n) for n in names)


class Gutkin16Backend:
    r"""Gutkin+2016 NEOGAL nebular line table, normalized per ionizing photon.

    Line luminosities are the stellar ionizing photon rate times a table entry:

    .. math::

        L_{\rm line} = Q_H(Z_\ast, \mathrm{SFH}) \times
        10^{\log_{10}(L_{\rm line}/Q_H)(Z_{\rm gas}, U_S, \xi_d, n_H, {\rm C/O})}

    where :math:`Q_H` is the ionizing photon rate of the stellar population
    [photons/s], summed over its young age bins in the log domain, and
    :math:`Z_{\rm gas}` is the gas-phase metallicity. The LyC share from
    ``neb_fesc`` and ``neb_fdust_frac`` multiplies :math:`Q_H` through
    :func:`~tengri.components.lyc.lyc_shares`.

    Parameters
    ----------
    mup : float, optional
        Upper stellar mass cutoff [Msun]. Must be a node of the table (100 or 300).
        Default 100.
    continuum : {"analytic", "none"}, optional
        ``"analytic"`` adds the free-free plus two-photon continuum; ``"none"``
        returns zero continuum. Default ``"analytic"``.
    grid_path : str or Path, optional
        HDF5 grid file. ``None`` uses ``data/gutkin16_grid.h5``.
    ssp_data : object, optional
        SSP grid supplying Q_H, with ``ssp_wave``, ``ssp_flux``, ``ssp_lgmet`` and
        ``ssp_lg_age_gyr``. Required to predict nebular emission.

    Attributes
    ----------
    name : str
        Registry key, ``"gutkin16"``.
    has_free_params : bool
        True: the gas metallicity, ionization parameter, density, C/O and dust
        fraction are fit parameters.
    has_continuum : bool
        True: the backend supplies a continuum through
        :meth:`predict_nebular_continuum`, which is zero under ``continuum="none"``.
    grid : Gutkin16GridData
        The loaded table, threaded into JIT by ``SEDModel._template_data_for_jit``.

    Raises
    ------
    ValueError
        If ``continuum`` is not ``"analytic"`` or ``"none"``.
    Gutkin16MupError
        If ``mup`` is not a node of the table.

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives. The table is
    passed as ``template_data`` so it is not baked into the compiled graph.

    **Gradient-safe**: yes, differentiable through ``neb_logU``, ``neb_logZ_gas``,
    ``neb_fesc``, ``neb_fdust_frac``, ``neb_log_nH``, ``neb_xi_d`` and
    ``neb_co_ratio``.

    Pure float32 is supported: the Q_H sum and the table lookup stay in log10 until
    the final conversion to :math:`L_\odot`.

    Implements the same nebular model as BEAGLE (Chevallard & Charlot 2016 [2]_),
    from the published NEOGAL tables; validated against them.

    References
    ----------
    .. [1] Gutkin, J., Charlot, S., & Bruzual, G. 2016, "Modelling the nebular
       emission from primeval to present-day star-forming galaxies," MNRAS, 462,
       1757. doi:10.1093/mnras/stw1716. arXiv:1607.06086.
    .. [2] Chevallard, J., & Charlot, S. 2016, "Modelling and interpreting spectral
       energy distributions of galaxies with BEAGLE," MNRAS, 462, 1415.
       doi:10.1093/mnras/stw1756. arXiv:1603.03037.
    .. [3] Osterbrock, D. E., & Ferland, G. J. 2006, "Astrophysics of Gaseous Nebulae
       and Active Galactic Nuclei," 2nd ed. (University Science Books), Tables 4.2
       and 4.4.
    .. [4] Plat, A., Charlot, S., Bruzual, G., et al. 2019, "Constraints on the
       production and escape of ionizing radiation from the emission-line spectra of
       metal-poor star-forming galaxies," MNRAS, 490, 978.
       doi:10.1093/mnras/stz2616.
    """

    name = "gutkin16"
    has_free_params = True
    has_continuum = True

    def __init__(
        self,
        mup: float = 100.0,
        continuum: str = "analytic",
        grid_path: str | Path | None = None,
        ssp_data=None,
    ) -> None:
        if continuum not in ("analytic", "none"):
            raise ValueError(f"continuum must be 'analytic' or 'none', got {continuum!r}")
        self.mup = float(mup)
        self.continuum = continuum
        self.grid_path = _resolve_path(grid_path)
        self.grid = load_gutkin16_grid(filepath=self.grid_path, mup=self.mup)
        self.line_names = gutkin16_line_names(self.grid_path)
        self._hbeta_index = self.line_names.index(_HBETA_NAME)
        # Young-age cut, log10(yr): only bins at or below 100 Myr ionize the gas.
        self._max_neb_log_age = 8.0
        self._qh_table = None
        self._qh_log_met = None
        self._qh_log_age = None
        self._young_idx = None
        self._log_qh_scale = 0.0
        if ssp_data is not None:
            self._precompute_qh(ssp_data)

    def _precompute_qh(self, ssp_data) -> None:
        """Build the peak-normalized Q_H table from the SSP grid, once.

        Shares the chain with the Cloudy CB19 backend (log-domain Q_H, peak
        normalization, sanitization, age axis, young-age cut).
        """
        CB19Backend._precompute_qh(self, ssp_data)

    def _get_log_qh_at(self, log_z, log_age_yr):
        """``log10 Q_H`` [dex re photons/s] at a stellar metallicity and SSP age."""
        qh_norm = _qh_bilinear(
            self._qh_table,
            self._qh_log_met,
            self._qh_log_age,
            log_z,
            log_age_yr,
            missing=1.0,
        )
        return log_qh_from_normalized(qh_norm, self._log_qh_scale)

    def _require_qh(self) -> None:
        if self._qh_table is None or self._young_idx is None:
            raise BackendError(
                "Gutkin16Backend needs ssp_data to compute Q_H; construct it with ssp_data=..."
            )

    def _log_ionizing_rate(self, ssp_weights, ssp_log_ages_yr, log_z):
        """``log10 sum_i w_i Q_H(Z, age_i)`` over the young SSP bins [dex re photons/s]."""
        self._require_qh()
        young = self._young_idx
        young_ages = ssp_log_ages_yr[young]
        young_w = ssp_weights[young]
        log_qh = jax.vmap(lambda la: self._get_log_qh_at(log_z, la))(young_ages)
        positive = young_w > 0
        log_w = jnp.where(positive, jnp.log10(jnp.where(positive, young_w, 1.0)), -jnp.inf)
        return _log10_sum_of_finite(log_w + log_qh)

    def _log_lyc_share(self, neb_fesc, neb_fdust_frac):
        """``log10`` of the LyC k-factor from the escape and dust shares."""
        _, f_dust, _ = lyc_shares(neb_fesc, neb_fdust_frac)
        return jnp.log10(lyc_dust_escape_factor(neb_fesc, f_dust))

    def predict_nebular_line_luminosities(
        self,
        ssp_weights: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_fesc: float = 0.0,
        neb_fesc_lya: float = 1.0,
        neb_fdust_frac: float = 0.0,
        neb_log_nH: float = 2.0,
        neb_xi_d: float = 0.3,
        neb_co_ratio: float = 1.0,
        template_data: Any | None = None,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Emission line wavelengths and luminosities, including Case B Lyα.

        .. math::

            L_{\rm line} = \sum_i w_i\, Q_H(Z_\ast, \mathrm{age}_i)\,
            10^{\log_{10}(L_{\rm line}/Q_H)}\, k(f_{\rm esc}, f_{\rm dust})

        where :math:`w_i` is the stellar mass per SSP age bin [Msun], and
        :math:`k` is the LyC k-factor. Lyα is
        :math:`L_{\rm Ly\alpha} = 2.863 \times 8.72\, L_{\rm H\beta}`, then scaled
        by ``neb_fesc_lya``.

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            Stellar mass per SSP age bin [Msun].
        ssp_log_ages_yr : array, shape (n_age,)
            log10(age / yr) of the SSP age bins [dex].
        log_z : float
            Stellar metallicity, absolute log10(Z). Sets Q_H [dex].
        neb_logU : float, optional
            log10(U_S) at the Strömgren radius [dex]. Grid range [-4, -1]. Default -3.
        neb_logZ_gas : float or None, optional
            Gas-phase metallicity, absolute log10(Z) [dex]. ``None`` uses ``log_z``.
            Default None.
        neb_fesc : float, optional
            Ionizing photon escape fraction [dimensionless, 0 to 1]. Default 0.
        neb_fesc_lya : float, optional
            Lyα escape fraction [dimensionless, 0 to 1], applied to Lyα only.
            Default 1.
        neb_fdust_frac : float, optional
            Dust share of the non-escaping ionizing budget [dimensionless, 0 to 1].
            Default 0.
        neb_log_nH : float, optional
            log10(n_H / cm^-3) [dex]. Grid range [2, 3]. Default 2.
        neb_xi_d : float, optional
            Dust-to-metal ratio [dimensionless]. Grid range [0.1, 0.5]. Default 0.3.
        neb_co_ratio : float, optional
            (C/O)/(C/O)_sun [dimensionless]. Grid range [0.1, 1.4]. Default 1.
        template_data : Gutkin16GridData, optional
            Grid threaded as a JIT argument. ``None`` uses ``self.grid``.

        Returns
        -------
        wavelengths : ndarray, shape (19,)
            Vacuum rest wavelengths [Angstrom]: the 18 table lines, then Lyα.
        luminosities : ndarray, shape (19,)
            Line luminosities [Lsun].

        Notes
        -----
        **JIT-compatible**: yes. **Gradient-safe**: yes, see the class Notes.

        Approximation of the Gutkin+2016 table lookup: node-exact monotone-cubic
        interpolation in every axis, through :func:`tengri.utils.grid_interp.library_lookup`.
        The n_H axis has two nodes, so the lookup is linear along it.
        """
        grid = template_data if template_data is not None else self.grid
        axes = (
            grid.logZ_axis,
            grid.logU_axis,
            grid.xi_d_axis,
            grid.log_nH_axis,
            grid.co_ratio_axis,
        )
        logz_abs = log_z if neb_logZ_gas is None else neb_logZ_gas
        point = (logz_abs, neb_logU, neb_xi_d, neb_log_nH, neb_co_ratio)
        log_line_per_qh = library_lookup(grid.log_line_per_qh, axes, point)

        log_qh = self._log_ionizing_rate(ssp_weights, ssp_log_ages_yr, log_z)
        log_k = self._log_lyc_share(neb_fesc, neb_fdust_frac)
        # erg/photon * photons/s = erg/s, then to Lsun. Summed in log10 throughout.
        log_lines_lsun = log_line_per_qh + log_qh + log_k - jnp.log10(_LSUN_ERG)
        log_lya_lsun = log_lines_lsun[self._hbeta_index] + np.log10(LYA_OVER_HB_CASE_B)
        log_all = jnp.concatenate([log_lines_lsun, jnp.reshape(log_lya_lsun, (1,))])

        wavelengths = jnp.concatenate(
            [jnp.asarray(grid.line_wavelengths_aa), jnp.array([LYA_WAVE_AA])]
        )
        luminosities = apply_lya_escape(pow10(log_all), wavelengths, neb_fesc_lya)
        return wavelengths, luminosities

    def predict_nebular_continuum(
        self,
        ssp_weights: jnp.ndarray,
        ssp_wave: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_fesc: float = 0.0,
        neb_fdust_frac: float = 0.0,
        **_kwargs,
    ) -> jnp.ndarray:
        r"""Nebular continuum on the SSP wavelength grid [erg/s/Hz].

        Free-free plus two-photon emission from
        ``compute_analytic_nebular_continuum``,
        driven by the absorbed ionizing photon rate :math:`k\,Q_H`. Zero under
        ``continuum="none"``.

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            Stellar mass per SSP age bin [Msun].
        ssp_wave : array, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].
        ssp_log_ages_yr : array, shape (n_age,)
            log10(age / yr) of the SSP age bins [dex].
        log_z : float
            Stellar metallicity, absolute log10(Z). Sets Q_H [dex].
        neb_fesc : float, optional
            Ionizing photon escape fraction [dimensionless, 0 to 1]. Default 0.
        neb_fdust_frac : float, optional
            Dust share of the non-escaping ionizing budget [dimensionless, 0 to 1].
            Default 0.

        Returns
        -------
        ndarray, shape (n_wave,)
            Nebular continuum spectral luminosity density [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes. **Gradient-safe**: yes.

        Approximation of the Gutkin+2016 continuum: the analytic continuum is
        free-free plus two-photon emission only, with no free-bound jump at the
        Lyman edge.
        """
        if self.continuum == "none":
            return jnp.zeros_like(jnp.asarray(ssp_wave))
        log_qh = self._log_ionizing_rate(ssp_weights, ssp_log_ages_yr, log_z)
        log_k = self._log_lyc_share(neb_fesc, neb_fdust_frac)
        return compute_analytic_nebular_continuum(
            jnp.asarray(ssp_wave), log_z_abs=log_z, log10_q_h=log_qh + log_k
        )

    def predict_nebular_sed(
        self,
        ssp_weights: jnp.ndarray,
        ssp_wave: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_fesc: float = 0.0,
        neb_fesc_lya: float = 1.0,
        neb_fdust_frac: float = 0.0,
        neb_log_nH: float = 2.0,
        neb_xi_d: float = 0.3,
        neb_co_ratio: float = 1.0,
        line_sigma_aa: float = 0.0,
        line_sigma_kms: float = 0.0,
        template_data: Any | None = None,
        **_kwargs,
    ) -> jnp.ndarray:
        """Nebular SED on the SSP wavelength grid: continuum plus rendered lines [erg/s/Hz].

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            Stellar mass per SSP age bin [Msun].
        ssp_wave : array, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].
        ssp_log_ages_yr : array, shape (n_age,)
            log10(age / yr) of the SSP age bins [dex].
        log_z : float
            Stellar metallicity, absolute log10(Z). Sets Q_H [dex].
        neb_logU, neb_logZ_gas, neb_fesc, neb_fesc_lya, neb_fdust_frac, neb_log_nH,
        neb_xi_d, neb_co_ratio, template_data
            As in :meth:`predict_nebular_line_luminosities`.
        line_sigma_aa : float, optional
            Fixed Gaussian line width [Angstrom]. Takes precedence when positive.
            Default 0.
        line_sigma_kms : float, optional
            Velocity-triweight line width [km/s]. Used when ``line_sigma_aa`` is 0.
            Default 0.

        Returns
        -------
        ndarray, shape (n_wave,)
            Nebular emission SED [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes. **Gradient-safe**: yes.

        Approximation of the Gutkin+2016 continuum: free-free plus two-photon
        emission only, with no free-bound jump at the Lyman edge. Lyα is Case B,
        :math:`L_{\rm Ly\alpha}/L_{\rm H\beta} = 2.863 \times 8.72`.
        """
        waves, lums = self.predict_nebular_line_luminosities(
            ssp_weights,
            ssp_log_ages_yr,
            log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            neb_fesc=neb_fesc,
            neb_fesc_lya=neb_fesc_lya,
            neb_fdust_frac=neb_fdust_frac,
            neb_log_nH=neb_log_nH,
            neb_xi_d=neb_xi_d,
            neb_co_ratio=neb_co_ratio,
            template_data=template_data,
        )
        lines_sed = render_nebular_lines(
            waves, lums, jnp.asarray(ssp_wave), line_sigma_aa, line_sigma_kms
        )
        continuum = self.predict_nebular_continuum(
            ssp_weights,
            ssp_wave,
            ssp_log_ages_yr,
            log_z,
            neb_fesc=neb_fesc,
            neb_fdust_frac=neb_fdust_frac,
        )
        # Lines are in Lsun; the continuum is already erg/s/Hz.
        return lines_sed * _LSUN_ERG + continuum

    def __repr__(self) -> str:
        return (
            f"Gutkin16Backend(mup={self.mup:g}, continuum={self.continuum!r}, "
            f"grid_path={str(self.grid_path)!r})"
        )

    def cache_key(self) -> tuple:
        """Return a hashable cache key for this backend's structure.

        Returns
        -------
        tuple
            Cache key derived from the grid arrays and the configuration. The Q_H
            table is a deterministic function of ``ssp_data``, which is content-keyed
            at the model level, so it is excluded here.
        """
        return derive_key(self, _GUTKIN16_BACKEND_CACHE_KEY_POLICY)


_GUTKIN16_BACKEND_CACHE_KEY_POLICY: KeyPolicy = {
    "mup": content("upper mass cutoff selects a different table slice"),
    "continuum": content("analytic or zero continuum changes the emitted SED"),
    "grid_path": content("grid file identity"),
    "grid": content("Gutkin+2016 table defines the emitted line luminosities"),
    "line_names": content("line order fixes the output index of each line"),
    "_hbeta_index": exclude("index read off line_names, already content-keyed"),
    "_max_neb_log_age": content("young-age cutoff changes which SSP bins contribute"),
    "_qh_table": exclude(
        "Q_H precompute table, a deterministic function of ssp_data "
        "(content-keyed at the model level)"
    ),
    "_qh_log_met": exclude("stashed ssp_data.ssp_lgmet, already content-keyed at the model level"),
    "_qh_log_age": exclude("stashed ssp_data age grid, already content-keyed at the model level"),
    "_young_idx": exclude("indices into the age grid above, same derivation"),
    "_log_qh_scale": exclude("normalization of _qh_table, same derivation"),
}

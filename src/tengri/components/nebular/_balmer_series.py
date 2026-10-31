r"""Case-B Balmer series H11-H50 relative to H-beta, for the lines Cue's catalog lacks.

Cue's trained catalog ends at H10 (3798.98 A vacuum); the upper levels of the
hydrogen Balmer series keep emitting up to the series limit at 3647.02 A. Their
summed flux is not negligible: at :math:`T_e = 10^4` K and :math:`n_e = 10^2`
cm\ :sup:`-3` the Case-B ratios to H-beta are :math:`R_{11} = 0.0397`,
:math:`R_{20} = 0.00662`, :math:`\sum_{n=21}^{50} R_n = 0.0560`, and the analytic
tail beyond :math:`n = 50` is about 0.020, so the series above :math:`n = 20` is
7.6% of H-beta and cannot be cut at :math:`n = 20`.

Source of the ratios
--------------------
Storey & Hummer (1995) [1]_ tabulate the hydrogen Case-B emissivities
:math:`j_{n \to 2}` through :math:`n = 50`. The ratio table below is

.. math::

    R_n = \frac{j(\mathrm{H}n)}{j(\mathrm{H}\beta)}
        = \frac{j_{n \to 2}(T_e, n_e)}{j_{4 \to 2}(T_e, n_e)},

with :math:`T_e` in K and :math:`n_e` in cm\ :sup:`-3`, evaluated once at
:math:`(10^4\ \mathrm{K}, 10^2\ \mathrm{cm^{-3}})` (the Cue default
``gas_logn = 2``) with PyNeb 1.1.30 [2]_::

    import pyneb as pn

    H1 = pn.RecAtom("H", 1)
    den = H1.getEmissivity(tem=1e4, den=1e2, lev_i=4, lev_j=2)
    R = {n: H1.getEmissivity(tem=1e4, den=1e2, lev_i=n, lev_j=2) / den for n in range(11, 51)}

The luminosity of each added line is :math:`L(\mathrm{H}n) = R_n\,L(\mathrm{H}\beta)`
with :math:`L(\mathrm{H}\beta)` the emulator's H-beta after the ionizing-photon
escape and dust factors, so every added line carries exactly the factors the
native lines carry.

Approximation flags
-------------------
Fixed :math:`(T_e, n_e)`: the ratios do not follow the nebula's electron
temperature or density. The relative change of :math:`R_n` against the shipped
values, over the three other conditions (5000 K and 20000 K at
:math:`n_e = 10^2`; :math:`n_e = 10^3` at 10^4 K), is the maximum below
(``BALMER_RATIO_SENSITIVITY``): at most 6.7% for :math:`n \le 24`, rising to
27% at :math:`n = 50` (5000 K). Per condition the maxima over :math:`n = 11` to
50 are 27.0% (5000 K, at :math:`n = 50`), 20.0% (20000 K, at :math:`n = 44`)
and 25.7% (:math:`n_e = 10^3`, at :math:`n = 42`).

Truncation: the series stops at :math:`n = 50`, the last level tabulated in
[1]_; the flux beyond is about 2% of H-beta (power-law extrapolation of
:math:`R_n \propto n^{-2.44}` fitted between :math:`n = 30` and 50, summed
analytically) and is not added.

Blending: the line spacing falls below the R = 3000 resolution element
(1.220 A at 3660 A) at :math:`n = 29`; H30 lies 16.28 A and H50 5.85 A
redward of the Balmer limit (3647.018 A). The high members therefore blend into
the pseudo-continuum below the limit at that resolution; the flux is
conserved, only the individual lines are not resolved.

Wavelengths
-----------
Vacuum Ritz wavelengths of the :math:`n \to 2` transitions,
:math:`\lambda = 10^8 / (E_n - E_2)` A with the energies in cm\ :sup:`-1`,
from the NIST Atomic Spectra Database [3]_ H I energy levels
(``https://physics.nist.gov/cgi-bin/ASD/energy1.pl?spectrum=H+I&units=0&format=2&output=0&page_size=15&multiplet_ordered=0&conf_out=on&term_out=on&level_out=on&unc_out=1&j_out=on&g_out=on&submit=Retrieve+Data``,
fetched 2026-10-31). :math:`E_n` is the g-weighted centroid of the fine-structure
components of shell :math:`n` for :math:`n \le 12`, the NIST shell-averaged level
for :math:`n = 13`-40, and :math:`E_\infty\,(1 - 1/n^2)` with the NIST series
limit :math:`E_\infty = 109678.77174` cm\ :sup:`-1` for :math:`n = 41`-50 (residual of that
form against the NIST levels at :math:`n = 13`-40: at most 0.0064 cm\ :sup:`-1`,
0.001 A). :math:`E_2 = 82259.1109` cm\ :sup:`-1` is the centroid of the n = 2
fine-structure components, whose shift against the bare Rydberg formula is
about 1e-5 relative (0.03 A at H10); the bare Rydberg constant
109677.583 cm\ :sup:`-1` sits 0.035 A off at :math:`n \ge 13`, which is why the NIST limit
is the anchor. Validation: the H3 to H10 wavelengths agree with Cue's catalog
(Cloudy air labels converted by
:func:`tengri.components.nebular._line_ingest.catalog_air_to_vacuum`, which
calls :func:`tengri.utils.air_vacuum.air_to_vac`) to 0.0041 A at most, and the H-alpha
to H8 wavelengths agree with FastSpecFit's ``emlines.ecsv`` to 0.0295 A.

Model sigma
-----------
:func:`balmer_series_sigma_frac` returns the fractional 1-sigma of each added
line, the emulator's H-beta sigma combined in quadrature with that line's
condition sensitivity,

.. math::

    \sigma_n = \sqrt{\sigma_{\mathrm{H}\beta}^2 + s_n^2},

where :math:`s_n` is the maximum relative change of :math:`R_n` over the
other three conditions (dimensionless). This is the stated approximation of
the fixed-condition ratios, not a measured emulator error.

References
----------
.. [1] P. J. Storey and D. G. Hummer, "Recombination line intensities for
   hydrogenic ions - IV. Total recombination coefficients and machine-readable
   tables for Z = 1 to 8," MNRAS, 272, 41 (1995).
   https://doi.org/10.1093/mnras/272.1.41
.. [2] V. Luridiana, C. Morisset, and R. A. Shaw, "PyNeb: A Python Package
   for Analyzing Emission Lines from Ionized Nebulae," A&A, 573, A42 (2015).
   arXiv:1412.6345. https://doi.org/10.1051/0004-6361/201323152
.. [3] A. Kramida, Yu. Ralchenko, J. Reader and NIST ASD Team, "NIST Atomic
   Spectra Database," National Institute of Standards and Technology,
   Gaithersburg, MD. https://physics.nist.gov/asd
"""

from __future__ import annotations

import numpy as np

#: First Balmer upper level absent from Cue's catalog (its last native line is H10).
BALMER_SERIES_N_FIRST: int = 11

#: Last Balmer upper level tabulated by Storey & Hummer (1995) and shipped here.
BALMER_SERIES_N_LAST: int = 50

#: Default ``balmer_series_n_max`` of the Cue backend: every level tabulated.
BALMER_SERIES_N_MAX_DEFAULT: int = BALMER_SERIES_N_LAST

#: Series limit of the Balmer lines [Angstrom, vacuum]: 1e8 / (E_inf - E_2) from NIST ASD.
BALMER_LIMIT_VACUUM_AA: float = 3647.0181

#: Vacuum Ritz wavelength of H-beta [Angstrom] from the NIST ASD levels, the anchor line.
HBETA_VACUUM_AA: float = 4862.6805

#: ``(n_upper, vacuum_wavelength_aa, ratio)``: Case-B j(Hn)/j(Hbeta) at 1e4 K, 1e2 cm^-3.
BALMER_CASE_B_RATIOS_TO_HBETA: tuple[tuple[int, float, float], ...] = (
    (11, 3771.7016, 3.9716599e-02),
    (12, 3751.2178, 3.0534413e-02),
    (13, 3735.4299, 2.3991903e-02),
    (14, 3722.9970, 1.9206478e-02),
    (15, 3713.0269, 1.5611336e-02),
    (16, 3704.9068, 1.2866397e-02),
    (17, 3698.2039, 1.0736842e-02),
    (18, 3692.6054, 9.0526316e-03),
    (19, 3687.8807, 7.7068826e-03),
    (20, 3683.8564, 6.6178138e-03),
    (21, 3680.4001, 5.7271255e-03),
    (22, 3677.4097, 4.9927126e-03),
    (23, 3674.8047, 4.3821862e-03),
    (24, 3672.5215, 3.8696356e-03),
    (25, 3670.5092, 3.4380567e-03),
    (26, 3668.7264, 3.0700574e-03),
    (27, 3667.1394, 2.7578947e-03),
    (28, 3665.7206, 2.4882591e-03),
    (29, 3664.4470, 2.2566802e-03),
    (30, 3663.2993, 2.0550607e-03),
    (31, 3662.2615, 1.8793522e-03),
    (32, 3661.3200, 1.7255061e-03),
    (33, 3660.4632, 1.5902834e-03),
    (34, 3659.6813, 1.4704453e-03),
    (35, 3658.9656, 1.3643725e-03),
    (36, 3658.3091, 1.2696356e-03),
    (37, 3657.7053, 1.1854251e-03),
    (38, 3657.1486, 1.1093117e-03),
    (39, 3656.6344, 1.0412955e-03),
    (40, 3656.1584, 9.7894737e-04),
    (41, 3655.7170, 9.2307692e-04),
    (42, 3655.3068, 8.7287449e-04),
    (43, 3654.9249, 8.2672065e-04),
    (44, 3654.5689, 7.8404858e-04),
    (45, 3654.2364, 7.4518219e-04),
    (46, 3653.9254, 7.0939271e-04),
    (47, 3653.6340, 6.7619433e-04),
    (48, 3653.3608, 6.4542510e-04),
    (49, 3653.1041, 6.1676113e-04),
    (50, 3652.8627, 5.8987854e-04),
)

#: ``(n_upper, s_n)``: maximum relative change of R_n over the conditions
#: (5000 K, 1e2), (20000 K, 1e2) and (1e4 K, 1e3) [K, cm^-3].
BALMER_RATIO_SENSITIVITY: tuple[tuple[int, float], ...] = (
    (11, 0.0260),
    (12, 0.0238),
    (13, 0.0214),
    (14, 0.0195),
    (15, 0.0172),
    (16, 0.0150),
    (17, 0.0131),
    (18, 0.0145),
    (19, 0.0196),
    (20, 0.0262),
    (21, 0.0345),
    (22, 0.0440),
    (23, 0.0546),
    (24, 0.0665),
    (25, 0.0788),
    (26, 0.0925),
    (27, 0.1057),
    (28, 0.1199),
    (29, 0.1338),
    (30, 0.1483),
    (31, 0.1627),
    (32, 0.1769),
    (33, 0.1900),
    (34, 0.2034),
    (35, 0.2152),
    (36, 0.2263),
    (37, 0.2350),
    (38, 0.2432),
    (39, 0.2491),
    (40, 0.2544),
    (41, 0.2567),
    (42, 0.2568),
    (43, 0.2556),
    (44, 0.2527),
    (45, 0.2487),
    (46, 0.2444),
    (47, 0.2386),
    (48, 0.2450),
    (49, 0.2579),
    (50, 0.2705),
)

#: ``(n_upper, vacuum_wavelength_aa)`` for H-alpha to H10 from the same NIST levels, for
#: consistency checks against catalogs that already carry these lines.
BALMER_RITZ_VACUUM_AA_N3_TO_N10: tuple[tuple[int, float], ...] = (
    (3, 6564.6098),
    (4, 4862.6805),
    (5, 4341.6821),
    (6, 4102.8912),
    (7, 3971.1953),
    (8, 3890.1509),
    (9, 3836.4722),
    (10, 3798.9761),
)


def validate_balmer_series_n_max(value: object) -> int | None:
    """Validate ``balmer_series_n_max``: ``None`` (off) or an int from 10 to 50.

    Parameters
    ----------
    value : int or None
        Highest Balmer upper level added; ``None`` adds none. 10 also adds none
        (H10 is Cue's own last line).

    Returns
    -------
    int or None
        The validated value.

    Raises
    ------
    ValueError
        If ``value`` is not ``None`` or an integer in ``[10, 50]``.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"balmer_series_n_max must be None or an integer in "
            f"[{BALMER_SERIES_N_FIRST - 1}, {BALMER_SERIES_N_LAST}], got {value!r}."
        )
    if not BALMER_SERIES_N_FIRST - 1 <= int(value) <= BALMER_SERIES_N_LAST:
        raise ValueError(
            f"balmer_series_n_max={int(value)} is outside [{BALMER_SERIES_N_FIRST - 1}, "
            f"{BALMER_SERIES_N_LAST}]: Storey & Hummer (1995) tabulate the series through n = 50 "
            "and Cue's own catalog ends at H10."
        )
    return int(value)


def balmer_series_arrays(
    n_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The added Balmer lines up to ``n_max`` as plain numpy arrays.

    Parameters
    ----------
    n_max : int
        Highest upper level, from 10 to 50.

    Returns
    -------
    n_upper : ndarray, shape (n_added,), int
        Upper levels 11 to ``n_max``.
    wavelength : ndarray, shape (n_added,)
        Vacuum wavelengths [Angstrom].
    ratio : ndarray, shape (n_added,)
        Case-B j(Hn)/j(Hbeta) [dimensionless] at 1e4 K and 1e2 cm^-3.
    sensitivity : ndarray, shape (n_added,)
        Maximum relative change of the ratio over the other three conditions
        [dimensionless].
    """
    keep = [i for i, row in enumerate(BALMER_CASE_B_RATIOS_TO_HBETA) if row[0] <= n_max]
    rows = [BALMER_CASE_B_RATIOS_TO_HBETA[i] for i in keep]
    sens = [BALMER_RATIO_SENSITIVITY[i][1] for i in keep]
    return (
        np.array([r[0] for r in rows], dtype=np.int64),
        np.array([r[1] for r in rows], dtype=np.float64),
        np.array([r[2] for r in rows], dtype=np.float64),
        np.array(sens, dtype=np.float64),
    )


def balmer_series_sigma_frac(hbeta_sigma_frac: float, sensitivity: np.ndarray) -> np.ndarray:
    """Fractional 1-sigma of each added line: H-beta's sigma plus the condition sensitivity.

    Parameters
    ----------
    hbeta_sigma_frac : float
        The emulator's fractional 1-sigma on H-beta [dimensionless].
    sensitivity : ndarray, shape (n_added,)
        ``s_n`` from :func:`balmer_series_arrays` [dimensionless].

    Returns
    -------
    ndarray, shape (n_added,)
        ``sqrt(hbeta_sigma_frac**2 + s_n**2)`` [dimensionless].
    """
    return np.sqrt(hbeta_sigma_frac**2 + np.asarray(sensitivity) ** 2)

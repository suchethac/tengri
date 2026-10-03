# SPDX-License-Identifier: BSD-3-Clause
"""AGB circumstellar dust-shell weighting (Villaume, Conroy & Johnson 2015).

FSPS bakes the circumstellar dust-shell reprocessing of thermally-pulsing
AGB (TP-AGB) stars into every MIST SSP spectrum at its own default weight
(``agb_dust=1.0``, which scales the DUSTY shell optical depth tau(1 micron)
of each TP-AGB template). This module adds a runtime-tunable lever on that
weight: the top-level ``agb_dust={'type': 'fsps_shell', 'weight': ...}``
grammar group, with one free parameter ``agb_dust_weight`` (default
``Fixed(1.0)`` -- the grid as shipped -- free ``Uniform(0, 3)``).

Physics
-------
The corrected SSP spectrum is the shipped grid times a resampled ratio
template measured directly from FSPS:

.. math::

    L_\\nu(Z, t, \\lambda; w) = L_\\nu^{\\mathrm{grid}}(Z, t, \\lambda)
        \\cdot R(w; Z, t, \\lambda)

    R(w; Z, t, \\lambda) = \\frac{f_{\\mathrm{FSPS}}(\\mathrm{agb\\_dust}=w)}
        {f_{\\mathrm{FSPS}}(\\mathrm{agb\\_dust}=1)}

measured at FSPS's own MIST metallicity and age nodes (``scripts/
generate_agb_dust_shell_ratios.py``) and resampled onto the loaded SSP
grid's (Z, age, lambda) axes: linearly in log10(Z) and log10(age), and (for
the wavelength axis) linearly in log10(lambda). ``R = 1`` exactly outside the
stored wavelength window and at w=1.

**Provenance and scope.** The ratio was computed with FSPS 0.4.7 (libfsps
v3.2-61-g82a8735) using MIST isochrones, the MILES spectral library and a
Chabrier IMF, and is reused unchanged for the ``c3k_a`` grids and for the
Kroupa and Salpeter IMFs. The IMF dependence was measured only at Z_sun, for
ages 0.3, 1 and 3 Gyr and w = 0 and 3: at most 4e-3 (Kroupa) and 1.9e-2
(Salpeter) in R. The spectral-library dependence cannot be measured with this
FSPS build (MILES only) and is a stated limitation.

**The weight ladder.** R(w) is neither linear nor smooth in w, so the file
stores 13 planes (w = 0, 1/1024, 1/128, 1/8, 7/32, 5/16, 7/16, 9/16, 13/16,
5/4, 13/8, 2, 3; w = 1 is the exact identity) and interpolates linearly
between them. Error of that interpolation against direct FSPS at the midpoint
of every interval (max over 2-30 um, Z_sun, 0.3/1/3 Gyr):
``[0, 1/1024]`` 3.6 % over 2-30 um (4.0 % over 0.3-100 um),
``[1/1024, 1/128]`` 0.06 %, ``[1/128, 1/8]`` 1.6 %,
``[1/8, 7/32]`` 1.1 %, ``[7/32, 5/16]`` 0.7 %, ``[5/16, 7/16]`` 0.7 %,
``[7/16, 9/16]`` 0.7 %, ``[9/16, 13/16]`` 1.0 %, ``[13/16, 1]`` 0.6 %,
``[1, 5/4]`` 0.5 %, ``[5/4, 13/8]`` 0.8 %, ``[13/8, 2]`` 0.4 %,
``[2, 3]`` 1.1 %. The first interval is limited by a near-discontinuity of R at
w = 0 (FSPS switches the shell model off there): the midpoint error of
``[0, h]`` falls only slowly with ``h`` (6.2 % at 1/256, 5.4 % at 1/512,
3.6 % at 1/1024), so no plane spacing resolves it and it affects 0.03 % of
the prior range: a weight near 0 carries a few-percent error in the shell's
ratio. The
interpolant has a kink at every stored weight, so the gradient with respect to
``agb_dust_weight`` is the slope of the bracketing segment and is
discontinuous across a node (at w = 1, where R peaks, it changes sign at
10 um).

**Window and extremes.** R is stored exactly as FSPS gives it, with a flux
guard (R = 1 where the w=1 spectrum is below 1e-6 of its own peak) and no
clamping: R spans 0.055 (far infrared at w = 0) to 236 (old, metal-poor
populations at 1200-1500 A, where the w=1 flux is ~1e-6 of the peak). The
stored window is 284 A to 3.4e7 A, where ``|R - 1| > 1e-4`` anywhere; its
lower edge is set by a few 45-100 Myr populations at low metallicity at the
threshold (ionizing-wavelength flux), not by shell absorption, which acts
above ~1 micron. Nebular Q_H tables built from the cube are unaffected to
below 1e-4 dex up to 10 Myr, and the constant-SFH integrated rate changes by
less than 1e-5.

The correction multiplies the SSP flux cube *before* the metallicity
interpolation and the SFH age-weight sum (``StellarSEDComponent.apply``'s
``ssp_flux_for_csp`` seam), so it is exact per SSP and differentiable in w,
for every downstream consumer (exact spectrum, exact photometry, and every
precompute LUT).

- ``weight`` **Fixed**: the ratio is baked directly into a new ``SSPData``
  at ``SEDModel.__init__`` time, before any LUT is built, so the
  exact path and every precompute LUT (``WavePrecomp``, ``SpectrumPrecomp``,
  ``FeaturePrecomp``) stay bit-exact with no extra runtime cost.
- ``weight`` **free**: baking is impossible (the weight is only known per
  sample), so the ratio is instead applied live inside
  ``StellarSEDComponent.apply`` (and the SED-free ionizing rate) from a
  resampled template threaded onto the component at construction time. Every
  table built once from the SSP cube refuses at build time, naming the exact
  path: ``WavePrecomp``, ``SpectrumPrecomp``, the ``FeaturePrecomp`` SSP window
  table and ``approx=True`` index / line-flux measurement. The ``FeaturePrecomp``
  Cue grid reads the live ionizing rate and stays valid. ``approx="auto"``
  resolves to the exact path, for single fits and for batch fits.

Only supported on FSPS MIST grids: FSPS's ``add_agb_dust_model`` Fortran
routine refuses non-MIST isochrones, so the shipped ratio template (measured
against MIST) is valid only for tengri's ``fsps_mist_*`` grids. The grid's
isochrone tag is read from ``ssp_data.source`` (the loaded file's stem),
the same mechanism the Cloudy-grid selection (#2426) uses to detect the SSP
family: no filename guessing in this module.

References
----------
.. [1] Villaume, A., Conroy, C., & Johnson, B. D. (2015), "Circumstellar Dust
       around AGB Stars and Implications for Infrared Emission from
       Galaxies", ApJ, 806, 82.
       arXiv:1504.00900, doi:10.1088/0004-637X/806/1/82
.. [2] Conroy, C., Gunn, J. E., & White, M. (2009), "The Propagation of
       Uncertainties in Stellar Population Synthesis Modeling. I.",
       ApJ, 699, 486. arXiv:0905.0391, doi:10.1088/0004-637X/699/1/486
"""

from __future__ import annotations

from typing import NamedTuple

import h5py
import jax.numpy as jnp
import numpy as np

from tengri.components.sed_model_component import SEDModelComponent
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.parameters.priors import Uniform
from tengri.protocols.component import ParamDeclaration

#: fsps_mist_* SSP grids this module supports (MIST isochrones only --
#: FSPS's ``add_agb_dust_model`` Fortran routine refuses any other
#: isochrone family).
SUPPORTED_GRIDS: tuple[str, ...] = (
    "fsps_mist_miles_chabrier",
    "fsps_mist_c3k_a_chabrier",
    "fsps_mist_c3k_a_kroupa",
    "fsps_mist_c3k_a_salpeter",
)

_TEMPLATE_FILENAME = "agb_dust_shell_ratios_mist.h5"


class AGBDustShell(SEDModelComponent):
    """AGB circumstellar dust-shell weighting component.

    Declares the single free parameter ``agb_dust_weight`` and the MIST-grid
    refusal / template loading logic. This component does not add to the
    additive SED chain: like :class:`~tengri.components.stellar.component.
    StellarSEDComponent`, it modifies the SSP flux cube itself, which no
    SSP-additive component can do through the normal ``predict(p, sed_in,
    wave, **inputs)`` contract (see ``docs/dev/model-construction.md``,
    "Advanced fallback"). :meth:`predict` is therefore a pass-through;
    the physics is applied by :func:`bake_agb_dust_shell` (Fixed weight, at
    :class:`~tengri.SEDModel` construction) or by
    :func:`prepare_free_agb_dust_shell` + :func:`agb_dust_ratio` (free
    weight, threaded into ``StellarSEDComponent`` and applied live in its
    ``apply``).

    Notes
    -----
    **JIT-compatible**: :func:`agb_dust_ratio` is pure JAX and differentiable
    in ``w``; template loading and resampling (:func:`load_agb_dust_shell_
    template`, :func:`resample_agb_dust_shell`) are eager numpy, run once at
    build time.
    """

    name = "fsps_shell"
    parameter_prefix = "agb_dust_"

    #: Single declaration serving both roles (ADR-0011): the registry
    #: default is ``Fixed(weight.default)`` = ``Fixed(1.0)`` (the grid as
    #: shipped, agb_dust=1 baked in by FSPS); freeing it (``all_params:
    #: FREE`` or an explicit prior) uses this same ``Uniform(0, 3)`` range.
    weight = Uniform(
        0.0,
        3.0,
        "AGB circumstellar dust-shell weight: FSPS's agb_dust parameter, "
        "scaling tau(1 micron) of every TP-AGB star's DUSTY shell template. "
        "1.0 (default) reproduces the shipped SSP grid exactly; 0 disables "
        "the shell; >1 strengthens it.",
        units="",
        default=1.0,
    )

    inputs: dict[str, str] = {}  # noqa: RUF012
    outputs: dict[str, str] = {}  # noqa: RUF012

    def load(self, wave: jnp.ndarray | None = None) -> AGBDustShellTemplate:
        """Load the ratio template from disk.

        Parameters
        ----------
        wave : ndarray, optional
            Unused; the template carries its own native wavelength window
            and is resampled onto a specific SSP grid by
            :func:`resample_agb_dust_shell`, not here.

        Returns
        -------
        AGBDustShellTemplate
            The raw loaded template (not yet resampled to any SSP grid).
        """
        del wave
        return load_agb_dust_shell_template()

    def predict(self, p, sed_in, wave, **inputs):
        """Pass-through: the ratio is applied at the stellar SSP-cube seam.

        Parameters
        ----------
        p : Mapping
            Parameters with the ``agb_dust_`` prefix stripped (unused here;
            read directly from the flat params dict by
            ``StellarSEDComponent.apply`` instead, since the correction must
            be applied to the SSP cube, not the assembled SED).
        sed_in : ndarray, shape (n_wave,)
            Rest-frame L_nu [erg/s/Hz] from upstream.
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].

        Returns
        -------
        tuple[ndarray, dict]
            ``(sed_in, {})`` unchanged.

        Notes
        -----
        **JIT-compatible**: yes (identity).
        """
        del p, wave, inputs
        return sed_in, {}


#: Single source of truth for ``agb_dust_weight``'s free-parameter
#: declaration, shared with :attr:`AGBDustShell.weight` (same object) so the
#: component's auto-discovery and the legacy flat-param-registry bucket
#: (``tengri.parameters._builders``) cannot drift apart.
PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        name="agb_dust_weight",
        prior=AGBDustShell._priors["weight"],
        description=AGBDustShell._priors["weight"].description,
        bound_check=lambda lo, hi: 0.0 <= lo <= hi <= 3.0,
        bound_error=(
            "must be within [0, 3] (FSPS's agb_dust range this template was measured over)"
        ),
        units=AGBDustShell._priors["weight"].units,
    ),
)


def _require_mist_grid(ssp_data: SSPData) -> None:
    """Refuse at build time when ``ssp_data`` is not an FSPS MIST grid.

    Parameters
    ----------
    ssp_data : SSPData
        The loaded SSP grid.

    Raises
    ------
    ValueError
        If the grid's ``source`` does not name the ``mist`` isochrone.

    Notes
    -----
    Uses the same isochrone-tag detection as the Cloudy-grid selection
    (#2426, ``tengri.parameters.parameters._neb_isochrone_tag_from_ssp``):
    whole ``_``-delimited tokens of ``ssp_data.source`` (the loaded file's
    stem), not a filename guess.
    """
    from tengri.parameters.parameters import _neb_isochrone_tag_from_ssp

    tag = _neb_isochrone_tag_from_ssp(ssp_data)
    if tag != "mist":
        source = getattr(ssp_data, "source", "") or "<unknown>"
        raise ValueError(
            "agb_dust={'type': 'fsps_shell', ...} requires an FSPS MIST SSP grid: "
            "FSPS's add_agb_dust_model Fortran routine refuses non-MIST isochrones, "
            f"so this correction's ratio template is MIST-only. Got ssp_data.source={source!r}"
            f" (isochrone tag={tag!r}). Supported grids: {', '.join(SUPPORTED_GRIDS)}."
        )


class AGBDustShellTemplate(NamedTuple):
    """Raw AGB dust-shell ratio template, as loaded from disk.

    Attributes
    ----------
    log_z : ndarray, shape (n_met,)
        Absolute log10(Z) metallicity nodes (FSPS's native MIST zlegend),
        matching the convention of ``SSPData.ssp_lgmet``.
    log_age_yr : ndarray, shape (n_age,)
        log10(age / yr) nodes, matching ``SSPData.ssp_lg_age_gyr + 9``.
    wave_angstrom : ndarray, shape (n_wave_window,)
        Rest-frame wavelength nodes [Angstrom] of the stored window (where
        the shell measurably departs from unity); ``R = 1`` outside it.
    weights : ndarray, shape (n_w,)
        Ascending weight nodes of the stored planes, including ``w = 1``
        (reconstructed by the loader as an exact identity plane).
    ratio_planes : ndarray, shape (n_w, n_met, n_age, n_wave_window)
        ``R`` at every weight node [dimensionless]; interpolated linearly in
        ``w`` between nodes.
    """

    log_z: np.ndarray
    log_age_yr: np.ndarray
    wave_angstrom: np.ndarray
    weights: np.ndarray
    ratio_planes: np.ndarray


class ResampledAGBDustShell(NamedTuple):
    """AGB dust-shell template resampled onto one SSP grid's own axes.

    A plain ``NamedTuple`` (JAX's native pytree support, no explicit
    registration needed) so this can be threaded through
    ``StellarSEDComponent.agb_dust_ratio`` as a JIT-traced leaf set.

    Attributes
    ----------
    weights : ndarray, shape (n_w,)
        Static ascending weight nodes (including ``w = 1``).
    ratio_planes : ndarray, shape (n_w, n_met_ssp, n_age_ssp, n_wave_ssp)
        ``R`` at every weight node on the SSP grid's axes [dimensionless];
        exactly 1 outside the measured window and at ``w = 1``, so multiplying
        the SSP cube by :func:`agb_dust_ratio` is a no-op there.
    """

    weights: np.ndarray
    ratio_planes: jnp.ndarray


def load_agb_dust_shell_template() -> AGBDustShellTemplate:
    """Load the AGB dust-shell ratio template through the data locator.

    Returns
    -------
    AGBDustShellTemplate
        Raw template, not yet resampled to any SSP grid.

    Raises
    ------
    FileNotFoundError
        If the template is not in any directory :func:`tengri.data_path`
        searches. Regenerate with ``scripts/generate_agb_dust_shell_ratios.py``.

    Notes
    -----
    **JIT-compatible**: not applicable; eager file I/O, called once at
    :class:`tengri.SEDModel` construction.
    """
    from tengri import data_path

    path = data_path(_TEMPLATE_FILENAME)
    with h5py.File(path, "r") as f:
        log_z = np.asarray(f["log_z"][:], dtype=np.float64)
        log_age_yr = np.asarray(f["log_age_yr"][:], dtype=np.float64)
        wave_angstrom = np.asarray(f["wave_angstrom"][:], dtype=np.float64)
        weights_stored = np.asarray(f["weights"][:], dtype=np.float64)
        ratio_stored = np.asarray(f["ratio"][:], dtype=np.float32)

    # The file stores only the non-trivial weights: R(w=1) is the exact
    # identity by construction, so the interpolation grid always has a node
    # exactly at w=1.
    insert_at = int(np.searchsorted(weights_stored, 1.0))
    weights_full = np.insert(weights_stored, insert_at, 1.0)
    ones_plane = np.ones((1, *ratio_stored.shape[1:]), dtype=np.float32)
    ratio_full = np.insert(ratio_stored, insert_at, ones_plane, axis=0)

    return AGBDustShellTemplate(
        log_z=log_z,
        log_age_yr=log_age_yr,
        wave_angstrom=wave_angstrom,
        weights=weights_full,
        ratio_planes=ratio_full,
    )


def _interp_1d_clamped(
    x_new: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    axis: int,
    outside: float | None = None,
) -> np.ndarray:
    """Linear interpolation along one axis, held constant outside the range.

    Parameters
    ----------
    x_new : ndarray, shape (n_new,)
        Query coordinates.
    x_ref : ndarray, shape (n_ref,)
        Strictly increasing reference coordinates.
    y_ref : ndarray
        Reference values; interpolated along ``axis``.
    axis : int
        Axis of ``y_ref`` corresponding to ``x_ref``.
    outside : float, optional
        Value outside ``[x_ref[0], x_ref[-1]]``. Default: the edge value.

    Returns
    -------
    ndarray
        ``y_ref`` resampled onto ``x_new`` along ``axis``.
    """
    y_ref = np.moveaxis(y_ref, axis, -1)
    flat = y_ref.reshape(-1, y_ref.shape[-1])
    out = np.empty((flat.shape[0], x_new.shape[0]), dtype=np.float32)
    for i in range(flat.shape[0]):
        left = flat[i, 0] if outside is None else outside
        right = flat[i, -1] if outside is None else outside
        out[i] = np.interp(x_new, x_ref, flat[i], left=left, right=right)
    out = out.reshape((*y_ref.shape[:-1], x_new.shape[0]))
    return np.moveaxis(out, -1, axis)


def resample_agb_dust_shell(
    template: AGBDustShellTemplate,
    ssp_log_z: np.ndarray,
    ssp_log_age_yr: np.ndarray,
    ssp_wave: np.ndarray,
) -> ResampledAGBDustShell:
    """Resample the ratio template onto one SSP grid's own (Z, age, wave) axes.

    Linear interpolation in log10(Z), log10(age/yr), and log10(wavelength).
    The metallicity and age axes hold the edge value beyond the template's
    range; the wavelength axis is exactly 1 outside the stored window (the
    window is, by construction, where ``|R - 1|`` exceeds the generation
    threshold anywhere, so ``R = 1`` to within that threshold outside it).

    Parameters
    ----------
    template : AGBDustShellTemplate
        Raw template from :func:`load_agb_dust_shell_template`.
    ssp_log_z : ndarray, shape (n_met_ssp,)
        Absolute log10(Z) nodes of the target SSP grid (``ssp_data.ssp_lgmet``).
    ssp_log_age_yr : ndarray, shape (n_age_ssp,)
        log10(age/yr) nodes of the target SSP grid
        (``ssp_data.ssp_lg_age_gyr + 9``).
    ssp_wave : ndarray, shape (n_wave_ssp,)
        Rest-frame wavelength grid [Angstrom] of the target SSP grid.

    Returns
    -------
    ResampledAGBDustShell
        Planes with shape ``(n_w, n_met_ssp, n_age_ssp, n_wave_ssp)``, ready
        for :func:`agb_dust_ratio`; exactly 1 outside the template's measured
        wavelength window.

    Notes
    -----
    **JIT-compatible**: not applicable; eager numpy, run once at build time
    (``SEDModel.__init__`` for a Fixed weight, or ``StellarSEDComponent``
    construction for a free weight).
    """
    ssp_log_z = np.asarray(ssp_log_z, dtype=np.float64)
    ssp_log_age_yr = np.asarray(ssp_log_age_yr, dtype=np.float64)
    ssp_wave = np.asarray(ssp_wave, dtype=np.float64)

    log_wave_template = np.log10(template.wave_angstrom)
    log_wave_ssp = np.log10(ssp_wave)

    def _resample_cube(cube: np.ndarray) -> np.ndarray:
        # (n_met_t, n_age_t, n_wave_t) -> (n_met_ssp, n_age_ssp, n_wave_ssp)
        out = _interp_1d_clamped(log_wave_ssp, log_wave_template, cube, axis=2, outside=1.0)
        out = _interp_1d_clamped(ssp_log_age_yr, template.log_age_yr, out, axis=1)
        out = _interp_1d_clamped(ssp_log_z, template.log_z, out, axis=0)
        return out.astype(np.float32)

    n_w = template.ratio_planes.shape[0]
    resampled_planes = np.stack(
        [_resample_cube(template.ratio_planes[i]) for i in range(n_w)], axis=0
    )
    return ResampledAGBDustShell(
        weights=template.weights,
        ratio_planes=jnp.asarray(resampled_planes),
    )


def _lerp_along_weight_axis(
    w: jnp.ndarray, weights: np.ndarray, planes: jnp.ndarray
) -> jnp.ndarray:
    """Differentiable linear interpolation along ``planes``'s leading axis.

    Parameters
    ----------
    w : scalar array
        Query weight.
    weights : ndarray, shape (n_w,)
        Static (design-time) ascending node coordinates.
    planes : ndarray, shape (n_w, ...)
        Values at each node.

    Returns
    -------
    ndarray, shape planes.shape[1:]
        Linearly interpolated value at ``w``, clamped to the edge planes
        outside ``[weights[0], weights[-1]]``.

    Notes
    -----
    **JIT-compatible**: yes; ``jnp.searchsorted`` + ``jnp.take`` with a
    traced scalar index are both JIT/grad-safe. The gradient wrt ``w`` is
    the finite-difference slope between the bracketing planes.
    """
    weights_j = jnp.asarray(weights)
    idx_hi = jnp.clip(jnp.searchsorted(weights_j, w), 1, weights_j.shape[0] - 1)
    idx_lo = idx_hi - 1
    w_lo = weights_j[idx_lo]
    w_hi = weights_j[idx_hi]
    denom = jnp.where(w_hi > w_lo, w_hi - w_lo, 1.0)
    frac = jnp.where(w_hi > w_lo, (w - w_lo) / denom, 0.0)
    plane_lo = jnp.take(planes, idx_lo, axis=0)
    plane_hi = jnp.take(planes, idx_hi, axis=0)
    return plane_lo + frac * (plane_hi - plane_lo)


def agb_dust_ratio(resampled: ResampledAGBDustShell, weight: jnp.ndarray) -> jnp.ndarray:
    """Evaluate R(w; Z, age, lambda) on the resampled SSP grid.

    .. math::

        R(w) = R(w_i) + \\frac{w - w_i}{w_{i+1} - w_i}\\,[R(w_{i+1}) - R(w_i)],
        \\qquad w_i \\le w \\le w_{i+1}

    Parameters
    ----------
    resampled : ResampledAGBDustShell
        From :func:`resample_agb_dust_shell`.
    weight : scalar array
        The ``agb_dust_weight`` value [dimensionless] (traced under JIT/grad
        for a free parameter; a Python float for the Fixed/build-time path).

    Returns
    -------
    ndarray, shape (n_met_ssp, n_age_ssp, n_wave_ssp)
        Multiplicative correction, exactly 1 at ``weight == 1``.

    Notes
    -----
    Linear interpolation between the stored weight nodes (see the module
    docstring for its measured error against direct FSPS output). The
    interpolant has a kink at every stored node, so ``jax.grad`` in ``weight``
    is the slope of the bracketing segment and is discontinuous across a
    node, and R(w) itself has a step at ``w = 0`` that the first segment
    cannot resolve.

    **JIT-compatible**: yes; differentiable in ``weight`` between nodes.
    """
    return _lerp_along_weight_axis(jnp.asarray(weight), resampled.weights, resampled.ratio_planes)


def align_ratio_to_cube(ratio: jnp.ndarray, ssp_flux: jnp.ndarray) -> jnp.ndarray:
    """Broadcast the (n_met, n_age, n_wave) ratio onto an SSP flux cube.

    A 3-D cube ``(n_met, n_age, n_wave)`` takes the ratio as is. An
    alpha-enhanced 4-D cube ``(n_met, n_alpha, n_age, n_wave)`` gets an
    explicit length-1 [alpha/Fe] axis, so the ratio is applied per metallicity
    and age and is the same for every alpha slice (without the explicit axis,
    NumPy would broadcast the ratio's metallicity axis against the alpha axis).

    Parameters
    ----------
    ratio : array, shape (n_met, n_age, n_wave)
        R(w) on the SSP grid's axes [dimensionless].
    ssp_flux : array, shape (n_met, n_age, n_wave) or (n_met, n_alpha, n_age, n_wave)
        The cube the ratio multiplies.

    Returns
    -------
    array
        ``ratio`` with a [alpha/Fe] axis inserted for a 4-D cube.

    Raises
    ------
    ValueError
        If the ratio does not match the cube's metallicity, age and wavelength axes.

    Notes
    -----
    The template carries no [alpha/Fe] dependence: the ratio is taken
    independent of [alpha/Fe]. That is an assumption of the template, which was
    computed at FSPS's solar-scaled abundances, not a measurement.

    **JIT-compatible**: yes (shape logic only).
    """
    cube_axes = (ssp_flux.shape[0], *ssp_flux.shape[-2:])
    if tuple(ratio.shape) != cube_axes or ssp_flux.ndim not in (3, 4):
        raise ValueError(
            f"AGB dust-shell ratio of shape {tuple(ratio.shape)} does not match an SSP cube "
            f"of shape {tuple(ssp_flux.shape)} (expected (n_met, [n_alpha,] n_age, n_wave))."
        )
    return ratio[:, None, :, :] if ssp_flux.ndim == 4 else ratio


def bake_agb_dust_shell(ssp_data: SSPData, weight: float) -> SSPData:
    """Bake a Fixed AGB dust-shell weight into a new ``SSPData``.

    Multiplies ``ssp_data.ssp_flux`` by :func:`agb_dust_ratio` evaluated at
    the (Python float) ``weight``, once, before any LUT is built from the
    result. At ``weight == 1.0`` the multiply is by an array that is
    exactly 1 everywhere (the template's w=1 plane is exact by
    construction), which is bit-identical to not touching the grid
    (``x * 1.0 == x`` for every finite ``x`` under IEEE754).

    Parameters
    ----------
    ssp_data : SSPData
        The loaded SSP grid (an FSPS MIST grid; see :func:`_require_mist_grid`).
    weight : float
        The Fixed ``agb_dust_weight`` value.

    Returns
    -------
    SSPData
        A new ``SSPData`` (immutable update via ``_replace``) with
        ``ssp_flux`` multiplied by R(weight).

    Raises
    ------
    ValueError
        If ``ssp_data`` is not an FSPS MIST grid.

    Notes
    -----
    For an alpha-enhanced 4-D cube the ratio is applied per metallicity and age
    and is the same for every [alpha/Fe] slice (:func:`align_ratio_to_cube`).

    **JIT-compatible**: not applicable; eager, called once at
    :class:`tengri.SEDModel` construction, before any tracing.
    """
    _require_mist_grid(ssp_data)
    template = load_agb_dust_shell_template()
    resampled = resample_agb_dust_shell(
        template,
        np.asarray(ssp_data.ssp_lgmet),
        np.asarray(ssp_data.ssp_lg_age_gyr) + 9.0,
        np.asarray(ssp_data.ssp_wave),
    )
    ratio = np.asarray(agb_dust_ratio(resampled, float(weight)))
    cube = np.asarray(ssp_data.ssp_flux)
    new_flux = cube * np.asarray(align_ratio_to_cube(jnp.asarray(ratio), jnp.asarray(cube)))
    return ssp_data._replace(ssp_flux=jnp.asarray(new_flux, dtype=ssp_data.ssp_flux.dtype))


def prepare_free_agb_dust_shell(ssp_data: SSPData) -> ResampledAGBDustShell:
    """Prepare the live (per-sample) ratio template for a free weight.

    Parameters
    ----------
    ssp_data : SSPData
        The loaded SSP grid (an FSPS MIST grid; see :func:`_require_mist_grid`).

    Returns
    -------
    ResampledAGBDustShell
        Resampled onto ``ssp_data``'s own axes, ready for
        :func:`agb_dust_ratio` inside ``StellarSEDComponent.apply``.

    Raises
    ------
    ValueError
        If ``ssp_data`` is not an FSPS MIST grid.

    Notes
    -----
    **JIT-compatible**: not applicable; eager, called once at
    ``StellarSEDComponent`` construction. The returned arrays then live on
    the (frozen-dataclass) component instance as JAX constants.
    """
    _require_mist_grid(ssp_data)
    template = load_agb_dust_shell_template()
    return resample_agb_dust_shell(
        template,
        np.asarray(ssp_data.ssp_lgmet),
        np.asarray(ssp_data.ssp_lg_age_gyr) + 9.0,
        np.asarray(ssp_data.ssp_wave),
    )

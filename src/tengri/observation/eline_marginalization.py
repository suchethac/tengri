# SPDX-License-Identifier: BSD-3-Clause
"""Analytical emission-line marginalization for spectral fitting.

When fitting spectra, individual emission-line amplitudes are nuisance
parameters.  Rather than sampling them, we analytically marginalize them
out of the likelihood under a Gaussian prior, following the approach in
Prospector (Johnson et al. 2021).

Given observed data *d*, a continuum model *m*, noise *sigma*, and a
design matrix *G* whose columns are Gaussian line profiles, the model
with free line amplitudes is::

    model = m + G @ a

The marginalized log-likelihood integrates out *a* analytically:

    ln L_marg = -0.5 * (chi2_marg + prior_penalty - log_det_correction)

where chi2_marg accounts for the improvement from fitting lines, and the
prior penalty and determinant correction come from the Gaussian prior on
line amplitudes.

All functions are pure JAX, JIT-compatible, and differentiable w.r.t.
the continuum model (enabling gradient-based inference of physical
parameters).
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri.observation.eline_catalog import (
    DEFAULT_LINE_NAMES,  # noqa: F401, re-exported for convenience
    DEFAULT_LINE_WAVELENGTHS,  # noqa: F401, re-exported for convenience
)
from tengri.observation.linear_marginal import marginalize_linear_nuisance
from tengri.utils.physics_constants import C_AA
from tengri.utils.scale import log10_four_pi_dl2

# ── Public API ────────────────────────────────────────────────────


@jax.jit
def _build_eline_design_matrix_jitted(
    wave_obs: jnp.ndarray,
    line_wavelengths: jnp.ndarray,
    spectral_resolution: float,
    redshift: float,
    eline_sigma_kms: float,
    eline_delta_v_kms: float,
) -> jnp.ndarray:
    """Internal JIT-compiled implementation of build_eline_design_matrix."""
    c_kms = 299792.458

    def _single_column(lam_rest):
        """Build a normalized Gaussian profile for a single emission line."""
        lam_obs = lam_rest * (1.0 + redshift)
        # Velocity offset: convert km/s to wavelength shift
        delta_lam = lam_obs * eline_delta_v_kms / c_kms
        lam_center = lam_obs + delta_lam
        # Instrument resolution
        sigma_inst = lam_obs / (2.355 * spectral_resolution)
        # Intrinsic velocity broadening
        sigma_vel = lam_obs * eline_sigma_kms / c_kms
        # Combined sigma
        sigma = jnp.sqrt(sigma_inst**2 + sigma_vel**2)
        profile = jnp.exp(-0.5 * ((wave_obs - lam_center) / sigma) ** 2) / (
            jnp.sqrt(2.0 * jnp.pi) * sigma
        )
        return profile

    # vmap over lines -> (n_lines, n_pix), then transpose -> (n_pix, n_lines)
    columns = jax.vmap(_single_column)(line_wavelengths)
    return columns.T


def build_eline_design_matrix(
    wave_obs: jnp.ndarray,
    line_wavelengths: jnp.ndarray,
    spectral_resolution: float,
    redshift: float,
    eline_sigma_kms: float = 0.0,
    eline_delta_v_kms: float = 0.0,
) -> jnp.ndarray:
    """Build the (n_pix, n_lines) Gaussian design matrix.

    Each column *j* is a normalized Gaussian profile centered at the
    redshifted line wavelength::

        G_j(lambda) = exp(-0.5 * ((lambda - lam_j*(1+z)) / sig_j)^2)
                      / (sqrt(2*pi) * sig_j)

    where ``sig_j = sqrt(sig_inst^2 + sig_vel^2)`` combines instrument
    resolution and intrinsic velocity broadening. The profile can also
    be offset by a velocity shift (eline_delta_v_kms).

    Parameters
    ----------
    wave_obs : array, shape (n_pix,)
        Observed-frame wavelength grid [Angstrom].
    line_wavelengths : array, shape (n_lines,)
        Rest-frame vacuum wavelengths of the lines [Angstrom].
    spectral_resolution : float
        Instrument spectral resolution R = lambda / delta_lambda
        [dimensionless].
    redshift : float
        Source redshift [dimensionless].
    eline_sigma_kms : float
        Intrinsic velocity broadening [km/s]. Default 0.
    eline_delta_v_kms : float
        Velocity offset from systemic [km/s]. Default 0.

    Returns
    -------
    array, shape (n_pix, n_lines)
        Design matrix *G* [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, delegates to ``_build_eline_design_matrix_jitted``
    which is JIT-decorated. Differentiable w.r.t. all parameters.

    Each profile is normalized to unit integral (``int G_j dlam = 1``) to
    enable proper line amplitude marginalization in
    :func:`marginalize_emission_lines`.
    """
    return _build_eline_design_matrix_jitted(
        wave_obs,
        line_wavelengths,
        spectral_resolution,
        redshift,
        eline_sigma_kms,
        eline_delta_v_kms,
    )


def build_broad_design_matrix(
    wave_obs: jnp.ndarray,
    line_wavelengths: jnp.ndarray,
    spectral_resolution: float,
    redshift: float,
    broad_sigma_kms: float,
    eline_delta_v_kms: float = 0.0,
) -> jnp.ndarray:
    """Build design matrix for broad emission line component.

    Same as build_eline_design_matrix but uses broad_sigma_kms as the
    intrinsic velocity dispersion (typically 500-5000 km/s for AGN BLR).
    Instrument resolution is still added in quadrature.

    Parameters
    ----------
    wave_obs : array, shape (n_pix,)
        Observed-frame wavelength grid [Angstrom].
    line_wavelengths : array, shape (n_lines,)
        Rest-frame vacuum wavelengths of broad-line candidates [Angstrom].
    spectral_resolution : float
        Instrument spectral resolution R = lambda/delta_lambda
        [dimensionless].
    redshift : float
        Source redshift [dimensionless].
    broad_sigma_kms : float
        Broad component velocity dispersion [km/s]. Typical: 500–5000.
    eline_delta_v_kms : float
        Velocity offset of broad component from systemic [km/s]. Default 0.

    Returns
    -------
    array, shape (n_pix, n_lines)
        Broad-component design matrix G_broad [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, delegates to :func:`build_eline_design_matrix`
    and is itself JIT-decorated. Differentiable w.r.t. all parameters.

    Used for modeling broad AGN emission lines (e.g., broad H-alpha, H-beta)
    in conjunction with narrow lines from :func:`build_eline_design_matrix`.
    """
    # Delegate to build_eline_design_matrix with broad_sigma_kms as eline_sigma_kms
    return build_eline_design_matrix(
        wave_obs,
        line_wavelengths,
        spectral_resolution,
        redshift,
        eline_sigma_kms=broad_sigma_kms,
        eline_delta_v_kms=eline_delta_v_kms,
    )


# Apply JIT
build_broad_design_matrix = jax.jit(build_broad_design_matrix)


def apply_doublet_constraints(
    design_matrix: jnp.ndarray,
    constraint_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Apply doublet ratio constraints to the design matrix.

    Reduces the design matrix from shape ``(n_pix, n_lines)`` to
    ``(n_pix, n_independent)`` by enforcing fixed flux ratios between
    doublet pairs (e.g., [OIII] 5007/4959 = 2.98).

    Parameters
    ----------
    design_matrix : array, shape (n_pix, n_lines)
        Full design matrix with one column per emission line.
    constraint_matrix : array, shape (n_lines, n_independent)
        Constraint matrix from ``LineList.build_constraint_matrix()``.
        Encodes doublet ratios as a linear transformation.

    Returns
    -------
    array, shape (n_pix, n_independent)
        Constrained design matrix. Marginalizing over ``n_independent``
        amplitudes automatically enforces all doublet ratio constraints.

    Notes
    -----
    **JIT-compatible**: yes, pure matrix multiplication via ``jnp.dot``.

    Examples
    --------
    ::

        cat = LineList.default_optical()
        C = cat.build_constraint_matrix()
        G = build_eline_design_matrix(wave_obs, cat.wavelengths, R, z)
        G_eff = apply_doublet_constraints(G, C)  # (n_pix, n_independent)

    """
    return design_matrix @ constraint_matrix


def expand_constrained_amplitudes(
    a_hat: jnp.ndarray,
    a_cov: jnp.ndarray,
    constraint_matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Expand independent amplitudes back to full line set.

    After marginalization with a constrained design matrix, the posterior
    amplitudes have shape ``(n_independent,)``. This function expands them
    back to ``(n_lines,)`` by applying the constraint matrix, giving the
    flux for every emission line including constrained doublet secondaries.

    Parameters
    ----------
    a_hat : array, shape (n_independent,)
        Posterior-mean amplitudes for the independent parameters.
    a_cov : array, shape (n_independent, n_independent)
        Posterior covariance for the independent parameters.
    constraint_matrix : array, shape (n_lines, n_independent)
        Constraint matrix from ``LineList.build_constraint_matrix()``.

    Returns
    -------
    a_hat_full : ndarray, shape (n_lines,)
        Amplitudes for all lines including constrained doublet secondaries
        [erg/s] or [erg/s/Angstrom] depending on line fitting context.
    a_cov_full : ndarray, shape (n_lines, n_lines)
        Full covariance, propagated through the constraint matrix:
        ``C @ a_cov @ C.T``.

    Notes
    -----
    **JIT-compatible**: yes, matrix multiplications via ``jnp.dot``.

    Examples
    --------
    ::

        G_eff = apply_doublet_constraints(G, C)
        ln_l, a_hat_ind, a_cov_ind = marginalize_emission_lines(resid, noise, G_eff)
        a_hat_full, a_cov_full = expand_constrained_amplitudes(a_hat_ind, a_cov_ind, C)

    """
    a_hat_full = constraint_matrix @ a_hat
    a_cov_full = constraint_matrix @ a_cov @ constraint_matrix.T
    return a_hat_full, a_cov_full


class UnitLinePrior(NamedTuple):
    """Unit-normal form of a Gaussian line prior, as returned by :func:`unit_line_prior`.

    Attributes
    ----------
    design : ndarray, shape (n_pix, n_lines)
        Design matrix with each column multiplied by its prior width.
    prior_mean : ndarray, shape (n_lines,)
        Prior mean of the unit coefficients, ``mu / w``.
    prior_precision : ndarray, shape (n_lines, n_lines)
        Identity: the unit coefficients carry a unit normal prior.
    scale : ndarray, shape ()
        Coefficient scale that keeps the whitened design at order unity.
    """

    design: jnp.ndarray
    prior_mean: jnp.ndarray
    prior_precision: jnp.ndarray
    scale: jnp.ndarray


def unit_line_prior(
    design: jnp.ndarray,
    prior_width: jnp.ndarray,
    *,
    noise: jnp.ndarray,
    pix_mask: jnp.ndarray | None = None,
    prior_mean: jnp.ndarray | None = None,
) -> UnitLinePrior:
    r"""Rewrite a Gaussian line prior as a unit normal on rescaled amplitudes.

    Each line amplitude is written as :math:`a_j = w_j\, a''_j` with
    :math:`a''_j \sim \mathcal{N}(\mu_j / w_j, 1)`. The line model :math:`G a` becomes
    :math:`G\,\mathrm{diag}(w)\, a''`, with no separate :math:`G\mu` term, so the design
    is the column-scaled matrix and the prior mean is carried by ``prior_mean``. By the
    determinant lemma, the marginal over :math:`a''` under this unit-normal prior equals
    the marginal over :math:`a` under :math:`\mathcal{N}(\mu, \mathrm{diag}(w^2))`
    exactly, and the normalization offset of that marginal is
    :math:`-\sum_j \ln w_j`. The prior precision :math:`1/w_j^2` is never formed,
    which matters in float32 at F_nu scale, where :math:`w_j \sim 10^{-21}` makes
    :math:`1/w_j^2` overflow.

    Parameters
    ----------
    design : array_like, shape (n_pix, n_lines)
        Design matrix :math:`G` [data units per line-amplitude unit].
    prior_width : scalar or array_like, shape (n_lines,)
        Prior standard deviation :math:`w_j` of each amplitude [data units]. A scalar
        applies to every line (the flat-prior case).
    noise : array_like, shape (n_pix,)
        Per-pixel noise standard deviation :math:`\sigma_i` [data units]. Sets the
        scale.
    pix_mask : array_like, shape (n_pix,), optional
        Pixel mask :math:`w_i \in \{0, 1\}`. Masked pixels are left out of the scale.
        Default: all ones.
    prior_mean : array_like, shape (n_lines,), optional
        Prior mean :math:`\mu_j` of each amplitude [data units]. Default: zeros.

    Returns
    -------
    UnitLinePrior
        ``(design, prior_mean, prior_precision, scale)`` for
        :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`. The
        design is :math:`G\,\mathrm{diag}(w)`, and the prior is on the width-scaled
        coefficients :math:`a''_j = a_j / w_j`: the mean is :math:`\mu_j / w_j` and the
        precision is one, so :math:`E[a''_j] = \mu_j / w_j` and
        :math:`\mathrm{Var}[a''_j] = 1`. The primitive's ``coef_hat`` is :math:`a''`,
        and the amplitudes are :math:`a = w\, a''`.

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp`` operations.

    **Gradient-safe**: yes, with respect to ``design``, ``prior_width``, ``noise`` and
    ``prior_mean``. The scale is the geometric mean of ``noise`` over the unmasked
    pixels divided by the largest width, so it is order unity for any prior width.

    **vmap**: yes over any leading batch axis.

    The scale :math:`s` is an exact change of variables: the marginal value does not
    depend on it, as :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`
    documents. The per-galaxy wrapper and the batched path both call this helper, so a
    line prior is written in one form on both.

    Examples
    --------
    ::

        line = unit_line_prior(G, w, noise=sigma, prior_mean=mu)
        result = marginalize_linear_nuisance(
            data,
            model0,
            line.design,
            sigma=sigma,
            prior_mean=line.prior_mean,
            prior_precision=line.prior_precision,
            scale=line.scale,
        )
        a_hat = width * result.coef_hat  # amplitudes in data units

    """
    n_lines = design.shape[1]
    width = jnp.broadcast_to(jnp.atleast_1d(prior_width), (n_lines,))
    mean = jnp.zeros(n_lines, dtype=design.dtype) if prior_mean is None else prior_mean
    weight = jnp.ones_like(noise) if pix_mask is None else pix_mask
    log_noise_mean = jnp.sum(weight * jnp.log(noise)) / jnp.sum(weight)
    return UnitLinePrior(
        design=design * width,
        prior_mean=mean / width,
        prior_precision=jnp.eye(n_lines, dtype=design.dtype),
        scale=jnp.exp(log_noise_mean) / jnp.max(width),
    )


class ModelLinePrior(NamedTuple):
    """Gaussian line prior centred on the model's own line prediction.

    Attributes
    ----------
    prior_mean : ndarray, shape (n_lines,)
        Model line amplitude :math:`\\mu_l` [data units]; zero for flat lines.
    prior_width : ndarray, shape (n_lines,)
        Prior standard deviation :math:`w\\,\\mu_l` [data units] of a Gaussian line; one
        for a flat line, whose unit coefficient carries no prior.
    gaussian : ndarray, shape (n_lines,)
        One where the line has a Gaussian prior, zero where it is flat.
    unit_mean : ndarray, shape (n_lines,), optional
        Prior mean in units of the prior width, :math:`\\mu_l/w_l = 1/(10^{\\Delta} - 1)`,
        formed without dividing two float32-scale amplitudes. When absent the ratio
        ``prior_mean / prior_width`` is used.
    """

    prior_mean: jnp.ndarray
    prior_width: jnp.ndarray
    gaussian: jnp.ndarray
    unit_mean: jnp.ndarray | None = None


def match_model_lines(
    model_wavelengths: np.ndarray,
    catalog_wavelengths: np.ndarray,
    tolerance_aa: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Index the model line matching each catalog line by vacuum wavelength.

    Parameters
    ----------
    model_wavelengths : array_like, shape (n_model,)
        Rest-frame vacuum wavelengths of the model's discrete line catalog [Angstrom].
    catalog_wavelengths : array_like, shape (n_lines,)
        Rest-frame vacuum wavelengths of the emission-line catalog [Angstrom].
    tolerance_aa : float, optional
        Largest wavelength separation that counts as a match [Angstrom]. Default 1.0.

    Returns
    -------
    match_index : ndarray, shape (n_lines,), int
        Index of the nearest model line for each catalog line (zero where unmatched).
    matched : ndarray, shape (n_lines,), bool
        True where a model line lies within ``tolerance_aa`` of the catalog line.

    Notes
    -----
    Runs in NumPy at build time. The result is a static map, so the Gaussian/flat split
    of the line prior is fixed by the catalog and is not a traced quantity.
    """
    model = np.asarray(model_wavelengths, dtype=np.float64).reshape(-1)
    catalog = np.asarray(catalog_wavelengths, dtype=np.float64).reshape(-1)
    if model.size == 0:
        return np.zeros(catalog.size, dtype=int), np.zeros(catalog.size, dtype=bool)
    separation = np.abs(catalog[:, None] - model[None, :])
    nearest = np.argmin(separation, axis=1)
    rows = np.arange(catalog.size)
    return nearest.astype(int), separation[rows, nearest] <= tolerance_aa


def model_line_match(model, catalog_wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index the lines a model's nebular backend publishes against a catalog.

    Parameters
    ----------
    model : SEDModel
        Model whose nebular backend publishes a discrete line catalog (Cue).
    catalog_wavelengths : array_like, shape (n_lines,)
        Rest-frame vacuum wavelengths of the independent catalog lines [Angstrom].

    Returns
    -------
    match_index : ndarray, shape (n_lines,), int
        Index of the nearest published model line for each catalog line.
    matched : ndarray, shape (n_lines,), bool
        True where a published line lies within 1 Angstrom of the catalog line.

    Raises
    ------
    ValueError
        If the model has no nebular backend that publishes a line catalog.

    Notes
    -----
    Build-time NumPy function, not JIT-compatible. The per-galaxy likelihood and the
    batched likelihood both call it, so the Gaussian/flat split is one rule.
    """
    backend = getattr(model, "_nebular_backend", None)
    if backend is None or not hasattr(backend, "published_line_wavelengths"):
        raise ValueError(
            'eline_prior_type="model" needs a nebular backend that publishes a line '
            "catalog (cue); the model's line wavelengths are read from the backend."
        )
    return match_model_lines(backend.published_line_wavelengths(), catalog_wavelengths)


def line_foreground_transmission(
    catalog_wavelengths, redshift: float, ebmv_mw, rv, law: str
) -> jnp.ndarray:
    r"""Observed-frame Milky Way transmission at each catalog line.

    Parameters
    ----------
    catalog_wavelengths : array_like, shape (n_lines,)
        Rest-frame vacuum wavelengths of the catalog lines [Angstrom].
    redshift : float
        Source redshift [dimensionless].
    ebmv_mw : float or array_like
        Foreground reddening E(B-V) [mag].
    rv : float or array_like
        Total-to-selective extinction ratio R_V [dimensionless].
    law : str
        Foreground law, a key of
        :data:`~tengri.components.dust.foreground.FOREGROUND_LAWS`.

    Returns
    -------
    ndarray, shape (n_lines,)
        Transmission :math:`T_l` of the screen at :math:`\lambda_l (1+z)` [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes; ``law`` is static. **Gradient-safe**: yes, with respect to
    ``ebmv_mw`` and ``rv``. **Vmap-safe**: yes over ``ebmv_mw``.
    """
    from tengri.components.dust.foreground import foreground_transmission

    lam_obs = jnp.asarray(catalog_wavelengths) * (1.0 + redshift)
    return foreground_transmission(lam_obs, ebmv_mw, rv, law)


def model_line_prior(
    log_line_lums: jnp.ndarray,
    match_index: np.ndarray,
    matched: np.ndarray,
    catalog_wavelengths: np.ndarray,
    redshift: float,
    dl_cm: float,
    aperture_frac: float,
    width_dex: float,
    transmission: jnp.ndarray | None = None,
) -> ModelLinePrior:
    r"""Line amplitude prior centred on the model's attenuated line luminosity.

    The model's line luminosity :math:`L_l` [erg/s] is projected onto the spectrum's
    flux-density scale with the factor that the spectral projection applies to a line.
    The integrated observed flux of a line is :math:`L_l/(4\pi d_L^2)` [erg/s/cm^2], and
    :math:`\int F_\nu\,d\lambda = F_l\,\lambda_{\rm obs}^2/c` converts it to the
    amplitude of a unit-integral profile in data units [F_nu Angstrom]. The prior mean
    and width are then

    .. math::

        \mu_l = f_\mathrm{ap}\,T_l\,\frac{L_l\,\lambda_{l,\rm obs}^2}{4\pi d_L^2\, c},
        \qquad
        w_l = (10^{\,\Delta} - 1)\,\mu_l,

    with :math:`f_\mathrm{ap}` the line aperture fraction, :math:`T_l` the observed-frame
    transmission of the Milky Way foreground screen at the line [dimensionless],
    :math:`\lambda_{l,\rm obs} = \lambda_l (1+z)` the observed line wavelength
    [Angstrom], :math:`c` the speed of light [Angstrom/s] and :math:`\Delta` the prior
    width [dex]. :math:`T_l = 1` when no transmission is given. A line the model does
    not predict has :math:`\mu_l = 0` and is flat.

    Parameters
    ----------
    log_line_lums : array_like, shape (n_model,)
        Attenuated model line luminosities :math:`\log_{10}(L_l / \mathrm{erg\,s^{-1}})`
        [dex], as published by the dust component
        (``state.derived["log_line_lums_attenuated"]``).
    match_index : ndarray, shape (n_lines,), int
        Model-line index of each catalog line, from :func:`match_model_lines`.
    matched : ndarray, shape (n_lines,), bool
        Whether each catalog line has a model counterpart, from :func:`match_model_lines`.
    catalog_wavelengths : array_like, shape (n_lines,)
        Rest-frame vacuum wavelengths of the catalog lines [Angstrom].
    redshift : float
        Source redshift [dimensionless].
    dl_cm : float
        Luminosity distance [cm].
    aperture_frac : float or scalar array
        Line aperture fraction :math:`f_\mathrm{ap}` [dimensionless]; the share of the
        line light that the fiber collects.
    width_dex : float
        Prior width :math:`\Delta` [dex].
    transmission : array_like, shape (n_lines,), optional
        Observed-frame transmission :math:`T_l` of each catalog line [dimensionless], at
        :math:`\lambda_l (1+z)`. The per-galaxy path passes the model's Milky Way screen
        (:func:`line_foreground_transmission`); the batched path passes the galaxy's own
        ``ebmv_mw``. Default: no screen, :math:`T_l = 1`.

    Returns
    -------
    ModelLinePrior
        Prior mean :math:`\mu_l` [data units], width :math:`w_l` [data units] and the
        Gaussian indicator of each line.

    Notes
    -----
    **JIT-compatible**: yes; ``match_index`` and ``matched`` are static, and the
    logarithmic form keeps :math:`\lambda^2/(4\pi d_L^2 c)` representable in float32.

    **Gradient-safe**: yes, with respect to ``log_line_lums``, ``aperture_frac``,
    ``redshift`` and ``dl_cm``. Unmatched entries are selected, not floored: the
    logarithm of an unmatched line is replaced by zero before the exponential.

    **vmap**: yes over any leading batch axis of ``log_line_lums``.

    The transmission is the Milky Way foreground screen only. The intergalactic
    transmission is not included: every catalog line lies at rest wavelength above
    1216 Angstrom, where the shipped IGM models transmit exactly one, so the foreground
    is the only observed-frame transmission a line sees.
    """
    catalog = jnp.asarray(catalog_wavelengths)
    lam_obs = catalog * (1.0 + redshift)
    log_projection = jnp.log10(lam_obs**2 / C_AA) - log10_four_pi_dl2(dl_cm)
    gaussian = jnp.asarray(matched, dtype=log_line_lums.dtype)
    safe_log = jnp.where(gaussian > 0, jnp.asarray(log_line_lums)[match_index], 0.0)
    mu = jnp.where(
        gaussian > 0,
        aperture_frac * 10.0 ** (safe_log + log_projection),
        0.0,
    )
    if transmission is not None:
        mu = mu * jnp.asarray(transmission)
    width = jnp.where(gaussian > 0, (10.0**width_dex - 1.0) * mu, 1.0)
    unit_mean = jnp.where(gaussian * (mu > 0) > 0, 1.0 / (10.0**width_dex - 1.0), 0.0)
    return ModelLinePrior(prior_mean=mu, prior_width=width, gaussian=gaussian, unit_mean=unit_mean)


def unit_model_line_prior(
    design: jnp.ndarray,
    prior: ModelLinePrior,
    *,
    noise: jnp.ndarray,
    pix_mask: jnp.ndarray | None = None,
) -> UnitLinePrior:
    r"""Rewrite a model-centred line prior as a unit-normal form for the linear primitive.

    Each amplitude is :math:`a_l = \mu_l + w_l\,a'_l`. A Gaussian line has
    :math:`a'_l \sim \mathcal{N}(0,1)`, and a flat line has no prior on :math:`a'_l`,
    which is marked by a zero entry of ``prior_precision``. A line with
    :math:`\mu_l = 0` has a zero design column and a unit prior, which contributes
    exactly nothing to the marginal, so it equals deleting the column.

    Parameters
    ----------
    design : array_like, shape (n_pix, n_lines)
        Design matrix :math:`G` [data units per line-amplitude unit].
    prior : ModelLinePrior
        Output of :func:`model_line_prior`.
    noise : array_like, shape (n_pix,)
        Per-pixel noise standard deviation :math:`\sigma_i` [data units].
    pix_mask : array_like, shape (n_pix,), optional
        Pixel mask. Default: all ones.

    Returns
    -------
    UnitLinePrior
        ``(design, prior_mean, prior_precision, scale)`` for
        :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`, where
        ``prior_precision`` is diagonal, one for Gaussian lines and zero for flat lines.

    Notes
    -----
    **JIT-compatible**: yes. Denominators are selected with ``jnp.where`` so a zero
    width gives a zero mean with a finite gradient.

    **Gradient-safe**: yes, with respect to ``design``, ``noise`` and the prior fields.
    The coefficient scale is an exact change of variables (the marginal does not depend
    on it), so it is held out of the gradient; its derivative would divide by the squared
    line width, which overflows in float32 at F_nu scale.

    **vmap**: yes over any leading batch axis.
    """
    width = prior.prior_width
    has_width = width > 0
    safe_width = jnp.where(has_width, width, 1.0)
    weight = jnp.ones_like(noise) if pix_mask is None else pix_mask
    log_noise_mean = jnp.sum(weight * jnp.log(noise)) / jnp.sum(weight)
    max_width = jnp.max(width)
    scale_width = jnp.where(max_width > 0, max_width, 1.0)
    return UnitLinePrior(
        design=design * width,
        prior_mean=jnp.where(
            has_width,
            prior.prior_mean / safe_width if prior.unit_mean is None else prior.unit_mean,
            0.0,
        ),
        prior_precision=jnp.diag(prior.gaussian),
        scale=jax.lax.stop_gradient(jnp.exp(log_noise_mean) / scale_width),
    )


def marginalize_emission_lines(
    residual: jnp.ndarray,
    noise: jnp.ndarray,
    design_matrix: jnp.ndarray,
    prior_variance: jnp.ndarray | None = None,
    *,
    pix_mask: jnp.ndarray | None = None,
    col_mask: jnp.ndarray | None = None,
    scale: jnp.ndarray | None = None,
    prior_sigma: jnp.ndarray | None = None,
) -> tuple:
    r"""Analytically marginalize emission-line amplitudes under a Gaussian prior.

    Integrates the linear line amplitudes out of the residual model through
    :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`, with a
    zero model offset, so the residual is the data.

    Parameters
    ----------
    residual : array_like, shape (n_pix,)
        Data minus continuum model, ``d - m`` [data units].
    noise : array_like, shape (n_pix,)
        Per-pixel noise standard deviation :math:`\sigma` [data units]. Must be
        strictly positive wherever ``pix_mask`` is nonzero.
    design_matrix : array_like, shape (n_pix, n_lines)
        Gaussian design matrix from :func:`build_eline_design_matrix`
        [data units per line-amplitude unit].
    prior_variance : scalar or array_like, shape (n_lines,), optional
        Prior variance :math:`\Lambda` of each line amplitude [data units squared].
        Large values give an uninformative prior. Default: ``1e10`` per line.
    pix_mask : array_like, shape (n_pix,), optional
        Pixel mask :math:`w \in \{0, 1\}`. Masked pixels contribute nothing to any
        term. Default: all ones.
    col_mask : array_like, shape (n_lines,), optional
        Line mask :math:`c \in \{0, 1\}`. A masked line is fixed out of the model.
        Default: all ones.
    prior_sigma : scalar or array_like, shape (n_lines,), optional
        Prior standard deviation :math:`\sqrt{\Lambda}` of each line amplitude
        [data units]. Pass this instead of ``prior_variance`` when the variance
        would underflow the working precision (below about 1e-19 in float32).
        Exactly one of the two may be given.
    scale : scalar, optional
        Coefficient scale :math:`s` that keeps the whitened design at order unity.
        The value does not depend on :math:`s` in exact arithmetic. Default: the
        geometric mean of ``noise`` over the unmasked pixels,
        :math:`\exp(\sum_i w_i \ln\sigma_i / N_\mathrm{eff})`, so float32 runs at
        flux scale stay finite.

    Returns
    -------
    ln_L_marg : ndarray, shape ()
        Marginal log-likelihood (dimensionless).
    a_hat : ndarray, shape (n_lines,)
        Posterior-mean line amplitudes [data units].
    a_cov : ndarray, shape (n_lines, n_lines)
        Posterior covariance of the line amplitudes [data units squared].

    Notes
    -----
    **JIT-compatible**: yes; ``prior_variance`` may be traced.

    **Gradient-safe**: yes, with respect to ``residual``, ``noise``,
    ``design_matrix`` and ``prior_variance``, in float32 as well as float64.

    The value is the exact marginal likelihood of the residual under
    :math:`r = G a + \epsilon` with :math:`\epsilon \sim \mathcal{N}(0, \mathrm{diag}\,\sigma^2)`
    and :math:`a \sim \mathcal{N}(0, \Lambda)`,

    .. math::

        \ln L = -\tfrac12 \ln\det\!\big(\mathrm{diag}\,\sigma^2 + G\Lambda G^\mathsf{T}\big)
        - \tfrac12\, r^\mathsf{T}\big(\mathrm{diag}\,\sigma^2 + G\Lambda G^\mathsf{T}\big)^{-1} r
        + \sum_i \ln\sigma_i,

    with the constant :math:`-\tfrac12 N_\mathrm{eff}\ln 2\pi` and the
    :math:`-\sum_i \ln\sigma_i` of the normalized Gaussian left out. Equivalently
    :math:`\ln L = -\tfrac12\,\tilde r^\mathsf{T}\tilde r + \tfrac12\,\hat a^\mathsf{T}
    (\tilde G^\mathsf{T}\tilde G)\,\hat a + \tfrac12\,\hat a^\mathsf{T}\Lambda^{-1}\hat a
    + \tfrac12 \ln\det(\Lambda^{-1})
    - \tfrac12\ln\det(\tilde G^\mathsf{T}\tilde G + \Lambda^{-1})`,
    with :math:`\tilde G = G/\sigma`, :math:`\tilde r = r/\sigma` and
    :math:`\hat a = (\tilde G^\mathsf{T}\tilde G + \Lambda^{-1})^{-1}\tilde G^\mathsf{T}\tilde r`.
    The prior penalty :math:`\hat a^\mathsf{T}\Lambda^{-1}\hat a` enters this value
    once. A form that subtracts it a second time is lower by exactly
    :math:`\hat a^\mathsf{T}\Lambda^{-1}\hat a`.

    The masks and the scale are handled by
    :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`; a masked
    line or pixel is selected out rather than floored.

    """
    n_lines = design_matrix.shape[1]
    if prior_sigma is not None and prior_variance is not None:
        raise ValueError("pass prior_variance or prior_sigma, not both")
    if prior_sigma is None:
        _pv = jnp.full((n_lines,), 1e10) if prior_variance is None else prior_variance
        width = jnp.sqrt(jnp.broadcast_to(jnp.atleast_1d(_pv), (n_lines,)))
    else:
        width = jnp.broadcast_to(jnp.atleast_1d(prior_sigma), (n_lines,))
    line = unit_line_prior(design_matrix, width, noise=noise, pix_mask=pix_mask)
    weight = jnp.ones_like(noise) if pix_mask is None else pix_mask
    n_eff = jnp.sum(weight)
    log_noise_sum = jnp.sum(weight * jnp.log(noise))
    coef_scale = line.scale if scale is None else scale

    result = marginalize_linear_nuisance(
        residual,
        jnp.zeros_like(residual),
        line.design,
        sigma=noise,
        pix_mask=pix_mask,
        prior_mean=line.prior_mean,
        prior_precision=line.prior_precision,
        col_mask=col_mask,
        scale=coef_scale,
    )
    ln_l_marg = result.log_like + 0.5 * n_eff * math.log(2.0 * math.pi) + log_noise_sum
    a_hat = width * result.coef_hat
    a_cov = (width[:, None] * width[None, :]) * result.coef_cov
    return ln_l_marg, a_hat, a_cov


# JIT-compiled version (prior_variance must be traced, not static)
marginalize_emission_lines = jax.jit(marginalize_emission_lines)


@jax.jit
def predict_with_marginalized_lines(
    model_continuum: jnp.ndarray,
    design_matrix: jnp.ndarray,
    a_hat: jnp.ndarray,
) -> jnp.ndarray:
    """Return the full model including optimized emission lines.

    Parameters
    ----------
    model_continuum : array, shape (n_pix,)
        Continuum-only model spectrum.
    design_matrix : array, shape (n_pix, n_lines)
        Gaussian design matrix.
    a_hat : array, shape (n_lines,)
        Optimal line amplitudes from :func:`marginalize_emission_lines`.

    Returns
    -------
    ndarray, shape (n_pix,)
        Full model spectrum including continuum and optimized emission lines
        [erg/s/Hz] or [erg/s/Angstrom] depending on wavelength unit.

    Notes
    -----
    **JIT-compatible**: yes, matrix multiplication via ``jnp.dot``.

    **Gradient-safe**: yes, fully differentiable.

    """
    return model_continuum + design_matrix @ a_hat


def build_line_design_matrix(
    wave_obs: jnp.ndarray,
    narrow_wavelengths: jnp.ndarray,
    broad_wavelengths: jnp.ndarray | None = None,
    spectral_resolution: float = 2000.0,
    redshift: float = 0.0,
    narrow_sigma_kms: float = 0.0,
    broad_sigma_kms: float = 5000.0,
    delta_v_kms: float = 0.0,
) -> jnp.ndarray:
    """Unified emission line design matrix for narrow and/or broad lines.

    Returns a ``(n_pix, n_narrow + n_broad)`` design matrix suitable for
    analytical marginalization via ``marginalize_emission_lines()``.

    Narrow line columns come first, broad line columns follow. If no
    ``broad_wavelengths`` are provided, returns only the narrow columns.

    Parameters
    ----------
    wave_obs : array, shape (n_pix,)
        Observed wavelength grid [Angstrom].
    narrow_wavelengths : array, shape (n_narrow,)
        Rest-frame narrow line wavelengths [Angstrom].
    broad_wavelengths : array or None, shape (n_broad,)
        Rest-frame broad line wavelengths [Angstrom].
        If ``None``, only narrow columns are returned.
    spectral_resolution : float
        Spectral resolution R = lambda/delta_lambda. Default 2000.
    redshift : float
        Redshift for shifting lines to observed frame. Default 0.
    narrow_sigma_kms : float
        Intrinsic narrow line width [km/s]. Default 0 (instrument-limited).
    broad_sigma_kms : float
        Intrinsic broad line width [km/s]. Default 5000.
    delta_v_kms : float
        Systematic velocity offset [km/s]. Default 0.

    Returns
    -------
    ndarray, shape (n_pix, n_narrow [+ n_broad])
        Design matrix. Each column is a normalized Gaussian profile
        [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, delegates to :func:`build_eline_design_matrix` and
    :func:`build_broad_design_matrix`, which are JIT-decorated.

    If ``broad_wavelengths`` is ``None``, returns only the narrow-line portion.

    Examples
    --------
    >>> A = build_line_design_matrix(wave_obs, DEFAULT_LINE_WAVELENGTHS, redshift=0.1)

    """
    A_narrow = build_eline_design_matrix(
        wave_obs,
        narrow_wavelengths,
        spectral_resolution=spectral_resolution,
        redshift=redshift,
        eline_sigma_kms=narrow_sigma_kms,
        eline_delta_v_kms=delta_v_kms,
    )

    if broad_wavelengths is None:
        return A_narrow

    A_broad = build_broad_design_matrix(
        wave_obs,
        broad_wavelengths,
        spectral_resolution=spectral_resolution,
        redshift=redshift,
        broad_sigma_kms=broad_sigma_kms,
        eline_delta_v_kms=delta_v_kms,
    )

    return jnp.concatenate([A_narrow, A_broad], axis=1)


def eline_scalar_resolution(resolution) -> float:
    """Return the scalar resolution R that sets the emission-line profile width.

    Parameters
    ----------
    resolution : float, array_like or None
        Spectral resolution R = lambda / delta_lambda [dimensionless]: a scalar,
        a per-pixel array, or ``None`` when the observation declares none.

    Returns
    -------
    float
        The scalar itself; the median of the array; or 2000.0 for ``None``.

    Notes
    -----
    Pure NumPy, evaluated at Python level. An array reduces to its median because
    the Gaussian design matrix takes one number and R varies slowly across a band.
    """
    if resolution is None:
        return 2000.0
    if getattr(resolution, "ndim", 0):
        return float(np.median(np.asarray(resolution)))
    return float(resolution)


def eline_amplitude_names(catalog) -> list[str]:
    """Return the ``eline_amp_<name>`` parameter names of the independent lines.

    Parameters
    ----------
    catalog : LineList
        Line catalog whose doublet secondaries are not sampled independently.

    Returns
    -------
    list of str
        One name per line that is not a doublet secondary, in catalog order.

    Notes
    -----
    Pure Python. The names are the keys of the amplitude priors that the fit
    adds to the spectroscopic specification.
    """
    secondary = {dc.secondary_idx for dc in catalog.doublets}
    return [f"eline_amp_{nm}" for i, nm in enumerate(catalog.names) if i not in secondary]

# SPDX-License-Identifier: BSD-3-Clause
"""StellarSEDComponent: composite stellar population assembly from SFH and SSP.

Merges the SFH and SSP sub-modules into a single ``SEDComponent``.

Model selection (see :class:`StellarSEDComponentConfig`):

- ``sfh_model`` — any name in ``SFH_REGISTRY`` (``tsnorm``, ``dpl``,
  ``dense_basis``, …); validated at construction.
- ``metallicity_model`` — any name in ``MET_REGISTRY`` (``delta``,
  ``bins``, ``table``, …); validated at construction.
- ``field=True`` adds the stochastic GP-field parameters on top of the
  chosen mean SFH.
- ``sps_backend="dsps"`` (DSPS native CSP integration) is the only
  backend.

The component publishes derived quantities (stellar mass, SFR history, etc.)
in ``state.derived`` that downstream components (dust, nebular, radio, X-ray)
read to compute their own emission.

Architectural note: the SSP grid is held on the component instance
(constructor field) and treated as a fixed input baked in at construction time,
not an output of a separate precompute step.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from tengri.parameters.resolve import require_redshift


class SFHBeforeBigBangWarning(UserWarning):
    """Part of the SFH forms stars before the Big Bang at the given redshift.

    Emitted from the eager forward path when the star formation history places
    a non-negligible fraction of its formed stellar mass at lookback times
    older than the age of the universe at ``redshift``. That mass is truncated
    (clamped to cosmic time) so the prediction does not reflect the requested
    SFH. Bound the SFH age parameter or the redshift to silence it. The check
    is skipped under ``jax.jit`` / inference, where exploring such draws is
    expected. See suchethac/tengri#683.
    """


class SFHBeyondSSPGridWarning(UserWarning):
    """A tabulated SFH forms stars older than the oldest SSP template age.

    The counterpart of :class:`SFHBeforeBigBangWarning` at the *other* end of
    the same lookback axis. No template represents stars older than the grid's
    oldest age, so their mass is assigned to that oldest template: mass is
    conserved, but those stars are rendered with the colors of a population as
    old as the grid allows rather than their true age.

    Only a **tabulated** history can trigger this. Parametric and non-parametric
    families renormalize their age weights to ``log_total_mass`` after landing
    them on the grid, so whatever falls off the end is scaled back in; a table
    carries absolute Msun/yr and has no such step — which is why the mass simply
    vanished before this warning existed. The check is skipped under
    ``jax.jit`` / inference, like its sibling. See suchethac/tengri#1522.
    """


from tengri.components.stellar.sfh.gp_sfh import log_age_grid_step, make_log_age_grid
from tengri.components.stellar.sfh.metallicity_history import (
    massmap_box_metallicity,
    massmap_lin_metallicity,
    metallicity_bins_continuity_on_ssp_grid,
    metallicity_bins_on_ssp_grid,
    psb_two_step_metallicity,
    tabulated_metallicity_on_ssp_grid,
    two_step_metallicity,
)
from tengri.components.stellar.sps.dsps_wrapper import (
    LSUN_ERG_PER_S,
    SSPData,
    canonical_dsps_kwargs,
    compute_log_z_evolving,
    compute_surviving_mass,
    effective_metallicity,
    enforce_increasing_cosmic_time,
    has_alpha_grid,
    interpolate_alpha_only,
    interpolate_mass_remaining,
)
from tengri.parameters.translate import LOG10_ZSUN
from tengri.utils.scale import _not_computable, log10_magnitude, pow10

# Default time bins for ``metallicity_model="bins"`` /
# ``"bins_continuity"`` — log-spaced from 1 Myr to 13.7 Gyr,
# 7 edges → 6 bins, matching ``MET_REGISTRY``'s
# ``_N_MET_BINS_DEFAULT``.
_DEFAULT_MET_BIN_EDGES_LOG_YR = jnp.array([6.0, 7.5, 8.5, 9.0, 9.5, 9.9, 10.14])

#: Accepted ``age_kernel`` values — how the SFH is integrated onto the SSP age
#: grid. See :class:`StellarSEDComponentConfig` for the accuracy/cost tradeoff.
VALID_AGE_KERNELS = ("cic", "dsps")

#: Kernel chosen on the non-field path when ``age_kernel`` is left unset
#: (``None`` = auto). ``"cic"`` is the accuracy default: the DSPS histogram
#: kernel zeroes the first SSP node older than the SFH start and biases the
#: optical CSP +1.2 % vs FSPS / bagpipes / a dense reference (#964). Flipping
#: this one name changes the default for every non-field model.
DEFAULT_AGE_KERNEL = "cic"


def _resolve_age_kernel(config) -> str:
    """Which age-weight kernel this config selects — ``"cic"`` or ``"dsps"``.

    Centralizes the choice so :meth:`StellarSEDComponent.apply` and the SED-free
    :meth:`StellarSEDComponent.compute_joint_weights` fast path cannot drift
    apart — a divergence there is invisible until the two disagree on a fit
    (#982). Validates the value here rather than at each branch so a typo fails
    loudly at the first prediction instead of silently selecting the default.

    Parameters
    ----------
    config : StellarSEDComponentConfig
        The component config; reads ``age_kernel`` and ``field``.

    Returns
    -------
    str
        ``"cic"`` or ``"dsps"``.

    Raises
    ------
    ValueError
        ``age_kernel`` is not in :data:`VALID_AGE_KERNELS`.
    NotImplementedError
        ``age_kernel="cic"`` was requested with ``field=True``. The GP-field
        draw is defined on its own coarse lookback grid, so there is no dense
        integrand to cloud-in-cell (#964). Returning DSPS weights anyway would
        make an explicit request a silent no-op.

    Notes
    -----
    **JIT/grad/vmap-safe**: returns a static Python string from static config
    fields; never touches traced values.
    """
    kernel = getattr(config, "age_kernel", None)
    if kernel is not None and kernel not in VALID_AGE_KERNELS:
        raise ValueError(
            f"Unknown age_kernel {kernel!r}. Valid: {', '.join(VALID_AGE_KERNELS)} "
            f"(or None to auto-select). 'cic' is the accuracy default (dense "
            f"cloud-in-cell integrand); 'dsps' is DSPS's histogram kernel, "
            f"offered for cross-code comparison and known to bias the optical "
            f"CSP +1.2 % (#964)."
        )
    if config.field:
        # The field draw lives on the coarse lookback grid by construction, so
        # DSPS is the only implemented kernel here. Auto-select resolves to it
        # silently (that is today's behavior); an EXPLICIT 'cic' must not.
        if kernel == "cic":
            raise NotImplementedError(
                "age_kernel='cic' is not supported with a GP-field SFH — the "
                "field draw is defined on its own coarse lookback grid, so "
                "there is no dense integrand to cloud-in-cell (#964). Drop the "
                "field modulator to use the CIC kernel, or set "
                "age_kernel='dsps' explicitly to acknowledge the field path's "
                "kernel."
            )
        return "dsps"
    return DEFAULT_AGE_KERNEL if kernel is None else kernel


def _sfh_bin_edges_yr(fn, sfh_kwargs):
    """Lazy proxy to :func:`...sfh.nonparametric.sfh_bin_edges_yr` (#765)."""
    from tengri.components.stellar.sfh.nonparametric import sfh_bin_edges_yr

    return sfh_bin_edges_yr(fn, sfh_kwargs)


def _fast_path_unsupported_sfh_fns():
    """Map each unsupported SFH function to *why* the fast path refuses it (#950, #1395).

    The SED-free :meth:`StellarSEDComponent.compute_joint_weights` fast path
    integrates a closed-form SFR on a dense age grid. Two disjoint families
    cannot be served that way, and they are kept apart here because the reason
    a caller sees decides what they do next:

    * the non-parametric families (Leja+2019 continuity, Dirichlet, PSB) — the
      fast path has no bin basis for them, and never did (#950);
    * the tabulated SFH — its registry ``fn`` is an all-zero *placeholder*,
      because the real history is wired in at
      :meth:`StellarSEDComponent.apply`, which the fast path never reaches.
      Evaluating the placeholder returns a zero SFH, zero mass and zero lines,
      which the weight normalization's ``1e-300`` clamp then launders into a
      finite zero — silent, and beside correct photometry (#1395).

    Keyed on the function object rather than the model name so that a second
    registry entry sharing one of these implementations is guarded too.
    Imported lazily to avoid import-order coupling at module load.

    Returns
    -------
    dict
        ``{sfh_fn: reason_str}`` — the reason is interpolated into the
        ``ValueError`` raised by :meth:`StellarSEDComponent.compute_joint_weights`.
    """
    from tengri.components.stellar.sfh.nonparametric import (
        continuity,
        continuity_flex,
        dirichlet,
        psb_continuity,
    )
    from tengri.components.stellar.sfh.registry import _table_sfh_placeholder

    binned = "the fast path integrates a closed-form SFR and has no bin basis for it"
    return {
        continuity: binned,
        continuity_flex: binned,
        dirichlet: binned,
        psb_continuity: binned,
        _table_sfh_placeholder: (
            "the tabulated history is supplied at apply(), which the fast path never "
            "reaches, so the fast path would evaluate an all-zero placeholder and "
            "return zero weights, zero mass and zero lines without raising (#1395)"
        ),
    }


_FAST_PATH_UNSUPPORTED_SFH_FNS = _fast_path_unsupported_sfh_fns()


def _apply_gp_field(sfr_history, params, n_grid, log_age_grid, centering: float = 1.0):
    """Apply the multiplicative GP-field modulation to a smooth SFR history.

    :math:`\\mathrm{SFR}(t) = \\mathrm{SFR}_{\\rm mean}(t)\\,\\exp(x(t) - K_0/2)`,
    where :math:`x(t)` is the PSD-governed Gaussian process and :math:`K_0/2` is
    the log-normal bias correction (so the ensemble mean is preserved). The single
    source of the field modulation — shared by :meth:`StellarSEDComponent.apply`
    (exact SED) and :meth:`StellarSEDComponent.compute_joint_weights` (fast
    line/nebular window LUT) so the two paths cannot diverge.

    Parameters
    ----------
    sfr_history : ndarray, shape (n_grid,)
        Smooth (mean) SFR on the lookback grid [Msun/yr].
    params : Mapping
        Free-parameter dict carrying ``sfh_field_psd_sigma`` [dex],
        ``sfh_field_psd_tau_myr`` [Myr], and the latent ``sfh_field_xi``.
    n_grid : int
        SFH grid resolution (the field latent dimension).
    log_age_grid : ndarray, shape (n_grid,)
        ``log10(age/yr)`` grid the field lives on.
    centering : float, optional
        Parameterization of the field latent, in ``[0, 1]`` [dimensionless].
        ``1.0`` (default) is the non-centered map ``s = L(sigma, tau) xi``;
        ``a < 1`` moves amplitude dependence out of it (#1355). Must be paired
        with the matching latent prior — see
        :func:`~tengri.components.stellar.sfh.gp_sfh.drw_latent_log_prior`.

    Returns
    -------
    ndarray, shape (n_grid,)
        Field-modulated SFR history [Msun/yr].

    Notes
    -----
    **JIT-compatible**: yes — ``log_age_grid_step`` recomputes the step from the
    static ``n_grid`` (a traced ``log_age_grid`` cannot be indexed under jit).
    """
    from tengri.components.stellar.sfh.registry import compute_field_gp

    psd_sigma = jnp.asarray(params["sfh_field_psd_sigma"])
    psd_tau_yr = jnp.asarray(params["sfh_field_psd_tau_myr"]) * 1e6
    # ``sfh_field_xi`` is the ONLY spelling that reaches here: the forward
    # pipeline filters params down to declared names, so a dict carrying the
    # sampler's ``psd_xi`` arrives with no latents at all and this default fires.
    # Producers must therefore publish ``sfh_field_xi`` -- see
    # ``Fitter._to_physical`` and ``_unstandardize_parameters``, both of which
    # emit both spellings for exactly this reason (#1271).
    xi = jnp.asarray(params.get("sfh_field_xi", jnp.zeros(n_grid)))
    gp_x, k0_half = compute_field_gp(
        xi,
        psd_sigma,
        psd_tau_yr,
        n_grid,
        log_age_grid_step(n_grid),
        field_model="drw",
        log_age_grid=log_age_grid,
        centering=centering,
    )
    return sfr_history * jnp.exp(gp_x - k0_half)


def _refine_sfh_table_ages(ssp_ages_yr, factor: int = 16):
    """Dense log-spaced age grid spanning the SSP template ages [yr] (#758).

    Non-parametric SFHs (continuity / dirichlet / post-starburst) are
    piecewise-constant in lookback time. Sampling them only at the ~107 coarse
    SSP template ages linearly smears each bin-edge transition when DSPS
    integrates the SFH table, producing a 2-4.5 % optical residual vs
    Prospector (#758). Evaluating the SFH on this ``factor``x denser grid
    resolves the edges. DSPS still returns age weights on ``ssp_lg_age_gyr``,
    so ``age_weights`` and every downstream consumer are unchanged — only the
    integrand resolution improves.

    Parameters
    ----------
    ssp_ages_yr : ndarray, shape (n_ssp,)
        Ascending SSP template ages [yr].
    factor : int, optional
        Sub-sampling factor between adjacent SSP ages (static; default 16).

    Returns
    -------
    ndarray, shape ((n_ssp - 1) * factor + 1,)
        Dense ascending age grid [yr], log-spaced over the SSP age span.

    Notes
    -----
    **Age-0 anchor templates** (#1016, #1030): a leading age = 0 template
    (bc03 stelib) would give ``log_lo = -inf`` and collapse the grid to
    ``[0, ..., 0, age_max]`` — the CIC mass then vanishes (#1016) or lands
    entirely on the youngest node (#1030), depending on the SFH's support.
    Spanning from the smallest *positive* template age keeps the grid finite
    and strictly ascending at full per-decade resolution (a 1 yr floor would
    halve it, #758); the ``[0, age_1]`` sliver is integrated by
    :func:`_cic_parcels`'s lookback-0 extension, not by this grid.
    """
    n = ssp_ages_yr.shape[0]
    lo_yr = jnp.min(jnp.where(ssp_ages_yr > 0.0, ssp_ages_yr, jnp.inf))
    log_lo = jnp.log10(lo_yr)
    log_hi = jnp.log10(ssp_ages_yr[-1])
    return 10.0 ** jnp.linspace(log_lo, log_hi, (n - 1) * factor + 1)


def _extend_integrand_to_history(fine_age_yr, tab_lbt_yr, ssp_ages_yr, factor: int = 16):
    r"""Extend the dense integrand past the oldest SSP template age (#1522).

    :func:`_refine_sfh_table_ages` spans ``[ssp_ages_yr[0], ssp_ages_yr[-1]]``, so
    a tabulated history reaching further back than the oldest template is never
    sampled there and its mass is simply absent from the quadrature — silently,
    since nothing downstream can tell a history that formed no old stars from one
    whose old stars were never integrated. Measured before this fix: an oldest bin
    of 8 Gyr at z=0.05 lost 27.5 % of the requested mass, and 46 % at z=0 on the
    real PARSEC/MILES grid (oldest bin 12.589 Gyr) for a history anchored at the
    Big Bang.

    Appending the tail lets :func:`_cic_parcels` see those parcels. Every one of
    them has ``log10(age) > lg_nodes[-1]``, so its interpolation fraction clips to
    ``1.0`` and it lands wholly on the **oldest** template — the closest thing the
    grid has to a star that old. Mass is then conserved exactly; what remains is an
    approximation in *color*, which :class:`SFHBeyondSSPGridWarning` reports.

    This is the old-age counterpart of the lookback-0 extension
    :func:`_cic_parcels` already performs at the young end, and of the
    :func:`_youngest_bin_lookback_multiplier` edge correction (#821): the same
    axis, the same "no template out here" problem, now handled at both ends.

    Only tabulated histories need it. Parametric and non-parametric families
    renormalize their age weights to ``log_total_mass`` *after* landing them on
    the grid, so mass falling off the end is scaled back in and their totals were
    never wrong; extending their integrand would move weight onto the oldest
    template and change long-settled numbers for no correctness gain.

    Parameters
    ----------
    fine_age_yr : ndarray, shape (n_fine,)
        Dense ascending lookback-age grid [yr] from :func:`_refine_sfh_table_ages`.
    tab_lbt_yr : ndarray, shape (n_t,)
        The tabulated history's own lookback ages [yr]; may be traced. Already
        capped at cosmic time by :func:`_tabulated_sfh`, so the tail never runs
        past the Big Bang.
    ssp_ages_yr : ndarray, shape (n_age,)
        Ascending SSP template ages [yr].
    factor : int, optional
        Number of tail samples appended (static; default 16).

    Returns
    -------
    age_yr : ndarray, shape (n_fine + factor,)
        The integrand grid with the tail appended, ascending.
    top_yr : ndarray, shape ()
        The tail's upper limit [yr] — the clip bound for any edge knots injected
        afterwards, so the table's own nodes out there stay representable.

    Notes
    -----
    **JIT/grad/vmap-safe**: static shape, pure ``jnp``. When the history stops
    inside the SSP grid the tail collapses onto repeats of ``ssp_ages_yr[-1]``;
    those carry zero trapezoid width, so the result is a no-op rather than a
    special case — which is what keeps the shape static.
    """
    hi = ssp_ages_yr[-1]
    top = jnp.maximum(jnp.max(tab_lbt_yr), hi)
    tail = 10.0 ** jnp.linspace(jnp.log10(hi), jnp.log10(top), factor + 1)[1:]
    return jnp.concatenate([fine_age_yr, tail]), top


def _warn_if_history_exceeds_ssp_grid(age_yr, sfr, ssp_ages_yr, tab_lbt_yr, conserved=True):
    """Warn when a tabulated SFH forms stars older than any SSP template (#1522).

    Eager-path only, mirroring the pre-Big-Bang guard: the ``float()`` casts raise
    :exc:`~jax.errors.ConcretizationTypeError` under any jax transform, where
    exploring such draws is expected and there is no concrete value to report.

    Parameters
    ----------
    age_yr, sfr : ndarray, shape (n,)
        The extended integrand — lookback ages [yr] and SFR [Msun/yr].
    ssp_ages_yr : ndarray, shape (n_age,)
        Ascending SSP template ages [yr].
    tab_lbt_yr : ndarray or None
        The tabulated history's lookback nodes, or None for a non-tabulated SFH
        (which cannot trigger this — see :class:`SFHBeyondSSPGridWarning`).
    conserved : bool, optional
        Whether the caller's kernel accumulates that mass onto the oldest
        template (the CIC path, default) or drops it (``age_kernel="dsps"``,
        whose histogram kernel has no bin out there). Only the wording differs;
        both cases must be announced.
    """
    if tab_lbt_yr is None:
        return
    try:
        hi_yr = float(ssp_ages_yr[-1])
        total = float(jnp.trapezoid(sfr, age_yr))
        beyond = float(jnp.trapezoid(jnp.where(age_yr > hi_yr, sfr, 0.0), age_yr))
    except jax.errors.ConcretizationTypeError:
        return  # tracing — no concrete values to inspect
    if total <= 0.0:
        return
    frac = beyond / total
    if frac <= 0.01:
        return
    fate = (
        f"that mass is assigned to the oldest one: the total is conserved, but "
        f"those stars carry the colors of a {hi_yr / 1e9:.2f} Gyr population "
        f"rather than their true age"
        if conserved
        else (
            "that mass is DROPPED — DSPS's histogram kernel has no bin beyond the "
            "grid, so the galaxy comes back lighter than the history you supplied. "
            "Use age_kernel='cic' (the default), which accumulates it onto the "
            "oldest template instead"
        )
    )
    warnings.warn(
        f"The tabulated star formation history forms {frac:.0%} of its stellar "
        f"mass at lookback ages older than the oldest SSP template "
        f"({hi_yr / 1e9:.2f} Gyr). No template represents stars that old, so "
        f"{fate}. Use an SSP grid that spans the galaxy's age, or start the "
        f"history later, to remove the approximation.",
        SFHBeyondSSPGridWarning,
        stacklevel=2,
    )


def _warn_if_dsps_kernel_truncates_history(ssp_ages_yr, sfh_fn, sfh_kwargs, tab_lbt_yr):
    """``age_kernel="dsps"`` still truncates a tabulated history — say so (#1522).

    The CIC fix extends the *integrand*; DSPS's histogram kernel bins onto
    ``ssp_lg_age_gyr`` itself and has no bin past the oldest template, so mass out
    there is still lost. That kernel is opt-in for cross-code comparison (#964),
    so the behavior stands — but it must not be silent, which was the whole of
    #1522. Cheap: returns immediately for every non-tabulated SFH.
    """
    if tab_lbt_yr is None:
        return
    fine_age_yr, _top = _extend_integrand_to_history(
        _refine_sfh_table_ages(ssp_ages_yr), tab_lbt_yr, ssp_ages_yr
    )
    _warn_if_history_exceeds_ssp_grid(
        fine_age_yr,
        sfh_fn(fine_age_yr, **sfh_kwargs),
        ssp_ages_yr,
        tab_lbt_yr,
        conserved=False,
    )


def _cic_integrand(ssp_ages_yr, sfh_fn, sfh_kwargs, sfh_spec_fn, tab_lbt_yr):
    """The dense (age, SFR) integrand every CIC weight kernel consumes.

    One source for the three call sites — :meth:`StellarSEDComponent.apply`'s
    delta and per-age-metallicity branches, and the SED-free
    :meth:`StellarSEDComponent.compute_joint_weights` fast path. They were
    identical line-for-line and had to stay that way: #982 exists because a
    divergence between ``apply`` and the fast path is invisible until the two
    disagree on a fit. Building the grid here means an edge fix (#1522) cannot
    land in two of the three.

    Parameters
    ----------
    ssp_ages_yr : ndarray, shape (n_age,)
        Ascending SSP template ages [yr].
    sfh_fn : callable
        ``sfh_fn(age_yr, **sfh_kwargs) -> SFR [Msun/yr]``.
    sfh_kwargs : dict
        Keyword arguments for ``sfh_fn``.
    sfh_spec_fn : callable
        The registry's SFH function, for bin-edge discovery on binned families.
    tab_lbt_yr : ndarray or None
        A tabulated history's own lookback nodes [yr], else None.

    Returns
    -------
    age_yr : ndarray, shape (n,)
        Ascending dense lookback-age grid [yr].
    sfr : ndarray, shape (n,)
        SFR on that grid [Msun/yr].
    """
    fine_age_yr = _refine_sfh_table_ages(ssp_ages_yr)
    hi_yr = ssp_ages_yr[-1]
    if tab_lbt_yr is not None:
        # #1522: a table can carry mass older than the oldest template. Parametric
        # families renormalize past the grid edge, so they neither need this nor
        # should get it.
        fine_age_yr, hi_yr = _extend_integrand_to_history(fine_age_yr, tab_lbt_yr, ssp_ages_yr)
    # #765: inject the SFH's exact bin edges as knots so the step transitions of
    # binned SFHs are represented sharply. A tabulated SFH's own nodes ARE its
    # exact knots. Parametric families have no bin edges (None).
    edges_yr = tab_lbt_yr if tab_lbt_yr is not None else _sfh_bin_edges_yr(sfh_spec_fn, sfh_kwargs)
    if edges_yr is not None:
        fine_age_yr = _inject_edge_knots(fine_age_yr, edges_yr, ssp_ages_yr[0], hi_yr)
    sfr = sfh_fn(fine_age_yr, **sfh_kwargs)
    _warn_if_history_exceeds_ssp_grid(fine_age_yr, sfr, ssp_ages_yr, tab_lbt_yr)
    return fine_age_yr, sfr


def _youngest_bin_lookback_multiplier(ssp_lg_age_gyr):
    r"""Per-age weight multiplier extending the youngest physical SSP bin to lookback 0 (#821).

    DSPS derives its age-bin edges as log-midpoints of ``ssp_lg_age_gyr``, so the
    youngest *physical* (finite-age) bin spans lookback :math:`[e_{lo}, e_{hi}]`
    with :math:`e_{lo} = 10^{\,lg_0 - \Delta lg/2} > 0`. The :math:`[0, e_{lo}]`
    sliver holds the most ionizing stars (``n_ly`` drops ~300x by 10 Myr), so
    dropping it biases the ionizing-photon rate Q_H low vs the exact SFH->SSP
    convolution — measured ~4% on FSPS/MILES grids and up to ~31% for BPASS
    (binary stars sustain ionizing output to later ages). The #809 lookback-0
    table knot feeds the SFH down to the observation time, but DSPS still clips
    the *bin* at :math:`e_{lo}`.

    SFR is constant to <0.1% over the ~0.1 Myr sliver, so the youngest bin's mass
    is

    .. math::

        M[0, e_{hi}] = \frac{e_{hi}}{e_{hi} - e_{lo}} \, M[e_{lo}, e_{hi}],

    a grid-only factor :math:`f = e_{hi}/(e_{hi} - e_{lo})`. Scaling the youngest
    weight column by ``f`` and renormalizing recovers the true age PDF
    :math:`M[0, e_{hi}] / M[0, e_{top}]` exactly (the boost numerator becomes
    :math:`M[0, e_{hi}]`, the renormalization divides by :math:`M[0, e_{top}]`),
    while leaving ``total_mass`` unchanged — so mass conservation holds.

    A leading ``age = 0`` template (``lg = -inf``, e.g. BC03 stelib) already
    collapses the youngest physical bin's lower edge to lookback 0 inside DSPS,
    giving :math:`e_{lo} = 0` and ``f = 1`` — a correct no-op.

    Parameters
    ----------
    ssp_lg_age_gyr : array_like, shape (n_age,)
        log10(SSP template age / Gyr), ascending. A leading ``-inf`` flags an
        ``age = 0`` template.

    Returns
    -------
    ndarray, shape (n_age,)
        Per-age multiplier: :math:`f` at the youngest finite-age bin, ``1.0``
        elsewhere.

    Notes
    -----
    Pure JAX (no host-side branching or static indexing), so the correction is
    JIT/grad/vmap-safe even when the SSP grid is threaded as a *traced* argument
    (the SSP-as-JIT-parameter path). The factor depends only on the SSP
    age grid, never on a fitted parameter, so no gradient flows through it. The
    double ``where`` is the standard JAX safe-divide guard for the ``-inf`` edge
    of an ``age = 0`` template (avoids ``0/0`` in that unused entry).
    """
    lg = jnp.asarray(ssp_lg_age_gyr)
    mid = 0.5 * (lg[:-1] + lg[1:])  # interior log-midpoint edges
    lo = jnp.concatenate([lg[:1] - 0.5 * (lg[1:2] - lg[:1]), mid])  # per-bin lower edge
    hi = jnp.concatenate([mid, lg[-1:] + 0.5 * (lg[-1:] - lg[-2:-1])])  # per-bin upper edge
    e_lo = 10.0**lo  # -> 0 where lo = -inf (the age=0 neighbor)
    e_hi = 10.0**hi
    denom = e_hi - e_lo
    boost = jnp.where(denom > 0.0, e_hi / jnp.where(denom > 0.0, denom, 1.0), 1.0)
    finite = jnp.isfinite(lg)
    is_youngest = finite & (jnp.cumsum(finite.astype(jnp.int32)) == 1)  # one-hot at j0
    return jnp.where(is_youngest, boost, 1.0)


@jax.custom_jvp
def _mass_scale_lnu(per_msun_lsun, total_mass):
    r"""``total_mass * per_msun_lsun * L_sun`` with a float32-safe reverse pass (#1206).

    Scales a per-solar-mass SSP luminosity [Lsun/(Hz*Msun)] to the physical
    quantity [erg/s/Hz] by the formed stellar mass and the solar-luminosity
    constant. Used for the full-wave stellar SED and for every per-filter /
    sub-band / spectrum photometry-LUT tensor that carries the same scaling.

    Parameters
    ----------
    per_msun_lsun : array_like
        Per-solar-mass weighted SSP luminosity. [Lsun/(Hz*Msun)]
    total_mass : array_like, scalar
        Formed stellar mass. [Msun]

    Returns
    -------
    ndarray
        ``total_mass * per_msun_lsun * L_sun`` [erg/s/Hz], same shape as input.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes. The forward is the plain triple product, so the
    value is bit-identical to writing it inline. The custom VJP exists only to
    pin the reverse pass's multiply *order*, which autodiff otherwise leaves to
    XLA:

    * ``d/d(per_msun) = (g * total_mass) * L_sun`` -- ``g * total_mass`` first,
      so the standalone Jacobian ``total_mass * L_sun`` ~3.8e43 (``inf`` in
      float32) is never formed.
    * ``d/d(total_mass) = sum(g * (per_msun * L_sun))`` -- ``per_msun * L_sun``
      (~3.8e18) first, avoiding the ``g * per_msun`` (~1e-41) float32 underflow.

    Under a plain product, XLA's fused reverse pass materializes ``total_mass *
    L_sun`` as a standalone Jacobian and overflows float32 to ``inf``. Multiplied
    by any incoming cotangent that is itself finite, the ``inf`` still poisons
    the SSP-contraction ``dot_general`` (``inf``/``nan``) regardless of the
    cotangent's magnitude. Folding L_sun into the einsum operand does not survive
    either: when the SSP grid is threaded as an XLA ``Parameter`` (the inference
    hot path) the algebraic simplifier pulls the constant back out. A
    custom-rule boundary is the one thing XLA reassociation cannot cross.
    Identical in float64 (the gradient differs only at the last bit from the
    multiply reorder), so this is **one path for both precisions** — float64
    does not need the pin but is unharmed by it, and a dtype-branched
    implementation would mean the float64 tests stopped exercising what
    float32 actually runs.

    The rule is a ``custom_jvp``, not a ``custom_vjp``. Both pin the order, but
    a ``custom_vjp`` is **opaque to forward mode** — ``jvp`` raises
    ``TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp
    function`` — and geoVI builds its metric with forward-mode autodiff, so the
    ``custom_vjp`` spelling turned ``test_geovi_mode_stable_convergence`` red.
    A ``custom_jvp`` serves forward mode directly and reverse mode by
    transposition, and the transpose of the groupings below is exactly the
    hand-written VJP this replaces.
    """
    return total_mass * per_msun_lsun * LSUN_ERG_PER_S


@_mass_scale_lnu.defjvp
def _mass_scale_lnu_jvp(primals, tangents):
    """Grouped so neither mode forms ``total_mass * L_sun`` (~3.8e43, inf in float32).

    Transposing these two terms gives ``d/d(per_msun) = (g*total_mass)*L_sun``
    and ``d/d(total_mass) = sum(g*(per_msun*L_sun))`` — the safe reverse pass.
    """
    per_msun_lsun, total_mass = primals
    d_per_msun, d_total_mass = tangents
    primal_out = total_mass * per_msun_lsun * LSUN_ERG_PER_S
    # ``optimization_barrier`` is what keeps the grouping: a ``custom_jvp``'s
    # transpose is inlined into the backward jaxpr and XLA is then free to
    # re-associate it back into ``total_mass * L_sun`` (3.8e43, inf in float32)
    # — measured, nine float32 gradient tests went red without these barriers.
    # Unlike ``custom_vjp``, a barrier blocks reassociation WITHOUT making the
    # function opaque to forward mode, so geoVI can still differentiate it.
    barrier = jax.lax.optimization_barrier
    tangent_out = barrier(d_per_msun * total_mass) * LSUN_ERG_PER_S + d_total_mass * barrier(
        per_msun_lsun * LSUN_ERG_PER_S
    )
    return primal_out, tangent_out


@jax.custom_jvp
def _flux_weighted_node(num, den):
    r"""``num / den`` (a flux-weighted mean wavelength) with a float32-safe VJP (#1206).

    The sub-band node wavelength is ``λ_k = Σ(w·λ·φ) / Σ(w·φ)`` — a ratio whose
    value is a well-defined ~5000 Å regardless of how small the denominator
    ``den = Σ(w·φ)`` is, but whose *autodiff* Jacobian ``d/d(den) = -num/den^2``
    overflows float32 for a tiny ``den`` (a near-zero-weight sub-band). The
    downstream cotangent is itself weighted by ``~den`` (a near-zero-weight node
    barely affects the dust law), so the *true* gradient is finite — the ``den``
    cancels — but XLA's fused reverse pass materializes ``num/den^2`` standalone
    first and hits ``inf``. When the node is not consumed at all the cotangent
    is exactly 0 and the result is ``0*inf = nan``.

    Parameters
    ----------
    num, den : array_like
        Numerator ``Σ(w·λ·φ)`` and denominator ``Σ(w·φ)`` of the flux-weighted
        mean wavelength. ``den`` is pre-floored away from exact zero by the
        caller.

    Returns
    -------
    ndarray
        ``num / den``, same shape.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes. The custom VJP forms ``g/den`` *first*
    (finite: the cotangent scales with ``den``, and is 0 when the node is
    unused) and only then multiplies by ``num/den`` — the standalone ``den^2``
    is never materialized. Identical in float64, and deliberately **one path
    for both precisions**: see :func:`_mass_scale_lnu` for why this is a
    ``custom_jvp`` rather than a ``custom_vjp`` (forward mode, which geoVI
    needs, cannot cross a ``custom_vjp``).
    """
    return num / den


@_flux_weighted_node.defjvp
def _flux_weighted_node_jvp(primals, tangents):
    """``d(num/den) = (d_num - (num/den)*d_den) / den`` — never forms ``den**2``.

    The naive form ``d_num/den - num*d_den/den**2`` squares the denominator;
    factoring the ``1/den`` out leaves one division by ``den`` and reuses the
    primal quotient, which is a well-behaved ~5000 Angstrom. Transposed, this
    gives ``g/den`` and ``-(g/den)*(num/den)`` — the safe reverse pass.
    """
    num, den = primals
    d_num, d_den = tangents
    quotient = num / den
    return quotient, (d_num - quotient * d_den) / den


def _age_weights_cic(age_yr, sfr, ssp_ages_yr, t_obs_gyr):
    r"""Interpolation-weighted (cloud-in-cell) SSP age weights (#964).

    Convolves the SFH with the SSP grid the way FSPS does: each mass parcel
    :math:`dM = \mathrm{SFR}(t)\,dt` at lookback age :math:`t` is split
    between the two bracketing SSP template ages with linear weights in
    :math:`\log_{10} t` — equivalent to evaluating a log-age-interpolated SSP
    spectrum at the parcel's exact age. DSPS's histogram kernel
    (``calc_age_weights_from_sfh_table``) instead assigns each parcel wholly
    to its log-midpoint age bin, and interpolates :math:`\log_{10} M(<t)` in
    :math:`\log_{10} t` across bin edges — which annihilates the mass in any
    table segment straddling the SFH start (measured: the 5.012 Gyr node got
    exactly zero weight for a delayed-τ SFH with age = 5 Gyr, re-attributing
    3.8 % of the mass to younger, brighter nodes → a +1.2 % optical CSP bias
    vs FSPS, bagpipes, and a dense reference; #964).

    Parameters
    ----------
    age_yr : ndarray, shape (n,)
        Ascending lookback ages [yr] of the SFH integrand (the dense grid
        from :func:`_refine_sfh_table_ages`, plus any edge knots).
    sfr : ndarray, shape (n,)
        Star-formation rate at ``age_yr`` [Msun/yr].
    ssp_ages_yr : ndarray, shape (n_age,)
        Ascending SSP template ages [yr].
    t_obs_gyr : float
        Cosmic age at the observation redshift [Gyr]; mass at lookback ages
        older than this (pre-Big-Bang) is dropped, matching
        :func:`_build_dsps_sfh_table`'s invalid-bin zeroing.

    Returns
    -------
    age_weights : ndarray, shape (n_age,)
        Normalized (sum = 1) SSP age weights.
    total_mass : ndarray, shape ()
        Trapezoidal mass formed on ``age_yr`` [Msun], excluding the
        prepended ``[0, age_yr[0]]`` lookback segment — that sliver
        *redistributes* mass into the youngest bin (the #538 young-knot
        contract), it must not inflate the normalization.

    Notes
    -----
    **JIT/grad/vmap-safe**: static shapes, pure ``jnp``; the scatter-adds
    are ``.at[].add``. Replaces both the #538 young-boundary knot (the
    prepended lookback-0 parcel lands wholly on the youngest node) and the
    #821 youngest-bin multiplier (no log-midpoint bin edge exists to clip),
    so callers must NOT also apply ``_youngest_bin_lookback_multiplier``.
    A leading ``age = 0`` SSP template (lg = -inf, e.g. BC03 stelib) falls
    back to linear-in-age interpolation weights for parcels in its segment.
    """
    contrib, idx, f, total_mass, _ = _cic_parcels(age_yr, sfr, ssp_ages_yr, t_obs_gyr)
    n_age = ssp_ages_yr.shape[0]
    w = (
        jnp.zeros(n_age, dtype=contrib.dtype)
        .at[idx]
        .add(contrib * (1.0 - f))
        .at[idx + 1]
        .add(contrib * f)
    )
    return w / jnp.maximum(jnp.sum(w), 1e-300), total_mass


def _cic_parcels(age_yr, sfr, ssp_ages_yr, t_obs_gyr):
    """Shared parcel machinery for the CIC weight builders (#964).

    Returns ``(contrib, idx, f, total_mass, age)``: per-parcel trapezoid
    masses [Msun] on the lookback-0-extended integrand, the lower bracketing
    SSP node index and its log-age interpolation fraction, the #538-contract
    total mass, and the extended age grid [yr].
    """
    # Extend the integrand to lookback 0 holding SFR constant over the
    # first ~0.1 Myr (#538): the [0, age0] mass belongs to the youngest node.
    age = jnp.concatenate([jnp.zeros((1,), age_yr.dtype), age_yr])
    sfr_ext = jnp.concatenate([sfr[:1], sfr])
    sfr_valid = jnp.where(age < t_obs_gyr * 1e9, sfr_ext, 0.0)

    # Trapezoid node masses: dM_i = SFR_i * dt_i with midpoint widths.
    d = jnp.diff(age)
    dt = jnp.concatenate([0.5 * d[:1], 0.5 * (d[:-1] + d[1:]), 0.5 * d[-1:]])
    contrib = sfr_valid * dt

    # Split each parcel between bracketing SSP nodes, linear in log10(age)
    # (FSPS's isochrones are log-spaced; the dense-reference arbitration in
    # #964 found FSPS consistent with log-age interpolation to 1e-4).
    n_age = ssp_ages_yr.shape[0]
    lg_nodes = jnp.log10(jnp.maximum(ssp_ages_yr, 1e-30))
    lg_age = jnp.log10(jnp.maximum(age, 1e-30))
    idx = jnp.clip(jnp.searchsorted(lg_nodes, lg_age) - 1, 0, n_age - 2)
    f_log = (lg_age - lg_nodes[idx]) / (lg_nodes[idx + 1] - lg_nodes[idx])
    # Fallback for a leading age = 0 template (lg = -inf): linear in age.
    f_lin = (age - ssp_ages_yr[idx]) / jnp.maximum(ssp_ages_yr[idx + 1] - ssp_ages_yr[idx], 1e-30)
    f = jnp.clip(jnp.where(jnp.isfinite(f_log), f_log, f_lin), 0.0, 1.0)
    total_mass = jnp.maximum(jnp.trapezoid(sfr_valid[1:], age_yr), 0.0)
    return contrib, idx, f, total_mass, age


def _lgmet_weights(log_z, lgmet_scatter, ssp_lgmet):
    """DSPS lognormal-MDF metallicity weights, operands canonicalized to one dtype.

    Parameters
    ----------
    log_z : array_like, scalar
        Absolute ``log10(Z)`` of the parcel. [dex]
    lgmet_scatter : array_like, scalar
        Width of the lognormal MDF. [dex]
    ssp_lgmet : array_like, shape (n_met,)
        SSP metallicity axis, absolute ``log10(Z)``. [dex]

    Returns
    -------
    ndarray, shape (n_met,)
        Non-negative weights summing to 1.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes.

    ``ssp_lgmet`` is the cached host SSP grid, built once at load time, so it
    stays float64 even inside ``jax.enable_x64(False)`` while the parameters
    arrive as float32 tracers. DSPS then sizes its ``dt`` array from the float64
    grid and scatters an f32-derived value into it
    (``dt.at[1:-1].set(dtmids)`` in ``dsps.sed.metallicity_weights``), which JAX
    reports today as::

        FutureWarning: scatter inputs have incompatible types: cannot safely
        cast value from dtype=float32 to dtype=float64 ...
        In future JAX releases this will result in an error.

    Canonicalizing all three operands first removes the mixed-dtype scatter.
    Under ``x64=True`` the canonical float *is* float64, so this is a no-op
    there and float64 results are bit-unchanged — the property that makes the
    pattern safe to apply broadly. Same treatment as
    :func:`tengri.utils.interpolation.compute_grid_weights` (#1206, #1448).

    All three DSPS lognormal-MDF call sites in this module route through here so
    the canonicalization cannot drift between them.
    """
    from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf

    args = canonical_dsps_kwargs(log_z=log_z, lgmet_scatter=lgmet_scatter, ssp_lgmet=ssp_lgmet)
    return calc_lgmet_weights_from_lognormal_mdf(
        args["log_z"], args["lgmet_scatter"], args["ssp_lgmet"]
    )


def _joint_weights_cic_met_table(
    age_yr, sfr, ssp_ages_yr, t_obs_gyr, lgmet_on_ssp_ages, lgmet_scatter, ssp_lgmet
):
    r"""CIC joint (met, age) weights for a per-age metallicity table (#964).

    The per-age-metallicity analog of :func:`_age_weights_cic`: each mass
    parcel is split between its bracketing SSP age nodes with log-age CIC
    weights, and simultaneously distributed over the metallicity axis with
    the lognormal MDF centered on the parcel's metallicity —
    ``lgmet_on_ssp_ages`` interpolated (linear in log-age) to the parcel age.
    Keeps the ramp / chem_evol paths consistent with the delta path, so
    degenerate per-age modes (constant table, zero step, ...) reduce to the
    delta result exactly (pinned by ``test_met_modes_components``).

    Parameters
    ----------
    age_yr, sfr, ssp_ages_yr, t_obs_gyr
        As in :func:`_age_weights_cic`.
    lgmet_on_ssp_ages : ndarray, shape (n_age,)
        log10(Z) absolute at each SSP age (ascending lookback).
    lgmet_scatter : float
        Lognormal MDF scatter [dex].
    ssp_lgmet : ndarray, shape (n_met,)
        SSP grid metallicities, log10(Z) absolute.

    Returns
    -------
    joint_weights : ndarray, shape (n_met, n_age)
        Normalized (sum = 1) joint weights.
    total_mass : ndarray, shape ()
        The #538-contract trapezoidal mass [Msun].

    Notes
    -----
    **JIT/grad/vmap-safe**: static shapes; the met axis uses DSPS's own
    lognormal-MDF kernel vmapped over parcels.
    """
    contrib, idx, f, total_mass, age = _cic_parcels(age_yr, sfr, ssp_ages_yr, t_obs_gyr)
    lg_nodes = jnp.log10(jnp.maximum(ssp_ages_yr, 1e-30))
    lg_age = jnp.log10(jnp.maximum(age, 1e-30))
    lgmet_par = jnp.interp(lg_age, lg_nodes, lgmet_on_ssp_ages)
    met_w = jax.vmap(lambda g: _lgmet_weights(g, lgmet_scatter, ssp_lgmet))(
        lgmet_par
    )  # (n_parcel, n_met)
    n_met = ssp_lgmet.shape[0]
    n_age = ssp_ages_yr.shape[0]
    joint = (
        jnp.zeros((n_met, n_age), dtype=contrib.dtype)
        .at[:, idx]
        .add(met_w.T * (contrib * (1.0 - f))[None, :])
        .at[:, idx + 1]
        .add(met_w.T * (contrib * f)[None, :])
    )
    return joint / jnp.maximum(jnp.sum(joint), 1e-300), total_mass


def _tabulated_sfh(params, t_obs_gyr):
    """Interp closure + lookback knots for a runtime tabulated SFH (#996).

    **The single source** shared by :meth:`StellarSEDComponent.apply` (the exact
    forward) and :meth:`StellarSEDComponent.compute_joint_weights` (the SED-free
    fast path). Reading a simulation history two different ways is precisely the
    divergence #1395 was — there, the fast path never learned to read it at all
    and silently evaluated an all-zero placeholder — so both routes call this.

    SFR is edge-clamped outside the table (the ``jnp.interp`` convention);
    lookbacks older than ``t_obs`` are dropped later by the CIC ``t_obs`` cutoff.
    Dict-key presence is static under jit while the array *values* stay traced,
    so two calls with different same-length tables share one compile.

    Parameters
    ----------
    params : Mapping
        Must contain ``sfh_t_gyr`` (cosmic time [Gyr]) and ``sfh_sfr`` [Msun/yr].
    t_obs_gyr : ndarray, shape ()
        Cosmic time at the observed redshift [Gyr].

    Returns
    -------
    sfh_fn : callable
        ``f(t_lookback_yr, **kwargs) -> SFR [Msun/yr]``, matching the registry
        SFH-callable signature so every downstream consumer is unchanged.
    lbt_yr : ndarray, shape (n_t,)
        Table nodes as ascending lookback time [yr]. Doubles as the exact-knot
        edge set for :func:`_inject_edge_knots`.
    order : ndarray, shape (n_t,)
        The descending-cosmic-time argsort, so a caller can reorder a companion
        array (e.g. ``met_history``) onto the same nodes.

    Raises
    ------
    ValueError
        If either runtime array is absent. Raising is required, not optional:
        the registry ``fn`` for ``sfh_model='table'`` is an all-zero placeholder,
        so proceeding would return zero weights and zero mass, finite and
        unwarned (#1395).

    Notes
    -----
    **JIT-compatible**: yes — ``argsort`` / ``interp`` on traced values.
    """
    if "sfh_t_gyr" not in params or "sfh_sfr" not in params:
        raise ValueError(
            "sfh_model='table' requires the runtime arrays "
            "params['sfh_t_gyr'] (cosmic time [Gyr]) and "
            "params['sfh_sfr'] [Msun/yr] (#996)."
        )
    # Descending cosmic time == ascending lookback; argsort keeps the pairing
    # under jit for any input ordering.
    order = jnp.argsort(-jnp.asarray(params["sfh_t_gyr"]))
    lbt_yr = jnp.maximum((t_obs_gyr - jnp.asarray(params["sfh_t_gyr"])[order]) * 1e9, 0.0)
    sfr = jnp.asarray(params["sfh_sfr"])[order]

    def sfh_fn(t_lookback_yr, **_kw):
        return jnp.interp(t_lookback_yr, lbt_yr, sfr)

    return sfh_fn, lbt_yr, order


def _tabulated_lgmet_on_ssp_ages(params, config, ssp_lg_age_gyr, tab_lbt_yr, tab_order):
    """Per-age log10(Z) absolute on the SSP ages, for ``metallicity_model='table'``.

    The single source shared by the exact forward and the fast path, for the
    same reason as :func:`_tabulated_sfh`. Two sources: a build-time table on
    the component config, or the runtime ``params['met_history']`` whose time
    axis **is** the tabulated SFH's ``sfh_t_gyr`` nodes (#996).

    Parameters
    ----------
    params : Mapping
        May contain ``met_history``, log10(Z/Zsun) at the SFH table's nodes.
    config : StellarSEDComponentConfig
        Supplies the optional build-time ``met_table_log_age_yr`` /
        ``met_table_log_z_abs``.
    ssp_lg_age_gyr : ndarray, shape (n_age,)
        SSP grid log10(age/Gyr).
    tab_lbt_yr : ndarray, shape (n_t,) or None
        Ascending lookback nodes from :func:`_tabulated_sfh`. Required for the
        runtime path — the Z(t) nodes have no time axis of their own.
    tab_order : ndarray, shape (n_t,) or None
        The matching argsort, applied to ``met_history`` so Z and SFR stay paired.

    Returns
    -------
    lgmet_on_ssp_ages : ndarray, shape (n_age,)
        log10(Z) **absolute** at each SSP age.
    met_log_age_yr : ndarray, shape (n_t,)
        The resolved table's age axis — returned so ``apply`` can reuse the very
        same table for its SFH-grid diagnostic instead of re-resolving it.
    met_log_z_abs : ndarray, shape (n_t,)
        The resolved table's log10(Z) absolute values.

    Raises
    ------
    NotImplementedError
        If ``met_history`` is given without ``sfh_model='table'``.
    ValueError
        If neither the build-time table nor ``met_history`` is available.

    Notes
    -----
    **JIT-compatible**: yes.
    """
    has_cfg_table = (
        config.met_table_log_age_yr is not None and config.met_table_log_z_abs is not None
    )
    if has_cfg_table:
        met_log_age_yr = jnp.asarray(config.met_table_log_age_yr)
        met_log_z_abs = jnp.asarray(config.met_table_log_z_abs)
    elif "met_history" in params:
        if tab_lbt_yr is None:
            raise NotImplementedError(
                "met_history needs sfh_model='table' for its time "
                "axis (the Z(t) nodes are the sfh_t_gyr nodes, #996)."
            )
        met_log_age_yr = jnp.log10(jnp.maximum(tab_lbt_yr, 1.0))
        met_log_z_abs = jnp.asarray(params["met_history"])[tab_order] + LOG10_ZSUN
    else:
        raise ValueError(
            "metallicity_model='table' requires met_table_log_age_yr "
            "+ met_table_log_z_abs on StellarSEDComponentConfig, or "
            "the runtime params['met_history'] array (#996)."
        )
    lgmet_on_ssp_ages = tabulated_metallicity_on_ssp_grid(
        ssp_lg_age_gyr, met_log_age_yr, met_log_z_abs
    )
    return lgmet_on_ssp_ages, met_log_age_yr, met_log_z_abs


def _inject_edge_knots(fine_age_yr, edges_yr, lo_yr, hi_yr):
    """Add SFH bin edges as exact knots to a dense lookback-age integrand (#765).

    The dense log grid from :func:`_refine_sfh_table_ages` never lands exactly
    on a step SFH's bin edges, so DSPS interpolates across each transition and
    smears the mass — a resolution-insensitive 2-4.5 % optical residual vs
    Prospector (#765, follow-up to the #758/#764 dense integrand). Each edge is
    doubled just inside/outside (±1e-6 fractional) so the step is represented
    sharply; knots are clamped into the SSP age span and the result re-sorted
    ascending. The output size is static (``len(fine_age_yr) + 2*len(edges_yr)``),
    so this stays JIT/grad/vmap-safe.

    Parameters
    ----------
    fine_age_yr : ndarray, shape (n_fine,)
        Dense ascending lookback-age grid [yr].
    edges_yr : ndarray, shape (n_edges,)
        SFH bin edges in lookback time [yr] (may be traced).
    lo_yr, hi_yr : float
        SSP age span bounds [yr] to clamp knots into.

    Returns
    -------
    ndarray, shape (n_fine + 2*n_edges,)
        Ascending lookback-age grid with edge knots merged in.
    """
    eps = 1e-6
    knots = jnp.concatenate([edges_yr * (1.0 - eps), edges_yr * (1.0 + eps)])
    knots = jnp.clip(knots, lo_yr * (1.0 + eps), hi_yr * (1.0 - eps))
    return jnp.sort(jnp.concatenate([fine_age_yr, knots]))


def _build_dsps_sfh_table(age_yr, sfr, t_obs_gyr, add_young_knot=False):
    """Ascending cosmic-time (t, SFR) table for DSPS, NaN-safe at the high-z edge.

    SSP ages older than the universe at the observation redshift imply negative
    cosmic time; those bins are clamped onto a strictly-increasing ramp and
    zeroed so they contribute nothing (avoids the DSPS NaN at #683).

    Parameters
    ----------
    age_yr : ndarray, shape (n,)
        Ascending lookback ages [yr].
    sfr : ndarray, shape (n,)
        Star-formation rate at ``age_yr`` [Msun/yr].
    t_obs_gyr : float
        Cosmic age at the observation redshift [Gyr].
    add_young_knot : bool, optional
        Prepend a lookback-0 knot so DSPS integrates the youngest SSP bin down
        to the observation time (#538). Default ``False``.

    Returns
    -------
    t_cosmic_asc : ndarray, shape (n,) or (n+1,)
        Strictly-increasing cosmic time [Gyr]. Length ``n+1`` when
        ``add_young_knot`` is set.
    sfr_asc : ndarray, shape (n,) or (n+1,)
        SFR aligned to ``t_cosmic_asc`` [Msun/yr] (pre-Big-Bang bins zeroed).
    total_mass : float
        Trapezoidal mass formed [Msun], EXCLUDING the young-boundary knot's
        ``[0, age0]`` segment so the knot redistributes — not inflates — mass.
    """
    T_TABLE_MIN = 0.01  # Gyr; matches dsps.constants.T_TABLE_MIN
    # Young-boundary knot (#538): ``age_yr`` starts at the youngest SSP age
    # (~1 Myr), so the mass formed between lookback 0 and that age is never
    # integrated into the youngest SSP bin — under-weighting the ionizing
    # population and biasing Q_H ~16 % low vs the analytic (and CIGALE) SFH->SSP
    # convolution (n_ly drops 3+ dex past ~10 Myr, so the youngest bin dominates
    # Q_H). Prepend a lookback-0 knot holding SFR constant from the youngest
    # sample (SFR varies negligibly over the first Myr) so DSPS redistributes the
    # youngest-bin mass down to the observation time. Static size (n -> n+1) keeps
    # the path JIT/grad/vmap-safe. Gated off by default so the shared coarse
    # table (mass guard + per-age-metallicity path, which aligns a length-n
    # metallicity table) is byte-unchanged.
    if add_young_knot:
        age_yr = jnp.concatenate([jnp.zeros((1,), age_yr.dtype), age_yr])
        sfr = jnp.concatenate([sfr[:1], sfr])
    age_gyr = age_yr / 1e9
    n = age_yr.shape[0]
    t_cosmic_raw = t_obs_gyr - age_gyr
    t_cosmic_floor = jnp.maximum(t_cosmic_raw, T_TABLE_MIN)
    valid_asc = (t_cosmic_raw > 0.0)[::-1]
    t_cosmic_asc_raw = t_cosmic_floor[::-1]
    sfr_asc_raw = sfr[::-1]
    n_invalid = jnp.sum(~valid_asc)
    idx_pos = jnp.arange(n)
    is_invalid_pos = idx_pos < n_invalid
    ramp = T_TABLE_MIN + (T_TABLE_MIN * 0.5) * (idx_pos + 1) / jnp.maximum(n_invalid, 1)
    t_cosmic_asc = jnp.where(is_invalid_pos, ramp, t_cosmic_asc_raw)
    # Guarantee strictly-increasing knots: at high z boundary-valid bins can
    # clamp to T_TABLE_MIN below the invalid-bin ramp, which DSPS NaNs on. #683
    t_cosmic_asc = enforce_increasing_cosmic_time(t_cosmic_asc)
    sfr_asc = jnp.where(is_invalid_pos, 0.0, sfr_asc_raw)
    if add_young_knot:
        # Exclude the young-boundary knot's [0, age0] segment (the last ascending
        # element, at t_cosmic = t_obs) from the normalization: the knot
        # REDISTRIBUTES mass into the youngest SSP bin via DSPS's (sum-to-one)
        # weights, it must not inflate the total. Trapezoid over all-but-the-knot
        # is exactly the as-given SFH mass, so ``sum(age_weights) ==
        # 10**log_total_mass`` is preserved (#538).
        total_mass = jnp.maximum(jnp.trapezoid(sfr_asc[:-1], t_cosmic_asc[:-1] * 1e9), 0.0)
    else:
        total_mass = jnp.maximum(jnp.trapezoid(sfr_asc, t_cosmic_asc * 1e9), 0.0)
    return t_cosmic_asc, sfr_asc, total_mass


from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.utils.physics_constants import C_AA, H_PLANCK

__all__ = [
    "StellarSEDComponent",
    "StellarSEDComponentConfig",
    "StellarSEDComponentState",
]


#: Smallest sub-band weight whose node wavelength ``sub_num / sub_phi`` still has
#: a representable **derivative** (#1397). The value is a ratio of two tiny
#: numbers and stays finite far below this, but the quotient rule needs
#: ``sub_phi**2``, which underflows to zero once ``sub_phi`` drops below
#: ``sqrt(2.2e-308) ~ 1.5e-154`` — and ``x / 0`` is the NaN that poisons the
#: gradient. Testing ``sub_phi != 0.0`` therefore does not protect autodiff: a
#: narrow SFH drives sub-band fluxes to 1e-250 and below while every one of them
#: is still nonzero. Sub-bands under this floor carry no measurable flux, so
#: their node falls back to the band effective wavelength exactly as an
#: identically-zero sub-band does.
def _subband_live_floor() -> float:
    """The ``1e-150`` liveness floor above, made representable (#1568).

    ``1e-150`` is far below float32's smallest subnormal (1.4e-45), so in
    float32 ``jnp.abs(sub_phi) > _subband_live_floor()`` degenerated to
    ``> 0.0`` — exactly the ``sub_phi != 0.0`` test the comment above explains
    is insufficient, and which #1397 replaced this floor *because* it does not
    protect the backward pass.

    Evaluated at trace time so it resolves against the working dtype; float64
    keeps ``1e-150`` unchanged.
    """
    from tengri.utils.scale import representable_floor

    return representable_floor(1e-150)


# Lyman limit — wavelengths below this contribute to the ionizing
# photon rate (matches :mod:`tengri.components.nebular.ionizing_spectrum`).
_HI_LIMIT_AA: float = 911.76


def _integrate_nion_log10(
    sed_lnu: jnp.ndarray, wave: jnp.ndarray, log10_scale: float = 0.0
) -> jnp.ndarray:
    r"""Log-domain ionizing photon rate (core Q_H integral for float32 safety).

    THE single source of the Q_H integral — log-domain computation to prevent
    float32 overflow (Q_H ~ 1e56 exceeds float32 max ~3.4e38). Integrates
    :math:`Q_H = \int_{\nu>\nu_{912}} L_\nu/(h\nu)\,d\nu` with the partial-bin
    Lyman-limit correction (#537): the boundary bin's contribution is a rectangle
    from ``nu_edge`` to the last ionizing grid point, not the trapezoid triangle
    a hard mask would give.

    The computation normalizes the SED by its peak, defers the Planck constant
    division, and performs the trapezoid integral in linear-normalized space,
    keeping all intermediates within float32 range (issue #1206).

    Parameters
    ----------
    sed_lnu : ndarray, shape (n_wave,)
        Rest-frame stellar :math:`L_\nu` [erg/s/Hz] (pre-dust intrinsic SED).
    wave : ndarray, shape (n_wave,)
        Wavelength grid [Angstrom]; must span the Lyman limit (a few points
        above 911.76 A suffice — the boundary bin needs the first non-ionizing
        point).
    log10_scale : float, optional
        Log10-scale offset [dex] to apply to the result. Default 0.0 (no scaling).
        Used for mass-scaling: ``log10_scale = log10(total_mass * LSUN_ERG_PER_S)``.

    Returns
    -------
    ndarray, shape ()
        Log10 ionizing photon rate [dex relative to photons/s].

    Notes
    -----
    **JIT-compatible**: yes. Bit-exact whether given the full SSP grid or an
    ionizing-only slice. Peak normalization and deferred 1/h keep all
    intermediates within float32 range.
    """
    # stop_gradient: pure factorization constant (#1436) — log10(peak) is added back
    # below, so the peak cancels analytically.
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(sed_lnu), initial=0.0))  # #1207
    peak = jnp.where(peak > 0, peak, jnp.ones_like(peak))
    ell = sed_lnu / peak  # O(1) normalized L_nu
    nu = C_AA / wave
    nu_edge = C_AA / _HI_LIMIT_AA
    integrand = ell / nu  # NO H_PLANCK division — that's deferred to avoid f32 overflow
    ionizing_mask = wave < _HI_LIMIT_AA
    integrand_masked = jnp.where(ionizing_mask, integrand, 0.0)
    idx_below = jnp.argmax(jnp.where(ionizing_mask, jnp.arange(wave.shape[0]), -1))
    idx_above = idx_below + 1
    integrand_below = integrand[idx_below]
    # Boundary bin: subtract the trapezoid triangle, add the true rectangle.
    triangle_overcount = 0.5 * integrand_below * jnp.abs(nu[idx_below] - nu[idx_above])
    rectangle_correct = integrand_below * jnp.abs(nu[idx_below] - nu_edge)
    nion_bulk = jnp.abs(jnp.trapezoid(integrand_masked, nu))
    norm = nion_bulk - triangle_overcount + rectangle_correct  # #537 correction BEFORE the log
    # log10_magnitude keeps "no ionizing flux" (-inf) apart from "the SED was
    # corrupt" (+inf). The hand-rolled ``norm > 0`` here was False for NaN, so a
    # non-finite ionizing SED gave log_nion = -inf, pow10 -> 0, and nebular
    # emission silently switched off entirely — the #1001 fail-open class, in
    # the quantity Tier B introduced to avoid it (#1527).
    log10_norm = log10_magnitude(norm)
    offsets = jnp.log10(peak) - jnp.log10(H_PLANCK) + log10_scale
    # -inf + finite is -inf (true zero) and +inf + finite is +inf (corrupt), so
    # both sentinels survive the offset addition unchanged; only a +inf peak
    # could turn one into NaN, and that is itself corrupt.
    return jnp.where(_not_computable(log10_norm), jnp.inf, log10_norm + offsets)


def _integrate_nion(sed_lnu: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    r"""Ionizing photon rate :math:`Q_H` [photons/s] from a rest-frame L_nu SED.

    Thin wrapper around :func:`_integrate_nion_log10` that exponentiates the
    log-domain result. Linear photons/s contract for user-facing APIs. Zero
    ionizing flux returns exactly 0.0 (exp(-inf) == 0.0).

    Parameters
    ----------
    sed_lnu : ndarray, shape (n_wave,)
        Rest-frame stellar :math:`L_\nu` [erg/s/Hz] (pre-dust intrinsic SED).
    wave : ndarray, shape (n_wave,)
        Wavelength grid [Angstrom]; must span the Lyman limit (a few points
        above 911.76 A suffice — the boundary bin needs the first non-ionizing
        point).

    Returns
    -------
    ndarray, shape ()
        Ionizing photon rate [photons/s].

    Notes
    -----
    **JIT-compatible**: yes. Bit-exact whether given the full SSP grid or an
    ionizing-only slice (the non-ionizing region contributes zero to the mask).
    """
    return pow10(_integrate_nion_log10(sed_lnu, wave))


@dataclass(frozen=True)
class StellarSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for :class:`StellarSEDComponent`.

    Parameters
    ----------
    name : str
        Diagnostic identifier.
    sfh_model : str
        Registered SFH name. Currently supports ``"tsnorm"``, ``"dpl"``,
        ``"continuity"``, ``"dirichlet"``, ``"dense_basis"``, and several
        parametric/bursty variants.
    field : bool
        If ``True``, applies stochastic log-normal GP modulation to the mean SFH.
        Default ``False`` (no field).
    n_grid : int
        Lookback-time grid resolution for SFH evaluation and the published
        ``state.derived["sfh_grid_lbt_yr"]`` array.
    metallicity_model : str
        Metallicity evolution model. Currently supports ``"delta"`` (constant Z),
        ``"ramp"`` (linear Z(t)), ``"two_step"`` (step function), ``"bins"``
        (piecewise-constant per age bin), ``"table"`` (user-provided), and
        ``"chem_evol"`` (closed-box chemical evolution).
    sps_backend : str
        Stellar population synthesis backend. Currently supports ``"dsps"``
        (DSPS native triweight-MDF CSP integration).
    age_kernel : str or None
        How the SFH is integrated onto the SSP age grid — ``"cic"``, ``"dsps"``,
        or ``None`` (default) to auto-select: :data:`DEFAULT_AGE_KERNEL` on the
        non-field path, ``"dsps"`` on the GP-field path. ``"cic"`` evaluates the
        SFH on a
        :func:`_refine_sfh_table_ages` dense integrand (16x the SSP nodes) and
        splits each ``SFR(t)*dt`` parcel between its bracketing SSP nodes with
        log-age cloud-in-cell weights. ``"dsps"`` hands the coarse per-SSP-age
        table to DSPS's histogram kernel
        (:func:`~tengri.components.stellar.sps.dsps_wrapper.compute_dsps_age_weights`),
        which interpolates ``log10(M(<t))`` in ``log10(t)``.

        The two are NOT equivalent: the DSPS kernel annihilates the mass of any
        table segment straddling the SFH's maximum age, zeroing the first SSP
        node older than the SFH start (3.8 % of the total for a delayed-tau with
        age = 5 Gyr) and biasing the CSP +1.2 % in the optical with a blue-ward
        tilt vs FSPS / bagpipes / a dense reference (#964). ``"cic"`` is
        therefore the accuracy default; ``"dsps"`` is offered for cross-code
        comparison against DSPS-native pipelines and pre-#964 tengri.

        **How large the error is depends on the SSP age grid**, since what is
        lost is one node's share of the mass. The 3.8 % above is for the grid
        that measurement used; on the finer 93-node ProGeny/MILES grid the same
        delayed-tau at age = 5 Gyr relocates 0.64 %, and a double power law
        0.13-0.29 % (rising with ``age_gyr``), which moves ``ugriz`` photometry
        by 0.14-0.19 %. Re-measure on your own grid rather than quoting a
        number; the mechanism is grid-independent, the magnitude is not.

        **Pre-#964 equivalence is exact, verified against the pre-fix source**
        (parent of ``d5a78433b``): on the parametric delta path this branch runs
        the identical sequence — the same ``sfr_on_ssp`` (untouched by #964),
        ``_build_dsps_sfh_table(..., add_young_knot=True)`` (#538),
        ``calc_rest_sed_sfh_table_lognormal_mdf(...).weights``, the #821
        youngest-bin multiplier, then normalization. The one deliberate
        difference is that normalization now floors the divisor
        (``jnp.maximum(sum, 1e-300)``) so a degenerate all-zero SFH yields zero
        rather than NaN; on any non-degenerate input the result is unchanged.

        It is **not** a speed knob, and it is the slower of the two: measured
        end-to-end, ``"cic"`` is ~3.5 % faster on the exact path and ~13 %
        faster under ``WavePrecomp`` — DSPS compiles to about twice as many
        ``while`` loops, which precompute cannot shrink. (Do not judge this by
        timing :func:`compute_dsps_age_weights`; it has no call sites here.)

        Only consulted on the non-field path. A GP-field SFH always uses the
        DSPS kernel: the field draw is defined on its own coarse lookback grid,
        so there is no dense integrand to cloud-in-cell (#964). Asking for
        ``age_kernel="cic"`` together with ``field=True`` raises rather than
        silently returning DSPS weights.
    use_alpha_grid : bool
        Whether the SSP grid carries an α/Fe axis. Currently ``False``.
    lgmet_scatter : float
        Gaussian scatter in log10(Z) (dex) for the DSPS triweight kernel.
        Default 0.2 dex matches Prospector / DSPS convention.
    """

    name: str = "stellar"
    sfh_model: str = "tsnorm"
    field: bool = False
    n_grid: int = 256
    metallicity_model: str = "delta"
    sps_backend: str = "dsps"
    age_kernel: str | None = None
    field_centering: float = 1.0
    use_alpha_grid: bool = False
    lgmet_scatter: float = 0.2
    # Number of bins for ``metallicity_model="bins"`` /
    # ``"bins_continuity"``. Defaults to 6 to match
    # ``MET_REGISTRY``'s ``_N_MET_BINS_DEFAULT`` and the
    # ``met_bin_<i>`` / ``met_d_log_z_<i>`` parameter declarations.
    met_n_bins: int = 6
    # Bin edges in ``log10(age/yr)``, sorted ascending. Used by the
    # ``"bins"`` and ``"bins_continuity"`` metallicity modes.
    # ``None`` falls back to :data:`_DEFAULT_MET_BIN_EDGES_LOG_YR`
    # (log-spaced from 1 Myr to 13.7 Gyr).
    met_bin_edges_log_yr: Any = None
    # User-provided Z(t) table for ``metallicity_model="table"``.
    # ``met_table_log_age_yr`` is the table's age axis in log10(age/yr),
    # sorted ascending; ``met_table_log_z_abs`` is absolute log10(Z) at
    # each table age. Leave both None to instead supply the runtime
    # ``params["met_history"]`` array — log10(Z/Zsun) at the tabular
    # SFH's ``sfh_t_gyr`` nodes (requires ``sfh_model="table"``; #996).
    met_table_log_age_yr: Any = None
    met_table_log_z_abs: Any = None


@dataclass(frozen=True)
class StellarSEDComponentState(SEDComponentState):
    """Marker state. SSP tensors are held on the component instance.

    The ``precompute`` method returns an empty marker or (when wave_precomp
    is enabled) a state carrying the pre-computed SSP×filter LUT (fixed-z)
    or ztable (free-z). When ``approx=SpectrumPrecomp()`` is set, it instead
    carries the pre-rebinned SSP×pixel LUT (``ssp_spec_lut``, fixed-z) or
    its redshift table (``ssp_spec_ztable``, free-z) — the spectroscopic
    analog of the photometric LUT.
    """

    name: str = "stellar"
    ssp_phot_lut: Any | None = None
    ssp_phot_ztable: Any | None = None
    # The SSP grid preintegrated through each filter placed in the REST frame
    # (a :class:`RestBandPrecomputation`), i.e. at z=0 — what ``phot_rest_fnu``
    # actually needs (#1148). Redshift-independent, so ONE constant serves the
    # fixed-z LUT and the free-z z-table alike.
    restband_lut: Any | None = None
    # SpectrumPrecomp: SSP flux pre-rebinned to spectrum pixel
    # centers in the galaxy rest frame. ``ssp_spec_lut`` is a
    # :class:`SpectroscopicPrecomputation` (fixed-z); ``ssp_spec_ztable``
    # is a :class:`SpectroscopicZTable` (free-z).
    ssp_spec_lut: Any | None = None
    ssp_spec_ztable: Any | None = None
    # Zero-padded observed-frame filter curves (n_filters, max_len), carried so
    # ``apply`` can publish them to ``state.derived`` for the additive-emitter
    # exact-projection path (dust IR / radio / X-ray / AGN). Static; the same
    # for fixed-z and free-z (redshift is applied inside the integral).
    phot_fw_padded: Any | None = None
    phot_ft_padded: Any | None = None
    #: Number of SSP wavelength bins in the ionizing region (lambda < 2*911.76 A),
    #: a static structural constant of the fixed SSP grid. Computed at build time
    #: (concrete grid) and carried as static meta so :meth:`apply` can integrate
    #: Q_H over the ionizing slice ALONE, decoupling ``nion`` from the full-grid
    #: ``sed_intrinsic`` — that lets the WavePrecomp LUT path prune the full
    #: stellar SED einsum instead of forcing it just to publish Q_H (#950).
    n_ion_bins: int | None = None


@dataclass(frozen=True)
class StellarSEDComponent:
    """SEDComponent adapter for stellar emission.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX. The ``SSPData``
    NamedTuple registers as a JAX pytree, so ``self.ssp_data`` is a
    leaf-set of traced arrays under JIT.

    **Pipeline ordering**: stellar runs **first** in any chain. It
    writes ``state.sed_intrinsic`` from scratch and publishes the full
    set of stellar quantities other components consume.

    Construction
    ------------
    ``ssp_data`` is required at construction time. The component is a
    frozen dataclass; build it once at session start and reuse::

        ssp = load_ssp_data("data/ssp_miles.h5")
        stellar = StellarSEDComponent(ssp_data=ssp)
        result = run_components([stellar, ...], state, params)

    Cross-component publications (``state.derived``)
    ------------------------------------------------
    These keys are the stable contract every downstream component relies
    on.

    - ``log_mstar`` (scalar, dex) — log10(surviving stellar mass / Msun).
      Falls back to ``log_mstar_formed`` when the SSP grid lacks a
      ``ssp_mass_remaining`` table.
    - ``log_mstar_formed`` (scalar, dex) — log10(formed mass / Msun).
    - ``sfr`` (scalar, Msun/yr) — SFR at lookback ≈ 0 (i.e. the youngest
      grid point of ``sfr_history``).
    - ``sfr_10myr`` (scalar, Msun/yr) — time-weighted SFR over the last
      10 Myr of the SFH on the lookback grid.
    - ``sfr_100myr`` (scalar, Msun/yr) — same for 100 Myr.
    - ``L_age`` (ndarray, shape ``(n_age,)``, erg/s) — bolometric L per
      SSP age bin (∫ L_ν dν).
    - ``lnu_age`` (ndarray, shape ``(n_age, n_wave)``, erg/s/Hz) —
      per-age L_nu cube. Memory cost ~3 MB for n_age=140, n_wave=2700.
    - ``nion`` (scalar, photons/s) — ionizing photon production rate
      (∫_{λ<911.76 Å} L_ν / (hν) dν, total over all ages).
    - ``sfh_grid_lbt_yr`` (ndarray, shape ``(n_grid,)``, yr) — SFH
      lookback-time grid (log-spaced, 1e5 yr → AGEMAX_YR).
    - ``sfr_history`` (ndarray, shape ``(n_grid,)``, Msun/yr) — SFR on
      the SFH grid.
    - ``log_metallicity_history`` (ndarray, shape ``(n_grid,)``, dex) —
      per-time-bin metallicity (constant for ``metallicity_model="delta"``).
    - ``stellar_phot_lnu_precomp`` (ndarray, shape ``(n_filter,)``, erg/s/Hz) —
      stellar contribution to photometry from the LUT. Published only when
      ``approx=WavePrecomp()`` is set at model construction.

    """

    config: StellarSEDComponentConfig = field(default_factory=StellarSEDComponentConfig)
    ssp_data: SSPData | None = None
    name: str = "stellar"
    parameter_prefix: tuple[str, ...] = ("sfh_", "met_", "chem_")
    _state: StellarSEDComponentState | None = None

    def citations(self) -> tuple[str, ...]:
        """The stellar component is structurally built on DSPS; SFH-family
        and SSP-grid citations are config-driven via
        :mod:`tengri.citations.associations`."""
        return ("dsps",)

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns.

        Pulled from :data:`tengri.components.stellar.sfh.registry.SFH_REGISTRY`
        for the configured ``sfh_model`` plus a metallicity block keyed
        by ``metallicity_model``. Field parameters are added when
        ``config.field`` is ``True``.
        """
        # Lazy-import the registries so this module remains importable even
        # if the SFH registry temporarily fails to build.
        from tengri.components.stellar.sfh.met_registry import MET_REGISTRY
        from tengri.components.stellar.sfh.registry import SFH_REGISTRY

        if self.config.sfh_model not in SFH_REGISTRY:
            raise ValueError(
                f"sfh_model={self.config.sfh_model!r} not in SFH_REGISTRY. "
                f"Available: {list(SFH_REGISTRY.keys())}"
            )
        if self.config.metallicity_model not in MET_REGISTRY:
            raise ValueError(
                f"metallicity_model={self.config.metallicity_model!r} not in MET_REGISTRY. "
                f"Available: {list(MET_REGISTRY.keys())}"
            )

        decls: list[ParamDeclaration] = []
        sfh_spec = SFH_REGISTRY[self.config.sfh_model]
        for pname, pdef in sfh_spec.params.items():
            decls.append(ParamDeclaration(pname, pdef.default, pdef.description))

        met_spec = MET_REGISTRY[self.config.metallicity_model]
        for pname, pdef in getattr(met_spec, "params", {}).items():
            decls.append(ParamDeclaration(pname, pdef.default, pdef.description))

        if self.config.field:
            field_spec = SFH_REGISTRY.get("field")
            if field_spec is not None:
                for pname, pdef in field_spec.params.items():
                    decls.append(ParamDeclaration(pname, pdef.default, pdef.description))

        return decls

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this stellar component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.
        """
        return (
            DerivedKey("log_mstar", "dex", "log10(surviving stellar mass / Msun)"),
            DerivedKey(
                "log_mstar_surviving",
                "dex",
                "log10(surviving stellar mass / Msun); NaN when the SSP grid has no "
                "mass-remaining table (unlike log_mstar, this never falls back to the "
                "formed mass)",
            ),
            DerivedKey("log_mstar_formed", "dex", "log10(formed stellar mass / Msun)"),
            DerivedKey("sfr", "Msun/yr", "SFR at lookback ~ 0"),
            DerivedKey("sfr_10myr", "Msun/yr", "Time-weighted SFR over last 10 Myr"),
            DerivedKey("sfr_100myr", "Msun/yr", "Time-weighted SFR over last 100 Myr"),
            DerivedKey("L_age", "erg/s", "Bolometric L per SSP age bin"),
            DerivedKey("log_L_age", "dex", "log10(L per SSP age bin / (erg/s)); float32-safe"),
            DerivedKey("lnu_age", "erg/s/Hz", "Per-age L_nu cube, shape (n_age, n_wave)"),
            DerivedKey(
                "joint_weights",
                "",
                "DSPS joint (metallicity, age) weights, shape (n_met, n_age)",
            ),
            DerivedKey(
                "stellar_mass_scale",
                "erg/s/Hz",
                "total_mass x L_sun: scales SSP per-Msun luminosities to erg/s/Hz",
            ),
            DerivedKey(
                "log_stellar_mass_scale",
                "dex",
                "log10(total_mass x L_sun): the float32-safe form of "
                "stellar_mass_scale, which is ~1e43 and so overflows float32",
            ),
            DerivedKey("ssp_ages_yr", "yr", "SSP age axis"),
            DerivedKey("age_weights", "Msun", "CSP mass weights per SSP age bin"),
            DerivedKey("nion", "photons/s", "Ionizing photon rate (lambda < 911.76 A)"),
            DerivedKey(
                "log_nion",
                "dex",
                "log10(ionizing photon rate / (photons/s)); lambda < 911.76 A",
            ),
            DerivedKey("sfh_grid_lbt_yr", "yr", "SFH lookback-time grid"),
            DerivedKey("sfr_history", "Msun/yr", "SFR on SFH grid"),
            DerivedKey("log_metallicity_history", "dex", "log10(Z) per SFH time bin"),
            DerivedKey(
                "stellar_phot_lnu_precomp",
                "erg/s/Hz",
                "stellar contribution to photometry from LUT (approx.wave_precomp only)",
            ),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
        redshift_spec: dict[str, Any] | None = None,
        spec_wave_obs: jnp.ndarray | None = None,
    ) -> StellarSEDComponentState:
        """Build SSP×filter LUT (WavePrecomp) or SSP×pixel LUT (SpectrumPrecomp).

        Reads the ``wave_precomp`` / ``spectrum_precomp`` flags from
        ``approx``. For ``wave_precomp`` (with ``filters``), calls
        :func:`precompute_photometry` (fixed-z) or
        :func:`precompute_photometry_ztable` (free-z). For
        ``spectrum_precomp`` (with ``spec_wave_obs``), calls
        :func:`precompute_spectroscopy` (fixed-z) or
        :func:`precompute_spectroscopy_ztable` (free-z), pre-rebinning the
        SSP grid to the spectrum pixel centers in the galaxy rest frame.
        Otherwise returns an empty state marker.

        Parameters
        ----------
        approx : Mapping[str, bool] | None
            Approximation flags. Reads ``"wave_precomp"`` and
            ``"spectrum_precomp"``.
        filters : tuple of (filter_wave_obs, filter_trans) pairs, optional
            Required when ``wave_precomp=True``. Tuple shape:
            ((fw_0, ft_0), (fw_1, ft_1), ...) where each pair is a pair of
            1-D arrays. The filter_wave is observed-frame.
        redshift_spec : dict[str, Any] | None
            Redshift specification for precomputation. If None or
            mode="fixed", builds a fixed-z LUT.

            - mode="fixed", value=float: builds LUT at that fixed z.
            - mode="free", z_min=float, z_max=float, n_z=int: builds
              ztable via precompute_photometry_ztable with the given grid.

        spec_wave_obs : array_like, shape (n_pix,), optional
            Observed-frame spectrum pixel wavelengths [Angstrom]. Required
            when ``spectrum_precomp=True``.
        """
        del wave_grid
        approx = approx or {}

        from dataclasses import replace as _replace_state

        state = StellarSEDComponentState(name=self.name)

        # Static ionizing-bin count from the concrete build-time SSP grid, so
        # ``apply`` can compute Q_H over the ionizing slice alone (see the field
        # docstring). The grid is ascending, so lambda < 2*911.76 A is a prefix.
        _ssp_for_nion = ssp_data if ssp_data is not None else self.ssp_data
        if _ssp_for_nion is not None:
            import numpy as _np

            _wave = _np.asarray(_ssp_for_nion.ssp_wave)
            n_ion = int(_np.count_nonzero(_wave < 2.0 * _HI_LIMIT_AA))
            state = _replace_state(state, n_ion_bins=n_ion)

        # SpectrumPrecomp — pre-rebin SSP to spectrum pixel centers.
        # Part A (joint): build the spectrum LUT *alongside* the photometry LUT
        # below (not an early return) so a joint photometry+spectroscopy model
        # carries BOTH families in one state. ``_precompute_spectrum`` populates
        # ``ssp_spec_lut`` (fixed-z) or ``ssp_spec_ztable`` (free-z).
        if approx.get("spectrum_precomp") and spec_wave_obs is not None:
            spec_state = self._precompute_spectrum(spec_wave_obs, redshift_spec)
            state = _replace_state(
                state,
                ssp_spec_lut=spec_state.ssp_spec_lut,
                ssp_spec_ztable=spec_state.ssp_spec_ztable,
            )

        # WavePrecomp: photometry SSP×filter LUT. Requires filters + SSP grid.
        if approx.get("wave_precomp") and filters is not None and self.ssp_data is not None:
            from tengri.observation.photometry import pad_filters

            filter_waves, filter_trans = zip(*filters, strict=False)
            filter_list = [jnp.asarray(fw) for fw in filter_waves]
            filter_trans_list = [jnp.asarray(ft) for ft in filter_trans]

            # Padded observed-frame filter curves for the additive-emitter exact
            # projection (published to state.derived in ``apply``). Static and
            # shared by the fixed-z and free-z LUT paths.
            fw_pad, ft_pad, _ = pad_filters(filter_list, filter_trans_list)
            state = _replace_state(state, phot_fw_padded=fw_pad, phot_ft_padded=ft_pad)

            # Dispatch: fixed-z LUT or free-z ztable
            if redshift_spec is None or redshift_spec.get("mode") == "fixed":
                # Build the fixed-z LUT at the source's z so the
                # filter passband is correctly redshifted into the rest frame.
                # Cosmology ``(1+z)/(4π·dl²)`` is applied in
                # :meth:`Observation.predict_via_precomp`.
                from tengri.components.stellar.sps.precompute import precompute_photometry

                z_source = redshift_spec.get("value", 0.0) if redshift_spec else 0.0
                lut = precompute_photometry(
                    ssp_data=self.ssp_data,
                    filter_waves=filter_list,
                    filter_trans=filter_trans_list,
                    redshift=z_source,
                    dl_cm=1.0,  # placeholder; cosmology applied at projection time
                    # Ψ moment for the dust-attenuation Taylor correction (#617),
                    # toggled by approx=WavePrecomp(taylor_correction=...).
                    taylor_correction=approx.get("taylor_correction", False),
                    # Sub-band quadrature for the dust screen (#1122) — supersedes Ψ.
                    n_subbands=int(approx.get("n_subbands", 0)),
                )
                state = _replace_state(state, ssp_phot_lut=lut)
            else:  # mode == "free"
                # Free-z path: build ztable for redshift interpolation.
                from tengri.components.stellar.sps.precompute import (
                    precompute_photometry_ztable,
                )

                ztable = precompute_photometry_ztable(
                    ssp_data=self.ssp_data,
                    filter_waves=filter_list,
                    filter_trans=filter_trans_list,
                    z_min=redshift_spec.get("z_min", 0.001),
                    z_max=redshift_spec.get("z_max", 3.0),
                    n_z=redshift_spec.get("n_z", 100),
                    apply_igm=False,
                    # Ψ moment for the dust-attenuation Taylor correction (#617),
                    # toggled by approx=WavePrecomp(taylor_correction=...).
                    taylor_correction=approx.get("taylor_correction", False),
                    # Sub-band quadrature for the dust screen (#1122) — supersedes Ψ.
                    n_subbands=int(approx.get("n_subbands", 0)),
                )
                state = _replace_state(state, ssp_phot_ztable=ztable)

            # The REST-frame band (#1148), built ONCE for both dispatches above.
            # ``phot_rest_fnu`` is the SED reprojected at z=0, d_L=10 pc, so the
            # filter sits in the REST frame and samples the rest SED at its own
            # pivot — a different integral from ``ssp_phot``, which places the
            # filter in the observed frame and samples rest λ_eff/(1+z). Reusing
            # the observed-band tensor for the rest-frame flux is #1148: it put the
            # LUT 769 % from the exact path in des_g at z=0.5.
            #
            # Redshift does not enter, so this is one constant for fixed-z AND
            # free-z — no z-table, no interpolation, no runtime cost.
            from tengri.components.stellar.sps.precompute import (
                precompute_restband_photometry,
            )

            state = _replace_state(
                state,
                restband_lut=precompute_restband_photometry(
                    ssp_data=self.ssp_data,
                    filter_waves=filter_list,
                    filter_trans=filter_trans_list,
                    n_subbands=int(approx.get("n_subbands", 0)),
                ),
            )

        return state

    def _precompute_spectrum(
        self,
        spec_wave_obs: jnp.ndarray | None,
        redshift_spec: dict[str, Any] | None,
    ) -> StellarSEDComponentState:
        """Build the SSP×pixel LUT for ``approx=SpectrumPrecomp()``.

        Pre-rebins the SSP flux cube to the spectrum pixel centers in the
        galaxy rest frame. Unlike the photometric LUT, **no Taylor moment
        is needed**: a spectrum pixel is a single wavelength, so dust
        attenuation ``A(λ_pix)`` evaluated at the pixel center is exact —
        there is no wide-kernel integral to factorize.

        Fixed-z builds a single :class:`SpectroscopicPrecomputation`; free-z
        builds a :class:`SpectroscopicZTable` so the rest-frame pixel grid
        ``wave_obs / (1 + z)`` can be interpolated at runtime.
        """
        if spec_wave_obs is None or self.ssp_data is None:
            # No grid or no SSP — fall back to the full-grid path.
            return StellarSEDComponentState(name=self.name)

        spec_wave_obs = jnp.asarray(spec_wave_obs)

        if redshift_spec is None or redshift_spec.get("mode") == "fixed":
            from tengri.components.stellar.sps.precompute import precompute_spectroscopy

            z_source = redshift_spec.get("value", 0.0) if redshift_spec else 0.0
            lut = precompute_spectroscopy(
                ssp_data=self.ssp_data,
                wave_obs_pixels=spec_wave_obs,
                redshift=z_source,
                dl_cm=1.0,  # placeholder; cosmology applied at projection time
            )
            return StellarSEDComponentState(name=self.name, ssp_spec_lut=lut)

        # mode == "free": build the redshift table.
        from tengri.components.stellar.sps.precompute import precompute_spectroscopy_ztable

        ztable = precompute_spectroscopy_ztable(
            ssp_data=self.ssp_data,
            wave_obs_pixels=spec_wave_obs,
            z_min=redshift_spec.get("z_min", 0.001),
            z_max=redshift_spec.get("z_max", 3.0),
            n_z=redshift_spec.get("n_z", 100),
        )
        return StellarSEDComponentState(name=self.name, ssp_spec_ztable=ztable)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        """Compute stellar SED and publish derived quantities.

        Assembles the composite stellar population by convolving the
        star formation history with SSP templates via DSPS. Publishes
        stellar mass, ionizing photon rate, and age-dependent quantities
        for downstream components.

        Parameters
        ----------
        state : ForwardState
            Initial pipeline state. Carries ``wave`` (rest-frame Å); the
            component reads ``redshift`` from ``params`` (allowlist).
        params : mapping
            Receives ``sfh_*``, ``met_*``, ``chem_*`` keys plus the bare
            ``redshift`` from :data:`BARE_NAME_ALLOWLIST`.
        ssp_data : Any | None, optional
            SSP data passed as a JIT runtime input.
            When provided, uses this instead of ``self.ssp_data``. Enables
            SSP arrays to be ``Parameter`` ops in compiled code rather than
            ``Constant`` ops, reducing HLO size and compile time.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` set and 13 derived keys
            published.
        """
        # Use ssp_data if threaded as JIT input, otherwise fall
        # back to the closure (for non-JIT paths).
        ssp = ssp_data if ssp_data is not None else self.ssp_data
        if ssp is None:
            raise ValueError(
                "StellarSEDComponent.apply requires ssp_data set on the component. "
                "Pass it at construction: StellarSEDComponent(ssp_data=ssp)."
            )
        # SFH models are routed through SFH_REGISTRY's internal_param_map.
        # Each model is validated against legacy DSPS path via
        # tests/integration/test_stellar_integration.py.
        _SUPPORTED_SFH = (
            "tsnorm",
            "dpl",
            "continuity",
            "dirichlet",
            "dense_basis",
            "lnorm",
            "snorm",
            "snorm_burst",
            "tsnorm_burst",
            "norm",
            "const",
            "const_exp",
            "continuity_flex",
            "psb",
            "psb_suess2022",
            "delayed_bq",
            "dense_basis_pure",
            "exp",
            "dexp",
            "tau",
            "delayed",
            "periodic",
            "sfh2exp",
            "buat08",
            "table",
        )
        if self.config.sfh_model not in _SUPPORTED_SFH:
            raise NotImplementedError(
                f"sfh_model={self.config.sfh_model!r} not yet validated "
                f"against legacy DSPS. Supported modes: {_SUPPORTED_SFH}."
            )
        _SUPPORTED_MET = (
            "delta",
            "ramp",
            "chem_evol",
            "two_step",
            "psb_two_step",
            "bins",
            "bins_continuity",
            "table",
            "massmap_lin",
            "massmap_box",
        )
        if self.config.metallicity_model not in _SUPPORTED_MET:
            raise NotImplementedError(
                f"metallicity_model={self.config.metallicity_model!r} not in "
                f"{_SUPPORTED_MET}. Add a branch in StellarSEDComponent.apply() "
                f"per docs/dev/20260506-met-mode-wiring-blueprint.md."
            )
        ssp_ages_yr = (10.0**ssp.ssp_lg_age_gyr) * 1e9
        n_grid = self.config.n_grid

        # ── 1. SFH lookback-time grid ───────────────────────────────────
        # Use the SAME grid construction as the legacy SEDModel path
        # (forward/sed_model.py:467). ``make_log_age_grid`` returns a
        # uniform grid in log10(age/yr) over [6.0, 10.14] (1 Myr →
        # 13.8 Gyr). This is critical for ``field=True`` parity:
        # ``compute_field_gp`` keys on n_grid + d_log_age to build
        # the GP correlation kernel, so both paths must construct
        # the grid identically. See tests/integration/test_stellar_integration.py.
        log_age_grid = make_log_age_grid(n_grid)
        sfh_lbt_grid = 10.0**log_age_grid

        # ── 1b. Cosmology: t_obs from redshift (hoisted above the SFH so
        # the runtime tabular SFH can convert cosmic time → lookback).
        # ``age_at_z`` is JIT-compatible (pure JAX under the hood).
        from tengri.cosmology import age_at_z as _age_at_z

        z = jnp.asarray(require_redshift(params, "components.stellar.component.apply"))
        t_obs_gyr = jnp.asarray(_age_at_z(z)).reshape(())

        # ── 2. Evaluate mean SFH on grid (registry-driven) ──────────────
        # Translate user-facing public params → SFH-function kwargs via
        # the registry's ``internal_param_map``: each entry is
        # ``(internal_name, scale, offset)`` and the conversion is
        # ``internal = public * scale + offset``. This ensures both this
        # component and legacy SEDModel paths see the same units and naming.
        from tengri.components.stellar.sfh.registry import SFH_REGISTRY

        sfh_spec = SFH_REGISTRY[self.config.sfh_model]
        sfh_kwargs = {}
        for public_name, (internal_name, scale, offset) in sfh_spec.internal_param_map.items():
            if public_name in params:
                raw = params[public_name]
            else:
                # Fall back to the registry default for any declared parameter
                # the caller omitted. Required so callers that pass a partial
                # params dict (e.g. the low-level run_components path, or flat-
                # kwarg specs predating the dpl/lnorm ``age`` anchor of #514)
                # still resolve every positional argument of the SFH callable.
                # Mirrors the dense_basis age_universe injection just below.
                pdef = sfh_spec.params.get(public_name)
                default_scalar = pdef.default.default if pdef is not None else None
                if default_scalar is None:
                    continue
                raw = default_scalar
            sfh_kwargs[internal_name] = jnp.asarray(raw) * scale + offset

        # Mode-specific settings that are NOT free parameters.
        # ``dense_basis`` needs an explicit ``age_universe_yr`` derived
        # from the configured cosmology; default of 13.47 Gyr matches
        # the registry setting (FlatLambdaCDM, H0=70, Omega_m=0.3, z=0).
        if self.config.sfh_model == "dense_basis":
            age_universe_gyr = sfh_spec.settings.get("sfh_db_age_universe_gyr", 13.47)
            sfh_kwargs["age_universe_yr"] = float(age_universe_gyr) * 1e9

        # ── 2a′. Runtime tabular SFH (sfh_model="table", #996) ──────────
        # Simulation SFHs arrive as runtime arrays: ``params["sfh_t_gyr"]``
        # (cosmic time [Gyr]) + ``params["sfh_sfr"]`` [Msun/yr]. The registry
        # entry is a placeholder; override it with an interp closure so every
        # consumer (coarse history, sfr_on_ssp, dense CIC integrand, derived
        # quantities) sees the table through the one SFH function. Dict-key
        # presence is static under jit; the array VALUES stay traced, so two
        # calls with different same-length tables share one compile. SFR is
        # edge-clamped outside the table (the legacy jnp.interp convention);
        # lookbacks older than t_obs are dropped by the CIC t_obs cutoff.
        sfh_fn = sfh_spec.fn
        _tab_lbt_yr = None
        _tab_order = None
        if self.config.sfh_model == "table":
            if self.config.field:
                raise NotImplementedError(
                    "sfh_model='table' with field=True is not supported — the "
                    "GP field draw modulates parametric SFHs only (#996)."
                )
            # Shared with compute_joint_weights so the exact forward and the
            # SED-free fast path cannot read a simulation history differently
            # (#1395/#1396).
            sfh_fn, _tab_lbt_yr, _tab_order = _tabulated_sfh(params, t_obs_gyr)
        elif "sfh_t_gyr" in params or "sfh_sfr" in params:
            raise NotImplementedError(
                "sfh_t_gyr/sfh_sfr passed but sfh_model="
                f"{self.config.sfh_model!r} — the table would be silently "
                "ignored. Build with mean_sfh_type='table' (#996)."
            )
        if "met_history" in params and self.config.metallicity_model != "table":
            raise NotImplementedError(
                "met_history passed but metallicity_model="
                f"{self.config.metallicity_model!r} — it would be silently "
                "ignored. Build with met_mode='table' (#996)."
            )

        sfr_history = sfh_fn(sfh_lbt_grid, **sfh_kwargs)

        # ── 2b. GP-field modulation ───────────────────────────────────
        # Multiplicative log-normal modulation: SFR_total = SFR_mean ×
        # exp(x(t) - K(0)/2), where x(t) is a PSD-governed Gaussian
        # process and K(0)/2 is the lognormal bias correction so the
        # ensemble mean equals SFR_mean. ``compute_field_gp`` lives in
        # the SFH registry next to the prior on ``sfh_field_xi``.
        if self.config.field:
            sfr_history = _apply_gp_field(
                sfr_history, params, n_grid, log_age_grid, self.config.field_centering
            )

        # ── 3. Resample to SSP age grid for CSP integration ─────────────
        # For deterministic (non-GP) parametric SFHs, evaluate the analytic
        # shape on ``ssp_ages_yr`` directly. The SSP age grid is linear-spaced
        # at 1 Myr cadence (13700 bins over 1 Myr → 13.7 Gyr), so SF-onset
        # cutoffs at ``t_lookback = age`` land on grid points instead of
        # being smeared across the coarser ~3 % log-spaced bins of
        # ``sfh_lbt_grid``. The closed form also self-normalizes through
        # ``_renormalize_to_mass``, so total mass formed = ``10**log_total_mass``
        # exactly. See suchethac/tengri#385.
        #
        # The GP-field path still goes through the log grid: ``compute_field_gp``
        # builds its DRW kernel keyed on ``n_grid`` and ``d_log_age``, so the
        # GP draw lives on the lookback grid by construction.
        if self.config.field:
            sfr_on_ssp = jnp.interp(ssp_ages_yr, sfh_lbt_grid, sfr_history)
        else:
            sfr_on_ssp = sfh_fn(ssp_ages_yr, **sfh_kwargs)

        # (t_obs_gyr hoisted to section 1b — needed by the tabular SFH.)

        # ── 4. Metallicity history Z(t) on SFH grid + per-SSP-age ───────
        # delta: scalar absolute log10(Z), constant in time.
        # ramp: linear interpolation between two endpoints.
        # chem_evol: closed-box gas regulator — Z(t) derived from SFH self-
        # consistently. Mirrors legacy sed_model.py:3578-3592.
        # 4D α-enhanced SSPs: collapse the [α/Fe] axis to a single
        # plane once, here, then pass the resulting 3D ssp_flux to the
        # downstream DSPS kernel (closes #226). The Z marginalization
        # remains the standard lognormal MDF for every met_mode, so the
        # 4D and 3D paths share the same Z bookkeeping — only the
        # ``ssp_flux`` that DSPS sees differs.
        _alpha_collapse_active = has_alpha_grid(ssp)
        if _alpha_collapse_active:
            _alpha_fe_value = jnp.asarray(params.get("met_alpha_fe", 0.0))
            ssp_flux_for_csp = interpolate_alpha_only(
                ssp.ssp_flux, ssp.ssp_alpha_fe, _alpha_fe_value
            )
        else:
            ssp_flux_for_csp = ssp.ssp_flux

        if self.config.metallicity_model == "delta":
            # Apply alpha-Fe enhancement via effective_metallicity for
            # 3D SSP grids (no native α axis): the α-shift is folded
            # into log_z via the Salaris+05 / DSPS canonical relation.
            # For 4D α-grid SSPs the α axis has already been collapsed
            # above, so we use ``met_logzsol`` directly without the
            # effective-Z approximation.
            alpha_fe = jnp.asarray(params.get("met_alpha_fe", 0.0))
            if _alpha_collapse_active:
                log_z_abs_scalar = jnp.asarray(params["met_logzsol"]) + LOG10_ZSUN
            else:
                log_z_eff = effective_metallicity(jnp.asarray(params["met_logzsol"]), alpha_fe)
                log_z_abs_scalar = log_z_eff + LOG10_ZSUN
            log_metallicity_history = jnp.full(n_grid, log_z_abs_scalar)
            lgmet_on_ssp_ages = jnp.full_like(ssp_ages_yr, log_z_abs_scalar)
            log_z_for_mr = log_z_abs_scalar
        elif self.config.metallicity_model == "ramp":
            log_z_init_abs = jnp.asarray(params["met_logzsol_0"]) + LOG10_ZSUN
            log_z_final_abs = jnp.asarray(params["met_logzsol_final"]) + LOG10_ZSUN
            # Build the per-age metallicity ramp on both grids (SFH grid for
            # diagnostics + SSP grid for the CSP integral).
            sfh_lg_age_gyr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0)) - 9.0
            log_metallicity_history = compute_log_z_evolving(
                sfh_lg_age_gyr, log_z_init_abs, log_z_final_abs, t_obs_gyr
            )
            lgmet_on_ssp_ages = compute_log_z_evolving(
                ssp.ssp_lg_age_gyr, log_z_init_abs, log_z_final_abs, t_obs_gyr
            )
            # For mass-remaining interpolation use the present-day metallicity
            # (newest stars dominate the mass-loss correction).
            log_z_for_mr = log_z_final_abs
        elif self.config.metallicity_model == "two_step":
            # Sigmoid-smoothed step at ``met_step_age_gyr``. Stars older than
            # the step get ``met_logzsol_old``, younger get ``met_logzsol_young``.
            log_z_old_abs = jnp.asarray(params["met_logzsol_old"]) + LOG10_ZSUN
            log_z_young_abs = jnp.asarray(params["met_logzsol_young"]) + LOG10_ZSUN
            step_age_gyr = jnp.asarray(params["met_step_age_gyr"])
            lgmet_on_ssp_ages = two_step_metallicity(
                ssp.ssp_lg_age_gyr, log_z_old_abs, log_z_young_abs, step_age_gyr
            )
            sfh_lg_age_gyr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0)) - 9.0
            log_metallicity_history = two_step_metallicity(
                sfh_lg_age_gyr, log_z_old_abs, log_z_young_abs, step_age_gyr
            )
            # Present-day Z (youngest SSP age, lookback ≈ 0).
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "psb_two_step":
            # Step tied to the PSB SFH burst onset
            # (``sfh_psb_burstage_gyr``). Pre-burst stars get
            # ``met_logzsol_old``, burst-and-younger get
            # ``met_logzsol_burst``.
            log_z_old_abs = jnp.asarray(params["met_logzsol_old"]) + LOG10_ZSUN
            log_z_burst_abs = jnp.asarray(params["met_logzsol_burst"]) + LOG10_ZSUN
            burstage_gyr = jnp.asarray(params.get("sfh_psb_burstage_gyr", 1.0))
            lgmet_on_ssp_ages = psb_two_step_metallicity(
                ssp.ssp_lg_age_gyr, log_z_old_abs, log_z_burst_abs, burstage_gyr
            )
            sfh_lg_age_gyr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0)) - 9.0
            log_metallicity_history = psb_two_step_metallicity(
                sfh_lg_age_gyr, log_z_old_abs, log_z_burst_abs, burstage_gyr
            )
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "bins":
            # Piecewise-constant Z per age bin. Bin edges from config
            # (defaults to log-spaced 1 Myr → 13.7 Gyr); per-bin
            # metallicities from ``met_bin_<i>`` params (i = 0..N-1).
            n_bins = self.config.met_n_bins
            bin_edges_log_yr = (
                self.config.met_bin_edges_log_yr
                if self.config.met_bin_edges_log_yr is not None
                else _DEFAULT_MET_BIN_EDGES_LOG_YR
            )
            metallicities_abs = (
                jnp.stack([jnp.asarray(params[f"met_bin_{i}"]) for i in range(n_bins)])
                + LOG10_ZSUN
            )
            lgmet_on_ssp_ages = metallicity_bins_on_ssp_grid(
                ssp.ssp_lg_age_gyr, jnp.asarray(bin_edges_log_yr), metallicities_abs
            )
            # SFH-grid history: same primitive applied to sfh_lbt_grid.
            sfh_lg_age_yr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0))
            log_metallicity_history = metallicity_bins_on_ssp_grid(
                sfh_lg_age_yr - 9.0, jnp.asarray(bin_edges_log_yr), metallicities_abs
            )
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "bins_continuity":
            # Cumulative delta-log-Z steps from oldest bin to youngest.
            # ``met_logzsol_base`` is the oldest bin; ``met_d_log_z_<i>``
            # are the N-1 steps. Reuses the binning primitive with
            # convolved metallicities.
            n_bins = self.config.met_n_bins
            bin_edges_log_yr = (
                self.config.met_bin_edges_log_yr
                if self.config.met_bin_edges_log_yr is not None
                else _DEFAULT_MET_BIN_EDGES_LOG_YR
            )
            log_z_base_abs = jnp.asarray(params["met_logzsol_base"]) + LOG10_ZSUN
            d_log_z = jnp.stack(
                [jnp.asarray(params[f"met_d_log_z_{i}"]) for i in range(n_bins - 1)]
            )
            lgmet_on_ssp_ages = metallicity_bins_continuity_on_ssp_grid(
                ssp.ssp_lg_age_gyr, jnp.asarray(bin_edges_log_yr), log_z_base_abs, d_log_z
            )
            sfh_lg_age_yr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0))
            log_metallicity_history = metallicity_bins_continuity_on_ssp_grid(
                sfh_lg_age_yr - 9.0,
                jnp.asarray(bin_edges_log_yr),
                log_z_base_abs,
                d_log_z,
            )
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "table":
            # Z(t) table from either (a) constructor-time config arrays, or
            # (b) the runtime ``met_history`` param — log10(Z/Zsun) at the
            # ``sfh_t_gyr`` nodes, the legacy simulation interface (#996).
            # (b) needs sfh_model='table' to supply the time axis.
            # Shared with compute_joint_weights (#1396) — see _tabulated_sfh
            # for why both routes must resolve the table through one function.
            lgmet_on_ssp_ages, met_log_age_yr, met_log_z_abs = _tabulated_lgmet_on_ssp_ages(
                params, self.config, ssp.ssp_lg_age_gyr, _tab_lbt_yr, _tab_order
            )
            sfh_lg_age_yr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0))
            log_metallicity_history = tabulated_metallicity_on_ssp_grid(
                sfh_lg_age_yr - 9.0, met_log_age_yr, met_log_z_abs
            )
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "massmap_lin":
            # Linear metallicity tied to cumulative stellar mass formed
            # (ProSpect Bellstedt+2020 massmap_lin model).
            log_z_start_abs = jnp.asarray(params["met_logzsol_start"]) + LOG10_ZSUN
            log_z_final_abs = jnp.asarray(params["met_logzsol_final"]) + LOG10_ZSUN
            # Per-age metallicity on the SSP grid
            lgmet_on_ssp_ages = massmap_lin_metallicity(
                ssp.ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start_abs, log_z_final_abs
            )
            # Z(t) on the SFH grid for diagnostics
            sfh_lg_age_gyr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0)) - 9.0
            log_metallicity_history = massmap_lin_metallicity(
                sfh_lg_age_gyr, sfh_lbt_grid, sfr_history, log_z_start_abs, log_z_final_abs
            )
            # Mass-remaining interpolation: use present-day Z (youngest SSP age).
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "massmap_box":
            # Closed-box chemical evolution tied to cumulative stellar mass formed
            # (ProSpect Bellstedt+2020 massmap_box model).
            log_z_start_abs = jnp.asarray(params["met_logzsol_start"]) + LOG10_ZSUN
            log_z_final_abs = jnp.asarray(params["met_logzsol_final"]) + LOG10_ZSUN
            yield_rho = jnp.asarray(params.get("met_yield", 0.03))
            # Per-age metallicity on the SSP grid
            lgmet_on_ssp_ages = massmap_box_metallicity(
                ssp.ssp_lg_age_gyr,
                ssp_ages_yr,
                sfr_on_ssp,
                log_z_start_abs,
                log_z_final_abs,
                yield_rho,
            )
            # Z(t) on the SFH grid for diagnostics
            sfh_lg_age_gyr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0)) - 9.0
            log_metallicity_history = massmap_box_metallicity(
                sfh_lg_age_gyr,
                sfh_lbt_grid,
                sfr_history,
                log_z_start_abs,
                log_z_final_abs,
                yield_rho,
            )
            # Mass-remaining interpolation: use present-day Z (youngest SSP age).
            log_z_for_mr = lgmet_on_ssp_ages[0]
        else:  # chem_evol
            from tengri.components.stellar.sfh.chemical_evolution import (
                chem_evol_metallicity_on_ssp_grid,
                closed_box_metallicity,
            )

            # jnp.asarray, not float(): traced under jit (ConcretizationTypeError).
            yield_y = jnp.asarray(params.get("chem_yield", 0.03))
            eta_outflow = jnp.asarray(params.get("chem_eta_outflow", 0.0))
            f_gas_init = jnp.asarray(params.get("chem_f_gas_init", 0.9))
            return_frac = jnp.asarray(params.get("chem_return_frac", 0.4))

            # Per-age metallicity on the SSP grid, in log10(age/yr) on both
            # grids; the SSP grid is ssp.ssp_lg_age_gyr + 9.0.
            ssp_log_ages_yr = ssp.ssp_lg_age_gyr + 9.0
            lgmet_on_ssp_ages = chem_evol_metallicity_on_ssp_grid(
                ssp_log_ages_yr,
                log_age_grid,
                sfr_history,
                yield_y=yield_y,
                eta_outflow=eta_outflow,
                f_gas_init=f_gas_init,
                return_frac=return_frac,
            )
            # Z(t) on the SFH grid for diagnostics — closed_box_metallicity
            # returns log10(Z/Zsun); add LOG10_ZSUN for absolute log10(Z).
            log_metallicity_history = (
                closed_box_metallicity(
                    sfh_lbt_grid,
                    sfr_history,
                    yield_y=yield_y,
                    eta_outflow=eta_outflow,
                    f_gas_init=f_gas_init,
                    return_frac=return_frac,
                )
                + LOG10_ZSUN
            )
            # Mass-remaining interpolation: use present-day Z (youngest SSP age).
            log_z_for_mr = lgmet_on_ssp_ages[0]

        # ── 6. CSP integral via DSPS ────────────────────────────────────
        # We call DSPS directly and use ``result.weights`` — the JOINT
        # (n_met, n_age) probability distribution — instead of the
        # separable approximation in compute_dsps_native_weights. The
        # separable form (lgmet_w × age_w) gave the right marginals but
        # the wrong product for non-trivial age-metallicity correlations,
        # over-scaling the CSP SED by orders of magnitude.
        from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_lognormal_mdf

        # NaN-safe cosmic-time prep mirroring
        # :func:`compute_dsps_age_weights`: when SSP ages exceed
        # ``t_obs`` (typical at z>0 with old SSPs), the implied cosmic
        # time is negative. Bare ``jnp.clip(min=1e-3)`` collapses
        # multiple such bins to the same boundary value, producing a
        # degenerate ``gal_t_table`` that DSPS NaNs on. Instead, we
        # build a strictly-monotonic ramp at the invalid end and zero
        # the SFR there so those bins contribute nothing.
        ssp_age_gyr = ssp_ages_yr / 1e9
        # Coarse (per-SSP-age) total formed mass — the conserved normalization
        # basis (without the young-boundary knot) shared by every DSPS path
        # below. Each path rebuilds its own (t, SFR) table: the non-parametric
        # delta path with a dense integrand (#758), and both the parametric
        # delta and per-age-metallicity paths with the young-boundary knot
        # (#538). The knot's [0, age0] segment is excluded from this total, so it
        # redistributes mass into the youngest bin without inflating it.
        _, _, total_mass = _build_dsps_sfh_table(ssp_ages_yr, sfr_on_ssp, t_obs_gyr)

        # Eager physicality guard: the masking above truncates any SFH mass at
        # lookback ages older than the universe at this redshift. When that
        # truncated fraction is non-negligible the prediction no longer matches
        # the requested SFH, so warn on the eager forward path. The ``float()``
        # conversions raise ConcretizationTypeError under *any* jax transform
        # (jit / grad / vmap, including the partial tracing of a population vmap
        # where ``redshift`` is concrete but the SFH params are batched), so we
        # catch that and skip silently — exploring such draws during inference is
        # expected and there is no concrete value to warn about while tracing.
        try:
            mass_total_sfh = float(jnp.trapezoid(sfr_on_ssp, ssp_ages_yr))
            mass_pre_bb = float(
                jnp.trapezoid(jnp.where(ssp_age_gyr > t_obs_gyr, sfr_on_ssp, 0.0), ssp_ages_yr)
            )
            z_val = float(z)
            t_obs_val = float(t_obs_gyr)
        except jax.errors.ConcretizationTypeError:
            mass_total_sfh = None  # tracing — no concrete values to inspect
        if mass_total_sfh is not None:
            frac_pre_bb = mass_pre_bb / max(mass_total_sfh, 1e-30)
            if frac_pre_bb > 0.01:
                warnings.warn(
                    f"Star formation history forms {frac_pre_bb:.0%} of its stellar "
                    f"mass before the Big Bang at z={z_val:.2f} (cosmic age "
                    f"{t_obs_val:.2f} Gyr). That mass is truncated, so the "
                    f"prediction does not reflect the requested SFH — bound the SFH "
                    f"age parameter or the redshift to keep star formation within "
                    f"cosmic time.",
                    SFHBeforeBigBangWarning,
                    stacklevel=2,
                )

        # Lognormal metallicity-distribution-function width (Carnall+2018 §3.2,
        # #506): DSPS's ``*_lognormal_mdf`` / ``*_met_table`` kernels already
        # spread the SSP weights as a Gaussian in log10(Z) of this width about
        # the (per-age) mean metallicity. It is fittable via the optional public
        # ``met_logzsol_scatter`` parameter — read here with the build-time
        # ``config.lgmet_scatter`` as the fallback (so models that do not free it
        # are byte-unchanged). The sigma -> 0 limit recovers the delta-in-Z SSP
        # weighting. Threaded into both the delta and per-age-metallicity DSPS
        # calls below so the two paths stay consistent.
        lgmet_scatter = jnp.asarray(params.get("met_logzsol_scatter", self.config.lgmet_scatter))

        _used_cic = False
        _age_kernel = _resolve_age_kernel(self.config)
        if self.config.metallicity_model == "delta":
            # Delta metallicity: separable joint weights. The age marginal
            # comes from tengri's cloud-in-cell kernel on a dense integrand
            # (#964) — DSPS's histogram kernel interpolates log10(M(<t)) in
            # log10(t), which annihilates the mass in any table segment
            # straddling the SFH's maximum age (3.8 % of the total for the
            # delayed-tau age = 5 Gyr fiducial) and biased the CSP +1.2 % in
            # the optical vs FSPS / bagpipes / a dense reference. The GP-field draw
            # lives on the coarse lookback grid by construction, so the field path
            # keeps DSPS — a deliberate <~1% parametric-vs-field systematic (#964).
            # ``age_kernel`` makes that choice explicit and selectable; see
            # :func:`_resolve_age_kernel`.
            if _age_kernel == "cic":
                _fine_age_yr, _fine_sfr = _cic_integrand(
                    ssp_ages_yr, sfh_fn, sfh_kwargs, sfh_spec.fn, _tab_lbt_yr
                )
                age_w_cic, total_mass = _age_weights_cic(
                    _fine_age_yr, _fine_sfr, ssp_ages_yr, t_obs_gyr
                )
                lgmet_w = _lgmet_weights(log_z_abs_scalar, lgmet_scatter, ssp.ssp_lgmet)
                joint_weights = lgmet_w[:, None] * age_w_cic[None, :]
                _used_cic = True
            else:
                # GP-field SFH: coarse per-SSP-age integrand (the field draw
                # is defined on this grid) through DSPS's kernel, plus the
                # young-boundary knot so the youngest SSP bin captures the
                # [0, age0] mass — the delayed-tau Q_H fix (#538). total_mass
                # stays the conserved coarse value from above (the knot's
                # segment is excluded), so mass conservation is unaffected.
                _warn_if_dsps_kernel_truncates_history(
                    ssp_ages_yr, sfh_fn, sfh_kwargs, _tab_lbt_yr
                )
                gal_t_table, gal_sfr_table, _ = _build_dsps_sfh_table(
                    ssp_ages_yr, sfr_on_ssp, t_obs_gyr, add_young_knot=True
                )
                dsps_result = calc_rest_sed_sfh_table_lognormal_mdf(
                    **canonical_dsps_kwargs(
                        gal_t_table=gal_t_table,
                        gal_sfr_table=gal_sfr_table,
                        gal_lgmet=log_z_abs_scalar,
                        gal_lgmet_scatter=lgmet_scatter,
                        ssp_lgmet=ssp.ssp_lgmet,
                        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
                        ssp_flux=ssp_flux_for_csp,
                        t_obs=t_obs_gyr,
                    )
                )
                joint_weights = dsps_result.weights  # (n_met, n_age)
        else:  # ramp / chem_evol — per-age metallicity table
            if _age_kernel == "cic":
                # CIC joint weights on the dense integrand (#964), so the
                # per-age metallicity modes stay consistent with the delta
                # path and their degenerate configurations (constant table,
                # zero step, ...) reduce to it exactly.
                _fine_age_yr, _fine_sfr = _cic_integrand(
                    ssp_ages_yr, sfh_fn, sfh_kwargs, sfh_spec.fn, _tab_lbt_yr
                )
                joint_weights, total_mass = _joint_weights_cic_met_table(
                    _fine_age_yr,
                    _fine_sfr,
                    ssp_ages_yr,
                    t_obs_gyr,
                    lgmet_on_ssp_ages,
                    lgmet_scatter,
                    ssp.ssp_lgmet,
                )
                _used_cic = True
            else:
                from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_met_table

                # GP-field SFH: coarse per-SSP-age integrand through DSPS's
                # kernel, with the young-boundary knot (#538). The knot is the
                # last ascending element (t_cosmic = t_obs), so the per-age
                # metallicity table is extended by the youngest-age value.
                _t_k, _sfr_k, _ = _build_dsps_sfh_table(
                    ssp_ages_yr, sfr_on_ssp, t_obs_gyr, add_young_knot=True
                )
                _lgmet_k = jnp.concatenate([lgmet_on_ssp_ages[::-1], lgmet_on_ssp_ages[:1]])
                dsps_result = calc_rest_sed_sfh_table_met_table(
                    **canonical_dsps_kwargs(
                        gal_t_table=_t_k,
                        gal_sfr_table=_sfr_k,
                        gal_lgmet_table=_lgmet_k,
                        gal_lgmet_scatter=lgmet_scatter,
                        ssp_lgmet=ssp.ssp_lgmet,
                        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
                        ssp_flux=ssp_flux_for_csp,
                        t_obs=t_obs_gyr,
                    )
                )

                # ``dsps_result.weights`` is the joint (n_met, n_age)
                # probability distribution (sums to 1) over SSP grid points.
                # The age axis is already aligned with tengri's ssp_flux
                # ordering (ascending lookback age) — no flip needed.
                joint_weights = dsps_result.weights  # (n_met, n_age)

        if not _used_cic:
            # Youngest-bin edge-clip correction (#821) for the DSPS histogram
            # kernel paths (GP-field, per-age metallicity): DSPS's log-midpoint
            # age-bin edges put the youngest physical bin's lower edge at
            # lookback e_lo > 0, clipping the most ionizing recent stars and
            # biasing Q_H low (~4% on FSPS/MILES grids, up to ~31% for BPASS).
            # Scale that weight column by the grid-only factor e_hi/(e_hi-e_lo)
            # and renormalize. A no-op on grids with an age=0 template (BC03).
            # The CIC kernel (#964) assigns the [0, age0] mass to the youngest
            # node natively, so applying this there would double-count.
            _young_mult = _youngest_bin_lookback_multiplier(ssp.ssp_lg_age_gyr)
            joint_weights = joint_weights * _young_mult[None, :]
        # Guarded normalization: a degenerate SFH (empty star-formation
        # window, e.g. reversed const bounds) yields all-zero CIC weights;
        # 0/0 here would NaN the whole SED. Zero weights → zero SED is the
        # honest answer (DSPS's kernel instead floors SFR to SFR_MIN and
        # returns uniform-ish garbage weights for the same input).
        joint_weights = joint_weights / jnp.maximum(joint_weights.sum(), 1e-300)
        # Per-age × per-Msun-formed weighted SSP flux in erg/s/Hz/Msun. L_sun is
        # folded into the (params-independent) SSP operand INSIDE the einsum, not
        # applied as a runtime factor in ``total_mass * X * L_sun`` below. The
        # forward value is fine either way, but autodiff's local Jacobian for
        # that product, ``d/dX = total_mass * L_sun`` ~ 3.8e43, overflows float32
        # (3.4e38) as a standalone intermediate under XLA's *fused* reverse pass
        # — even though the true gradient is in range (the unfused path is
        # finite). With L_sun carried on the zero-gradient SSP constant, the only
        # Jacobians autodiff forms are ``total_mass`` (~1e10) and the erg-scaled
        # SSP (~3.8e18), both representable. Identical in float64 (#1206).
        ssp_flux_at_age = jnp.einsum(
            "ma,maw->aw", joint_weights, ssp_flux_for_csp * LSUN_ERG_PER_S
        )
        # Per-age "mass" for downstream per-age operations (dust BC mask).
        # This is the marginalized age distribution × total_mass.
        age_weights = joint_weights.sum(axis=0) * total_mass  # (n_age,) Msun

        # ── 7. Stellar SED in erg/s/Hz ──────────────────────────────────
        # Reconstruct the CSP SED as ``total_mass × Σ_met(weights × ssp_flux)``
        # rather than using DSPS's own ``rest_sed`` (= ``sed_unit_mstar ×
        # mstar_obs``). Both integrate the same SFH and use the same joint
        # ``weights``; they differ ONLY by the formed-mass scalar:
        # ``mstar_obs`` is DSPS's cumulative-SFH quadrature on its internal
        # log-time ``T_TABLE``, whereas ``total_mass`` is the trapezoid
        # integral on the SSP-age grid. The post-2026-05-25 SFH normalization
        # contract defines formed mass as ``trapezoid(sfr, t) = 10**log_total_mass``
        # (see ``_renormalize_to_mass`` and ``predict_surviving_mass``), so
        # ``total_mass`` is canonical and ``mstar_obs`` deviates by up to ~6.6%
        # at low z (large ``t_obs``; the two quadratures coincide for z ≳ 0.1).
        # Using DSPS's ``rest_sed`` (the #394 choice) silently broke that
        # contract for the SED at low z and disagreed with every precompute LUT
        # (which already use ``total_mass``). Reconstructing here makes the
        # exact SED, the photometry/spectrum LUTs, ``lnu_age``, ``L_age``,
        # ``age_weights``, and ``pred.stellar_mass`` all honor the one
        # contract mass. ``ssp_flux_at_age`` (line above) is the per-met-summed
        # joint-weighted SSP flux per Msun formed, already in erg/s/Hz/Msun
        # (L_sun folded in); scaling by ``total_mass`` is exactly ``Σ_age
        # lnu_age``. (Reverses #394.)
        lnu_age = total_mass * ssp_flux_at_age
        sed_intrinsic = lnu_age.sum(axis=0)

        # The bare erg/s scale ``total_mass x L_sun``. Written out on its own it
        # is ~1e42 for a 1e9 Msun galaxy, which overflows float32 (max 3.4e38)
        # to ``inf`` — silently, since JAX neither warns nor NaNs. The SED above
        # never trips this because ``total_mass x ssp_flux_at_age`` lands first
        # and keeps the magnitude small; the two consumers below have no such
        # small factor to hide behind. Pin the scale at working precision so a
        # float32 SSP grid (bc03_*, pgny_*) cannot poison the nebular backends
        # through the ionizing SED (#1099).
        mass_scale_erg = total_mass.astype(jnp.result_type(float)) * LSUN_ERG_PER_S
        # Log10 of the mass scale, folded into the log-domain Q_H integral
        # (_integrate_nion_log10 log10_scale) so no ~1e42 intermediate is materialized.
        log10_mass_scale = jnp.log10(total_mass.astype(jnp.result_type(float))) + jnp.log10(
            LSUN_ERG_PER_S
        )

        # ── 8. Mass quantities ──────────────────────────────────────────
        #
        # Two keys, on purpose. ``log_mstar`` keeps its documented fallback to the
        # formed mass when the SSP grid carries no mass-remaining table, because
        # downstream normalization needs *a* mass and cannot take a NaN.
        # ``log_mstar_surviving`` does NOT fall back: it is the user-facing answer
        # to "how much stellar mass is left", and when the grid cannot say, the
        # honest answer is NaN — not the formed mass, which silently asserts zero
        # mass loss (typically 30-40% of the formed mass; #1131). The old
        # ``predict_sfh_quantities`` returned NaN here and was right to.
        log_mstar_formed = jnp.log10(jnp.maximum(jnp.sum(age_weights), 1e-30))
        if ssp.ssp_mass_remaining is not None:
            mr_at_met = interpolate_mass_remaining(
                ssp.ssp_mass_remaining, ssp.ssp_lgmet, log_z_for_mr
            )
            mstar_surv = compute_surviving_mass(age_weights, mr_at_met)
            log_mstar = jnp.log10(jnp.maximum(mstar_surv, 1e-30))
            log_mstar_surviving = log_mstar
        else:
            log_mstar = log_mstar_formed
            log_mstar_surviving = jnp.asarray(jnp.nan)

        # ── 9. SFR averages on the SFH grid ─────────────────────────────
        sfr_now = sfr_history[0]
        sfr_10myr = _time_weighted_sfr(sfr_history, sfh_lbt_grid, 1e7)
        sfr_100myr = _time_weighted_sfr(sfr_history, sfh_lbt_grid, 1e8)

        # ── 10. Bolometric L per SSP age bin ────────────────────────────
        # ν = c/λ ⟹ |dν| = c/λ² dλ. Trapezoid in wavelength with the
        # frequency Jacobian gives ∫ L_ν dν per age.
        wave = ssp.ssp_wave
        nu_jac = C_AA / (wave**2)
        # `log_L_age` must not materialize the ~1e46 erg/s product that overflows
        # float32 (#1534) — and must not cost an extra pass over the cube to avoid it.
        #
        # `ssp_flux_at_age` is the per-Msun cube and `lnu_age = total_mass *
        # ssp_flux_at_age`, so the offending scale is already factored out upstream.
        # Integrating the per-Msun cube and re-applying `total_mass` in log space
        # needs the *same single trapezoid* as before, plus a log over an (n_age,)
        # vector. Being per-Msun, its headroom does not vary with galaxy mass:
        # `ssp_flux_at_age * nu_jac` peaks ~1.7e35 against the 3.4e38 ceiling at any
        # total_mass.
        #
        # Two earlier attempts were wrong and the inventory sweep caught both:
        # `log10_magnitude(L_age)` is useless (L_age is already inf in float32, and
        # log of inf is inf), and peak-factoring `lnu_age * nu_jac` after forming it
        # is too late (the product is already inf, so its peak is inf). A third,
        # peak-factoring `lnu_age` by its own per-row max, was correct but cost +19%
        # of a full predict_state (287 -> 343 us) for a value most callers never read.
        #
        # log10(total_mass), not log10_mass_scale: L_sun is already folded into
        # ssp_flux_at_age, and log10_mass_scale carries it a second time.
        _per_msun_L = jnp.trapezoid(ssp_flux_at_age * nu_jac[None, :], wave, axis=1)
        L_age = total_mass * _per_msun_L
        _log_per_msun = log10_magnitude(_per_msun_L)
        log_L_age = jnp.where(
            _not_computable(_log_per_msun),
            jnp.inf,
            _log_per_msun + jnp.log10(total_mass.astype(jnp.result_type(float))),
        )

        # ── 11. Ionizing photon production rate (λ < 911.76 Å) ──────────
        # photons/s = ∫_{ν > c/λ_HI} L_ν / (hν) dν, summed over all ages.
        # Mirrors components/nebular/ionizing_spectrum.py:299.
        #
        # Partial-bin correction (#537): when 911.76 Å falls between two
        # grid points (true for BC03's 10 Å sampling: 905 and 915 Å are
        # the bracketing points), a hard ``wave < 911.76`` mask drops
        # the 905 → 911.76 portion of the boundary bin entirely. SSP
        # spectra have a near-discontinuous Lyman drop at 911.76 Å:
        # ionizing flux is well-defined right up to the limit, then
        # drops to zero. Linear interpolation between 905 and 915 Å
        # would under-estimate the boundary value (a half-value of the
        # ionizing side); the correct partial-bin contribution treats
        # ``L_ν`` as constant from the last ionizing grid point up to
        # 911.76 Å — a rectangle, not a trapezium. This matches the
        # physical Lyman discontinuity and produces a Q_H consistent
        # with CIGALE's tabulated ``stellar.n_ly`` to within numerical
        # noise at any SSP grid spacing.
        # Q_H via the single-sourced integral (partial-bin Lyman correction
        # lives in _integrate_nion, shared with compute_nion for bit-exactness).
        #
        # Integrate over the ionizing SLICE alone (lambda < 2*911.76 A, a prefix
        # of the ascending grid) rather than the full-grid sed_intrinsic. This
        # decouples nion from the 6000-wave stellar SED: under approx=WavePrecomp
        # the LUT path can then prune the full stellar einsum, which was
        # otherwise dragged into the graph solely to publish Q_H for the nebular
        # backend (#950). Bit-exact — sed_ion == sed_intrinsic[:n_ion]. Falls
        # back to the full integral when the static bound was not precomputed.
        _n_ion = self._state.n_ion_bins if self._state is not None else None
        if _n_ion is not None and _n_ion > 0:
            # Compute Q_H in log-domain to avoid float32 overflow (#1206).
            # The tensordot result is O(1); the scale rides the log integral.
            _tensordot_result = jnp.tensordot(
                joint_weights, ssp_flux_for_csp[:, :, :_n_ion], axes=([0, 1], [0, 1])
            )
            log_nion = _integrate_nion_log10(
                _tensordot_result, wave[:_n_ion], log10_scale=log10_mass_scale
            )
        elif _n_ion is not None:
            # n_ion_bins == 0 (static): no grid bins below the Lyman limit
            # (IR-focused configs) -> Q_H is identically zero. Skips the slice
            # machinery: max/argmax over zero-size arrays raise (#1193 fallout).
            log_nion = jnp.full((), -jnp.inf)
        else:
            log_nion = _integrate_nion_log10(sed_intrinsic, wave)
        nion = pow10(log_nion)  # linear transition surface; exp(-inf) == 0.0

        # ── 11b. Project to pipeline wavelength grid ────────────────
        # When the pipeline runs on a panchromatic grid (radio/X-ray
        # extension via ``make_panchromatic_grid``), ``state.wave`` is
        # wider than ``ssp.ssp_wave``. Both ``sed_intrinsic`` and the
        # per-age cube ``lnu_age`` MUST live on ``state.wave`` so
        # downstream additive emitters (radio, X-ray) and per-age
        # transforms (dust two-component) can broadcast. Linear interp
        # is exact at SSP grid points (panchromatic preserves them) and
        # zero is the physically correct extrapolation outside the SSP
        # range — the SSP templates carry no information there.
        #
        # The shape comparison is Python-level (both arrays exist at
        # trace time), so the no-extension case incurs zero JIT cost.
        if state.wave.shape[0] != wave.shape[0]:
            target = state.wave
            ssp_wave_arr = wave
            outside = (target < ssp_wave_arr[0]) | (target > ssp_wave_arr[-1])
            sed_intrinsic_proj = jnp.where(
                outside, 0.0, jnp.interp(target, ssp_wave_arr, sed_intrinsic)
            )
            from jax import vmap

            lnu_age_proj = vmap(lambda row: jnp.interp(target, ssp_wave_arr, row))(lnu_age)
            lnu_age_proj = jnp.where(outside[None, :], 0.0, lnu_age_proj)
            sed_intrinsic = sed_intrinsic_proj
            lnu_age = lnu_age_proj

        # ── 12b. Stellar photometry LUT (WavePrecomp) ──────────────────
        # When eager precomputation is enabled and the LUT is available,
        # compute stellar_phot_lnu_precomp and publish it to derived.
        derived_overrides = dict(
            log_mstar=log_mstar,
            log_mstar_formed=log_mstar_formed,
            log_mstar_surviving=log_mstar_surviving,
            sfr=sfr_now,
            sfr_10myr=sfr_10myr,
            sfr_100myr=sfr_100myr,
            L_age=L_age,
            # log companion (#1534), peak-factored at the source above rather than
            # taken from the overflowed linear value.
            log_L_age=log_L_age,
            lnu_age=lnu_age,
            # Per-(met, age) DSPS weights and the total_mass x L_sun scaling,
            # published so DustSEDComponent can evaluate the energy-balance
            # L_ir from a precomputed bolometric (tau_bc, tau_diff) LUT instead
            # of the full-wavelength stellar cube (WavePrecomp speed path).
            joint_weights=joint_weights,
            stellar_mass_scale=mass_scale_erg,
            # The float32-safe form of the same scale. ``mass_scale_erg`` is
            # ~1e43 for a 1e10 Msun galaxy and so is ``inf`` in pure float32
            # for any galaxy above ~9e4 Msun — it is total_mass times a
            # constant, with no SSP flux factor to keep it in range (#1206).
            log_stellar_mass_scale=log10_mass_scale,
            # CSP mass weights (Msun per SSP age bin), summed
            # over the metallicity axis. Published so downstream
            # nebular backends (Cue, CloudyGrid) can call their
            # high-level ``predict_nebular_*(ssp_weights=...)``
            # entry points and derive Q_H + ionizing spectrum
            # from the SSP, matching legacy parity.
            age_weights=age_weights,
            log_nion=log_nion,
            nion=nion,
            sfh_grid_lbt_yr=sfh_lbt_grid,
            sfr_history=sfr_history,
            log_metallicity_history=log_metallicity_history,
            # Published for downstream (dust two-component attenuation
            # needs the SSP age axis to apply the BC/diffuse split).
            ssp_ages_yr=ssp_ages_yr,
        )

        if self._state is not None and self._state.ssp_phot_lut is not None:
            # Fixed-z path — LUT built at source's z in precompute()
            ssp_phot = self._state.ssp_phot_lut.ssp_phot
            # (n_met, n_age, n_filt) in Lsun/Hz/Msun; sum over metallicity and
            # age axes weighted by joint distribution × total mass.
            # Convert to erg/s/Hz to match sed_intrinsic units.
            stellar_phot_lnu_precomp_rest = _mass_scale_lnu(
                jnp.einsum("ma,maf->f", joint_weights, ssp_phot), total_mass
            )
            derived_overrides["stellar_phot_lnu_precomp"] = stellar_phot_lnu_precomp_rest
            # Age-resolved per-filter LUT for two-component
            # dust attenuation. Marginalize over metallicity only; preserve
            # the age axis. Shape (n_age, n_filter). Sum over age == the
            # marginalized stellar_phot_lnu_precomp above.
            stellar_phot_lnu_per_age = _mass_scale_lnu(
                jnp.einsum("ma,maf->af", joint_weights, ssp_phot), total_mass
            )
            derived_overrides["stellar_phot_lnu_per_age_precomp"] = stellar_phot_lnu_per_age
            # Taylor moment Ψ — same einsum, units erg/s/Hz × Å.
            ssp_phot_moment = self._state.ssp_phot_lut.ssp_phot_moment
            if ssp_phot_moment is not None:
                stellar_phot_moment_precomp = _mass_scale_lnu(
                    jnp.einsum("ma,maf->f", joint_weights, ssp_phot_moment), total_mass
                )
                derived_overrides["stellar_phot_moment_precomp"] = stellar_phot_moment_precomp
                stellar_phot_moment_per_age = _mass_scale_lnu(
                    jnp.einsum("ma,maf->af", joint_weights, ssp_phot_moment), total_mass
                )
                derived_overrides["stellar_phot_moment_per_age_precomp"] = (
                    stellar_phot_moment_per_age
                )
            # Sub-band quadrature tensors (#1122). Φ_k carries the same mass and
            # L_sun scaling as Φ; the node λ_k is a RATIO, so those scalings cancel
            # and it is computed from the unscaled sums.
            ssp_sub_phot = self._state.ssp_phot_lut.ssp_subband_phot
            if ssp_sub_phot is not None:
                ssp_sub_waves = self._state.ssp_phot_lut.ssp_subband_waves_rest
                sub_phi = jnp.einsum("ma,mafk->afk", joint_weights, ssp_sub_phot)
                sub_num = jnp.einsum("ma,mafk->afk", joint_weights, ssp_sub_waves * ssp_sub_phot)
                derived_overrides["stellar_phot_lnu_per_age_subband_precomp"] = _mass_scale_lnu(
                    sub_phi, total_mass
                )
                # Sub-bands with no usable weight contribute nothing, but their
                # node still goes through the 1/λ dust law — keep it finite and
                # positive. The floor (not ``!= 0.0``) is what keeps the node's
                # DERIVATIVE finite; see ``_subband_live_floor`` (#1397).
                live = jnp.abs(sub_phi) > _subband_live_floor()
                derived_overrides["stellar_subband_waves_rest_precomp"] = jnp.where(
                    live,
                    _flux_weighted_node(sub_num, jnp.where(live, sub_phi, 1.0)),
                    jnp.asarray(self._state.ssp_phot_lut.effective_wavelengths_rest)[:, None],
                )
                # The same tensor with the IGM folded in at the nodes (#1135).
                # Identical einsum, on a constant that already carries T — the met
                # axis is contracted here, so T had to be evaluated on it (the node
                # moves with the free met_logzsol). Kept alongside the IGM-free
                # tensor rather than replacing it: phot_rest_fnu is projected at
                # z=0 and carries no IGM.
                ssp_sub_phot_igm = self._state.ssp_phot_lut.ssp_subband_phot_igm
                if ssp_sub_phot_igm is not None:
                    derived_overrides["stellar_phot_lnu_per_age_subband_igm_precomp"] = (
                        _mass_scale_lnu(
                            jnp.einsum("ma,mafk->afk", joint_weights, ssp_sub_phot_igm), total_mass
                        )
                    )
            # Publish filter pivot wavelengths so the dust LUT
            # (and future per-filter consumers like AGN and IGM) can use them.
            derived_overrides["filter_eff_waves"] = jnp.asarray(
                self._state.ssp_phot_lut.effective_wavelengths_rest
            )
            if self._state.phot_fw_padded is not None:
                derived_overrides["phot_filter_waves_padded"] = self._state.phot_fw_padded
                derived_overrides["phot_filter_trans_padded"] = self._state.phot_ft_padded

        elif self._state is not None and self._state.ssp_phot_ztable is not None:
            # Free-z path — smooth triweight
            # interp of the ztable at runtime z. Publishes the same derived
            # keys as the fixed-z path: stellar_phot_lnu_precomp,
            # stellar_phot_moment_precomp, stellar_phot_lnu_per_age_precomp,
            # stellar_phot_moment_per_age_precomp, filter_eff_waves.
            #
            # The original linear z-interp was O(h^2) and non-monotonic in
            # n_z at fixed test redshifts: doubling the grid can shift a
            # test point into a less-favorable cell and raise the error.
            # The triweight kernel (Hearin et al. 2023) is the canonical
            # smooth-grid interpolant used throughout tengri for SSP, CLOUDY,
            # and SKIRTOR grids — C²-continuous, kernel-supported on the
            # 3-bandwidth neighborhood. See issue #438.
            from tengri.utils.interpolation import (
                apply_grid_window,
                compute_grid_window,
                edges_for_grid,
            )

            ztable = self._state.ssp_phot_ztable
            z = jnp.asarray(require_redshift(params, "components.stellar.component.apply"))
            z_grid = ztable.z_grid
            z_edges = edges_for_grid(z_grid)
            # Match grid-cell width for the kernel bandwidth (Hearin 2023
            # convention): smooth across one neighbor on each side. Given in
            # CELLS, not in z: the window width is a shape, and 0.5 * (z_grid[1]
            # - z_grid[0]) is a tracer under jit even though z_grid is a
            # constant, so it cannot size anything.
            # Windowed, not dense. The kernel is supported on five nodes; the
            # other n_z - 5 weights are EXACTLY zero, so contracting the full
            # axis multiplied the whole (n_z, n_met, n_age, n_filt) table by
            # zeros. That was 87% of the free-redshift gradient arithmetic —
            # 128 MFLOP at n_z=250 against 2.7 MFLOP at fixed z. Identical
            # values and gradients; the cost simply stops tracking n_z.
            z_start, w_z = compute_grid_window(z, z_grid, bandwidth_cells=0.5, edges=z_edges)

            def _interp(table):
                # table: (n_z, ...). Contract the supported window of axis 0.
                return apply_grid_window(table, z_start, w_z)

            # ssp_phot_table: (n_z, n_met, n_age, n_filt); interp along axis 0.
            ssp_phot_at_z = _interp(ztable.ssp_phot_table)
            # Marginalized + age-resolved LUTs (parity with the fixed-z path).
            stellar_phot_lnu_precomp_rest = _mass_scale_lnu(
                jnp.einsum("ma,maf->f", joint_weights, ssp_phot_at_z), total_mass
            )
            stellar_phot_lnu_per_age = _mass_scale_lnu(
                jnp.einsum("ma,maf->af", joint_weights, ssp_phot_at_z), total_mass
            )
            derived_overrides["stellar_phot_lnu_precomp"] = stellar_phot_lnu_precomp_rest
            derived_overrides["stellar_phot_lnu_per_age_precomp"] = stellar_phot_lnu_per_age
            # Taylor moment Ψ at runtime z. Interpolate the
            # moment table the same way and publish marginalized + per-age.
            if ztable.ssp_phot_moment_table is not None:
                ssp_moment_at_z = _interp(ztable.ssp_phot_moment_table)
                stellar_phot_moment_precomp = _mass_scale_lnu(
                    jnp.einsum("ma,maf->f", joint_weights, ssp_moment_at_z), total_mass
                )
                stellar_phot_moment_per_age = _mass_scale_lnu(
                    jnp.einsum("ma,maf->af", joint_weights, ssp_moment_at_z), total_mass
                )
                derived_overrides["stellar_phot_moment_precomp"] = stellar_phot_moment_precomp
                derived_overrides["stellar_phot_moment_per_age_precomp"] = (
                    stellar_phot_moment_per_age
                )
            # Interpolate effective rest-frame wavelengths and publish for
            # downstream consumers (dust LUT, AGN, IGM).
            eff_waves_at_z = _interp(ztable.eff_waves_rest_table)
            derived_overrides["filter_eff_waves"] = eff_waves_at_z

            # Sub-band quadrature tensors at runtime z (#1122), same contract as
            # the fixed-z path. Φ_k carries the mass and L_sun scaling; the node
            # λ_k is a RATIO, so those cancel and it comes from the unscaled sums.
            if ztable.ssp_subband_phot_table is not None:
                sub_phot_at_z = _interp(ztable.ssp_subband_phot_table)
                sub_wave_at_z = _interp(ztable.subband_waves_rest_table)
                sub_phi = jnp.einsum("ma,mafk->afk", joint_weights, sub_phot_at_z)
                sub_num = jnp.einsum("ma,mafk->afk", joint_weights, sub_wave_at_z * sub_phot_at_z)
                derived_overrides["stellar_phot_lnu_per_age_subband_precomp"] = _mass_scale_lnu(
                    sub_phi, total_mass
                )
                # Sub-bands with no usable weight cannot change the result, but
                # their node still goes through the 1/λ dust law — keep it finite
                # and positive. The floor (not ``!= 0.0``) is what keeps the
                # node's DERIVATIVE finite; see ``_subband_live_floor`` (#1397).
                live = jnp.abs(sub_phi) > _subband_live_floor()
                derived_overrides["stellar_subband_waves_rest_precomp"] = jnp.where(
                    live,
                    _flux_weighted_node(sub_num, jnp.where(live, sub_phi, 1.0)),
                    eff_waves_at_z[:, None],
                )
                # IGM folded in at the nodes (#1135) — tabulated against this same
                # z grid at build time, so it rides the same triweight interpolation
                # as the sub-band photometry it multiplies.
                if ztable.ssp_subband_phot_igm_table is not None:
                    sub_phot_igm_at_z = _interp(ztable.ssp_subband_phot_igm_table)
                    derived_overrides["stellar_phot_lnu_per_age_subband_igm_precomp"] = (
                        _mass_scale_lnu(
                            jnp.einsum("ma,mafk->afk", joint_weights, sub_phot_igm_at_z),
                            total_mass,
                        )
                    )
            if self._state.phot_fw_padded is not None:
                derived_overrides["phot_filter_waves_padded"] = self._state.phot_fw_padded
                derived_overrides["phot_filter_trans_padded"] = self._state.phot_ft_padded

        # ── 12b-rest. The REST-frame band (#1148) ───────────────────────────
        # ``phot_rest_fnu`` is the SED reprojected at z=0, d_L=10 pc — the galaxy
        # as it is — so the filter sits in the REST frame and samples the rest SED
        # at its own pivot. That is a different integral from the observed-band
        # tensors above, which sample rest λ_eff/(1+z). The LUT used to reuse those
        # for the rest-frame flux, which put it 769 % from the exact path in des_g
        # at z=0.5 and orders of magnitude out in the blue.
        #
        # Published for the fixed-z and free-z paths alike, from ONE constant: the
        # rest band does not move with redshift, so there is nothing to interpolate.
        if self._state is not None and self._state.restband_lut is not None:
            rb = self._state.restband_lut
            derived_overrides["stellar_restband_lnu_precomp"] = _mass_scale_lnu(
                jnp.einsum("ma,maf->f", joint_weights, rb.ssp_restband_phot), total_mass
            )
            derived_overrides["filter_restband_eff_waves"] = jnp.asarray(rb.restband_eff_waves)
            if rb.ssp_restband_subband_phot is not None:
                rb_phi = jnp.einsum("ma,mafk->afk", joint_weights, rb.ssp_restband_subband_phot)
                rb_num = jnp.einsum(
                    "ma,mafk->afk",
                    joint_weights,
                    rb.ssp_restband_subband_waves * rb.ssp_restband_subband_phot,
                )
                derived_overrides["stellar_restband_lnu_per_age_subband_precomp"] = (
                    _mass_scale_lnu(rb_phi, total_mass)
                )
                # The node is a RATIO, so mass and L_sun cancel — take it from the
                # unscaled sums. Sub-bands with no usable weight keep a finite,
                # positive node: a zero would go to inf through the 1/λ dust law.
                # The floor (not ``!= 0.0``) is what keeps the node's DERIVATIVE
                # finite; see ``_subband_live_floor`` (#1397).
                rb_live = jnp.abs(rb_phi) > _subband_live_floor()
                derived_overrides["stellar_restband_subband_waves_precomp"] = jnp.where(
                    rb_live,
                    rb_num / jnp.where(rb_live, rb_phi, 1.0),
                    jnp.asarray(rb.restband_eff_waves)[:, None],
                )

        # ── 12c. Stellar spectrum LUT (SpectrumPrecomp) ─────────────────
        # Pre-rebinned SSP × pixel LUT: the continuum at the spectrum pixel
        # centers in the galaxy rest frame. Publishes:
        #   - ``stellar_spec_lnu_precomp`` (n_pix,) — rest-frame Lν [erg/s/Hz]
        #   - ``spec_eff_waves`` (n_pix,) — rest-frame pixel wavelengths [Å]
        # The latter routes downstream SEDModelComponents (dust / AGN / IGM /
        # nebular continuum) through their spectrum-LUT branch, mirroring how
        # ``filter_eff_waves`` drives the photometry LUT path.
        if self._state is not None and self._state.ssp_spec_lut is not None:
            ssp_on_pixels = self._state.ssp_spec_lut.ssp_on_pixels  # (n_met, n_age, n_pix)
            stellar_spec_lnu = _mass_scale_lnu(
                jnp.einsum("ma,map->p", joint_weights, ssp_on_pixels), total_mass
            )
            derived_overrides["stellar_spec_lnu_precomp"] = stellar_spec_lnu
            # Age-resolved per-pixel LUT for two-component (Charlot & Fall)
            # dust attenuation at the pixel grid (sum over age == marginalized).
            stellar_spec_lnu_per_age = _mass_scale_lnu(
                jnp.einsum("ma,map->ap", joint_weights, ssp_on_pixels), total_mass
            )
            derived_overrides["stellar_spec_lnu_per_age_precomp"] = stellar_spec_lnu_per_age
            derived_overrides["spec_eff_waves"] = jnp.asarray(
                self._state.ssp_spec_lut.wave_rest_pixels
            )

        elif self._state is not None and self._state.ssp_spec_ztable is not None:
            # Free-z: interpolate the SSP cube to the rest-frame pixel grid
            # ``wave_obs / (1 + z)`` at runtime. Exact (no z-grid interpolation
            # of absorption features) and differentiable in z. ``wave`` is
            # ``ssp.ssp_wave`` and ``ssp_flux_for_csp`` is the (n_met, n_age,
            # n_wave) cube already used for the full-grid CSP einsum above.
            from jax import vmap

            z = jnp.asarray(require_redshift(params, "components.stellar.component.apply"))
            wave_obs_pix = jnp.asarray(self._state.ssp_spec_ztable.wave_obs_pixels)
            wave_rest = wave_obs_pix / (1.0 + z)
            n_met_s, n_age_s = ssp_flux_for_csp.shape[0], ssp_flux_for_csp.shape[1]
            flat = ssp_flux_for_csp.reshape(n_met_s * n_age_s, -1)
            interp_flat = vmap(lambda row: jnp.interp(wave_rest, wave, row, left=0.0, right=0.0))(
                flat
            )
            ssp_on_pixels_at_z = interp_flat.reshape(n_met_s, n_age_s, -1)
            stellar_spec_lnu = _mass_scale_lnu(
                jnp.einsum("ma,map->p", joint_weights, ssp_on_pixels_at_z), total_mass
            )
            stellar_spec_lnu_per_age = _mass_scale_lnu(
                jnp.einsum("ma,map->ap", joint_weights, ssp_on_pixels_at_z), total_mass
            )
            derived_overrides["stellar_spec_lnu_precomp"] = stellar_spec_lnu
            derived_overrides["stellar_spec_lnu_per_age_precomp"] = stellar_spec_lnu_per_age
            derived_overrides["spec_eff_waves"] = wave_rest

        # ── 12. Assemble new state ──────────────────────────────────────
        return state.with_(
            sed_intrinsic=sed_intrinsic,
            derived=state.derived.with_(**derived_overrides),
        )

    def compute_joint_weights(self, params, ssp_data=None):
        """(met, age) CSP weights + total mass WITHOUT the full-wavelength SED.

        Reproduces exactly the weight computation inside :meth:`apply` for the
        delta-metallicity, non-field path — the registry SFH-kwargs translation,
        the cloud-in-cell age marginal on a dense integrand (#758/#964), and the
        lognormal-MDF metallicity marginal (#982) — but skips the ~5994-wavelength
        SED einsum entirely (it needs only the weights, not the reconstructed SED).

        This is the FeaturePrecomp fast-path entry: the wNE window-LUT gets
        ``joint_weights`` in microseconds and never reconstructs the full SED.

        Restricted to the supported configuration — **delta** metallicity, a
        **closed-form parametric** SFH (with or without the GP field, #1204),
        **no** alpha-Fe SSP grid — and raises for anything else, so the fast
        path can never silently diverge from the exact forward. Callers must
        fall back to the exact path.

        The non-parametric families and the **tabulated** SFH are both refused
        via :data:`_FAST_PATH_UNSUPPORTED_SFH_FNS`, which carries a per-family
        reason. The tabulated case is the one that used to slip through: its
        registry ``fn`` is an all-zero placeholder, so the fast path returned
        zero weights and zero mass, finite and unwarned (#1395).

        Parameters
        ----------
        params : Mapping
            Free-parameter dict (same shape as :meth:`apply`).
        ssp_data : SSPData, optional
            Override for the model's SSP grid.

        Returns
        -------
        joint_weights : ndarray, shape (n_met, n_age)
            Normalized (met, age) CSP weight distribution (sums to 1).
        total_mass : ndarray, shape ()
            Total formed stellar mass [Msun] (coarse, pre-young-knot basis).
        ssp_ages_yr : ndarray, shape (n_age,)
            SSP lookback ages [yr].

        Raises
        ------
        ValueError
            For any configuration outside delta metallicity / closed-form
            parametric SFH / no alpha-Fe grid — including the tabulated SFH
            (#1395). The caller must use the exact forward there.
        """
        from tengri.components.stellar.sfh.registry import SFH_REGISTRY
        from tengri.components.stellar.sps.dsps_wrapper import has_alpha_grid
        from tengri.cosmology import age_at_z as _age_at_z

        ssp = ssp_data if ssp_data is not None else self.ssp_data
        sfh_spec = SFH_REGISTRY[self.config.sfh_model]

        # ── supported-configuration guards (loud, never silent) ──
        # The GP field is supported (delta metallicity, parametric backbone): it
        # modulates the SFR on the lookback grid and is handled in the field
        # branch below via the shared ``_apply_gp_field`` + the same DSPS weight
        # function apply uses. The remaining guards still fall back to the exact
        # forward.
        if self.config.metallicity_model not in ("delta", "table"):
            raise ValueError(
                f"compute_joint_weights supports metallicity_model 'delta' and "
                f"'table' (got {self.config.metallicity_model!r}); use the exact "
                f"forward."
            )
        # A tabulated SFH is SERVED (#1396) via the shared runtime-table closure,
        # not refused — it is routed below before the registry placeholder is ever
        # evaluated. The map remains the backstop for the non-parametric families,
        # and for a tabulated SFH that somehow reached here unrouted.
        if self.config.sfh_model != "table":
            unsupported_reason = _FAST_PATH_UNSUPPORTED_SFH_FNS.get(sfh_spec.fn)
            if unsupported_reason is not None:
                raise ValueError(
                    f"compute_joint_weights does not support "
                    f"sfh_model={self.config.sfh_model!r}: {unsupported_reason}; "
                    f"use the exact forward."
                )
        if has_alpha_grid(ssp):
            raise ValueError(
                "compute_joint_weights does not support alpha-Fe SSP grids; use the exact forward."
            )

        ssp_ages_yr = (10.0**ssp.ssp_lg_age_gyr) * 1e9

        # SFH kwargs — identical registry translation to apply (§2)
        sfh_kwargs = {}
        for public_name, (internal_name, scale, offset) in sfh_spec.internal_param_map.items():
            if public_name in params:
                raw = params[public_name]
            else:
                pdef = sfh_spec.params.get(public_name)
                default_scalar = pdef.default.default if pdef is not None else None
                if default_scalar is None:
                    continue
                raw = default_scalar
            sfh_kwargs[internal_name] = jnp.asarray(raw) * scale + offset
        if self.config.sfh_model == "dense_basis":
            age_universe_gyr = sfh_spec.settings.get("sfh_db_age_universe_gyr", 13.47)
            sfh_kwargs["age_universe_yr"] = float(age_universe_gyr) * 1e9

        z = jnp.asarray(
            require_redshift(params, "components.stellar.component.compute_joint_weights")
        )
        t_obs_gyr = jnp.asarray(_age_at_z(z)).reshape(())

        # Runtime tabulated SFH (#996/#1396) — the SAME closure and lookback
        # knots the exact forward builds, from the single shared helper, so the
        # two routes cannot read a simulation history differently.
        sfh_fn = sfh_spec.fn
        _tab_lbt_yr = None
        _tab_order = None
        if self.config.sfh_model == "table":
            if self.config.field:
                raise NotImplementedError(
                    "sfh_model='table' with field=True is not supported — the "
                    "GP field draw modulates parametric SFHs only (#996)."
                )
            sfh_fn, _tab_lbt_yr, _tab_order = _tabulated_sfh(params, t_obs_gyr)

        lgmet_scatter = jnp.asarray(params.get("met_logzsol_scatter", self.config.lgmet_scatter))

        _age_kernel = _resolve_age_kernel(self.config)

        # Metallicity — delta gives one scalar log10(Z); table gives a per-age
        # curve that routes to the CIC met-table kernel below (matches apply §4).
        log_z_abs_scalar = None
        lgmet_on_ssp_ages = None
        if self.config.metallicity_model == "table":
            lgmet_on_ssp_ages, _met_log_age_yr, _met_log_z_abs = _tabulated_lgmet_on_ssp_ages(
                params, self.config, ssp.ssp_lg_age_gyr, _tab_lbt_yr, _tab_order
            )
        else:
            alpha_fe = jnp.asarray(params.get("met_alpha_fe", 0.0))
            log_z_abs_scalar = (
                effective_metallicity(jnp.asarray(params["met_logzsol"]), alpha_fe) + LOG10_ZSUN
            )

        # DSPS-histogram CSP weights — mirrors apply's DSPS path EXACTLY. The
        # (met, age) weights come from the SAME DSPS function
        # (``calc_ssp_weights_sfh_table_lognormal_mdf``) that apply's SED call uses
        # internally — so the fast and exact line paths cannot diverge. ``total_mass``
        # is the conserved coarse value (no young knot), matching apply §3.
        #
        # Reached by a GP-field SFH (whose draw lives on the coarse lookback grid,
        # so DSPS is the only implemented kernel) and by any non-field model that
        # explicitly selects ``age_kernel="dsps"`` (#964).
        if _age_kernel == "dsps":
            if lgmet_on_ssp_ages is not None:
                # The scalar-MDF DSPS call below has no per-age metallicity
                # axis; feeding it ``log_z_abs_scalar=None`` would fail deep
                # inside DSPS (or, worse, silently drop Z(t)). apply's
                # ``calc_rest_sed_sfh_table_met_table`` arm covers this
                # combination — the SED-free fast path does not.
                raise NotImplementedError(
                    "The SED-free fast path does not support the DSPS age "
                    "kernel with a per-age metallicity table "
                    f"(metallicity_model={self.config.metallicity_model!r}). "
                    "Use age_kernel='cic' (the default), or call predict()/"
                    "apply() instead of the line/nion fast path."
                )
            from dsps.sed.ssp_weights import calc_ssp_weights_sfh_table_lognormal_mdf

            if self.config.field:
                # The field modulates the SFR on the lookback grid
                # (``_apply_gp_field``, the single source shared with
                # :meth:`apply`), which is interpolated to the SSP ages.
                n_grid = self.config.n_grid
                log_age_grid = make_log_age_grid(n_grid)
                sfh_lbt_grid = 10.0**log_age_grid
                sfr_history = sfh_spec.fn(sfh_lbt_grid, **sfh_kwargs)
                sfr_history = _apply_gp_field(
                    sfr_history, params, n_grid, log_age_grid, self.config.field_centering
                )
                sfr_on_ssp = jnp.interp(ssp_ages_yr, sfh_lbt_grid, sfr_history)
            else:
                # Non-field: apply evaluates the closed-form SFH directly on the
                # SSP ages (§3) rather than through the log grid, so the fast
                # path must do the same or the two routes read one SFH
                # differently (#982).
                sfr_on_ssp = sfh_fn(ssp_ages_yr, **sfh_kwargs)
            _warn_if_dsps_kernel_truncates_history(ssp_ages_yr, sfh_fn, sfh_kwargs, _tab_lbt_yr)
            _, _, total_mass = _build_dsps_sfh_table(ssp_ages_yr, sfr_on_ssp, t_obs_gyr)
            gal_t, gal_sfr, _ = _build_dsps_sfh_table(
                ssp_ages_yr, sfr_on_ssp, t_obs_gyr, add_young_knot=True
            )
            _dsps_args = canonical_dsps_kwargs(
                gal_t=gal_t,
                gal_sfr=gal_sfr,
                gal_lgmet=log_z_abs_scalar,
                gal_lgmet_scatter=lgmet_scatter,
                ssp_lgmet=ssp.ssp_lgmet,
                ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
                t_obs=t_obs_gyr,
            )
            weights, _, _ = calc_ssp_weights_sfh_table_lognormal_mdf(
                _dsps_args["gal_t"],
                _dsps_args["gal_sfr"],
                _dsps_args["gal_lgmet"],
                _dsps_args["gal_lgmet_scatter"],
                _dsps_args["ssp_lgmet"],
                _dsps_args["ssp_lg_age_gyr"],
                _dsps_args["t_obs"],
            )
            # #821 youngest-bin edge-clip correction — apply applies this for ALL
            # DSPS-histogram-kernel paths (field / per-age metallicity); without it
            # the youngest bin (which carries the ionizing, line-emitting stars) is
            # clipped ~10% low. The CIC path (below) bakes it in instead.
            weights = weights * _youngest_bin_lookback_multiplier(ssp.ssp_lg_age_gyr)[None, :]
            joint_weights = weights / jnp.maximum(weights.sum(), 1e-300)
            return joint_weights, total_mass, ssp_ages_yr

        # Delta + non-field CSP weights — mirrors apply's delta path EXACTLY
        # (#982): a cloud-in-cell age marginal on a dense integrand (#758/#964,
        # with the SFH's exact bin-edge knots injected for binned families) times
        # the lognormal-MDF metallicity marginal. ``_age_weights_cic`` already
        # applies the youngest-bin lookback correction and returns the conserved
        # total_mass, so — unlike the DSPS histogram path — the caller must NOT
        # also multiply by ``_youngest_bin_lookback_multiplier`` here.
        # The SAME builder apply uses, so the two integrands are identical point
        # for point — the #982 contract, now enforced by construction.
        _fine_age_yr, _fine_sfr = _cic_integrand(
            ssp_ages_yr, sfh_fn, sfh_kwargs, sfh_spec.fn, _tab_lbt_yr
        )

        # Per-age metallicity → the joint CIC kernel apply uses (#964), which
        # spreads each mass parcel over the metallicity axis with the MDF
        # centered on that parcel's own Z. It normalizes internally.
        if lgmet_on_ssp_ages is not None:
            return (
                *_joint_weights_cic_met_table(
                    _fine_age_yr,
                    _fine_sfr,
                    ssp_ages_yr,
                    t_obs_gyr,
                    lgmet_on_ssp_ages,
                    lgmet_scatter,
                    ssp.ssp_lgmet,
                ),
                ssp_ages_yr,
            )

        age_w_cic, total_mass = _age_weights_cic(_fine_age_yr, _fine_sfr, ssp_ages_yr, t_obs_gyr)
        lgmet_w = _lgmet_weights(log_z_abs_scalar, lgmet_scatter, ssp.ssp_lgmet)
        joint_weights = lgmet_w[:, None] * age_w_cic[None, :]
        joint_weights = joint_weights / jnp.maximum(joint_weights.sum(), 1e-300)
        return joint_weights, total_mass, ssp_ages_yr

    def compute_log_nion(self, params, ssp_data=None):
        r"""SED-free log-domain ionizing photon rate — no full-wavelength SED.

        Log-domain variant of :meth:`compute_nion` that returns log10(Q_H) instead
        of Q_H. Reconstructs the stellar intrinsic SED on the **ionizing slice only**
        (:math:`\lambda` below ~2x the Lyman limit) from the SED-free CSP weights
        (:meth:`compute_joint_weights`) and integrates via the shared
        :func:`_integrate_nion_log10`. Matches the ``log_nion`` that :meth:`apply`
        publishes (both use the same log-domain integral).

        Parameters
        ----------
        params : Mapping
            Free-parameter dict (same shape as :meth:`apply`).
        ssp_data : SSPData, optional
            Override for the model's SSP grid.

        Returns
        -------
        ndarray, shape ()
            Log10 ionizing photon rate [dex relative to photons/s]. Raises (via
            :meth:`compute_joint_weights`) for unsupported SFH / metallicity.

        Notes
        -----
        **JIT-compatible**: yes. The ionizing-slice mask is static (SSP wave grid
        is concrete), so the slice shape is fixed under jit.
        """
        ssp = ssp_data if ssp_data is not None else self.ssp_data
        joint_weights, total_mass, _ = self.compute_joint_weights(params, ssp_data=ssp)
        wave = ssp.ssp_wave
        # Ionizing slice + a buffer so the boundary bin's first non-ionizing
        # point is included; the non-ionizing region contributes zero to the
        # Lyman mask, so slicing is bit-exact with the full-grid integral.
        #
        # The region λ < 2·912Å is a PREFIX of the ascending ``ssp_wave``, so its
        # length is a concrete structural constant. Take a STATIC slice rather
        # than a boolean mask: ``ssp.ssp_flux[:, :, mask]`` raises
        # NonConcreteBooleanIndexError when ``ssp_flux`` is a traced jit input
        # (the fast nebular line path differentiates through this), whereas a
        # static-length slice compiles cleanly. Prefer the build-time static bound
        # (``_state.n_ion_bins``) — it is jit-safe even when ``wave`` is a traced
        # jit input; ``int(jnp.sum(...))`` only works when ``wave`` is concrete
        # (eager) and is the fallback for a component with no precompute state.
        if self._state is not None and self._state.n_ion_bins is not None:
            n_ion = self._state.n_ion_bins
        else:
            n_ion = int(jnp.sum(wave < (2.0 * _HI_LIMIT_AA)))

        if n_ion == 0:
            # No grid bins below the Lyman limit (IR-focused configs)
            # → Q_H is identically zero. Skips the slice machinery to avoid
            # max/argmax over zero-size arrays (#1193 fallout, #1207 fix).
            return jnp.full((), -jnp.inf)

        sed_ion = jnp.tensordot(joint_weights, ssp.ssp_flux[:, :, :n_ion], axes=([0, 1], [0, 1]))
        log10_scale = jnp.log10(total_mass.astype(jnp.result_type(float))) + jnp.log10(
            LSUN_ERG_PER_S
        )
        return _integrate_nion_log10(sed_ion, wave[:n_ion], log10_scale=log10_scale)

    def compute_nion(self, params, ssp_data=None):
        r"""SED-free ionizing photon rate :math:`Q_H` — no full-wavelength SED.

        Thin wrapper around :meth:`compute_log_nion` that exponentiates the
        log-domain result. Reconstructs the stellar intrinsic SED on the
        **ionizing slice only** (:math:`\lambda` below ~2x the Lyman limit)
        from the SED-free CSP weights (:meth:`compute_joint_weights`) and
        integrates via the log-domain core. Matches the ``nion`` that
        :meth:`apply` publishes without the ~6000-wavelength einsum.

        Parameters
        ----------
        params : Mapping
            Free-parameter dict (same shape as :meth:`apply`).
        ssp_data : SSPData, optional
            Override for the model's SSP grid.

        Returns
        -------
        ndarray, shape ()
            Ionizing photon rate [photons/s]. Raises (via
            :meth:`compute_joint_weights`) for unsupported SFH / metallicity.

        Notes
        -----
        **JIT-compatible**: yes. The ionizing-slice mask is static (SSP wave grid
        is concrete), so the slice shape is fixed under jit.
        """
        return pow10(self.compute_log_nion(params, ssp_data=ssp_data))


def _time_weighted_sfr(
    sfr_history: jnp.ndarray,
    sfh_lbt_grid: jnp.ndarray,
    window_yr: float,
) -> jnp.ndarray:
    """Time-weighted SFR over the last ``window_yr`` years.

    Thin wrapper around the canonical helper in
    :mod:`tengri.components.stellar.sfh.sfr_window`. Kept for
    StellarSEDComponent's existing call sites; new code should import
    :func:`time_weighted_sfr` from there directly.
    """
    from tengri.components.stellar.sfh.sfr_window import time_weighted_sfr

    return time_weighted_sfr(sfr_history, sfh_lbt_grid, window_yr)


# ─────────────────────────────────────────────────────────────────────
# JAX pytree registration
# ─────────────────────────────────────────────────────────────────────
#
# Register StellarSEDComponent as a JAX pytree so ``self.ssp_data``
# flows through ``jax.jit`` as a TRACED input rather than being baked
# into the XLA graph as a literal constant. The SSP grid is ~8 MB
# (15 × 93 × 5994 doubles); without this registration the cold-compile
# time explodes to ~900 ms because XLA inlines the entire grid as constants
# at every call site. With registration cold-compile drops by an
# order of magnitude.
#
# ``ssp_data`` is the only data field (it's a JAX-pytree-compatible
# NamedTuple with ndarray leaves). Everything else is structural
# (config, name, parameter_prefix) → meta.

from jax import tree_util as _tree_util

_tree_util.register_dataclass(
    StellarSEDComponent,
    data_fields=("ssp_data",),
    meta_fields=("config", "name", "parameter_prefix", "_state"),
)

del _tree_util


# ─────────────────────────────────────────────────────────────────────
# Property registration (Phase 1A + Phase 1B)
# ─────────────────────────────────────────────────────────────────────
#
# StellarSEDComponent does NOT inherit from SEDModelComponent, so
# __init_subclass__ auto-collection is unavailable. Properties are
# registered manually at module initialization.

_TINY = 1e-30  # Floor for safe division

# ─ Phase 1A: SFH group ─


def _stellar_mass_fn(state, params):
    """Total stellar mass currently alive [Msun]."""
    log_mstar_formed = jnp.asarray(state.derived["log_mstar_formed"])
    return jnp.power(10.0, log_mstar_formed)


def _stellar_mass_surviving_fn(state, params):
    """Total surviving stellar mass [Msun]."""
    log_mstar = jnp.asarray(state.derived["log_mstar_surviving"])
    return jnp.power(10.0, log_mstar)


def _sfr_100myr_fn(state, params):
    """Star formation rate averaged over past 100 Myr [Msun/yr]."""
    return jnp.asarray(state.derived["sfr_100myr"])


def _sfr_10myr_fn(state, params):
    """Star formation rate averaged over past 10 Myr [Msun/yr]."""
    return jnp.asarray(state.derived["sfr_10myr"])


def _ssfr_fn(state, params):
    """Specific star formation rate (SFR / surviving stellar mass) [1/yr].

    Notes
    -----
    Reads ``log_mstar``, **not** ``log_mstar_surviving`` — and that asymmetry with
    :func:`_stellar_mass_surviving_fn` is deliberate, not an oversight. When the
    SSP grid carries no mass-remaining table, "how much mass survives" has no
    answer and the property says NaN; but sSFR is still a meaningful number
    against the formed mass, so it falls back rather than going dark. That is
    exactly what the method this replaces did::

        mass_for_ssfr = jnp.where(jnp.isnan(mass_surviving), mass_formed, mass_surviving)

    and the deprecation shim must stay bit-exact with it (#1049, contract §6).
    """
    log_mstar = jnp.asarray(state.derived["log_mstar"])
    stellar_mass_surviving = jnp.power(10.0, log_mstar)
    sfr_100myr = jnp.asarray(state.derived["sfr_100myr"])
    return sfr_100myr / jnp.maximum(stellar_mass_surviving, _TINY)


def _mass_weighted_age_gyr_fn(state, params):
    r"""Mass-weighted mean age of the stellar population [Gyr].

    .. math::

        t_\mathrm{mw} = \frac{\sum_i w_i\, t_i}{\sum_i w_i}

    :math:`w_i` are the CSP mass weights per SSP age bin [Msun] and :math:`t_i`
    the SSP isochrone ages [yr].

    Notes
    -----
    **JIT-compatible**: yes.

    Weighted on the **SSP age grid** — the stars that actually exist in the SED —
    not by integrating the raw SFH grid. The two are not equivalent: an SFH can
    place stellar mass at lookback times beyond the age of the universe at the
    model's redshift (the orchestrator already warns when it does), and the SED
    truncates that mass while a raw-SFH integral would still count it. Weighting
    the population that the SED actually contains keeps this quantity consistent
    with the spectrum it accompanies; the two definitions differed by ~4.6% and
    were both live under this one name until #1131.

    Shares :func:`~tengri.utils.sed_quantities.compute_mass_weighted_age` with
    ``predict_sfh_quantities`` and ``Prediction.sfh`` — one implementation, so
    they cannot drift apart again.
    """
    from tengri.utils.sed_quantities import compute_mass_weighted_age

    weights = jnp.asarray(state.derived["age_weights"])
    ssp_ages_yr = jnp.asarray(state.derived["ssp_ages_yr"])
    return compute_mass_weighted_age(weights, ssp_ages_yr)


def _mass_weighted_metallicity_fn(state, params):
    """Mass-weighted mean metallicity (log10 Z/Zsun) [dex]."""
    sfh_lbt = jnp.asarray(state.derived["sfh_grid_lbt_yr"])
    sfr_history = jnp.asarray(state.derived["sfr_history"])
    log_z_history = jnp.asarray(state.derived["log_metallicity_history"])
    bin_widths = jnp.gradient(sfh_lbt)
    bin_mass = jnp.maximum(sfr_history * bin_widths, 0.0)
    bin_mass_total = jnp.maximum(jnp.sum(bin_mass), _TINY)
    return jnp.sum(log_z_history * bin_mass) / bin_mass_total


# ─ Phase 1B: SED group ─


def _l_bol_fn(state, params):
    """Bolometric luminosity [Lsun]."""
    from tengri.utils.sed_quantities import compute_bolometric_luminosity

    # Delegate to the canonical reduction rather than re-inlining it: that helper
    # folds 1/L_sun into the integral so the ~1e43 erg/s value is never formed
    # (float32-safe, #1206). ``abs`` preserves the original sign convention —
    # wave ascends, so nu descends and the signed area is negative.
    return jnp.abs(compute_bolometric_luminosity(state.sed_intrinsic, state.wave))


def _l_tir_fn(state, params):
    """Infrared luminosity (8–1000 µm) [Lsun]."""
    from tengri.utils.sed_quantities import compute_l_tir

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_l_tir(sed, wave)


def _l_dust_absorbed_fn(state, params):
    """Dust-absorbed luminosity [Lsun]."""
    from tengri.utils.physics_constants import L_SUN

    l_absorbed = jnp.asarray(state.derived.get("L_absorbed", 0.0))
    return l_absorbed / L_SUN


def _irx_fn(state, params):
    r"""Infrared excess against the **monochromatic 1600 A** UV luminosity [dex].

    .. math::

        \mathrm{IRX} = \log_{10}\!\left(
            \frac{L_\mathrm{TIR}}{(\nu L_\nu)_{1600\,\mathrm{A}}}\right)

    :math:`L_\mathrm{TIR}` is the 8-1000 um dust luminosity [erg/s] and
    :math:`(\nu L_\nu)_{1600}` the monochromatic UV luminosity at rest-frame
    1600 A [erg/s]. This is the anchor of the IRX-beta relation (Meurer et al.
    1999 [1]_) and the definition tengri has reported all along.

    See :func:`_irx_fuv_fn` for the band-averaged FUV variant — the two anchors
    differ by ~0.12 dex and are not interchangeable.

    Notes
    -----
    **JIT-compatible**: yes.

    References
    ----------
    .. [1] Meurer, G. R., Heckman, T. M., & Calzetti, D. 1999, "Dust Absorption
       and the Ultraviolet Luminosity Density at z ~ 3 as Calibrated by Local
       Starburst Galaxies", ApJ, 521, 64. doi:10.1086/307523
    """
    from tengri.utils.sed_quantities import compute_irx, compute_l_tir, compute_uv_luminosity_1600

    sed = state.sed_intrinsic
    wave = state.wave
    l_tir = compute_l_tir(sed, wave)
    l_uv = compute_uv_luminosity_1600(sed, wave)
    return compute_irx(l_tir, l_uv)


def _irx_fuv_fn(state, params):
    r"""Infrared excess against the **band-averaged FUV** luminosity [dex].

    .. math::

        \mathrm{IRX_{FUV}} = \log_{10}\!\left(
            \frac{L_\mathrm{TIR}}{\nu_{1500}\,\langle L_\nu\rangle_\mathrm{FUV}}\right)

    :math:`\langle L_\nu \rangle_\mathrm{FUV}` is the mean :math:`L_\nu` over
    1000-1700 A [erg/s/Hz] — a GALEX-FUV-like window rather than a single
    wavelength — and :math:`\nu_{1500} = c / 1500\,\mathrm{A}` the pivot
    frequency [Hz].

    Notes
    -----
    **JIT-compatible**: yes.

    The pivot frequency takes ``C_AA`` from
    :mod:`tengri.utils.physics_constants`. It previously used a hardcoded
    ``2.998e15`` — the speed of light 1000x too small in [A/s] — which inflated
    every reported IRX by exactly :math:`\log_{10}(1000) = 3` dex (#1131).
    """
    from tengri.utils.physics_constants import C_AA
    from tengri.utils.sed_quantities import compute_fuv_flux, compute_irx, compute_l_tir

    sed = state.sed_intrinsic
    wave = state.wave
    l_tir = compute_l_tir(sed, wave)
    fuv = compute_fuv_flux(sed, wave)
    return compute_irx(l_tir, fuv * C_AA / 1500.0)


def _uv_slope_beta_fn(state, params):
    """UV slope (β in L_ν ∝ ν^β) [dimensionless]."""
    from tengri.utils.sed_quantities import compute_uv_slope_beta

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_uv_slope_beta(sed, wave)


def _dn4000_fn(state, params):
    """D n4000 break diagnostic [dimensionless]."""
    from tengri.utils.sed_quantities import compute_dn4000

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_dn4000(sed, wave)


def _balmer_break_fn(state, params):
    """Balmer break diagnostic [dimensionless]."""
    from tengri.utils.sed_quantities import compute_balmer_break

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_balmer_break(sed, wave)


def _m_uv_fn(state, params):
    """UV absolute magnitude (1600 Å) [AB mag]."""
    from tengri.utils.sed_quantities import compute_m_uv

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_m_uv(sed, wave)


def _fuv_flux_fn(state, params):
    """FUV flux (1000–1700 Å) [erg/s/Hz]."""
    from tengri.utils.sed_quantities import compute_fuv_flux

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_fuv_flux(sed, wave)


def _nuv_flux_fn(state, params):
    """NUV flux (1700–3000 Å) [erg/s/Hz]."""
    from tengri.utils.sed_quantities import compute_nuv_flux

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_nuv_flux(sed, wave)


def _fuv_flux_intrinsic_fn(state, params):
    """Intrinsic FUV flux before dust attenuation [erg/s/Hz]."""
    from tengri.utils.sed_quantities import compute_fuv_flux

    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)
    if "lnu_age" in derived:
        sed_stellar_intrinsic = jnp.sum(jnp.asarray(derived["lnu_age"]), axis=0)
        wave = state.wave
        return compute_fuv_flux(sed_stellar_intrinsic, wave)
    else:
        return nan_scalar


def _nuv_flux_intrinsic_fn(state, params):
    """Intrinsic NUV flux before dust attenuation [erg/s/Hz]."""
    from tengri.utils.sed_quantities import compute_nuv_flux

    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)
    if "lnu_age" in derived:
        sed_stellar_intrinsic = jnp.sum(jnp.asarray(derived["lnu_age"]), axis=0)
        wave = state.wave
        return compute_nuv_flux(sed_stellar_intrinsic, wave)
    else:
        return nan_scalar


def _rest_uv_color_fn(state, params):
    """Rest-frame UV color (FUV–NUV) [AB mag]."""
    from tengri.utils.sed_quantities import compute_rest_uv_color

    sed = state.sed_intrinsic
    wave = state.wave
    return compute_rest_uv_color(sed, wave)


# ─ Phase 1B: Luminosity-weighted SFH properties ─


def _luminosity_weighted_age_gyr_fn(state, params):
    """Luminosity-weighted mean age [Gyr]."""
    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)

    if "L_age" in derived and "ssp_ages_yr" in derived:
        L_age = jnp.asarray(derived["L_age"])
        ssp_ages_yr = jnp.asarray(derived["ssp_ages_yr"])
        L_total = jnp.maximum(jnp.sum(L_age), _TINY)
        lw_age_yr = jnp.sum(ssp_ages_yr * L_age) / L_total
        return lw_age_yr / 1e9
    else:
        return nan_scalar


def _luminosity_weighted_metallicity_fn(state, params):
    """Luminosity-weighted mean metallicity [dex, log10(Z/Zsun)]."""
    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)

    if "L_age" in derived and "ssp_ages_yr" in derived:
        L_age = jnp.asarray(derived["L_age"])
        ssp_ages_yr = jnp.asarray(derived["ssp_ages_yr"])
        L_total = jnp.maximum(jnp.sum(L_age), _TINY)

        if "log_metallicity_history" in derived and "sfh_grid_lbt_yr" in derived:
            lz_per_ssp = jnp.interp(
                ssp_ages_yr,
                jnp.asarray(derived["sfh_grid_lbt_yr"]),
                jnp.asarray(derived["log_metallicity_history"]),
            )
            return jnp.sum(lz_per_ssp * L_age) / L_total
        else:
            return nan_scalar
    else:
        return nan_scalar


# ─ Phase 1B: Ionizing group ─


def _q_h_fn(state, params):
    """Ionizing photon production rate [photons/s]."""
    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)
    return jnp.asarray(derived.get("nion", nan_scalar))


def _log_q_h_fn(state, params):
    """log10 ionizing photon production rate [dex re photons/s] — float32-safe."""
    derived = state.derived
    log_nion = derived.get("log_nion")
    return jnp.asarray(log_nion) if log_nion is not None else jnp.asarray(jnp.nan)


def _xi_ion_fn(state, params):
    """Ionizing photon efficiency [Hz/erg]."""
    from tengri.utils.sed_quantities import compute_xi_ion_from_log_qh

    derived = state.derived
    nan_scalar = jnp.asarray(jnp.nan)
    sed = state.sed_intrinsic
    if sed is None:
        return nan_scalar
    else:
        log_nion = jnp.asarray(derived.get("log_nion", -jnp.inf))
        return compute_xi_ion_from_log_qh(log_nion, sed, state.wave)


# Register properties in the global registry
from tengri.forward.properties import Property, register_properties

_SFH_PROPERTIES = {
    "stellar_mass": Property(
        units="Msun",
        group="sfh",
        doc="Total stellar mass currently alive",
        fn=_stellar_mass_fn,
    ),
    "stellar_mass_surviving": Property(
        units="Msun",
        group="sfh",
        doc="Total surviving stellar mass",
        fn=_stellar_mass_surviving_fn,
    ),
    "sfr_100myr": Property(
        units="Msun/yr",
        group="sfh",
        doc="Star formation rate averaged over past 100 Myr",
        fn=_sfr_100myr_fn,
    ),
    "sfr_10myr": Property(
        units="Msun/yr",
        group="sfh",
        doc="Star formation rate averaged over past 10 Myr",
        fn=_sfr_10myr_fn,
    ),
    "ssfr": Property(
        units="1/yr",
        group="sfh",
        doc="Specific star formation rate (SFR / stellar_mass)",
        fn=_ssfr_fn,
    ),
    "mass_weighted_age_gyr": Property(
        units="Gyr",
        group="sfh",
        doc="Mass-weighted mean age of stellar population",
        fn=_mass_weighted_age_gyr_fn,
    ),
    "mass_weighted_metallicity": Property(
        units="dex",
        group="sfh",
        doc="Mass-weighted mean metallicity (log10 Z/Zsun)",
        fn=_mass_weighted_metallicity_fn,
    ),
    "luminosity_weighted_age_gyr": Property(
        units="Gyr",
        group="sfh",
        doc="Luminosity-weighted mean age of stellar population",
        fn=_luminosity_weighted_age_gyr_fn,
    ),
    "luminosity_weighted_metallicity": Property(
        units="dex",
        group="sfh",
        doc="Luminosity-weighted mean metallicity (log10 Z/Zsun)",
        fn=_luminosity_weighted_metallicity_fn,
    ),
}

_SED_PROPERTIES = {
    "l_bol": Property(
        units="Lsun",
        group="sed",
        doc="Bolometric luminosity",
        fn=_l_bol_fn,
    ),
    "l_tir": Property(
        units="Lsun",
        group="sed",
        doc="Infrared luminosity (8–1000 µm)",
        fn=_l_tir_fn,
    ),
    "l_dust_absorbed": Property(
        units="Lsun",
        group="sed",
        doc="Dust-absorbed luminosity",
        fn=_l_dust_absorbed_fn,
    ),
    "irx": Property(
        units="dex",
        group="sed",
        doc="Infrared excess, log10(L_TIR / nu*L_nu at 1600 A) — the Meurer+99 IRX-beta anchor",
        fn=_irx_fn,
    ),
    "irx_fuv": Property(
        units="dex",
        group="sed",
        doc="Infrared excess against the band-averaged FUV (1000-1700 A), pivoted at 1500 A",
        fn=_irx_fuv_fn,
    ),
    "uv_slope_beta": Property(
        units="",
        group="sed",
        doc="UV slope (β in L_ν ∝ ν^β)",
        fn=_uv_slope_beta_fn,
    ),
    "dn4000": Property(
        units="",
        group="sed",
        doc="D n4000 break diagnostic",
        fn=_dn4000_fn,
    ),
    "balmer_break": Property(
        units="",
        group="sed",
        doc="Balmer break diagnostic",
        fn=_balmer_break_fn,
    ),
    "m_uv": Property(
        units="AB mag",
        group="sed",
        doc="UV absolute magnitude (1600 Å)",
        fn=_m_uv_fn,
    ),
    "fuv_flux": Property(
        units="erg/s/Hz",
        group="sed",
        doc="FUV flux (1000–1700 Å)",
        fn=_fuv_flux_fn,
    ),
    "nuv_flux": Property(
        units="erg/s/Hz",
        group="sed",
        doc="NUV flux (1700–3000 Å)",
        fn=_nuv_flux_fn,
    ),
    "fuv_flux_intrinsic": Property(
        units="erg/s/Hz",
        group="sed",
        doc="Intrinsic FUV flux before dust attenuation",
        fn=_fuv_flux_intrinsic_fn,
    ),
    "nuv_flux_intrinsic": Property(
        units="erg/s/Hz",
        group="sed",
        doc="Intrinsic NUV flux before dust attenuation",
        fn=_nuv_flux_intrinsic_fn,
    ),
    "rest_uv_color": Property(
        units="AB mag",
        group="sed",
        doc="Rest-frame UV color (FUV–NUV)",
        fn=_rest_uv_color_fn,
    ),
}

_IONIZING_PROPERTIES = {
    "q_h": Property(
        units="photons/s",
        group="ionizing",
        doc="Ionizing photon production rate",
        fn=_q_h_fn,
    ),
    "log_q_h": Property(
        units="dex",
        group="ionizing",
        doc="log10(ionizing photon production rate / (photons/s)) — float32-safe form of q_h",
        fn=_log_q_h_fn,
    ),
    "xi_ion": Property(
        units="Hz/erg",
        group="ionizing",
        doc="Ionizing photon efficiency",
        fn=_xi_ion_fn,
    ),
}

register_properties("stellar", _SFH_PROPERTIES)
register_properties("stellar", _SED_PROPERTIES)
register_properties("stellar", _IONIZING_PROPERTIES)

del (
    Property,
    register_properties,
    _SFH_PROPERTIES,
    _SED_PROPERTIES,
    _IONIZING_PROPERTIES,
)

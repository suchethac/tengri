# SPDX-License-Identifier: BSD-3-Clause
"""Metallicity-indexed L_line/Q_H table — the FeaturePrecomp line channel (#950).

Cue's emulator makes each nebular line luminosity **exactly linear in the
hydrogen-ionizing photon rate** Q_H (``L_line = Q_H \\cdot \\ell(\\theta)``; the
per-Q_H factor ``\\ell`` is the network's ``Lsun/Q_H`` output, cue.py). The
per-Q_H factor depends on the *shape* of the ionizing spectrum (set by stellar
metallicity) and the fixed gas conditions (logU, logZ_gas, fesc) — **not on the
star-formation-history shape**: the SFH enters lines only through the scalar
Q_H (validated to CV = 0 % across SFH draws).

So a joint fit that varies the SFH can skip Cue's ~3 ms neural forward every
evaluation: precompute ``\\ell`` once on a dense stellar-metallicity grid, then

    L_line(params) = nion(params) \\cdot interp_met(ell, met_logzsol)

where ``nion`` is the stellar-published ionizing rate (``nion == q_h``). The
stellar metallicity enters Cue **nonlinearly** (via the ionizing-spectrum
shape), so a coarse SSP-grid interpolation is not enough (1-60 % line errors);
a dense grid (~40 points) with linear interpolation reaches < 4e-4 on the
strong DESI lines — three orders of magnitude below the measurement floor.

The table is built in **observed-flux space** (``predict_line_fluxes`` output)
so reconstruction matches that method bit-for-bit at fixed redshift; the
cosmological ``1/(4 pi d_L^2)`` factor is identical between build and eval and
cancels in ``flux_ref / nion_ref``.

Requires FIXED nebular ionization (logU, logZ_gas, fesc); ``met_logzsol`` may be
free (it is the LUT axis). See issue #950.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class LinePerQHTable:
    """Dense metallicity grid of per-Q_H line fluxes for the FeaturePrecomp line path.

    Attributes
    ----------
    met_grid : ndarray, shape (n_met,)
        ``met_logzsol`` grid points [dex], ascending.
    line_per_qh : ndarray, shape (n_met, n_lines)
        Observed line flux per unit ``nion`` at each grid metallicity —
        ``predict_line_fluxes(ref) / nion(ref)`` [erg/s/cm^2 per (photons/s)].
    wavelengths : ndarray, shape (n_lines,)
        Rest-frame vacuum line wavelengths [Angstrom], matching the target
        lines the table was built for.
    """

    met_grid: jnp.ndarray = dataclasses.field()
    line_per_qh: jnp.ndarray = dataclasses.field()
    wavelengths: jnp.ndarray = dataclasses.field()


def _nion_of_state(state) -> jnp.ndarray:
    """Total ionizing photon rate published by the stellar component."""
    nion = state.derived["nion"]
    return jnp.sum(nion) if jnp.ndim(nion) else nion


def precompute_line_per_qh(
    model,
    wavelengths,
    *,
    met_lo: float = -1.8,
    met_hi: float = 0.4,
    n_met: int = 40,
    ref_params: dict | None = None,
) -> LinePerQHTable:
    """Build the metallicity-indexed ``L_line / nion`` table.

    Evaluates the exact nebular forward at a fixed reference SFH across a dense
    ``met_logzsol`` grid. Because ``line_per_qh`` is SFH-shape-independent, the
    reference SFH is arbitrary; the table depends only on ``met_logzsol`` (and
    the model's fixed nebular ionization).

    Parameters
    ----------
    model : SEDModel
        A model with a Cue (or other Q_H-linear) nebular backend and FIXED
        ``neb_logU`` / ``neb_logZ_gas`` / ``neb_fesc``.
    wavelengths : array_like, shape (n_lines,)
        Rest-frame vacuum target line wavelengths [Angstrom].
    met_lo, met_hi : float
        Grid bounds in ``met_logzsol`` [dex] — cover the fit's metallicity prior.
    n_met : int, default 40
        Grid points. 40 gives < 4e-4 on strong DESI lines; raise for tighter.
    ref_params : dict, optional
        Reference parameter dict (the metallicity is overwritten per grid
        point). Defaults to a mid-range single sample.

    Returns
    -------
    LinePerQHTable
        The dense-met table for :func:`reconstruct_line_lums`.

    Notes
    -----
    **Build cost**: ``n_met`` forward evaluations, once at construction.
    Not JIT'd (a build-time loop over concrete metallicities).
    """
    wavelengths = jnp.asarray(wavelengths)
    if ref_params is None:
        ref_params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    else:
        ref_params = dict(ref_params)

    met_grid = jnp.linspace(met_lo, met_hi, n_met)
    rows = []
    for mz in met_grid:
        p = dict(ref_params)
        p["met_logzsol"] = jnp.asarray(float(mz))
        flux = model.predict_line_fluxes(p, target_wavelengths=wavelengths)
        nion = _nion_of_state(model.predict_state(p))
        rows.append(jnp.asarray(flux) / jnp.maximum(nion, 1e-30))
    return LinePerQHTable(
        met_grid=met_grid,
        line_per_qh=jnp.stack(rows),  # (n_met, n_lines)
        wavelengths=wavelengths,
    )


def reconstruct_line_lums(
    nion: jnp.ndarray, met_logzsol: jnp.ndarray, table: LinePerQHTable
) -> jnp.ndarray:
    """Reconstruct observed line fluxes from the table without a Cue forward.

    ``L_line = nion * interp_met(line_per_qh, met_logzsol)``.

    Parameters
    ----------
    nion : float
        Ionizing photon rate for this evaluation (stellar-published; == q_h).
    met_logzsol : float
        Stellar metallicity for this evaluation [dex].
    table : LinePerQHTable
        The dense-met table from :func:`precompute_line_per_qh`.

    Returns
    -------
    ndarray, shape (n_lines,)
        Observed line fluxes [erg/s/cm^2], matching ``predict_line_fluxes`` to
        < 4e-4 on strong lines.

    Notes
    -----
    **JIT-compatible**: yes — ``jnp.interp`` + a scalar multiply. This is the
    per-evaluation hot path that replaces the ~3 ms Cue neural forward.
    """
    mz = jnp.asarray(met_logzsol)
    # per-line linear interpolation across the metallicity grid
    lpq = jax.vmap(lambda col: jnp.interp(mz, table.met_grid, col), in_axes=1)(table.line_per_qh)
    return jnp.asarray(nion) * lpq

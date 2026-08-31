# SPDX-License-Identifier: BSD-3-Clause
"""Metallicity-indexed L_line/Q_H table: the FeaturePrecomp line channel (#950).

Cue's emulator makes each nebular line luminosity **exactly linear in the
hydrogen-ionizing photon rate** Q_H (``L_line = Q_H \\cdot \\ell(\\theta)``; the
per-Q_H factor ``\\ell`` is the network's ``Lsun/Q_H`` output, cue.py). The
per-Q_H factor depends on the *shape* of the ionizing spectrum (set by stellar
metallicity) and the fixed gas conditions (logU, logZ_gas, fesc): **not on the
star-formation-history shape**: the SFH enters lines only through the scalar
Q_H (validated to CV = 0 % across SFH draws).

The reconstruction is

    F_line(params) = nion(params) * interp_met(ell, met_logzsol) / (4 pi d_L(z)^2)

where ``nion`` is the stellar-published ionizing rate (``nion == q_h``,
independently verified). The stored ``ell`` is a **luminosity** per Q_H
(distance-independent), so the cosmology is applied at evaluation with the
**evaluation** redshift: the table is valid at any (per-galaxy or free)
redshift. The stellar metallicity enters Cue **nonlinearly** (via the
ionizing-spectrum shape), so a coarse SSP-grid interpolation is not enough
(1-60 % line errors); a dense grid (~40 points) with linear interpolation
reaches < 4e-4 on the strong DESI lines.

Requires fixed nebular ionization (logU, logZ_gas, fesc): guarded at build;
``met_logzsol`` may be free (it is the LUT axis). See issue #950.

.. warning::

    **Not wired into the forward, and not a performance win as of #949.** This
    module is a *validated physics record* (line-Q_H linearity, SFH-shape
    independence, metallicity nonlinearity). After #949 made the objective
    genuinely JIT-compiled, the Cue line forward is ~0.5 ms (not the ~85 ms
    un-JIT'd figure #950 was scoped against), so ``reconstruct_line_lums`` is
    NOT faster than running Cue. The expensive joint-fit channel is spectral
    **indices** (Dn4000 forces the full-grid SED), not lines; see the
    index-window LUT (#949) and the #950 benchmark comments.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from tengri.components.stellar.reference_history import reference_history_params
from tengri.utils.scale import apply_log10_scale

#: Nebular ionization parameters that MUST be fixed for the table to be valid:
#: they change ``line_per_qh`` (line ratios), so a free one would make the
#: single-metallicity-axis table wrong away from its baked reference value.
_REQUIRED_FIXED = ("neb_logU", "neb_logZ_gas", "neb_fesc")


@dataclasses.dataclass(frozen=True)
class LinePerQHTable:
    """Dense metallicity grid of per-Q_H line **luminosities** for the line path.

    Attributes
    ----------
    met_grid : ndarray, shape (n_met,)
        ``met_logzsol`` grid points [dex], ascending.
    line_per_qh : ndarray, shape (n_met, n_lines)
        Line **luminosity** per unit ``nion`` at each grid metallicity:
        ``L_line(ref) / nion(ref)`` [erg/s per (photons/s)]. **Distance-
        independent** (luminosity, not observed flux) so the table is valid at
        any redshift; :func:`reconstruct_line_lums` applies the cosmology at the
        evaluation redshift. (Storing observed flux here would bake the
        reference distance and be silently wrong at any other z.)
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


def _log10_four_pi_dl2(redshift) -> jnp.ndarray:
    """log10(4 pi d_L(z)^2) [dex]: the line luminosity → observed flux divisor.

    Log, not linear: the divisor is ~1e57 and ``inf`` in float32 at every
    distance, so it is applied with :func:`~tengri.utils.scale.apply_log10_scale`
    rather than materialized (#1859).
    """
    from tengri.cosmology import luminosity_distance
    from tengri.utils.scale import log10_four_pi_dl2

    dl_cm = jnp.asarray(luminosity_distance(jnp.asarray(redshift))).reshape(())
    return log10_four_pi_dl2(dl_cm)


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
    ``met_logzsol`` grid. ``line_per_qh`` is SFH-shape-independent only to ~0.2 %
    (#1018), so the reference SFH is nearly but not exactly arbitrary; the table
    depends (to that accuracy) only on ``met_logzsol`` (and
    the model's fixed nebular ionization).

    Parameters
    ----------
    model : SEDModel
        A model with a Cue (or other Q_H-linear) nebular backend and fixed
        ``neb_logU`` / ``neb_logZ_gas`` / ``neb_fesc``.
    wavelengths : array_like, shape (n_lines,)
        Rest-frame vacuum target line wavelengths [Angstrom].
    met_lo, met_hi : float
        Grid bounds in ``met_logzsol`` [dex]: cover the fit's metallicity prior.
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
    # Guard the "fixed ionization" precondition: a free logU/logZ_gas/fesc
    # changes line_per_qh (line ratios), so the single-metallicity-axis table
    # would be silently wrong away from the baked reference value. Raise rather
    # than let the caller build an invalid table.
    free_ion = [p for p in _REQUIRED_FIXED if p in set(model.spec.free_params)]
    if free_ion:
        raise ValueError(
            f"precompute_line_per_qh requires fixed nebular ionization, but "
            f"{free_ion} are free. The table has a single metallicity axis; a "
            f"free ionization parameter changes line ratios and would make "
            f"reconstruction wrong away from its reference value. Fix these "
            f"parameters, or extend the table with extra axes (see #950)."
        )

    wavelengths = jnp.asarray(wavelengths)
    if ref_params is None:
        ref_params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    else:
        ref_params = dict(ref_params)

    # Same #1718 gap as the grid builder: `spec.sample` cannot produce the
    # runtime arrays of a tabulated SFH, which declares no parameters. Legitimate
    # to stand in for, and for the reason this module already states below:
    # the table is per-Q_H, a property of the gas, not of the reference SFH.
    ref_params = {
        **reference_history_params(model, redshift=ref_params.get("redshift", 0.0)),
        **ref_params,
    }

    # Recover distance-independent LUMINOSITY: predict_line_fluxes returns
    # observed flux L / (4 pi d_L(z_ref)^2); multiply by the reference divisor
    # so the stored table carries L_line / nion, valid at any evaluation z.
    ref_z = ref_params.get("redshift", 0.0)
    log10_ref_divisor = _log10_four_pi_dl2(ref_z)

    met_grid = jnp.linspace(met_lo, met_hi, n_met)
    rows = []
    for mz in met_grid:
        p = dict(ref_params)
        p["met_logzsol"] = jnp.asarray(float(mz))
        # INTRINSIC line-per-Q_H (redden=False): the table is the nebular line
        # luminosity per ionizing photon: a property of the gas, independent of
        # the reference SFH *and* the reference dust. Dust reddening (which now
        # defaults on in predict_line_fluxes) is applied downstream, not baked in.
        flux = model.predict_line_fluxes(p, target_wavelengths=wavelengths, redden=False)
        nion = _nion_of_state(model.predict_state(p))
        # observed flux → line luminosity, without materializing the ~1e57 divisor
        lum = apply_log10_scale(jnp.asarray(flux), log10_ref_divisor)
        rows.append(lum / jnp.maximum(nion, 1e-30))
    return LinePerQHTable(
        met_grid=met_grid,
        line_per_qh=jnp.stack(rows),  # (n_met, n_lines): luminosity per Q_H
        wavelengths=wavelengths,
    )


def reconstruct_line_lums(
    nion: jnp.ndarray,
    met_logzsol: jnp.ndarray,
    redshift: jnp.ndarray,
    table: LinePerQHTable,
) -> jnp.ndarray:
    """Reconstruct observed line fluxes from the table without a Cue forward.

    .. math::

        F_{\\rm line} = \\frac{n_{\\rm ion}\\,
            \\mathrm{interp\\_met}(\\ell, Z_\\star)}{4\\pi\\,d_L(z)^2}

    where :math:`\\ell` is the stored luminosity-per-Q_H and :math:`d_L(z)` is
    the luminosity distance at the **evaluation** redshift: so the same table
    is correct at any (per-galaxy or free) redshift.

    Parameters
    ----------
    nion : float
        Ionizing photon rate for this evaluation (stellar-published; == q_h).
    met_logzsol : float
        Stellar metallicity for this evaluation [dex].
    redshift : float
        Evaluation redshift: the cosmology is applied here, NOT baked into the
        table (that was the redshift-lock bug).
    table : LinePerQHTable
        The dense-met table from :func:`precompute_line_per_qh`.

    Returns
    -------
    ndarray, shape (n_lines,)
        Observed line fluxes [erg/s/cm^2], matching ``predict_line_fluxes`` to
        < 4e-4 on strong lines at the evaluation redshift.

    Notes
    -----
    **JIT-compatible**: yes, ``jnp.interp`` + cosmology + a scalar multiply.
    """
    mz = jnp.asarray(met_logzsol)
    # per-line linear interpolation across the metallicity grid → L_line / nion
    lpq = jax.vmap(lambda col: jnp.interp(mz, table.met_grid, col), in_axes=1)(table.line_per_qh)
    lum = jnp.asarray(nion) * lpq  # line luminosity [erg/s]
    # → observed flux at THIS redshift; the divisor stays an exponent (#1859)
    return apply_log10_scale(lum, -_log10_four_pi_dl2(redshift))

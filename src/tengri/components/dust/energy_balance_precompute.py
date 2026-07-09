# SPDX-License-Identifier: BSD-3-Clause
r"""Build-time LUT for the two-component dust energy balance (``L_ir``).

Under ``approx=WavePrecomp()`` the photometry is projected from per-filter
LUTs, so the full-wavelength stellar SED cube is normally dead-code-eliminated
by XLA. Enabling dust IR re-emission, however, makes ``L_ir`` — the
energy-balance absorbed luminosity — feed the output, and the exact
:func:`L_absorbed` integral is taken over the full ``(n_met, n_age, n_wave)``
stellar cube. That single dependency resurrects the cube and costs ~40× per
evaluation (30 µs → 1.2 ms on a photometry-only fit).

The integral factorizes. With per-age transmission ``T_a(λ)`` independent of the
SFH,

.. math::

    L_{\rm abs}^{\star} = M_\star \, L_\odot \sum_{m,a} w_{m,a}
        \left[ B_{m,a} - G_{m,a}(\tau_{\rm bc}, \tau_{\rm diff}) \right]

with

.. math::

    B_{m,a}      &= \int \mathrm{SSP}_{m,a}(\lambda)\, d\nu \\
    G_{m,a}(\boldsymbol\tau) &= \int \mathrm{SSP}_{m,a}(\lambda)\,
        T_a(\lambda;\boldsymbol\tau)\, d\nu

where :math:`w_{m,a}` are the runtime DSPS joint (metallicity, age) weights and
:math:`M_\star L_\odot` the runtime mass scaling. ``B`` and ``G`` depend only on
the fixed SSP grid, the (fixed-shape) attenuation curves, and the optical depths
:math:`(\tau_{\rm bc}, \tau_{\rm diff})`. They are precomputed once on a small
:math:`(\tau_{\rm bc}, \tau_{\rm diff})` grid; at runtime ``G`` is bilinearly
interpolated and contracted with the weights — no full-wavelength cube.

The spectral integral is held at full SSP resolution (so #622's far-IR exactness
is preserved); the only approximation is the smooth bilinear interpolation in
the two optical-depth axes, where :math:`\int \mathrm{SSP}\, e^{-\tau k}\, d\nu`
is monotone and well behaved.

This LUT is the precomputed factorization of the canonical energy-balance
integral :func:`tengri.forward.energy_balance.bolometric_absorbed` — same
signed :math:`\int (L_\nu^{\rm intr} - L_\nu^{\rm att})\, d\nu` with the same
912 Å Lyman-continuum mask (#922). The two must agree; the contract is pinned
by ``tests/contract/test_energy_balance_lut.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from tengri.components.dust.attenuation import two_component_dust
from tengri.utils.physics_constants import C_AA

__all__ = ["EnergyBalanceLUT", "build_energy_balance_lut", "lut_l_absorbed_stellar"]


class EnergyBalanceLUT(NamedTuple):
    """Precomputed bolometric absorption LUT for the stellar energy balance.

    Attributes
    ----------
    B : ndarray, shape (n_met, n_age)
        Intrinsic bolometric SSP luminosity per unit mass, ``∫ SSP dν`` (signed,
        masked to λ ≥ 912 Å). [erg/s/Hz · Hz per Lsun-flux unit]
    G : ndarray, shape (n_met, n_age, n_tau_bc, n_tau_diff)
        Attenuated bolometric SSP luminosity ``∫ SSP·T_a dν`` on the optical-depth
        grid.
    tau_bc_grid : ndarray, shape (n_tau_bc,)
        Birth-cloud optical-depth grid nodes.
    tau_diff_grid : ndarray, shape (n_tau_diff,)
        Diffuse-ISM optical-depth grid nodes.
    """

    B: jnp.ndarray
    G: jnp.ndarray
    tau_bc_grid: jnp.ndarray
    tau_diff_grid: jnp.ndarray


def _axis_grid(lo: float, hi: float, n: int) -> jnp.ndarray:
    """Grid nodes for one optical-depth axis; a single node when ``lo == hi``."""
    if hi <= lo:
        return jnp.asarray([lo])
    return jnp.linspace(lo, hi, n)


def build_energy_balance_lut(
    ssp_flux: jnp.ndarray,
    ssp_wave: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    *,
    law_bc: str,
    law_diff: str,
    f_obscuration: float = 0.0,
    t_birth_yr: float = 1e7,
    transition_width_dex: float = 0.3,
    bc_params: dict | None = None,
    diff_params: dict | None = None,
    lyman_cutoff_aa: float = 0.0,
    eb_include_lyc: bool = False,
    tau_bc_grid: jnp.ndarray,
    tau_diff_grid: jnp.ndarray,
) -> EnergyBalanceLUT:
    r"""Precompute ``B`` and ``G`` for the two-component energy balance.

    The transmission ``T_a(λ)`` is built with the *same*
    :func:`two_component_dust` the runtime path uses, so the LUT reproduces the
    exact spectral integral at every grid node.

    Parameters
    ----------
    ssp_flux : ndarray, shape (n_met, n_age, n_wave)
        SSP specific luminosity per unit mass [Lsun/Hz/Msun].
    ssp_wave : ndarray, shape (n_wave,)
        Rest-frame SSP wavelength grid [Å], ascending.
    ssp_ages_yr : ndarray, shape (n_age,)
        SSP age axis [yr].
    law_bc, law_diff : str
        Attenuation-law registry keys (fixed shape).
    f_obscuration, t_birth_yr, transition_width_dex, bc_params, diff_params,
    lyman_cutoff_aa
        Passed verbatim to :func:`two_component_dust` for node-exact agreement.
    eb_include_lyc : bool, optional
        FSPS-parity toggle (#961): when True, the LyC (λ < 912 Å) is kept in
        the absorbed-luminosity integrand — all absorbed energy heats dust —
        instead of the canonical LyC mask (#922). Must match the runtime
        ``DustSEDComponent.config.eb_include_lyc``.
    tau_bc_grid, tau_diff_grid : ndarray
        Optical-depth grid nodes (keyword-only).

    Returns
    -------
    EnergyBalanceLUT
    """
    nu = C_AA / ssp_wave  # (n_wave,)
    mask = jnp.ones_like(ssp_wave, dtype=bool) if eb_include_lyc else (ssp_wave >= 912.0)
    sspm = ssp_flux * mask[None, None, :]  # (n_met, n_age, n_wave)
    B = jnp.trapezoid(sspm, nu, axis=-1)  # (n_met, n_age), signed

    bc_params = bc_params or {}
    diff_params = diff_params or {}

    def g_at(tb, td):
        transmission = two_component_dust(
            wavelength=ssp_wave,
            age_grid=ssp_ages_yr,
            tau_v1=jnp.asarray(tb),
            tau_v2=jnp.asarray(td),
            law_bc=law_bc,
            law_diff=law_diff,
            f_obscuration=jnp.asarray(f_obscuration),
            t_birth=t_birth_yr,
            transition_width=transition_width_dex,
            bc_params={k: jnp.asarray(v) for k, v in bc_params.items()},
            diff_params={k: jnp.asarray(v) for k, v in diff_params.items()},
            lyman_cutoff_aa=lyman_cutoff_aa,
        )  # (n_age, n_wave)
        integrand = sspm * transmission[None, :, :]  # (n_met, n_age, n_wave)
        return jnp.trapezoid(integrand, nu, axis=-1)  # (n_met, n_age)

    # Build-time Python loop over the (small) optical-depth grid.
    G = jnp.stack(
        [jnp.stack([g_at(tb, td) for td in tau_diff_grid], axis=-1) for tb in tau_bc_grid],
        axis=-2,
    )  # (n_met, n_age, n_tau_bc, n_tau_diff)

    return EnergyBalanceLUT(
        B=B, G=G, tau_bc_grid=jnp.asarray(tau_bc_grid), tau_diff_grid=jnp.asarray(tau_diff_grid)
    )


def _interp_weight_vector(grid: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Linear-interpolation weights over *all* nodes of a uniform ascending grid.

    Returns a ``(n_nodes,)`` vector that is zero except on the two nodes
    bracketing ``x`` (carrying weights ``1 - f`` and ``f``). Using a dense
    weight vector + contraction — rather than a dynamic gather — keeps the
    downstream interpolation a plain tensor product that XLA does not try to
    constant-fold. A single-node grid returns ``[1.0]`` (no interpolation).
    """
    n = grid.shape[0]
    if n == 1:
        return jnp.ones((1,), dtype=grid.dtype)
    dx = grid[1] - grid[0]  # uniform linspace spacing
    return jnp.clip(1.0 - jnp.abs(x - grid) / dx, 0.0, 1.0)


def lut_l_absorbed_stellar(
    lut: EnergyBalanceLUT,
    joint_weights: jnp.ndarray,
    mass_scale: jnp.ndarray,
    tau_bc: jnp.ndarray,
    tau_diff: jnp.ndarray,
) -> jnp.ndarray:
    r"""Signed stellar bolometric absorbed luminosity from the LUT.

    Computes :math:`M_\star L_\odot \sum_{m,a} w_{m,a}(B_{m,a} - G_{m,a})` with
    ``G`` bilinearly interpolated at ``(tau_bc, tau_diff)``. Returned *signed*
    (the caller adds the nebular term and takes the absolute value), matching
    ``jnp.trapezoid(absorbed_lnu, nu)`` on the full grid.

    Parameters
    ----------
    lut : EnergyBalanceLUT
        Precomputed ``B``/``G``.
    joint_weights : ndarray, shape (n_met, n_age)
        Runtime DSPS joint (metallicity, age) weights.
    mass_scale : float
        ``total_mass × L_sun`` scaling applied to the SSP luminosities.
    tau_bc, tau_diff : float
        Runtime optical depths.

    Returns
    -------
    float
        Signed stellar absorbed bolometric luminosity.
    """
    w_bc = _interp_weight_vector(lut.tau_bc_grid, tau_bc)  # (n_bc,)
    w_diff = _interp_weight_vector(lut.tau_diff_grid, tau_diff)  # (n_diff,)
    # Bilinear interpolation as a contraction over the two optical-depth axes.
    g_interp = jnp.einsum("maij,i,j->ma", lut.G, w_bc, w_diff)  # (n_met, n_age)
    return mass_scale * jnp.sum(joint_weights * (lut.B - g_interp))

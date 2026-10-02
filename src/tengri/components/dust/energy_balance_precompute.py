# SPDX-License-Identifier: BSD-3-Clause
r"""Build-time LUT for the two-component dust energy balance (``L_ir``).

Under ``approx=WavePrecomp()`` the photometry is projected from per-filter
LUTs, so the full-wavelength stellar SED cube is normally dead-code-eliminated
by XLA. Enabling dust IR re-emission, however, makes ``L_ir``: the
energy-balance absorbed luminosity: feed the output, and the exact
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
interpolated and contracted with the weights: no full-wavelength cube.

The spectral integral is held at full SSP resolution (so #622's far-IR exactness
is preserved); the only approximation is the smooth bilinear interpolation in
the two optical-depth axes, where :math:`\int \mathrm{SSP}\, e^{-\tau k}\, d\nu`
is monotone and well behaved.

This LUT is the precomputed factorization of the canonical energy-balance
integral :func:`tengri.forward.energy_balance.bolometric_absorbed`: same
signed :math:`\int (L_\nu^{\rm intr} - L_\nu^{\rm att})\, d\nu` with the same
Lyman-continuum mask (#922; edge at
:data:`tengri.components.lyc.LYMAN_LIMIT_AA`). The two must agree; the
contract is pinned by ``tests/contract/test_energy_balance_lut.py``. Both
``B``/``G`` (and their fesc-linear ``B_fesc``/``G_fesc`` twins) are reduced
through :func:`tengri.components.lyc.edge_trapezoid` rather than a plain
``jnp.trapezoid``, so the SSP grid cell straddling the edge gets the SAME
step-model rectangle split the exact path applies (one Lyman edge, see that
module's docstring); the per-node ``mask_nonlyc``/``weight_a0``/``weight_a1``
region weighting below is unchanged, only the final reduction is edge-aware.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tengri.components.dust._params import DEFAULT_DUST_F_OBSCURATION
from tengri.components.dust.attenuation import two_component_dust
from tengri.components.lyc import LYMAN_LIMIT_AA, edge_trapezoid, ionizing_mask

__all__ = [
    "EnergyBalanceLUT",
    "build_energy_balance_lut",
    "lut_l_absorbed_stellar",
    "lut_l_absorbed_stellar_log10",
]


class EnergyBalanceLUT(NamedTuple):
    """Precomputed bolometric absorption LUT for the stellar energy balance.

    Attributes
    ----------
    B : ndarray, shape (n_met, n_age)
        Intrinsic bolometric SSP luminosity per unit mass, ``∫ SSP dν`` (signed,
        masked to the non-ionizing side of LYMAN_LIMIT_AA). [erg/s/Hz · Hz per Lsun-flux unit]
    G : ndarray, shape (n_met, n_age, n_tau_bc, n_tau_diff)
        Attenuated bolometric SSP luminosity ``∫ SSP·T_a dν`` on the optical-depth
        grid.
    tau_bc_grid : ndarray, shape (n_tau_bc,)
        Birth-cloud optical-depth grid nodes.
    tau_diff_grid : ndarray, shape (n_tau_diff,)
        Diffuse-ISM optical-depth grid nodes.
    B_fesc, G_fesc : ndarray or None
        The fesc-LINEAR-COEFFICIENT family (#2539 item 1), same shapes as
        ``B``/``G``: the stellar absorbed integral is affine in the live
        nebular escape fraction, ``A(fesc) = A_0 + fesc * A_1``, when
        ``eb_include_lyc=True`` unmasks the Lyman continuum. ``B``/``G`` above
        are then the fesc-INDEPENDENT ``A_0`` family (non-LyC everywhere, plus
        the fesc=0 remainder of the LyC region); ``B_fesc``/``G_fesc`` are the
        ``A_1`` coefficient (the young/birth-cloud-weighted, or -- under
        ``lyc_absorb_all=True`` -- unweighted, LyC-only contribution). ``None``
        when the model never needs fesc-exactness (single-component dust,
        ``eb_include_lyc=False``, or no live photoionized nebular component):
        ``lut_l_absorbed_stellar_log10`` then behaves exactly as before.
    """

    B: jnp.ndarray
    G: jnp.ndarray
    tau_bc_grid: jnp.ndarray
    tau_diff_grid: jnp.ndarray
    B_fesc: jnp.ndarray | None = None
    G_fesc: jnp.ndarray | None = None


def build_energy_balance_lut(
    ssp_flux: jnp.ndarray,
    ssp_wave: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    *,
    law_bc: str,
    law_diff: str,
    f_obscuration: float = DEFAULT_DUST_F_OBSCURATION,
    t_birth_yr: float = 1e7,
    transition_width_dex: float = 0.3,
    bc_params: dict | None = None,
    diff_params: dict | None = None,
    lyman_cutoff_aa: float = 0.0,
    eb_include_lyc: bool = False,
    tau_bc_grid: jnp.ndarray,
    tau_diff_grid: jnp.ndarray,
    fesc_exact: bool = False,
    lyc_absorb_all: bool = False,
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
        FSPS-parity toggle (#961): when True, the LyC (ionizing side of LYMAN_LIMIT_AA) is kept in
        the absorbed-luminosity integrand (all absorbed energy heats dust)
        instead of the canonical LyC mask (#922). Must match the runtime
        ``DustSEDComponent.config.eb_include_lyc``.
    tau_bc_grid, tau_diff_grid : ndarray
        Optical-depth grid nodes (keyword-only).
    fesc_exact : bool, optional
        Also build the ``B_fesc``/``G_fesc`` family (#2539 item 1): the
        stellar absorbed integral's fesc-linear coefficient, so
        :func:`lut_l_absorbed_stellar_log10` can be exact in a live nebular
        escape fraction instead of declining the LUT outright. Only
        meaningful when ``eb_include_lyc=True`` (otherwise the LyC region is
        masked out of the integral regardless of fesc, and ``B``/``G`` alone
        are already exact); ignored otherwise.
    lyc_absorb_all : bool, optional
        Mirrors ``DustSEDComponent.config.lyc_absorb_all`` (#961/#2539): when
        True, ALL stellar ages route their LyC through the gas uniformly (no
        young/old split), so the fesc-linear ``A_1`` term is the UNWEIGHTED
        LyC contribution of every age, and ``A_0`` excludes the LyC region
        entirely. When False (default), only the young/birth-cloud population
        (``t_birth_yr``/``transition_width_dex``-weighted) is credited, matching
        ``two_component.py``'s ``lyc_factor = 1 - y_age*(1 - fesc)``.

    Returns
    -------
    EnergyBalanceLUT
    """
    mask_nonlyc = ~ionizing_mask(ssp_wave, edge_aa=LYMAN_LIMIT_AA)

    sspm_fesc = None
    if fesc_exact and eb_include_lyc:
        # A(fesc) = A_0 + fesc * A_1 (#2539 item 1): lyc_factor(age) =
        # 1 - y_age*(1-fesc) is affine in fesc, so the per-(met,age,wave) SSP
        # weight in the LyC region splits into a fesc-independent piece
        # ((1-y_age), or 0 under lyc_absorb_all=True) and a fesc-linear piece
        # (y_age, or 1 under lyc_absorb_all=True). Outside the LyC region the
        # weight is always 1 for A_0 and 0 for A_1 -- unaffected by fesc,
        # matching the runtime exact integral (two_component.py §2a/§3).
        from tengri.components.dust.two_component import _young_indicator

        y_age = _young_indicator(ssp_ages_yr, t_birth_yr, transition_width_dex)  # (n_age,)
        ones_age = jnp.ones_like(y_age)[:, None]
        if lyc_absorb_all:
            weight_a0 = jnp.where(mask_nonlyc[None, :], ones_age, 0.0)
            weight_a1 = jnp.where(mask_nonlyc[None, :], 0.0, ones_age)
        else:
            weight_a0 = jnp.where(mask_nonlyc[None, :], ones_age, (1.0 - y_age)[:, None])
            weight_a1 = jnp.where(mask_nonlyc[None, :], 0.0, y_age[:, None])
        sspm = ssp_flux * weight_a0[None, :, :]  # (n_met, n_age, n_wave)
        sspm_fesc = ssp_flux * weight_a1[None, :, :]
    else:
        mask = jnp.ones_like(ssp_wave, dtype=bool) if eb_include_lyc else mask_nonlyc
        sspm = ssp_flux * mask[None, None, :]  # (n_met, n_age, n_wave)

    # side="all": each array (sspm / sspm_fesc) already carries its own
    # per-region weight (mask_nonlyc / weight_a0 / weight_a1 above), so the
    # edge-aware reduction just needs to sum everything it holds -- the
    # bracket cell straddling LYMAN_LIMIT_AA is then split into its two
    # rectangles using whatever weight is actually present on each side
    # (zero on a masked-out ionizing node), which is bit-identical to
    # selecting "nonionizing"/"ionizing" explicitly when one side is zeroed.
    B = edge_trapezoid(
        sspm, ssp_wave, side="all", edge_aa=LYMAN_LIMIT_AA, axis=-1
    )  # (n_met, n_age)

    bc_params = bc_params or {}
    diff_params = diff_params or {}

    def g_at(sspm_in, tb, td):
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
        integrand = sspm_in * transmission[None, :, :]  # (n_met, n_age, n_wave)
        return edge_trapezoid(
            integrand, ssp_wave, side="all", edge_aa=LYMAN_LIMIT_AA, axis=-1
        )  # (n_met, n_age)

    # Build-time Python loop over the (small) optical-depth grid. Eagerly,
    # the 576-node loop spends its time in per-op Python dispatch inside
    # the attenuation law, not math: jit once and reuse. The SSP cube is
    # threaded as an argument so it enters the graph as a runtime input,
    # not a constant to fold.
    g_at_compiled = jax.jit(g_at)

    def _build_g(sspm_in):
        return jnp.stack(
            [
                jnp.stack([g_at_compiled(sspm_in, tb, td) for td in tau_diff_grid], axis=-1)
                for tb in tau_bc_grid
            ],
            axis=-2,
        )  # (n_met, n_age, n_tau_bc, n_tau_diff)

    G = _build_g(sspm)

    B_fesc = None
    G_fesc = None
    if sspm_fesc is not None:
        B_fesc = edge_trapezoid(sspm_fesc, ssp_wave, side="all", edge_aa=LYMAN_LIMIT_AA, axis=-1)
        G_fesc = _build_g(sspm_fesc)

    return EnergyBalanceLUT(
        B=B,
        G=G,
        tau_bc_grid=jnp.asarray(tau_bc_grid),
        tau_diff_grid=jnp.asarray(tau_diff_grid),
        B_fesc=B_fesc,
        G_fesc=G_fesc,
    )


def _interp_bracket(grid: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Lower bracketing node index and the two linear weights for ``x``.

    Linear interpolation on a uniform grid touches exactly two nodes, so the
    dense ``(n_nodes,)`` weight vector this replaces was zero everywhere except
    at ``i0`` and ``i0 + 1``. Returning just the bracket lets the caller contract
    a two-node slice of ``G`` instead of all ``n_nodes`` of it.

    The node wavelengths are reconstructed arithmetically (``grid`` is a uniform
    linspace) rather than gathered, so no indexing op touches ``grid`` itself.

    Parameters
    ----------
    grid : ndarray, shape (n_nodes,)
        Uniform ascending grid.
    x : ndarray, shape ()
        Query point. May lie outside ``grid``.

    Returns
    -------
    i0 : ndarray, shape (), int32
        Lower node index, clipped to ``[0, n_nodes - 2]``.
    weights : ndarray, shape (2,)
        Weights on nodes ``i0`` and ``i0 + 1``. Both are zero when ``x`` lies
        more than one spacing outside the grid, reproducing the dense form.

    Notes
    -----
    JIT/grad/vmap safe. ``i0`` is piecewise constant, so it carries no gradient;
    the derivative with respect to ``x`` flows entirely through ``weights``,
    which is the correct derivative of a piecewise-linear interpolant.
    """
    n = grid.shape[0]
    if n == 1:
        return jnp.zeros((), dtype=jnp.int32), jnp.ones((1,), dtype=grid.dtype)

    dx = grid[1] - grid[0]  # uniform linspace spacing
    i0 = jnp.clip(jnp.floor((x - grid[0]) / dx).astype(jnp.int32), 0, n - 2)
    nodes = grid[0] + dx * (i0.astype(grid.dtype) + jnp.arange(2, dtype=grid.dtype))
    weights = jnp.clip(1.0 - jnp.abs(x - nodes) / dx, 0.0, 1.0)
    return i0, weights


def _contract_bg(
    B: jnp.ndarray,
    G: jnp.ndarray,
    tau_bc_grid: jnp.ndarray,
    tau_diff_grid: jnp.ndarray,
    joint_weights: jnp.ndarray,
    tau_bc: jnp.ndarray,
    tau_diff: jnp.ndarray,
) -> jnp.ndarray:
    r"""Per-unit-mass signed absorbed luminosity, :math:`\sum_{m,a} w(B - G)`.

    The mass scaling is deliberately *not* applied here: this contraction is
    O(1) (the SSP integrals are per unit mass), whereas ``mass_scale`` is
    ~1e43 and carries the whole dynamic-range problem. Keeping them separate
    lets the log form fold the scale into an exponent instead of a product.

    Takes explicit ``B``/``G`` (rather than an :class:`EnergyBalanceLUT`) so
    :func:`lut_l_absorbed_stellar_log10` can contract the fesc-linear
    ``B_fesc``/``G_fesc`` family (#2539 item 1) through the SAME bilinear
    interpolation, instead of duplicating it.
    """
    i0, w_bc = _interp_bracket(tau_bc_grid, tau_bc)  # (), (2,)
    j0, w_diff = _interp_bracket(tau_diff_grid, tau_diff)  # (), (2,)

    # Bilinear interpolation touches four nodes of ``G``, so slice those four
    # out before contracting. Contracting the whole optical-depth grid instead
    # (which is what a dense weight vector forces) costs n_met x n_age x
    # n_bc x n_diff multiply-adds to use n_met x n_age x 4 of them: on a
    # (15, 93, 24, 24) LUT that is 803,520 versus 5,580, a 144x overshoot, and
    # it dominated the whole WavePrecomp forward pass.
    n_met, n_age = B.shape
    g_sub = jax.lax.dynamic_slice(
        G,
        (jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32), i0, j0),
        (n_met, n_age, w_bc.shape[0], w_diff.shape[0]),
    )
    g_interp = jnp.einsum("maij,i,j->ma", g_sub, w_bc, w_diff)  # (n_met, n_age)
    return jnp.sum(joint_weights * (B - g_interp))


def _lut_contract(
    lut: EnergyBalanceLUT,
    joint_weights: jnp.ndarray,
    tau_bc: jnp.ndarray,
    tau_diff: jnp.ndarray,
) -> jnp.ndarray:
    """Per-unit-mass signed absorbed luminosity of the ``A_0`` (or the only) family."""
    return _contract_bg(
        lut.B, lut.G, lut.tau_bc_grid, lut.tau_diff_grid, joint_weights, tau_bc, tau_diff
    )


def lut_l_absorbed_stellar_log10(
    lut: EnergyBalanceLUT,
    joint_weights: jnp.ndarray,
    log10_mass_scale: jnp.ndarray,
    tau_bc: jnp.ndarray,
    tau_diff: jnp.ndarray,
    fesc: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""log10 magnitude and sign of the stellar absorbed luminosity.

    The float32-safe form of :func:`lut_l_absorbed_stellar`: the ~1e43
    ``mass_scale`` is carried as a log10 offset and added to the log of the
    O(1) contraction, so the product is never materialized (float32 ceiling
    3.4e38, #1206).

    Parameters
    ----------
    lut : EnergyBalanceLUT
        Precomputed ``B``/``G`` (and, when built with ``fesc_exact=True``,
        ``B_fesc``/``G_fesc``).
    joint_weights : ndarray, shape (n_met, n_age)
        Runtime DSPS joint (metallicity, age) weights.
    log10_mass_scale : ndarray, shape ()
        ``log10(total_mass x L_sun)`` [dex].
    tau_bc, tau_diff : ndarray, shape ()
        Runtime optical depths.
    fesc : ndarray, shape (), optional
        Runtime nebular escape fraction (#2539 item 1). When the LUT carries
        the fesc-linear ``B_fesc``/``G_fesc`` family (built with
        ``fesc_exact=True``), the exact affine combine
        :math:`A(\mathrm{fesc}) = A_0 + \mathrm{fesc} \cdot A_1` is applied
        BEFORE the mass scale / log10 conversion, at the O(1) per-unit-mass
        contraction level -- a plain linear combine, not a log-domain one,
        because both ``A_0`` and ``A_1`` contractions are already O(1)
        (:func:`_contract_bg`'s docstring) and share the SAME grid-orientation
        sign (both are ``sum(w*(B-G))`` over the same descending-``nu`` grid),
        so ordinary addition reproduces the correct signed sum with no
        overflow risk and no need for :func:`tengri.utils.scale.log10_add`.
        Ignored (exactly reproduces the pre-#2539-item-1 answer) when ``None``
        or when the LUT lacks the fesc family.

    Returns
    -------
    log_magnitude : ndarray, shape ()
        :math:`\log_{10}|L_{\rm abs}^\star / (\mathrm{erg/s})|` [dex]. ``-inf``
        when nothing is absorbed; ``+inf`` when the contraction is non-finite.
    sign : ndarray, shape ()
        Sign of the signed luminosity (follows the grid orientation), so the
        caller can combine it with other terms via
        :func:`tengri.utils.scale.log10_add`. ``NaN`` when the contraction is
        non-finite.

    Notes
    -----
    JIT/grad/vmap-safe; the where-dummy keeps the zero case NaN-free.

    Carries the same corrupt/zero split as
    :func:`tengri.forward.energy_balance.bolometric_absorbed_log10` (#1527),
    and must: this is the **stellar** half of the LUT branch in
    ``two_component.py``, the path taken whenever ``approx=WavePrecomp(...)``
    is set. Leaving it fail-open while the exact form is strict would tighten
    only the nebular term on the configuration most fits actually use.

    ``positive = magnitude > 0`` is False for NaN, so before this the whole
    stellar absorbed luminosity silently became ``-inf`` (i.e. exactly 0.0)
    on a corrupt contraction.
    """
    from tengri.utils.scale import _not_computable, log10_magnitude

    contracted = _lut_contract(lut, joint_weights, tau_bc, tau_diff)
    if fesc is not None and lut.B_fesc is not None and lut.G_fesc is not None:
        contracted_fesc = _contract_bg(
            lut.B_fesc,
            lut.G_fesc,
            lut.tau_bc_grid,
            lut.tau_diff_grid,
            joint_weights,
            tau_bc,
            tau_diff,
        )
        contracted = contracted + jnp.asarray(fesc) * contracted_fesc
    log_relative = log10_magnitude(contracted)
    corrupt = _not_computable(log_relative)
    log_mag = jnp.where(corrupt, jnp.inf, log_relative + log10_mass_scale)
    return log_mag, jnp.where(corrupt, jnp.nan, jnp.sign(contracted))


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

    Notes
    -----
    ``mass_scale`` is ~1e43, so this product overflows float32. Use
    :func:`lut_l_absorbed_stellar_log10` on a pure-float32 path (#1206).
    """
    return mass_scale * _lut_contract(lut, joint_weights, tau_bc, tau_diff)

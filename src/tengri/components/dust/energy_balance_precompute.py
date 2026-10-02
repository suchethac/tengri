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
912 Å Lyman-continuum mask (#922). The two must agree; the contract is pinned
by ``tests/contract/test_energy_balance_lut.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tengri.components.dust._apply import birth_cloud_age_weight, two_component_curves
from tengri.components.dust._params import DEFAULT_DUST_F_OBSCURATION
from tengri.components.dust.attenuation import two_component_dust
from tengri.utils.physics_constants import C_AA

__all__ = [
    "EnergyBalanceLUT",
    "build_energy_balance_lut",
    "build_energy_balance_lut_over_z",
    "lut_l_absorbed_stellar",
    "lut_l_absorbed_stellar_log10",
    "nebular_grid_absorbed_log10",
]


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
    #: ``ln(1+z)`` axis of ``G`` for an attenuation law that reads the model
    #: redshift, else ``None``. When set, ``G`` carries a leading redshift axis,
    #: shape ``(n_z, n_met, n_age, n_tau_bc, n_tau_diff)``, and the contraction
    #: needs the evaluation redshift. ``B`` is intrinsic, so it has no z axis.
    ln1pz: jnp.ndarray | None = None


def build_energy_balance_lut_over_z(
    ssp_flux: jnp.ndarray,
    ssp_wave: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    *,
    ln1pz: jnp.ndarray,
    params_at_z,
    law_bc: str,
    law_diff: str,
    f_obscuration: float = DEFAULT_DUST_F_OBSCURATION,
    t_birth_yr: float = 1e7,
    transition_width_dex: float = 0.3,
    lyman_cutoff_aa: float = 0.0,
    eb_include_lyc: bool = False,
    tau_bc_grid: jnp.ndarray,
    tau_diff_grid: jnp.ndarray,
) -> EnergyBalanceLUT:
    r"""Energy-balance LUT for a law whose curve moves with the evaluation redshift.

    Same ``B`` and ``G`` as :func:`build_energy_balance_lut`, with ``G`` built on
    every ``ln(1+z)`` node so the absorbed luminosity follows the curve at the
    redshift the model is *evaluated* at (a free redshift, or a per-galaxy
    runtime redshift under ``WavePrecomp(catalog_z_range=...)``), not the one the
    spec happened to carry when the table was built.

    Parameters
    ----------
    ssp_flux, ssp_wave, ssp_ages_yr, law_bc, law_diff, f_obscuration,
    t_birth_yr, transition_width_dex, lyman_cutoff_aa, eb_include_lyc,
    tau_bc_grid, tau_diff_grid
        As :func:`build_energy_balance_lut`.
    ln1pz : ndarray, shape (n_z,)
        Ascending redshift nodes, :math:`\ln(1+z)`. [dimensionless]
    params_at_z : callable
        ``params_at_z(z) -> (bc_params, diff_params)`` of jit-safe scalars: the
        resolved law parameters at redshift ``z``.

    Returns
    -------
    EnergyBalanceLUT
        With ``ln1pz`` set and ``G`` of shape ``(n_z, n_met, n_age, n_bc, n_diff)``.

    Notes
    -----
    **JIT-compatible**: no, build time.

    The transmission factorizes, :math:`T = f + (1 - f)\, e^{-w_a \tau_{\rm bc}
    k_{\rm bc}(\lambda)}\, e^{-\tau_{\rm diff} k_{\rm diff}(\lambda)}`, so on
    each node

    .. math::

        G_{ma}(\tau_{\rm bc}, \tau_{\rm diff}) = f\, B_{ma} + (1 - f)
        \sum_\lambda q_\lambda\, S_{ma\lambda}\,
        e^{-w_a \tau_{\rm bc} k_{\rm bc, \lambda}}\,
        e^{-\tau_{\rm diff} k_{\rm diff, \lambda}}

    with :math:`q_\lambda` the trapezoid weights of :math:`\int d\nu`. That is one
    small matrix product per age instead of one full
    ``(n_met, n_age, n_wave)`` pass per optical-depth node, which is what keeps a
    table over tens of redshifts affordable. It agrees with
    :func:`build_energy_balance_lut` at a node to floating-point round-off.
    """
    nu = C_AA / ssp_wave
    mask = jnp.ones_like(ssp_wave, dtype=bool) if eb_include_lyc else (ssp_wave >= 912.0)
    sspm = ssp_flux * mask[None, None, :]
    B = jnp.trapezoid(sspm, nu, axis=-1)
    # Trapezoid weights q with sum(q * y) == trapezoid(y, nu).
    dnu = jnp.diff(nu)
    q = 0.5 * (jnp.concatenate([dnu, jnp.zeros(1)]) + jnp.concatenate([jnp.zeros(1), dnu]))
    weighted = jnp.transpose(sspm * q[None, None, :], (1, 0, 2))  # (n_age, n_met, n_wave)
    age_weight = birth_cloud_age_weight(ssp_ages_yr, t_birth_yr, transition_width_dex)
    tau_bc = jnp.asarray(tau_bc_grid)
    tau_diff = jnp.asarray(tau_diff_grid)
    f_obs = jnp.asarray(f_obscuration)

    @jax.jit
    def g_at(z):
        bc_params, diff_params = params_at_z(z)
        k_bc, k_diff = two_component_curves(
            ssp_wave,
            law_bc,
            law_diff,
            {k: jnp.asarray(v) for k, v in bc_params.items()},
            {k: jnp.asarray(v) for k, v in diff_params.items()},
            lyman_cutoff_aa,
        )
        diffuse = jnp.exp(-k_diff[:, None] * tau_diff[None, :])  # (n_wave, n_diff)

        def one_age(args):
            s_a, w_a = args  # (n_met, n_wave), ()
            birth = jnp.exp(-(w_a * k_bc)[:, None] * tau_bc[None, :])  # (n_wave, n_bc)
            return jnp.einsum("ml,lb,ld->mbd", s_a, birth, diffuse)

        g_age = jax.lax.map(one_age, (weighted, age_weight))  # (n_age, n_met, n_bc, n_diff)
        g_age = jnp.transpose(g_age, (1, 0, 2, 3))
        return f_obs * B[:, :, None, None] + (1.0 - f_obs) * g_age

    G = jnp.stack([g_at(z) for z in jnp.expm1(ln1pz)])
    return EnergyBalanceLUT(
        B=B,
        G=G,
        tau_bc_grid=tau_bc,
        tau_diff_grid=tau_diff,
        ln1pz=jnp.asarray(ln1pz),
    )


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
        the absorbed-luminosity integrand (all absorbed energy heats dust)
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
        return jnp.trapezoid(integrand, nu, axis=-1)  # (n_met, n_age)

    # Build-time Python loop over the (small) optical-depth grid. Eagerly,
    # the 576-node loop spends its time in per-op Python dispatch inside
    # the attenuation law, not math: jit once and reuse. The SSP cube is
    # threaded as an argument so it enters the graph as a runtime input,
    # not a constant to fold.
    g_at_compiled = jax.jit(g_at)
    G = jnp.stack(
        [
            jnp.stack([g_at_compiled(sspm, tb, td) for td in tau_diff_grid], axis=-1)
            for tb in tau_bc_grid
        ],
        axis=-2,
    )  # (n_met, n_age, n_tau_bc, n_tau_diff)

    return EnergyBalanceLUT(
        B=B, G=G, tau_bc_grid=jnp.asarray(tau_bc_grid), tau_diff_grid=jnp.asarray(tau_diff_grid)
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


def _lut_contract(
    lut: EnergyBalanceLUT,
    joint_weights: jnp.ndarray,
    tau_bc: jnp.ndarray,
    tau_diff: jnp.ndarray,
    redshift: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Per-unit-mass signed absorbed luminosity, :math:`\sum_{m,a} w(B - G)`.

    The mass scaling is deliberately *not* applied here: this contraction is
    O(1) (the SSP integrals are per unit mass), whereas ``mass_scale`` is
    ~1e43 and carries the whole dynamic-range problem. Keeping them separate
    lets the log form fold the scale into an exponent instead of a product.
    """
    i0, w_bc = _interp_bracket(lut.tau_bc_grid, tau_bc)  # (), (2,)
    j0, w_diff = _interp_bracket(lut.tau_diff_grid, tau_diff)  # (), (2,)

    # Bilinear interpolation touches four nodes of ``G``, so slice those four
    # out before contracting. Contracting the whole optical-depth grid instead
    # (which is what a dense weight vector forces) costs n_met x n_age x
    # n_bc x n_diff multiply-adds to use n_met x n_age x 4 of them: on a
    # (15, 93, 24, 24) LUT that is 803,520 versus 5,580, a 144x overshoot, and
    # it dominated the whole WavePrecomp forward pass.
    n_met, n_age = lut.B.shape
    zero = jnp.zeros((), jnp.int32)
    if lut.ln1pz is None:
        g_sub = jax.lax.dynamic_slice(
            lut.G,
            (zero, zero, i0, j0),
            (n_met, n_age, w_bc.shape[0], w_diff.shape[0]),
        )
        g_interp = jnp.einsum("maij,i,j->ma", g_sub, w_bc, w_diff)  # (n_met, n_age)
    else:
        # The curve moves with redshift: G carries a leading ln(1+z) axis and the
        # contraction reads it at the EVALUATION redshift, linear in ln(1+z).
        if redshift is None:
            raise ValueError(
                "this energy-balance LUT is tabulated over redshift (its attenuation "
                "law reads the model redshift), so the contraction needs the "
                "evaluation redshift; none was passed."
            )
        from tengri.components._z_response import z_bracket

        k0, w_z = z_bracket(lut.ln1pz, redshift)  # (), (2,)
        g_sub = jax.lax.dynamic_slice(
            lut.G,
            (k0, zero, zero, i0, j0),
            (w_z.shape[0], n_met, n_age, w_bc.shape[0], w_diff.shape[0]),
        )
        g_interp = jnp.einsum("zmaij,z,i,j->ma", g_sub, w_z, w_bc, w_diff)
    return jnp.sum(joint_weights * (lut.B - g_interp))


def lut_l_absorbed_stellar_log10(
    lut: EnergyBalanceLUT,
    joint_weights: jnp.ndarray,
    log10_mass_scale: jnp.ndarray,
    tau_bc: jnp.ndarray,
    tau_diff: jnp.ndarray,
    redshift: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""log10 magnitude and sign of the stellar absorbed luminosity.

    The float32-safe form of :func:`lut_l_absorbed_stellar`: the ~1e43
    ``mass_scale`` is carried as a log10 offset and added to the log of the
    O(1) contraction, so the product is never materialized (float32 ceiling
    3.4e38, #1206).

    Parameters
    ----------
    lut : EnergyBalanceLUT
        Precomputed ``B``/``G``.
    joint_weights : ndarray, shape (n_met, n_age)
        Runtime DSPS joint (metallicity, age) weights.
    log10_mass_scale : ndarray, shape ()
        ``log10(total_mass x L_sun)`` [dex].
    tau_bc, tau_diff : ndarray, shape ()
        Runtime optical depths.
    redshift : ndarray, shape (), optional
        Evaluation redshift. Required when ``lut`` is tabulated over redshift
        (``lut.ln1pz`` is set); ignored otherwise. [dimensionless]

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

    contracted = (
        _lut_contract(lut, joint_weights, tau_bc, tau_diff)
        if redshift is None
        else _lut_contract(lut, joint_weights, tau_bc, tau_diff, redshift)
    )
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
    redshift: jnp.ndarray | None = None,
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
    redshift : float, optional
        Evaluation redshift, required when the LUT is tabulated over redshift.

    Returns
    -------
    float
        Signed stellar absorbed bolometric luminosity.

    Notes
    -----
    ``mass_scale`` is ~1e43, so this product overflows float32. Use
    :func:`lut_l_absorbed_stellar_log10` on a pure-float32 path (#1206).
    """
    return mass_scale * (
        _lut_contract(lut, joint_weights, tau_bc, tau_diff)
        if redshift is None
        else _lut_contract(lut, joint_weights, tau_bc, tau_diff, redshift)
    )


def nebular_grid_absorbed_log10(
    grid_abs: jnp.ndarray,
    log_nion: jnp.ndarray,
    tau_grids: tuple,
    tau_a: jnp.ndarray,
    tau_b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Absorbed nebular luminosity from the per-Q_H grid: log10 magnitude and sign.

    .. math::

        L_{\rm abs}^{\rm neb} = Q_H \sum_{a,b} w_a \, A_{ab} \, w_b

    where :math:`A_{ab}` is the signed absorbed luminosity per unit
    :math:`Q_H` [erg/s per (photon/s)] on the optical-depth nodes and
    :math:`w_a`, :math:`w_b` the two-node linear weights at
    ``(tau_a, tau_b)``.

    Parameters
    ----------
    grid_abs : ndarray, shape (n_tau_a, n_tau_b)
        Signed absorbed luminosity per unit Q_H at the nebular grid point, in
        the orientation of the frequency integral (negative on an ascending
        wavelength grid) [erg/s per (photon/s)].
    log_nion : ndarray, shape ()
        :math:`\log_{10} Q_H` [dex re photon/s].
    tau_grids : tuple
        ``(tau_a_nodes, tau_b_nodes)``, each a uniform ascending sequence.
    tau_a, tau_b : ndarray, shape ()
        Optical depths of the evaluation point.

    Returns
    -------
    log_magnitude : ndarray, shape ()
        :math:`\log_{10} |L_{\rm abs}^{\rm neb}|` [dex re erg/s]; ``-inf`` when
        nothing is absorbed.
    sign : ndarray, shape ()
        Sign of the signed value, the same convention
        :func:`tengri.forward.energy_balance.bolometric_absorbed_log10` returns.

    Notes
    -----
    JIT/grad/vmap-safe: the empty case takes the where-dummy path, so no NaN
    reaches the backward pass. The magnitude per unit Q_H is ~1e-11 and
    ``log_nion`` ~53, so the product is formed in the log domain (#1206).
    """
    grid_a = jnp.asarray(tau_grids[0], dtype=grid_abs.dtype)
    grid_b = jnp.asarray(tau_grids[1], dtype=grid_abs.dtype)
    ia, wa = _interp_bracket(grid_a, tau_a)
    ib, wb = _interp_bracket(grid_b, tau_b)
    sub = jax.lax.dynamic_slice(grid_abs, (ia, ib), (wa.shape[0], wb.shape[0]))
    per_qh = jnp.einsum("a,ab,b->", wa, sub, wb)
    magnitude = jnp.abs(per_qh)
    nonzero = magnitude > 0
    safe = jnp.where(nonzero, magnitude, 1.0)
    log_abs = jnp.where(nonzero, jnp.asarray(log_nion) + jnp.log10(safe), -jnp.inf)
    return log_abs, jnp.sign(per_qh)

# SPDX-License-Identifier: BSD-3-Clause
r"""Build-time LUT for the two-component dust energy balance (``L_ir``).

Under ``approx=WavePrecomp()`` the photometry is projected from per-filter
LUTs, so the full-wavelength stellar SED cube is normally dead-code-eliminated
by XLA. Enabling dust IR re-emission, however, makes ``L_ir``: the
energy-balance absorbed luminosity: feed the output, and the exact
:func:`L_absorbed` integral is taken over the full ``(n_met, n_age, n_wave)``
stellar cube. That single dependency resurrects the cube and costs ~40× per
evaluation (30 µs → 1.2 ms on a photometry-only fit).

The integral factorizes over stellar *populations*. A node :math:`(m, a)`
holds the fraction :math:`y_a` of its formed mass in the young population
(birth cloud + diffuse screen) and :math:`1 - y_a` in the old one (diffuse
screen only); the absorbed energy is linear in the populations, so with
transmissions :math:`T_p(\lambda)` independent of the SFH,

.. math::

    L_{\rm abs}^{\star} = M_\star \, L_\odot \sum_{m,a} w_{m,a}
        \sum_{p} y_{a,p} \left[ B_{m,a} - G^{p}_{m,a}(\tau_{\rm bc}, \tau_{\rm diff}) \right]

(:math:`y_{a,{\rm young}} = y_a`, :math:`y_{a,{\rm old}} = 1 - y_a`) with

.. math::

    B_{m,a}      &= \int \mathrm{SSP}_{m,a}(\lambda)\, d\nu \\
    G^{p}_{m,a}(\boldsymbol\tau) &= \int \mathrm{SSP}_{m,a}(\lambda)\,
        T_p(\lambda;\boldsymbol\tau)\, d\nu

where :math:`w_{m,a}` are the runtime DSPS joint (metallicity, age) weights and
:math:`M_\star L_\odot` the runtime mass scaling. ``B`` and ``G`` depend only on
the fixed SSP grid, the (fixed-shape) attenuation curves, and the optical depths
:math:`(\tau_{\rm bc}, \tau_{\rm diff})`: the node's age is not baked in, only
combined at runtime through :math:`y_a`. They are precomputed once on a small
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
module's docstring); the per-population Lyman-continuum region weights
below are applied before the edge-aware reduction.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tengri.components.dust._params import DEFAULT_DUST_F_OBSCURATION
from tengri.components.lyc import LYMAN_LIMIT_AA, edge_trapezoid, ionizing_mask

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

    The integral is linear in the stellar populations an SSP node holds, so the
    table keeps one family per POPULATION and the runtime mixes them with the
    node's young fraction (``y`` from the stellar component's
    ``age_boundary_younger_fraction``): nothing about the node's age is baked
    in.  The young population sees the birth cloud and the diffuse ISM, the old
    one the diffuse ISM only.

    Attributes
    ----------
    B : ndarray, shape (n_met, n_age)
        Intrinsic bolometric SSP luminosity per unit mass, ``∫ SSP dν`` of the
        YOUNG population (signed, masked to the non-ionizing side of
        LYMAN_LIMIT_AA). [erg/s/Hz · Hz per Lsun-flux unit]
    G : ndarray, shape (n_met, n_age, n_tau_bc, n_tau_diff)
        Attenuated bolometric SSP luminosity ``∫ SSP·T_young dν`` on the
        optical-depth grid.
    tau_bc_grid : ndarray, shape (n_tau_bc,)
        Birth-cloud optical-depth grid nodes.
    tau_diff_grid : ndarray, shape (n_tau_diff,)
        Diffuse-ISM optical-depth grid nodes.
    B_fesc, G_fesc : ndarray or None
        The fesc-LINEAR-COEFFICIENT family (#2539 item 1), same shapes as
        ``B``/``G``: the stellar absorbed integral is affine in the live
        nebular escape fraction, ``A(fesc) = A_0 + fesc * A_1``, when
        ``lyc_in_energy_balance=True`` unmasks the Lyman continuum. ``B``/``G`` above
        are then the fesc-INDEPENDENT ``A_0`` family; ``B_fesc``/``G_fesc`` are the
        ``A_1`` coefficient. ``None`` when the model never needs fesc-exactness
        (single-component dust, ``lyc_in_energy_balance=False``, or no live
        photoionized nebular component).
    B_old, G_old, B_fesc_old, G_fesc_old : ndarray or None
        The same families for the OLD population (diffuse ISM only; the gas
        gate does not act on it unless ``lyc_reprocessed_by='all'``).  ``None``
        on a single-population table (single-screen dust), where the young
        family is the whole answer.
    """

    B: jnp.ndarray
    G: jnp.ndarray
    tau_bc_grid: jnp.ndarray
    tau_diff_grid: jnp.ndarray
    B_fesc: jnp.ndarray | None = None
    G_fesc: jnp.ndarray | None = None
    B_old: jnp.ndarray | None = None
    G_old: jnp.ndarray | None = None
    B_fesc_old: jnp.ndarray | None = None
    G_fesc_old: jnp.ndarray | None = None
    #: ``ln(1+z)`` axis of every ``G`` family for an attenuation law that reads
    #: the model redshift, else ``None``. When set, each ``G`` (``G``, ``G_fesc``,
    #: ``G_old``, ``G_fesc_old``) carries a leading redshift axis, e.g.
    #: ``(n_z, n_met, n_age, n_tau_bc, n_tau_diff)``, and the contraction needs
    #: the evaluation redshift. The ``B`` families are intrinsic, so they have no
    #: z axis.
    ln1pz: jnp.ndarray | None = None


def build_energy_balance_lut(
    ssp_flux: jnp.ndarray,
    ssp_wave: jnp.ndarray,
    *,
    law_bc: str,
    law_diff: str,
    f_obscuration: float = DEFAULT_DUST_F_OBSCURATION,
    bc_params: dict | None = None,
    diff_params: dict | None = None,
    lyman_cutoff_aa: float = 0.0,
    lyc_in_energy_balance: bool = False,
    tau_bc_grid: jnp.ndarray,
    tau_diff_grid: jnp.ndarray,
    fesc_exact: bool = False,
    lyc_reprocessed_by: str = "young",
    lyc_escape_geometry: str = "screened",
    single_population: bool = False,
) -> EnergyBalanceLUT:
    r"""Precompute the young and old population families of the energy balance.

    The transmissions are built with the *same*
    :func:`two_component_interval_transmission` the runtime path uses, so the
    LUT reproduces the exact spectral integral at every grid node.  Nothing
    depends on the node's age: the runtime mixes the two populations with the
    node's young fraction.

    Parameters
    ----------
    ssp_flux : ndarray, shape (n_met, n_age, n_wave)
        SSP specific luminosity per unit mass [Lsun/Hz/Msun].
    ssp_wave : ndarray, shape (n_wave,)
        Rest-frame SSP wavelength grid [Å], ascending.
    law_bc, law_diff : str
        Attenuation-law registry keys (fixed shape).
    f_obscuration, bc_params, diff_params, lyman_cutoff_aa
        Passed verbatim to :func:`two_component_interval_transmission`.
    lyc_in_energy_balance : bool, optional
        FSPS-parity toggle (#961): when True, the LyC (ionizing side of LYMAN_LIMIT_AA) is kept in
        the absorbed-luminosity integrand (all absorbed energy heats dust)
        instead of the canonical LyC mask (#922). Must match the runtime
        ``DustSEDComponent.config.lyc_in_energy_balance``.
    tau_bc_grid, tau_diff_grid : ndarray
        Optical-depth grid nodes (keyword-only).
    fesc_exact : bool, optional
        Also build the ``B_fesc``/``G_fesc`` family (#2539 item 1): the
        stellar absorbed integral's fesc-linear coefficient, so
        :func:`lut_l_absorbed_stellar_log10` can be exact in a live nebular
        escape fraction. Only meaningful when ``lyc_in_energy_balance=True``
        (otherwise the LyC region is masked out regardless of fesc).
    lyc_reprocessed_by : str, optional
        Mirrors ``DustSEDComponent.config.lyc_reprocessed_by``: ``'all'``
        routes every population's LyC through the gas, ``'young'`` only the
        young population's.
    lyc_escape_geometry : str, optional
        Mirrors ``DustSEDComponent.config.lyc_escape_geometry`` (#2529):
        ``'birth_cloud_holes'`` / ``'clear'`` build the affine-in-fesc family
        of :func:`tengri.components.lyc.hole_young_transmission` across the
        whole spectrum (a hole bypasses the birth-cloud screen at every
        wavelength); refused with ``lyc_reprocessed_by='all'`` upstream.
    single_population : bool, optional
        Build the young family only (single-screen dust: no birth cloud, so
        young and old transmissions coincide at ``tau_bc = 0``).

    Returns
    -------
    EnergyBalanceLUT
    """
    from tengri.components.dust._apply import two_component_interval_transmission

    ion = ionizing_mask(ssp_wave, edge_aa=LYMAN_LIMIT_AA).astype(ssp_flux.dtype)
    mask = jnp.ones_like(ion) if lyc_in_energy_balance else 1.0 - ion
    bc_params = {k: jnp.asarray(v) for k, v in (bc_params or {}).items()}
    diff_params = {k: jnp.asarray(v) for k, v in (diff_params or {}).items()}
    f_obs = jnp.asarray(f_obscuration)
    geometry = lyc_escape_geometry != "screened"
    has_fesc = geometry or (fesc_exact and lyc_in_energy_balance)
    gate_all = lyc_reprocessed_by == "all"

    def transmissions(tb, td):
        kw = dict(
            law_bc=law_bc,
            law_diff=law_diff,
            bc_params=bc_params,
            diff_params=diff_params,
            lyman_cutoff_aa=lyman_cutoff_aa,
        )
        wrapped = two_component_interval_transmission(
            ssp_wave, jnp.asarray(tb), jnp.asarray(td), f_obscuration=f_obs, **kw
        )
        raw = two_component_interval_transmission(
            ssp_wave, jnp.asarray(tb), jnp.asarray(td), f_obscuration=0.0, **kw
        )
        return wrapped, raw

    def observed(population, tb, td):
        """Per-wavelength observed weights ``(obs_0, obs_1)`` of one population at (tb, td)."""
        wrapped, raw = transmissions(tb, td)
        t_pop = wrapped[population]
        if not has_fesc:
            return t_pop, None
        if geometry and population == 0:
            t_hole = raw[1] if lyc_escape_geometry == "birth_cloud_holes" else jnp.ones_like(ion)
            t_cov = raw[0] * (1.0 - ion)
            obs0 = f_obs + (1.0 - f_obs) * t_cov
            obs1 = (1.0 - f_obs) * (t_hole - t_cov)
            return obs0, obs1
        if population == 1 and not gate_all:
            return t_pop, jnp.zeros_like(t_pop)
        # LyC gate of the gas, A(f) = A_0 + f A_1: obs = T (1 - ion + f ion).
        return t_pop * (1.0 - ion), t_pop * ion

    def intrinsic(population):
        if not has_fesc:
            return jnp.ones_like(ion), None
        if geometry and population == 0:
            return 1.0 - ion, ion
        if population == 1 and not gate_all:
            return jnp.ones_like(ion), jnp.zeros_like(ion)
        return 1.0 - ion, ion

    # edge_trapezoid is linear in its integrand for a fixed grid, so every integral
    # below is the SSP cube contracted with one quadrature-weight vector; the
    # weights are read off the rule itself (its gradient), so they cannot drift
    # from it. ``quad`` folds in the LyC mask.
    quad = mask * jax.grad(
        lambda y: edge_trapezoid(y, ssp_wave, side="all", edge_aa=LYMAN_LIMIT_AA, axis=-1)
    )(jnp.ones(ssp_wave.shape, dtype=jnp.result_type(ssp_flux, ssp_wave)))

    def integrate(weights):
        """``∫ ssp · mask · w`` for weights of shape (..., n_wave) -> (n_met, n_age, ...)."""
        return jnp.tensordot(ssp_flux, weights * quad, axes=([-1], [-1]))

    def family(population):
        i0, i1 = intrinsic(population)

        # The observed weights at every (tau_bc, tau_diff) node in one jitted
        # vmap. The intrinsic weights ride in the SAME contraction: where a
        # transmission is exactly 1 (tau = 0) the observed column then equals
        # the intrinsic one bit for bit, so L_absorbed = B - G cancels to zero
        # exactly, as the per-node integrals did.
        def weights_at(tb, td):
            o0, o1 = observed(population, tb, td)
            return o0[None] if o1 is None else jnp.stack([o0, o1])

        tb_nodes, td_nodes = jnp.meshgrid(
            jnp.asarray(tau_bc_grid), jnp.asarray(tau_diff_grid), indexing="ij"
        )
        w = jax.jit(jax.vmap(jax.vmap(weights_at)))(tb_nodes, td_nodes)
        ntb, ntd, k, n_wave = w.shape  # k = 1, or 2 with an fesc family
        intrinsic_w = jnp.stack([i0] if i1 is None else [i0, i1])
        cols = integrate(jnp.concatenate([intrinsic_w, w.reshape(ntb * ntd * k, n_wave)]))
        b = cols[..., :k]
        g = cols[..., k:].reshape(*cols.shape[:-1], ntb, ntd, k)
        if i1 is None:
            return b[..., 0], g[..., 0], None, None
        return b[..., 0], g[..., 0], b[..., 1], g[..., 1]

    B, G, B_fesc, G_fesc = family(0)
    if single_population:
        B_old = G_old = B_fesc_old = G_fesc_old = None
    else:
        B_old, G_old, B_fesc_old, G_fesc_old = family(1)
    return EnergyBalanceLUT(
        B=B,
        G=G,
        tau_bc_grid=jnp.asarray(tau_bc_grid),
        tau_diff_grid=jnp.asarray(tau_diff_grid),
        B_fesc=B_fesc,
        G_fesc=G_fesc,
        B_old=B_old,
        G_old=G_old,
        B_fesc_old=B_fesc_old,
        G_fesc_old=G_fesc_old,
    )


def build_energy_balance_lut_over_z(
    ssp_flux: jnp.ndarray,
    ssp_wave: jnp.ndarray,
    *,
    ln1pz: jnp.ndarray,
    params_at_z,
    **kwargs,
) -> EnergyBalanceLUT:
    r"""Energy-balance LUT for a law whose curve moves with the evaluation redshift.

    The same families as :func:`build_energy_balance_lut`, with every ``G`` built
    on each ``ln(1+z)`` node so the absorbed luminosity follows the curve at the
    redshift the model is *evaluated* at (a free redshift, or a per-galaxy
    runtime redshift under ``WavePrecomp(catalog_z_range=...)``), not the one the
    spec carried when the table was built.

    Parameters
    ----------
    ssp_flux : ndarray, shape (n_met, n_age, n_wave)
        SSP specific luminosity per unit mass. [Lsun/Hz/Msun]
    ssp_wave : ndarray, shape (n_wave,)
        Rest-frame SSP wavelength grid, ascending. [Angstrom]
    ln1pz : ndarray, shape (n_z,)
        Ascending redshift nodes, :math:`\ln(1+z)`. [dimensionless]
    params_at_z : callable
        ``params_at_z(z) -> (bc_params, diff_params)`` of jit-safe scalars: the
        resolved law parameters at redshift ``z``.
    **kwargs
        Every other argument of :func:`build_energy_balance_lut` except
        ``bc_params`` / ``diff_params``.

    Returns
    -------
    EnergyBalanceLUT
        With ``ln1pz`` set and every ``G`` family of shape
        ``(n_z, n_met, n_age, n_tau_bc, n_tau_diff)``.

    Notes
    -----
    Build time, not JIT-compatible. Each node is the table
    :func:`build_energy_balance_lut` returns at that redshift, so the two agree
    at a node to round-off by construction.
    """
    nodes = [
        build_energy_balance_lut(ssp_flux, ssp_wave, bc_params=bc, diff_params=diff, **kwargs)
        for bc, diff in (params_at_z(z) for z in jnp.expm1(jnp.asarray(ln1pz)))
    ]
    first = nodes[0]

    def stack(name):
        if getattr(first, name) is None:
            return None
        return jnp.stack([getattr(n, name) for n in nodes])

    return first._replace(
        G=stack("G"),
        G_fesc=stack("G_fesc"),
        G_old=stack("G_old"),
        G_fesc_old=stack("G_fesc_old"),
        ln1pz=jnp.asarray(ln1pz),
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
    ln1pz: jnp.ndarray | None = None,
    redshift: jnp.ndarray | None = None,
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
    zero = jnp.zeros((), jnp.int32)
    if ln1pz is None:
        g_sub = jax.lax.dynamic_slice(
            G,
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

        k0, w_z = z_bracket(ln1pz, redshift)  # (), (2,)
        g_sub = jax.lax.dynamic_slice(
            G,
            (k0, zero, zero, i0, j0),
            (w_z.shape[0], n_met, n_age, w_bc.shape[0], w_diff.shape[0]),
        )
        g_interp = jnp.einsum("zmaij,z,i,j->ma", g_sub, w_z, w_bc, w_diff)
    return jnp.sum(joint_weights * (B - g_interp))


def _population_contract(
    B,
    G,
    B_old,
    G_old,
    lut: EnergyBalanceLUT,
    joint_weights: jnp.ndarray,
    younger_fraction: jnp.ndarray | None,
    tau_bc: jnp.ndarray,
    tau_diff: jnp.ndarray,
    redshift: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-unit-mass signed absorbed luminosity of one family, young and old mixed.

    Each node holds the fraction ``y`` of its mass in the young population and
    ``1 - y`` in the old one; the absorbed energy is linear in the populations,
    so the weights ``w * y`` and ``w * (1 - y)`` contract the two families.
    A single-population table (no old family) contracts the young one alone.
    """
    grids = (lut.tau_bc_grid, lut.tau_diff_grid)
    if G_old is None or younger_fraction is None:
        return _contract_bg(B, G, *grids, joint_weights, tau_bc, tau_diff, lut.ln1pz, redshift)
    y = jnp.asarray(younger_fraction)[None, :]
    young = _contract_bg(B, G, *grids, joint_weights * y, tau_bc, tau_diff, lut.ln1pz, redshift)
    old = _contract_bg(
        B_old,
        G_old,
        *grids,
        joint_weights * (1.0 - y),
        tau_bc,
        tau_diff,
        lut.ln1pz,
        redshift,
    )
    return young + old


def _lut_contract(
    lut: EnergyBalanceLUT,
    joint_weights: jnp.ndarray,
    tau_bc: jnp.ndarray,
    tau_diff: jnp.ndarray,
    *,
    younger_fraction: jnp.ndarray | None = None,
    redshift: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-unit-mass signed absorbed luminosity of the ``A_0`` (or the only) family."""
    return _population_contract(
        lut.B,
        lut.G,
        lut.B_old,
        lut.G_old,
        lut,
        joint_weights,
        younger_fraction,
        tau_bc,
        tau_diff,
        redshift,
    )


def lut_l_absorbed_stellar_log10(
    lut: EnergyBalanceLUT,
    joint_weights: jnp.ndarray,
    log10_mass_scale: jnp.ndarray,
    tau_bc: jnp.ndarray,
    tau_diff: jnp.ndarray,
    fesc: jnp.ndarray | None = None,
    younger_fraction: jnp.ndarray | None = None,
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
    younger_fraction : ndarray, shape (n_age,), optional
        Per-node formed-mass fraction younger than the birth-cloud lifetime
        (row 0 of the stellar component's ``age_boundary_younger_fraction``).
        Mixes the young and old population families of a two-population LUT;
        ignored by a single-population one.
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

    contracted = _lut_contract(
        lut, joint_weights, tau_bc, tau_diff, younger_fraction=younger_fraction, redshift=redshift
    )
    if fesc is not None and lut.B_fesc is not None and lut.G_fesc is not None:
        contracted_fesc = _population_contract(
            lut.B_fesc,
            lut.G_fesc,
            lut.B_fesc_old,
            lut.G_fesc_old,
            lut,
            joint_weights,
            younger_fraction,
            tau_bc,
            tau_diff,
            redshift,
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
    younger_fraction: jnp.ndarray | None = None,
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
    younger_fraction : ndarray, shape (n_age,), optional
        Per-node young fraction mixing the two population families.
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
    return mass_scale * _lut_contract(
        lut, joint_weights, tau_bc, tau_diff, younger_fraction=younger_fraction, redshift=redshift
    )


def nebular_grid_absorbed_log10(
    grid_abs: jnp.ndarray,
    log_nion: jnp.ndarray,
    tau_grids: tuple,
    tau_a: jnp.ndarray,
    tau_b: jnp.ndarray,
    weights: jnp.ndarray | None = None,
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
    grid_abs : ndarray, shape (n_tau_a, n_tau_b) or (K, n_tau_a, n_tau_b)
        Signed absorbed luminosity per unit Q_H at the nebular grid point,
        positively oriented (+1 for a net absorber) [erg/s per (photon/s)].
        A leading channel axis holds one table per age interval's screen (the
        absorbed energy is linear in the screen, so channels mix linearly with
        ``weights``).
    log_nion : ndarray, shape ()
        :math:`\log_{10} Q_H` [dex re photon/s].
    tau_grids : tuple
        ``(tau_a_nodes, tau_b_nodes)``, each a uniform ascending sequence.
    tau_a, tau_b : ndarray, shape ()
        Optical depths of the evaluation point.
    weights : ndarray, shape (K,), optional
        Interval weights mixing the channels (the nebular screen's
        ionizing-luminosity shares); required when ``grid_abs`` has channels.

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
    if grid_abs.ndim == 3:
        if weights is None:
            raise ValueError("nebular_grid_absorbed_log10: a channelled grid needs `weights`.")
        k = grid_abs.shape[0]
        sub = jax.lax.dynamic_slice(
            grid_abs, (jnp.zeros((), jnp.int32), ia, ib), (k, wa.shape[0], wb.shape[0])
        )
        per_channel = jnp.einsum("a,kab,b->k", wa, sub, wb)
        per_qh = jnp.dot(jnp.asarray(weights)[:k], per_channel)
    else:
        sub = jax.lax.dynamic_slice(grid_abs, (ia, ib), (wa.shape[0], wb.shape[0]))
        per_qh = jnp.einsum("a,ab,b->", wa, sub, wb)
    magnitude = jnp.abs(per_qh)
    nonzero = magnitude > 0
    safe = jnp.where(nonzero, magnitude, 1.0)
    log_abs = jnp.where(nonzero, jnp.asarray(log_nion) + jnp.log10(safe), -jnp.inf)
    return log_abs, jnp.sign(per_qh)

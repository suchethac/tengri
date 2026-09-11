# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the adaptive-axis nebular grid reconstructs line fluxes with variable
ionization (logU, gas-phase metallicity), and the axes adapt to which params are
free (#950).

Q_H-linearity is orthogonal to ionization, so freeing logU / logZ_gas / met just
adds interpolation axes; fixed params are baked. Validated: axes track
free_params (0-3 D), and Q_H*interp(grid) matches the exact Cue forward across
(logU, logZ_gas, SFH) draws. Interpolation is in log space (line luminosities
span decades across the ionization grid).

Data-gated (needs a bare SSP + cue_weights.npz); skips in CI.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.components.nebular.dig import mix_dig_grid_reconstruction
from tengri.components.nebular.line_precompute import _log10_four_pi_dl2
from tengri.components.nebular.nebular_grid_precompute import (
    precompute_nebular_grid,
    reconstruct_nebular_line_log_lums,
    reconstruct_nebular_line_lums,
    reconstruct_nebular_lines,
    reconstruct_nebular_phot,
)
from tengri.observation.line_flux_data import LineFluxData
from tengri.utils.scale import apply_log10_scale, pow10

pytestmark = pytest.mark.contract

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_LINES = ("Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717")
_LINE_DATA = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
_LW = _LINE_DATA.wavelengths
Z = 0.15


def _require():
    if not Path(_BARE).is_file():
        pytest.skip(f"missing bare SSP {_BARE}")
    if not Path("data/cue_weights.npz").is_file():
        pytest.skip("Cue weights (data/cue_weights.npz) not present")


def _model(neb, sfh_wild=FREE, met=None):
    import warnings

    _require()
    ssp = load_ssp_data(_BARE)
    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r"]), line_fluxes=_LINE_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs = dict(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": sfh_wild},
            dust_attenuation=None,
            neb=neb,
            redshift=Fixed(Z),
        )
        if met is not None:
            kwargs["met"] = met
        return SEDModel.build(**kwargs)


def _nion(m, p):
    return float(np.sum(np.asarray(m.predict_state(p).derived["nion"])))


def _log_nion(m, p):
    return float(np.asarray(m.predict_state(p).derived["log_nion"]))


_BANDS = ["galex_fuv", "galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1", "wise_w2"]


def _wave_model(neb, sfh_wild=FREE, met=None):
    """WavePrecomp Cue model with dust off — so it publishes nebular_phot_lnu_precomp."""
    import warnings

    _require()
    ssp = load_ssp_data(_BARE)
    obs = Observation(photometry=Photometry.from_names(_BANDS), line_fluxes=_LINE_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs = dict(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": sfh_wild},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_diff": Fixed(0.0),
                "tau_bc": Fixed(0.0),
            },
            neb=neb,
            redshift=Fixed(Z),
            approx=WavePrecomp(),
        )
        if met is not None:
            kwargs["met"] = met
        return SEDModel.build(**kwargs)


def test_axes_adapt_to_free_ionization():
    """The grid axes are exactly the free {met, logU, logZ_gas}, in order."""
    # both gas params fixed, SFH free -> met-only axis
    t0 = precompute_nebular_grid(
        _model({"type": "cue", "all_params": Fixed(DEFAULT)}, met={"logzsol": FREE}), _LW, n_grid=3
    )
    assert t0.axis_names == ("met_logzsol",), t0.axis_names
    # met + logU + logZ_gas all free -> 3 axes. The gas axes sit at exactly n_grid;
    # the met axis is n_grid uniform points PLUS the interior SSP metallicity nodes
    # (#1020), so it is larger and its length depends on the SSP grid, not on a
    # hard-coded factor.
    m3 = _model(
        {
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Uniform(-4.0, -1.0),
            "logZ_gas": Uniform(-1.0, 0.4),
        },
        met={"logzsol": FREE},
    )
    t3 = precompute_nebular_grid(m3, _LW, n_grid=3)
    assert t3.axis_names == ("met_logzsol", "neb_logU", "neb_logZ_gas"), t3.axis_names
    n_met = int(t3.axes[0].shape[0])
    assert n_met > 3, f"met axis should gain the SSP nodes, got {n_met}"
    assert t3.log_line_per_qh.shape == (n_met, 3, 3, len(_LINES)), t3.log_line_per_qh.shape
    # explicit per-axis dict sets the UNIFORM part; the met axis still gains the nodes
    t3d = precompute_nebular_grid(
        m3, _LW, n_grid={"met_logzsol": 5, "neb_logU": 3, "neb_logZ_gas": 4}
    )
    assert t3d.log_line_per_qh.shape[1:] == (3, 4, len(_LINES)), t3d.log_line_per_qh.shape
    assert int(t3d.axes[0].shape[0]) >= 5, t3d.axes[0].shape
    # ... and turning snapping off restores the plain densified uniform axis
    t3u = precompute_nebular_grid(m3, _LW, n_grid=3, snap_met_to_ssp_nodes=False)
    assert t3u.log_line_per_qh.shape == (6, 3, 3, len(_LINES)), t3u.log_line_per_qh.shape
    # met fixed (sfh 'all_params':FIXED fixes met), logU+logZ_gas free -> 2 axes
    m2 = _model(
        {
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Uniform(-4.0, -1.0),
            "logZ_gas": Uniform(-1.0, 0.4),
        },
        sfh_wild=Fixed(DEFAULT),
    )
    t2 = precompute_nebular_grid(m2, _LW, n_grid=3)
    assert t2.axis_names == ("neb_logU", "neb_logZ_gas"), t2.axis_names


def test_reconstruct_matches_exact_variable_ionization():
    """Q_H x interp(grid) matches the exact Cue forward across (logU, logZ_gas, SFH).

    2-axis (met fixed) — the 'mostly logU + gas metallicity' setup. Log-space
    node-exact PCHIP; strong DESI lines to < few percent on a 14-pt grid.
    """
    m = _model(
        {
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Uniform(-4.0, -1.0),
            "logZ_gas": Uniform(-1.0, 0.4),
        },
        sfh_wild=Fixed(DEFAULT),
    )
    table = precompute_nebular_grid(m, _LW, n_grid=14)
    assert table.axis_names == ("neb_logU", "neb_logZ_gas")
    worst = 0.0
    for i in range(6):
        p = dict(m.spec.sample(jax.random.PRNGKey(300 + i)))
        exact = np.asarray(m.predict_line_fluxes(p, target_wavelengths=_LW, redden=False))
        fast = np.asarray(reconstruct_nebular_lines(_nion(m, p), p, float(p["redshift"]), table))
        strong = np.abs(exact) > 1e-3 * np.max(np.abs(exact))
        rel = np.max(np.abs(fast - exact)[strong] / (np.abs(exact)[strong] + 1e-40))
        worst = max(worst, rel)
    assert worst < 3e-2, f"variable-ionization reconstruction off by {worst:.2e}"


def test_reconstruct_is_jittable_and_gradient_safe():
    """reconstruct is JIT + grad safe in logU (the fit-path requirement)."""
    m = _model(
        {"type": "cue", "all_params": Fixed(DEFAULT), "logU": Uniform(-4.0, -1.0)},
        sfh_wild=Fixed(DEFAULT),
    )
    table = precompute_nebular_grid(m, _LW, n_grid=8)
    assert table.axis_names == ("neb_logU",)
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    nion = _nion(m, p)

    def total(logU):
        return jnp.sum(reconstruct_nebular_lines(nion, dict(p, neb_logU=logU), Z, table))

    val = jax.jit(total)(jnp.asarray(-2.5))
    g = jax.jit(jax.grad(total))(jnp.asarray(-2.5))
    assert np.isfinite(float(val)) and np.isfinite(float(g))
    assert np.any(float(g) != 0.0), (
        "`float(g)` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


def test_axis_range_reads_prior_not_default():
    """The grid axis spans the PRIOR support, not the fallback default range.

    Regression: _axis_range read the wrong Uniform attributes (.low/.high) and
    silently fell back to _DEFAULT_RANGE for every prior — the 'grid adapts to the
    prior' claim was dead. A prior WIDER than the default (logU here) must widen
    the axis; the old bug would clamp it to the default (-4, -1).
    """
    m = _model(
        {"type": "cue", "all_params": Fixed(DEFAULT), "logU": Uniform(-5.0, 0.0)},
        sfh_wild=Fixed(DEFAULT),
    )
    table = precompute_nebular_grid(m, _LW, n_grid=4)
    ax = np.asarray(table.axes[table.axis_names.index("neb_logU")])
    assert ax.min() == pytest.approx(-5.0, abs=1e-6), f"axis min {ax.min()} != prior -5.0"
    assert ax.max() == pytest.approx(0.0, abs=1e-6), f"axis max {ax.max()} != prior 0.0"


def test_phot_channel_reconstructs_nebular_precomp():
    """Q_H x interp(phot_grid) matches the exact nebular_phot_lnu_precomp publish.

    The broadband analog of the line channel. A WavePrecomp model publishes
    the intrinsic filter-integrated nebular L_nu (``nebular_phot_lnu_precomp``,
    the key predict_via_precomp consumes). The grid captures it per Q_H at build
    time; the reconstruction is what the fast forward would publish instead of
    the per-eval Cue forward + filter integration.

    met FIXED + logU free (the 'sometimes met is fixed' sweet spot) — the grid is
    over the smooth gas axis, so the intrinsic-channel error stays tight.
    """
    m = _wave_model(
        {"type": "cue", "all_params": Fixed(DEFAULT), "logU": Uniform(-4.0, -1.0)},
        sfh_wild=Fixed(DEFAULT),
    )
    table = precompute_nebular_grid(m, _LW, n_grid=14)
    assert table.axis_names == ("neb_logU",), table.axis_names
    assert table.log_phot_per_qh is not None, "photometry channel missing"
    assert table.log_phot_per_qh.shape == (14, len(_BANDS)), table.log_phot_per_qh.shape

    worst = 0.0
    for i in range(6):
        p = dict(m.spec.sample(jax.random.PRNGKey(500 + i)))
        st = m.predict_state(p)
        exact = np.asarray(st.derived["nebular_phot_lnu_precomp"])  # rest-frame L_nu
        log_nion = float(np.asarray(st.derived["log_nion"]))
        fast = np.asarray(reconstruct_nebular_phot(log_nion, p, table))
        strong = np.abs(exact) > 1e-3 * np.max(np.abs(exact))
        rel = np.max(np.abs(fast - exact)[strong] / (np.abs(exact)[strong] + 1e-40))
        worst = max(worst, rel)
    assert worst < 3e-2, f"nebular photometry reconstruction off by {worst:.2e}"


def test_phot_channel_absent_without_wave_precomp():
    """A line-only grid (no WavePrecomp filters) has no photometry channel, and
    reconstruct_nebular_phot raises loudly rather than silently returning garbage."""
    m = _model(
        {"type": "cue", "all_params": Fixed(DEFAULT), "logU": Uniform(-4.0, -1.0)},
        sfh_wild=Fixed(DEFAULT),
    )
    table = precompute_nebular_grid(m, _LW, n_grid=4)
    assert table.log_phot_per_qh is None
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    with pytest.raises(ValueError, match="no photometry channel"):
        reconstruct_nebular_phot(_log_nion(m, p), p, table)


def test_unsnapped_met_axis_warns_and_snapped_one_does_not():
    """A uniform met axis straddles the SSP-node kinks — that must warn (#1020).

    The exact per-Q_H emissivity is C0 at every ``ssp_lgmet`` node (the ionizing-
    spectrum tables interpolate bilinearly in met), so a uniform axis converges
    only as O(h) on the collisionally-excited lines. Snapping is the default and
    must be silent; opting out must not be.
    """
    import warnings as _w

    m = _model(
        {"type": "cue", "all_params": Fixed(DEFAULT), "logU": Uniform(-4.0, -1.0)},
        sfh_wild=FREE,
        met={"logzsol": FREE},
    )

    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        t = precompute_nebular_grid(m, _LW, n_grid=3, snap_met_to_ssp_nodes=False)
    assert "met_logzsol" in t.axis_names, "fixture should free met"
    assert any("met_logzsol" in str(x.message) and "UNIFORM" in str(x.message) for x in rec), (
        "expected a met-axis under-resolution warning on the unsnapped axis"
    )

    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        precompute_nebular_grid(m, _LW, n_grid=3)  # snapped: the default
    assert not [x for x in rec if "met_logzsol" in str(x.message)], (
        "the snapped met axis resolves the kinks — it must not warn"
    )


def test_met_axis_snaps_to_ssp_nodes_and_interpolates_linearly():
    """Knots land on every interior SSP metallicity node, and that axis is linear.

    Regression for #1020. Node-exactness alone is not enough: PCHIP estimates the
    tangent at a knot from BOTH sides, so at a C0 kink it is wrong by O(1) and the
    neighboring cells decay only as O(h). The met axis must therefore be flagged
    ``'linear'`` while the smooth gas axes stay ``'pchip'``.
    """
    from tengri.components.nebular.nebular_grid_precompute import _ssp_met_nodes

    m = _model(
        {"type": "cue", "all_params": Fixed(DEFAULT), "logU": Uniform(-4.0, -1.0)},
        sfh_wild=FREE,
        met={"logzsol": FREE},
    )
    t = precompute_nebular_grid(m, _LW, n_grid=4)

    assert t.axis_names[0] == "met_logzsol"
    assert t.axis_kinds == ("linear", "pchip"), t.axis_kinds

    axis = np.asarray(t.axes[0])
    nodes = _ssp_met_nodes(m)
    interior = nodes[(nodes > axis[0]) & (nodes < axis[-1])]
    assert interior.size > 0, "fixture must span several SSP metallicity nodes"
    for node in interior:
        assert np.min(np.abs(axis - node)) < 1e-9, f"SSP node {node:.4f} is not a grid knot"
    assert np.all(np.diff(axis) > 0), "axis must be strictly ascending"


def test_gas_only_axes_do_not_warn():
    """met FIXED + gas axes free (the reliable production config) — no met warning."""
    import warnings as _w

    m = _model(
        {
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Uniform(-4.0, -1.0),
            "logZ_gas": Uniform(-1.0, 0.4),
        },
        sfh_wild=Fixed(DEFAULT),
    )
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        t = precompute_nebular_grid(m, _LW, n_grid=3)
    assert t.axis_names == ("neb_logU", "neb_logZ_gas")
    assert not any("not reliably" in str(x.message).lower() for x in rec)


@pytest.mark.slow
def test_snapped_met_axis_beats_uniform_on_a_dense_sweep():
    """Accuracy contract for #1020, measured the only way that works.

    A **dense sweep strictly inside** the grid bounds. Random draws under-sample
    narrow features (they once made a 10 % met-axis error look like 0.09 %), and
    sweeping past a bound clamps and manufactures a fake resolution-independent
    error. Worst-case relative error over the sweep is the quantity that matters.

    The snapped + linear axis must beat the uniform + PCHIP axis on the shape-
    sensitive [OIII] line **while using no more grid points** — the snapped axis
    resolves the SSP-node kinks, so the cubic's cross-kink tangent error is gone.
    """
    m = _model(
        {"type": "cue", "all_params": Fixed(DEFAULT)},
        sfh_wild=FREE,
        met={"logzsol": FREE},
    )
    assert "met_logzsol" in m.spec.free_params
    lo, hi = -1.8, 0.2
    rng = {"met_logzsol": (lo, hi)}
    base = dict(m.spec.sample(jax.random.PRNGKey(0)))
    o3 = _LINES.index("OIII_5007")

    snapped = precompute_nebular_grid(
        m, _LW, n_grid={"met_logzsol": 16}, ranges=rng, ref_params=base
    )
    with pytest.warns(UserWarning, match="UNIFORM"):
        uniform = precompute_nebular_grid(
            m,
            _LW,
            n_grid={"met_logzsol": int(snapped.axes[0].shape[0])},
            ranges=rng,
            ref_params=base,
            snap_met_to_ssp_nodes=False,
        )
    assert uniform.axes[0].shape[0] >= snapped.axes[0].shape[0], "uniform must not be handicapped"

    worst = {"snapped": 0.0, "uniform": 0.0}
    for met in np.linspace(lo + 0.05, hi - 0.05, 121):  # strictly inside
        p = {**base, "met_logzsol": jnp.asarray(float(met))}
        exact = np.asarray(m.predict_line_fluxes(p, target_wavelengths=_LW, redden=False))[o3]
        nion = _nion(m, p)
        for tag, table in (("snapped", snapped), ("uniform", uniform)):
            got = np.asarray(reconstruct_nebular_line_lums(nion, p, table))[o3]
            # luminosity -> observed flux; the ~1e57 divisor stays an exponent (#1859)
            got = np.asarray(apply_log10_scale(got, -_log10_four_pi_dl2(float(p["redshift"]))))
            worst[tag] = max(worst[tag], abs(got - exact) / max(abs(exact), 1e-40))

    assert worst["snapped"] < worst["uniform"], (
        f"snapped met axis ({worst['snapped']:.2%}) must beat uniform "
        f"({worst['uniform']:.2%}) at equal size"
    )
    assert worst["snapped"] < 0.01, f"[OIII] worst-case {worst['snapped']:.2%} exceeds 1 %"


# ── DIG mixing on the grid (#2222) ──────────────────────────────────────────
#
# Before #2222, arming the grid with an active ``neb_dig_frac`` (free, or
# fixed non-zero) raised ``DIGNotOnNebularGridError``: the grid tabulated one
# photoionization regime and had no axis to represent a second one. The grid
# now extends ``neb_logU``'s axis (never clips, never refuses) to also cover
# the DIG-shifted query point ``neb_logU + neb_dig_delta_logU``, and mixes two
# lookups against the SAME table via ``dig.mix_dig_grid_reconstruction``. See
# ``nebular_grid_precompute._dig_may_be_active`` /
# ``_dig_extended_logU_range`` / ``_preserve_spacing_n``.
#
# The table itself always stores the undiluted HII term per node (build-time
# forwards force ``neb_dig_frac = 0.0``, see the comment on that line in
# ``precompute_nebular_grid``): DIG mixing is applied only at reconstruction.


def _reconstruct_line_flux(nion, point, table):
    """3-arg adapter: ``reconstruct_nebular_lines`` at this module's fixed ``Z``.

    ``mix_dig_grid_reconstruction`` expects a 3-arg ``reconstruct(amplitude,
    point, table)`` (the shape ``reconstruct_nebular_phot`` /
    ``reconstruct_nebular_restband`` / ``reconstruct_nebular_line_lums``
    already have); ``reconstruct_nebular_lines`` additionally takes a
    redshift, so this binds it at the fixture's fixed ``Z``, giving observed
    flux instead of intrinsic luminosity (matching what
    ``model.predict_line_fluxes(..., redden=False)`` returns).
    """
    return reconstruct_nebular_lines(nion, point, Z, table)


def _reconstruct_line_flux_log(log_nion, point, table):
    """Log10-domain 3-arg adapter: log10 observed flux at this module's fixed ``Z``.

    The log-domain analog of :func:`_reconstruct_line_flux`: mirrors
    ``reconstruct_nebular_lines``'s own body (log10 intrinsic luminosity via
    ``reconstruct_nebular_line_log_lums``, then subtract the log10 cosmology
    divisor) without the final ``pow10``, so ``mix_dig_grid_reconstruction``
    can mix the HII and DIG evaluations in log space (``log_domain=True``) --
    the branch ``predict_line_fluxes`` actually takes on the shipped
    ``FeaturePrecomp`` fast path (#2263 review I1: the linear adapter above
    exercises a branch production code never takes for lines).
    """
    return reconstruct_nebular_line_log_lums(log_nion, point, table) - _log10_four_pi_dl2(Z)


def _worst_rel(fast, exact):
    """Worst-case relative error, ignoring entries below 0.1 % of the peak."""
    fast = np.asarray(fast)
    exact = np.asarray(exact)
    strong = np.abs(exact) > 1e-3 * np.max(np.abs(exact))
    return float(np.max(np.abs(fast - exact)[strong] / (np.abs(exact)[strong] + 1e-40)))


def _dig_parity(m, table, *, n_seeds, seed0):
    """Worst-case (photometry, lines) relative error of the two-lookup DIG mix.

    Compares ``mix_dig_grid_reconstruction`` against ``m``'s own exact-path
    forward (``mix_dig_emission`` / ``mix_dig_line_luminosities``, unaffected
    by #2222), sampling ``m.spec`` (so ``neb_dig_frac`` / ``neb_dig_delta_logU``
    take whatever disposition -- Fixed or FREE -- ``m`` was built with).

    The line channel goes through ``_reconstruct_line_flux_log`` with
    ``log_domain=True`` (#2263 review I1): that is the branch
    ``predict_line_fluxes`` actually takes on the shipped ``FeaturePrecomp``
    fast path (``reconstruct_nebular_line_log_lums`` mixed via
    :func:`~tengri.components.nebular.dig._log10_weighted_mix`), not the
    linear ``reconstruct_nebular_lines`` / ``_linear_mix`` pair the plain
    ``_reconstruct_line_flux`` adapter exercises.
    """
    worst_phot = worst_line = 0.0
    for i in range(n_seeds):
        p = dict(m.spec.sample(jax.random.PRNGKey(seed0 + i)))
        st = m.predict_state(p)
        exact_phot = np.asarray(st.derived["nebular_phot_lnu_precomp"])
        log_nion = float(np.asarray(st.derived["log_nion"]))
        fast_phot = np.asarray(
            mix_dig_grid_reconstruction(
                reconstruct_nebular_phot,
                log_nion,
                p,
                table,
                neb_dig_frac=p["neb_dig_frac"],
                neb_dig_delta_logU=p["neb_dig_delta_logU"],
            )
        )
        worst_phot = max(worst_phot, _worst_rel(fast_phot, exact_phot))

        exact_line = np.asarray(
            m.predict_line_fluxes(p, target_wavelengths=_LW, redden=False, state=st)
        )
        fast_line = np.asarray(
            pow10(
                mix_dig_grid_reconstruction(
                    _reconstruct_line_flux_log,
                    log_nion,
                    p,
                    table,
                    neb_dig_frac=p["neb_dig_frac"],
                    neb_dig_delta_logU=p["neb_dig_delta_logU"],
                    log_domain=True,
                )
            )
        )
        worst_line = max(worst_line, _worst_rel(fast_line, exact_line))
    return worst_phot, worst_line


def test_reconstruct_matches_exact_with_dig_mixing_fixed_frac():
    r"""Two-lookup grid DIG mix matches the exact DIG-mixed forward (#2222).

    ``neb_dig_frac = 0.3`` fixed, ``neb_dig_delta_logU = -1.0`` fixed,
    ``neb_logU`` the only free axis (``Uniform(-4, -1)``), auto-extended to
    ``(-5, -1)`` (14 nodes to 19) because DIG could be active. Measured
    worst-case relative error over 10 seeds against the exact path (this
    fixture, FSPS/MILES, dpl SFH, z = 0.15): 1.17e-3 (photometry) / 1.41e-3
    (lines), inside this module's 3e-2 parity ceiling and no worse than the
    DIG-absent baseline on the SAME fixture (``neb_dig_frac`` pinned at its
    declared 0.0, un-extended 14-node axis): 1.72e-3 / 2.04e-3.
    """
    neb = {
        "type": "cue",
        "all_params": Fixed(DEFAULT),
        "logU": Uniform(-4.0, -1.0),
        "dig_frac": Fixed(0.3),
        "dig_delta_logU": Fixed(-1.0),
    }
    m = _wave_model(neb, sfh_wild=Fixed(DEFAULT))
    table = precompute_nebular_grid(m, _LW, n_grid=14)
    assert table.axis_names == ("neb_logU",), table.axis_names
    ax = np.asarray(table.axes[0])
    assert ax.min() == pytest.approx(-5.0, abs=1e-6), f"axis min {ax.min()} != -5.0"

    worst_phot, worst_line = _dig_parity(m, table, n_seeds=10, seed0=700)
    assert worst_phot < 3e-2, f"DIG-mixed photometry reconstruction off by {worst_phot:.2e}"
    assert worst_line < 3e-2, f"DIG-mixed line reconstruction off by {worst_line:.2e}"


def test_reconstruct_matches_exact_with_dig_mixing_free_frac():
    """As above, but ``neb_dig_frac`` is FREE and sampled per draw, not pinned.

    Confirms the two-lookup mix and its axis extension hold across the whole
    ``neb_dig_frac`` prior, not merely at one fixed value.
    """
    neb = {
        "type": "cue",
        "all_params": Fixed(DEFAULT),
        "logU": Uniform(-4.0, -1.0),
        "dig_frac": FREE,
        "dig_delta_logU": Fixed(-1.0),
    }
    m = _wave_model(neb, sfh_wild=Fixed(DEFAULT))
    table = precompute_nebular_grid(m, _LW, n_grid=14)
    assert table.axis_names == ("neb_logU",), table.axis_names

    worst_phot, worst_line = _dig_parity(m, table, n_seeds=10, seed0=800)
    assert worst_phot < 3e-2, f"DIG-mixed photometry reconstruction off by {worst_phot:.2e}"
    assert worst_line < 3e-2, f"DIG-mixed line reconstruction off by {worst_line:.2e}"


def test_dig_extended_axis_preserves_zero_frac_parity():
    """Extending ``neb_logU`` for DIG must not degrade ``neb_dig_frac = 0`` parity.

    Widening the axis to cover the DIG-shifted query point, at the SAME node
    count, would thin the nodes covering the original HII-only region and
    degrade this baseline purely because DIG *could* be active elsewhere in
    the spec (both ``neb_dig_frac`` and ``neb_dig_delta_logU`` are FREE here,
    the widest extension this fixture can produce: axis widens from
    ``(-4, -1)``/14 nodes to ``(-8, -1)``/32 nodes). ``_preserve_spacing_n``
    scales the node count to hold the pre-extension spacing fixed, so parity
    at a query with ``neb_dig_frac`` forced to a Python ``0.0`` must match the
    DIG-absent baseline (1.72e-3 photometry / 2.04e-3 lines, un-extended
    14-node axis, ``test_reconstruct_matches_exact_with_dig_mixing_fixed_frac``'s
    docstring), the spacing rule #2222 requires. Measured here: 4.6e-4 / 1.35e-3
    -- as tight as, or tighter than, the baseline.
    """
    neb = {
        "type": "cue",
        "all_params": Fixed(DEFAULT),
        "logU": Uniform(-4.0, -1.0),
        "dig_frac": FREE,
        "dig_delta_logU": FREE,
    }
    m = _wave_model(neb, sfh_wild=Fixed(DEFAULT))
    table = precompute_nebular_grid(m, _LW, n_grid=14)
    assert table.axis_names == ("neb_logU",), table.axis_names
    ax = np.asarray(table.axes[0])
    assert ax.min() < -4.0, "axis should be extended: dig_frac is free here"

    worst_phot = worst_line = 0.0
    for i in range(6):
        p = dict(m.spec.sample(jax.random.PRNGKey(900 + i)))
        p["neb_dig_frac"] = 0.0  # Python literal: forces the short-circuit
        st = m.predict_state(p)
        exact_phot = np.asarray(st.derived["nebular_phot_lnu_precomp"])
        log_nion = float(np.asarray(st.derived["log_nion"]))
        fast_phot = np.asarray(reconstruct_nebular_phot(log_nion, p, table))
        worst_phot = max(worst_phot, _worst_rel(fast_phot, exact_phot))

        exact_line = np.asarray(
            m.predict_line_fluxes(p, target_wavelengths=_LW, redden=False, state=st)
        )
        fast_line = np.asarray(_reconstruct_line_flux(_nion(m, p), p, table))
        worst_line = max(worst_line, _worst_rel(fast_line, exact_line))

    assert worst_phot < 3e-2, f"zero-frac photometry parity degraded to {worst_phot:.2e}"
    assert worst_line < 3e-2, f"zero-frac line parity degraded to {worst_line:.2e}"


def test_dig_extends_logU_axis_and_includes_fixed_logU():
    """``neb_logU``'s axis extends for DIG, and joins the axes even when Fixed.

    With ``neb_logU`` ``Uniform(-4, -1)`` and ``neb_dig_delta_logU`` fixed at
    -1.0, the built axis low end is -5.0 (own range union the DIG-shifted
    range). With ``neb_logU`` ``Fixed(-2.5)`` and DIG active, ``neb_logU``
    still joins ``axis_names``: the DIG lookup always needs a second query
    point distinct from the HII one.
    """
    m_free = _model(
        {
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Uniform(-4.0, -1.0),
            "dig_frac": Fixed(0.3),
            "dig_delta_logU": Fixed(-1.0),
        },
        sfh_wild=Fixed(DEFAULT),
    )
    table_free = precompute_nebular_grid(m_free, _LW, n_grid=14)
    assert table_free.axis_names == ("neb_logU",)
    ax = np.asarray(table_free.axes[0])
    assert ax.min() == pytest.approx(-5.0, abs=1e-6), f"axis min {ax.min()} != -5.0"
    assert ax.max() == pytest.approx(-1.0, abs=1e-6), f"axis max {ax.max()} != -1.0"

    m_fixed_logu = _model(
        {
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Fixed(-2.5),
            "dig_frac": Fixed(0.3),
            "dig_delta_logU": Fixed(-1.0),
        },
        sfh_wild=Fixed(DEFAULT),
    )
    assert m_fixed_logu.spec.fixed_value("neb_logU") == pytest.approx(-2.5)
    table_fixed = precompute_nebular_grid(m_fixed_logu, _LW, n_grid=14)
    assert "neb_logU" in table_fixed.axis_names, (
        "neb_logU must join the axes purely because DIG could be active, "
        "even though it is itself Fixed"
    )


def test_grid_line_fluxes_match_full_params_dict_when_keys_are_omitted():
    r"""Omitting a Fixed key from params must not change grid-path line fluxes.

    #2222 review I1. ``neb_logU`` Fixed(-2.0), ``neb_dig_frac`` Fixed(0.3),
    ``neb_dig_delta_logU`` Fixed(-1.0), served through
    :meth:`~tengri.SEDModel.enable_fast_nebular`. Before the fix, dropping
    any of the three keys from the params dict substituted a registry-default
    literal for the model's own Fixed value in
    ``SEDModel.predict_line_fluxes``'s grid branch -- measured 9.323e-1
    (``neb_logU`` dropped) / 3.996e-1 (``neb_dig_frac`` dropped) worst-case
    relative error, silently, because the exact path's own
    ``full_params = {**fixed_values, **params}`` merge (``predict_state``)
    makes the identical dict *correct* there. The fix reads the same merged
    dict on the grid path (``self.spec.get_fixed_values()``), so a dict
    omitting any Fixed key is bit-identical to the full dict.
    """
    neb = {
        "type": "cue",
        "all_params": Fixed(DEFAULT),
        "logU": Fixed(-2.0),
        "dig_frac": Fixed(0.3),
        "dig_delta_logU": Fixed(-1.0),
    }
    m = _model(neb, sfh_wild=Fixed(DEFAULT))
    m.enable_fast_nebular(_LW, n_grid=14)

    full_p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    full_lums = np.asarray(m.predict_line_fluxes(full_p, target_wavelengths=_LW, redden=False))
    for drop in ("neb_logU", "neb_dig_frac", "neb_dig_delta_logU"):
        stripped = {k: v for k, v in full_p.items() if k != drop}
        lums = np.asarray(m.predict_line_fluxes(stripped, target_wavelengths=_LW, redden=False))
        np.testing.assert_allclose(
            lums,
            full_lums,
            rtol=1e-12,
            atol=0.0,
            err_msg=f"dropping {drop!r} from params changed the grid-path line fluxes",
        )

    exact_m = _model(neb, sfh_wild=Fixed(DEFAULT))
    exact_lums = np.asarray(
        exact_m.predict_line_fluxes(full_p, target_wavelengths=_LW, redden=False)
    )
    rel = _worst_rel(full_lums, exact_lums)
    assert rel < 3e-2, f"grid vs exact line fluxes off by {rel:.2e}"


def test_explicit_ranges_still_extends_for_dig_and_parity_holds():
    r"""An explicit ``ranges['neb_logU']`` must not bypass the DIG extension.

    #2222 review I2. ``ranges={'neb_logU': (-4, -1)}`` is the prior support
    -- the documented default and the natural thing to write before DIG is
    armed. Before the fix this bypassed the extension entirely and
    ``interp_nd_pchip`` clipped the DIG query silently: measured 5.399e-2
    worst-case relative line-flux error, above this module's own 3e-2 parity
    ceiling. The fix applies the same Minkowski-sum extension on top of a
    user-supplied range that it applies on top of the prior support, so the
    built axis low end is still -5.0 and parity still holds.
    """
    neb = {
        "type": "cue",
        "all_params": Fixed(DEFAULT),
        "logU": Uniform(-4.0, -1.0),
        "dig_frac": Fixed(0.3),
        "dig_delta_logU": Fixed(-1.0),
    }
    m = _wave_model(neb, sfh_wild=Fixed(DEFAULT))
    table = precompute_nebular_grid(m, _LW, n_grid=14, ranges={"neb_logU": (-4.0, -1.0)})
    ax = np.asarray(table.axes[table.axis_names.index("neb_logU")])
    assert ax.min() == pytest.approx(-5.0, abs=1e-6), f"axis min {ax.min()} != -5.0"

    worst_phot, worst_line = _dig_parity(m, table, n_seeds=6, seed0=1100)
    assert worst_phot < 3e-2, f"DIG-mixed photometry (explicit ranges=) off by {worst_phot:.2e}"
    assert worst_line < 3e-2, f"DIG-mixed line (explicit ranges=) off by {worst_line:.2e}"


def test_degenerate_dig_extension_drops_the_axis_and_matches_hii():
    r"""``neb_logU`` Fixed + ``neb_dig_delta_logU`` Fixed(0.0) needs no axis.

    #2222 review I3; the grid twin of
    ``test_bug_2195_...::test_pure_dig_with_no_offset_is_the_hii_solution``.
    Before the fix, the Minkowski-sum extension collapsed to a single point
    (``own_lo == own_hi == v``, ``delta_lo == delta_hi == 0.0``: the HII and
    DIG query points coincide), giving an all-identical axis and a **NaN**
    reconstruction from ``interp_nd_pchip``'s degenerate PCHIP slopes, with
    no error raised, where #2195 used to raise ``DIGNotOnNebularGridError``.
    ``neb_dig_frac = 1.0`` (pure DIG) with ``delta = 0.0`` means the DIG term
    equals the HII term everywhere, so the correct reconstruction is exactly
    the pure-HII solution.
    """
    m = _model(
        {
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Fixed(-3.0),
            "dig_frac": Fixed(1.0),
            "dig_delta_logU": Fixed(0.0),
        },
        sfh_wild=Fixed(DEFAULT),
    )
    table = precompute_nebular_grid(m, _LW, n_grid=6)
    assert "neb_logU" not in table.axis_names, (
        "a degenerate DIG extension must drop neb_logU from the axes, not "
        f"build an all-identical one: {table.axis_names}"
    )

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    nion = _nion(m, p)
    recon = np.asarray(reconstruct_nebular_line_lums(nion, p, table))
    assert np.all(np.isfinite(recon)), f"reconstruction is not finite: {recon}"

    m_hii = _model(
        {
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Fixed(-3.0),
            "dig_frac": Fixed(0.0),
        },
        sfh_wild=Fixed(DEFAULT),
    )
    exact_hii = np.asarray(m_hii.predict_line_fluxes(p, target_wavelengths=_LW, redden=False))
    recon_flux = np.asarray(apply_log10_scale(recon, -_log10_four_pi_dl2(Z)))
    rel = _worst_rel(recon_flux, exact_hii)
    assert rel < 1e-10, f"degenerate-DIG reconstruction vs pure HII off by {rel:.2e}"


def test_dig_grid_reconstruction_is_jittable_and_gradient_safe():
    """jit + grad through the two-lookup mix, w.r.t. both DIG parameters (#2222).

    Finite is not enough: a clipped or otherwise inert parameter gives a
    gradient of exactly zero, which is finite -- and a DIG parameter the
    likelihood cannot see is #2195's disease (#2100 gradient-assertion
    convention). Both gradients are asserted nonzero too (measured
    ``d/d(delta) = 5.71e-17`` at ``delta = -1.0``, ``frac = 0.3`` -- small in
    absolute terms but 2.0e-2 relative, review M6).
    """
    m = _model(
        {
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Uniform(-4.0, -1.0),
            "dig_frac": Fixed(0.3),
            "dig_delta_logU": Fixed(-1.0),
        },
        sfh_wild=Fixed(DEFAULT),
    )
    table = precompute_nebular_grid(m, _LW, n_grid=14)
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    nion = _nion(m, p)

    def total_delta(delta):
        return jnp.sum(
            mix_dig_grid_reconstruction(
                _reconstruct_line_flux,
                nion,
                p,
                table,
                neb_dig_frac=jnp.asarray(0.3),
                neb_dig_delta_logU=delta,
            )
        )

    val = jax.jit(total_delta)(jnp.asarray(-1.0))
    g = jax.jit(jax.grad(total_delta))(jnp.asarray(-1.0))
    assert np.isfinite(float(val)) and np.isfinite(float(g))
    assert float(g) != 0.0, (
        "d/d(neb_dig_delta_logU) is identically zero -- finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )

    def total_frac(frac):
        return jnp.sum(
            mix_dig_grid_reconstruction(
                _reconstruct_line_flux,
                nion,
                p,
                table,
                neb_dig_frac=frac,
                neb_dig_delta_logU=jnp.asarray(-1.0),
            )
        )

    val2 = jax.jit(total_frac)(jnp.asarray(0.3))
    g2 = jax.jit(jax.grad(total_frac))(jnp.asarray(0.3))
    assert np.isfinite(float(val2)) and np.isfinite(float(g2))
    assert float(g2) != 0.0, (
        "d/d(neb_dig_frac) is identically zero -- finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


def test_zero_frac_calls_interpolator_once_per_channel(monkeypatch):
    """A Python-literal ``neb_dig_frac = 0.0`` must not pay a second interpolation.

    The short-circuit lives in ``dig.py``'s ``_mix_dig_backend_evaluations``;
    this counts calls into ``interp_nd_pchip`` to prove
    ``mix_dig_grid_reconstruction`` actually skips the DIG lookup rather than
    computing and discarding it, when called with a genuine Python ``0.0``.

    **Function-level only.** End to end (``SEDModel.build`` ->
    ``predict_photometry`` / ``predict_line_fluxes``), ``params`` is always a
    JAX array or tracer by the time it reaches this call -- never a Python
    float -- so the short-circuit does not fire there today, mirroring #2221's
    own measured finding for the exact path (its ``M2`` review note). Measured
    on a ``Fixed(0.0)`` ``FeaturePrecomp`` cue model, no dust, WavePrecomp
    photometry + 2 target lines: ``predict_photometry`` makes 4 interpolator
    calls (2 channels x 2 lookups) and ``predict_line_fluxes`` makes 2 (1
    channel x 2 lookups) -- twice what this test demonstrates is possible.
    Not fixed here -- tracked as #2262, alongside #2221's identical gap for
    the exact path; this test only proves the mixing function's own
    short-circuit contract.
    """
    import tengri.components.nebular.nebular_grid_precompute as ngp

    neb = {
        "type": "cue",
        "all_params": Fixed(DEFAULT),
        "logU": Uniform(-4.0, -1.0),
        "dig_frac": Fixed(0.3),
        "dig_delta_logU": Fixed(-1.0),
    }
    m = _wave_model(neb, sfh_wild=Fixed(DEFAULT))
    table = precompute_nebular_grid(m, _LW, n_grid=14)
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    nion = _nion(m, p)
    log_nion = _log_nion(m, p)

    calls = []
    real = ngp.interp_nd_pchip

    def _counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ngp, "interp_nd_pchip", _counting)

    for reconstruct, amplitude in (
        (ngp.reconstruct_nebular_phot, log_nion),
        (ngp.reconstruct_nebular_line_lums, nion),
    ):
        calls.clear()
        mix_dig_grid_reconstruction(
            reconstruct, amplitude, p, table, neb_dig_frac=0.0, neb_dig_delta_logU=-1.0
        )
        assert len(calls) == 1, (
            f"{reconstruct.__name__}: expected exactly one interpolator call at "
            f"neb_dig_frac=0.0, got {len(calls)}"
        )

        calls.clear()
        mix_dig_grid_reconstruction(
            reconstruct, amplitude, p, table, neb_dig_frac=0.3, neb_dig_delta_logU=-1.0
        )
        assert len(calls) == 2, (
            f"{reconstruct.__name__}: expected two interpolator calls at "
            f"neb_dig_frac=0.3, got {len(calls)}"
        )

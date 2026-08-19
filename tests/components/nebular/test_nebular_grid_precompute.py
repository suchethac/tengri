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
    FIXED,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.components.nebular.line_precompute import _log10_four_pi_dl2
from tengri.components.nebular.nebular_grid_precompute import (
    precompute_nebular_grid,
    reconstruct_nebular_line_lums,
    reconstruct_nebular_lines,
    reconstruct_nebular_phot,
)
from tengri.observation.line_flux_data import LineFluxData
from tengri.utils.scale import apply_log10_scale

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
            sfh={"type": "dpl", "*": sfh_wild},
            dust=None,
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
            sfh={"type": "dpl", "*": sfh_wild},
            dust={
                "law_diff": "calzetti",
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
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
        _model({"type": "cue", "*": FIXED}, met={"logzsol": FREE}), _LW, n_grid=3
    )
    assert t0.axis_names == ("met_logzsol",), t0.axis_names
    # met + logU + logZ_gas all free -> 3 axes. The gas axes sit at exactly n_grid;
    # the met axis is n_grid uniform points PLUS the interior SSP metallicity nodes
    # (#1020), so it is larger and its length depends on the SSP grid, not on a
    # hard-coded factor.
    m3 = _model(
        {"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0), "logZ_gas": Uniform(-1.0, 0.4)},
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
    # met fixed (sfh '*':FIXED fixes met), logU+logZ_gas free -> 2 axes
    m2 = _model(
        {"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0), "logZ_gas": Uniform(-1.0, 0.4)},
        sfh_wild=FIXED,
    )
    t2 = precompute_nebular_grid(m2, _LW, n_grid=3)
    assert t2.axis_names == ("neb_logU", "neb_logZ_gas"), t2.axis_names


def test_reconstruct_matches_exact_variable_ionization():
    """Q_H x interp(grid) matches the exact Cue forward across (logU, logZ_gas, SFH).

    2-axis (met fixed) — the 'mostly logU + gas metallicity' setup. Log-space
    node-exact PCHIP; strong DESI lines to < few percent on a 14-pt grid.
    """
    m = _model(
        {"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0), "logZ_gas": Uniform(-1.0, 0.4)},
        sfh_wild=FIXED,
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
    m = _model({"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0)}, sfh_wild=FIXED)
    table = precompute_nebular_grid(m, _LW, n_grid=8)
    assert table.axis_names == ("neb_logU",)
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    nion = _nion(m, p)

    def total(logU):
        return jnp.sum(reconstruct_nebular_lines(nion, dict(p, neb_logU=logU), Z, table))

    val = jax.jit(total)(jnp.asarray(-2.5))
    g = jax.jit(jax.grad(total))(jnp.asarray(-2.5))
    assert np.isfinite(float(val)) and np.isfinite(float(g))


def test_axis_range_reads_prior_not_default():
    """The grid axis spans the PRIOR support, not the fallback default range.

    Regression: _axis_range read the wrong Uniform attributes (.low/.high) and
    silently fell back to _DEFAULT_RANGE for every prior — the 'grid adapts to the
    prior' claim was dead. A prior WIDER than the default (logU here) must widen
    the axis; the old bug would clamp it to the default (-4, -1).
    """
    m = _model({"type": "cue", "*": FIXED, "logU": Uniform(-5.0, 0.0)}, sfh_wild=FIXED)
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
    m = _wave_model({"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0)}, sfh_wild=FIXED)
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
    m = _model({"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0)}, sfh_wild=FIXED)
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
        {"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0)},
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
        {"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0)},
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
        {"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0), "logZ_gas": Uniform(-1.0, 0.4)},
        sfh_wild=FIXED,
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
    m = _model({"type": "cue", "*": FIXED}, sfh_wild=FREE)
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

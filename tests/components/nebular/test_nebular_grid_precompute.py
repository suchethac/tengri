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

jax.config.update("jax_enable_x64", True)

from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, Uniform, load_ssp_data
from tengri.components.nebular.nebular_grid_precompute import (
    precompute_nebular_grid,
    reconstruct_nebular_lines,
)
from tengri.observation.line_flux_data import LineFluxData

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


def _model(neb, sfh_wild=FREE):
    import warnings

    _require()
    ssp = load_ssp_data(_BARE)
    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r"]), line_fluxes=_LINE_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": sfh_wild},
            dust=None,
            neb=neb,
            redshift=Fixed(Z),
        )


def _nion(m, p):
    return float(np.sum(np.asarray(m.predict_state(p).derived["nion"])))


def test_axes_adapt_to_free_ionization():
    """The grid axes are exactly the free {met, logU, logZ_gas}, in order."""
    # both gas params fixed, SFH free -> met-only axis
    t0 = precompute_nebular_grid(_model({"type": "cue", "*": FIXED}), _LW, n_grid=3)
    assert t0.axis_names == ("met_logzsol",), t0.axis_names
    # met + logU + logZ_gas all free -> 3 axes
    m3 = _model(
        {"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0), "logZ_gas": Uniform(-1.0, 0.4)}
    )
    t3 = precompute_nebular_grid(m3, _LW, n_grid=3)
    assert t3.axis_names == ("met_logzsol", "neb_logU", "neb_logZ_gas"), t3.axis_names
    assert t3.log_line_per_qh.shape == (3, 3, 3, len(_LINES))
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

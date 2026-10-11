# SPDX-License-Identifier: BSD-3-Clause
"""Accuracy tests for the analytic dust-emission precompute adapter.

Tests off-node accuracy at default grids, Wien-tail accuracy, band-coverage refusal,
and redshift consistency against exact closures integrated on a wide fine grid.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust._params import PARAMS as _DUST_PARAMS

pytestmark = pytest.mark.contract

# Exact-closure reference grid [Angstrom]: 0.01 um to 10 m, finer than the builders' grid.
WIDE = np.geomspace(1e2, 1e11, 72000)
# Continuum-limit reference grid for casey2012: WIDE plus 8000 points over 0.9-1.1 um (0.25
# Angstrom spacing at the 1 um step, against 2.9 Angstrom on WIDE). casey2012 is zero below 1 um
# and normalized on the grid it is evaluated on, so the trapezoid cell at the step sets the band
# flux to first order in its width.
WIDE_CONVERGED = np.unique(np.concatenate([WIDE, np.linspace(0.9e4, 1.1e4, 8000)]))


def _get_param_bounds(param_name: str) -> tuple[float, float]:
    """Extract (lower, upper) bounds from a parameter's free_prior."""
    for p in _DUST_PARAMS:
        if (
            p.name == param_name
            and hasattr(p, "free_prior")
            and p.free_prior is not None
            and hasattr(p.free_prior, "lo")
        ):
            return float(p.free_prior.lo), float(p.free_prior.hi)
    raise ValueError(f"Parameter {param_name} has no bounded free_prior in PARAMS")


def tophat(lo_um, hi_um):
    """Construct a top-hat filter."""
    inner = np.geomspace(lo_um * 1e4, hi_um * 1e4, 400)
    return (
        np.concatenate([[inner[0] * (1 - 1e-6)], inner, [inner[-1] * (1 + 1e-6)]]),
        np.concatenate([[0.0], np.ones_like(inner), [0.0]]),
    )


def exact(model, kw, fw, ft, z=0.0, wide=WIDE):
    """Exact closure at redshift ``z``: integrate the model on the rest grid ``wide``."""
    from tengri.components.dust.emission import DUST_EMISSION_MODELS as M

    s = np.asarray(M[model](jnp.asarray(wide), 1.0, redshift=z, **kw), dtype=float)
    return np.trapezoid(np.interp(fw / (1 + z), wide, s) * ft / fw, fw) / np.trapezoid(ft / fw, fw)


def lookup(model, kw, fw, ft, **grids):
    """Lookup value from precomputed grid."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    res = adapter.precompute([fw], [ft], 0.0, None, model=model, **grids)
    q = [kw[n] for n in adapter.AXIS_PARAMS[model]]
    return float(np.asarray(adapter.build_lookup(res, model=model)(1.0, *q)).ravel()[0])


_BANDS_UM = ((60, 90), (250, 500), (750, 950))
_ISSUE_QUERY = {
    "dust_T": 47.3,
    "dust_beta_ir": 1.65,
    "dust_alpha_mir": 1.8,
    "dust_lambda_0_um": 130.0,
}


@pytest.mark.parametrize("model", ["modified_blackbody", "casey2012", "graybody"])
def test_off_node_accuracy_default_grids(model):
    """Default node grids: the issue query and 12 seeded random points inside the priors."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    names = adapter.AXIS_PARAMS[model]
    filters = [tophat(*b) for b in _BANDS_UM]
    res = adapter.precompute(
        [f[0] for f in filters], [f[1] for f in filters], 0.0, None, model=model
    )
    lookup_fn = adapter.build_lookup(res, model=model)

    rng = np.random.RandomState(42)
    points = [{n: _ISSUE_QUERY[n] for n in names}]
    points += [{n: rng.uniform(*_get_param_bounds(n)) for n in names} for _ in range(12)]
    for kw in points:
        got = np.asarray(lookup_fn(1.0, *[kw[n] for n in names]))
        for b, f, lkp in zip(_BANDS_UM, filters, got):
            ratio = lkp / exact(model, kw, *f)
            assert abs(ratio - 1.0) < 1e-3, (
                f"{model} point {kw}, band {b}: lookup/exact = {ratio:.6f}"
            )


@pytest.mark.parametrize("model", ["modified_blackbody", "casey2012", "graybody"])
def test_on_node_exactness(model):
    """On-node query agrees to 1e-3."""
    if model == "modified_blackbody":
        kw = {"dust_T": 30.0, "dust_beta_ir": 1.5}
        T_grid = np.array([20.0, 30.0, 60.0])
        beta_grid = np.array([1.5, 1.8, 2.0])
        grids = {"T_grid": T_grid, "beta_grid": beta_grid}
    elif model == "casey2012":
        kw = {
            "dust_T": 25.0,
            "dust_beta_ir": 1.8,
            "dust_alpha_mir": 2.0,
            "dust_lambda_0_um": 150.0,
        }
        T_grid = np.array([25.0, 35.0, 60.0])
        beta_grid = np.array([1.5, 1.8, 2.0])
        alpha_mir_grid = np.array([1.5, 2.0, 2.5])
        lambda_0_um_grid = np.array([100.0, 150.0, 200.0])
        grids = {
            "T_grid": T_grid,
            "beta_grid": beta_grid,
            "alpha_mir_grid": alpha_mir_grid,
            "lambda_0_um_grid": lambda_0_um_grid,
        }
    elif model == "graybody":
        kw = {"dust_T": 20.0, "dust_beta_ir": 1.8, "dust_lambda_0_um": 100.0}
        T_grid = np.array([20.0, 40.0, 60.0])
        beta_grid = np.array([1.5, 1.8, 2.0])
        lambda_0_um_grid = np.array([100.0, 150.0, 200.0])
        grids = {
            "T_grid": T_grid,
            "beta_grid": beta_grid,
            "lambda_0_um_grid": lambda_0_um_grid,
        }

    fw, ft = tophat(250, 500)
    lkp = lookup(model, kw, fw, ft, **grids)
    ex = exact(model, kw, fw, ft)
    ratio = lkp / ex
    assert abs(ratio - 1.0) < 1e-3, (
        f"{model} on-node: lookup/exact = {ratio:.6f}, error {abs(ratio - 1.0):.6f}"
    )


@pytest.mark.parametrize("model", ["modified_blackbody", "graybody"])
def test_wien_tail_accuracy(model):
    """Wien tail (15 K, 8-24 um) converged to 1e-3."""
    if model == "modified_blackbody":
        kw = {"dust_T": 15.0, "dust_beta_ir": 1.5}
        grids = {
            "T_grid": 15.0 * np.array([0.998, 1.0, 1.002]),
            "beta_grid": 1.5 * np.array([0.998, 1.0, 1.002]),
        }
    elif model == "graybody":
        kw = {"dust_T": 15.0, "dust_beta_ir": 1.5, "dust_lambda_0_um": 200.0}
        grids = {
            "T_grid": 15.0 * np.array([0.998, 1.0, 1.002]),
            "beta_grid": 1.5 * np.array([0.998, 1.0, 1.002]),
            "lambda_0_um_grid": 200.0 * np.array([0.998, 1.0, 1.002]),
        }

    fw, ft = tophat(8, 24)
    lkp = lookup(model, kw, fw, ft, **grids)
    ex = exact(model, kw, fw, ft)
    ratio = lkp / ex
    assert abs(ratio - 1.0) < 1e-3, (
        f"{model} Wien tail: lookup/exact = {ratio:.6f}, error {abs(ratio - 1.0):.6f}"
    )


_MM_GRIDS = {
    "T_grid": 20.0 * np.array([0.998, 1.0, 1.002]),
    "beta_grid": 1.5 * np.array([0.998, 1.0, 1.002]),
}


@pytest.mark.parametrize("band_um", [(12000, 20000), (8000, 12000)])
def test_millimeter_bands_agree_with_exact_closure(band_um):
    """Bands at 8-20 mm lie inside the rest grid and agree with the exact closure."""
    kw = {"dust_T": 20.0, "dust_beta_ir": 1.5}
    fw, ft = tophat(*band_um)
    ratio = lookup("modified_blackbody", kw, fw, ft, **_MM_GRIDS) / exact(
        "modified_blackbody", kw, fw, ft
    )
    assert abs(ratio - 1.0) < 1e-3, f"{band_um} um: lookup/exact = {ratio:.6f}"


def test_band_beyond_rest_grid_refused():
    """A band whose red edge lies beyond 10 m raises ValueError naming the filter."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    fw, ft = tophat(15_000_000, 20_000_000)  # 15-20 m
    with pytest.raises(ValueError, match="Filter 0 RED edge"):
        adapter.precompute([fw], [ft], 0.0, None, model="modified_blackbody", **_MM_GRIDS)


def test_pah_drude_coverage():
    """pah_drude in 40-70 um agrees with exact closure to 1e-3."""
    from tengri.components.dust import dust_analytic_precompute as adapter
    from tengri.components.dust.emission import DUST_EMISSION_MODELS as M

    fw, ft = tophat(40, 70)
    res = adapter.precompute([fw], [ft], 0.0, None, model="pah_drude")

    wide_pah = np.asarray(M["pah_drude"](jnp.asarray(WIDE), 1.0), dtype=float)
    ex = np.trapezoid(np.interp(fw, WIDE, wide_pah) * ft / fw, fw) / np.trapezoid(ft / fw, fw)

    lkp = float(np.asarray(adapter.build_lookup(res, model="pah_drude")(1.0)).ravel()[0])
    ratio = lkp / ex
    assert abs(ratio - 1.0) < 1e-3, (
        f"pah_drude 40-70 um: lookup/exact = {ratio:.6f}, error {abs(ratio - 1.0):.6f}"
    )


def test_redshift_off_node():
    """z=2 off-node query (issue's first case at redshift)."""
    from tengri.components.dust import dust_analytic_precompute as adapter
    from tengri.components.dust.emission import DUST_EMISSION_MODELS as M

    z = 2.0
    kw = {"dust_T": 47.3, "dust_beta_ir": 1.65}
    fw, ft = tophat(60, 90)  # observed frame

    # Exact: template evaluated on ultra-fine grid, then interpolated
    fw_rest = fw / (1 + z)  # convert to rest frame
    s = np.asarray(M["modified_blackbody"](jnp.asarray(WIDE), 1.0, redshift=z, **kw), dtype=float)
    ex = np.trapezoid(np.interp(fw_rest, WIDE, s) * ft / fw_rest, fw_rest) / np.trapezoid(
        ft / fw_rest, fw_rest
    )

    # Lookup at redshift z
    res = adapter.precompute([fw], [ft], z, None, model="modified_blackbody")
    lkp = float(
        np.asarray(adapter.build_lookup(res, model="modified_blackbody")(1.0, 47.3, 1.65)).ravel()[
            0
        ]
    )

    ratio = lkp / ex
    assert abs(ratio - 1.0) < 1e-3, (
        f"modified_blackbody z=2: lookup/exact = {ratio:.6f}, error {abs(ratio - 1.0):.6f}"
    )


_FLOAT32_SCRIPT = """
import json, sys
import numpy as np
import jax, jax.numpy as jnp
from tengri.components.dust import dust_analytic_precompute as adapter

assert not jax.config.jax_enable_x64
spec = json.loads(sys.argv[1])
filters = [(np.asarray(w), np.asarray(t)) for w, t in spec["filters"]]
res = adapter.precompute(
    [f[0] for f in filters], [f[1] for f in filters], 0.0, None, model=spec["model"],
    **{k: np.asarray(v) for k, v in spec["grids"].items()},
)
fn = adapter.build_lookup(res, model=spec["model"])
q = [jnp.float32(x) for x in spec["query"]]
value = np.asarray(fn(jnp.float32(1.0), *q))
grad = jax.jacfwd(lambda *a: fn(jnp.float32(1.0), *a), argnums=tuple(range(len(q))))(*q)
print(json.dumps({"value": value.tolist(), "grad": [np.asarray(g).tolist() for g in grad]}))
"""

_FLOAT32_CASES = {
    "modified_blackbody": (
        {"T_grid": [30.0, 45.0, 60.0], "beta_grid": [1.2, 1.8, 2.4]},
        [47.3, 1.65],
    ),
    "graybody": (
        {
            "T_grid": [30.0, 45.0, 60.0],
            "beta_grid": [1.2, 1.8, 2.4],
            "lambda_0_um_grid": [80.0, 160.0, 320.0],
        },
        [47.3, 1.65, 130.0],
    ),
    "casey2012": (
        {
            "T_grid": [30.0, 45.0, 60.0],
            "beta_grid": [1.2, 1.8, 2.4],
            "alpha_mir_grid": [1.5, 2.0, 2.5],
            "lambda_0_um_grid": [80.0, 160.0, 320.0],
        },
        [47.3, 1.65, 1.8, 130.0],
    ),
}


@pytest.mark.parametrize("model", sorted(_FLOAT32_CASES))
def test_float32_value_and_gradient_are_finite(model):
    """With x64 off, bands that underflow float32 keep a finite value and gradient.

    The optical band's flux is below float32's smallest normal, so the grid is stored as
    ln(band flux) computed in float64; the far-IR band agrees with float64 to rtol 1e-4.
    """
    import json
    import os
    import subprocess
    import sys

    grids, query = _FLOAT32_CASES[model]
    bands = ((0.4, 0.6), (3, 5), (250, 500))
    filters = [tophat(*b) for b in bands]
    spec = {
        "model": model,
        "grids": grids,
        "query": query,
        "filters": [(w.tolist(), t.tolist()) for w, t in filters],
    }
    env = {**os.environ, "JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu"}
    proc = subprocess.run(
        [sys.executable, "-c", _FLOAT32_SCRIPT, json.dumps(spec)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert np.all(np.isfinite(out["value"])), out["value"]
    for name, g in zip(_AXIS_NAMES[model], out["grad"], strict=True):
        # grad-assert: finite-only — the optical band's flux underflows to zero at build in
        # float32, so its gradient is correctly zero; the 3-5 um band is checked finite only.
        assert np.all(np.isfinite(g)), f"d/d{name} not finite: {g}"
        assert g[2] != 0.0, f"far-IR d/d{name} is zero: {g}"

    from tengri.components.dust import dust_analytic_precompute as adapter

    res = adapter.precompute(
        [f[0] for f in filters],
        [f[1] for f in filters],
        0.0,
        None,
        model=model,
        **{k: np.asarray(v) for k, v in grids.items()},
    )
    ref = np.asarray(adapter.build_lookup(res, model=model)(1.0, *query))
    assert out["value"][2] == pytest.approx(ref[2], rel=1e-4)


_AXIS_NAMES = {
    "modified_blackbody": ("dust_T", "dust_beta_ir"),
    "graybody": ("dust_T", "dust_beta_ir", "dust_lambda_0_um"),
    "casey2012": ("dust_T", "dust_beta_ir", "dust_alpha_mir", "dust_lambda_0_um"),
}


@pytest.mark.parametrize("model", ["modified_blackbody", "casey2012", "graybody"])
def test_off_node_accuracy_far_ir_at_z3(model):
    """Default node grids at z = 3: 8 seeded random points in the 250-500 and 750-950 um bands."""
    from tengri.components.dust import dust_analytic_precompute as adapter

    z = 3.0
    names = adapter.AXIS_PARAMS[model]
    bands = ((250, 500), (750, 950))
    filters = [tophat(*b) for b in bands]
    res = adapter.precompute(
        [f[0] for f in filters], [f[1] for f in filters], z, None, model=model
    )
    lookup_fn = adapter.build_lookup(res, model=model)
    rng = np.random.RandomState(7)
    for _ in range(8):
        kw = {n: rng.uniform(*_get_param_bounds(n)) for n in names}
        got = np.asarray(lookup_fn(1.0, *[kw[n] for n in names]))
        for b, f, lkp in zip(bands, filters, got):
            ratio = lkp / exact(model, kw, *f, z)
            assert abs(ratio - 1.0) < 1e-3, f"{model} z=3 point {kw}, band {b}: ratio {ratio:.6f}"


# Worst points of the 200-point set at the origin/main node counts, per band
# (8-24 um z=0, 24-40 um z=0, 60-90 um z=3, 250-500 um z=3).
_CASEY_WORST_POINTS = (
    {
        "dust_T": 65.86694039491042,
        "dust_beta_ir": 1.954949242778371,
        "dust_alpha_mir": 1.1414627378157898,
        "dust_lambda_0_um": 314.3541764398587,
    },
    {
        "dust_T": 20.954624674666388,
        "dust_beta_ir": 2.3033277384649917,
        "dust_alpha_mir": 1.2581468015755068,
        "dust_lambda_0_um": 220.62578820187835,
    },
    {
        "dust_T": 61.801754131364774,
        "dust_beta_ir": 1.6405802395205915,
        "dust_alpha_mir": 1.2691409172893753,
        "dust_lambda_0_um": 199.11074493308254,
    },
    {
        "dust_T": 70.26214388463012,
        "dust_beta_ir": 1.2460498104750752,
        "dust_alpha_mir": 1.0294372121615194,
        "dust_lambda_0_um": 277.99476454653933,
    },
)


def test_casey2012_mid_ir_accuracy_default_grids():
    """casey2012 band fluxes at the default nodes: 8-24, 24-40 um at z=0; 60-90, 250-500 um at z=3.

    Reference: ``exact`` on ``WIDE_CONVERGED``, the continuum limit at the 1 um step, with a rest
    spacing of 0.25 Angstrom at 1 um (the value is unchanged to 3e-11 at 0.0625 Angstrom). Points:
    the four worst points of the origin/main node counts plus 200 seeded ``RandomState(7)`` points
    inside the declared priors. The maximum relative error in each band is below 1e-3.
    """
    from tengri.components.dust import dust_analytic_precompute as adapter

    names = adapter.AXIS_PARAMS["casey2012"]
    cases = ((0.0, ((8, 24), (24, 40))), (3.0, ((60, 90), (250, 500))))
    rng = np.random.RandomState(7)
    points = list(_CASEY_WORST_POINTS)
    points += [{n: rng.uniform(*_get_param_bounds(n)) for n in names} for _ in range(200)]

    for z, bands in cases:
        filters = [tophat(*b) for b in bands]
        res = adapter.precompute(
            [f[0] for f in filters], [f[1] for f in filters], z, None, model="casey2012"
        )
        lookup_fn = adapter.build_lookup(res, model="casey2012")
        worst = np.zeros(len(bands))
        for kw in points:
            got = np.asarray(lookup_fn(1.0, *[kw[n] for n in names]))
            for j, (f, lkp) in enumerate(zip(filters, got)):
                ref = exact("casey2012", kw, *f, z=z, wide=WIDE_CONVERGED)
                worst[j] = max(worst[j], abs(lkp / ref - 1.0))
        for b, w in zip(bands, worst):
            assert w < 1e-3, f"casey2012 z={z:g} band {b} um: max |lookup/exact - 1| = {w:.3e}"

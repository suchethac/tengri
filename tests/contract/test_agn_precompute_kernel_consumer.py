# SPDX-License-Identifier: BSD-3-Clause
"""AGN disc/torus precompute kernel consumers: the six wired adapters.

powerlaw_disc, ss_disc, cigale_disc, qsogen, silva04, cat3d_wind.

Six classes held one test each, all named ``test_smoke_lookup_jit_compatible``
and all asserting only that the returned array had the right shape. A lookup
returning zeros, or one whose normalization changed by nine orders of
magnitude, passed every one of them. They are one table now, and each row
carries what its adapter actually produces.

Two things this file could not see
----------------------------------

**The silva04 and cat3d_wind tests had never run.** Their guard read
``not (_DATA / grid).exists()`` where ``_DATA`` was
``Path(__file__).resolve().parents[4] / "data"`` -- from ``tests/contract/``
that is two levels *above* the repository, so the guard was permanently true
while both grids sat in ``data/`` all along. #1431 is the same defect in
``test_agn_cat3d_wind.py``. The root now comes from ``tests/_data_skip``, which
computes it once.

**The two adapter families disagree about the leading argument.** Measured by
sweeping it from 8 to 14 with every other axis fixed:

===============  ==========================================
adapter          response to +1 in the leading argument
===============  ==========================================
powerlaw_disc    x1.0 per unit -- output is *linear* in it
ss_disc          x1.0 per unit -- linear
cigale_disc      x1.0 per unit -- linear
qsogen           x1.0 per unit -- linear
silva04          **x10** per unit -- output goes as 10**arg
cat3d_wind       **x10** per unit -- 10**arg
===============  ==========================================

The disc curves are straight lines through the origin: powerlaw_disc gives
6.9396e29 at 8 and 8.6745e29 at 10, a constant 8.674e28 per unit. silva04
gives 3.4313e26, 3.4313e27, 3.4313e28 at 8, 9, 10 -- exactly a decade apart.

Whether the disc kernels are *meant* to take a linear amplitude while the torus
kernels take ``log10(L/Lsun)`` is a question about the pipeline that feeds them,
not something this file can answer. What matters here is that nothing asserted
either convention, so a kernel that switched from one to the other would change
its output by ~1e9 at a typical ``agn_log_lbol`` and every test would stay
green. ``test_leading_argument_scaling`` pins the measured behavior per
adapter. See the issue linked from that test.
"""

from __future__ import annotations

import importlib

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tests._data_skip import (
    CAT3D_WIND_GRID,
    SILVA04_GRID,
    requires_cat3d_wind,
    requires_silva04,
)
from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.contract

#: How the output responds to a +1 change in the leading argument.
_LINEAR = "linear"  # output is proportional to the argument
_DECADE = "decade"  # output is proportional to 10**argument


@pytest.fixture(scope="module")
def simple_agn_filters():
    """Simple 3-filter set (UV, optical, IR) for AGN testing."""
    centers = np.array([2000, 5000, 30000])  # Angstrom
    widths = np.array([500, 1500, 5000])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(c - 3 * w, c + 3 * w, 32)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


#: (module under tengri.components.agn, precompute kwargs, build_lookup kwargs,
#:  axes after the leading argument, lookup keyword args, output ndim, scaling).
#:
#: ``powerlaw_disc``'s single axis is the power-law index, which the adapter
#: declared as ``agn_alpha_pl`` -- a name no Parameters can hold, so it could
#: never collapse (#1738). ``ss_disc`` has two, agn_log_mbh and agn_log_lbol
#: (#902). ``cigale_disc`` has none: it is a pure scaling, and it is the one
#: adapter that returns a leading axis, shape (1, n_filters).
_ADAPTERS = [
    pytest.param(
        "disc_precompute",
        {"model": "powerlaw_disc"},
        {"model": "powerlaw_disc"},
        (-1.0,),
        {},
        1,
        _LINEAR,
        id="powerlaw_disc",
    ),
    pytest.param(
        "disc_precompute",
        {"model": "ss_disc"},
        {"model": "ss_disc"},
        (8.0, 11.0),
        {},
        1,
        _LINEAR,
        id="ss_disc",
    ),
    pytest.param(
        "disc_precompute",
        {"model": "cigale_disc"},
        {"model": "cigale_disc"},
        (),
        {},
        2,
        _LINEAR,
        id="cigale_disc",
    ),
    pytest.param(
        "qsogen_precompute",
        {},
        {},
        (-0.35, 0.1),
        {},
        1,
        _LINEAR,
        id="qsogen",
    ),
    pytest.param(
        "silva04_precompute",
        {"grid_path": str(SILVA04_GRID)},
        {},
        (21.5,),
        {"agn_torus_frac": 0.5},
        1,
        _DECADE,
        id="silva04",
        marks=requires_silva04,
    ),
    pytest.param(
        "cat3d_precompute",
        {"grid_path": str(CAT3D_WIND_GRID)},
        {},
        (0.5, 0.5, 0.3),
        {"agn_torus_frac": 0.5},
        1,
        _DECADE,
        id="cat3d_wind",
        marks=requires_cat3d_wind,
    ),
]

_ARGS = ("module", "pre_kwargs", "build_kwargs", "axes", "call_kwargs", "ndim", "scaling")


def _build(module, pre_kwargs, build_kwargs, waves, trans):
    """Import the adapter, precompute against the filter set, return its lookup."""
    adapter = importlib.import_module(f"tengri.components.agn.{module}")
    result = adapter.precompute(waves, trans, redshift=0.1, parameters=None, **pre_kwargs)
    return adapter.build_lookup(result, **build_kwargs)


def _call(lookup, leading, axes, call_kwargs):
    args = (jnp.float64(leading), *(jnp.float64(a) for a in axes))
    return lookup(*args, **{k: jnp.float64(v) for k, v in call_kwargs.items()})


@pytest.mark.parametrize(_ARGS, _ADAPTERS)
def test_lookup_is_jit_compatible_and_emits(
    simple_agn_filters, module, pre_kwargs, build_kwargs, axes, call_kwargs, ndim, scaling
):
    """The lookup traces, matches eager, and returns a non-empty spectrum.

    ``assert_jit_matches_eager`` is the JIT half. The rest is what the six
    shape-only assertions this replaces could not distinguish: an adapter
    returning zeros has the right shape and is finite and non-negative.
    """
    waves, trans = simple_agn_filters
    lookup = _build(module, pre_kwargs, build_kwargs, waves, trans)

    phot = assert_jit_matches_eager(
        lookup,
        jnp.float64(10.5),
        *(jnp.float64(a) for a in axes),
        **{k: jnp.float64(v) for k, v in call_kwargs.items()},
    )
    arr = np.asarray(phot)

    assert arr.ndim == ndim, f"expected ndim {ndim}, got shape {arr.shape}"
    assert arr.shape[-1] == len(waves), f"expected last dim {len(waves)}, got {arr.shape}"
    chex.assert_tree_all_finite(arr)
    assert np.all(arr >= 0.0), f"negative photometry: min {arr.min():.4g}"
    assert np.any(arr > 0.0), "lookup returned an identically-zero spectrum"


@pytest.mark.parametrize(_ARGS, _ADAPTERS)
def test_leading_argument_scaling(
    simple_agn_filters, module, pre_kwargs, build_kwargs, axes, call_kwargs, ndim, scaling
):
    """Pin whether the leading argument acts linearly or as a power of ten.

    The four disc adapters are linear in it; the two torus adapters go as
    ``10**arg``. Both conventions are in the tree at once and nothing asserted
    either, so a kernel switching between them would move its output by ~1e9 at
    a typical ``agn_log_lbol`` with every test still green.

    This test does not claim which is correct -- that depends on what the
    forward pipeline hands these kernels, which is outside this file. It claims
    only that the behavior is what it is today, so a change is visible.
    """
    waves, trans = simple_agn_filters
    lookup = _build(module, pre_kwargs, build_kwargs, waves, trans)

    lo = float(np.asarray(_call(lookup, 10.0, axes, call_kwargs)).max())
    hi = float(np.asarray(_call(lookup, 11.0, axes, call_kwargs)).max())

    assert lo > 0.0, "cannot measure scaling against a zero baseline"

    if scaling is _DECADE:
        assert hi / lo == pytest.approx(10.0, rel=1e-6), (
            f"{module}: +1 in the leading argument scaled the output by {hi / lo:.4g}, "
            f"not 10 — the torus kernels take log10(L/Lsun)"
        )
    else:
        assert hi / lo == pytest.approx(11.0 / 10.0, rel=1e-6), (
            f"{module}: +1 in the leading argument scaled the output by {hi / lo:.4g}, "
            f"not 1.1 — the disc kernels are linear in it, not logarithmic"
        )

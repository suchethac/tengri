# SPDX-License-Identifier: BSD-3-Clause
"""Regression: AGNfitter-rX torus/disc shape params are wired through the builder.

The cat3d_wind, silva04, and skirtor_agnfitter torus blocks shipped with their
distinctive shape parameters declared in the block functions but NOT threaded
through ``SEDModel.build`` (missing the 5-layer wiring), so setting them via the
composable-AGN grammar was a silent no-op. skirtor_agnfitter additionally was
not registered as a torus block, and its exact-runtime SED skipped the
``L_SUN x integral`` normalization (emitting ~1e-18, negligible in a composite).

This test pins both failure modes:

1. **No-op guard** — every torus shape parameter must change ``predict()`` when
   set through the builder (the user's "verify a param is not a no-op" rule).
2. **Normalization guard** — the skirtor_agnfitter torus must emit at a physical
   magnitude (comparable to the disc it reprocesses), not the ~1e-18 the
   un-normalized implementation produced.

Shape-only crossvals cannot catch either: a peak-normalized comparison is blind
to absolute scale and to dead parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri

pytestmark = pytest.mark.regression_bug

_SFH = {"type": "const", "*": tengri.FIXED, "log_total_mass": -10.0}
_DUST = {
    "law": "power_law",
    "type": "two_component",
    "*": tengri.FIXED,
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}
_DISC = {
    "disc": {"type": "multicolor", "*": tengri.FIXED},
    "*": tengri.FIXED,
    "log_lbol": 12.0,
    "frac": 1.0,
}


@pytest.fixture(scope="module")
def ssp():
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")


def _sed(ssp, torus: dict) -> np.ndarray:
    model = tengri.SEDModel.build(
        ssp,
        sfh=_SFH,
        dust_attenuation=_DUST,
        agn=dict(_DISC, torus=torus),
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    return np.asarray(model.predict_rest_sed(p).sed)


@pytest.mark.parametrize(
    "torus_type,key,v1,v2",
    [
        ("cat3d_wind", "a_cat3d", -2.5, -1.5),
        # fwd grid is [1.0, 2.25] — AGNfitter's rows-210+ sub-library. The
        # pre-fix values (0.2 vs 1.5) straddled the fabricated flat region
        # below 1.0 and only passed because 0.2 clamped onto the (cloned)
        # fwd=1.0 plane while 1.5 hit a real one (#1036).
        ("cat3d_wind", "fwd_cat3d", 1.0, 2.25),
        ("silva04", "log_nh_silva", 22.0, 25.0),
        ("skirtor_agnfitter", "oa_skirtor", 20.0, 70.0),
        ("skirtor_agnfitter", "incl_skirtor", 10.0, 80.0),
        ("skirtor_agnfitter", "tv_skirtor", 3.0, 11.0),
    ],
)
def test_torus_param_is_not_a_noop(ssp, torus_type, key, v1, v2):
    """Setting a torus shape param via SEDModel.build must change predict()."""
    s1 = _sed(ssp, {"type": torus_type, "*": tengri.FIXED, key: v1})
    s2 = _sed(ssp, {"type": torus_type, "*": tengri.FIXED, key: v2})
    rel = float(np.abs(s1 - s2).max() / max(np.abs(s1).max(), 1e-99))
    assert rel > 1e-3, (
        f"{torus_type}.{key} is a no-op via the builder (rel diff {rel:.2e}); "
        "the 5-layer wiring (params/groups/forward) is incomplete."
    )


def test_skirtor_agnfitter_emits_physical_magnitude(ssp):
    """The skirtor_agnfitter torus must reprocess a sizeable fraction of L_bol.

    Guards the normalization bug where the exact runtime emitted ~1e-18 erg/s/Hz
    (no L_SUN x integral normalization) and was invisible in the composite.
    """
    base = tengri.SEDModel.build(
        ssp,
        sfh=_SFH,
        dust_attenuation=_DUST,
        agn={
            "disc": {"type": "multicolor", "*": tengri.FIXED},
            "*": tengri.FIXED,
            "log_lbol": 12.0,
            "frac": 1.0,
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(base.spec.sample(jax.random.PRNGKey(0)))
    disc = np.asarray(base.predict_rest_sed(p).sed)
    total = _sed(ssp, {"type": "skirtor_agnfitter", "*": tengri.FIXED})
    # Interpolate disc onto total's grid if needed, then require the torus adds
    # a contribution of order the disc luminosity (not ~1e-18).
    disc_max = float(np.abs(disc).max())
    total_max = float(np.abs(total).max())
    assert total_max > 0.1 * disc_max, (
        f"skirtor_agnfitter torus emits negligibly (max {total_max:.2e} vs disc "
        f"{disc_max:.2e}) — normalization regression."
    )


def test_skirtor_agnfitter_runtime_normalized():
    """The bundled-grid skirtor_agnfitter SED peaks at a physical L_nu, not ~1e-18.

    Uses only the committed torus grid (no SSP data), so it runs in CI and guards
    the normalization regression even where SSP-gated tests skip.
    """
    from tengri.components.agn.skirtor_agnfitter import skirtor_agnfitter_sed

    wave = jnp.geomspace(1e3, 1e7, 400)
    sed = np.asarray(
        skirtor_agnfitter_sed(
            wave,
            agn_log_lbol=12.0,
            agn_oa_skirtor=40.0,
            agn_incl_skirtor=30.0,
            agn_tv_skirtor=7.0,
            agn_torus_frac=1.0,
        )
    )
    assert sed.max() > 1e20, f"skirtor_agnfitter SED peak {sed.max():.2e} is unphysically faint"
    peak_um = float(wave[np.nanargmax(sed)] / 1e4)
    assert 15.0 < peak_um < 40.0, f"peak {peak_um:.1f} um outside the IR torus band"


def test_analytic_aliases_still_resolve():
    """Deprecated *_analytic names forward to the renamed *_sed functions."""
    from tengri.components.agn import (
        cat3d_wind_analytic,
        cat3d_wind_sed,
        silva04_analytic,
        silva04_sed,
    )

    wave = jnp.geomspace(1e3, 1e7, 200)
    with pytest.warns(DeprecationWarning):
        a = np.asarray(cat3d_wind_analytic(wave, agn_log_lbol=12.0))
    b = np.asarray(cat3d_wind_sed(wave, agn_log_lbol=12.0))
    np.testing.assert_allclose(a, b)
    assert silva04_analytic is not silva04_sed  # alias wraps, not identity

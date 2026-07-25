# SPDX-License-Identifier: BSD-3-Clause
r"""Float32 status of every composable-AGN disc block (#1206).

A durable inventory: for each registered disc block, build a composable AGN
(disc + SKIRTOR torus, CIGALE-joint norm) and compare ``sed_agn`` between
float64 and pure float32.

The **shape-class** discs — whose spectral shape depends on L_bol (temperature),
so the float32 path must take the TRUE L_bol for the shape while normalizing
MAGNITUDE to a reference — are all fixed: ``multicolor``, ``kubota_done`` and
``adaf`` carry log-space (or L_sun-unit) internals plus the
``agn_log_lbol_shape`` split. The shape-invariant discs (``powerlaw``,
``richards2006``, ``skirtor``, ``qsogen``, ``schartmann2005``) are exact under
the plain reference-evaluation + rescale.

``adaf_lopez2024`` is exact too: its CIGALE piecewise power law formed
``wavelength**coef * norm`` where, on a steep segment at long wavelength, the two
factors leave the float32 window in OPPOSITE directions (``wavelength**-4``
~1e-40 flushes to 0 while the continuity ``norm`` ~1e40 overflows) — ``0 * inf =
nan``. It is now built as one log10 sum, peak-factored before exponentiating.

Three float32 failures remain — **grid/other-class** (``relagn``,
``slone_netzer``, ``grahsp_sbpl``): non-finite in float32 *even at the reference
L_bol*, each from a DIFFERENT cause. ``grahsp_sbpl`` is blocked on a linear erg/s
*parameter* (``agn_grahsp_l5100``, LogUniform(1e42, 1e47) — the value itself is
``inf`` in float32), so it needs a log-space parameter, not a kernel fix (#1206
item 3). ``relagn`` is finite in EAGER float32 and only ``inf`` under jit — an XLA
fusion artifact, not a range wall. ``slone_netzer`` underflows its template
normalization to zero.

This test pins the exact discs (regression guard) and ``xfail``\ s the rest
(progress tracker: fixing one turns its ``xfail`` into an unexpected pass). It is
the enforced record of "checked every AGN disc component".
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

# Discs whose spectral shape is invariant under L_bol (template / power-law) OR
# whose L_bol-dependent shape is now handled on the float32 path (multicolor,
# kubota_done — log-space internals + shape/normalization split).
_EXACT_DISCS = [
    "multicolor",
    "kubota_done",
    "adaf",
    "powerlaw",
    "richards2006",
    "skirtor",
    "qsogen",
    "schartmann2005",
    "adaf_lopez2024",
    "slone_netzer",
    "relagn",
]

# Shape depends on L_bol; float32 reference evaluation gives the wrong shape.
_SHAPE_CLASS_XFAIL = []

# Non-finite in float32 even at the reference L_bol — a distinct internal overflow.
_GRID_CLASS_XFAIL = ["grahsp_sbpl"]


def _sed_agn(ssp, disc, dtype):
    obs = Observation(photometry=Photometry.from_names(["sdss_r", "wise_w3", "wise_w4"]))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_diff": 0.3,
            "tau_bc": 0.0,
        },
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": {"type": disc, "all_params": FIXED},
            "torus": {"type": "skirtor", "all_params": FIXED},
            "norm": "cigale_joint",
            "log_lbol": Uniform(9.0, 13.0),
            "fracAGN": 0.1,
        },
        redshift=Fixed(0.1),
    )
    p = {
        k: jnp.asarray(v, dtype=dtype)
        for k, v in {"sfh_delayed_log_total_mass": 10.0, "agn_log_lbol": 11.0}.items()
    }
    return np.asarray(model.predict_state(p).derived["sed_agn"])


#: Disc blocks backed by a large HDF5 template grid. Holding the float64 and
#: float32 copies at once (plus their compiled executables) is enough to OOM a
#: parallel pytest worker — the RELAGN grid alone is 26 MB — so the float64 side is
#: released before the float32 model is built.
_LARGE_GRID_DISCS = frozenset({"relagn", "slone_netzer"})


def _f32_matches_f64(ssp, disc):
    with jax.enable_x64(True):
        ref = _sed_agn(ssp, disc, jnp.float64)
    if disc in _LARGE_GRID_DISCS:
        import gc

        jax.clear_caches()
        gc.collect()
    with jax.enable_x64(False):
        f32 = _sed_agn(ssp, disc, jnp.float32)
    if not np.all(np.isfinite(f32)):
        return False, "non-finite in float32"
    peak = np.abs(ref).max()
    live = np.abs(ref) > 1e-6 * peak
    rel = np.abs(f32[live] - ref[live]) / np.abs(ref[live])
    return rel.max() < 1e-3, f"max_rel={rel.max():.2e}"


@pytest.mark.parametrize("disc", _EXACT_DISCS)
def test_disc_is_exact_in_float32(ssp_bare, disc):
    """These disc blocks must match float64 to float32 eps (regression guard)."""
    ok, detail = _f32_matches_f64(ssp_bare, disc)
    assert ok, f"disc '{disc}' regressed on the pure-float32 path: {detail}"


@pytest.mark.parametrize("disc", _SHAPE_CLASS_XFAIL + _GRID_CLASS_XFAIL)
@pytest.mark.xfail(reason="#1206 follow-up: disc not yet float32-hardened", strict=True)
def test_disc_float32_pending(ssp_bare, disc):
    """Progress tracker — fixing one of these flips its xfail to an unexpected pass."""
    ok, detail = _f32_matches_f64(ssp_bare, disc)
    assert ok, f"disc '{disc}' still float32-broken: {detail}"


@pytest.mark.parametrize("disc", _GRID_CLASS_XFAIL)
def test_grid_class_disc_warns_in_float32(ssp_bare, disc):
    """A non-float32-safe disc must warn (loudly) when evaluated in float32.

    Until these discs are hardened they silently corrupt a float32 fit; the
    ``Float32UnsafeAGNWarning`` makes the failure visible. It fires only in
    float32 — never in float64.
    """
    import contextlib
    import warnings

    from tengri.components.agn.component import Float32UnsafeAGNWarning

    # float32: must warn.
    with jax.enable_x64(False):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.suppress(Exception):
                _sed_agn(ssp_bare, disc, jnp.float32)
        assert any(issubclass(w.category, Float32UnsafeAGNWarning) for w in caught), (
            f"disc '{disc}' is float32-broken but emitted no Float32UnsafeAGNWarning"
        )

    # float64: must NOT warn (the disc works there).
    with jax.enable_x64(True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _sed_agn(ssp_bare, disc, jnp.float64)
        assert not any(issubclass(w.category, Float32UnsafeAGNWarning) for w in caught), (
            f"disc '{disc}' wrongly warned about float32 while running in float64"
        )


@pytest.mark.parametrize("disc", ["multicolor", "kubota_done", "adaf"])
def test_float32_safe_disc_does_not_warn(ssp_bare, disc):
    """The float32-exact discs must NOT emit the unsafe warning in float32."""
    import warnings

    from tengri.components.agn.component import Float32UnsafeAGNWarning

    with jax.enable_x64(False):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _sed_agn(ssp_bare, disc, jnp.float32)
        assert not any(issubclass(w.category, Float32UnsafeAGNWarning) for w in caught), (
            f"float32-exact disc '{disc}' wrongly warned as unsafe"
        )

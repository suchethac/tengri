# SPDX-License-Identifier: BSD-3-Clause
"""Continuity-family SFH normalization survives extreme cumulative log ratios.

The continuity SFHs turned the cumulative log10 SFR into a linear rate, summed
it, and divided: ``10**log_sfr / sum(10**log_sfr * dt)``. The ratio priors are
untruncated Student-t(df=2, scale=0.3), so cumulative values of a few hundred
dex are in support (a Laplace posterior reached them in 7/500 to 264/500 draws
per cell). Above ~308 dex (~38 in float32) the sum overflowed and the division
returned ``inf/inf = NaN`` for the whole history; below ~-324 (~-45 float32)
every bin underflowed to an exactly-zero history.

Oracle: a float64 numpy log-space reference (``log10`` of a max-shifted sum),
plus agreement with the naive linear formula at moderate ratios. Equations:
Leja et al. 2019 (arXiv:1905.11997), the continuity and ContinuityFlex
normalizations to a fixed formed mass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh.nonparametric import (
    _continuity_flex_edges_yr,
    continuity,
    continuity_flex,
    psb_continuity,
)

pytestmark = [pytest.mark.regression_bug]

LOG_M = 10.0
GYR = 1e9
# (per-ratio magnitude, float32?, mass rtol). Cumulative log SFR is
# n_ratios * magnitude: ~450 in float64, ~75 in float32.
EXTREME = [
    pytest.param(75.0, False, 1e-9, id="f64_plus450"),
    pytest.param(-75.0, False, 1e-9, id="f64_minus450"),
    pytest.param(15.0, True, 1e-4, id="f32_plus75"),
    pytest.param(-15.0, True, 1e-4, id="f32_minus75"),
]


def _x64(use_f32):
    return jax.enable_x64(not use_f32)


def _ref_log_sfr(log_s, widths_yr, log_m):
    """float64 log-space normalization: log10 SFR per bin [dex]."""
    log_w = log_s + np.log10(widths_yr)
    top = np.max(log_w)
    big_l = top + np.log10(np.sum(10.0 ** (log_w - top)))
    return log_s + log_m - big_l


def _assert_matches(sfr, ref_log, atol):
    sfr = np.asarray(sfr, dtype=np.float64)
    assert np.all(np.isfinite(sfr)), sfr
    assert np.all(sfr >= 0.0)
    live = (ref_log > -25.0) & (ref_log < 30.0)
    assert live.any()
    np.testing.assert_allclose(np.log10(sfr[live]), ref_log[live], atol=atol)


def _mass(fn, edges_yr, **kw):
    mid = 0.5 * (edges_yr[1:] + edges_yr[:-1])
    widths = edges_yr[1:] - edges_yr[:-1]
    sfr = fn(jnp.asarray(mid), **kw)
    return float(jnp.sum(sfr * jnp.asarray(widths))), sfr


def _grads_finite(fn, edges_yr, ratio_names, ratio_vals, **fixed):
    mid = 0.5 * (edges_yr[1:] + edges_yr[:-1])
    widths = edges_yr[1:] - edges_yr[:-1]
    wt = jnp.asarray(widths * mid / np.sum(widths * mid))

    def objective(ratios, log_m):
        kw = dict(zip(ratio_names, ratios, strict=True))
        sfr = fn(jnp.asarray(mid), log_total_mass=log_m, **kw, **fixed)
        return jnp.sum(sfr * wt)

    g_r, g_m = jax.grad(objective, argnums=(0, 1))(jnp.asarray(ratio_vals), jnp.asarray(LOG_M))
    assert np.all(np.isfinite(np.asarray(g_r)))
    assert np.isfinite(float(g_m))


CONT_EDGES_GYR = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])  # 6 bins, 5 ratios
CONT_NAMES = [f"ratio_{i}" for i in range(5)]


class TestContinuity:
    @pytest.mark.parametrize(("mag", "f32", "rtol"), EXTREME)
    def test_extreme_cumulative_ratios_finite_and_mass_exact(self, mag, f32, rtol):
        with _x64(f32):
            edges = jnp.asarray(CONT_EDGES_GYR)
            kw = dict.fromkeys(CONT_NAMES, mag)
            mass, sfr = _mass(
                continuity, CONT_EDGES_GYR * GYR, log_total_mass=LOG_M, bin_edges_gyr=edges, **kw
            )
            assert np.all(np.isfinite(np.asarray(sfr)))
            assert mass == pytest.approx(10.0**LOG_M, rel=rtol)
            log_s = np.append(np.cumsum([mag] * 5)[::-1], 0.0)
            ref = _ref_log_sfr(log_s, np.diff(CONT_EDGES_GYR) * GYR, LOG_M)
            _assert_matches(sfr, ref, 5e-4 if f32 else 1e-9)

    @pytest.mark.parametrize(("mag", "f32", "_rtol"), EXTREME)
    def test_gradients_finite(self, mag, f32, _rtol):
        with _x64(f32):
            _grads_finite(
                lambda a, **k: continuity(a, bin_edges_gyr=jnp.asarray(CONT_EDGES_GYR), **k),
                CONT_EDGES_GYR * GYR,
                CONT_NAMES,
                [mag] * 5,
            )

    def test_moderate_ratios_match_naive_formula(self):
        ratios = np.array([0.3, -0.2, 0.5, 0.1, -0.4])
        kw = dict(zip(CONT_NAMES, ratios, strict=True))
        mid = 0.5 * (CONT_EDGES_GYR[1:] + CONT_EDGES_GYR[:-1]) * GYR
        got = np.asarray(continuity(jnp.asarray(mid), LOG_M, jnp.asarray(CONT_EDGES_GYR), **kw))
        log_s = np.append(np.cumsum(ratios[::-1])[::-1], 0.0)
        unnorm = 10.0**log_s
        naive = unnorm * 10.0**LOG_M / np.sum(unnorm * np.diff(CONT_EDGES_GYR) * GYR)
        np.testing.assert_allclose(got, naive, rtol=1e-12)


PSB_KW = {
    "tlast_gyr": 0.2,
    "tflex_gyr": 0.8,
    "bin_edges_gyr": np.array([0.8, 1.0, 3.0, 6.0, 13.7]),  # only [1:] is read
}
PSB_EDGES_GYR = np.array([0.0, 0.2, 0.8, 1.0, 3.0, 6.0, 13.7])
PSB_NAMES = ["ratio_young", "ratio_old_0", "ratio_old_1", "ratio_old_2", "ratio_old_3"]


def _psb_log_s(ry, link, adj):
    fixed = np.append(np.cumsum(np.asarray(adj)[::-1])[::-1], 0.0)
    flex = fixed[0] + link
    return np.concatenate([[flex + ry], [flex], fixed])


class TestPsbContinuity:
    @pytest.mark.parametrize(("mag", "f32", "rtol"), EXTREME)
    def test_extreme_cumulative_ratios_finite_and_mass_exact(self, mag, f32, rtol):
        with _x64(f32):
            kw = {"ratio_young": 0.0, "ratio_old_0": 0.0, **dict.fromkeys(PSB_NAMES[2:], mag)}
            mass, sfr = _mass(
                psb_continuity, PSB_EDGES_GYR * GYR, log_total_mass=LOG_M, **PSB_KW, **kw
            )
            assert np.all(np.isfinite(np.asarray(sfr)))
            assert mass == pytest.approx(10.0**LOG_M, rel=rtol)
            ref = _ref_log_sfr(
                _psb_log_s(0.0, 0.0, [mag] * 3), np.diff(PSB_EDGES_GYR) * GYR, LOG_M
            )
            _assert_matches(sfr, ref, 5e-4 if f32 else 1e-9)

    @pytest.mark.parametrize(("mag", "f32", "_rtol"), EXTREME)
    def test_gradients_finite(self, mag, f32, _rtol):
        with _x64(f32):
            _grads_finite(
                lambda a, **k: psb_continuity(a, **PSB_KW, **k),
                PSB_EDGES_GYR * GYR,
                PSB_NAMES,
                [0.0, 0.0, mag, mag, mag],
            )

    def test_moderate_ratios_match_naive_formula(self):
        vals = [0.4, -0.1, 0.3, -0.2, 0.25]
        kw = dict(zip(PSB_NAMES, vals, strict=True))
        mid = 0.5 * (PSB_EDGES_GYR[1:] + PSB_EDGES_GYR[:-1]) * GYR
        got = np.asarray(psb_continuity(jnp.asarray(mid), LOG_M, **PSB_KW, **kw))
        unnorm = 10.0 ** _psb_log_s(vals[0], vals[1], vals[2:])
        naive = unnorm * 10.0**LOG_M / np.sum(unnorm * np.diff(PSB_EDGES_GYR) * GYR)
        np.testing.assert_allclose(got, naive, rtol=1e-12)


FLEX_NAMES = ["ratio_young", "flex_0", "flex_1", "flex_2", "ratio_old"]
FLEX_RATIO_NAMES = FLEX_NAMES[1:4]
T_Y, T_O, T_MAX = 0.0316e9, 5.012e9, 13.7e9
FLEX_SCALE = 5.0 / 3.0  # 3 flex ratios reach the same cumulative dex as the 5 above


def _flex_ref_log_sfr(ry, flex, ro, log_m):
    """float64 log-space ContinuityFlex: log10 SFR per bin, young, flex..., old."""
    c = np.concatenate([[0.0], np.cumsum(flex)])
    top = c.max()
    big_l = top + np.log10(np.sum(10.0 ** (c - top)))
    log_dt = np.log10(T_O - T_Y) + c - big_l
    terms = np.array(
        [
            np.log10(len(c)),
            ry + np.log10(T_Y) - log_dt[0],
            ro + np.log10(T_MAX - T_O) - log_dt[-1],
        ]
    )
    top = terms.max()
    log_mbin = log_m - (top + np.log10(np.sum(10.0 ** (terms - top))))
    return np.concatenate(
        [[ry + log_mbin - log_dt[0]], log_mbin - log_dt, [ro + log_mbin - log_dt[-1]]]
    )


def _flex_edges(flex_vals):
    kw = dict(zip(FLEX_RATIO_NAMES, flex_vals, strict=True))
    return np.asarray(_continuity_flex_edges_yr(kw), dtype=np.float64)


class TestContinuityFlex:
    @pytest.mark.parametrize(("mag", "f32", "rtol"), EXTREME)
    def test_extreme_cumulative_ratios_finite_and_mass_exact(self, mag, f32, rtol):
        with _x64(f32):
            flex = [mag * FLEX_SCALE] * 3
            edges = _flex_edges(flex)
            kw = {
                "ratio_young": 0.0,
                "ratio_old": 0.0,
                **dict.fromkeys(FLEX_RATIO_NAMES, mag * FLEX_SCALE),
            }
            mass, sfr = _mass(continuity_flex, edges, log_total_mass=LOG_M, **kw)
            assert np.all(np.isfinite(np.asarray(sfr)))
            assert mass == pytest.approx(10.0**LOG_M, rel=rtol)
            ref = _flex_ref_log_sfr(0.0, flex, 0.0, LOG_M)
            live = np.diff(edges) > 0
            _assert_matches(np.asarray(sfr)[live], ref[live], 5e-4 if f32 else 1e-9)

    @pytest.mark.parametrize(("mag", "f32", "_rtol"), EXTREME)
    def test_gradients_finite(self, mag, f32, _rtol):
        with _x64(f32):
            _grads_finite(
                continuity_flex,
                _flex_edges([mag * FLEX_SCALE] * 3),
                FLEX_NAMES,
                [0.0, *([mag * FLEX_SCALE] * 3), 0.0],
            )

    def test_moderate_ratios_match_naive_formula(self):
        vals = [0.3, 0.2, -0.1, 0.15, -0.25]
        kw = dict(zip(FLEX_NAMES, vals, strict=True))
        edges = _flex_edges(vals[1:4])
        mid = 0.5 * (edges[1:] + edges[:-1])
        got = np.asarray(continuity_flex(jnp.asarray(mid), LOG_M, **kw))
        cum = np.concatenate([[1.0], np.cumprod(10.0 ** np.asarray(vals[1:4]))])
        dt = (T_O - T_Y) * cum / cum.sum()
        denom = len(dt) + 10.0 ** vals[0] * T_Y / dt[0] + 10.0 ** vals[4] * (T_MAX - T_O) / dt[-1]
        mbin = 10.0**LOG_M / denom
        naive = np.concatenate(
            [[10.0 ** vals[0] * mbin / dt[0]], mbin / dt, [10.0 ** vals[4] * mbin / dt[-1]]]
        )
        np.testing.assert_allclose(got, naive, rtol=1e-12)


class TestContinuityFlexEdges:
    @pytest.mark.parametrize(("mag", "f32", "_rtol"), EXTREME)
    def test_extreme_edges_finite_ascending_and_span_exact(self, mag, f32, _rtol):
        with _x64(f32):
            edges = _flex_edges([mag * FLEX_SCALE] * 3)
            assert np.all(np.isfinite(edges))
            assert np.all(np.diff(edges) >= 0.0)
            assert edges[-2] == pytest.approx(T_O, rel=1e-5 if f32 else 1e-12)
            assert edges[-1] == pytest.approx(T_MAX, rel=1e-6)

    @pytest.mark.parametrize(("mag", "f32", "_rtol"), EXTREME)
    def test_edges_gradient_finite(self, mag, f32, _rtol):
        with _x64(f32):

            def interior_sum(r):
                kw = {f"flex_{i}": r[i] for i in range(3)}
                return jnp.sum(_continuity_flex_edges_yr(kw)[2:-1])

            assert np.all(
                np.isfinite(np.asarray(jax.grad(interior_sum)(jnp.full(3, mag * FLEX_SCALE))))
            )

    def test_moderate_edges_match_naive_formula(self):
        vals = np.array([0.2, -0.1, 0.15])
        cum = np.concatenate([[1.0], np.cumprod(10.0**vals)])
        naive = T_Y + np.cumsum((T_O - T_Y) * cum / cum.sum())
        np.testing.assert_allclose(_flex_edges(vals)[2:-1], naive, rtol=1e-12)

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

from tengri.components.stellar.component import _cic_integrand, _cic_parcels
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


def _grads_finite(fn, edges_yr, ratio_names, ratio_vals, *, edges_saturate=False, **fixed):
    mid = 0.5 * (edges_yr[1:] + edges_yr[:-1])
    widths = edges_yr[1:] - edges_yr[:-1]
    wt = jnp.asarray(widths * mid / np.sum(widths * mid))

    def objective(ratios, log_m):
        kw = dict(zip(ratio_names, ratios, strict=True))
        sfr = fn(jnp.asarray(mid), log_total_mass=log_m, **kw, **fixed)
        return jnp.sum(sfr * wt)

    g_r, g_m = jax.grad(objective, argnums=(0, 1))(jnp.asarray(ratio_vals), jnp.asarray(LOG_M))
    # grad-assert: finite-only — with edges_saturate the flex bin edges sit at their limits, so the
    # objective is flat in the ratios there and a zero ratio gradient is correct; otherwise the
    # non-zero half is asserted on the next line.
    assert np.all(np.isfinite(np.asarray(g_r))), "a ratio gradient overflowed or is NaN"
    if not edges_saturate:
        assert np.any(np.asarray(g_r) != 0.0), "the ratios lost all influence on the SFR"
    assert np.isfinite(float(g_m)), "the mass gradient overflowed or is NaN"
    assert float(g_m) != 0.0, "the SFR no longer scales with log_total_mass"


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
                edges_saturate=True,
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


# (ratio_young, flex_0, flex_1, flex_2, ratio_old) as multiples of a magnitude
# r. With both anchors suppressed (-r) the flex bins carry the mass, and a sign
# change in the cumulative ratio collapses a bin that still owns an equal share:
# the pre-fix formed mass was short by one share per collapsed bin.
MASS_PATTERNS = {
    "anchors_down_flex_mixed": (-1.0, -1.0, 1.0, 1.0, -1.0),
    "anchors_down_flex_down_up": (-1.0, -1.0, -1.0, 1.0, -1.0),
    "anchors_down_flex_up_down": (-1.0, 1.0, -1.0, 1.0, -1.0),
    "anchors_up": (1.0, 1.0, 1.0, 1.0, 1.0),
    "uneven": (-0.3, 1.0, -0.37, 0.61, -1.0),
}
MASS_MAGNITUDES = (10.0, 30.0, 100.0, 450.0)
SSP_AGES_YR = np.logspace(6.0, 10.14, 80)
T_OBS_GYR = 13.8


def _flex_mass_kwargs(pattern, mag):
    return dict(zip(FLEX_NAMES, (mag * c for c in MASS_PATTERNS[pattern]), strict=True))


def _edge_mass(kw):
    """sum(SFR * dt) over the function's own edges: bookkeeping mass [Msun]."""
    edges = np.asarray(_continuity_flex_edges_yr(kw), dtype=np.float64)
    mid = 0.5 * (edges[1:] + edges[:-1])
    sfr = np.asarray(continuity_flex(jnp.asarray(mid), log_total_mass=LOG_M, **kw), np.float64)
    return float(np.sum(sfr * np.diff(edges)))


def _integrated_mass(kw):
    """Mass of the dense integrand the stellar component builds its age weights from."""
    ssp = jnp.asarray(SSP_AGES_YR)
    sfh_kwargs = {"log_total_mass": LOG_M, **kw}

    def fn(age, **k):
        return continuity_flex(age, **k)

    age, sfr = _cic_integrand(ssp, fn, sfh_kwargs, continuity_flex, None)
    contrib, *_ = _cic_parcels(age, sfr, ssp, T_OBS_GYR)
    return float(jnp.sum(contrib))


class TestContinuityFlexFormedMass:
    @pytest.mark.parametrize("mag", MASS_MAGNITUDES)
    @pytest.mark.parametrize("pattern", sorted(MASS_PATTERNS))
    def test_edge_bookkeeping_mass_exact(self, pattern, mag):
        assert _edge_mass(_flex_mass_kwargs(pattern, mag)) == pytest.approx(10.0**LOG_M, rel=1e-10)

    @pytest.mark.parametrize("mag", MASS_MAGNITUDES)
    @pytest.mark.parametrize("pattern", sorted(MASS_PATTERNS))
    def test_integrand_mass_exact(self, pattern, mag):
        got = _integrated_mass(_flex_mass_kwargs(pattern, mag))
        assert got == pytest.approx(10.0**LOG_M, rel=1e-5)

    @pytest.mark.parametrize("mag", [5.0, 15.0])
    @pytest.mark.parametrize("pattern", sorted(MASS_PATTERNS))
    def test_float32_edge_mass(self, pattern, mag):
        with _x64(True):
            kw = _flex_mass_kwargs(pattern, mag)
            assert _edge_mass(kw) == pytest.approx(10.0**LOG_M, rel=1e-4)

    def test_resolved_bins_keep_their_rate(self):
        """Merging touches only unresolved bins: moderate ratios are the naive formula."""
        vals = [0.3, 0.2, -0.1, 0.15, -0.25]
        kw = dict(zip(FLEX_NAMES, vals, strict=True))
        assert _edge_mass(kw) == pytest.approx(10.0**LOG_M, rel=1e-12)

    def test_unresolved_bin_is_served_a_finite_rate(self):
        kw = _flex_mass_kwargs("anchors_down_flex_mixed", 30.0)
        edges = np.asarray(_continuity_flex_edges_yr(kw), dtype=np.float64)
        mid = 0.5 * (edges[1:] + edges[:-1])
        sfr = np.asarray(continuity_flex(jnp.asarray(mid), log_total_mass=LOG_M, **kw))
        assert np.all(np.isfinite(sfr))
        assert np.all(sfr > 0.0)

    def test_narrow_but_resolved_bins_are_not_merged(self):
        """Bins ~1e-4 of their edge wide are in support: each keeps M_bin / dt."""
        vals = [0.0, -4.0, 0.0, 0.0, 0.0]
        kw = dict(zip(FLEX_NAMES, vals, strict=True))
        edges = _flex_edges(vals[1:4])
        mid = 0.5 * (edges[1:] + edges[:-1])
        got = np.asarray(continuity_flex(jnp.asarray(mid), LOG_M, **kw))
        cum = np.concatenate([[1.0], np.cumprod(10.0 ** np.asarray(vals[1:4]))])
        dt = (T_O - T_Y) * cum / cum.sum()
        assert dt[1] / edges[3] < 1e-3  # would be absorbed by a coarser floor
        denom = len(dt) + T_Y / dt[0] + (T_MAX - T_O) / dt[-1]
        mbin = 10.0**LOG_M / denom
        naive = np.concatenate([[mbin / dt[0]], mbin / dt, [mbin / dt[-1]]])
        np.testing.assert_allclose(got, naive, rtol=1e-6)

# SPDX-License-Identifier: BSD-3-Clause
"""Standalone bisect probe for #1439's kubota_done hot-corona float32 gradient.

Reproduces the strict xfail's measurement (float64 grad, float32 grad, float32
central difference), then bisects by zone and by intermediate to localize the
-0.034x sign-flipped reverse-mode gradient reported in the xfail's reason text.

Status (2026-09-11, time-boxed investigation): NOT fully re-verified end to
end. A full run on a shared, heavily-loaded machine (multiple concurrent
pytest/agent sessions) was killed by the harness before printing output, and
the time box closed before a rerun could complete. Posted to #1439 with that
caveat; see the issue comment for what was and was not confirmed.

One methodological pitfall this script fixes and is worth keeping documented:
``jax.enable_x64`` is a context manager, not a per-array dtype. Building the
model and computing ``jax.grad`` must both happen INSIDE the same ``with
jax.enable_x64(x64):`` block (as the xfail's own ``_band_gradient`` does) --
computing the gradient after that block exits silently reverts to tengri's
global default (``jax_enable_x64=True``, set at import in ``tengri/__init__.py``),
which makes a "float32" measurement meaningless (an early pass here made
exactly this mistake and produced a spuriously clean ratio of 0.999990).

Run with:
    PYTHONPATH=<repo>/src taskset -c 9-11 JAX_PLATFORMS=cpu \
        <venv>/bin/python bench/scripts/probe_kubota_done_float32_grad.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.components.agn import disc

_SFH = {
    "type": "delayed",
    "all_params": Fixed(DEFAULT),
    "log_total_mass": Uniform(9.0, 11.0),
    "tau_gyr": 1.0,
    "age_gyr": 5.0,
}
_DUST = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_diff": 0.3,
    "tau_bc": 0.0,
}
_TRUTH = {"sfh_delayed_log_total_mass": 10.0, "agn_log_lbol": 11.0}


def _agn_groups(disc_type):
    return dict(
        sfh=_SFH,
        dust_attenuation=_DUST,
        agn={
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "disc": {"type": disc_type, "all_params": Fixed(DEFAULT)},
            "log_lbol": Uniform(9.0, 12.0),
            "fracAGN": 0.1,
        },
    )


def _ssp():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data("data/fsps_prsc_miles_chabrier.h5")


def _obs():
    return Observation(photometry=Photometry.from_names(["sdss_r", "wise_w1"]))


def measure(ssp, obs, groups, *, lo, hi, x64, dtype, agn_f_hard=None):
    """EXACTLY mirrors the xfail's ``_band_gradient``: build, grad and central
    difference must all happen inside the same ``jax.enable_x64`` context,
    because ``jax.enable_x64`` is a context manager, not a per-array dtype --
    calling ``jax.grad`` after it exits silently reverts to tengri's global
    default (x64=True), which would make a "float32" measurement meaningless."""
    with jax.enable_x64(x64):
        model = SEDModel.build(ssp_data=ssp, observation=obs, redshift=Fixed(0.1), **groups)
        names = sorted(n for n in model.spec.free_params if n in _TRUTH)
        wave = np.asarray(model._rest_wavelength, dtype=np.float64)
        mask = jnp.asarray((wave >= lo) & (wave < hi))

        def band_sum(values):
            params = {k: values[i] for i, k in enumerate(names)}
            if agn_f_hard is not None:
                params["agn_f_hard"] = jnp.asarray(agn_f_hard, dtype=dtype)
            return jnp.sum(jnp.where(mask, model.predict(params).rest_sed(), 0.0))

        base = [jnp.asarray(_TRUTH[k], dtype=dtype) for k in names]
        value = float(np.asarray(band_sum(base)))
        grad = np.array([float(np.asarray(g)) for g in jax.grad(band_sum)(base)])
        return names, value, grad


def central_diff(ssp, obs, groups, *, lo, hi, dtype, h, agn_f_hard=None):
    """Float32 central difference of d(band_sum)/d(agn_log_lbol) at step h."""
    with jax.enable_x64(False):
        model = SEDModel.build(ssp_data=ssp, observation=obs, redshift=Fixed(0.1), **groups)
        names = sorted(n for n in model.spec.free_params if n in _TRUTH)
        wave = np.asarray(model._rest_wavelength, dtype=np.float64)
        mask = jnp.asarray((wave >= lo) & (wave < hi))
        idx = names.index("agn_log_lbol")
        base = [jnp.asarray(_TRUTH[k], dtype=dtype) for k in names]

        def band_sum(values):
            params = {k: values[i] for i, k in enumerate(names)}
            if agn_f_hard is not None:
                params["agn_f_hard"] = jnp.asarray(agn_f_hard, dtype=dtype)
            return jnp.sum(jnp.where(mask, model.predict(params).rest_sed(), 0.0))

        def scalar_at(x):
            b = list(base)
            b[idx] = jnp.asarray(x, dtype=dtype)
            return float(np.asarray(band_sum(b)))

        x0 = _TRUTH["agn_log_lbol"]
        fp = scalar_at(x0 + h)
        fm = scalar_at(x0 - h)
        return (fp - fm) / (2 * h)


print("=" * 78)
print("STEP 1: full-model reproduction (matches the xfail's own measurement)")
print("=" * 78)

ssp = _ssp()
obs = _obs()
groups = _agn_groups("kubota_done")
kw = dict(lo=0.0, hi=1e12)

names64, v64, g64 = measure(ssp, obs, groups, x64=True, dtype=jnp.float64, **kw)
names32, v32, g32 = measure(ssp, obs, groups, x64=False, dtype=jnp.float32, **kw)
idx_lbol = names64.index("agn_log_lbol")
print(f"names={names64}")
print(f"f64: value={v64:.6e} grad={g64}")
print(f"f32: value={v32:.6e} grad={g32}")
ratio = g32[idx_lbol] / g64[idx_lbol]
print(f"d(sum L_nu)/d(agn_log_lbol): f64={g64[idx_lbol]:.6e} f32={g32[idx_lbol]:.6e} "
      f"ratio(f32/f64)={ratio:.6f}")

print()
print("Float32 central-difference step-convergence scan:")
for h in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
    cd = central_diff(ssp, obs, groups, dtype=jnp.float32, h=h, **kw)
    print(f"  h={h:.0e}  central_diff={cd:.6e}  ratio_to_f64={cd / g64[idx_lbol]:.6f}")

print()
print("=" * 78)
print("STEP 1b: A/B on agn_f_hard (re-verify the xfail's localization claim)")
print("=" * 78)
for f_hard in [0.0, 1e-6, 0.02, 0.05, 0.1]:
    _, v64f, g64f = measure(
        ssp, obs, groups, x64=True, dtype=jnp.float64, agn_f_hard=f_hard, **kw
    )
    _, v32f, g32f = measure(
        ssp, obs, groups, x64=False, dtype=jnp.float32, agn_f_hard=f_hard, **kw
    )
    r = g32f[idx_lbol] / g64f[idx_lbol] if g64f[idx_lbol] != 0 else float("nan")
    print(f"  agn_f_hard={f_hard:<8g} f64_grad={g64f[idx_lbol]:.6e} "
          f"f32_grad={g32f[idx_lbol]:.6e} ratio={r:.6f}")


print()
print("=" * 78)
print("STEP 2: capture the real intermediates _hot_corona_lnu is called with")
print("=" * 78)

_captured = {}


def _make_capturing_hot_corona_lnu(tag):
    _orig = disc._hot_corona_lnu

    def wrapper(nu, l_hot_erg, gamma_hard, kt_hot_erg, nu_seed_hz=0.0):
        _captured[tag] = dict(
            nu=np.asarray(nu),
            l_hot_erg=np.asarray(l_hot_erg),
            gamma_hard=np.asarray(gamma_hard),
            kt_hot_erg=np.asarray(kt_hot_erg),
            nu_seed_hz=np.asarray(nu_seed_hz),
        )
        return _orig(nu, l_hot_erg, gamma_hard, kt_hot_erg, nu_seed_hz)

    return wrapper, _orig


def _make_capturing_zone_lum(tag):
    _orig = disc._compute_zone_luminosities

    def wrapper(*args, **kwargs):
        # Positional signature: nu, r_isco_cm, r_hot_cm, r_warm_cm, r_out_cm, t_in,
        # agn_cos_inc, n_radii, agn_gamma_warm, agn_kt_warm, agn_gamma_hard,
        # agn_kt_hot, agn_f_hard, l_edd, l_bol_erg, agn_self_consistent_gamma,
        # float32=..., agn_log_mbh=..., agn_log_lbol_shape=..., nthcomp_table=...
        _captured.setdefault(tag, {})
        _captured[tag]["l_edd"] = np.asarray(args[13])
        _captured[tag]["l_bol_erg"] = np.asarray(args[14])
        _captured[tag]["agn_f_hard"] = np.asarray(args[12])
        _captured[tag]["agn_log_mbh"] = np.asarray(kwargs.get("agn_log_mbh"))
        _captured[tag]["agn_log_lbol_shape"] = np.asarray(kwargs.get("agn_log_lbol_shape"))
        _captured[tag]["float32"] = kwargs.get("float32")
        return _orig(*args, **kwargs)

    return wrapper, _orig


for tag, x64 in [("f64", True), ("f32", False)]:
    dtype = jnp.float64 if x64 else jnp.float32
    hot_wrap, hot_orig = _make_capturing_hot_corona_lnu(tag)
    zone_wrap, zone_orig = _make_capturing_zone_lum(tag)
    disc._hot_corona_lnu = hot_wrap
    disc._compute_zone_luminosities = zone_wrap
    try:
        with jax.enable_x64(x64):
            model = SEDModel.build(ssp_data=ssp, observation=obs, redshift=Fixed(0.1), **groups)
            params = {
                "agn_log_lbol": jnp.asarray(_TRUTH["agn_log_lbol"], dtype=dtype),
                "sfh_delayed_log_total_mass": jnp.asarray(
                    _TRUTH["sfh_delayed_log_total_mass"], dtype=dtype
                ),
            }
            _ = model.predict(params).rest_sed()
    finally:
        disc._hot_corona_lnu = hot_orig
        disc._compute_zone_luminosities = zone_orig

for tag in ["f64", "f32"]:
    c = _captured[tag]
    print(f"\n[{tag}] captured from real forward pass:")
    print(f"  l_hot_erg      = {c['l_hot_erg']!r}")
    print(f"  gamma_hard_eff = {c['gamma_hard']!r}")
    print(f"  kt_hot_erg     = {c['kt_hot_erg']!r}")
    print(f"  nu_seed_hz     = {c['nu_seed_hz']!r}")
    print(f"  l_edd          = {c['l_edd']!r}")
    print(f"  l_bol_erg      = {c['l_bol_erg']!r}")
    print(f"  agn_f_hard     = {c['agn_f_hard']!r}")
    print(f"  agn_log_mbh    = {c['agn_log_mbh']!r}")
    print(f"  agn_log_lbol_shape = {c['agn_log_lbol_shape']!r}")
    print(f"  float32 branch = {c['float32']!r}")
    # Recompute the jnp.minimum's two competing branches from the captured
    # concrete operands, to check whether the SELECTED branch of
    # ``l_hot_erg = jnp.minimum(f_hard_safe * l_edd[_lsun], lbol_term * 0.5)``
    # differs between float64 and float32 -- a branch flip would route the
    # log_lbol dependency through a completely different term.
    f_hard_safe = float(np.clip(c["agn_f_hard"], 1e-6, 0.5))
    if c["float32"]:
        term_a = f_hard_safe * float(disc._L_EDD_1MSUN_LSUN) * 10.0 ** float(c["agn_log_mbh"])
        term_b = 10.0 ** float(c["agn_log_lbol_shape"]) * 0.5
    else:
        term_a = f_hard_safe * float(c["l_edd"])
        term_b = float(c["l_bol_erg"]) * 0.5
    print(f"  minimum() branches: term_a(f_hard*l_edd)={term_a:.6e}  "
          f"term_b(lbol_shape*0.5)={term_b:.6e}  "
          f"selected={'A' if term_a < term_b else 'B'}  "
          f"l_hot_erg(recomputed)={min(term_a, term_b):.6e} vs captured "
          f"{float(c['l_hot_erg']):.6e}")

print()
print("=" * 78)
print("STEP 3: isolated hot-corona-zone gradient, using REAL captured f64 magnitudes")
print("=" * 78)
c64 = _captured["f64"]
nu64 = jnp.asarray(c64["nu"], dtype=jnp.float64)
gamma64 = jnp.asarray(c64["gamma_hard"], dtype=jnp.float64)
kt64 = jnp.asarray(c64["kt_hot_erg"], dtype=jnp.float64)
seed64 = jnp.asarray(c64["nu_seed_hz"], dtype=jnp.float64)
l_hot64 = jnp.asarray(c64["l_hot_erg"], dtype=jnp.float64)


def _corona_sum(l_hot, nu, gamma, kt, seed):
    return jnp.sum(disc._hot_corona_lnu(nu, l_hot, gamma, kt, seed))


for argname, idxset in [
    ("l_hot_erg", 0),
    ("gamma_hard", 1),
    ("kt_hot_erg", 2),
    ("nu_seed_hz", 3),
]:
    args64 = [l_hot64, nu64, gamma64, kt64, seed64]
    args32 = [jnp.asarray(a, dtype=jnp.float32) for a in args64]

    def f(x, which=idxset, base=args64):
        a = list(base)
        a[which + 1] = x
        return _corona_sum(*a)

    def f32fn(x, which=idxset, base=args32):
        a = list(base)
        a[which + 1] = x
        return _corona_sum(*a)

    with jax.enable_x64(True):
        v64, gr64 = jax.value_and_grad(f)(args64[idxset + 1])
    with jax.enable_x64(False):
        v32, gr32 = jax.value_and_grad(f32fn)(args32[idxset + 1])
        h = jnp.asarray(max(1e-3 * abs(float(args32[idxset + 1])), 1e-6), dtype=jnp.float32)
        xp = args32[idxset + 1] + h
        xm = args32[idxset + 1] - h
        cd = (f32fn(xp) - f32fn(xm)) / (2 * h)
    r = float(gr32) / float(gr64) if float(gr64) != 0 else float("nan")
    print(f"  d(corona sum)/d({argname}):"
          f" f64={float(gr64):.6e}  f32={float(gr32):.6e}  ratio={r:.6f}  "
          f"f32_central_diff={float(cd):.6e}")

print()
print("=" * 78)
print("STEP 4: _nthcomp_interp primal/jvp parity + FD step scan (warm zone)")
print("=" * 78)
from tengri.components.agn._nthcomp import _get_nthcomp_templates, _nthcomp_interp

gamma_g, kte_g, ktbb_g, nu_g, table_g, avail = _get_nthcomp_templates()
if not avail:
    print("  nthcomp templates unavailable -- skipping")
else:
    # off-node probe point (grid nodes are kinks; avoid them)
    gamma0 = float(gamma_g[len(gamma_g) // 2]) + 0.037
    kte0 = float(kte_g[len(kte_g) // 2]) + 0.011
    ktbb0 = float(ktbb_g[len(ktbb_g) // 2]) + 0.003
    nu_probe = jnp.geomspace(float(nu_g[0]) * 1.1, float(nu_g[-1]) * 0.9, 64)

    class _T:
        pass

    def _table(dtype):
        t = _T()
        t.gamma_grid = jnp.asarray(gamma_g, dtype=dtype)
        t.kte_grid = jnp.asarray(kte_g, dtype=dtype)
        t.ktbb_grid = jnp.asarray(ktbb_g, dtype=dtype)
        t.nu_grid = jnp.asarray(nu_g, dtype=dtype)
        t.table_log = jnp.asarray(table_g, dtype=dtype)
        return t

    with jax.enable_x64(True):
        tab64 = _table(jnp.float64)
        primal64 = _nthcomp_interp(
            tab64,
            nu_probe.astype(jnp.float64),
            jnp.asarray(gamma0, dtype=jnp.float64),
            jnp.asarray(kte0, dtype=jnp.float64),
            jnp.asarray(ktbb0, dtype=jnp.float64),
        )

        def g64fn(gamma):
            return jnp.sum(
                _nthcomp_interp(
                    tab64,
                    nu_probe.astype(jnp.float64),
                    gamma,
                    jnp.asarray(kte0, dtype=jnp.float64),
                    jnp.asarray(ktbb0, dtype=jnp.float64),
                )
            )

        _gamma0_64 = jnp.asarray(gamma0, dtype=jnp.float64)
        _, jvp64 = jax.jvp(g64fn, (_gamma0_64,), (jnp.ones_like(_gamma0_64),))

    with jax.enable_x64(False):
        tab32 = _table(jnp.float32)
        primal32 = _nthcomp_interp(
            tab32,
            nu_probe.astype(jnp.float32),
            jnp.asarray(gamma0, dtype=jnp.float32),
            jnp.asarray(kte0, dtype=jnp.float32),
            jnp.asarray(ktbb0, dtype=jnp.float32),
        )
        _primal_abs_err = jnp.abs(primal32.astype(jnp.float64) - primal64)
        rel_primal = float(jnp.max(_primal_abs_err / jnp.maximum(primal64, 1e-300)))

        def g32fn(gamma):
            return jnp.sum(
                _nthcomp_interp(
                    tab32,
                    nu_probe.astype(jnp.float32),
                    gamma,
                    jnp.asarray(kte0, dtype=jnp.float32),
                    jnp.asarray(ktbb0, dtype=jnp.float32),
                )
            )

        _gamma0_32 = jnp.asarray(gamma0, dtype=jnp.float32)
        _, jvp32 = jax.jvp(g32fn, (_gamma0_32,), (jnp.ones_like(_gamma0_32),))
        print(f"  primal max rel error (f32 vs f64): {rel_primal:.3e}")
        print(f"  jvp(d gamma) f64={float(jvp64):.6e}  f32={float(jvp32):.6e}  "
              f"ratio={float(jvp32) / float(jvp64):.6f}")

        for step in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
            xp = jnp.asarray(gamma0 + step, dtype=jnp.float32)
            xm = jnp.asarray(gamma0 - step, dtype=jnp.float32)
            cd = (g32fn(xp) - g32fn(xm)) / (2 * step)
            print(f"    FD step h={step:.0e}: central_diff={float(cd):.6e}  "
                  f"ratio_to_f64_jvp={float(cd) / float(jvp64):.6f}")

print()
print("DONE")

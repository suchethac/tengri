"""Benchmark precompute vs analytic-runtime paths for closed-form components.

For each adapter that wraps a closed-form analytic spectrum (radio power-laws,
X-ray power-law-with-cutoff, ``powerlaw_disc``, ``ss_disc``,
``modified_blackbody``, ``casey2012``), measure:

* **Precompute path**: the JIT-compiled ``build_lookup`` closure that does an
  N-D triweight interpolation on a grid of band fluxes pre-integrated through
  the supplied filter set.
* **Runtime path**: the source physics function evaluated on a fixed
  rest-frame wavelength grid (1024 points), with band fluxes computed by
  ``jnp.trapz`` of ``L_nu × T(nu) / nu`` against each filter.  This is the
  apples-to-apples comparison: same filter set, same precision target,
  no caching.

Each path is JIT-compiled and warmed up before timing.  Reports steady-state
per-call wall time and the precompute speed-up factor.

Run::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_precompute_analytic.py

The intent is to answer: for these closed-form components, is the precompute
layer actually faster than just running the analytic spectrum on a wavelength
grid every gradient step?  If the speed-up factor is close to 1, the
precompute layer is theatre and should either be retired or reserved as
scaffolding for future tabulated models.

Notes
-----
This is a microbenchmark — it measures only the per-component cost in
isolation, not the cost in a full SED forward pass where the wavelength grid
already exists for stars+dust.  In a full forward pass the marginal cost of
adding one analytic component on the existing grid is typically well below
this microbenchmark suggests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# ── synthetic filter set covering UV–radio ────────────────────────────────


def _make_filter_set(
    centers: np.ndarray, widths: np.ndarray, n_per_filt: int = 64
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(max(c - 3 * w, 1e-2), c + 3 * w, n_per_filt)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


_FILTERS_UV_OPT = _make_filter_set(
    np.array([1500.0, 2300.0, 4500.0, 6500.0, 8500.0, 12000.0]),
    np.array([300.0, 400.0, 800.0, 1000.0, 1200.0, 1500.0]),
)
_FILTERS_FIR = _make_filter_set(
    np.array([7e4, 1.5e5, 5e5, 1e6, 5e6]),
    np.array([1e4, 3e4, 1e5, 2e5, 5e5]),
)
_FILTERS_RADIO = _make_filter_set(
    np.array([3e5, 1e7, 1e8, 1e10]),
    np.array([1e5, 3e6, 3e7, 3e9]),
)
_FILTERS_XRAY = _make_filter_set(
    np.array([1.0, 5.0, 50.0, 500.0]),
    np.array([0.3, 1.5, 15.0, 150.0]),
)


# ── runtime-path band-flux helper ─────────────────────────────────────────


def _make_runtime_band_flux(
    filter_waves_obs: list[np.ndarray],
    filter_trans: list[np.ndarray],
    redshift: float,
) -> Callable[[Callable, dict], jnp.ndarray]:
    """Apples-to-apples runtime band-flux evaluator.

    Builds a JIT-able function that:
      1. Takes a JAX-callable ``spectrum_fn(wave_rest, **kwargs) -> L_nu``.
      2. Samples it on a fixed 1024-point rest-frame log-wavelength grid.
      3. For each filter, interpolates onto the filter's observed-frame
         wavelengths (rest = obs / (1+z)) and trapz-integrates
         ``L_nu × T(nu) / nu`` to a single band flux.

    Mirrors ``forward.precompute.templates.precompute_template_photometry``'s
    integration semantics so the comparison is fair.
    """
    wave_rest = jnp.logspace(0.0, 13.0, 1024)
    f_obs_jax = [jnp.asarray(fw) for fw in filter_waves_obs]
    t_jax = [jnp.asarray(ft) for ft in filter_trans]
    one_plus_z = 1.0 + float(redshift)

    def runtime_band_flux(spectrum_fn: Callable, kwargs: dict) -> jnp.ndarray:
        l_nu_rest = spectrum_fn(wave_rest, **kwargs)
        bands = []
        for fw_obs, t in zip(f_obs_jax, t_jax):
            wave_rest_filt = fw_obs / one_plus_z
            l_nu_at_filt = jnp.interp(wave_rest_filt, wave_rest, l_nu_rest, left=0.0, right=0.0)
            nu = 2.998e18 / fw_obs  # Hz, rough — only proportionality matters here
            integrand = l_nu_at_filt * t / nu
            bands.append(jnp.trapezoid(integrand, fw_obs))
        return jnp.stack(bands)

    return runtime_band_flux


# ── case definitions ───────────────────────────────────────────────────────


@dataclass
class BenchCase:
    name: str
    adapter_module_path: str
    model_kwarg: str | None  # e.g. "powerlaw_disc" for multi-model adapters; None for single
    filter_set: tuple[list[np.ndarray], list[np.ndarray]]
    redshift: float
    runtime_fn_path: str  # "tengri.components.radio.radio:radio_sfr_bell2003"
    runtime_kwargs: dict  # static kwargs that are not the precompute axis
    lookup_args: tuple[float, ...]  # (scale, *axes) for build_lookup probe


CASES: list[BenchCase] = [
    BenchCase(
        name="radio_synchrotron",
        adapter_module_path="tengri.components.radio.radio_precompute",
        model_kwarg="radio_synchrotron",
        filter_set=_FILTERS_RADIO,
        redshift=0.5,
        runtime_fn_path="tengri.components.radio.radio:radio_sfr_bell2003",
        runtime_kwargs={"L_ir": 1.0e44, "alpha_sf": 0.7},
        lookup_args=(1.0, 0.7),
    ),
    BenchCase(
        name="radio_freefree",
        adapter_module_path="tengri.components.radio.radio_precompute",
        model_kwarg="radio_freefree",
        filter_set=_FILTERS_RADIO,
        redshift=0.5,
        runtime_fn_path="tengri.components.radio.radio:radio_freefree",
        runtime_kwargs={"L_ir": 1.0e44, "alpha_ff": -0.1},
        lookup_args=(1.0, -0.1),
    ),
    BenchCase(
        name="xray_corona",
        adapter_module_path="tengri.components.xray.xray_precompute",
        model_kwarg="xray_corona",
        filter_set=_FILTERS_XRAY,
        redshift=0.5,
        runtime_fn_path="tengri.components.xray.xray:xray_agn_corona",
        runtime_kwargs={"L_agn_bol": 1.0e44, "gamma": 1.8, "alpha_ox": -1.4},
        lookup_args=(1.0, 1.8, -1.4),
    ),
    BenchCase(
        name="powerlaw_disc",
        adapter_module_path="tengri.components.agn.disc_precompute",
        model_kwarg="powerlaw_disc",
        filter_set=_FILTERS_UV_OPT,
        redshift=1.0,
        runtime_fn_path="tengri.components.agn.disc:powerlaw_disc",
        runtime_kwargs={
            "agn_log_lbol": 45.0,
            "agn_frac": 1.0,
            "agn_alpha": -1.5,
            "agn_T_max": 1.0e5,
        },
        lookup_args=(1.0, -1.5),
    ),
    BenchCase(
        name="modified_blackbody",
        adapter_module_path="tengri.components.dust.dust_analytic_precompute",
        model_kwarg="modified_blackbody",
        filter_set=_FILTERS_FIR,
        redshift=1.0,
        runtime_fn_path="tengri.components.dust.emission:modified_blackbody",
        runtime_kwargs={"L_absorbed": 1.0e44, "dust_T": 35.0, "dust_beta_ir": 1.8},
        lookup_args=(1.0, 35.0, 1.8),
    ),
]


def _resolve_runtime_fn(spec: str) -> Callable:
    import importlib

    mod_path, fn_name = spec.split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, fn_name)


def _time_call(fn: Callable, args: tuple, n_warmup: int = 5, n_iter: int = 200) -> float:
    """Return median per-call wall time in microseconds."""
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(times))


def run_case(case: BenchCase) -> dict:
    import importlib

    adapter = importlib.import_module(case.adapter_module_path)
    waves, trans = case.filter_set

    # Build precompute path
    precomp_kwargs = {"model": case.model_kwarg} if case.model_kwarg else {}
    preint = adapter.precompute(
        waves, trans, redshift=case.redshift, parameters=None, **precomp_kwargs
    )
    lookup = adapter.build_lookup(preint, **precomp_kwargs)
    lookup_jit = jax.jit(lookup)

    # Build runtime path
    runtime_band_flux = _make_runtime_band_flux(waves, trans, case.redshift)
    runtime_fn = _resolve_runtime_fn(case.runtime_fn_path)
    runtime_jit = jax.jit(lambda: runtime_band_flux(runtime_fn, case.runtime_kwargs))

    lookup_args = tuple(jnp.asarray(a, dtype=jnp.float64) for a in case.lookup_args)

    t_lookup = _time_call(lookup_jit, lookup_args)
    t_runtime = _time_call(runtime_jit, ())

    speedup = t_runtime / t_lookup if t_lookup > 0 else float("nan")
    return {
        "name": case.name,
        "lookup_us": t_lookup,
        "runtime_us": t_runtime,
        "speedup": speedup,
    }


def main() -> None:
    print(f"{'component':28s}  {'precompute (us)':>18s}  {'runtime (us)':>14s}  {'speedup':>10s}")
    print("-" * 78)
    for case in CASES:
        try:
            r = run_case(case)
        except Exception as e:
            print(f"{case.name:28s}  ERROR: {e}")
            continue
        print(
            f"{r['name']:28s}  {r['lookup_us']:>18.2f}  {r['runtime_us']:>14.2f}  "
            f"{r['speedup']:>9.2f}x"
        )


if __name__ == "__main__":
    main()

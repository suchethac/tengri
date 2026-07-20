# SPDX-License-Identifier: BSD-3-Clause
"""Standalone multi-device check for sharded population (hierarchical) VI.

Run as a subprocess with fake devices, e.g.::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 python _population_sharded_parity_check.py

Prints ``PARITY_OK`` on success, ``SKIP_NO_DEVICES`` if fewer than 2 devices are
visible (so the flag did not take). Driven by ``test_population_sharded.py`` —
kept as a separate module (not a ``test_`` file) because the device count must
be set via ``XLA_FLAGS`` *before* JAX initializes, which only a fresh process
can do.

Two independent things are checked, because either alone can pass while the
feature is broken:

1. **The work is divided, not duplicated.** A galaxy axis that is *sharded but
   not partitioned* returns correct numbers while every device redundantly
   computes every galaxy — the failure mode measured for ``lax.map`` under
   GSPMD (8.00x = n_devices exactly). Presence of an ``all-reduce`` in the
   sharded Hamiltonian and a drop in per-device FLOPs together distinguish
   partitioned work from replicated work; neither does so alone.
2. **The answer is unchanged.** Sharded and single-device fits must agree to
   float reassociation tolerance.
"""

import warnings

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

if jax.device_count() < 2:
    print("SKIP_NO_DEVICES")
    raise SystemExit(0)

from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.forward.forward_model import ForwardModel
from tengri.forward.population_sed_model import PopulationSEDModel
from tengri.inference.fitter import Fitter
from tengri.inference.sharding import shard_leading_axis
from tengri.observation.photometry import FilterCurve

N_GRID = 64


def _template():
    """A tiny stochastic-SFH model on a synthetic SSP (no data files needed)."""
    ssp = SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, 100),
        ssp_flux=jnp.abs(jax.random.normal(jax.random.PRNGKey(123), (3, 20, 100))) * 1e-3 + 1e-5,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 20),
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
    )
    curves = tuple(
        FilterCurve(wave=w, trans=jnp.ones(50) * 0.5, name=f"b{i}")
        for i, w in enumerate(
            [
                jnp.linspace(3500, 4500, 50),
                jnp.linspace(5000, 6500, 50),
                jnp.linspace(7500, 9000, 50),
            ]
        )
    )
    obs = Observation(photometry=Photometry(filters=curves))
    spec = Parameters(
        sfh_tsnorm_log_total_mass=Uniform(8.0, 12.0),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
        sfh_tsnorm_skew=Fixed(0.0),
        sfh_tsnorm_trunc=Fixed(5.0),
        sfh_field_psd_sigma=Uniform(0.1, 4.0),
        sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.05),
        mean_sfh_type=["tsnorm", "field"],
        n_grid=N_GRID,
    )
    return SEDModel(spec, ssp, observation=obs), obs


def _build_fitter(n_gal):
    template, obs = _template()
    galaxies = []
    for i in range(n_gal):
        k = jax.random.fold_in(jax.random.PRNGKey(42), i)
        p = dict(template.spec.sample(k))
        p["sfh_field_psd_sigma"] = jnp.array(2.0)
        p["sfh_field_psd_tau_myr"] = jnp.array(20.0)
        f = template.predict_photometry(p)
        noise = jnp.abs(f) * 0.10 + 1e-3
        galaxies.append(
            {"flux_obs": f + noise * jax.random.normal(k, shape=f.shape), "noise": noise}
        )
    pop = PopulationSEDModel(sed=template, galaxies=galaxies, data_type="photometry")
    forward = ForwardModel.build(population=pop, observation=obs)
    return Fitter(forward, compile_modes=None)


def check_work_is_divided(fitter, devices):
    """Sharding must partition the galaxy loop, not merely relabel it.

    Compiles the population Hamiltonian against replicated and sharded
    ``data_args`` and compares the two lowerings.
    """
    pos = fitter._initialize_unbounded(jax.random.PRNGKey(0))
    engine = fitter._get_or_build_engine(pos)
    xi = ravel_pytree(pos)[0]

    ham = jax.jit(engine["hamiltonian"])
    replicated = fitter._data_args
    sharded = shard_leading_axis(replicated, devices)

    c_repl = ham.lower(xi, replicated).compile()
    c_shard = ham.lower(xi, sharded).compile()
    hlo_repl, hlo_shard = c_repl.as_text(), c_shard.as_text()

    # The chi2 reduction spans galaxies; once those live on different devices it
    # cannot complete without a cross-device sum.
    assert "all-reduce" in hlo_shard, "sharded Hamiltonian has no all-reduce: chi2 never combined"
    assert "all-reduce" not in hlo_repl, "replicated Hamiltonian unexpectedly has an all-reduce"

    def _flops(compiled):
        try:
            cost = compiled.cost_analysis()
            return float((cost[0] if isinstance(cost, list) else cost).get("flops", float("nan")))
        except Exception:
            return float("nan")

    f_repl, f_shard = _flops(c_repl), _flops(c_shard)
    print(f"per-device flops: replicated={f_repl:,.0f}  sharded={f_shard:,.0f}")
    if np.isfinite(f_repl) and np.isfinite(f_shard) and f_repl > 0:
        # Duplicated work would leave per-device FLOPs flat. Partitioned work
        # divides them by ~n_devices; 0.75 leaves room for the shared
        # hyperparameter terms, which every device computes.
        assert f_shard < 0.75 * f_repl, (
            f"per-device flops did not fall ({f_shard:,.0f} vs {f_repl:,.0f}): "
            "every device is computing every galaxy"
        )


def check_parity(n_gal, n_dev):
    """Sharded and single-device fits must agree.

    Both run on *one* fitter, sharded first. That ordering is deliberate: it is
    the case where a multi-device run could leave its mesh behind and silently
    hand the next single-device run galaxy-sharded arrays.
    """
    common = dict(
        key=jax.random.PRNGKey(0),
        n_iterations=2,
        n_samples=2,
        n_posterior_samples=20,
        n_seeds=1,
        verbose=False,
    )
    fitter = _build_fitter(n_gal)
    sharded = fitter.run("native_vi_linear", devices="all", **common)
    single = fitter.run("native_vi_linear", devices=None, **common)

    assert single.diagnostics["n_devices"] == 1, single.diagnostics
    assert sharded.diagnostics["n_devices"] == n_dev, sharded.diagnostics

    maxdiff = 0.0
    for name, val in single.params.items():
        a = np.asarray(val)
        b = np.asarray(sharded.params[name])
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        scale = max(1.0, float(np.max(np.abs(a))))
        maxdiff = max(maxdiff, float(np.max(np.abs(a - b))) / scale)
    # Batched matmul reassociates the chi2 reduction across devices, so this is
    # a reassociation tolerance, not bit-equality.
    assert maxdiff < 1e-6, f"parity failed: max relative diff {maxdiff:.3e}"
    print(f"max relative param diff single vs sharded ({n_dev} devices): {maxdiff:.2e}")


def check_rejects_indivisible(n_dev):
    """An uneven galaxy axis must refuse to run rather than drop galaxies.

    Lives here, not in the pytest module, because the rejection path is
    unreachable with a single device — as a unit test it would skip everywhere
    and read as coverage it never provided.
    """
    tree = {"data": jnp.ones((n_dev + 1, 3))}
    try:
        shard_leading_axis(tree, "all")
    except ValueError as exc:
        assert "divisible" in str(exc), f"wrong error: {exc}"
    else:
        raise AssertionError(f"sharding {n_dev + 1} galaxies over {n_dev} devices was allowed")


def check_warns_for_unsupported_method(n_gal):
    """A method that cannot shard must say so, not accept the argument quietly.

    Silently running single-device is indistinguishable from running
    distributed until someone times it, which is the whole failure class here.
    """
    fitter = _build_fitter(n_gal)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        post = fitter.run("map", devices="all", n_steps=2, verbose=False)
    assert any("devices=" in str(w.message) for w in caught), (
        f"no devices= warning from an unsupported method; got {[str(w.message) for w in caught]}"
    )
    # And it must genuinely have fallen back rather than half-sharded.
    assert fitter._n_devices == 1, fitter._n_devices
    assert post is not None


def main():
    n_dev = jax.device_count()
    n_gal = 2 * n_dev  # divisible by n_dev so every device gets an equal slice
    print(f"devices={n_dev}  n_gal={n_gal}")

    check_rejects_indivisible(n_dev)
    check_warns_for_unsupported_method(n_gal)
    check_work_is_divided(_build_fitter(n_gal), "all")
    check_parity(n_gal, n_dev)
    print("PARITY_OK")


if __name__ == "__main__":
    main()

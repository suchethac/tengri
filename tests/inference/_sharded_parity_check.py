# SPDX-License-Identifier: BSD-3-Clause
"""Standalone CPU↔multi-device parity check for sharded catalog sampling.

Run as a subprocess with fake devices, e.g.::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 python _sharded_parity_check.py

Prints ``PARITY_OK`` on success, ``SKIP_NO_DEVICES`` if fewer than 2 devices are
visible (so the flag did not take). Driven by ``test_catalog_sharded.py`` — kept
as a separate module (not a ``test_`` file) because the device count must be set
via ``XLA_FLAGS`` *before* JAX initializes, which only a fresh process can do.
"""

import warnings

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np

if jax.device_count() < 2:
    print("SKIP_NO_DEVICES")
    raise SystemExit(0)

from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, Uniform
from tengri.inference.catalog_fitter import CatalogFitter
from tengri.observation.photometry import FilterCurve
from tengri.sps.dsps_wrapper import SSPData


def _build():
    wave = jnp.linspace(3000.0, 10000.0, 100)
    ages = jnp.linspace(-1.0, 1.14, 20)
    flux = jnp.abs(jax.random.normal(jax.random.PRNGKey(123), (3, 20, 100))) * 1e-3 + 1e-5
    ssp = SSPData(
        ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages, ssp_lgmet=jnp.array([-1.5, -0.5, 0.0])
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
        sfh_dpl_log_total_mass=Uniform(-1.0, 3.0),
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(1.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    return SEDModel(spec, ssp, observation=obs)


def main():
    model = _build()
    n_dev = jax.device_count()
    # n_gal divisible by lcm(K=2, n_dev) so single & sharded pad identically —
    # otherwise the per-galaxy RNG key splits differ and parity is meaningless.
    n_gal = 2 * n_dev * 2
    galaxies = []
    for i in range(n_gal):
        k = jax.random.fold_in(jax.random.PRNGKey(0), i)
        tp = dict(model.spec.sample(k))
        tp["sfh_dpl_log_total_mass"] = jnp.array(float(i) * 0.2)
        f = model.predict_photometry(tp)
        noise = jnp.abs(f) * 0.02
        obs = f + noise * jax.random.normal(jax.random.fold_in(k, 1), f.shape)
        galaxies.append({"flux_obs": obs, "noise": noise})

    cat = CatalogFitter(model, galaxies, data_type="photometry")
    common = dict(
        key=jax.random.PRNGKey(7),
        forward_chunk_size=2,
        n_warmup=15,
        n_burnin=5,
        n_samples=15,
        verbose=False,
    )
    single = cat.run("mcmc_nuts", devices=None, **common)
    sharded = cat.run("mcmc_nuts", devices="all", **common)

    assert single.diagnostics["n_devices"] == 1
    assert sharded.diagnostics["n_devices"] == n_dev

    maxdiff = 0.0
    for i in range(n_gal):
        for name in single[i].samples:
            d = float(
                np.max(
                    np.abs(
                        np.asarray(single[i].samples[name]) - np.asarray(sharded[i].samples[name])
                    )
                )
            )
            maxdiff = max(maxdiff, d)
    assert maxdiff < 1e-8, f"parity failed: max abs diff {maxdiff}"
    print(f"max abs diff single vs sharded ({n_dev} devices): {maxdiff:.2e}")
    print("PARITY_OK")


if __name__ == "__main__":
    main()

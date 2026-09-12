# SPDX-License-Identifier: BSD-3-Clause
"""``chain_parallel`` (jax.pmap over MCMC chains) on ``run_nuts``.

CPU exposes one JAX device unless ``XLA_FLAGS=--xla_force_host_platform_device_count=N``
is set before the first ``import jax`` (device count is fixed at backend
initialization). Every test here therefore runs in a fresh subprocess with
``TENGRI_HOST_DEVICES`` set in the environment -- the documented shortcut
(``tengri/__init__.py``) that appends the XLA flag before ``import jax`` --
rather than in-process, where the current process's single device is
already locked in.

Measured on ``ctl-dpl`` (D=8, 4 chains) with 4 forced host CPU devices: the
sampling phase drops 19.5s -> 3.5s versus ``jax.vmap`` on one device;
warmup (always single-chain) is unaffected.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.contract

_SCRIPT = r"""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

import tengri  # noqa: F401  (applies TENGRI_HOST_DEVICES before any other jax use)


def _build_model():
    from tengri import SEDModel, recipes
    from tengri.components.stellar.sps.dsps_wrapper import SSPData
    from tengri.observation import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    n_age = 25
    wave = jnp.logspace(2.0, 7.0, 1600)
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    ssp = SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)

    def _tophat(center, frac=0.16, n=40):
        w = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        t = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=w, trans=t, name=f"b{int(center)}")

    curves = tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0, 7600.0, 9000.0))
    obs = Observation(photometry=Photometry(filters=curves))
    return SEDModel.build(ssp_data=ssp, observation=obs, **recipes.mock_recovery_minimal())


def main():
    mode = sys.argv[1]

    if mode == "devices":
        print(json.dumps({"n_devices": len(jax.devices())}))
        return

    from tengri import ForwardModel, generate_mock

    model = _build_model()
    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(0), 3)
    truth = model.spec.sample(key_truth)
    mock = generate_mock(model, truth, key=key_mock, snr=30.0)
    flux_obs = jnp.asarray(mock["flux_obs"])
    noise = jnp.asarray(mock["noise"])

    if mode == "pmap_needs_devices":
        forward = ForwardModel.build(sed=model)
        try:
            forward.fit(
                flux_obs,
                noise,
                method="mcmc_nuts",
                key=jax.random.PRNGKey(99),
                n_chains=4,
                chain_parallel="pmap",
                n_warmup=20,
                n_burnin=5,
                n_samples=20,
                verbose=False,
            )
        except ValueError as exc:
            print(json.dumps({"raised": True, "message": str(exc)}))
            return
        print(json.dumps({"raised": False}))
        return

    if mode == "compare":
        results = {}
        for tag in ("pmap", "vmap"):
            forward = ForwardModel.build(sed=model)
            post = forward.fit(
                flux_obs,
                noise,
                method="mcmc_nuts",
                key=key_fit,
                n_chains=4,
                chain_parallel=tag,
                n_warmup=100,
                n_burnin=20,
                n_samples=100,
                verbose=False,
            )
            means, ses, shapes = {}, {}, {}
            for name in model.spec.free_params:
                draws = np.asarray(post.samples[name])
                means[name] = float(np.mean(draws))
                ses[name] = float(np.std(draws) / np.sqrt(draws.shape[0]))
                shapes[name] = list(draws.shape)
            results[tag] = {
                "means": means,
                "ses": ses,
                "shapes": shapes,
                "chain_parallel_diagnostic": post.diagnostics.get("chain_parallel"),
            }
        print(json.dumps(results))
        return

    raise ValueError(f"unknown mode {mode!r}")


main()
"""


def _run(mode: str, *, host_devices: int | None):
    # Inherit the full parent environment (PYTHONPATH in particular) so the
    # subprocess imports *this* tengri checkout rather than whatever else
    # might be importable on a bare interpreter's default path -- forcing
    # only JAX_PLATFORMS and TENGRI_HOST_DEVICES.
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    if host_devices is not None:
        env["TENGRI_HOST_DEVICES"] = str(host_devices)
    else:
        env.pop("TENGRI_HOST_DEVICES", None)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"subprocess failed (mode={mode!r}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_tengri_host_devices_env_gives_four_cpu_devices():
    """The documented shortcut: ``TENGRI_HOST_DEVICES=4`` -> 4 CpuDevices."""
    out = _run("devices", host_devices=4)
    assert out["n_devices"] == 4


def test_pmap_without_enough_devices_raises_value_error():
    """``chain_parallel='pmap'`` on a single-device process names the env hook."""
    out = _run("pmap_needs_devices", host_devices=None)
    assert out["raised"] is True
    assert "TENGRI_HOST_DEVICES" in out["message"]
    assert "n_chains=4" in out["message"]


def test_pmap_and_vmap_agree_on_shapes_and_means():
    """``chain_parallel='pmap'`` vs ``'vmap'`` on the same key: same posterior.

    4 forced host CPU devices, ``n_chains=4``, ``mcmc_nuts`` on
    ``recipes.mock_recovery_minimal()``. Both executors must return the same
    sample shapes and per-parameter means within 3 combined MC standard
    errors -- pmap changes *where* the chains run (one device vs four), not
    the sampled distribution.
    """
    out = _run("compare", host_devices=4)
    pmap, vmap = out["pmap"], out["vmap"]

    assert pmap["chain_parallel_diagnostic"] == "pmap"
    assert vmap["chain_parallel_diagnostic"] == "vmap"

    assert pmap["shapes"].keys() == vmap["shapes"].keys()
    for name in pmap["shapes"]:
        assert pmap["shapes"][name] == vmap["shapes"][name], name

    for name in pmap["means"]:
        se = np.sqrt(pmap["ses"][name] ** 2 + vmap["ses"][name] ** 2)
        diff = abs(pmap["means"][name] - vmap["means"][name])
        assert diff <= 3.0 * se, (
            f"{name}: pmap mean {pmap['means'][name]:.4g} vs vmap mean "
            f"{vmap['means'][name]:.4g}, diff {diff:.4g} > 3*SE {3 * se:.4g}"
        )

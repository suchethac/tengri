# SPDX-License-Identifier: BSD-3-Clause
"""Tests for GRAHSP Netzer accretion-disc multilinear interpolation kernel.

Tests Item D1 and D2 from the F1 design document: kernel equivalence at grid
nodes, midpoint interpolation, gradient properties, and finite-difference
validation.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds

FIXTURE = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "grahsp" / "netzer_disc.npz"


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURE, allow_pickle=True)


@pytest.fixture(scope="module")
def templates():
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    return load_grahsp_templates()


# Item D1: Kernel equivalence at all 16 grid nodes
class TestNetzerDiscInterpNodeEquality:
    """Verify interpolation kernel equals select_disc_model at all 16 nodes."""

    def test_netzer_disc_interp_exists(self, templates):
        """Check that netzer_disc_interp can be imported."""
        from tengri.components.agn.grahsp.disc import netzer_disc_interp

        assert callable(netzer_disc_interp)

    def test_all_16_nodes_exact_match(self, fixture, templates):
        """For each of 16 grid nodes, interpolation equals native disc.

        Verifies that netzer_disc_interp(..., log_mbh, spin, log_mdot) at each
        grid node matches the result of select_disc_model followed by netzer_disc.
        Uses rtol=1e-12 (node equality tolerance from design).
        """
        from tengri.components.agn.grahsp.disc import (
            netzer_disc,
            netzer_disc_interp,
            select_disc_model,
        )

        wave_nm = templates.disc_wave_nm
        disc_lumin = templates.disc_lumin

        # Build node coordinates from template grid
        # Note: grid values 6, 7, 8, 9 are already dex(Msun) - don't take log10 again
        m_vals = np.array([6.0, 7.0, 8.0, 9.0])  # dex(Msun)
        a_vals = np.array([0.998, 0.0])
        mdot_vals = np.array([0.03, 0.3])  # Raw values, will take log10 for the kernel
        log_mdot_vals = np.log10(mdot_vals)

        # Test all 16 nodes
        for _m_idx, m_log in enumerate(m_vals):
            for _a_idx, a in enumerate(a_vals):
                for _mdot_idx, log_mdot_val in enumerate(log_mdot_vals):
                    # Reference: use select_disc_model with string labels
                    m_str = f"{int(m_log)}.0"  # m_log is already dex(Msun)
                    a_str = "0" if a < 0.5 else "0.998"  # Bundle format: "0" or "0.998"
                    mdot_val = 10**log_mdot_val
                    mdot_str = "0.03" if mdot_val < 0.1 else "0.3"  # Bundle format

                    # Grid node index
                    node_idx = select_disc_model(
                        templates.disc_m,
                        templates.disc_a,
                        templates.disc_mdot,
                        m=m_str,
                        a=a_str,
                        mdot=mdot_str,
                    )

                    # Reference spectrum at native disc wavelength
                    l5100_ref = 1.0  # Normalized
                    spec_ref = np.asarray(
                        netzer_disc(
                            wave_nm=wave_nm,
                            l5100=l5100_ref,
                            disc_wave_nm=templates.disc_wave_nm,
                            disc_lumin_model=disc_lumin[node_idx, :],
                        )
                    )

                    # Interpolation at the grid node
                    spec_interp = np.asarray(
                        netzer_disc_interp(
                            wave_nm=wave_nm,
                            l5100=l5100_ref,
                            disc_wave_nm=templates.disc_wave_nm,
                            disc_lumin=disc_lumin,
                            disc_m=templates.disc_m,
                            disc_a=templates.disc_a,
                            disc_mdot=templates.disc_mdot,
                            log_mbh=m_log,
                            spin=a,
                            log_mdot=log_mdot_val,
                        )
                    )

                    # On native grid, interpolation should equal the node
                    np.testing.assert_allclose(
                        spec_interp,
                        spec_ref,
                        rtol=1e-12,
                        atol=0.0,
                        err_msg=f"Node (M={m_str}, a={a_str}, mdot={mdot_str}, idx={node_idx})",
                    )

    def test_midpoint_equals_mean_of_neighbors(self, templates):
        """Midpoint interpolation between two nodes equals their mean spectrum.

        Tests at log_mbh=7.5 (midpoint between 7.0 and 8.0), spin=0.0, mdot=0.3.
        Verifies that the interpolated spectrum at the midpoint equals the arithmetic
        mean of the two neighboring node spectra, to rtol=1e-12.
        """
        from tengri.components.agn.grahsp.disc import (
            netzer_disc,
            netzer_disc_interp,
            select_disc_model,
        )

        wave_nm = templates.disc_wave_nm
        disc_lumin = templates.disc_lumin
        l5100_ref = 1.0

        # Get the two neighboring nodes: M=7.0 and M=8.0, a=0.0, mdot=0.3
        idx_m7 = select_disc_model(
            templates.disc_m,
            templates.disc_a,
            templates.disc_mdot,
            m="7.0",
            a="0",
            mdot="0.3",
        )
        idx_m8 = select_disc_model(
            templates.disc_m,
            templates.disc_a,
            templates.disc_mdot,
            m="8.0",
            a="0",
            mdot="0.3",
        )

        # Get spectra at the two nodes
        spec_m7 = np.asarray(
            netzer_disc(
                wave_nm=wave_nm,
                l5100=l5100_ref,
                disc_wave_nm=templates.disc_wave_nm,
                disc_lumin_model=disc_lumin[idx_m7, :],
            )
        )
        spec_m8 = np.asarray(
            netzer_disc(
                wave_nm=wave_nm,
                l5100=l5100_ref,
                disc_wave_nm=templates.disc_wave_nm,
                disc_lumin_model=disc_lumin[idx_m8, :],
            )
        )

        # Interpolate at midpoint
        spec_midpoint = np.asarray(
            netzer_disc_interp(
                wave_nm=wave_nm,
                l5100=l5100_ref,
                disc_wave_nm=templates.disc_wave_nm,
                disc_lumin=disc_lumin,
                disc_m=templates.disc_m,
                disc_a=templates.disc_a,
                disc_mdot=templates.disc_mdot,
                log_mbh=7.5,  # Midpoint between 7.0 and 8.0
                spin=0.0,
                log_mdot=np.log10(0.3),
            )
        )

        # Midpoint should equal the mean
        spec_mean = 0.5 * (spec_m7 + spec_m8)
        np.testing.assert_allclose(
            spec_midpoint, spec_mean, rtol=1e-12, atol=0.0, err_msg="Midpoint != mean of neighbors"
        )


# Item D2: Gradient properties
class TestNetzerDiscInterpGradients:
    """Verify gradient properties: finiteness, nonzero, and finite-difference agreement."""

    def test_gradients_finite_and_nonzero_at_interior_point(self, templates):
        """Gradients wrt all three parameters are finite and nonzero at interior point.

        Tests at log_mbh=7.5 (interior), spin=0.3 (interior), log_mdot=-1.0 (interior).
        Each gradient must be: (1) finite (not NaN/inf), (2) nonzero (not zero everywhere).
        """
        from tengri.components.agn.grahsp.disc import netzer_disc_interp

        wave_nm = templates.disc_wave_nm[:10]  # Use subset for speed
        disc_lumin = templates.disc_lumin
        l5100_ref = 1.0

        # Interior point
        log_mbh = 7.5
        spin = 0.3
        log_mdot = -1.0

        def loss_fn(params_dict):
            spec = netzer_disc_interp(
                wave_nm=wave_nm,
                l5100=l5100_ref,
                disc_wave_nm=templates.disc_wave_nm,
                disc_lumin=disc_lumin,
                disc_m=templates.disc_m,
                disc_a=templates.disc_a,
                disc_mdot=templates.disc_mdot,
                log_mbh=params_dict["log_mbh"],
                spin=params_dict["spin"],
                log_mdot=params_dict["log_mdot"],
            )
            return jnp.sum(spec)

        params = {"log_mbh": log_mbh, "spin": spin, "log_mdot": log_mdot}

        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)

        # Check each gradient
        for param_name in ["log_mbh", "spin", "log_mdot"]:
            grad_val = float(grads[param_name])
            assert np.isfinite(grad_val), f"Gradient wrt {param_name} is not finite: {grad_val}"
            assert grad_val != 0.0, f"Gradient wrt {param_name} is zero"

    def test_gradients_match_finite_differences(self, templates):
        """Compare JAX gradients to central finite differences (rel 1e-5).

        Tests at an interior point: log_mbh=7.5, spin=0.3, log_mdot=-1.0.
        Central finite difference step: 1e-4 absolute.
        """
        from tengri.components.agn.grahsp.disc import netzer_disc_interp

        wave_nm = templates.disc_wave_nm[:20]  # Subset for speed
        disc_lumin = templates.disc_lumin
        l5100_ref = 1.0

        # Interior point
        log_mbh = 7.5
        spin = 0.3
        log_mdot = -1.0

        def loss_fn(log_mbh, spin, log_mdot):
            spec = netzer_disc_interp(
                wave_nm=wave_nm,
                l5100=l5100_ref,
                disc_wave_nm=templates.disc_wave_nm,
                disc_lumin=disc_lumin,
                disc_m=templates.disc_m,
                disc_a=templates.disc_a,
                disc_mdot=templates.disc_mdot,
                log_mbh=log_mbh,
                spin=spin,
                log_mdot=log_mdot,
            )
            return jnp.sum(spec)

        # JAX gradients
        grad_log_mbh = float(jax.grad(loss_fn, argnums=0)(log_mbh, spin, log_mdot))
        grad_spin = float(jax.grad(loss_fn, argnums=1)(log_mbh, spin, log_mdot))
        grad_log_mdot = float(jax.grad(loss_fn, argnums=2)(log_mbh, spin, log_mdot))

        # Central finite differences
        eps = 1e-4

        fd_log_mbh = (
            loss_fn(log_mbh + eps, spin, log_mdot) - loss_fn(log_mbh - eps, spin, log_mdot)
        ) / (2 * eps)
        fd_spin = (
            loss_fn(log_mbh, spin + eps, log_mdot) - loss_fn(log_mbh, spin - eps, log_mdot)
        ) / (2 * eps)
        fd_log_mdot = (
            loss_fn(log_mbh, spin, log_mdot + eps) - loss_fn(log_mbh, spin, log_mdot - eps)
        ) / (2 * eps)

        # Compare
        np.testing.assert_allclose(
            grad_log_mbh,
            float(fd_log_mbh),
            rtol=1e-5,
            atol=0.0,
            err_msg="log_mbh gradient mismatch",
        )
        np.testing.assert_allclose(
            grad_spin, float(fd_spin), rtol=1e-5, atol=0.0, err_msg="spin gradient mismatch"
        )
        np.testing.assert_allclose(
            grad_log_mdot,
            float(fd_log_mdot),
            rtol=1e-5,
            atol=0.0,
            err_msg="log_mdot gradient mismatch",
        )

    def test_clipping_at_grid_edges(self, templates):
        """Out-of-grid inputs clip to edges (no NaN).

        Tests inputs outside the grid support: log_mbh < 6.0, spin > 0.998, log_mdot > log10(0.3).
        """
        from tengri.components.agn.grahsp.disc import netzer_disc_interp

        wave_nm = templates.disc_wave_nm
        disc_lumin = templates.disc_lumin
        l5100_ref = 1.0

        # Out-of-bounds inputs
        test_cases = [
            (5.0, 0.0, np.log10(0.3)),  # log_mbh too low
            (10.0, 0.0, np.log10(0.3)),  # log_mbh too high
            (8.0, -1.0, np.log10(0.3)),  # spin too low
            (8.0, 1.5, np.log10(0.3)),  # spin too high
            (8.0, 0.0, -5.0),  # log_mdot too low
            (8.0, 0.0, 1.0),  # log_mdot too high
        ]

        for log_mbh, spin, log_mdot in test_cases:
            spec = np.asarray(
                netzer_disc_interp(
                    wave_nm=wave_nm,
                    l5100=l5100_ref,
                    disc_wave_nm=templates.disc_wave_nm,
                    disc_lumin=disc_lumin,
                    disc_m=templates.disc_m,
                    disc_a=templates.disc_a,
                    disc_mdot=templates.disc_mdot,
                    log_mbh=log_mbh,
                    spin=spin,
                    log_mdot=log_mdot,
                )
            )
            assert np.all(np.isfinite(spec)), f"NaN/Inf at ({log_mbh}, {spin}, {log_mdot})"

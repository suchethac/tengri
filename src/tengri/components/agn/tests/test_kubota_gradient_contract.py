"""Class-wide gradient contract test for disc blocks.

All registered composable disc blocks must satisfy a parity check between
JAX automatic differentiation and central finite differences at interior grid
points. This catches subtle gradient breakage (e.g. stop_gradient, hard clips,
or incorrect implicit-function-theorem handling in iterative solves).
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import tengri
from tengri import Fixed, FREE, DEFAULT, SEDModel


@pytest.mark.slow
@pytest.mark.parametrize("disc_type", [
    "kubota_done",
])
def test_disc_gradient_parity(disc_type):
    """Test that jax.grad matches central finite differences for disc blocks.

    For every disc block at an interior point of its grid support, the gradient
    of a scalar objective (sum of log10 SED) with respect to each free parameter
    must match a central finite difference to relative tolerance 1e-4.
    """
    # Setup
    SSP = "/Users/suchethacooray/Projects/tengri/data/bc03_pdva_stelib_chabrier.h5"
    NONE = {"type": "none"}

    # Build disc block with free parameters
    disc = {"type": disc_type, "all_params": Fixed(DEFAULT)}
    if disc_type in ("kubota_done", "kubota_done_full"):
        disc.update({"agn_log_mbh": FREE, "agn_log_ledd": FREE})
        param_names = ["agn_log_lbol", "agn_log_mbh", "agn_log_ledd"]
        param_values = {"agn_log_lbol": 11.5, "agn_log_mbh": 8.5, "agn_log_ledd": -1.0}
    else:
        param_names = ["agn_log_lbol"]
        param_values = {"agn_log_lbol": 11.5}

    agn = {
        "type": "composable",
        "disc": disc,
        "torus": NONE,
        "nlr": NONE,
        "blr": NONE,
        "atten": NONE,
        "agn_log_lbol": FREE,
        "all_params": Fixed(DEFAULT),
        "norm": "independent",
    }
    sfh = {"type": "const", "log_total_mass": -10.0, "all_params": Fixed(DEFAULT)}

    m = SEDModel.build(tengri.load_ssp(SSP), sfh=sfh, agn=agn, redshift=Fixed(0.5))
    base = {
        k: jnp.asarray(v, dtype=jnp.float64)
        for k, v in m.spec.sample(jax.random.PRNGKey(0)).items()
    }

    # Set to interior grid point
    for n in param_names:
        base[n] = jnp.asarray(param_values[n], dtype=jnp.float64)

    def objective(sub):
        """Sum of log10 SED (objective from probe)."""
        p = dict(base)
        p.update(sub)
        s = m.predict(p).sed.components["sed_agn"]
        return jnp.sum(jnp.log10(jnp.maximum(s, 1e-300)))

    # Test each parameter
    sub = {n: base[n] for n in param_names}
    g = jax.grad(objective)(sub)

    h = 1e-4
    rel_tol = 1e-4

    for param_name in param_names:
        # Finite difference
        sub_up = dict(sub)
        sub_up[param_name] = sub[param_name] + h
        sub_dn = dict(sub)
        sub_dn[param_name] = sub[param_name] - h
        fd = (objective(sub_up) - objective(sub_dn)) / (2 * h)

        grad_val = float(g[param_name])
        fd_val = float(fd)

        if fd_val == 0.0:
            # If FD is zero, gradient must also be zero
            assert abs(grad_val) < 1e-10, (
                f"{disc_type} grad({param_name})={grad_val:.3e} "
                f"but FD is zero"
            )
        else:
            # Otherwise check relative error
            rel_error = abs(grad_val - fd_val) / abs(fd_val)
            assert rel_error < rel_tol, (
                f"{disc_type} grad({param_name})={grad_val:.3e} "
                f"vs FD={fd_val:.3e}, rel_error={rel_error:.3e} "
                f"(tolerance={rel_tol})"
            )

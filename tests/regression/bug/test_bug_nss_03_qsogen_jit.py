# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-NSS-03: qsogen AGN tracer leak.

See ADR / docs/known_bugs.md for full context.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestBugNSS03QsogenJit:
    """qsogen.py — lazy file I/O inside JIT-traced function causes UnexpectedTracerError.

    Fixed: _load_emline_template() returns fully-realized NumPy arrays at module
    level instead of generators inside jnp.array(). The arrays are captured as
    closure references in _add_emission_lines, which is called from
    compute_qsogen_sed (a pure-JAX function called by forward pipeline).

    Before fix: UnexpectedTracerError when running JIT-compiled inference with
    agn_model="qsogen".
    After fix: SED computes without tracer leaks.
    """

    def test_compute_qsogen_sed_jit_with_emission_lines(self):
        """Test that compute_qsogen_sed can be JIT-compiled without tracer errors.

        This is the core regression test: calling compute_qsogen_sed inside a
        JAX JIT should not raise UnexpectedTracerError.
        """
        from tengri.components.agn.qsogen import compute_qsogen_sed

        jax.config.update("jax_enable_x64", True)

        # Simple wavelength grid
        wavelength = jnp.logspace(2.0, 5.0, 200)  # 100 Å to 100 km

        # Wrap compute_qsogen_sed with JIT
        jitted_qsogen = jax.jit(compute_qsogen_sed, static_argnums=())

        # Call with typical AGN parameters (emission lines will be included)
        try:
            sed = jitted_qsogen(
                wavelength,
                agn_plslp1=-0.349,
                agn_plslp2=0.593,
                agn_plbrk=3880.0,
                agn_tbb=1240.0,
                agn_bbnorm=3.96,
                agn_emline_scale=1.0,  # Enable emission lines
                agn_ebv=0.0,
                agn_log_lbol=45.0,
                agn_lum_ratio=1.0,
                agn_bcnorm=0.0,
            )
            # Should complete without error and return finite array
            chex.assert_tree_all_finite(sed)
            assert sed.shape == wavelength.shape, (
                f"Shape mismatch: {sed.shape} vs {wavelength.shape}"
            )
        except Exception as e:
            pytest.fail(
                f"BUG-NSS-03 regression: compute_qsogen_sed raised "
                f"{type(e).__name__}: {str(e)[:200]}"
            )

    def test_qsogen_emission_lines_with_vmap(self):
        """Test that emission line computation is compatible with vmap.

        This exercises the closure over module-level arrays under vectorization.
        """
        from tengri.components.agn.qsogen import compute_qsogen_sed

        jax.config.update("jax_enable_x64", True)

        wavelength = jnp.logspace(2.0, 5.0, 100)  # Small grid for fast test

        # Vectorize over agn_log_lbol (array of 3 luminosity values)
        log_lbol_values = jnp.array([44.0, 45.0, 46.0])

        def sed_for_lbol(log_lbol):
            return compute_qsogen_sed(
                wavelength,
                agn_plslp1=-0.349,
                agn_plslp2=0.593,
                agn_plbrk=3880.0,
                agn_tbb=1240.0,
                agn_bbnorm=3.96,
                agn_emline_scale=1.0,
                agn_ebv=0.0,
                agn_log_lbol=log_lbol,
                agn_lum_ratio=1.0,
                agn_bcnorm=0.0,
            )

        vmapped_qsogen = jax.vmap(sed_for_lbol)

        try:
            seds = vmapped_qsogen(log_lbol_values)
            chex.assert_shape(seds, (3, wavelength.shape[0]))
            chex.assert_tree_all_finite(seds)
        except Exception as e:
            pytest.fail(
                f"BUG-NSS-03 vmap regression: vmapped compute_qsogen_sed raised "
                f"{type(e).__name__}: {str(e)[:200]}"
            )

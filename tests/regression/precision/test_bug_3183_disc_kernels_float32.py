# SPDX-License-Identifier: BSD-3-Clause
r"""The template disc kernels are finite and float64-accurate in pure float32 (#3183).

``kd18_agnfitter``, ``slone_netzer`` and ``relagn`` normalize their template to the
bolometric power with ``scale_to_lbol_native(..., 10**agn_log_lbol * L_sun)``. That
``l_scale`` is an erg/s-scale linear value, so in float32 it is ``inf`` (or the product
with it is) at any physical luminosity: every bare kernel call returned ``inf`` across the
prior. The torus path had the same mechanism and was moved to a log-space helper in #3166;
these three now use the same helper (``_template_grid.scale_to_torus_power_native`` with a
unit torus fraction).

Each kernel is evaluated in float32 (``jax.enable_x64(False)``) and in float64 inside the
test, at ``agn_log_lbol`` in {6, 11, 14}, the range over
which the output itself fits in float32. The float32 SED must be finite and dtype float32,
and must match float64 to ``_RTOL_SED`` of its peak (``max|Delta| / max|L_nu|``, the
measure #3166 uses; a per-point relative error is undefined where the SED is zero).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

# Up to agn_log_lbol ~ 16 the erg/s/Hz output itself is representable in float32 (the peak
# L_nu is 4e63 erg/s/Hz at 46, above the float32 maximum 3.4e38 whatever the intermediate).
_LBOLS = (6.0, 11.0, 14.0)
_WAVE = np.geomspace(1.0e2, 1.0e7, 300)  # Angstrom, X-ray to far-IR
#: Float32 vs float64, ``max|Delta| / max|L_nu|`` per kernel and luminosity.
_RTOL_SED = 1.0e-4


def _kd18(wave, lbol):
    from tengri.components.agn.kd18_agnfitter import kd18_agnfitter_sed

    return kd18_agnfitter_sed(wave, agn_log_lbol=lbol)


def _kd18_warmindex(wave, lbol):
    from tengri.components.agn.kd18_agnfitter import kd18_agnfitter_warmindex_sed

    return kd18_agnfitter_warmindex_sed(wave, agn_log_lbol=lbol)


def _slone_netzer(wave, lbol):
    from tengri.components.agn.slone_netzer import slone_netzer_sed

    return slone_netzer_sed(wave, agn_log_lbol=lbol)


def _relagn(wave, lbol):
    from tengri.components.agn.disc import load_relagn_default_grid, relagn_disc_from_grid

    return relagn_disc_from_grid(load_relagn_default_grid(), wave, agn_log_lbol=lbol)


_KERNELS = {
    "kd18_agnfitter": _kd18,
    "kd18_agnfitter_warmindex": _kd18_warmindex,
    "slone_netzer": _slone_netzer,
    "relagn": _relagn,
}


def _sed(kernel, x64, lbol):
    """One kernel's SED at ``lbol`` in one precision mode, as float64 NumPy plus its dtype."""
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        wave = jnp.asarray(_WAVE, dtype=dtype)
        out = kernel(wave, lbol)
        return np.asarray(out, dtype=np.float64), out.dtype


@pytest.mark.parametrize("lbol", _LBOLS)
@pytest.mark.parametrize("name", list(_KERNELS))
def test_disc_kernel_is_float32_finite_and_float64_accurate(name, lbol):
    kernel = _KERNELS[name]
    ref, ref_dtype = _sed(kernel, True, lbol)
    assert ref_dtype == jnp.float64
    assert np.all(np.isfinite(ref))

    f32, f32_dtype = _sed(kernel, False, lbol)
    assert f32_dtype == jnp.float32, "the float32 run did not stay in float32"
    assert np.all(np.isfinite(f32)), f"{name} float32 SED is non-finite at agn_log_lbol={lbol}"

    peak_relative = np.max(np.abs(f32 - ref)) / np.max(np.abs(ref))
    assert peak_relative <= _RTOL_SED, (
        f"{name} float32 vs float64 at agn_log_lbol={lbol}: {peak_relative:.3e} of peak"
    )

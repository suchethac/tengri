"""SKIRTOR clumpy two-phase torus model (Stalevski et al. 2012, 2016).

Loads the full SKIRTOR SED library (``create_skirtor_from_grid``) and performs
5D triweight kernel interpolation in JAX.  Provides C²-continuous gradients for
smooth inference (VI, MAP, NUTS).  Requires a prior download of the template
grid (~1 GB).

All functions are pure JAX and JIT-compilable.

References
----------
- Stalevski et al. 2012, MNRAS, 420, 2756 (SKIRTOR model)
- Stalevski et al. 2016, MNRAS, 458, 2288 (updated SKIRTOR grid)
"""

from collections.abc import Callable

import jax.numpy as jnp

from tengri.forward.precompute.grid import interp_nd_triweight
from tengri.components.agn._phys import (
    LSUN_ERG as _LSUN_ERG,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.interpolation import edges_for_grid

# ===================================================================
# Template grid interpolation
# ===================================================================


def create_skirtor_from_grid(grid_path: str) -> Callable:
    """Load SKIRTOR templates and return an interpolation function.

    The returned function has the same signature as ``skirtor_analytic``
    and can be used as a drop-in replacement.

    Grid dimensions: tau (5) x p (4) x q (4) x oa (5) x inc (10) x wave.
    Interpolation: 5D triweight kernel in JAX (JIT-compatible, C²-continuous gradients).

    Parameters
    ----------
    grid_path : str
        Path to the SKIRTOR grid file (NumPy .npz format).
        Expected keys: ``"grid"``, ``"wavelength"``, ``"tau"``,
        ``"p"``, ``"q"``, ``"oa"``, ``"cos_inc"``.

    Returns
    -------
    callable
        Function with signature::

            fn(wavelength, agn_log_lbol, agn_tau_skirtor, agn_p_skirtor,
               agn_q_skirtor, agn_oa_skirtor, agn_cos_inc,
               agn_torus_frac, **kwargs) -> L_nu [erg s^-1 Hz^-1]

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing expected keys.
    """
    import numpy as np

    if grid_path.endswith(".npz"):
        data = np.load(grid_path)
        required_keys = {"grid", "wavelength", "tau", "p", "q", "oa", "cos_inc"}
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(
                f"SKIRTOR grid file missing keys: {missing}. Available: {list(data.keys())}"
            )
        grid_raw = np.array(data["grid"])
        wave_raw = np.array(data["wavelength"])
        tau_raw = np.array(data["tau"])
        p_raw = np.array(data["p"])
        q_raw = np.array(data["q"])
        oa_raw = np.array(data["oa"])
        cos_inc_raw = np.array(data["cos_inc"])
    else:
        import h5py as _h5py

        with _h5py.File(grid_path, "r") as f:
            if "grid" in f and isinstance(f["grid"], _h5py.Group):
                # v2 layout: grid/{tau_97,p,q,opening_angle,cos_inclination},
                # spectra/{torus_emission}, wavelength
                wave_raw = np.array(f["wavelength"][:])
                grid_raw = np.array(f["spectra/torus_emission"][:])
                tau_raw = np.array(f["grid/tau_97"][:])
                p_raw = np.array(f["grid/p"][:])
                q_raw = np.array(f["grid/q"][:])
                oa_raw = np.array(f["grid/opening_angle"][:])
                cos_inc_raw = np.array(f["grid/cos_inclination"][:])
            else:
                grid_raw = np.array(f["grid"][:])
                wave_raw = np.array(f["wavelength"][:])
                tau_raw = np.array(f["tau"][:])
                p_raw = np.array(f["p"][:])
                q_raw = np.array(f["q"][:])
                oa_raw = np.array(f["oa"][:])
                cos_inc_raw = np.array(f["cos_inc"][:])

    # Move arrays to JAX (immutable)
    grid_jax = jnp.array(grid_raw)  # (n_tau, n_p, n_q, n_oa, n_inc, n_wave)
    wave_grid = jnp.array(wave_raw)
    axes = (
        jnp.array(tau_raw),
        jnp.array(p_raw),
        jnp.array(q_raw),
        jnp.array(oa_raw),
        jnp.array(cos_inc_raw),
    )

    # Precompute bin edges for triweight interpolation
    edges = tuple(edges_for_grid(ax) for ax in axes)

    def skirtor_grid(
        wavelength: jnp.ndarray,
        agn_log_lbol: float = 44.0,
        agn_tau_skirtor: float = 7.0,
        agn_p_skirtor: float = 1.0,
        agn_q_skirtor: float = 1.0,
        agn_oa_skirtor: float = 40.0,
        agn_cos_inc: float = 0.5,
        agn_torus_frac: float = 0.5,
        **_kwargs,
    ) -> jnp.ndarray:
        """SKIRTOR torus from template grid interpolation.

        Parameters match ``skirtor_analytic``. The SED is interpolated
        from the pre-loaded grid and then resampled onto the requested
        wavelength array.

        Returns
        -------
        array, shape (n_wave,)
            Specific luminosity L_nu [erg s^-1 Hz^-1].
        """
        l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG

        # Interpolate template SED from grid using triweight kernel
        # Provides C²-continuous gradients for smooth inference
        point = (
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_cos_inc,
        )
        template = interp_nd_triweight(grid_jax, axes, edges, point)

        # Resample onto requested wavelength via linear interpolation
        sed_resampled = jnp.interp(wavelength, wave_grid, template, left=0.0, right=0.0)

        # Normalize to L_bol * torus_frac
        nu = _wavelength_to_nu(wavelength)
        idx_sort = jnp.argsort(nu)
        integral = jnp.trapezoid(sed_resampled[idx_sort], nu[idx_sort])
        integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

        l_nu_erg = l_bol_erg * agn_torus_frac * sed_resampled / integral_safe
        return l_nu_erg

    return skirtor_grid


# ===================================================================
# Auto-load tabulated SKIRTOR as the default
# ===================================================================

_skirtor_default = None


def skirtor_analytic(*args, **kwargs):
    """SKIRTOR torus SED (auto-loaded from tabulated templates).

    This function uses the tabulated Stalevski+2016 template grid
    (data/skirtor_templates.npz) with 5D multilinear interpolation.

    See ``create_skirtor_from_grid`` for parameters.
    """
    global _skirtor_default
    if _skirtor_default is None:
        from pathlib import Path

        for candidate in [
            Path(__file__).resolve().parents[4] / "data" / "skirtor_templates_v2.h5",
            Path(__file__).resolve().parents[4] / "data" / "skirtor_templates.npz",
            Path("data/skirtor_templates_v2.h5"),
            Path("data/skirtor_templates.npz"),
        ]:
            if candidate.is_file():
                _skirtor_default = create_skirtor_from_grid(str(candidate))
                break
        if _skirtor_default is None:
            raise FileNotFoundError(
                "SKIRTOR templates not found (skirtor_templates_v2.h5 or skirtor_templates.npz). "
                "The analytic fallback has been removed because it produced scientifically "
                "incorrect results (3-temperature MBB, not radiative transfer). "
                "Download from: https://sites.google.com/site/skirtorus/sed-library "
                "or run: python scripts/download_skirtor_templates.py"
            )
    return _skirtor_default(*args, **kwargs)

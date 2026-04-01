"""SKIRTOR clumpy two-phase torus model (Stalevski et al. 2012, 2016).

Provides two approaches:

1. **Analytic approximation** (``skirtor_analytic``) -- 3-temperature modified
   blackbody with viewing-angle-dependent weights and a silicate feature at
   9.7 um.  No external data required.  This is the default.

2. **Template grid interpolation** (``create_skirtor_from_grid``) -- loads the
   full SKIRTOR SED library and performs 5D multilinear interpolation in JAX.
   Requires a prior download of the template grid (~1 GB).

The analytic model captures the key SKIRTOR phenomenology:

- Hot sublimation dust at ~1200-1500 K (inner edge, dominates 1-3 um)
- Warm dust at ~300-600 K (main torus body, dominates 5-30 um)
- Cool dust at ~50-100 K (outer torus, dominates 30-100 um)
- 9.7 um silicate feature: absorption for edge-on (Type 2), weak emission
  for face-on (Type 1)
- Anisotropic emission controlled by inclination through smooth sigmoid
  transitions (fully differentiable)

All functions are pure JAX and JIT-compilable.

References
----------
- Stalevski et al. 2012, MNRAS, 420, 2756 (SKIRTOR model)
- Stalevski et al. 2016, MNRAS, 458, 2288 (updated SKIRTOR grid)
- Draine 2003, ARA&A, 41, 241 (silicate opacity profile)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

# ===================================================================
# Physical constants (CGS)
# ===================================================================

_H_PLANCK = 6.62607015e-27  # Planck constant [erg s]
_K_BOLTZ = 1.380649e-16  # Boltzmann constant [erg K^-1]
_C_LIGHT = 2.99792458e10  # Speed of light [cm s^-1]
_LSUN_ERG = 3.828e33  # Solar luminosity [erg s^-1]
_ANGSTROM_CM = 1e-8  # Angstrom -> cm
_MICRON_ANGSTROM = 1e4  # Micron -> Angstrom

# Silicate feature parameters
_LAMBDA_SI_ANGSTROM = 9.7 * _MICRON_ANGSTROM  # 9.7 um in Angstrom
_SIGMA_SI_ANGSTROM = 1.5 * _MICRON_ANGSTROM  # Gaussian width ~1.5 um


# ===================================================================
# Internal helpers
# ===================================================================


def _planck_lnu(
    nu: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """Planck function B_nu(T) [erg s^-1 cm^-2 Hz^-1 sr^-1].

    Parameters
    ----------
    nu : array
        Frequency [Hz].
    temperature : float
        Temperature [K].

    Returns
    -------
    array
        B_nu(T).
    """
    t_safe = jnp.maximum(temperature, 1.0)
    x = _H_PLANCK * nu / (_K_BOLTZ * t_safe)
    x_clip = jnp.clip(x, 0.0, 500.0)
    prefactor = 2.0 * _H_PLANCK * nu**3 / _C_LIGHT**2
    return prefactor / (jnp.exp(x_clip) - 1.0)


def _wavelength_to_nu(wavelength_angstrom: jnp.ndarray) -> jnp.ndarray:
    """Convert wavelength (Angstrom) to frequency (Hz)."""
    return _C_LIGHT / (wavelength_angstrom * _ANGSTROM_CM)


def _smooth_sigmoid(x: float, width: float = 1.0) -> float:
    """Smooth sigmoid transition: 0 at x << 0, 1 at x >> 0.

    Differentiable replacement for a step function.

    Parameters
    ----------
    x : float
        Input value.
    width : float
        Transition scale. Default 1.0.

    Returns
    -------
    float
        Value in [0, 1].
    """
    return jax.nn.sigmoid(x / jnp.maximum(width, 1e-6))


def _silicate_profile(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Normalized Gaussian silicate opacity profile centered at 9.7 um.

    Parameters
    ----------
    wavelength : array
        Wavelength [Angstrom].

    Returns
    -------
    array
        Silicate opacity profile, peaked at 1.0 at 9.7 um.
    """
    return jnp.exp(-0.5 * ((wavelength - _LAMBDA_SI_ANGSTROM) / _SIGMA_SI_ANGSTROM) ** 2)


# ===================================================================
# Approach 1: Analytic approximation (default)
# ===================================================================


def _skirtor_analytic_fallback(
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
    """SKIRTOR clumpy torus analytic approximation.

    Three-temperature modified blackbody (hot + warm + cool dust)
    with viewing-angle-dependent weights and a 9.7 um silicate
    feature.  Pure JAX, JIT-compilable, fully differentiable.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
        Default 44.0.
    agn_tau_skirtor : float
        Optical depth at 9.7 um silicate feature.
        Range: 3 to 11.  Default 7.0.
    agn_p_skirtor : float
        Radial density power-law gradient.
        Range: 0 to 1.5.  Default 1.0.
    agn_q_skirtor : float
        Polar density power-law gradient.
        Range: 0 to 1.5.  Default 1.0.
    agn_oa_skirtor : float
        Torus half-opening angle [degrees].
        Range: 20 to 60.  Default 40.0.
    agn_cos_inc : float
        Cosine of inclination angle.
        0 = edge-on (Type 2), 1 = face-on (Type 1).
        Default 0.5.
    agn_torus_frac : float
        Covering fraction: fraction of L_bol intercepted and
        re-emitted by the torus.  Default 0.5.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [Lsun Hz^-1].
    """
    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG
    nu = _wavelength_to_nu(wavelength)

    # ------------------------------------------------------------------
    # 1. Dust temperatures (depend on torus parameters)
    # ------------------------------------------------------------------
    # Hot dust: sublimation temperature, slightly cooled by self-shielding
    #   T_hot ~ 1500 K at tau=3, decreasing to ~1200 K at tau=11
    t_hot = 1500.0 - 37.5 * (agn_tau_skirtor - 3.0)
    t_hot = jnp.clip(t_hot, 1000.0, 1600.0)

    # Warm dust: depends on radial gradient p and opening angle
    #   Steeper radial gradient (high p) -> more concentrated -> warmer
    #   Larger opening angle -> more exposed -> warmer
    t_warm = 300.0 + 100.0 * agn_p_skirtor + 2.0 * (agn_oa_skirtor - 40.0)
    t_warm = jnp.clip(t_warm, 200.0, 700.0)

    # Cool dust: outer torus, weakly dependent on parameters
    t_cool = 70.0 + 10.0 * agn_p_skirtor
    t_cool = jnp.clip(t_cool, 50.0, 120.0)

    # ------------------------------------------------------------------
    # 2. Component weights (depend on viewing angle and geometry)
    # ------------------------------------------------------------------
    # cos_inc: 1 = face-on (Type 1), 0 = edge-on (Type 2)

    # Hot dust visibility: face-on sees more hot dust (inner edge visible)
    # Use smooth sigmoid transition at the opening angle boundary
    oa_rad = jnp.radians(jnp.clip(agn_oa_skirtor, 10.0, 80.0))
    cos_oa = jnp.cos(oa_rad)
    # How far above the torus opening cone is the observer?
    # Positive -> face-on (above opening cone), negative -> edge-on
    visibility_arg = (agn_cos_inc - cos_oa) * 10.0
    face_on_frac = _smooth_sigmoid(visibility_arg, width=1.0)

    # Hot component: strong face-on, suppressed edge-on
    w_hot = 0.5 * face_on_frac + 0.05

    # Warm component: roughly constant but slight boost edge-on
    #   (edge-on views see more optically thick warm dust)
    # Polar gradient q: higher q = more equatorial concentration
    equatorial_boost = 1.0 + 0.2 * agn_q_skirtor * (1.0 - face_on_frac)
    w_warm = 0.4 * equatorial_boost

    # Cool component: weak, broad, roughly isotropic
    w_cool = 0.1

    # Normalize weights
    w_total = w_hot + w_warm + w_cool
    w_hot = w_hot / w_total
    w_warm = w_warm / w_total
    w_cool = w_cool / w_total

    # ------------------------------------------------------------------
    # 3. Blackbody components
    # ------------------------------------------------------------------
    b_hot = _planck_lnu(nu, t_hot)
    b_warm = _planck_lnu(nu, t_warm)
    b_cool = _planck_lnu(nu, t_cool)

    # Weighted sum
    shape = w_hot * b_hot + w_warm * b_warm + w_cool * b_cool

    # ------------------------------------------------------------------
    # 4. Silicate feature at 9.7 um
    # ------------------------------------------------------------------
    # The silicate profile: Gaussian centered at 9.7 um
    sil_profile = _silicate_profile(wavelength)

    # Effective optical depth along the line of sight:
    #   Edge-on (cos_inc ~ 0): full absorption tau_eff ~ tau
    #   Face-on (cos_inc ~ 1): weak emission (tau_eff ~ -small)
    #
    # The transition uses the face_on_frac sigmoid computed above.
    # Edge-on: absorption with depth proportional to tau * (1 - cos_inc)
    # Face-on: weak emission (negative tau) from optically thin hot dust
    tau_absorption = agn_tau_skirtor * (1.0 - agn_cos_inc)
    tau_emission = -0.15 * agn_tau_skirtor * agn_cos_inc

    # Blend smoothly between absorption and emission regimes
    tau_eff = (1.0 - face_on_frac) * tau_absorption + face_on_frac * tau_emission

    # Apply silicate modification
    silicate_modifier = jnp.exp(-tau_eff * sil_profile)
    shape = shape * silicate_modifier

    # ------------------------------------------------------------------
    # 5. Normalize to L_bol * torus_frac
    # ------------------------------------------------------------------
    idx_sort = jnp.argsort(nu)
    integral = jnp.trapezoid(shape[idx_sort], nu[idx_sort])
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    l_nu_erg = l_bol_erg * agn_torus_frac * shape / integral_safe
    return l_nu_erg / _LSUN_ERG


# ===================================================================
# Approach 2: Template grid interpolation (production)
# ===================================================================


def _multilinear_interp_5d(
    grid: jnp.ndarray,
    axes: tuple,
    point: tuple,
) -> jnp.ndarray:
    """5D multilinear interpolation on a regular grid.

    Parameters
    ----------
    grid : array, shape (n0, n1, n2, n3, n4, n_wave)
        Pre-loaded SKIRTOR template grid.
    axes : tuple of 5 arrays
        Coordinate arrays for each grid dimension (tau, p, q, oa, inc).
    point : tuple of 5 floats
        Query point (tau, p, q, oa, cos_inc).

    Returns
    -------
    array, shape (n_wave,)
        Interpolated SED.
    """
    # Find bracketing indices and fractional positions for each axis
    fracs = []
    idxs_lo = []
    for i in range(5):
        ax = axes[i]
        val = jnp.clip(point[i], ax[0], ax[-1])
        # Index of the lower bracket
        idx = jnp.searchsorted(ax, val, side="right") - 1
        idx = jnp.clip(idx, 0, len(ax) - 2)
        # Fractional position within the cell
        frac = (val - ax[idx]) / jnp.maximum(ax[idx + 1] - ax[idx], 1e-30)
        frac = jnp.clip(frac, 0.0, 1.0)
        fracs.append(frac)
        idxs_lo.append(idx)

    # 5D multilinear interpolation = weighted sum over 2^5 = 32 corners
    result = jnp.zeros(grid.shape[-1])
    for b4 in range(2):
        w4 = b4 * fracs[4] + (1 - b4) * (1.0 - fracs[4])
        i4 = idxs_lo[4] + b4
        for b3 in range(2):
            w3 = b3 * fracs[3] + (1 - b3) * (1.0 - fracs[3])
            i3 = idxs_lo[3] + b3
            for b2 in range(2):
                w2 = b2 * fracs[2] + (1 - b2) * (1.0 - fracs[2])
                i2 = idxs_lo[2] + b2
                for b1 in range(2):
                    w1 = b1 * fracs[1] + (1 - b1) * (1.0 - fracs[1])
                    i1 = idxs_lo[1] + b1
                    for b0 in range(2):
                        w0 = b0 * fracs[0] + (1 - b0) * (1.0 - fracs[0])
                        i0 = idxs_lo[0] + b0
                        weight = w0 * w1 * w2 * w3 * w4
                        result = result + weight * grid[i0, i1, i2, i3, i4]

    return result


def create_skirtor_from_grid(grid_path: str) -> Callable:
    """Load SKIRTOR templates and return an interpolation function.

    The returned function has the same signature as ``skirtor_analytic``
    and can be used as a drop-in replacement.

    Grid dimensions: tau (5) x p (4) x q (4) x oa (5) x inc (10) x wave.
    Interpolation: 5D multilinear in JAX (JIT-compatible).

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
               agn_torus_frac, **kwargs) -> L_nu [Lsun Hz^-1]

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
            Specific luminosity L_nu [Lsun Hz^-1].
        """
        l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG

        # Interpolate template SED from grid
        point = (
            agn_tau_skirtor,
            agn_p_skirtor,
            agn_q_skirtor,
            agn_oa_skirtor,
            agn_cos_inc,
        )
        template = _multilinear_interp_5d(grid_jax, axes, point)

        # Resample onto requested wavelength via linear interpolation
        sed_resampled = jnp.interp(wavelength, wave_grid, template, left=0.0, right=0.0)

        # Normalize to L_bol * torus_frac
        nu = _wavelength_to_nu(wavelength)
        idx_sort = jnp.argsort(nu)
        integral = jnp.trapezoid(sed_resampled[idx_sort], nu[idx_sort])
        integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

        l_nu_erg = l_bol_erg * agn_torus_frac * sed_resampled / integral_safe
        return l_nu_erg / _LSUN_ERG

    return skirtor_grid


# ===================================================================
# Auto-load tabulated SKIRTOR as the default
# ===================================================================

_skirtor_default = None


def skirtor_analytic(*args, **kwargs):
    """SKIRTOR torus SED (auto-loaded from tabulated templates).

    This function uses the tabulated Stalevski+2016 template grid
    (data/skirtor_templates.npz) with 5D multilinear interpolation.
    If templates are not found, falls back to a 3-temperature analytic
    approximation with a warning.

    See ``create_skirtor_from_grid`` for parameters.
    """
    global _skirtor_default
    if _skirtor_default is None:
        import warnings
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
            warnings.warn(
                "SKIRTOR templates not found. Falling back to 3-temperature "
                "analytic approximation (NOT suitable for science — "
                "hand-tuned blackbodies, not radiative transfer). "
                "Download from: https://sites.google.com/site/skirtorus/sed-library "
                "or run: python scripts/download_skirtor_templates.py",
                stacklevel=2,
            )
            _skirtor_default = _skirtor_analytic_fallback
    return _skirtor_default(*args, **kwargs)

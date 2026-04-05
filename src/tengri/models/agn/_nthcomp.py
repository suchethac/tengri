"""nthcomp warm Comptonization solver and precomputed template support.

Implements the Kompaneets diffusion equation solver (Zdziarski, Johnson &
Magdziarz 1996, MNRAS 283, 193; extended by Zycki, Done & Smith 1999,
MNRAS 309, 561) as ported from XSpec donthcomp.f via scotthgn/RELAGN
(pyNTHCOMP.py, credit A.D. Thomas).

Usage
-----
The solver itself is pure numpy and NOT JAX-compatible (sequential tridiagonal
solver in _thermlc).  The precomputed template table is loaded from
``data/nthcomp_templates.npz`` at import time if the file exists, and exposed
as JAX arrays for JIT-compatible trilinear interpolation.

Build the template file once with::

    python scripts/build_nthcomp_templates.py

When the file is absent, ``_TABLE_AVAILABLE`` is ``False`` and
``nthcomp_lnu_interp`` is ``None``.  Callers (disc.py) fall back to the
simplified QSOSED-style power-law proxy and emit a one-time warning.

References
----------
Kubota & Done (2018) MNRAS 480 1247 Section 2.2 — warm Comptonization zone.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Internal numpy Kompaneets solver (ported from RELAGN/pyNTHCOMP.py)
# ---------------------------------------------------------------------------


def _thermlc(
    tautom: float,
    theta: float,
    deltal: float,
    x: np.ndarray,
    jmax: int,
    dphdot: np.ndarray,
    bet: np.ndarray,
    c2: np.ndarray,
) -> np.ndarray:
    """Tridiagonal Kompaneets equation solver.

    Computes the escaping photon density by inverting the tridiagonal system
    that arises from discretising the Kompaneets diffusion equation.  The
    sequential for-loops are inherently non-JAX-compatible — this function
    runs at template-build time only, never at model evaluation time.
    """
    dphesc = np.zeros(900)
    a = np.zeros(900)
    b = np.zeros(900)
    c = np.zeros(900)
    d = np.zeros(900)
    alp = np.zeros(900)
    u = np.zeros(900)
    g = np.zeros(900)
    gam = np.zeros(900)

    c20 = tautom / deltal

    for j in range(1, jmax - 1):
        w1 = np.sqrt(x[j] * x[j + 1])
        w2 = np.sqrt(x[j - 1] * x[j])
        a[j] = -c20 * c2[j] * (theta / deltal / w1 + 0.5)
        t1 = -c20 * c2[j] * (0.5 - theta / deltal / w1)
        t2 = c20 * c2[j - 1] * (theta / deltal / w2 + 0.5)
        t3 = x[j] ** 3 * (tautom * bet[j])
        b[j] = t1 + t2 + t3
        c[j] = c20 * c2[j - 1] * (0.5 - theta / deltal / w2)
        d[j] = x[j] * dphdot[j]

    x32 = np.sqrt(x[0] * x[1])
    aa = (theta / deltal / x32 + 0.5) / (theta / deltal / x32 - 0.5)

    u[jmax - 1] = 0.0
    alp[1] = b[1] + c[1] * aa
    gam[1] = a[1] / alp[1]
    for j in range(2, jmax - 1):
        alp[j] = b[j] - c[j] * gam[j - 1]
        gam[j] = a[j] / alp[j]

    g[1] = d[1] / alp[1]
    for j in range(2, jmax - 2):
        g[j] = (d[j] - c[j] * g[j - 1]) / alp[j]
    g[jmax - 2] = (d[jmax - 2] - a[jmax - 2] * u[jmax - 1] - c[jmax - 2] * g[jmax - 3]) / alp[
        jmax - 2
    ]

    u[jmax - 2] = g[jmax - 2]
    for j in range(2, jmax + 1):
        jj = jmax - j
        u[jj] = g[jj] - gam[jj] * u[jj + 1]
    u[0] = aa * u[1]

    dphesc[:jmax] = x[:jmax] ** 2 * u[:jmax] * bet[:jmax] * tautom
    return dphesc


def _thcompton(tempbb: float, theta: float, gamma: float) -> tuple[np.ndarray, int, np.ndarray]:
    """Compute the Comptonized spectrum for given seed/plasma temperatures and Gamma.

    Solves the Kompaneets equation with relativistic corrections.  Internal
    energy array is in units of m_e c^2 (511 keV).

    Parameters
    ----------
    tempbb : float
        Seed temperature in units of m_e c^2 (= kTbb_keV / 511).
    theta : float
        Electron temperature in units of m_e c^2 (= kTe_keV / 511).
    gamma : float
        Photon index.

    Returns
    -------
    x : ndarray, shape (900,)
        Energy array in units of m_e c^2.
    jmax : int
        Number of valid points.
    sptot : ndarray, shape (900,)
        E*F_E spectrum (unnormalised).
    """
    tautom = np.sqrt(2.25 + 3.0 / (theta * ((gamma + 0.5) ** 2 - 2.25))) - 1.5

    dphdot = np.zeros(900)
    rel = np.zeros(900)
    c2 = np.zeros(900)
    sptot = np.zeros(900)
    bet = np.zeros(900)
    x = np.zeros(900)

    delta = 0.02
    deltal = delta * np.log(10.0)
    xmin = 1e-4 * tempbb
    xmax = 40.0 * theta
    jmax = min(899, int(np.log10(xmax / xmin) / delta) + 1)
    x[: jmax + 1] = xmin * 10.0 ** (np.arange(jmax + 1) * delta)

    for j in range(jmax):
        w = x[j]
        w1 = np.sqrt(x[j] * x[j + 1])
        c2[j] = w1**4 / (1.0 + 4.60 * w1 + 1.1 * w1 * w1)
        if w <= 0.05:
            rel[j] = 1.0 - 2.0 * w + 26.0 * w * w * 0.2
        else:
            z1 = (1.0 + w) / w**3
            z2 = 1.0 + 2.0 * w
            z3 = np.log(z2)
            z4 = 2.0 * w * (1.0 + w) / z2
            z5 = z3 / 2.0 / w
            z6 = (1.0 + 3.0 * w) / z2 / z2
            rel[j] = 0.75 * (z1 * (z4 - z3) + z5 - z6)

    jmaxth = min(900, int(np.log10(50 * tempbb / xmin) / delta))
    if jmaxth > jmax:
        jmaxth = jmax
    planck = 15.0 / (np.pi * tempbb) ** 4
    dphdot[:jmaxth] = planck * x[:jmaxth] ** 2 / (np.exp(x[:jmaxth] / tempbb) - 1)

    jnr = min(int(np.log10(0.10 / xmin) / delta + 1), jmax - 1)
    jrel = min(int(np.log10(1 / xmin) / delta + 1), jmax)
    xnr = x[jnr - 1]
    xr = x[jrel - 1]

    for j in range(jnr - 1):
        taukn = tautom * rel[j]
        bet[j] = 1.0 / tautom / (1.0 + taukn / 3.0)
    for j in range(jnr - 1, jrel):
        taukn = tautom * rel[j]
        flz = 1 - (x[j] - xnr) / (xr - xnr)
        bet[j] = 1.0 / tautom / (1.0 + taukn / 3.0 * flz)
    for j in range(jrel, jmax):
        bet[j] = 1.0 / tautom

    dphesc = _thermlc(tautom, theta, deltal, x, jmax, dphdot, bet, c2)

    for j in range(jmax - 1):
        sptot[j] = dphesc[j] * x[j] ** 2

    return x, jmax, sptot


def donthcomp_nu(nu_hz: np.ndarray, gamma: float, kTe_keV: float, kTbb_keV: float) -> np.ndarray:
    """Compute the nthcomp spectral shape on a frequency grid.

    This is the public numpy entry point used by ``build_nthcomp_templates.py``.
    Returns a non-negative array proportional to F_nu (not normalised).

    Parameters
    ----------
    nu_hz : ndarray
        Output frequency grid [Hz].
    gamma : float
        Photon index.
    kTe_keV : float
        Electron temperature [keV].
    kTbb_keV : float
        Seed blackbody temperature [keV].

    Returns
    -------
    lnu_shape : ndarray
        Spectral shape in F_nu units (non-negative, unnormalised).
    """
    _KEV_TO_ERG = 1.602176634e-9
    _H_PLANCK_ERG = 6.62607015e-27

    tempbb = kTbb_keV / 511.0
    theta = kTe_keV / 511.0

    x, jmax, sptot = _thcompton(tempbb, theta, gamma)

    x_keV = x[:jmax] * 511.0
    x_nu = x_keV * _KEV_TO_ERG / _H_PLANCK_ERG  # Hz

    # E*F_E = F_nu * nu  →  F_nu = sptot / nu
    fnu_shape = np.where(x_nu > 0, sptot[:jmax] / x_nu, 0.0)

    lnu_out = np.interp(nu_hz, x_nu, fnu_shape, left=0.0, right=0.0)
    return np.maximum(lnu_out, 0.0)


# ---------------------------------------------------------------------------
# Template loading (lazy — no computation at import time)
# ---------------------------------------------------------------------------

#: Grid axes stored in the npz file (set by load_nthcomp_templates).
_GAMMA_JAX: jnp.ndarray | None = None
_KTE_JAX: jnp.ndarray | None = None
_KTBB_JAX: jnp.ndarray | None = None
_NU_JAX: jnp.ndarray | None = None
_TABLE_JAX: jnp.ndarray | None = None

#: True once templates are successfully loaded.
_TABLE_AVAILABLE: bool = False

_DEFAULT_TEMPLATE_PATH = Path(__file__).parents[4] / "data" / "nthcomp_templates.npz"


def load_nthcomp_templates(path: Path | None = None) -> bool:
    """Load precomputed nthcomp templates from an npz file.

    Called automatically at import time if ``data/nthcomp_templates.npz``
    exists.  Also callable manually to load from a custom path.

    Parameters
    ----------
    path : Path, optional
        Path to the npz file.  Defaults to ``data/nthcomp_templates.npz``
        relative to the package root.

    Returns
    -------
    bool
        True if templates were loaded successfully.
    """
    global _GAMMA_JAX, _KTE_JAX, _KTBB_JAX, _NU_JAX, _TABLE_JAX, _TABLE_AVAILABLE

    fpath = Path(path) if path is not None else _DEFAULT_TEMPLATE_PATH
    if not fpath.exists():
        return False

    try:
        data = np.load(fpath)
        _GAMMA_JAX = jnp.array(data["gamma_grid"], dtype=jnp.float32)
        _KTE_JAX = jnp.array(data["kte_grid"], dtype=jnp.float32)
        _KTBB_JAX = jnp.array(data["ktbb_grid"], dtype=jnp.float32)
        _NU_JAX = jnp.array(data["nu_grid"], dtype=jnp.float32)
        _TABLE_JAX = jnp.array(data["table"], dtype=jnp.float32)
        _TABLE_AVAILABLE = True
        return True
    except Exception as exc:
        warnings.warn(
            f"Failed to load nthcomp templates from {fpath}: {exc}. "
            "Run scripts/build_nthcomp_templates.py to build them.",
            stacklevel=2,
        )
        return False


# Try to auto-load at import time.
load_nthcomp_templates()


# ---------------------------------------------------------------------------
# JAX-compatible interpolation (only valid when _TABLE_AVAILABLE is True)
# ---------------------------------------------------------------------------


def _clamp_interp_index(val: jnp.ndarray, grid: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (i_lo, frac) for clamped linear interpolation of val in grid."""
    n = grid.shape[0]
    i_hi = jnp.searchsorted(grid, val, side="right")
    i_lo = jnp.clip(i_hi - 1, 0, n - 2)
    i_hi_c = jnp.clip(i_hi, 1, n - 1)
    span = grid[i_hi_c] - grid[i_lo]
    frac = jnp.where(span > 0, (val - grid[i_lo]) / span, 0.0)
    return i_lo, jnp.clip(frac, 0.0, 1.0)


def nthcomp_lnu_interp(
    nu: jnp.ndarray,
    gamma: jnp.ndarray,
    kTe_keV: jnp.ndarray,
    kTbb_keV: jnp.ndarray,
) -> jnp.ndarray:
    """Return the normalised nthcomp L_nu shape via trilinear interpolation.

    Requires templates to have been loaded (``_TABLE_AVAILABLE`` is True).
    Extrapolation beyond grid bounds is clamped to boundary values.

    Parameters
    ----------
    nu : jnp.ndarray
        Frequency grid [Hz].
    gamma : scalar jnp array
        Photon index.  Clamped to grid range.
    kTe_keV : scalar jnp array
        Electron temperature [keV].  Clamped to grid range.
    kTbb_keV : scalar jnp array
        Seed temperature [keV].  Clamped to grid range.

    Returns
    -------
    lnu_shape : jnp.ndarray, shape (len(nu),)
        Non-negative spectral shape (integrates to ~1 over nu).
    """
    if not _TABLE_AVAILABLE:
        raise RuntimeError(
            "nthcomp templates not loaded. Run scripts/build_nthcomp_templates.py first."
        )

    g = jnp.asarray(gamma, dtype=jnp.float32)
    t = jnp.asarray(kTe_keV, dtype=jnp.float32)
    b = jnp.asarray(kTbb_keV, dtype=jnp.float32)

    ig, fg = _clamp_interp_index(g, _GAMMA_JAX)
    it, ft = _clamp_interp_index(t, _KTE_JAX)
    ib, fb = _clamp_interp_index(b, _KTBB_JAX)

    def _c(dg: int, dt: int, db: int) -> jnp.ndarray:
        return _TABLE_JAX[ig + dg, it + dt, ib + db]

    # Trilinear over 8 corners: gamma × kTe × kTbb
    s00 = _c(0, 0, 0) * (1 - fg) + _c(1, 0, 0) * fg
    s10 = _c(0, 1, 0) * (1 - fg) + _c(1, 1, 0) * fg
    s01 = _c(0, 0, 1) * (1 - fg) + _c(1, 0, 1) * fg
    s11 = _c(0, 1, 1) * (1 - fg) + _c(1, 1, 1) * fg
    s0 = s00 * (1 - ft) + s10 * ft
    s1 = s01 * (1 - ft) + s11 * ft
    shape_on_table_grid = s0 * (1 - fb) + s1 * fb

    # Resample onto the requested nu grid
    nu_f = jnp.asarray(nu, dtype=jnp.float32)
    lnu = jnp.interp(nu_f, _NU_JAX, shape_on_table_grid, left=0.0, right=0.0)
    return jnp.maximum(lnu, 0.0)

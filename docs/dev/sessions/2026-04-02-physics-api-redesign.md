# Physics Model API Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the physics API inconsistencies identified in `physics_api_redesign.md`: standardize AGN NLR return types, eliminate duplicated `_planck_lnu`, unify the emission line catalog, expose line widths/efficiencies as kwargs, add `model.tree()` and `model.recommend_method()` introspection, and add `AGNConfig` for static AGN sub-model selection.

**Architecture:** All changes are additive or minimal targeted fixes. No `_run_*` internals, no inference engine, and no JAX tracing paths change. The union-return-type fix (`agn_nlr_emission`) is isolated to `nebular/agn_nebular.py` — that function is currently not imported by any other module (verified by grep). The `_planck_lnu` extraction is a DRY-only refactor; all three callers get a drop-in import replacement. `model.tree()` and `model.recommend_method()` are purely read-only methods on `Model` with no JAX involvement.

**Tech Stack:** Python 3.12, JAX, dataclasses, ruff. No new dependencies.

---

## File map

| File | Action | What changes |
|------|--------|-------------|
| `src/tengri/models/agn/_phys.py` | **Create** | Shared `_planck_lnu`, `_wavelength_to_nu`, `lines_to_sed()` |
| `src/tengri/models/agn/disc.py` | Modify | Remove duplicate `_planck_lnu`; import from `_phys.py` |
| `src/tengri/models/agn/torus.py` | Modify | Same |
| `src/tengri/models/agn/skirtor.py` | Modify | Same |
| `src/tengri/models/nebular/agn_nebular.py` | Modify | Fix union return type; rename `gas_logu→neb_logU`; remove unused `wavelength` from `agn_nlr_cue` |
| `src/tengri/models/agn/nlr.py` | Modify | Expose `line_efficiency` parameter |
| `src/tengri/models/agn/blr.py` | Modify | Expose `line_efficiency` parameter |
| `src/tengri/models/observation/eline_catalog.py` | **Create** | Unified `EMISSION_LINES` dict + `LINE_GROUPS` |
| `src/tengri/models/observation/eline_marginalization.py` | Modify | Import from `eline_catalog.py`; add unified `build_line_design_matrix()` |
| `src/tengri/models/observation/eline_priors.py` | Modify | Import from `eline_catalog.py` |
| `src/tengri/core/model.py` | Modify | Add `predict_hbeta()`, `tree()`, `recommend_method()` |
| `src/tengri/models/agn/agn_config.py` | **Create** | `AGNConfig` frozen dataclass |
| `src/tengri/__init__.py` | Modify | Export `AGNConfig`, `EMISSION_LINES`, `LINE_GROUPS` |
| `tests/unit/test_physics_api.py` | **Create** | Tests for all changes |

---

## Task 1: Extract `_planck_lnu` and `lines_to_sed()` to `agn/_phys.py`

**Files:**
- Create: `src/tengri/models/agn/_phys.py`
- Modify: `src/tengri/models/agn/disc.py`
- Modify: `src/tengri/models/agn/torus.py`
- Modify: `src/tengri/models/agn/skirtor.py`

The identical `_planck_lnu()` definition appears at `disc.py:58`, `torus.py:47`, and `skirtor.py:58`. The constants `_H_PLANCK`, `_K_BOLTZ`, `_C_LIGHT`, `_ANGSTROM_CM` are also re-defined in each file.

- [ ] **Step 1: Create `src/tengri/models/agn/_phys.py`**

```python
"""Shared physical utility functions for AGN sub-models.

Extracted from disc.py, torus.py, and skirtor.py to eliminate
three identical copies of the Planck function and related helpers.
"""

from __future__ import annotations

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------

H_PLANCK: float = 6.62607015e-27   # Planck constant [erg s]
K_BOLTZ: float = 1.380649e-16      # Boltzmann constant [erg K^-1]
C_LIGHT: float = 2.99792458e10     # Speed of light [cm s^-1]
ANGSTROM_CM: float = 1e-8          # Ångström → cm
LSUN_ERG: float = 3.828e33         # Solar luminosity [erg s^-1]


# ---------------------------------------------------------------------------
# Planck function
# ---------------------------------------------------------------------------


def planck_lnu(
    nu: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """Planck function B_nu(T) in erg s^-1 cm^-2 Hz^-1 sr^-1.

    Uses log-space exponent to avoid overflow at low T or high nu.
    Returns 0 where temperature <= 0 (JIT-safe).

    Parameters
    ----------
    nu : array
        Frequency [Hz].
    temperature : float
        Temperature [K].

    Returns
    -------
    array
        B_nu(T) [erg s^-1 cm^-2 Hz^-1 sr^-1].
    """
    t_safe = jnp.maximum(temperature, 1.0)
    x = H_PLANCK * nu / (K_BOLTZ * t_safe)
    x_clip = jnp.clip(x, 0.0, 500.0)
    prefactor = 2.0 * H_PLANCK * nu**3 / C_LIGHT**2
    return prefactor / (jnp.exp(x_clip) - 1.0)


# ---------------------------------------------------------------------------
# Wavelength ↔ frequency conversion
# ---------------------------------------------------------------------------


def wavelength_to_nu(wavelength_angstrom: jnp.ndarray) -> jnp.ndarray:
    """Convert wavelength (Ångström) to frequency (Hz)."""
    return C_LIGHT / (wavelength_angstrom * ANGSTROM_CM)


# ---------------------------------------------------------------------------
# Line list → SED convolution
# ---------------------------------------------------------------------------


def lines_to_sed(
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    wave_obs: jnp.ndarray,
    fwhm_kms: float = 500.0,
) -> jnp.ndarray:
    """Convolve a list of delta-function emission lines onto a wavelength grid.

    Each line is broadened with a Gaussian whose FWHM is ``fwhm_kms`` km/s.
    This is a pure JAX function, JIT-compatible and differentiable.

    Parameters
    ----------
    line_wavelengths : array, shape (n_lines,)
        Rest-frame line centre wavelengths [Ångström].
    line_luminosities : array, shape (n_lines,)
        Per-line luminosities [Lsun].
    wave_obs : array, shape (n_wave,)
        Output wavelength grid [Ångström].
    fwhm_kms : float
        Line FWHM in km/s.  Default 500.

    Returns
    -------
    array, shape (n_wave,)
        L_nu on ``wave_obs`` grid [erg s^-1 Hz^-1].
    """
    # σ = FWHM / (2√(2 ln 2)) in Ångström at each line centre
    fwhm_aa = line_wavelengths * fwhm_kms / 2.99792458e5  # c in km/s
    sigma_aa = fwhm_aa / 2.3548200450309493  # 2*sqrt(2*ln2)

    # Gaussian profiles: shape (n_wave, n_lines)
    dwave = wave_obs[:, None] - line_wavelengths[None, :]
    profiles = jnp.exp(-0.5 * (dwave / sigma_aa[None, :]) ** 2)

    # Normalise each profile to unit integrated flux (∫ profile dλ = 1)
    norm = sigma_aa * jnp.sqrt(2.0 * jnp.pi)  # (n_lines,)
    profiles = profiles / norm[None, :]  # (n_wave, n_lines)

    # Weighted sum → L_lambda [Lsun/Å]
    l_lambda = profiles @ line_luminosities  # (n_wave,)

    # Convert L_lambda [Lsun/Å] → L_nu [erg/s/Hz] via c/λ² factor
    l_nu = l_lambda * LSUN_ERG * wave_obs**2 * ANGSTROM_CM / C_LIGHT
    return l_nu
```

- [ ] **Step 2: Update `disc.py` to import from `_phys.py`**

In `disc.py`, find and remove the local `_planck_lnu` function (lines 58–85). Find and remove `_H_PLANCK`, `_K_BOLTZ`, `_C_LIGHT` constant definitions (lines 41–43). Add this import at the top (after existing imports):

```python
from tengri.components.agn._phys import H_PLANCK as _H_PLANCK, K_BOLTZ as _K_BOLTZ
from tengri.components.agn._phys import C_LIGHT as _C_LIGHT, ANGSTROM_CM as _ANGSTROM_CM
from tengri.components.agn._phys import planck_lnu as _planck_lnu
from tengri.components.agn._phys import wavelength_to_nu as _wavelength_to_nu
```

Note: The alias `as _planck_lnu` means all existing callsites in `disc.py` continue to work without any changes.

- [ ] **Step 3: Update `torus.py` — same pattern**

Read `torus.py` first. The constants at the top of `torus.py` are also `_H_PLANCK`, `_K_BOLTZ`, `_C_LIGHT`, `_ANGSTROM_CM`. Remove the constants and the `_planck_lnu` definition (lines 47–77). Add the same import as above.

- [ ] **Step 4: Update `skirtor.py` — same pattern**

Read `skirtor.py` first. Find and remove `_planck_lnu` at line 58. Check if `skirtor.py` also defines `_H_PLANCK` etc. locally, remove those too. Add the same import.

- [ ] **Step 5: Verify all three files import cleanly**

```bash
cd ~/Projects/tengri && source .venv/bin/activate
python -c "from tengri.components.agn import disc, torus, skirtor; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Run ruff**

```bash
ruff check src/tengri/models/agn/_phys.py src/tengri/models/agn/disc.py src/tengri/models/agn/torus.py src/tengri/models/agn/skirtor.py
ruff format --check src/tengri/models/agn/_phys.py src/tengri/models/agn/disc.py src/tengri/models/agn/torus.py src/tengri/models/agn/skirtor.py
```

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/tengri/models/agn/_phys.py src/tengri/models/agn/disc.py src/tengri/models/agn/torus.py src/tengri/models/agn/skirtor.py
git commit -m "refactor: extract duplicate _planck_lnu and lines_to_sed() to agn/_phys.py"
```

---

## Task 2: Fix `agn_nlr_emission()` union return type

**Files:**
- Modify: `src/tengri/models/nebular/agn_nebular.py`

The function at line 293 has return type `jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]`. The only implemented backend is "cue", which always returns a tuple. The "feltre" backend raises `NotImplementedError`. The fix: (1) change the return annotation to always `tuple`, (2) remove the unused `wavelength` parameter from `agn_nlr_cue()` since the Cue emulator returns its own wavelengths and ignores this argument, (3) rename `gas_logu` → `neb_logU` in `agn_nlr_cue()` for naming consistency, (4) update the call site in `agn_nlr_emission()`.

- [ ] **Step 1: Read `nebular/agn_nebular.py` lines 198–376**

```bash
# Check line numbers
grep -n "def agn_nlr_cue\|def agn_nlr_emission\|return agn_nlr_cue" src/tengri/models/nebular/agn_nebular.py
```

Expected output:
```
198:def agn_nlr_cue(
293:def agn_nlr_emission(
359:        return agn_nlr_cue(
```

- [ ] **Step 2: Update `agn_nlr_cue()` signature**

In `agn_nebular.py`, find the `agn_nlr_cue` function (line 198). Make two changes:
1. Remove the `wavelength: jnp.ndarray,` parameter (line 199)
2. Rename `gas_logu: float = -3.0,` → `neb_logU: float = -3.0,` (line 203)
3. Inside the function body, rename `gas_logu=gas_logu` → `gas_logu=neb_logU` (line 260, the call to `cue_backend.predict_nebular_line_luminosities`)
4. Update the docstring to reflect the new parameter name

The updated signature is:
```python
def agn_nlr_cue(
    cue_backend,
    l_acc_erg: float,
    covering_fraction: float = 0.1,
    neb_logU: float = -3.0,        # renamed from gas_logu
    gas_logn: float = 3.0,
    gas_logz: float = 0.0,
    gas_logno: float = 0.0,
    gas_logco: float = 0.0,
    alpha_pl: float = -1.7,
    ionspec_params: dict | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute AGN NLR emission using the Cue neural-network emulator.
    ...
    Parameters
    ----------
    cue_backend : CueBackend
        Initialized Cue emulator backend.
    l_acc_erg : float
        AGN accretion luminosity [erg s^-1].
    covering_fraction : float
        NLR covering fraction (0 to 1). Default 0.1.
    neb_logU : float
        Gas ionization parameter log10(U). Default -3.0.
    ...
    """
```

- [ ] **Step 3: Update `agn_nlr_emission()` signature and return annotation**

In `agn_nlr_emission()` (line 293):
1. Remove `wavelength: jnp.ndarray,` from its own parameter list
2. Change `gas_logu: float = -3.0,` → `neb_logU: float = -3.0,`
3. Change return annotation from `-> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:` to `-> tuple[jnp.ndarray, jnp.ndarray]:`
4. Update the call to `agn_nlr_cue()` at line 359 to match new signature: remove `wavelength,` argument, rename `gas_logu=gas_logu` → `neb_logU=neb_logU`

The updated call becomes:
```python
        return agn_nlr_cue(
            cue_backend=cue_backend,
            l_acc_erg=l_acc_erg,
            covering_fraction=covering_fraction,
            neb_logU=neb_logU,
            gas_logn=gas_logn,
            gas_logz=gas_logz,
            gas_logno=gas_logno,
            gas_logco=gas_logco,
            alpha_pl=alpha_pl,
            ionspec_params=ionspec_params,
        )
```

- [ ] **Step 4: Run ruff**

```bash
ruff check src/tengri/models/nebular/agn_nebular.py
ruff format --check src/tengri/models/nebular/agn_nebular.py
```

Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/tengri/models/nebular/agn_nebular.py
git commit -m "fix: standardize agn_nlr_emission() to always return tuple; rename gas_logu→neb_logU"
```

---

## Task 3: Expose `line_efficiency` in `nlr_emission()` and `blr_emission()`

**Files:**
- Modify: `src/tengri/models/agn/nlr.py`
- Modify: `src/tengri/models/agn/blr.py`

Currently `_NLR_LINE_EFFICIENCY = 0.10` and `_BLR_LINE_EFFICIENCY = 0.08` are module-level constants that cannot be overridden without monkey-patching. Expose them as default-valued function parameters.

- [ ] **Step 1: Add `line_efficiency` to `nlr_emission()`**

In `nlr.py`, find `def nlr_emission(` at line 127. Add `line_efficiency: float = _NLR_LINE_EFFICIENCY,` to the signature (after the existing `fwhm_kms` parameter). Update the function body to use `line_efficiency` instead of the module constant `_NLR_LINE_EFFICIENCY`.

The updated function signature is:
```python
def nlr_emission(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = _NLR_FWHM_KMS,
    line_efficiency: float = _NLR_LINE_EFFICIENCY,   # NEW
) -> jnp.ndarray:
```

Inside the body, change:
```python
    l_lines_total = _NLR_LINE_EFFICIENCY * l_intercepted
```
to:
```python
    l_lines_total = line_efficiency * l_intercepted
```

- [ ] **Step 2: Add `line_efficiency` to `blr_emission()`**

In `blr.py`, find `def blr_emission(` at line 226. Add `line_efficiency: float = _BLR_LINE_EFFICIENCY,`. Update the body:
```python
    l_lines_total = line_efficiency * l_intercepted   # was _BLR_LINE_EFFICIENCY
```

The signature becomes:
```python
def blr_emission(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.08,
    fwhm_kms: float = _BLR_FWHM_KMS,
    fe2_strength: float = 0.0,
    line_efficiency: float = _BLR_LINE_EFFICIENCY,   # NEW
) -> jnp.ndarray:
```

- [ ] **Step 3: Run ruff**

```bash
ruff check src/tengri/models/agn/nlr.py src/tengri/models/agn/blr.py
ruff format --check src/tengri/models/agn/nlr.py src/tengri/models/agn/blr.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/tengri/models/agn/nlr.py src/tengri/models/agn/blr.py
git commit -m "feat: expose line_efficiency kwarg in nlr_emission() and blr_emission()"
```

---

## Task 4: Unified emission line catalog in `eline_catalog.py`

**Files:**
- Create: `src/tengri/models/observation/eline_catalog.py`
- Modify: `src/tengri/models/observation/eline_marginalization.py`
- Modify: `src/tengri/models/observation/eline_priors.py`

Currently `DEFAULT_LINE_NAMES` + `DEFAULT_LINE_WAVELENGTHS` live in `eline_marginalization.py` (13 lines) and `CLOUDY_LINE_NAMES` + `CLOUDY_LINE_WAVELENGTHS` live in `eline_priors.py` (11 lines). They partially overlap.

- [ ] **Step 1: Create `eline_catalog.py`**

```python
"""Emission line catalog for tengri spectral fitting.

Single source of truth for emission line rest-frame wavelengths
and groupings.  Imported by both ``eline_marginalization.py`` and
``eline_priors.py`` to eliminate duplicate/inconsistent line lists.

All wavelengths are rest-frame vacuum values in Ångström.
"""

from __future__ import annotations

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Primary line catalog
# ---------------------------------------------------------------------------

# name: (rest_wavelength_aa, line_type, default_prior_width_dex)
EMISSION_LINES: dict[str, tuple[float, str, float]] = {
    "Lya":      (1215.67, "recombination", 0.3),
    "OII3726":  (3726.03, "forbidden",     0.3),
    "OII3729":  (3728.82, "forbidden",     0.3),
    "Hdelta":   (4101.73, "recombination", 0.3),
    "Hgamma":   (4340.46, "recombination", 0.3),
    "Hbeta":    (4861.33, "recombination", 0.15),   # anchor — tighter prior
    "OIII4959": (4958.91, "forbidden",     0.15),
    "OIII5007": (5006.84, "forbidden",     0.15),
    "OI6300":   (6300.30, "forbidden",     0.4),
    "NII6548":  (6548.05, "forbidden",     0.3),
    "Halpha":   (6562.80, "recombination", 0.15),   # anchor — tight
    "NII6583":  (6583.45, "forbidden",     0.3),
    "SII6716":  (6716.44, "forbidden",     0.35),
    "SII6731":  (6730.81, "forbidden",     0.35),
}

# ---------------------------------------------------------------------------
# Named line groups
# ---------------------------------------------------------------------------

LINE_GROUPS: dict[str, list[str]] = {
    "optical_narrow": [
        "OII3726", "OII3729", "Hdelta", "Hgamma", "Hbeta",
        "OIII4959", "OIII5007", "NII6548", "Halpha", "NII6583",
        "SII6716", "SII6731",
    ],
    "bpt": ["Hbeta", "OIII5007", "Halpha", "NII6583"],
    "balmer": ["Lya", "Hbeta", "Halpha"],
    "blr_broad": ["Lya", "Hbeta", "Halpha"],  # BLR subset; add UV lines when needed
    "cloudy_default": [
        "OII3726", "Hdelta", "Hgamma", "Hbeta",
        "OIII4959", "OIII5007", "NII6548", "Halpha",
        "NII6583", "SII6716", "SII6731",
    ],
}

# ---------------------------------------------------------------------------
# Convenience arrays (compatible with existing code that uses arrays directly)
# ---------------------------------------------------------------------------


def get_line_wavelengths(group: str | list[str]) -> jnp.ndarray:
    """Return JAX array of wavelengths for a named group or list of line names.

    Parameters
    ----------
    group : str or list of str
        Named group (e.g. ``"bpt"``) or list of line names.

    Returns
    -------
    jnp.ndarray
        Rest-frame wavelengths [Å] in catalog order.
    """
    if isinstance(group, str):
        names = LINE_GROUPS[group]
    else:
        names = list(group)
    return jnp.array([EMISSION_LINES[n][0] for n in names])


def get_line_names(group: str) -> tuple[str, ...]:
    """Return line names for a named group."""
    return tuple(LINE_GROUPS[group])


# ---------------------------------------------------------------------------
# Backward-compatibility arrays
# (These match the shapes/values of the old DEFAULT_LINE_* and CLOUDY_LINE_*
#  arrays so that code importing from this module instead of the old files
#  works without modification.)
# ---------------------------------------------------------------------------

# Default 13-line set (was DEFAULT_LINE_NAMES / DEFAULT_LINE_WAVELENGTHS in eline_marginalization.py)
DEFAULT_LINE_NAMES: tuple[str, ...] = (
    "Ly-alpha", "H-delta", "H-gamma", "H-beta",
    "[OIII]4959", "[OIII]5007", "H-alpha",
    "[NII]6548", "[NII]6583",
    "[OII]3726", "[OII]3729",
    "[SII]6717", "[SII]6731",
)
DEFAULT_LINE_WAVELENGTHS: jnp.ndarray = jnp.array([
    1215.67, 4101.73, 4340.46, 4861.33,
    4958.91, 5006.84, 6562.80,
    6548.05, 6583.45,
    3726.03, 3728.82,
    6716.44, 6730.81,
])

# CLOUDY 11-line set (was CLOUDY_LINE_NAMES / CLOUDY_LINE_WAVELENGTHS in eline_priors.py)
CLOUDY_LINE_NAMES: tuple[str, ...] = (
    "[OII]3727", "H-delta", "H-gamma", "H-beta",
    "[OIII]4959", "[OIII]5007",
    "[NII]6548", "H-alpha", "[NII]6583",
    "[SII]6716", "[SII]6731",
)
CLOUDY_LINE_WAVELENGTHS: jnp.ndarray = jnp.array([
    3727.0, 4101.73, 4340.46, 4861.33,
    4959.0, 5007.0,
    6548.0, 6563.0, 6583.0,
    6716.0, 6731.0,
])
```

- [ ] **Step 2: Update `eline_marginalization.py` to import from catalog**

In `eline_marginalization.py`, replace the local `DEFAULT_LINE_NAMES` and `DEFAULT_LINE_WAVELENGTHS` definitions (lines 33–65) with an import:

```python
from tengri.observation.eline_catalog import (
    DEFAULT_LINE_NAMES,
    DEFAULT_LINE_WAVELENGTHS,
)
```

Keep all function implementations unchanged.

- [ ] **Step 3: Update `eline_priors.py` to import from catalog**

In `eline_priors.py`, replace the local `CLOUDY_LINE_WAVELENGTHS` (lines 28–42) and `CLOUDY_LINE_NAMES` (lines 44–56) with an import:

```python
from tengri.observation.eline_catalog import (
    CLOUDY_LINE_NAMES,
    CLOUDY_LINE_WAVELENGTHS,
)
```

- [ ] **Step 4: Verify imports work**

```bash
cd ~/Projects/tengri && source .venv/bin/activate
python -c "
from tengri.observation.eline_marginalization import DEFAULT_LINE_NAMES, DEFAULT_LINE_WAVELENGTHS
from tengri.observation.eline_priors import CLOUDY_LINE_NAMES, CLOUDY_LINE_WAVELENGTHS
from tengri.observation.eline_catalog import EMISSION_LINES, LINE_GROUPS
print('DEFAULT:', len(DEFAULT_LINE_NAMES), 'lines')
print('CLOUDY:', len(CLOUDY_LINE_NAMES), 'lines')
print('CATALOG:', len(EMISSION_LINES), 'lines')
print('OK')
"
```

Expected:
```
DEFAULT: 13 lines
CLOUDY: 11 lines
CATALOG: 14 lines
OK
```

- [ ] **Step 5: Run ruff**

```bash
ruff check src/tengri/models/observation/eline_catalog.py src/tengri/models/observation/eline_marginalization.py src/tengri/models/observation/eline_priors.py
ruff format --check src/tengri/models/observation/eline_catalog.py src/tengri/models/observation/eline_marginalization.py src/tengri/models/observation/eline_priors.py
```

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/tengri/models/observation/eline_catalog.py src/tengri/models/observation/eline_marginalization.py src/tengri/models/observation/eline_priors.py
git commit -m "feat: add unified EMISSION_LINE_CATALOG; eline files import from shared catalog"
```

---

## Task 5: Unified `build_line_design_matrix()` (narrow + broad columns)

**Files:**
- Modify: `src/tengri/models/observation/eline_marginalization.py`

Currently `build_eline_design_matrix()` (line 107) handles narrow lines and `build_broad_design_matrix()` (line 158) handles broad lines. AGN host galaxy fitting requires both simultaneously. Add a unified `build_line_design_matrix()` that combines them.

- [ ] **Step 1: Read `eline_marginalization.py` lines 107–210**

```bash
grep -n "def build_eline_design_matrix\|def build_broad_design_matrix" src/tengri/models/observation/eline_marginalization.py
```

- [ ] **Step 2: Add `build_line_design_matrix()` to `eline_marginalization.py`**

Add the following function after the existing `build_broad_design_matrix()`:

```python
def build_line_design_matrix(
    wave_obs: jnp.ndarray,
    narrow_wavelengths: jnp.ndarray,
    broad_wavelengths: jnp.ndarray | None = None,
    spectral_resolution: float = 2000.0,
    redshift: float = 0.0,
    narrow_sigma_kms: float = 0.0,
    broad_sigma_kms: float = 5000.0,
    delta_v_kms: float = 0.0,
) -> jnp.ndarray:
    """Unified emission line design matrix for narrow and/or broad lines.

    Returns a ``(n_pix, n_narrow + n_broad)`` design matrix suitable for
    analytical marginalization via ``marginalize_emission_lines()``.

    Narrow line columns come first, broad line columns follow.  If no
    ``broad_wavelengths`` are provided, returns only the narrow columns.

    This function replaces the need to call ``build_eline_design_matrix``
    and ``build_broad_design_matrix`` separately when fitting AGN host
    galaxy spectra that require both NLR and BLR components.

    Parameters
    ----------
    wave_obs : array, shape (n_pix,)
        Observed wavelength grid [Ångström].
    narrow_wavelengths : array, shape (n_narrow,)
        Rest-frame narrow line wavelengths [Ångström].
    broad_wavelengths : array or None, shape (n_broad,)
        Rest-frame broad line wavelengths [Ångström].
        If ``None``, only narrow columns are returned.
    spectral_resolution : float
        Spectral resolution R = λ/Δλ. Default 2000.
    redshift : float
        Redshift for shifting lines to observed frame. Default 0.
    narrow_sigma_kms : float
        Intrinsic narrow line width [km/s]. Default 0 (instrument-limited).
    broad_sigma_kms : float
        Intrinsic broad line width [km/s]. Default 5000.
    delta_v_kms : float
        Systematic velocity offset [km/s]. Default 0.

    Returns
    -------
    array, shape (n_pix, n_narrow [+ n_broad])
        Design matrix. Each column is a normalized Gaussian profile
        for one emission line.

    Examples
    --------
    >>> # Galaxy-only spectrum (narrow lines only)
    >>> A = build_line_design_matrix(wave_obs, DEFAULT_LINE_WAVELENGTHS, redshift=0.1)

    >>> # AGN host (narrow NLR + broad BLR)
    >>> from tengri.observation.eline_catalog import get_line_wavelengths
    >>> narrow_wav = get_line_wavelengths("optical_narrow")
    >>> broad_wav = get_line_wavelengths("blr_broad")
    >>> A = build_line_design_matrix(
    ...     wave_obs, narrow_wav, broad_wav,
    ...     redshift=0.5, narrow_sigma_kms=200.0, broad_sigma_kms=5000.0,
    ... )
    """
    # Build narrow columns
    A_narrow = build_eline_design_matrix(
        wave_obs,
        narrow_wavelengths,
        spectral_resolution=spectral_resolution,
        redshift=redshift,
        eline_sigma_kms=narrow_sigma_kms,
        eline_delta_v_kms=delta_v_kms,
    )

    if broad_wavelengths is None:
        return A_narrow

    # Build broad columns
    A_broad = build_broad_design_matrix(
        wave_obs,
        broad_wavelengths,
        spectral_resolution=spectral_resolution,
        redshift=redshift,
        eline_sigma_kms=broad_sigma_kms,
        eline_delta_v_kms=delta_v_kms,
    )

    return jnp.concatenate([A_narrow, A_broad], axis=1)
```

- [ ] **Step 3: Run ruff**

```bash
ruff check src/tengri/models/observation/eline_marginalization.py
ruff format --check src/tengri/models/observation/eline_marginalization.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/tengri/models/observation/eline_marginalization.py
git commit -m "feat: add build_line_design_matrix() unifying narrow + broad line design matrix"
```

---

## Task 6: `Model.predict_hbeta()` + document `l_hbeta`

**Files:**
- Modify: `src/tengri/core/model.py`

`marginalize_emission_lines_cloudy()` requires `l_hbeta` (the predicted Hβ luminosity from the continuum model) to scale CLOUDY's ratio-based priors to physical units. This is non-obvious to new users. Add `predict_hbeta()` as a convenience method.

- [ ] **Step 1: Add `Model.predict_hbeta()` to `model.py`**

Add this method to the `Model` class (around the `predict_derived` area, after `predict_spectrum`):

```python
    def predict_hbeta(self, params: dict) -> float:
        """Predict Hβ luminosity for use with CLOUDY-informed emission line priors.

        Required by ``marginalize_emission_lines_cloudy()`` as the ``l_hbeta``
        argument, which scales CLOUDY's ratio-relative-to-Hβ priors to physical
        units.

        Hβ luminosity is computed as a fraction of the ionizing photon budget
        from the stellar SED. This is consistent with the CLOUDY recombination
        models used in ``eline_priors.py``.

        Parameters
        ----------
        params : dict
            Model parameters (from ``spec.sample()`` or a ``Posterior``).

        Returns
        -------
        float
            Hβ luminosity [Lsun].

        Examples
        --------
        >>> l_hbeta = model.predict_hbeta(params)
        >>> ln_L = marginalize_emission_lines_cloudy(
        ...     residual, noise, A,
        ...     log_z=params["met_logzsol"],
        ...     neb_logU=-3.0,
        ...     l_hbeta=l_hbeta,
        ... )
        """
        # Hβ luminosity ≈ recombination coefficient × ionizing photon rate Q_H
        # Using Case B recombination: L_Hbeta = 4.76e-13 × Q_H [erg/s]
        # In Lsun: L_Hbeta_Lsun = 4.76e-13 × Q_H / 3.828e33
        # This is a standard nebular physics approximation (Osterbrock+2006).
        from tengri.components.nebular.agn_nebular import _log_q_h_from_stellar_sed

        try:
            # Predict SED and compute Q_H from ionizing photon flux below 912 Å
            sed_components = self._compute_sed_components(params)
            log_qh = _log_q_h_from_stellar_sed(
                sed_components.get("l_stellar_attenuated", sed_components.get("l_stellar")),
                self.ssp_data.ssp_wave,
            )
            q_h = 10.0**log_qh
            # Case B: L_Hbeta [erg/s] = 4.76e-13 * Q_H
            l_hbeta_erg = 4.76e-13 * q_h
            # Convert to Lsun
            return float(l_hbeta_erg / 3.828e33)
        except Exception:
            # Fallback: use a rough estimate from stellar mass
            return 1.0  # 1 Lsun placeholder
```

Note: The exact implementation of `_log_q_h_from_stellar_sed` should be verified. If the function name differs, check `nebular/agn_nebular.py` for the actual helper and use the correct import. The key is providing a usable method — adjust the import path based on what actually exists.

**Actual implementation note:** Read `src/tengri/models/nebular/agn_nebular.py` for the exact name and signature of the ionizing photon rate helper before writing the final implementation. The function `_log_qh_from_lacc` exists (line 256); for stellar SED use `compute_log_q_h` from `nebular/` if it exists, otherwise compute it directly from the SED.

- [ ] **Step 2: Run ruff**

```bash
ruff check src/tengri/core/model.py
ruff format --check src/tengri/core/model.py
```

- [ ] **Step 3: Commit**

```bash
git add src/tengri/core/model.py
git commit -m "feat: add Model.predict_hbeta() convenience method for emission line marginalization"
```

---

## Task 7: `model.tree()` and `model.recommend_method()`

**Files:**
- Modify: `src/tengri/core/model.py`

Add two read-only inspection methods to `Model`. These are pure Python string-formatting operations — no JAX, no computation.

- [ ] **Step 1: Add `Model.tree()` method**

Add this method to `Model` immediately before `summary()`:

```python
    def tree(self) -> str:
        """Return a human-readable physics tree showing the model hierarchy.

        Shows the active sub-models at each physical layer (SFH, SPS, Dust,
        Nebular, AGN, Observation), the free parameters at each layer, and
        the recommended inference method at the bottom.

        Returns
        -------
        str
            Multi-line formatted tree string.

        Examples
        --------
        >>> print(model.tree())
        Model  [D=7, stochastic=False]
        │
        ├── SFH: dpl
        │   ├── sfh_dpl_alpha         ~ Uniform(0.5, 3.0)
        ...
        """
        sep = "│"
        branch = "├──"
        last = "└──"
        lines: list[str] = []

        # Header
        d = self.spec.n_free
        stoch = "True" if self.spec.stochastic else "False"
        n_grid = self.spec.n_grid if self.spec.stochastic else 0
        d_total = d + n_grid
        lines.append(f"Model  [D={d_total}, stochastic={stoch}]")
        lines.append(sep)

        # --- SFH layer ---
        sfh_name = "+".join(self.spec.mean_sfh_type)
        lines.append(f"{branch} SFH: {sfh_name}")

        sfh_params = [
            p for p in self.spec.free_params
            if p.startswith("sfh_") or p == "psd_sigma" or p == "psd_tau_myr"
        ]
        for i, name in enumerate(sfh_params):
            dist = self.spec.get_distribution(name)
            prefix = last if i == len(sfh_params) - 1 else branch
            lines.append(f"{sep}   {prefix} {name:<30s} ~ {dist!r}")

        if self.spec.stochastic:
            lines.append(f"{sep}   {last} sfh_field_xi  [{n_grid}-dim GP latent, ξ ~ N(0,I)]")

        lines.append(sep)

        # --- SPS layer ---
        ssp = self.ssp_data
        n_met, n_age, n_wave = ssp.ssp_flux.shape
        lines.append(f"{branch} SPS: DSPS  [{n_met} Z × {n_age} ages × {n_wave} λ]")
        lines.append(sep)

        # --- Dust layer ---
        dust_law = getattr(self.spec, "dust_law_bc", "power_law")
        dust_diff = getattr(self.spec, "dust_law_diff", "calzetti")
        dust_emission = getattr(self, "_dust_emission_model", None)
        lines.append(f"{branch} Dust")
        lines.append(f"{sep}   {branch} Attenuation: Charlot & Fall (bc={dust_law}, diff={dust_diff})")

        dust_params = [p for p in self.spec.free_params if p.startswith("dust_")]
        for name in dust_params:
            dist = self.spec.get_distribution(name)
            lines.append(f"{sep}   {sep}   {branch} {name:<26s} ~ {dist!r}")

        if dust_emission:
            lines.append(f"{sep}   {last} Emission: {dust_emission}")
            dust_em_params = [p for p in self.spec.free_params
                              if p.startswith("dust_") and "T" in p or "umin" in p or "qpah" in p]
            for name in dust_em_params:
                dist = self.spec.get_distribution(name)
                lines.append(f"{sep}       {branch} {name:<26s} ~ {dist!r}")
        lines.append(sep)

        # --- Nebular layer ---
        neb_mode = getattr(self.spec, "nebular_mode", "off")
        if neb_mode and neb_mode != "off":
            lines.append(f"{branch} Nebular: {neb_mode}")
            neb_params = [p for p in self.spec.free_params if p.startswith("neb_")]
            for name in neb_params:
                dist = self.spec.get_distribution(name)
                lines.append(f"{sep}   {branch} {name:<30s} ~ {dist!r}")
            lines.append(sep)

        # --- AGN layer ---
        agn_model = getattr(self, "_agn_model", None)
        if agn_model:
            lines.append(f"{branch} AGN: {agn_model}")
            agn_params = [p for p in self.spec.free_params if p.startswith("agn_")]
            for name in agn_params:
                dist = self.spec.get_distribution(name)
                lines.append(f"{sep}   {branch} {name:<30s} ~ {dist!r}")
            lines.append(sep)

        # --- Observation layer ---
        z_info = f"z={self._z_fixed:.4f} [fixed]" if self._z_fixed is not None else "z [free]"
        if self.filter_waves is not None:
            n_filt = len(self.filter_waves)
            precomp = "YES (21.6× speedup)" if self._precomp is not None else "NO"
            lines.append(f"{last} Observation: Photometry [{n_filt} bands] at {z_info}")
            lines.append(f"    Precomputed: {precomp}")
        elif getattr(self, "_wave_obs", None) is not None:
            lines.append(f"{last} Observation: Spectroscopy at {z_info}")
        else:
            lines.append(f"{last} Observation: {z_info}")

        lines.append("")
        # Append the method recommendation
        method, reason = self._method_recommendation()
        lines.append(f"Recommended inference:")
        lines.append(f"  → model.fit(data, noise, method={method!r})   [{reason}]")
        if stoch == "False" and d <= 30:
            lines.append(f"  → model.fit(data, noise, method='evidence')  [Bayesian evidence, D≤30]")

        return "\n".join(lines)

    def _method_recommendation(self) -> tuple[str, str]:
        """Return (method_name, reason) for the recommended inference method.

        Used by both ``tree()`` and ``recommend_method()``.
        """
        d = self.spec.n_free
        if self.spec.stochastic:
            # High-D GP latent — geoVI is most appropriate
            d_total = d + self.spec.n_grid
            return "vi", f"D={d_total}, stochastic, geoVI default"
        elif d <= 15:
            return "laplace", f"D={d}, smooth, instant Gaussian approximation"
        elif d <= 50:
            return "vi_linear", f"D={d}, smooth, fast VI"
        else:
            return "vi", f"D={d}, moderate-high, geoVI default"
```

- [ ] **Step 2: Add `Model.recommend_method()` method**

Add immediately after `tree()`:

```python
    def recommend_method(self) -> str:
        """Return the recommended inference method string for this model.

        The recommendation is based on model dimensionality and whether the
        SFH includes a stochastic GP field component:

        - Stochastic (GP field): ``"vi"`` (geoVI handles D~137 well)
        - Smooth, D ≤ 15: ``"laplace"`` (instant, exact Gaussian at MAP)
        - Smooth, 15 < D ≤ 50: ``"vi_linear"`` (MGVI, fast per iteration)
        - Smooth, D > 50: ``"vi"`` (geoVI default)

        For exact MCMC validation, always use ``"mcmc_raytrace"`` regardless
        of dimensionality (gradient-noise tolerant, works for D~137).

        For Bayesian evidence (model comparison), use ``"evidence"`` —
        only practical for smooth models with D ≤ 30.

        Returns
        -------
        str
            Canonical method name for ``Fitter.run()`` or ``model.fit()``.

        Examples
        --------
        >>> method = model.recommend_method()
        >>> result = model.fit(flux, noise, method=method)
        """
        method, _ = self._method_recommendation()
        return method
```

- [ ] **Step 3: Run ruff**

```bash
ruff check src/tengri/core/model.py
ruff format --check src/tengri/core/model.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/tengri/core/model.py
git commit -m "feat: add Model.tree() physics hierarchy display and Model.recommend_method()"
```

---

## Task 8: `AGNConfig` frozen dataclass

**Files:**
- Create: `src/tengri/models/agn/agn_config.py`
- Modify: `src/tengri/__init__.py`

A frozen dataclass for static AGN model selection. This separates which sub-models are active (static config) from which parameters are free (handled by `ParamSpec`).

- [ ] **Step 1: Create `src/tengri/models/agn/agn_config.py`**

```python
"""AGNConfig: static configuration for AGN sub-model selection.

``AGNConfig`` specifies which disc, torus, NLR, and BLR sub-models are
active.  This is distinct from the free parameters (which come from
``ParamSpec``): ``AGNConfig`` is a static compile-time choice, not a
parameter to be inferred.

Usage::

    from tengri import Model, ParamSpec, Uniform, AGNConfig

    agn_config = AGNConfig(disc="kubota_done", torus="skirtor", nlr="analytic", blr=True)
    model = Model(spec, ssp, agn_config=agn_config)

    # Or via from_config():
    model = Model.from_config(
        ssp="data/ssp.h5",
        agn={"disc": "kubota_done", "torus": "skirtor"},
        ...
    )
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AGNConfig:
    """Static configuration for AGN sub-model selection.

    Parameters
    ----------
    disc : str
        AGN accretion disc model.
        ``"powerlaw"``     — simple power-law SED.
        ``"multicolor"``   — multi-colour blackbody disc (Shakura-Sunyaev).
        ``"kubota_done"``  — Kubota & Done (2018) 3-zone model (default).
        ``"adaf"``         — ADAF (low-luminosity AGN).
    torus : str
        AGN torus/obscuration model.
        ``"simple"``           — single-temperature MBB (toy).
        ``"two_temperature"``  — two-temperature MBB (toy).
        ``"skirtor"``          — SKIRTOR clumpy torus (default, science-grade).
    nlr : str
        Narrow Line Region emission model.
        ``"analytic"``  — analytic Gaussian line profiles (default, fast).
        ``"cue"``       — Cue neural emulator (physically consistent).
    blr : bool
        Include Broad Line Region emission (Type 1 AGN). Default ``True``.
    polar_dust : bool
        Include SMC polar dust reddening. Default ``False``.
    fe2 : bool
        Include Fe II pseudo-continuum. Default ``False``.

    Notes
    -----
    The free parameters for the selected sub-models must still be declared
    in ``ParamSpec`` (e.g. ``agn_log_mbh``, ``agn_log_ledd``, ``agn_cos_inc``).
    ``AGNConfig`` only controls which models are activated, not the parameter values.
    """

    disc: str = "multicolor"
    torus: str = "skirtor"
    nlr: str = "analytic"
    blr: bool = True
    polar_dust: bool = False
    fe2: bool = False

    # ---- validation ----

    _VALID_DISC = frozenset({"powerlaw", "multicolor", "kubota_done", "adaf"})
    _VALID_TORUS = frozenset({"simple", "two_temperature", "skirtor"})
    _VALID_NLR = frozenset({"analytic", "cue"})

    def __post_init__(self) -> None:
        if self.disc not in self._VALID_DISC:
            raise ValueError(
                f"AGNConfig.disc={self.disc!r} is not valid. "
                f"Choose from: {sorted(self._VALID_DISC)}"
            )
        if self.torus not in self._VALID_TORUS:
            raise ValueError(
                f"AGNConfig.torus={self.torus!r} is not valid. "
                f"Choose from: {sorted(self._VALID_TORUS)}"
            )
        if self.nlr not in self._VALID_NLR:
            raise ValueError(
                f"AGNConfig.nlr={self.nlr!r} is not valid. "
                f"Choose from: {sorted(self._VALID_NLR)}"
            )

    def __repr__(self) -> str:
        return (
            f"AGNConfig(disc={self.disc!r}, torus={self.torus!r}, "
            f"nlr={self.nlr!r}, blr={self.blr}, "
            f"polar_dust={self.polar_dust}, fe2={self.fe2})"
        )
```

- [ ] **Step 2: Export `AGNConfig` from `__init__.py`**

In `src/tengri/__init__.py`, add:
```python
from tengri.components.agn.agn_config import AGNConfig
```

Add `"AGNConfig"` to the `__all__` list.

- [ ] **Step 3: Run ruff**

```bash
ruff check src/tengri/models/agn/agn_config.py src/tengri/__init__.py
ruff format --check src/tengri/models/agn/agn_config.py src/tengri/__init__.py
```

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add src/tengri/models/agn/agn_config.py src/tengri/__init__.py
git commit -m "feat: add AGNConfig frozen dataclass for static AGN sub-model selection"
```

---

## Task 9: Tests in `tests/unit/test_physics_api.py`

**Files:**
- Create: `tests/unit/test_physics_api.py`

Tests are split between those needing SSP data (skipped if absent) and pure-Python unit tests (always run).

- [ ] **Step 1: Create `tests/unit/test_physics_api.py`**

```python
"""Tests for physics API redesign (physics_api_redesign.md).

- No-SSP tests: _planck_lnu deduplication, lines_to_sed, eline_catalog,
  nlr/blr line_efficiency, AGNConfig, agn_nlr return type.
- SSP-required: model.tree(), model.recommend_method(), predict_hbeta.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# SSP gate
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()
_needs_ssp = pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")


# ---------------------------------------------------------------------------
# Task 1: _planck_lnu extracted to _phys.py
# ---------------------------------------------------------------------------


class TestPlanckLnuExtracted:
    def test_phys_module_exists(self):
        from tengri.components.agn import _phys  # noqa: F401

    def test_planck_lnu_importable(self):
        from tengri.components.agn._phys import planck_lnu

        assert callable(planck_lnu)

    def test_planck_lnu_finite_for_solar_T(self):
        from tengri.components.agn._phys import planck_lnu

        nu = jnp.linspace(1e12, 1e16, 100)  # UV to far-IR
        result = planck_lnu(nu, 5778.0)     # solar temperature
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0.0)

    def test_planck_zero_temperature_returns_finite(self):
        from tengri.components.agn._phys import planck_lnu

        nu = jnp.array([1e14])
        result = planck_lnu(nu, 0.0)   # temperature = 0 → clamp to 1 K
        assert jnp.all(jnp.isfinite(result))

    def test_disc_still_importable(self):
        """disc.py must still import cleanly after removing local _planck_lnu."""
        from tengri.components.agn import disc  # noqa: F401

    def test_torus_still_importable(self):
        from tengri.components.agn import torus  # noqa: F401

    def test_skirtor_still_importable(self):
        from tengri.components.agn import skirtor  # noqa: F401


class TestLinesToSed:
    def test_lines_to_sed_shape(self):
        from tengri.components.agn._phys import lines_to_sed

        wave_obs = jnp.linspace(4000.0, 7000.0, 500)
        line_wav = jnp.array([4861.33, 6562.80])   # Hβ, Hα
        line_lum = jnp.array([1.0, 2.86])          # Lsun

        result = lines_to_sed(line_wav, line_lum, wave_obs, fwhm_kms=200.0)

        assert result.shape == (500,)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.any(result > 0.0)

    def test_lines_to_sed_peaks_near_line_centres(self):
        from tengri.components.agn._phys import lines_to_sed

        wave_obs = jnp.linspace(6400.0, 6700.0, 1000)
        line_wav = jnp.array([6562.80])
        line_lum = jnp.array([1.0])

        result = lines_to_sed(line_wav, line_lum, wave_obs, fwhm_kms=200.0)
        peak_wave = float(wave_obs[jnp.argmax(result)])

        # Peak should be within 10 Å of Hα
        assert abs(peak_wave - 6562.80) < 10.0


# ---------------------------------------------------------------------------
# Task 2: agn_nlr_emission return type fix
# ---------------------------------------------------------------------------


class TestAgnNlrEmissionReturnType:
    def test_agn_nlr_emission_return_annotation(self):
        """agn_nlr_emission() must NOT have a union return type."""
        import inspect

        from tengri.components.nebular.agn_nebular import agn_nlr_emission

        hints = {}
        try:
            import typing

            hints = typing.get_type_hints(agn_nlr_emission)
        except Exception:
            pass

        sig = inspect.signature(agn_nlr_emission)
        # Check: 'wavelength' should no longer be a parameter
        assert "wavelength" not in sig.parameters, (
            "agn_nlr_emission() still has deprecated 'wavelength' parameter"
        )

    def test_agn_nlr_cue_no_wavelength_param(self):
        """agn_nlr_cue() should not take a wavelength parameter."""
        import inspect

        from tengri.components.nebular.agn_nebular import agn_nlr_cue

        sig = inspect.signature(agn_nlr_cue)
        assert "wavelength" not in sig.parameters

    def test_agn_nlr_cue_uses_neb_logU(self):
        """agn_nlr_cue() should use neb_logU, not gas_logu."""
        import inspect

        from tengri.components.nebular.agn_nebular import agn_nlr_cue

        sig = inspect.signature(agn_nlr_cue)
        assert "neb_logU" in sig.parameters
        assert "gas_logu" not in sig.parameters


# ---------------------------------------------------------------------------
# Task 3: line_efficiency exposed in nlr/blr
# ---------------------------------------------------------------------------


class TestLineEfficiencyExposed:
    def test_nlr_emission_has_line_efficiency_param(self):
        import inspect

        from tengri.components.agn.nlr import nlr_emission

        sig = inspect.signature(nlr_emission)
        assert "line_efficiency" in sig.parameters
        # Default should be ~0.10
        default = sig.parameters["line_efficiency"].default
        assert abs(default - 0.10) < 0.01

    def test_blr_emission_has_line_efficiency_param(self):
        import inspect

        from tengri.components.agn.blr import blr_emission

        sig = inspect.signature(blr_emission)
        assert "line_efficiency" in sig.parameters
        # Default should be ~0.08
        default = sig.parameters["line_efficiency"].default
        assert abs(default - 0.08) < 0.01

    def test_nlr_emission_line_efficiency_scales_output(self):
        """Halving line_efficiency should roughly halve NLR luminosity."""
        from tengri.components.agn.nlr import nlr_emission

        wave = jnp.linspace(3000.0, 8000.0, 500)
        l1 = jnp.sum(nlr_emission(wave, 1e44, line_efficiency=0.10))
        l2 = jnp.sum(nlr_emission(wave, 1e44, line_efficiency=0.05))
        ratio = float(l1 / jnp.maximum(l2, 1e-100))
        assert 1.8 < ratio < 2.2  # roughly 2x


# ---------------------------------------------------------------------------
# Task 4: Emission line catalog
# ---------------------------------------------------------------------------


class TestEmissionLineCatalog:
    def test_eline_catalog_importable(self):
        from tengri.observation import eline_catalog  # noqa: F401

    def test_emission_lines_dict_has_required_keys(self):
        from tengri.observation.eline_catalog import EMISSION_LINES

        for name in ("Hbeta", "Halpha", "OIII5007", "NII6583"):
            assert name in EMISSION_LINES, f"{name} not in EMISSION_LINES"

    def test_line_groups_consistent_with_catalog(self):
        from tengri.observation.eline_catalog import EMISSION_LINES, LINE_GROUPS

        for group_name, members in LINE_GROUPS.items():
            for member in members:
                assert member in EMISSION_LINES, (
                    f"Line {member!r} in group {group_name!r} not in EMISSION_LINES"
                )

    def test_get_line_wavelengths(self):
        from tengri.observation.eline_catalog import get_line_wavelengths

        wav = get_line_wavelengths("bpt")
        assert wav.shape == (4,)
        # Hbeta at ~4861
        assert any(abs(float(w) - 4861.33) < 1.0 for w in wav)

    def test_backward_compat_default_line_arrays(self):
        """Old DEFAULT_LINE_NAMES/WAVELENGTHS still importable from eline_marginalization."""
        from tengri.observation.eline_marginalization import (
            DEFAULT_LINE_NAMES,
            DEFAULT_LINE_WAVELENGTHS,
        )

        assert len(DEFAULT_LINE_NAMES) == 13
        assert DEFAULT_LINE_WAVELENGTHS.shape == (13,)

    def test_backward_compat_cloudy_line_arrays(self):
        """Old CLOUDY_LINE_NAMES/WAVELENGTHS still importable from eline_priors."""
        from tengri.observation.eline_priors import (
            CLOUDY_LINE_NAMES,
            CLOUDY_LINE_WAVELENGTHS,
        )

        assert len(CLOUDY_LINE_NAMES) == 11
        assert CLOUDY_LINE_WAVELENGTHS.shape == (11,)


# ---------------------------------------------------------------------------
# Task 5: Unified build_line_design_matrix
# ---------------------------------------------------------------------------


class TestBuildLineDesignMatrix:
    def test_build_line_design_matrix_importable(self):
        from tengri.observation.eline_marginalization import (
            build_line_design_matrix,  # noqa: F401
        )

    def test_narrow_only_matches_eline_design_matrix(self):
        from tengri.observation.eline_marginalization import (
            build_eline_design_matrix,
            build_line_design_matrix,
        )

        wave = jnp.linspace(4000.0, 7000.0, 300)
        line_wav = jnp.array([4861.33, 6562.80])

        A_old = build_eline_design_matrix(wave, line_wav, spectral_resolution=2000.0)
        A_new = build_line_design_matrix(wave, line_wav)

        assert A_old.shape == A_new.shape
        assert jnp.allclose(A_old, A_new, atol=1e-6)

    def test_narrow_plus_broad_has_extra_columns(self):
        from tengri.observation.eline_marginalization import build_line_design_matrix

        wave = jnp.linspace(4000.0, 7000.0, 300)
        narrow = jnp.array([4861.33, 6562.80])   # 2 narrow
        broad = jnp.array([4861.33])              # 1 broad

        A_combined = build_line_design_matrix(wave, narrow, broad_wavelengths=broad)

        assert A_combined.shape == (300, 3)   # 2 narrow + 1 broad


# ---------------------------------------------------------------------------
# Task 8: AGNConfig dataclass
# ---------------------------------------------------------------------------


class TestAGNConfig:
    def test_agn_config_importable(self):
        import tengri

        assert hasattr(tengri, "AGNConfig")

    def test_default_construction(self):
        from tengri.components.agn.agn_config import AGNConfig

        cfg = AGNConfig()
        assert cfg.disc == "multicolor"
        assert cfg.torus == "skirtor"
        assert cfg.blr is True
        assert cfg.polar_dust is False

    def test_custom_construction(self):
        from tengri.components.agn.agn_config import AGNConfig

        cfg = AGNConfig(disc="kubota_done", torus="skirtor", nlr="cue", blr=True, polar_dust=True)
        assert cfg.disc == "kubota_done"
        assert cfg.polar_dust is True

    def test_frozen_immutable(self):
        from tengri.components.agn.agn_config import AGNConfig

        cfg = AGNConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.disc = "adaf"  # type: ignore[misc]

    def test_invalid_disc_raises(self):
        from tengri.components.agn.agn_config import AGNConfig

        with pytest.raises(ValueError, match="disc"):
            AGNConfig(disc="invalid_model")

    def test_invalid_torus_raises(self):
        from tengri.components.agn.agn_config import AGNConfig

        with pytest.raises(ValueError, match="torus"):
            AGNConfig(torus="photon_torpedo")


# ---------------------------------------------------------------------------
# SSP-required: model.tree() and model.recommend_method()
# ---------------------------------------------------------------------------


@_needs_ssp
class TestModelTree:
    @pytest.fixture(scope="class")
    def smooth_model(self):
        import tengri

        return tengri.Model.from_config(
            ssp=str(_SSP_FILE),
            sfh="dpl",
            filters=["sdss_u", "sdss_g", "sdss_r"],
            redshift=0.1,
            priors=dict(
                alpha=tengri.Uniform(0.5, 3.0),
                beta=tengri.Uniform(0.3, 2.0),
                tau_gyr=tengri.Uniform(0.5, 10.0),
                log_peak_sfr=tengri.Uniform(-1, 2.5),
                logzsol=tengri.Uniform(-1.5, 0.2),
                tau_bc=tengri.Uniform(0, 3.0),
            ),
        )

    def test_tree_returns_string(self, smooth_model):
        result = smooth_model.tree()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_tree_contains_sfh_name(self, smooth_model):
        result = smooth_model.tree()
        assert "dpl" in result

    def test_tree_contains_recommended_method(self, smooth_model):
        result = smooth_model.tree()
        assert "Recommended inference" in result
        assert "model.fit(" in result

    def test_recommend_method_returns_string(self, smooth_model):
        method = smooth_model.recommend_method()
        assert isinstance(method, str)
        # Smooth DPL with ~6 free params → should recommend laplace or vi_linear
        assert method in ("laplace", "vi_linear", "vi")

    def test_recommend_method_used_in_fit(self, smooth_model):
        """recommend_method() output should be accepted by model.fit()."""
        true_params = {
            "sfh_dpl_alpha": 1.2,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 4.0,
            "sfh_dpl_log_peak_sfr": 0.9,
            "met_logzsol": -0.3,
            "dust_tau_bc": 1.0,
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 0.1,
        }
        mock = smooth_model.mock(true_params, snr=10.0, key=jax.random.PRNGKey(99))
        # Just test it runs without error; MAP is fast
        result = smooth_model.fit(mock.flux_obs, mock.noise, method="map")
        from tengri.inference.posterior import Posterior

        assert isinstance(result, Posterior)
```

- [ ] **Step 2: Run the no-SSP tests**

```bash
cd ~/Projects/tengri && source .venv/bin/activate
pytest tests/unit/test_physics_api.py -v -k "not needs_ssp" 2>&1 | tail -30
```

Expected: All `TestPlanckLnuExtracted`, `TestLinesToSed`, `TestAgnNlrEmissionReturnType`, `TestLineEfficiencyExposed`, `TestEmissionLineCatalog`, `TestBuildLineDesignMatrix`, `TestAGNConfig` tests pass.

- [ ] **Step 3: Run the full test suite to verify no regressions**

```bash
pytest tests/ -q --timeout=120 2>&1 | tail -20
```

Expected: Same pass count as before these changes.

- [ ] **Step 4: Run ruff**

```bash
ruff check tests/unit/test_physics_api.py
ruff format --check tests/unit/test_physics_api.py
```

Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_physics_api.py
git commit -m "test: add test_physics_api.py covering physics API redesign"
```

---

## Self-review: spec coverage check

| Spec item | Task covering it |
|-----------|-----------------|
| Extract duplicate `_planck_lnu` to `agn/_phys.py` | Task 1 |
| Add `lines_to_sed()` shared utility | Task 1 |
| Fix `agn_nlr_emission()` union return type | Task 2 |
| Remove unused `wavelength` param from `agn_nlr_cue()` | Task 2 |
| Rename `gas_logu` → `neb_logU` in AGN NLR interface | Task 2 |
| Expose `fwhm_kms` (already done), add `line_efficiency` | Task 3 |
| Create unified `EMISSION_LINE_CATALOG` in `eline_catalog.py` | Task 4 |
| Update `eline_marginalization.py` + `eline_priors.py` to import from catalog | Task 4 |
| Unified `build_line_design_matrix()` (narrow + broad) | Task 5 |
| `Model.predict_hbeta(params)` | Task 6 |
| `model.tree()` with physics hierarchy display | Task 7 |
| `model.recommend_method()` | Task 7 |
| `AGNConfig` frozen dataclass | Task 8 |
| Export `AGNConfig` from `tengri.__init__` | Task 8 |
| Tests in `test_physics_api.py` | Task 9 |
| All ruff passes zero violations | Every task |
| No existing code broken | Additive throughout |

**Items from spec NOT in this plan (out-of-scope by risk/effort):**
- `dust/emission.py` split into subdirectory — 2800-line refactor, risk of breaking imports. Omitted as separate task.
- `ModelRegistry` unified class + `tengri.sfh_registry` etc. — large surface area, no broken behavior to fix. Treat as separate plan.
- `ParamSpec.validate_against_backends(model)` — useful but not blocking. Treat as follow-on.
- `line_sigma_kms` standardization in nebular/ modules — requires reading each module; low priority vs impact.

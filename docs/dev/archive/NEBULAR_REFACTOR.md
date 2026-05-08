# Nebular Emission Refactor Plan

> **Scope:** The `src/tengri/models/nebular/` module as it exists today — six backends, three helper modules,
> ~3,100 lines spread across nine files. This document records what was built, what went wrong, what should have
> been done first, and the concrete steps to fix it.

---

## 1. What exists today

| File | Backends / utilities | Lines | Continuum? |
|------|----------------------|-------|------------|
| `baked_in.py` | `BakedInBackend` | ~50 | SSP already includes it (no separate prediction) |
| `cloudy_grid.py` | `CloudyGridBackend` | ~420 | **Yes** — lines + continuum from CLOUDY grid |
| `cloudy_cb19.py` | `CB19Backend` | ~520 | **No** — stub returns `([5000 Å], [0.0])` |
| `mappings_photo.py` | `MappingsPhotoStellarBackend`, `MappingsPhotoAGNBackend` | ~680 | **No** — no method at all |
| `cue.py` | `CueBackend` | ~800 | **Yes** — lines + continuum from neural emulator |
| `shock.py` | `shock_line_ratios`, `shock_emission_sed` | ~350 | **No** — stub pattern |
| `ionizing_spectrum.py` | `fit_ionizing_spectrum`, `interpolate_ionizing_params` | ~230 | utility only |
| `dig.py` | `mix_dig_emission` | ~80 | passthrough |
| `agn_nebular.py` | `agn_ionspec_from_alpha_pl`, `agn_nlr_emission` | ~240 | via Cue |

Nine files. No abstract base class. No shared constants module. No shared utility module.

---

## 2. What should have been done first (pre-implementation checklist)

These six steps — which mirror Step 0A–0F in `docs/dev/REFACTOR.md` — should have been completed
**before writing a single line of backend code**.

### Step 0A — Write an API contract script

A one-page Python file that exercises every backend with identical call signatures and asserts the
output shapes and units:

```python
# nebular_api_contract.py  (should have existed before any backend was written)
backend = SomeBackend(grid_path="...")
wave, lines = backend.line_luminosities(logU, log_z_gas, log_age, log_qh, ssps)
wave_c, cont  = backend.continuum_luminosity(logU, log_z_gas, log_age, log_qh, ssps)
# Both return arrays in Lsun/Hz, vacuum wavelengths in Ångström
```

Writing this script first would have forced:
- a **unified method-name decision** (`line_luminosities` vs `predict_nebular_line_luminosities`
  vs `predict_nebular_line_fluxes`)
- a **unified return-value contract** (luminosities in Lsun/Hz, vacuum Å)
- a **decision about what to do when a backend has no continuum** (raise? return zeros? fall back?)

### Step 0B — Write a unit contract table

All nebular-relevant physical quantities, their units, and which convention each caller uses:

| Quantity | Symbol | Unit (tengri internal) | Notes |
|----------|--------|------------------------|-------|
| Ionizing photon rate | Q_H | photons s⁻¹ | log10 stored |
| Wavelength | λ | Å (vacuum) | NOT air |
| Emission line luminosity | L_line | L☉ | integrated over profile |
| Continuum specific luminosity | L_ν | L☉ Hz⁻¹ | per Hz, not per Å |
| Gas metallicity (user-facing) | neb_logZ_gas | log10(Z/Z☉) | param_map adds LOG10_ZSUN |
| Gas metallicity (DSPS) | log_z_abs | log10(Z) absolute | |
| Gas metallicity (CB19) | log(O/H) | CLOUDY c17.01 solar scale | offset –1.222 from absolute |
| Gas metallicity (Mappings) | ζ_O | solar-relative O abundance | |
| Gas metallicity (Cue) | log10(Z/Z☉) | same as user-facing | direct |
| Ionization parameter | logU | dimensionless | log10(Q_H / 4πr²cn_H) |
| Escape fraction (ionizing) | neb_fesc | [0, 1] | attenuates Q_H |
| Escape fraction (Ly-α only) | neb_fesc_lya | [0, 1] | separate from neb_fesc |

Without this table, each backend invented its own metallicity axis. Four different conventions are
now in production simultaneously with no single conversion utility.

### Step 0C — Define the module boundary tree

```
nebular/
  _constants.py        # ONE place for H_PLANCK, C_CGS, LSUN_ERG, LYMAN_LIMIT, LOG10_ZSUN
  _shared.py           # compute_qh(), interp_index_weight(), place_line_profile()
  _protocol.py         # NebularBackend Protocol (ABC alternative, JAX-compatible)
  backends/
    baked_in.py
    cloudy_grid.py
    cloudy_cb19.py
    cue.py
    mappings_photo.py
    mappings_shock.py
  agn_nebular.py       # AGN NLR dispatcher (uses CueBackend or MappingsPhotoAGN)
  dig.py               # DIG mixing (depends on _protocol, not concrete backends)
  ionizing_spectrum.py # Ionizing SED parameterization for Cue
  __init__.py          # Public exports only
```

Having this tree before writing code would have prevented:
- physical constants being defined independently in five files
- `compute_qh()` being re-implemented three times
- line profile code being duplicated four times

### Step 0D — Decide settings vs. parameters

The central JAX JIT constraint: **anything that dispatches code paths must be a Python-level
setting, not a JAX traced parameter.**

| Item | Category | Why |
|------|----------|-----|
| Which backend to use | Setting (init-time) | Determines JAX function to call |
| logU | Parameter (traced) | Gradient flows through it |
| neb_logZ_gas | Parameter (traced) | Gradient flows through it |
| neb_fesc | Parameter (traced) | Linear scale on Q_H |
| log_nH | Setting or Parameter | Should be documented; currently ad-hoc per backend |
| DIG fraction | Parameter (traced) | Mixed with linear weight |
| Ionizing spectrum shape (Cue α_EUV) | Parameter (traced) | Differentiable EUV parameterization |

This decision should have been made in writing before any `__init__` was coded.

### Step 0E — Define the ionizing spectrum contract per backend

Before writing a single backend, document what ionizing radiation source each one assumes and what
warning the user must receive. (See Section 3 for the warnings.)

### Step 0F — Write finite-difference gradient tests first

Every backend that accepts traced parameters must have a gradient test before any production use.
The FD test for line luminosities ∂L/∂logU and ∂L/∂log_z_gas is ~10 lines of pytest and catches
unit errors, shape errors, and NaN propagation automatically.

No such tests existed when the backends were written.

---

## 3. Ionizing spectrum warnings (per backend)

> **This is the most important section for physical correctness.**
> The ionizing photon spectrum — the radiation field that excites nebular emission —
> differs between backends. Users must understand this before interpreting results.

### 3.1 `CloudyGridBackend`

**Ionizing source:** BPASS v2.1 binary stellar populations (Stanway & Eldridge 2018) at fixed
metallicity grid points. Q_H is computed from the *tengri* DSPS SSPs via `compute_qh()`, then the
grid is renormalized to that Q_H.

**Warning to emit at backend initialization:**

```
⚠ CloudyGridBackend: Q_H is computed from your DSPS SSPs (correct for
  stellar mass accounting), but the *ionizing spectral shape* used to run
  the CLOUDY grid was BPASS v2.1 binary stars. If your SSP has a
  significantly harder/softer ionizing SED than BPASS (e.g., single-star
  SSPs, stripped-star prescriptions, very young/old populations), the
  predicted line ratios will be biased. Use CueBackend or CB19Backend with
  an explicit ionizing SED shape if this matters for your science case.
```

**Implementation:** Emit via `warnings.warn(..., IonizingSpectrumMismatchWarning)` inside
`__init__` when a non-BPASS SSP path is detected, or unconditionally at first call with a one-shot
flag.

### 3.2 `CB19Backend`

**Ionizing source:** BPASS v2.1 binary stars (Chevallard & Charlot 2016 / Byler+2019 updates).
Same ionizing SED assumption as CloudyGridBackend. Additionally, the grid uses fixed ionization
parameter geometry (plane-parallel Strömgren sphere).

**Warning:**

```
⚠ CB19Backend: The CLOUDY c17.01 grids were computed with BPASS v2.1
  binary stars as the ionizing source. The 6D parameter space (O/H, age,
  logU, nH, C/O, ΔN/O) does NOT include variation in ionizing SED hardness.
  For AGN-ionized or shock-excited regions, use MappingsPhotoAGNBackend or
  ShockEmission instead.
```

### 3.3 `MappingsPhotoStellarBackend`

**Ionizing source:** Starburst99 + BPASS grids (Allen+2008; MAPPINGS V; Byler+2018).
**This is NOT the same SSP as your DSPS stellar continuum.**

This is the single most dangerous inconsistency in the nebular module. The stellar continuum
comes from DSPS (which wraps FSPS/BaSeL/MILES), but the ionizing source that drives line
predictions is a completely separate Starburst99/BPASS grid baked into MAPPINGS V. For a galaxy
at 10 Myr with a hard ionizing SED, the DSPS model and the MAPPINGS ionizing model may differ
significantly.

**Error (CRITICAL — raises by default, opt-out required):**

```python
# Default: raises IonizingSpectrumInconsistencyError
backend = MappingsPhotoStellarBackend(...)

# Opt-in to warning only
backend = MappingsPhotoStellarBackend(..., ionizing_source_warning="warn")

# Fully suppress (only if you have verified this is acceptable)
backend = MappingsPhotoStellarBackend(..., ionizing_source_warning="suppress")
```

**Implementation:** `IonizingSpectrumInconsistencyError` raised in `__init__` by default.
Exported from `tengri.components.nebular`.

### 3.4 `MappingsPhotoAGNBackend`

**Ionizing source:** Power-law AGN continuum parameterized by MBH, Eddington ratio, and spectral
slope — consistent with the AGN disc model in `models/agn/`. Q_H is passed in from the caller
(estimated from accretion luminosity), not from SSPs.

**Warning:**

```
⚠ MappingsPhotoAGNBackend: Q_H must be supplied by the AGN disc model —
  it is not self-consistently derived from SSPs. Ensure you are passing
  Q_H from `_log_qh_from_lacc()` or an equivalent accretion model.
  The ionizing shape is a power law; for a composite starburst+AGN ionized
  region, use a mixed-source model.
```

### 3.5 `CueBackend`

**Ionizing source:** Explicitly parameterized by 7 piecewise power-law coefficients below 912 Å
(`alpha_EUV_1` through `alpha_EUV_7`). These parameters should be computed from the *actual*
DSPS SSP via `fit_ionizing_spectrum()` and `interpolate_ionizing_params()` in `ionizing_spectrum.py`.

This is the most self-consistent backend for stellar nebular emission.

**Warning (informational):**

```
ℹ CueBackend: The ionizing SED shape is parameterized by 7 piecewise power-law
  slopes computed from your DSPS SSPs via fit_ionizing_spectrum(). This is the
  most self-consistent stellar nebular backend. If you are using default
  ionizing params without calling precompute_ionizing_params_table(), the
  shapes default to a solar-metallicity BPASS spectrum — verify this is
  appropriate for your parameter range.
```

### 3.6 `BakedInBackend`

**Ionizing source:** Whatever was assumed when the SSP HDF5 was generated (typically FSPS
nebular at logU = −3, fixed escape fraction).

**Warning:**

```
⚠ BakedInBackend: Nebular emission is baked into your SSP file at a FIXED
  logU and FIXED escape fraction determined when the SSP grid was generated
  (depends on the SSP file — commonly logU = −3 but not guaranteed). The
  ionization parameter and escape fraction are NOT free parameters. Check
  your SSP file's nebular assumptions. Switch to CloudyGridBackend or
  CueBackend to vary nebular properties.
```

### 3.7 `ShockEmission`

**Ionizing source:** MAPPINGS III/V shock models (Allen+2008; 3MdBs Alarie & Morisset 2019).
The "ionizing source" is the post-shock plasma itself, not stellar photons.

**Warning:**

```
ℹ ShockEmission: Ionization is driven by shock heating, not stellar photons.
  Q_H from SSPs is not used. Ensure your target galaxy has evidence for
  shock excitation (e.g., BPT position above the Kewley+2001 maximum
  starburst line) before adding this component.
```

### Summary table

| Backend | Ionizing source | Self-consistent with DSPS? | Warning severity |
|---------|-----------------|---------------------------|-----------------|
| `BakedInBackend` | SSP-baked (FSPS logU=−3) | Partially | ⚠ WARN |
| `CloudyGridBackend` | BPASS v2.1 shape + DSPS Q_H | Mostly | ⚠ INFO |
| `CB19Backend` | BPASS v2.1 | No shape variation | ⚠ INFO |
| `MappingsPhotoStellarBackend` | SB99/BPASS (separate grid) | **NO** | 🔴 CRITICAL |
| `MappingsPhotoAGNBackend` | AGN power law + DSPS Q_H | For AGN: yes | ⚠ INFO |
| `CueBackend` | DSPS SSP via fit_ionizing_spectrum | **YES** | ℹ INFO |
| `ShockEmission` | Post-shock plasma | N/A | ℹ INFO |

---

## 4. The continuum gap and fallback strategy

Four of seven nebular components provide **no nebular continuum**:

| Backend | Continuum status |
|---------|-----------------|
| `CloudyGridBackend` | ✅ Full continuum from CLOUDY grid |
| `CueBackend` | ✅ Full continuum from neural emulator |
| `BakedInBackend` | ✅ Baked into SSP |
| `CB19Backend` | ❌ Stub: `([5000 Å], [0.0])` |
| `MappingsPhotoStellarBackend` | ❌ No method |
| `MappingsPhotoAGNBackend` | ❌ No method |
| `ShockEmission` | ❌ Stub |

For star-forming galaxies at z > 2, nebular continuum contributes 10–40% of the rest-frame UV
flux (Byler+2017). Silently returning zeros is physically wrong.

### 4.1 Fallback hierarchy

When a backend has no continuum, **automatically fall back** to the following in order:

1. **CueBackend continuum** — if a `CueBackend` instance is already active for this galaxy
   (most common case: MappingsPhotoStellarBackend used for lines, Cue already loaded for AGN NLR)
2. **CloudyGridBackend continuum** — if a CloudyGrid is loaded
3. **Analytic nebular continuum** — two-photon + free-free + free-bound from `_compute_analytic_nebular_continuum(logU, log_z_gas, log_T_e, log_qh)` (to be implemented in `_shared.py`)
4. **Raise `NebularContinuumUnavailableError`** with a clear message

### 4.2 Implementation: `NebularContinuumFallback` protocol

```python
# _protocol.py
from typing import Protocol
import jax.numpy as jnp

class NebularBackend(Protocol):
    """Unified interface all nebular backends must satisfy."""

    def line_luminosities(
        self,
        logU: jnp.ndarray,
        log_z_gas: jnp.ndarray,   # log10(Z/Zsun) — user convention
        log_age: jnp.ndarray,     # log10(age/yr)
        log_qh: jnp.ndarray,      # log10(photons/s)
        ssp_wave: jnp.ndarray,    # Å vacuum
        ssp_lnu: jnp.ndarray,     # Lsun/Hz
        catalog: LineCatalog,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (line_wavelengths [Å vacuum], line_luminosities [Lsun])."""
        ...

    def has_continuum(self) -> bool:
        """Return True if this backend provides nebular continuum."""
        ...

    def continuum_luminosity(
        self,
        logU: jnp.ndarray,
        log_z_gas: jnp.ndarray,
        log_age: jnp.ndarray,
        log_qh: jnp.ndarray,
        ssp_wave: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (wave [Å vacuum], L_nu [Lsun/Hz]).
        Must raise NebularContinuumUnavailableError if has_continuum() is False
        and no fallback is configured.
        """
        ...
```

### 4.3 The `NebularContinuumFallback` wrapper

```python
# _shared.py
class NebularContinuumFallback:
    """Wraps any NebularBackend and provides continuum via fallback hierarchy."""

    def __init__(
        self,
        primary: NebularBackend,
        fallback: NebularBackend | None = None,  # CueBackend or CloudyGridBackend
        fallback_mode: str = "analytic",          # "cue", "cloudy", "analytic", "error"
    ):
        self.primary = primary
        self.fallback = fallback
        self.fallback_mode = fallback_mode

    def continuum_luminosity(self, ...):
        if self.primary.has_continuum():
            return self.primary.continuum_luminosity(...)
        if self.fallback is not None:
            return self.fallback.continuum_luminosity(...)
        if self.fallback_mode == "analytic":
            return _compute_analytic_nebular_continuum(...)
        raise NebularContinuumUnavailableError(
            f"{type(self.primary).__name__} provides no nebular continuum. "
            "Pass fallback=CueBackend(...) or fallback_mode='analytic'."
        )
```

---

## 5. Code duplication inventory

These are the four categories of duplication that must be eliminated.

### 5.1 Physical constants (defined in 5 files)

```python
# Currently duplicated in:
# cloudy_grid.py, cloudy_cb19.py, mappings_photo.py, agn_nebular.py, ionizing_spectrum.py

_H_PLANCK = 6.626e-27    # erg s
_C_CGS    = 2.998e10     # cm/s
_LSUN_ERG = 3.846e33     # erg/s
_LYMAN_LIMIT = 912.0     # Å
_LOG10_ZSUN  = -1.848    # log10(Z_sun)
```

**Fix:** Create `_constants.py` and import everywhere. Zero duplication.

### 5.2 `compute_qh()` — three independent implementations

| File | Function name | Differences |
|------|---------------|-------------|
| `cloudy_grid.py` | `compute_qh(ssp_wave, ssp_lnu, ages, sfh_weights)` | vectorized via vmap |
| `cloudy_cb19.py` | `_compute_qh_spectrum(ssp_wave, ssp_lnu)` | single-SSP only |
| `mappings_photo.py` | `_compute_qh(ssp_wave, ssp_lnu, ages, sfh_weights)` | copy of cloudy_grid version |

**Fix:** Move the canonical version to `_shared.py`:

```python
# _shared.py
@partial(jax.jit, static_argnums=())
def compute_qh(ssp_wave: jnp.ndarray, ssp_lnu: jnp.ndarray) -> jnp.ndarray:
    """Q_H [photons/s] from a single SSP spectrum.
    ssp_wave: Å vacuum. ssp_lnu: Lsun/Hz.
    """
    mask = ssp_wave < _LYMAN_LIMIT
    nu = _C_CGS / (ssp_wave * 1e-8)           # Hz
    lnu_cgs = ssp_lnu * _LSUN_ERG             # erg/s/Hz
    integrand = lnu_cgs / (_H_PLANCK * nu)    # photons/s/Hz
    return jnp.trapz(jnp.where(mask, integrand, 0.0), nu)


def compute_qh_sfh_weighted(
    ssp_wave: jnp.ndarray,
    ssp_lnu_grid: jnp.ndarray,   # (n_ages, n_wave)
    sfh_weights: jnp.ndarray,    # (n_ages,)
) -> jnp.ndarray:
    """Q_H integrated over an SFH."""
    qh_per_age = jax.vmap(lambda lnu: compute_qh(ssp_wave, lnu))(ssp_lnu_grid)
    return jnp.dot(sfh_weights, qh_per_age)
```

### 5.3 `_interp_index_weight()` — two copies

Defined identically in `cloudy_grid.py` line ~85 and `mappings_photo.py` line ~90.

**Fix:** Move to `_shared.py`. Both files already import from each other in one direction
(`cloudy_cb19.py` already imports `_interp_index_weight` from `cloudy_grid.py`), so the
abstraction is already half-done.

### 5.4 Line profile placement — four copies

The Gaussian/delta line profile code appears in:
- `cloudy_grid.py:predict_nebular_sed()`
- `cloudy_cb19.py:predict_nebular_sed()`
- `mappings_photo.py:predict_{stellar,agn}_line_sed()`
- `shock.py:shock_emission_sed()`

**Fix:** Extract to `_shared.py`:

```python
def place_line_profiles(
    wave_obs: jnp.ndarray,          # Å vacuum, output wavelength grid
    line_waves: jnp.ndarray,        # Å vacuum, line centers
    line_lums: jnp.ndarray,         # Lsun, integrated line luminosities
    sigma_aa: float | jnp.ndarray,  # Å, line width (instrumental + intrinsic)
    delta_fallback: bool = False,   # use delta function if sigma_aa == 0
) -> jnp.ndarray:
    """Return L_nu [Lsun/Hz] on wave_obs grid with Gaussian line profiles."""
    ...
```

---

## 6. Abstract base class / Protocol

The current backends have three different method name conventions:

| Backend | Lines method | Continuum method |
|---------|-------------|-----------------|
| `CloudyGridBackend` | `predict_nebular_line_luminosities` | `predict_nebular_continuum` |
| `CB19Backend` | `predict_nebular_line_luminosities` | `predict_nebular_continuum` (stub) |
| `MappingsPhotoStellarBackend` | `predict_stellar_line_luminosities` | — |
| `MappingsPhotoAGNBackend` | `predict_agn_line_luminosities` | — |
| `BakedInBackend` | `predict_nebular_line_fluxes` | — |
| `CueBackend` | `predict_lines` + `predict_continuum` | `predict_continuum` |
| `ShockEmission` (functions) | `shock_line_ratios` | — |

**Fix:** Enforce `NebularBackend` Protocol (see Section 4.2). All backends implement
`line_luminosities()` and `continuum_luminosity()`. The caller always uses the same interface.

`ShockEmission` remains a module-level function for now (it is not parameterized by logU in the
same way) but wraps to a `ShockBackend` class for Protocol compliance.

---

## 7. Metallicity convention table and converter

Four metallicity axes in production simultaneously:

| Backend | Internal axis | Conversion to log10(Z/Zsun) |
|---------|--------------|----------------------------|
| DSPS / CloudyGrid | log10(Z) absolute | `log_z_abs - LOG10_ZSUN` |
| CB19 | log10(O/H) + CLOUDY offset | `log_oh + 1.222 + LOG10_ZSUN` |
| Mappings | ζ_O solar-relative | `log10(zeta_O)` |
| Cue | log10(Z/Zsun) | identity |
| User-facing (`neb_logZ_gas`) | log10(Z/Zsun) | identity |

**Fix:** Add to `_shared.py`:

```python
def neb_logzsol_to_cloudy_logoh(logzsol: jnp.ndarray) -> jnp.ndarray:
    """log10(Z/Zsun) → log10(O/H) on CLOUDY c17.01 solar scale."""
    return logzsol + _LOG10_ZSUN - _LOG_OH_OFFSET   # _LOG_OH_OFFSET = -1.222

def neb_logzsol_to_mappings_zeta(logzsol: jnp.ndarray) -> jnp.ndarray:
    """log10(Z/Zsun) → ζ_O (MAPPINGS V solar-relative)."""
    return 10.0 ** logzsol

def neb_logzsol_to_log_z_abs(logzsol: jnp.ndarray) -> jnp.ndarray:
    """log10(Z/Zsun) → log10(Z) absolute (DSPS convention)."""
    return logzsol + _LOG10_ZSUN
```

Each backend's internal `__call__` converts from the user-facing `neb_logZ_gas` (log10(Z/Zsun))
to its own axis using these functions. The user never sees the internal convention.

---

## 8. Refactoring plan (phased)

### Phase N-1: Shared infrastructure (no behavior change)

1. Create `_constants.py` — move all five sets of physical constants
2. Create `_shared.py` — move `compute_qh`, `_interp_index_weight`, `place_line_profiles`,
   metallicity converters
3. Update all five backend files to import from `_constants.py` and `_shared.py`
4. Run `pytest tests/ -q` — must be green with zero changes to test logic

**Verification:** `grep -r "_H_PLANCK" src/tengri/models/nebular/` returns only `_constants.py`.

### Phase N-2: Protocol + unified method names

1. Create `_protocol.py` with `NebularBackend` Protocol and `NebularContinuumUnavailableError`
2. Rename all backend methods to `line_luminosities` / `continuum_luminosity`
3. Keep old names as deprecated aliases (one release cycle)
4. Add `has_continuum() -> bool` to each backend
5. Add type annotations to all signatures
6. Update `sed_pipeline.py` call sites

**Verification:** `mypy src/tengri/models/nebular/` passes.

### Phase N-3: Ionizing spectrum warnings

1. Add `IonizingSpectrumMismatchWarning(UserWarning)` to `__init__.py`
2. Add `PhysicalInconsistencyWarning(UserWarning)` to `__init__.py`
3. Emit warnings per Section 3 in each backend's `__init__`
4. Write tests that `pytest.warns(IonizingSpectrumMismatchWarning)` fires for each relevant backend

### Phase N-4: Continuum fallback

1. Implement `_compute_analytic_nebular_continuum()` in `_shared.py`
   (two-photon + Lyman-α free-bound + Balmer free-bound + free-free; see Dopita & Sutherland 2003 §2)
2. Implement `NebularContinuumFallback` wrapper in `_shared.py`
3. For `CB19Backend`: use `NebularContinuumFallback(self, fallback=None, fallback_mode="analytic")`
   as the default; promote to `CueBackend` fallback if a Cue instance is passed
4. For `MappingsPhotoStellarBackend`: same
5. For `ShockEmission`: same
6. Write tests asserting that `continuum_luminosity()` returns finite arrays for all backends
7. Write FD gradient test: `∂(sum(continuum_lnu)) / ∂logU` is finite and matches JAX gradient

### Phase N-5: FD gradient tests for all backends

For every backend with traced parameters, add to `tests/unit/test_nebular_gradients.py`:

```python
@pytest.mark.parametrize("backend_cls", [CloudyGridBackend, CueBackend, CB19Backend])
def test_nebular_line_grad_logU(backend_cls, backend_fixture):
    backend = backend_fixture(backend_cls)
    eps = 1e-3
    logU = jnp.array(-2.5)
    L_plus  = jnp.sum(backend.line_luminosities(logU + eps, ...)[1])
    L_minus = jnp.sum(backend.line_luminosities(logU - eps, ...)[1])
    fd_grad = (L_plus - L_minus) / (2 * eps)
    jax_grad = jax.grad(lambda u: jnp.sum(backend.line_luminosities(u, ...)[1]))(logU)
    assert jnp.allclose(fd_grad, jax_grad, rtol=1e-2), f"FD/AD mismatch: {fd_grad} vs {jax_grad}"
```

---

## 9. Files to create vs. files to modify

### Create (new)

| File | Purpose |
|------|---------|
| `nebular/_constants.py` | Physical constants (one source of truth) |
| `nebular/_shared.py` | `compute_qh`, `interp_index_weight`, `place_line_profiles`, metallicity converters, `NebularContinuumFallback` |
| `nebular/_protocol.py` | `NebularBackend` Protocol, `NebularContinuumUnavailableError`, `IonizingSpectrumMismatchWarning`, `PhysicalInconsistencyWarning` |
| `tests/unit/test_nebular_gradients.py` | FD gradient tests for all backends |
| `tests/unit/test_nebular_continuum_fallback.py` | Fallback tests for no-continuum backends |
| `tests/unit/test_nebular_warnings.py` | Warning emission tests |

### Modify (existing)

| File | Changes |
|------|---------|
| `cloudy_grid.py` | Import from `_constants.py`, `_shared.py`; rename method; add `has_continuum = True` |
| `cloudy_cb19.py` | Same; use `NebularContinuumFallback` |
| `mappings_photo.py` | Same + `PhysicalInconsistencyWarning` for Stellar backend |
| `baked_in.py` | Rename `predict_nebular_line_fluxes` → `line_luminosities`; add warning |
| `cue.py` | Rename to unified API; add `has_continuum = True` |
| `shock.py` | Wrap functions in `ShockBackend` class; add continuum fallback |
| `agn_nebular.py` | Add AGN ionizing spectrum warning |
| `dig.py` | Update to call unified `line_luminosities()` API |
| `__init__.py` | Export new warning classes and Protocol |

---

## 10. What we would do differently from scratch

1. **Write Step 0A–0F first.** The API contract script, unit table, and module boundary tree would
   have been the first three artifacts — before any backend code.

2. **One backend, proven end-to-end, before writing the second.** `CloudyGridBackend` should have
   been the template. Its design would have been frozen and documented, then every subsequent
   backend would have been a variant of it, sharing `_constants.py` and `_shared.py` from day one.

3. **Protocol first.** `_protocol.py` with the `NebularBackend` Protocol would have been committed
   before any concrete backend. New backends would fail mypy until they satisfy the Protocol.

4. **Continuum is not optional.** Every backend would have been required to implement
   `continuum_luminosity()`. Backends that truly cannot provide it (Mappings, CB19) would have
   returned a `NebularContinuumFallback`-wrapped result using the analytic approximation by default.
   "Return zeros" would never have been an acceptable implementation.

5. **Warn about ionizing spectrum at the point of physics divergence.** The MAPPINGS stellar
   backends' use of a separate SB99 ionizing grid is a physical inconsistency that should have been
   a showstopper comment in the very first PR review. It would have been documented in the unit
   contract table and flagged with a warning at init time before the backend was merged.

6. **FD gradient tests before any VI run.** No parameter would have been called "differentiable"
   without a passing FD gradient test. The tests would have been written before the forward pass.

---

## Appendix A: Duplication summary

| Item | Copies | Files |
|------|--------|-------|
| `_H_PLANCK`, `_C_CGS`, etc. | 5 | cloudy_grid, cloudy_cb19, mappings_photo, agn_nebular, ionizing_spectrum |
| `compute_qh()` | 3 | cloudy_grid, cloudy_cb19, mappings_photo |
| `_interp_index_weight()` | 2 | cloudy_grid, mappings_photo |
| Line profile (Gaussian+delta) | 4 | cloudy_grid, cloudy_cb19, mappings_photo, shock |
| Metallicity conversion | 4 conventions | no shared converter |

---

## Appendix B: Test coverage gaps (pre-refactor)

- No FD gradient test for any backend
- No test that `continuum_luminosity()` returns non-zero for CB19/Mappings/Shock
- No test that `IonizingSpectrumMismatchWarning` fires
- No test for metallicity conversion round-trips
- No test for `mix_dig_emission` with non-zero `neb_dig_frac`
- No test for `agn_nlr_emission` dispatcher with `method="feltre"` (should raise `NotImplementedError`)

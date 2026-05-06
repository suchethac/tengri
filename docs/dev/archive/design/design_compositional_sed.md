# Compositional Rest-Frame SED Kernel — Implementation Blueprint

## Motivation

The current forward model has two paths:
1. **Fused kernel** (fast, ~140μs): Evaluates at filter effective wavelengths. Only supports a subset of physics components. Mixes physics + observation.
2. **Exact path** (slow, ~500-1000μs): Full-resolution SED. Supports all components. Python dispatch overhead per component.

The compositional architecture introduces a **Tier 2** path between these:
- Produces a rest-frame SED at SSP wavelength resolution (like exact)
- But JIT-compiled end-to-end (like fused)
- Supports ALL physics components
- Observation model is a separate thin wrapper → free-z spectroscopy works

## Architecture

```
Tier 1: Fused photometry (existing)
  → Fixed-z + fused-compatible components only
  → ~140μs, evaluates at effective wavelengths

Tier 2: Compositional rest-frame SED + observation wrapper (NEW)
  → All components, any z, photometry or spectroscopy
  → ~300-500μs estimated
  → JIT'd physics engine (z-independent) + JIT'd observation wrapper

Tier 3: Exact path (existing fallback)
  → Python dispatch, no end-to-end JIT
  → ~500-1000μs
```

## Key Files to Modify

| File | Changes |
|------|---------|
| `src/tengri/core/fused_kernels.py` | Add `build_fused_rest_sed()` factory |
| `src/tengri/core/model.py` | Add `_fused_rest_sed`, three-tier dispatch in `predict_photometry`/`predict_spectrum` |
| `src/tengri/core/sed_pipeline.py` | Extract component sub-functions |

## Implementation Plan

### 1. Extract component sub-functions from sed_pipeline.py

Each component in `compute_sed_components()` becomes a standalone pure JAX function:

```python
# In sed_pipeline.py or new file: src/tengri/core/sed_components.py

def build_ssp_component(model):
    """Returns: (sfr_on_ssp, log_z_abs, alpha_fe) -> ssp_flux_at_z, weights"""
    ssp_data = model.ssp_data
    ssp_ages_yr = model.ssp_ages_yr
    ...
    def ssp_fn(sfr_on_ssp, log_z_abs, alpha_fe=0.0):
        weights = compute_csp_weights(sfr_on_ssp, ssp_ages_yr)
        ssp_flux_at_z = interpolate_metallicity(...)
        return ssp_flux_at_z, weights
    return ssp_fn


def build_dust_atten_component(model):
    """Returns: (ssp_flux_at_z, weights, dust_params) -> sed_attenuated, sed_intrinsic, L_absorbed"""
    ...
    def dust_fn(ssp_flux_at_z, weights, tau_bc, tau_diff, dust_slope, **kwargs):
        # Two-component or single-component dust
        sed_attenuated = compute_csp_sed(weights, ssp_flux_at_z, dust_atten)
        sed_intrinsic = compute_csp_sed(weights, ssp_flux_at_z, ones)
        L_absorbed = -jnp.trapezoid(sed_intrinsic - sed_attenuated, nu)
        return sed_attenuated, sed_intrinsic, L_absorbed
    return dust_fn


def build_dust_emission_component(model):
    """Returns: (wave, L_absorbed, dust_params) -> dust_ir_sed"""
    emission_fn = get_emission_model(model._dust_emission_model)
    def dust_emit_fn(wave, L_ir, **dust_params):
        return emission_fn(wave, L_ir, **dust_params)
    return dust_emit_fn


def build_agn_component(model):
    """Returns: (wave, agn_params) -> agn_sed"""
    agn_model_fn = get_agn_model(model._agn_model)
    def agn_fn(wave, **agn_params):
        return agn_model_fn(wave, **agn_params)
    return agn_fn


def build_nebular_component(model):
    """Returns: (weights, wave, log_z, neb_params) -> neb_sed"""
    backend = model._nebular_backend
    def neb_fn(weights, wave, ssp_log_ages_yr, log_z, **neb_params):
        return backend.predict_nebular_sed(
            ssp_weights=weights, ssp_wave=wave,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z, **neb_params,
        )
    return neb_fn


def build_radio_component(model):
    """Returns: (wave, L_ir, L_agn_bol, radio_params) -> radio_sed"""
    ...

def build_xray_component(model):
    """Returns: (wave, sfr, mstar, L_agn_bol, xray_params) -> xray_sed"""
    ...
```

### 2. Compose into build_fused_rest_sed()

```python
# In fused_kernels.py

def build_fused_rest_sed(model):
    """Build a JIT'd function: internal_params -> rest-frame SED.
    
    Composes all enabled physics components into a single JIT'd function.
    Disabled components are excluded from the XLA graph at trace time.
    """
    ssp_fn = build_ssp_component(model)
    dust_atten_fn = build_dust_atten_component(model)
    
    # Optional components (Python if at trace time)
    has_dust_em = model._dust_emission_model is not None
    has_agn = model._agn_model is not None
    has_nebular = (model._nebular_backend is not None 
                   and model._nebular_backend.has_free_params)
    has_radio = model._radio_enabled
    has_xray = model._xray_enabled
    
    if has_dust_em:
        dust_emit_fn = build_dust_emission_component(model)
    if has_agn:
        agn_fn = build_agn_component(model)
    if has_nebular:
        nebular_fn = build_nebular_component(model)
    if has_radio:
        radio_fn = build_radio_component(model)
    if has_xray:
        xray_fn = build_xray_component(model)
    
    wave = model.ssp_data.ssp_wave
    ssp_log_ages_yr = model.ssp_log_ages_yr
    
    def rest_sed(p):
        """Compute rest-frame SED from internal parameters dict."""
        # 1. SSP + metallicity
        sfr = model._compute_sfr(p)
        sfr_on_ssp = jnp.interp(model.ssp_log_ages_yr, model.log_age_grid, sfr)
        ssp_flux_at_z, weights = ssp_fn(sfr_on_ssp, p["log_z_abs"])
        
        # 2. Dust attenuation
        sed, sed_intrinsic, L_absorbed = dust_atten_fn(
            ssp_flux_at_z, weights,
            p["tau_bc"], p["tau_diff"], p["dust_slope"],
        )
        
        # 3. Nebular
        if has_nebular:
            neb = nebular_fn(weights, wave, ssp_log_ages_yr, p["log_z_abs"],
                             neb_logU=p.get("neb_logU", -3.0),
                             neb_logZ_gas=p.get("neb_logZ_gas", None),
                             neb_fesc=p.get("neb_fesc", 0.0),
                             neb_fesc_lya=p.get("neb_fesc_lya", 0.0))
            sed = sed + neb
        
        # 4. Dust emission
        if has_dust_em:
            eta = p.get("dust_eta_balance", 1.0)
            L_ir = jnp.maximum(L_absorbed * eta, 0.0)
            dust_ir = dust_emit_fn(wave, L_ir,
                                    dust_T=p.get("dust_T", 35.0),
                                    dust_beta_ir=p.get("dust_beta_ir", 1.6))
            sed = sed + dust_ir
        
        # 5. AGN
        agn_bol = 0.0
        if has_agn:
            agn_params = {k: p.get(k, v) for k, v in AGN_DEFAULTS.items()}
            agn_sed = agn_fn(wave, **agn_params)
            sed = sed + agn_sed
            agn_bol = 10.0 ** p.get("agn_log_lbol", 10.0)
        
        # 6. Radio
        if has_radio:
            radio_sed = radio_fn(wave, L_ir, agn_bol, ...)
            sed = sed + radio_sed
        
        # 7. X-ray
        if has_xray:
            xray_sed = xray_fn(wave, sfr[-1], jnp.sum(weights), agn_bol, ...)
            sed = sed + xray_sed
        
        return sed
    
    return jax.jit(rest_sed)
```

### 3. Observation wrappers

Already exist in the codebase — `compute_spectrum()` in spectroscopy.py (line 304) is just `jnp.interp` + scaling. For photometry, `compute_flux_density()` integrates through filters.

```python
# In model.py — thin observation dispatch

@jax.jit
def _observe_spectrum(rest_sed, wave_rest, wave_obs, z, dl_cm):
    """Apply redshift + interpolate. ~1μs."""
    wave_rest_query = wave_obs / (1 + z)
    sed_at_pixels = jnp.interp(wave_rest_query, wave_rest, rest_sed)
    return (1 + z) / (4 * jnp.pi * dl_cm**2) * sed_at_pixels

@jax.jit
def _observe_photometry(rest_sed, wave_rest, z, dl_cm, filter_data):
    """Apply redshift + filter integration. ~10μs."""
    wave_obs = wave_rest * (1 + z)
    if igm_enabled:
        rest_sed = rest_sed * igm_transmission(wave_obs, z)
    # ... filter integration
```

### 4. Three-tier dispatch in Model

```python
def predict_photometry(self, params):
    # Tier 1: existing fused photometry (unchanged)
    if self._fused_photometry is not None and ...:
        return self._predict_photometry_fast(params)
    
    # Tier 2: compositional rest-frame SED + observation
    if self._fused_rest_sed is not None:
        p = self._get_internal_params(params)
        rest_sed = self._fused_rest_sed(p)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)
        return _observe_photometry(rest_sed, self.ssp_wave, z, dl_cm, ...)
    
    # Tier 3: exact path (unchanged)
    return self._predict_photometry_exact(params)


def predict_spectrum(self, params, wave_obs=None):
    # Tier 1: existing fused spectrum (fixed z only)
    if self._fused_spectrum is not None:
        return self._predict_spectrum_fast(params)
    
    # Tier 2: compositional SED + observation (handles free z!)
    if self._fused_rest_sed is not None:
        p = self._get_internal_params(params)
        rest_sed = self._fused_rest_sed(p)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)
        return _observe_spectrum(rest_sed, self.ssp_wave, wave_obs, z, dl_cm)
    
    # Tier 3: exact path
    ...
```

### 5. Build at Model.__init__ time

```python
# In Model.__init__()
# After existing precomputation

# Tier 2: compositional rest-frame SED (always available)
try:
    self._fused_rest_sed = build_fused_rest_sed(self)
except Exception as e:
    warnings.warn(f"Compositional SED kernel failed: {e}")
    self._fused_rest_sed = None
```

### 6. Testing

- Unit test: `test_fused_rest_sed.py`
  - Test each component builder independently
  - Test composed kernel matches exact path to <0.1%
  - Test free-z spectroscopy uses Tier 2
  - Test adding/removing components doesn't break other tiers
- Regression: compare `predict_photometry` and `predict_spectrum` outputs between Tier 2 and Tier 3 for all component combinations
- Performance: benchmark Tier 2 vs Tier 3 wall time

### 7. Edge Cases

- **Tabulated SFH**: `sfh_t_gyr + sfh_sfr` in params bypasses standard SFR computation. The compositional kernel needs a separate path or falls back to Tier 3.
- **Evolving metallicity**: Needs per-age Z interpolation. Can be handled in `ssp_fn` but adds complexity.
- **Chemical evolution**: Derives Z(t) from SFH — self-referential. Must be inside the JIT boundary.
- **DSPS table path**: Uses DSPS's own CSP weight computation. Incompatible with our `ssp_fn`. Falls back to Tier 3.

### 8. Estimated Effort

- Component extraction: ~2 hours (mostly refactoring existing code)
- Composition + JIT: ~1 hour
- Observation wrappers: ~30 min (mostly exists)
- Three-tier dispatch: ~30 min
- Testing: ~2 hours
- **Total: ~6 hours**

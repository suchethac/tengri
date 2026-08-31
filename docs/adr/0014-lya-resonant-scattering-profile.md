# ADR-0014: Lyman-α resonant-scattering profile

**Status:** Proposed (2026-05-24)

**Stakeholders:** Suchetha; users fitting JWST/NIRSpec EoR Lyα emitters, planning Lyα morphology studies.

**Closes design for:** #304 (Lyα emitted as scalar luminosity — no profile model).

## Context

Lyα is currently treated as a discrete scalar line at 1215.67 Å (vacuum) with a single luminosity value, scaled by an `fesc_lya` escape fraction. Real Lyα emerging from neutral-hydrogen gas is *not* a delta: it has the asymmetric red-wing shape produced by resonant scattering, and at high column densities a characteristic double-peak structure. This is a primary diagnostic of ISM kinematics for high-redshift star-forming galaxies — without it, tengri cannot meaningfully fit JWST/NIRSpec Lyα morphology.

The library currently exposes:

- Cue emulator returns Lyα as one scalar element of `predict_all_lines()`.
- CloudyGrid backends similarly return Lyα as a single wavelength bin.
- Only one Lyα-specific parameter: `neb_fesc_lya` (scalar escape fraction).

What's missing:

- No `N_HI`, `v_outflow`, or profile-shape parameters.
- No mechanism for nebular-output entries to carry *profile-bearing* spectra alongside scalar luminosities.

PR #320 (closes #303) introduced `EmissionLines.all_waves` / `all_lums` and a `.get(wavelength)` accessor — this gives us the substrate for a profile-bearing Lyα, but it doesn't change the line-as-scalar model.

## Decision

Ship two coupled additions:

### 1. Profile model selector

A new optional `lya_profile` block on `SEDModel.build`:

```python
model = SEDModel.build(
    ssp_data=ssp,
    sfh={...}, dust={...}, neb={'type': 'cue', 'all_params': Fixed(DEFAULT)},
    lya_profile={
        'type': 'verhamme2006',
        'log_nhi': Uniform(18.0, 22.0, units='log10 cm^-2'),
        'v_outflow': Uniform(-200.0, 600.0, units='km/s'),
    },
)
```

Registered options:

| `type` | Backing data | Cost |
|---|---|---|
| `'delta'` (default) | None — preserves current behavior | 0 |
| `'verhamme2006'` | Coarse 2-D lookup on (log_NHI, v_outflow), tabulated profiles from Verhamme+2006 Table 1 / appendix | ~µs per evaluation |
| `'gronke2015_emulator'` | Small NN emulator over the Gronke+2015 grid (asymmetric drift + temperature) | ~10 µs (JIT) |

`'delta'` is the no-op for existing fits — semantically the current behavior. `'verhamme2006'` is the table-based pilot; `'gronke2015_emulator'` is the differentiable / extensible follow-up.

### 2. Profile-bearing entries in the nebular output

Extend the line-publishing protocol so a nebular backend can publish either:

- **Scalar entry**: ``(wave, luminosity)`` — current behavior, used for every non-Lyα species.
- **Profile entry**: ``(wave_grid, lnu_grid)`` — wavelength array + Lν spectrum, contributes to the broadband nebular SED via direct addition (no Gaussian / delta scatter step) and to `EmissionLines` via integrated luminosity over a window.

`EmissionLines` gains:

```python
@dataclass(frozen=True)
class LineProfile:
    wave: jnp.ndarray   # (n_pix,) Å, vacuum, rest-frame
    lnu:  jnp.ndarray   # (n_pix,) erg/s/Hz
```

```python
class EmissionLines(NamedTuple):
    # ... existing fields ...
    lya_profile: LineProfile | None   # None when type='delta'
```

`Prediction.lines.lya_profile` is `None` for the default `'delta'` model — back-compat preserved.

### Composition with `fesc_lya`

`neb_fesc_lya` continues to scale the *integrated* Lyα luminosity: the profile is computed first (shape from N_HI / v_outflow), then renormalized so its integral equals the scalar `(1 − fesc_lya) × L_lya_intrinsic`. This keeps the existing energy-balance contract; only the *shape* is new.

### Dust attenuation

Profile-bearing entries flow through `attenuate_emission()` exactly like the existing `all_lums` catalog did in #320 (PR landed in cs/fix-nebular-surface). Each pixel of the profile multiplies by the attenuation at its wavelength. Birth-cloud vs diffuse selection follows the same `_neb_dust_mode` rule as the rest of the nebular surface.

## Consequences

### Positive

- **Closes #304** with a small, opt-in API. Default `'delta'` keeps every existing fit bit-exact.
- **Builds on #303's flexible nebular surface** instead of bolting on a parallel one.
- **Two implementation tiers** (`verhamme2006` table, `gronke2015_emulator`) let us ship a useful pilot quickly and upgrade later without API breakage.
- **Differentiable** — the NN emulator path makes N_HI / v_outflow inferrable from data via the standard JAX gradient path. The table path supports forward modeling only (returns NaN gradients for these parameters).

### Negative

- **New optional asset:** the Verhamme table or Gronke weights file must ship in `data/`. Adds ~MB to the install, with a graceful fallback (raise clear `ModuleNotFoundError`-style hint at build time).
- **`EmissionLines` becomes heterogeneous.** Adding `lya_profile: LineProfile | None` mixes a scalar-mostly NamedTuple with one optional pytree leaf. Acceptable cost but worth noting — sets precedent for future profile-bearing species (Mg II, C IV resonant fluorescence).
- **JIT discipline:** the `'delta'` and `'profile'` branches must share a structurally-equal pytree at the JIT boundary, or the cache splits. Resolution: `lya_profile=None` becomes a static-False sentinel at compile time, so two models with `type='delta'` and `type='verhamme2006'` get separate compiles — same pattern as other approx-policy splits today.

### Migration

- Default behavior unchanged (`type='delta'`).
- Users wanting profile fitting opt in via the new `lya_profile={...}` block.
- Existing `neb_fesc_lya` parameter unchanged in meaning.

## Implementation phasing

Depends on PR #320 landing (Lyα currently extracted from `state.derived["line_lums"]` as a scalar — needs the new flexible mapping surface).

1. **Verhamme+2006 table lookup** (smallest, no NN training). One file `components/nebular/lya_profile_verhamme.py`. New `LineProfile` dataclass. Plumbing in `NebularSEDComponent` to call the profile function and publish `state.derived["lya_profile"]`. `state_to_emission_lines` reads it.
2. **Gronke+2015 NN emulator** (follow-up). Reuses the same plumbing; swaps in a JAX-evaluable network.
3. **Gallery script** demonstrating profile shape across N_HI / v_outflow regimes (the script user mentioned in #304 wanting to write).

Phase 1 is the smallest closes-the-issue PR; phase 2 is the differentiable-fitting upgrade; phase 3 is the documentation/visualization deliverable.

## References

- Verhamme et al. 2006, A&A, 460, 397 — analytic + table fits for resonant Lyα profiles.
- Gronke et al. 2015, ApJ, 812, 123 — fitting framework + parameter grids.
- #304: open issue this design closes.
- #303 / PR #320: the nebular-output redesign this proposal depends on.
- ADR-0011: SEDModelComponent base — `lya_profile={}` follows the standard nested-dict grammar.

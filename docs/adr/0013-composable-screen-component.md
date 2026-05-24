# ADR-0013: `ScreenComponent` — composable transmission screens

**Status:** Proposed (2026-05-24)

**Stakeholders:** Suchetha; users fitting Milky-Way foreground, obscured AGN, edge-on Seyferts.

**Closes design for:** #292 (X-ray N_H absorption), #294 (AGN torus screening disc), #297 (MW foreground extinction).

## Context

Three open issues describe *missing physics that already exists in the codebase as components* — they fail because tengri's component model can only *emit* (`predict(p, sed_in, wave) -> (sed_in + new_emission, derived)`), not *screen* (multiply an upstream SED by a transmission curve).

| Issue | Physics | What's missing |
|---|---|---|
| #297 | Milky-Way foreground extinction (Cardelli+1989) | The `MilkyWay` dust component exists at `components/dust/mw_model.py` but shares `parameter_prefix='dust_'` with the host-galaxy `two_component` block — only one can be active at a time. Users at low Galactic latitudes can't fit through a MW screen *and* a host-galaxy two-component dust model. |
| #292 | X-ray photoelectric absorption (Wilms+2000 `tbabs`) | `XRayAird` exposes spectral indices and break energy but no `N_H` parameter. Obscured AGN (Compton-thin to thick) are not expressible. |
| #294 | Torus screening of the disc (Stalevski+2016) | `AGNSEDComponent` adds disc + torus + lines additively. Edge-on inclinations don't obscure the disc UV bump — `agn['torus']['inclination_deg']` sweeps don't reproduce the Seyfert 1 → Seyfert 2 transition. The SKIRTOR table includes the transmitted-disc component but it's not wired in. |

All three share the same *shape* — a transmission `T(λ, params) ∈ [0, 1]` (or `T(E, ...)` for X-ray) applied to an upstream SED at a chosen pipeline slot — and all three are blocked by the same architectural gap.

## Decision

Introduce a new component subclass `ScreenComponent(SEDModelComponent)` that *modifies* `state.sed_intrinsic` (or a chosen state field) by an upstream-evaluated transmission curve, without contributing emission.

```python
class MilkyWayForegroundScreen(ScreenComponent):
    name = "mw_foreground"
    parameter_prefix = "mw_"
    target = "sed_observed"        # apply to observed-frame SED after IGM
    apply_at = "post_observation"  # pipeline slot

    ebmv = Uniform(0.0, 1.0, "MW E(B-V)", units="mag")
    rv   = Fixed(3.1, "MW R_V",           units="")

    def transmission(self, p, wave_obs):
        k_lambda = cardelli(wave_obs, r_v=p["rv"])
        a_v      = p["rv"] * p["ebmv"]
        return jnp.exp(-a_v * k_lambda / 1.086)

    # Default predict: sed_in × transmission, no emission added
```

The base class auto-implements `predict(p, sed_in, wave, **inputs)` as `(sed_in * self.transmission(p, wave), {})`. Subclasses override `transmission()` only.

### Pipeline slot dispatch

A new `apply_at: str` class attribute selects when the screen runs in the orchestrator chain:

| `apply_at` | Runs after | Sees |
|---|---|---|
| `"pre_dust"` | StellarSEDComponent | Stellar SED (rest-frame, intrinsic) |
| `"post_dust"` | DustSEDComponent | Stellar + dust + nebular + AGN (rest-frame, attenuated) |
| `"post_observation"` | IGMSEDComponent | Final observed-frame SED |
| `"on_component"` (advanced) | A specific named component output | Slot-targeted (e.g. disc-only for torus screening) |

The orchestrator adds a screen-application phase after each named step.

### Three example users

```python
# Issue #297 — MW foreground (post-observation, observer-frame screen)
model = SEDModel.build(
    ssp_data=ssp,
    sfh={...}, dust={...},          # host galaxy
    foreground={'type': 'mw', 'ebmv': Fixed(0.05)},
)

# Issue #292 — X-ray absorption (post-emission, rest-frame screen on X-ray)
model = SEDModel.build(
    ssp_data=ssp,
    sfh={...}, dust={...},
    xray={'type': 'aird', 'screen': {'type': 'tbabs', 'log_nh': Fixed(22.0)}},
)

# Issue #294 — Torus screens disc (intra-AGN slot)
model = SEDModel.build(
    ssp_data=ssp,
    agn={
        'disc': {'type': 'qsogen'},
        'torus': {'type': 'skirtor', 'inclination_deg': Fixed(85)},
        'torus_screens_disc': True,   # opt-in switch
    },
)
```

### What stays the same

- `SEDModelComponent` base, the `_REGISTRY`, and existing components — untouched.
- `ScreenComponent` *is a* `SEDModelComponent` with `inputs = {}` and `outputs = {}` by default. Existing chain machinery (orchestrator, factory) sees it as an ordinary component, plus a small dispatch branch in `state_to_*` for the slot annotation.

### What changes

- New base class in `src/tengri/components/screen_component.py`.
- New `apply_at` enum (`"pre_dust"`, `"post_dust"`, `"post_observation"`, `"on_component"`).
- New grammar slots: `foreground={}` at top level, `screen={}` nested in any component dict.
- Three concrete screens: `MilkyWayForegroundScreen` (closes #297), `TbabsAbsorption` (closes #292), `TorusDiscScreen` (closes #294, wires the SKIRTOR transmission table).

## Consequences

### Positive

- **Closes three open issues without touching `SEDModelComponent`** — the screen subclass is purely additive.
- **The composability gap is named and locked.** Future "law applied to upstream SED" requests (e.g. patchy IGM at high-z, polarisation screens) inherit the slot grammar.
- **Self-describing** — `list_screens()` / `describe_screen("tbabs")` fits the same introspection surface as #310 proposes for AGN models and recipes.

### Negative

- **More slot-dispatch logic.** The orchestrator gains a routing table (`apply_at` → insertion point). Mitigated by keeping the enum closed (four values, all observable in `chain_summary()`).
- **Documentation surface grows.** A new "screens" concept to teach alongside emission components. Mitigated by treating ScreenComponent as a *subclass* in docs ("Screens are emission components that don't emit — they multiply").
- **Per-component-slot screens** (`screen={}` nested in X-ray, AGN disc) require the orchestrator to carry component-named state slots. This is the riskiest piece; deferring `on_component` slot to a follow-up keeps the initial impl small.

### Migration

- `MilkyWay` and `MWExtinction` dust-model variants stay as deprecated aliases, with `DeprecationWarning` pointing at `foreground={'type': 'mw', ...}`.
- `XRayAird` gets a new optional `screen={}` slot; existing models without it are unchanged.
- AGN composer gains a `torus_screens_disc` flag, default `False` (preserves current behaviour).

## Implementation phasing

1. **ScreenComponent base + `post_observation` slot** (smallest, closes #297). One new file, one new top-level grammar slot, one concrete subclass.
2. **`pre_*` / `post_*` slots in the orchestrator chain** (closes #292 and any future post-stellar / post-dust screens via the same mechanism).
3. **`on_component` slot for intra-AGN torus screening** (closes #294). Trickiest because it needs the SKIRTOR table's transmitted-disc payload to flow through state.
4. **Self-describing surface** (`list_screens`, `describe_screen`) — folds into the #310 introspection PR.

Each phase ships independently; #297 alone unblocks a meaningful gallery class (low-latitude sources).

## References

- ADR-0011: SEDModelComponent base — defines the additive component contract this builds on.
- #292 / #294 / #297: open issues this design closes.
- #310 proposal 4 (MW foreground as top-level block): directly addressed by phase 1.
- Stalevski et al. 2016 (SKIRTOR): inclination-dependent disc transmission table.
- Wilms, Allen & McCray 2000 (`tbabs`): photoelectric absorption cross-sections.
- Cardelli, Clayton & Mathis 1989: MW extinction law (already in `components/dust/attenuation.cardelli`).

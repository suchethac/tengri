# Spatial wiring into ForwardModel + joint spec-phot demo plan

> Item #6 of the post-tracer-bullet architecture sequence. Lands the
> scientific main path: a user can construct a `ForwardModel` with a
> spatial submodel, run `forward.predict(params)`, and get back a
> prediction dict that correctly handles fiber-aperture vs total-flux
> mismatch.

**Depends on:** PR #180 (SpatialComponent Protocol + concrete profiles), PR #181 (SpatialModel + SpatialSEDModel composers).

**Goal:** A user can write

```python
forward = ForwardModel.build(
    sed=SEDModel.build(...),
    spatial=SpatialModel(components=[Sersic()]),
    observation=JointObservation(
        photometry=Photometry(...),
        fiber_spec=FiberSpectroscopy(aperture_arcsec=1.5, ...),
    ),
)
fit = forward.predict(params)
# fit["phot_fnu"] — full-galaxy photometry
# fit["spec_fnu"] — fiber-aperture-integrated spectroscopy
```

…and the photometry sees the total flux while the spectroscopy sees only the fraction of the spatial profile inside the fiber footprint. **No flat-slab approximation.**

## Open design choices

### 1. Where does the spatial grid `(x_kpc, y_kpc)` come from?

Three candidates. The choice affects API ergonomics, observation-model coupling, and JIT cache stability.

**A. On the `SpatialModel`.** `SpatialModel(components=[Sersic()], grid_kpc=...)`. The grid is the sub-model's static configuration. JIT-cache stable, but the user has to choose the resolution up front.

**B. On the `Observation`.** Imaging observation provides the pixel grid (in arcsec); cosmology converts arcsec → kpc. Fiber-spec provides the aperture mask. JIT-cache stable per observation but multiple instruments mean multiple grids per fit.

**C. On `ForwardModel.build(spatial_grid_kpc=...)`.** Top-level passthrough. Pragmatic, easy to override, but exposes a detail at the outer-shell level that maybe shouldn't live there.

**My lean: A** (on `SpatialModel`). The grid is part of the spatial physics resolution, not the instrument. Observation models consume the spatial profile and convolve with their own PSF/LSF as needed.

### 2. How does aperture integration work for the fiber spectrum?

Given a `spatial_profile_2d` and a fiber aperture, the fiber sees only the fraction of the integrated profile inside the aperture footprint. Two implementations:

**a. Analytic.** For circular fiber + circular Sersic, the integral has closed form. Fast, exact, but locks us to specific profile/aperture combinations.

**b. Numerical.** Multiply profile by aperture mask, integrate. Works for any profile/aperture. Default choice.

The aperture-fraction scales the SED: `fiber_spec ∝ aperture_fraction × full_SED`. Architecturally this lives in `FiberSpectroscopyObservation.predict(state, params)`.

### 3. New ObservationModel adapters?

Two new classes are required for a credible joint spec-phot demo:

- **`FiberSpectroscopyObservation`** — wraps `Spectroscopy` + a circular fiber aperture (arcsec). `predict(state, params)` reads `spatial_profile_2d`, computes the aperture-fraction integral, scales the spectroscopy.
- **`TotalPhotometryObservation`** — the existing `Photometry`/`PhotometryObservationModel` but explicitly labeled "total flux" so the joint composer can distinguish per-channel.

Both follow the same `predict(state, params) → dict` shape (Protocol exists already).

### 4. `JointObservation` composer

Composer that holds N `ObservationModel`s and returns a merged dict:

```python
class JointObservation:
    def __init__(self, photometry, fiber_spec=None, imaging=None):
        self._models = ...

    def predict(self, state, params):
        out = {}
        for m in self._models:
            out.update(m.predict(state, params))
        return out
```

If the existing `Observation` class already handles photometry+spec jointly, this is just refactoring its `predict` into a composable form.

---

## Task breakdown

### Task 1: SpatialModel gets a `grid_kpc` attribute

- Extend `SpatialModel` to carry `grid_kpc: tuple[ndarray, ndarray]` (default: a 64×64 grid from -10 kpc to +10 kpc as a sensible smoke-test default; users override for production).
- `run(state, params)` inserts `state.derived["spatial_grid_xy_kpc"] = self.grid_kpc` before calling components.
- Add a test that constructing a Sersic into a SpatialModel works without the user manually setting up the grid.

### Task 2: Wire `ForwardModel.build(spatial=...)` and `ForwardModel.predict`

- Accept `spatial=SpatialModel(...)` kwarg.
- Construct `Population(name="default", sed=sed, spatial=spatial)`.
- In `predict`, when `pop.spatial is not None`, thread state through both: `pop.sed.run(state) → pop.spatial.run(state) → observation.predict(state)`.
- Test: numerical equivalence with the no-spatial path when `spatial=None`; smoke test with `Sersic` + photometry observation.

### Task 3: `FiberSpectroscopyObservation` adapter

- New file: `src/tengri/observation/fiber_spectroscopy.py`.
- Wraps existing Spectroscopy + a `fiber_radius_arcsec` field (and optional `fiber_offset_arcsec_xy`).
- `predict(state, params)` reads `state.derived["spatial_profile_2d"]` and computes the aperture fraction by numerical integration of profile × mask.
- Returns `{"spec_fnu": aperture_frac × spectrum}`.
- Tests: fraction → 1 for very-large aperture; fraction → 0 for very-small aperture; smooth limit between.

### Task 4: `JointObservation` composer

- New file: `src/tengri/observation/joint.py`.
- Holds an iterable of observation models.
- `predict(state, params)` calls each, merges dict outputs.
- No new physics. Test: ensures dict merging is consistent (no key collisions, all keys present).

### Task 5: The joint spec-phot demo notebook

`notebooks/19_joint_spec_phot.py` (jupytext percent-format):

- Build a synthetic mock: Sersic galaxy + SDSS-like 5-band photometry + 2-arcsec-fiber 4000–9000 Å spectrum.
- Build the same model two ways:
  1. **Flat-slab** (current standard practice): use `FlatSlab` spatial, observe photometry + spec with no aperture-correction step.
  2. **Sérsic + fiber**: use `Sersic` spatial, `FiberSpectroscopyObservation` with the correct aperture.
- Fit both, show the recovered stellar mass differs by ~factor 2 (the scientific motivation: flat-slab gets it wrong).
- The notebook is the demonstration of the §2 architecture-spec motivation.

### Task 6: Docs + CHANGELOG

- New entries:
  - `ForwardModel.build(spatial=...)` is now functional
  - `FiberSpectroscopyObservation`
  - `JointObservation`
- New section in `where-things-live.md` for spatial-observation models.

### Task 7: Push + PR

- Stacked on #181 (SpatialModel + SpatialSEDModel) since it needs both `SpatialModel` and the typed-bundle spatial keys.
- Demo notebook is the headline.

---

## Out of scope

- IFU resolved spectroscopy (the spatial profile is convolved with a PSF + sampled per-spaxel). Architecture supports it; the demo notebook doesn't.
- Per-age spatial profiles (B path) — `PerAgeSersic` component lands when color-gradient fits become a real ask.
- `BulgeDisk` composer profile (`Sersic` + `Exponential` summed).
- Multi-population (item #3 / ADR-0012).

---

## Self-review checklist

- [ ] `ForwardModel.build(spatial=...)` constructs a `Population` with both slots filled.
- [ ] `ForwardModel.predict` threads SED → Spatial → Observation correctly.
- [ ] `FiberSpectroscopyObservation` aperture-fraction integral is JIT-clean and differentiable.
- [ ] The demo notebook actually demonstrates the scientific point (recovered M_* differs by ~2× between flat-slab and Sérsic).
- [ ] No existing notebook is broken (spatial=None remains the default).
- [ ] All new tests pass.

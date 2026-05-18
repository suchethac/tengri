# Spatial Model Extension Path (forward-looking design note)

> Status: **deferred — not started.** Captured 2026-05-03 so the
> Phase II-1 protocol surface (SEDComponent / ObservationModel /
> Likelihood) doesn't paint itself into a corner before the spatial
> model lands.

## What the user wants

A spatial forward model that joins:

1. **Resolved imaging** — 2D F_nu maps per filter (e.g. HST, JWST, MUSE
   continuum slices).
2. **Broadband photometry** — single-aperture or total fluxes (the
   current `Photometry` channel).
3. **Fiber spectroscopy** — 1D spectrum at a known sky position with a
   fiber aperture (e.g. SDSS, DESI, MOONS, MaNGA single-fiber
   extractions).

All three observe the *same physical galaxy*; the differences are the
spatial sampling (resolved vs aperture-integrated vs fiber-localised)
and the spectral resolution (broad-band vs narrow-band vs R~few
hundred to several thousand). A joint fit needs:

- a spatial profile (e.g. Sérsic, exponential, GP-textured) that
  carries SED-per-pixel at the model's intrinsic resolution;
- a PSF / LSF / fiber-aperture model that maps that intrinsic profile
  onto each instrument's sampling;
- a likelihood that scores all three channels against their data
  simultaneously, accounting for noise correlated within an
  instrument and (typically) uncorrelated across instruments.

## How the existing Phase II-1 surface absorbs this

The protocol abstractions already in place generalise cleanly. **No
breaking change to the contracts is anticipated.**

### Forward model side — SEDComponent

Add one new component that publishes the spatial profile:

```python
class SpatialProfileSEDComponent:
    name = "spatial_profile"
    parameter_prefix = "spatial_"

    def declared_parameters(self):
        return [
            ParamDeclaration("spatial_log_re_kpc", ..., "Effective radius"),
            ParamDeclaration("spatial_n_sersic", ..., "Sérsic index"),
            ParamDeclaration("spatial_axis_ratio", ..., "Axis ratio b/a"),
            ParamDeclaration("spatial_pa_deg", ..., "Position angle"),
        ]

    def apply(self, state, params):
        # Publish a unit-normalised 2D surface-brightness profile
        # (the *spatial* part of the SED — the spectral part is
        # already in state.sed_intrinsic / sed_attenuated).
        # Typed bundle write (ADR-0007). When new fields like
        # ``spatial_profile_2d`` are added, add a matching field on
        # ``tengri.core.DerivedBundle`` and an entry in
        # ``tengri.forward.orchestrator._CANONICAL_UNITS`` in the
        # same PR — the canonical-units check enforces this on every
        # ``build_components`` call.
        return state.with_(
            derived=state.derived.with_(
                spatial_profile_2d=sersic_profile(
                    params["spatial_log_re_kpc"], params["spatial_n_sersic"], ...
                ),
                spatial_grid_xy_kpc=(x_grid, y_grid),
            ),
        )
```

This component reads no other component and writes only to
`state.derived` — it composes orthogonally with the existing 6 SED
adapters.

For sub-galactic *colour gradients* (each SSP age has a different
spatial extent), the component publishes a richer
`state.derived["spatial_profile_per_age"]` of shape
`(n_age, ny, nx)`. The `StellarSEDComponent` (Phase II-2) already
publishes `state.derived["lnu_age"]` of shape `(n_age, n_wave)`; the
spatial component just needs to publish a parallel age-resolved
spatial cube.

### Observation side — three new ObservationModel adapters

Each instrument gets its own `ObservationModel` adapter; each reads
from `state` what it needs and publishes its predicted observable
under a unique prediction-dict key:

```python
class ImagingObservationModel:
    """Convolves spatial profile × per-filter SED with the PSF, then
    samples onto the imaging pixel grid."""
    name = "imaging"
    def predict(self, state, params):
        # Reads state.sed_attenuated AND state.derived["spatial_profile_per_age"]
        # Convolves with self.psf_kernel, samples onto self.pixel_grid
        return {"imaging_fnu_pixel": <ndarray, shape (n_filter, ny, nx)>}

class FiberSpectroscopyObservationModel:
    """Integrates spatial profile inside the fiber footprint, then
    applies the LSF to produce a 1D spectrum at the fiber position."""
    name = "fiber_spec"
    def predict(self, state, params):
        # Reads state.sed_attenuated AND state.derived["spatial_profile_per_age"]
        # Integrates over self.fiber_aperture_mask, applies self.lsf
        return {"fiber_spec_fnu": <ndarray, shape (n_pixels,)>}

class TotalPhotometryObservationModel:
    """Existing PhotometryObservationModel, just renamed for clarity
    — integrates the *whole* spatial profile through each filter."""
    name = "total_photometry"
    def predict(self, state, params):
        return {"phot_fnu": <ndarray, shape (n_filter,)>}
```

A user with imaging + photometry + fiber spectroscopy would compose:

```python
observation = JointObservationModel(
    ImagingObservationModel(psf=hst_psf, pixel_grid=hst_pixels),
    TotalPhotometryObservationModel(filters=broadband_filters),
    FiberSpectroscopyObservationModel(
        fiber_position_arcsec=(0.0, 0.0),
        fiber_diameter_arcsec=2.0,
        lsf=desi_lsf,
    ),
)
```

`JointObservationModel` is the observation-side analogue of
`CompositeLikelihood`: takes a list of `ObservationModel`s, calls
`.predict()` on each, merges the resulting dicts, returns the union.
A single new ~30-line class.

### Likelihood side — three new Likelihood adapters

Same pattern as the existing `PhotometryLikelihood` /
`SpectroscopyLikelihood`, just keyed differently:

```python
likelihood = CompositeLikelihood(
    ImagingLikelihood(imaging_data, imaging_err),       # reads "imaging_fnu_pixel"
    PhotometryLikelihood(phot_fnu, phot_err),           # reads "phot_fnu"
    FiberSpectroscopyLikelihood(spec_fnu, spec_err),    # reads "fiber_spec_fnu"
)
```

`ImagingLikelihood` may want to ride a per-pixel covariance (read
noise + photon noise + sky residual covariance) rather than diagonal
errors — that's a `gp_log_prob` helper analogous to
`diag_gaussian_log_prob`, lives in
`tengri/inference/likelihoods/gp.py` when needed.

## What the current Phase II-1 surface needs to allow

Three things, all already true:

1. **Components publish to `state.derived` with stable keys.** Spatial
   profile data (or per-age cubes) goes into `state.derived` like any
   other cross-component handshake. ✓
2. **ObservationModels return a dict.** Multiple observation channels
   coexist by using different dict keys; downstream Likelihoods read
   only what they need. ✓
3. **Likelihoods compose via `CompositeLikelihood`.** Joint scoring
   across instruments is one constructor call, no god-class. ✓

## What the current Phase II-1 surface should *not* commit to

- **Don't bake a `wave` axis convention into `PipelineState`.** Today
  `state.wave` is a 1-D rest-frame wavelength grid. The spatial work
  needs wavelength to coexist with `(y, x)` spatial axes. When the
  spatial model lands, `state.wave` stays 1-D (the spectral axis); the
  spatial axes live in `state.derived["spatial_grid_xy_kpc"]` and the
  per-pixel SED is `(n_age, ny, nx, n_wave)` reconstructed inside each
  `ObservationModel.predict`. This is already how it works.
- **Don't assume `Likelihood.log_prob` operates on a single channel.**
  The existing `prediction` mapping signature already handles multiple
  channels via dict keys. ✓
- **Don't hard-code `phot_fnu` / `spec_fnu` as the only valid keys
  anywhere upstream of Likelihoods.** The orchestrator's prediction
  dict construction (`build_loss_fn`) currently only emits these two,
  but downstream likelihoods choose what they read. New keys
  (`imaging_fnu_pixel`, `fiber_spec_fnu`) get added the same way.

## Acceptance test (when this lands)

A single end-to-end test exercising the three-instrument case:

```python
def test_joint_imaging_phot_fiberspec_recovery():
    chain = [
        StellarSEDComponent(...),
        DustAttenuationSEDComponent(...),
        SpatialProfileSEDComponent(...),  # NEW
    ]
    obs = JointObservationModel(
        ImagingObservationModel(psf=mock_psf, pixel_grid=mock_grid),
        TotalPhotometryObservationModel(filters=mock_filters),
        FiberSpectroscopyObservationModel(
            fiber_position_arcsec=(0, 0), fiber_diameter_arcsec=1.5
        ),
    )
    likelihood = CompositeLikelihood(
        ImagingLikelihood(imaging_data, imaging_err),
        PhotometryLikelihood(phot_data, phot_err),
        FiberSpectroscopyLikelihood(spec_data, spec_err),
    )
    fitter = tengri.Fitter(
        components=chain, observation=obs, likelihood=likelihood,
        parameters=parameters, method="vi",
    )
    posterior = fitter.run(key)
    # Recovered M_*, SFR, R_e within mock truth ± 2σ.
```

This is Path 3 (`Fitter(components=, observation=, likelihood=,
parameters=)`), still deferred. When it lands, the spatial model is a
straightforward extension.

## Status

- Captured in:
  - `~/.claude/projects/-Users-suchethacooray-Projects-tengri/memory/project_spatial_extension.md`
  - This document.
- No code change today.
- Reviewed before any future change to `core/component.py`,
  `core/observation.py`, `core/likelihood.py` to confirm the spatial
  use-case still fits.

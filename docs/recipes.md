# Recipes

`tengri.recipes` provides curated starting configurations — full nested-dict
model specifications you can hand straight to `SEDModel.build`. They are the
fastest way to a sensible model without assembling every block by hand.

```python
import tengri
from tengri import SEDModel

ssp = tengri.load_ssp("...")             # see the recipe's SSP requirement
model = SEDModel.build(ssp_data=ssp, observation=obs,
                       **tengri.recipes.star_forming_photometry())
```

The current set, with each recipe's SSP requirement, is a function call:

```python
tengri.list_recipes()                    # table: name, short_doc, ssp_requirement
tengri.describe("star_forming_photometry")
```

## The six recipes

| Recipe | For | SSP requirement |
|--------|-----|-----------------|
| `star_forming_photometry` | Star-forming galaxies, broadband photometry, 0 < z < 6 | bare-stellar (Cue nebular) |
| `quiescent_z0` | Quiescent galaxies at z ≈ 0.05 | bare-stellar (Cue nebular) |
| `agn_panchromatic` | AGN-dominated, multi-wavelength (disc + torus + NLR + radio + X-ray) | bare-stellar (Cue nebular) |
| `stochastic_sfh_jwst` | JWST high-z with IFT correlated-field stochastic SFH | bare-stellar (Cue nebular) |
| `mock_recovery_minimal` | Minimal model for recovery tests / benchmarking | any (nebular disabled) |
| `dust_demo` | Forward-only dust-attenuation gallery sweeps | wNE (with-nebular-emission) |

## SSP requirement — read it first

Four of the six recipes use the **Cue** nebular backend, which requires a
**bare-stellar** SSP grid (no nebular emission baked in). Passing a `wNE`
("with nebular emission") SSP to a Cue recipe raises at construction. `dust_demo`
is the opposite — it expects a `wNE` grid. `tengri.describe("<recipe>")` prints
the exact requirement and an example filename. Pre-formatted grids are mirrored
at <https://halos.as.arizona.edu/suchethacooray/ssp-spectra/>.

## They are starting points, not black boxes

A recipe returns a plain nested dict. Inspect it, edit it, or override blocks
before building:

```python
spec = tengri.recipes.star_forming_photometry()
spec["redshift"] = tengri.Fixed(0.1)     # pin redshift
spec["dust"]["law_bc"] = "calzetti"      # swap the birth-cloud law
model = SEDModel.build(ssp_data=ssp, observation=obs, **spec)
model.spec.summary()                      # provenance-tagged view of every parameter
```

All recipes set `approx=WavePrecomp()` for the photometry lookup-table speedup.
The full nested-dict grammar is worked through in
[notebook 04](spine/04_building_models) (and documented for contributors in
`docs/dev/api_migration_v0.x.md`).

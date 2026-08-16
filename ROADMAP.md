# Tengri Roadmap

Planned physics modules not yet implemented. These are tracked here rather
than in the Sphinx docs to avoid documenting unimplemented features.

Each entry below was checked against the live registries before being listed.
If you are adding one, check the same way rather than trusting this file — the
previous version of this page listed eight modules as unimplemented that had
already shipped, because nothing kept it honest.

## Planned Physics Modules

### MAGPHYS-style Dust
Energy-balance dust model from da Cunha+2008.
Motivation: alternative to DL07 for high-z submillimeter-selected sources.
Not implemented: `DUST_EMISSION_MODELS` has no `magphys` entry, and the name
appears in the tree only as prose comparing tengri's energy balance to what
CIGALE and MAGPHYS enforce.

### TEA Dust Attenuation Model
Infrared excess attenuation model for Milky Way environments.
Motivation: improved UV-to-IR energy balance consistency.
Not implemented: no attenuation model of this name exists in the tree.

## Already delivered

These were listed as planned long after they shipped. Each is a live registry
entry or public builder today, and most have a rendered example in the gallery:

| Was planned as | Reach it via |
|---|---|
| Chemical Evolution Z(t) | `chem_evol` in the metallicity-mode registry |
| Shock Emission (MAPPINGS) | `compute_shock_sed`, `shock_line_ratios` from `tengri.components.nebular` |
| ADAF Disc | `builders.agn.disc.adaf`, `builders.agn.disc.adaf_lopez2024` |
| THEMIS Dust | `themis` in `DUST_EMISSION_MODELS` |
| Patchy IGM Reionization | `builders.igm.inoue14(patchy=True, bubble_mpc=...)` |
| PAH Emission Features | `draine2021_pah`, `pah_drude` in `DUST_EMISSION_MODELS` |
| Astrodust + PAH | `astrodust` in `DUST_EMISSION_MODELS` |
| BOSA Templates | `bosa` in `DUST_EMISSION_MODELS` |

To see what is actually registered rather than what any document claims:

```python
from tengri.components.dust.emission import DUST_EMISSION_MODELS
from tengri.components.igm import IGM_MODELS
from tengri.components.nebular import NEBULAR_MODELS

sorted(DUST_EMISSION_MODELS)
```

The gallery has worked examples for several of them — see the dust emission and
nebular sections of the published examples index.

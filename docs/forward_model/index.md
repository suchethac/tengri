# Forward model

The differentiable forward chain follows the standard SED-fitting cascade:
star formation history → simple stellar populations → nebular and AGN
emission → dust attenuation and re-emission → IGM absorption → the
observables (photometry and spectroscopy). Each step is a pure JAX
function, so the whole chain is JIT-compiled, gradient-traceable, and
batchable through `vmap`.

Spine notebooks that exercise the forward model:

| Notebook | Focus |
|----------|-------|
| [`02_sed_anatomy`](../spine/02_sed_anatomy) | The panchromatic SED, component by component |
| [`04_building_models`](../spine/04_building_models) | Building models with `Parameters`; swapping SFH families, dust laws, and IR templates |
| [`08_emission_lines`](../spine/08_emission_lines) | Nebular line fluxes, BPT diagnostics, Hα-derived SFR |

The pedagogical order in the spine is *SFH → dust → nebular → AGN →
multi-wavelength*, which mirrors the order of decisions a user typically
makes rather than the strict internal call order.

# Reproduction

The community has been doing galaxy SED inference for a long time. A
new code should earn trust by reproducing the outputs of the codes
already in use, band by band and component by component, before
claiming anything new.

This section collects those cross-checks. Each notebook walks one
external code block by block — stellar populations, star formation
history, dust attenuation, IR re-emission, nebular, IGM — putting that
code's output and tengri's on the same axes at matched parameters, and
reporting where they agree, where they disagree, and why. Each closes
with a full-SED head-to-head: tengri configured to emulate the external
code end to end, overlaid on its own panchromatic output with a residual
panel and an optical normalization ratio reported with its 16–84% spread.

The three notebooks share a layout, plotting style, and a `## Summary` /
`## References` close so they read as one series:

- **{doc}`cigale`** — CIGALE (Boquien et al. 2019). The widest stack:
  stellar, SFH, dust attenuation + Dale 2014 IR, nebular, AGN (SKIRTOR),
  X-ray, radio, and the Meiksin IGM.
- **{doc}`bagpipes`** — BAGPIPES (Carnall et al. 2018). JWST cosmic-noon
  focus: parametric and non-parametric SFHs, metallicity, the Inoue 2014
  IGM with the Asada 2025 CGM damping wing, SDSS photometry, and timing.
- **{doc}`prospector`** — Prospector / FSPS (Johnson et al. 2021). The
  core forward model: SSPs, delayed-τ SFH, Calzetti / Kriek & Conroy
  attenuation, Draine & Li 2007 IR, the Byler nebular grid, and Madau IGM.

```{toctree}
:maxdepth: 1

cigale
bagpipes
prospector
```

Comparisons with MAGPHYS, x-cigale, GRAHSP, and Synthesizer will land as
their reproduction notebooks come together.

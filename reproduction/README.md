# Reproduction studies

Component-by-component reproductions of tengri against the reference
SED-fitting codes the community already relies on. Each subfolder
holds one comparison.

```
reproduction/
├── cigale/                     # Boquien et al. 2019, A&A 622, A103
├── bagpipes/                   # Carnall et al. 2018, MNRAS 480, 4379
├── prospector/                 # Johnson et al. 2021, ApJS 254, 22
└── synthesizer/                # planned — Vijayan et al. 2024
```

Each comparison ships a notebook (`01_<code>.py`, jupytext percent
format), thin code-specific driver modules, and the rendered figures.
The CIGALE, BAGPIPES, and Prospector folders are ready; Synthesizer is
a scaffold.

## Shared conventions

The three ready notebooks are kept uniform so they read as one series:

- **Layout.** Each section sweeps one physics block — same SSP on both
  sides, external code (`C0-` solid) against tengri (`C1-`), shared
  `units.panel` / `units.two_panel_fig` axes labelled in erg/s/Hz. The
  per-code `_drivers/units.py` carry byte-identical `regrid`,
  `verify_unit_conversion`, `panel`, and `two_panel_fig`; only the
  luminosity converter and `L_SUN` constant differ between codes.
- **Helpers.** `save_fig` (one `_FIG_DPI`), `_assert_comparable(arr_ref,
  arr_t, *, name)`, `_norm_AV`, and `_bump_excess` are identical across
  the three.
- **Figures.** Every saved PNG is prefixed with its code —
  `cigale_*`, `bagpipes_*`, `prospector_*` — so the per-code `_figs/`
  never collide when aggregated.
- **Capstone.** Each notebook closes with a *full-SED head-to-head*
  section: tengri configured to emulate the external code end to end,
  overlaid on that code's own panchromatic output, with a fractional
  residual panel and an optical normalization ratio with its spread. Both end with a
  `## Summary` and a `## References` block.

## Running a notebook

```bash
cd reproduction/<code>
jupytext --to ipynb 01_<code>.py
jupyter nbconvert --to html --execute 01_<code>.ipynb
```

Prospector additionally needs `SPS_HOME` pointed at an FSPS checkout.
See the per-code README inside each subfolder for prerequisites and
data setup specific to that comparison.

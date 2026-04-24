# Citations still needing online lookup

This is the residue of the systematic citation audit. Everything listed below is
referenced in a docstring or component registration but could **not** be
confirmed verbatim in `~/writing-workspace/projects/tengri/99-references.bib`.
Do **not** invent metadata for these — look each up on NASA ADS and add it.

## Papers cited in tengri docstrings, not found in workspace .bib

| Label in tengri | Component / file | Minimum data we know | Action |
|---|---|---|---|
| Iyer et al. 2020 — stochastic SFH PSD | `components/sfh/psd_models.py` | MNRAS or arXiv 2020 | Need ADS bibcode. |
| Leja et al. 2019 — non-parametric SFH priors | various SFH docstrings | ApJ 2019 | Need ADS bibcode. |
| Salim et al. 2018 — dust attenuation curve | `dust/attenuation.py` | ApJ 2018 arXiv:1804.05850 (unverified) | Look up and add. |
| Nenkova et al. 2008 — CLUMPY torus | `components/agn/torus.py` | ApJ 2008 arXiv:0806.0512 (unverified) | Look up and add. |
| Vanden Berk et al. 2001 — SDSS quasar composite | `components/agn/blr.py` | AJ 122, 549 | Look up and add. |
| Murray, Adams & MacKay 2010 — elliptical slice sampling | `inference/backends/ess.py` | AISTATS 2010 | `Murray_2010` in the workspace is a *different* Murray paper on star formation — this one is not present. |
| Zhuang, Martínez-Ramírez et al. — AGNfitter-rx | `agn/cat3d_wind.py` | arXiv:2405.12111 (from tengri docstring) | Add once verified on ADS. |
| Richardson et al. 2021 — post-starburst spectroscopy | `agn/nlr.py` | arXiv not in docstring | Need ADS bibcode. |
| Draine & Li 2007 / Draine et al. 2014 — dust emission | `dust/emission.py` | 2007: ApJ 657, 810; 2014: ApJ 780, 172 | Look up and add. |
| Li et al. 2008 — clumpy medium attenuation | `dust/attenuation.py` | ApJ 686 | Look up; multiple Li 2008 entries likely exist. Verify title. |

## Misattributed in earlier registry (corrected in this pass)

- NIFTy software was pointed at `Arras_2022` (Nature Astronomy M87 imaging paper). The correct NIFTy.re / geoVI paper is `Edenhofer_2024` (JOSS, arXiv:2402.16683). Fixed.
- The tengri registry's `cue` previously had arXiv `2405.07657`; the real Cue paper is `Li_2024a`, arXiv:2405.04598. Fixed.
- BlackJAX arXiv corrected from `2402.00787` → `2402.10797` (`Cabezas_2024`).
- Inoue+2014 DOI corrected to `10.1093/mnras/stu936`.
- Madau 1995 DOI corrected to `10.1086/175332`.
- Johnson+2021 Prospector: workspace key is `Johnson_2021b` (the ApJS 254, 22 paper), not any `C_2021` entry.

## How to add a verified entry

1. Confirm on [NASA ADS](https://ui.adsabs.harvard.edu/) and copy the BibTeX
   export verbatim.
2. Append to `src/tengri/citations/references.bib` with:
   - a `registry_key = {<short_lowercase>}` line
   - a `category = {...}` line (one of `framework / ssp / sfh / dust_attenuation /
     dust_emission / nebular / agn / igm / preprocessing / inference /
     reference_code / other`)
   - a `short = {Author et al. (Year)}` line
   - a `role = {What tengri uses this for}` line
3. If it maps to a model-component value (dust law name, inference backend,
   …), add it to the relevant table in
   `src/tengri/citations/associations.py`.
4. Also append the exact ADS export to
   `~/writing-workspace/projects/tengri/99-references.bib` so Paper I's
   bibliography stays canonical.

Never invent DOIs, arXiv IDs, volume / page numbers.

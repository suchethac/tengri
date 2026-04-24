# Citations TODO

All 10 references flagged as "needs online lookup" in the previous audit have
been **verified on NASA ADS and added** to `src/tengri/citations/references.bib`
(commit `feat(citations): ADS-verify and wire the 10 previously-missing refs`).

See `docs/dev/CITATIONS_ADS_VERIFIED.md` for the per-paper ADS evidence URLs.

## Resolved (now in references.bib)

| Registry key            | Paper (ADS bibcode)                  | Role                                   |
|-------------------------|--------------------------------------|----------------------------------------|
| `salim2018`             | `2018ApJ...859...11S`                | Modified-Calzetti dust attenuation     |
| `iyer2020`              | `2020MNRAS.498..430I`                | PSD-governed SFH variability           |
| `leja2019`              | `2019ApJ...876....3L`                | Non-parametric SFH priors              |
| `clumpy_nenkova2008`    | `2008ApJ...685..147N`                | CLUMPY AGN torus (Paper I)             |
| `vandenberk2001`        | `2001AJ....122..549V`                | SDSS composite quasar spectrum         |
| `ess_murray2010`        | arXiv:1001.0175 (AISTATS 2010)       | Elliptical slice sampling              |
| `agnfitter_rx`          | `2024A&A...688A..46M`                | AGNfitter-rx peer SED-fitting code     |
| `draine_li2007`         | `2007ApJ...657..810D`                | Silicate-graphite-PAH dust emission    |
| `draine2014`            | `2014ApJ...780..172D`                | Updated Draine+Li templates            |
| `li2008_ext`            | `2008ApJ...685.1046L`                | Four-coefficient analytical extinction |

## Wired into `associations.py`

- `DustConfig.law_bc="salim"` → `salim2018` (was: misattributed to `noll2009`).
- `DustConfig.law_bc="li08"`  → `li2008_ext` (was: misattributed to `witt_gordon2000`).
- `DustConfig.emission="draine_li2007"` → `draine_li2007`.
- `DustConfig.emission="draine_li2014"` → `draine2014`.
- `AGNConfig.torus="clumpy"` (or `"nenkova"`) → `clumpy_nenkova2008`.
- `AGNConfig.blr="vanden_berk"` → `vandenberk2001`.
- `Fitter.run(backend="ess" | "elliptical_slice" | "mcmc_ess")` → `ess_murray2010`.

## Still unresolved

None of the named items are outstanding. Minor follow-up:

- `DustConfig.law_bc="vw07_bc"` / `"vw07_diff"` currently map to `witt_gordon2000`
  as a placeholder; the tengri source says Wild+2007 (`da Cunha & Charlot 2008`
  birth-cloud power-law). Wild+2007 is not in the workspace `.bib` and was not
  part of the batch-of-10 — leave as a noted placeholder and fix on the next
  ADS-verification pass.
- `DustConfig.law_bc="tea"` (Haskell+2024), `"narayanan_z"` (Narayanan+2018),
  `"hd23_mwrv31"` (Hensley & Draine 2023), `"prevot_smc"` (Prevot+1984),
  `"lmc"` (unspecified LMC curve), `"conroy2010"` (Conroy+2010 mixed MW+PL):
  each exists in tengri's dust module but has no citation mapped yet. Add on
  demand.

## Verification protocol reminder

Never add a citation without:
1. An authoritative ADS abstract page (or arXiv abs page for unpublished),
2. Exact URL recorded in `docs/dev/CITATIONS_ADS_VERIFIED.md`,
3. `registry_key` + `category` + `short` + `role` custom fields,
4. Matching entry in `src/tengri/citations/associations.py` (so the
   Bibliography machinery can surface the paper when the associated
   component value is set).

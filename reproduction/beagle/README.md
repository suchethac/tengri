# BEAGLE ↔ tengri

A component-by-component reproduction of BEAGLE 0.29.2 (Chevallard & Charlot 2016) against
tengri. The notebook is `reproduction/beagle/01_beagle.py`, rendered to `docs/reproduction/beagle.ipynb`.
BEAGLE is a binary fitter with its own templates, its own nebular grid and its own attenuation
curves, so the comparison is a validation round: the same physical inputs are pushed through both
codes, and each difference is classified as a library bracket, a convention or a physics offset.

BEAGLE 0.29.2 runs here, in its Docker image on its own licensed templates, and only its outputs
appear on the left-hand side; tengri's side uses public ingredients only and never reads a BEAGLE
template file. BEAGLE runs are executed once by the driver and cached outside the repository, so
rendering re-runs nothing, and no BEAGLE-derived data is committed: the figures and printed numbers
are the record.

## Prerequisites

- **A registered BEAGLE user.** Register individually on the BEAGLE homepage (https://www.iap.fr/beagle/),
  as section 2.3 of the BEAGLE Terms of Use requires. The repository does not contain the templates.
- **Docker Desktop** with amd64 emulation on Apple silicon. The image is pinned to
  `beagletool/beagle:0.29.2` and runs under `--platform linux/amd64`:

  ```bash
  docker pull --platform linux/amd64 beagletool/beagle:0.29.2
  ```

- **A BEAGLE working directory.** Clone the public `BEAGLE-general` repository, which holds `params/`,
  `filters/`, `dust/`, `data/`, `results/` and `Docker_env.list`, then point tengri at it:

  ```bash
  export BEAGLE_WORKDIR=~/beagle/BEAGLE-general
  ```

- **The templates**, fetched by you with your own credentials into `~/beagle/files/templates`:
  `bc03`, `cb_June16/mup100`, `ineb_June16/mup100` with `n2/C100`, and `AGN_NLR`. Keep the
  credentials in your shell session only; do not write them into a file in this repository.
- **Optional, for section 15b.** `export ART_SEDFITTING_DIR=...` names a local checkout of
  `camipacifici/art_sedfitting`, which holds the Paper I summaries.

## How the comparison is produced

`reproduction/beagle/_drivers/beagle_run.py` owns every BEAGLE run. It writes the parameter files,
invokes the containerized binary, parses the FITS output into tengri's units and caches the result.

```bash
python -m reproduction.beagle._drivers.beagle_run --render-params   # write every parameter file, no Docker
python -m reproduction.beagle._drivers.beagle_run --run-all         # run the mock matrix once and cache it
python -m reproduction.beagle._drivers.beagle_run --run s3_fiducial # run one section
python -m reproduction.beagle._drivers.beagle_run --status          # list which sections are cached
```

Each cached run lives under `$BEAGLE_WORKDIR/cache/`. The notebook never calls Docker: it reads the
cache, and when a section is missing it stops in Setup and prints the `--run` command that fills it.
`--fit-capstone` is reserved for the four Paper I fits. The current driver prints that no galaxy
identifiers are committed yet and runs nothing.

`reproduction/beagle/validate_matched_physics.py` is the strict companion check. It reads the cached
`s3_fiducial` and `s5_cf00_tauv` runs, builds the matching tengri models row by row and prints the
band-by-band ratio table. Its exit status is described under Sections.

## What is committed

- `data/beagle_workshop_catalog.h5`: the CANDELS GOODS-S photometry and the filter curves it uses.
- `reproduction/beagle/_drivers/`: the run driver, the run matrix, the output parser and the public
  analytic and table helpers.
- `reproduction/beagle/01_beagle.py` and its rendered notebook.
- `reproduction/beagle/validate_matched_physics.py` and its contract test.

No BEAGLE output, no template and no BEAGLE parameter file is committed. The cache, the results
directory and the templates stay in `$BEAGLE_WORKDIR`.

## Sections

- **Setup.** Backend guard, the SSP load, the catalog check, the cache check, and a provenance
  print: image tag, BEAGLE version, template names and cache directory.
- **§1 Stellar population.** BEAGLE's bc03 SSP against tengri's BC03 nodes at log age 7, 8 and 9 for
  Z = 0.02, as a library bracket; §1a sets BEAGLE's CB16 against its bc03 in a window table.
- **§2 Star-formation history.** The delayed history ψ(t), with SFR_10, SFR_100 and M_tot against
  tengri's delayed history with a 10 Myr current episode.
- **§3 Integrated SED.** The no-dust fiducial at z = 1, with nebular emission off on both sides,
  as overlay ratios and CANDELS band ratios.
- **§4 Attenuation curves.** The optical depth τ̂_λ = −ln(F_dust/F_nodust) of BEAGLE's CF00, Calzetti
  and CCWW13 runs, against tengri's power-law, calzetti and chevallard13 curves.
- **§5 Attenuation applied.** CF00 at τ_V = 0.3, 1 and 2 on the §3 history, as SED ratios and
  band ratios.
- **§8 Nebular emission.** The 48 gutkin16 mock objects: BEAGLE, tengri's gutkin16 and the raw
  table at the grid nodes; off-node behavior; trends in ionization parameter, metallicity and ξ_d;
  Hβ and ionizing-photon rates; the gutkin16, cue and cloudy backends; and the continuum in the
  line-free windows.
- **§9 AGN narrow-line region.** The Feltre block against the raw Feltre tables. A BEAGLE AGN run is
  compared only if a driver section exists for it; otherwise the comparison is at the table level.
- **§12 IGM.** Filter ratios with the Inoue (2014) transmission, integrated over the same top-hat
  bands for z = 1, 3 and 6.
- **§13 Where the codes differ.** One table of the conventions that separate the codes; the rows are
  listed below.
- **§14 Parameter translation.** BEAGLE keywords against tengri's grammar keys, with units and shifts.
- **§15 Capstone.** A BEAGLE-like fiducial mock with CANDELS photometry at z = 1, against tengri's
  BEAGLE-mode prediction. **§15b**, when `ART_SEDFITTING_DIR` is set, compares the Paper I fits with
  the workshop summaries and labels them fit-to-fit, with a different BEAGLE configuration.
- **Summary** and **References.**

## Where the codes differ

- **Stellar library.** BEAGLE reads the MILES-based `bc2003_hr_xmiless` family, and tengri reads the
  STELIB `bc03_pdva` grid. The difference is a library bracket, not a configuration mismatch, and
  it is the expected source of the few-percent residual in the blue and near-infrared.
- **Calzetti curve.** BEAGLE's `Calzetti` is Calzetti et al. (1994). tengri's `calzetti` is Calzetti
  et al. (2000), so the two curves differ by construction.
- **Milky Way curve.** BEAGLE's `mw`, `lmc` and `smc` are Pei (1992). tengri's `cardelli`, `lmc` and
  `smc` curves are the Cardelli et al. (1989) law and its LMC and SMC variants.
- **Nebular emission.** BEAGLE's nebular grid is the Gutkin et al. (2016) grid. tengri's Cue emulator
  is a different model, so the §8 comparison is made against tengri's gutkin16 backend, with the cue
  and cloudy backends shown as a sweep.
- **Solar metallicity.** BEAGLE uses Z⊙ = 0.01524 and tengri uses 0.0142. The shift
  `LOG10_ZSUN_SHIFT` moves BEAGLE's log Z/Z⊙ onto tengri's scale.
- **Solar luminosity.** BEAGLE uses L⊙ = 3.826 × 10³³ erg/s and tengri uses 3.828 × 10³³ erg/s.
- **Mass.** BEAGLE's `M_tot` is the formed mass, and tengri's formed `stellar_mass` matches it.
  BEAGLE's `M_star` is the surviving mass, which tengri names `stellar_mass_surviving`.
- **Specific SFR.** BEAGLE's sSFR is defined against M_tot, so a comparison of sSFR needs that
  normalization on both sides.
- **Noise floor.** BEAGLE's σ² floor is 0.02, and tengri's calibration term is `noise_frac_cal`.
- **Formation epoch.** BEAGLE's `formation_redshift` and tengri's `age_gyr` name the start of the
  history through different keys.
- **Prior ranges.** BEAGLE's τ prior reaches 31.6 Gyr and tengri's reaches 20 Gyr. BEAGLE's sSFR prior
  reaches −7 at the pole, and tengri stops at −7.3.
- **Air and vacuum.** BEAGLE writes wavelengths at or above 2000 Å in air, and the driver converts
  them once to vacuum. tengri is vacuum throughout.
- **Cosmology.** Both codes use Planck18 (H0 = 67.66, Ωm = 0.30966, ΩΛ = 0.69034). tengri's radiation
  terms move the luminosity distance by less than 10⁻⁴.
- **Ionization parameter.** tengri's `logU_S` is referred to the source, and BEAGLE's `logU` is
  referred to the inner face of the nebula.

## Terms of Use boundary

Section 6.2 of the BEAGLE Terms of Use bars using BEAGLE's models together with external software that
offers similar functionality. tengri therefore never reads a BEAGLE template, and BEAGLE's models run
only inside its Docker image; the page uses BEAGLE's outputs as a reference and nothing else. The
public nebular tables from Gutkin et al. (2016) and the CANDELS workshop photometry from Pacifici et al.
(2023) are published paper products, and tengri reads those.

## Citing

Section 2.2 of the BEAGLE Terms of Use asks for the citations below. Entries 1 and 2 apply to every
result on this page. The others apply when the page shows a result that depends on the named physics.

1. Chevallard J., Charlot S., 2016, MNRAS 462, 1415, doi:10.1093/mnras/stw1756, arXiv:1603.03037.
   Cite for every BEAGLE result.
2. Bruzual G., Charlot S., 2003, MNRAS 344, 1000, doi:10.1046/j.1365-8711.2003.06897.x; the June 2016
   update (Charlot & Bruzual, in preparation) as distributed with BEAGLE. Cite for every BEAGLE result.
3. Gutkin J., Charlot S., Bruzual G., 2016, MNRAS 462, 1757, doi:10.1093/mnras/stw1716,
   arXiv:1607.06086. Cite when nebular emission from the gutkin16 models is shown (§8, §15).
4. Calzetti D., Kinney A. L., Storchi-Bergmann T., 1994, ApJ 429, 582, doi:10.1086/174346; Calzetti D. et al.,
   2000, ApJ 533, 682, doi:10.1086/308692. Cite Calzetti et al. (1994) when BEAGLE's `Calzetti` curve is
   shown, and Calzetti et al. (2000) when tengri's `calzetti` curve is shown (§4, §5).
5. Charlot S., Fall S. M., 2000, ApJ 539, 718, doi:10.1086/309250. Cite when BEAGLE's CF00 attenuation is
   shown (§4, §5, §15).
6. Chevallard J., Charlot S., Wandelt B., Wild V., 2013, MNRAS 432, 2061, doi:10.1093/mnras/stt523. Cite
   when BEAGLE's CCWW13 attenuation is shown, together with entry 5 (§4).

Also cited:

7. Pei Y. C., 1992, ApJ 395, 130, doi:10.1086/171637.
8. Inoue A. K., Shimizu I., Iwata I., Tanaka M., 2014, MNRAS 442, 1805, doi:10.1093/mnras/stu936.
9. Feltre A., Charlot S., Gutkin J., 2016, MNRAS 456, 3354, doi:10.1093/mnras/stv2794.
10. Vidal-García A. et al., 2024, MNRAS 527, 7217, doi:10.1093/mnras/stad3252.
11. Plat A. et al., 2019, MNRAS 490, 978, doi:10.1093/mnras/stz2616.
12. Pacifici C. et al., 2023, ApJ 944, 141, doi:10.3847/1538-4357/acacff; repository camipacifici/art_sedfitting (MIT).
13. Planck Collaboration, 2016, A&A 594, A13, doi:10.1051/0004-6361/201525830.
14. Ferland G. J. et al., 2017, RMxAA 53, 385, arXiv:1705.10877.
15. Osterbrock D. E., Ferland G. J., 2006, Astrophysics of Gaseous Nebulae and Active Galactic Nuclei, 2nd ed.

## Validator

```bash
BEAGLE_WORKDIR=~/beagle/BEAGLE-general JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \
    .venv/bin/python reproduction/beagle/validate_matched_physics.py [--tol 0.02]
```

The validator needs the cache and the SSP grid, and it starts no container. It exits 0 when every band
of the s3 fiducial agrees within `--tol`, exits 1 when a band does not, and exits 2 when a prerequisite is
missing. The s5 table is printed and does not set the exit status. The validator reads
`data/bc03_pdva_stelib_chabrier.h5`; fetch it with `tengri.download_ssp('bc03_pdva_stelib_chabrier')`.

## Status

No component of this comparison has a row in `docs/dev/verification-protocol.md`, so the page assigns no
`VERIFIED`, `CROSSVAL`, `PARTIAL` or `NOT RUN` status. The record is the printed numbers. A band outside
`--tol` in the validator's output is a difference to classify, and the validator's exit status reports
that classification input only; it is not a verdict on either code.

# Citing tengri

If tengri shows up in a publication, cite the methods paper
(in preparation) and the upstream codes providing physics, grids, and
samplers. `tengri.print_components_bibtex(result)` prints BibTeX for every
model, SSP, and inference backend that ran in your fit, so the acknowledgement
stays in sync with the components you used. (`tengri.cite_all()` — no argument —
returns every citation registered in tengri, regardless of your fit.)

## The tengri methods papers

```bibtex
@ARTICLE{Cooray_2026,
  author  = {{Cooray}, Suchetha},
  title   = {{tengri}: Differentiable SED fitting with Information-Field-Theory
             star formation history priors. I. Framework and mock recovery},
  journal = {in preparation},
  year    = {2026}
}
```

Three papers are planned, and the citation keys are stable now even though
the metadata is not:

- **Paper I** — Cooray (2026), framework and mock recovery. Registry key
  `tengri`, BibTeX key `Cooray_2026`.
- **Paper II** — Cooray (2026), stochastic SFHs with IFT correlated-field
  priors and hierarchical population inference through geoVI. Registry key
  `tengri_paper2`, BibTeX key `Cooray_2026a`.
- **Paper III** — Cooray (2026), application to deep JWST surveys.
  Registry key `tengri_paper3`, BibTeX key `Cooray_2026b`.

DOIs, volumes, and ADS bibcodes are backfilled on acceptance. See
[CITATION.bib](https://github.com/suchethac/tengri/blob/main/CITATION.bib)
for the entries as shipped, or call `tengri.paper_citation()`.

## Acknowledgements

Tengri is built on a long stack of open-source astronomy and Bayesian
inference projects, and we are grateful to the authors and maintainers of
each one. They shape what tengri can do; we have just glued them together.

Everything below is registered in
[`src/tengri/citations/references.bib`](https://github.com/suchethac/tengri/blob/main/src/tengri/citations/references.bib),
which is the single source of truth — `tengri.cite("dsps")` returns the same
record this page quotes, and a CI guard fails the build when an entry in that
file has no acknowledgement here. Papers are grouped by the role they play in
the forward model, and listed oldest first within each group.

### Framework and inference

- [JAX](https://github.com/jax-ml/jax) — autodiff, JIT, and XLA compilation.
  Bradbury et al. (2018).
- [optax](https://github.com/google-deepmind/optax) — gradient-based
  optimizers behind the MAP routes.
- [NIFTy](https://github.com/NIFTy-PPL/NIFTy) — original NIFTy signal-inference
  library, foundation for the correlated-field SFH prior. Selig et al. (2013),
  [arXiv:1301.4499](https://arxiv.org/abs/1301.4499).
- Information Field Theory — the theoretical foundation for tengri's
  correlated-field SFH priors. Enßlin (2019),
  [arXiv:1804.03350](https://arxiv.org/abs/1804.03350).
- Bayesian Model Averaging. Hoeting et al. (1999),
  [doi:10.1214/ss/1009212519](https://doi.org/10.1214/ss/1009212519).
- Elliptical Slice Sampling — MCMC for Gaussian-prior models. Murray, Adams
  & MacKay (2010), [arXiv:1001.0175](https://arxiv.org/abs/1001.0175).
- Importance-sampling evidence estimation. Perrakis, Ntzoufras &
  Tsamardinos (2014), [arXiv:1106.5578](https://arxiv.org/abs/1106.5578).
- The Barker proposal — gradient-based MCMC robust to a step size that is
  wrong for one direction's scale. Livingstone & Zanella (2022),
  [arXiv:1908.11812](https://arxiv.org/abs/1908.11812).
- Optimal scaling of Langevin diffusions — the 0.574 acceptance rate the
  Barker and MALA backends tune to. Roberts & Rosenthal (1998),
  [doi:10.1111/1467-9868.00123](https://doi.org/10.1111/1467-9868.00123).
- Pathfinder — fast approximate-posterior sampler. Zhang et al. (2022),
  *Journal of Machine Learning Research* **23**(306), 1–49.
- [NIFTy.re](https://github.com/NIFTy-PPL/NIFTy) — information field theory,
  geoVI and MGVI. Edenhofer et al. (2024),
  [arXiv:2402.16683](https://arxiv.org/abs/2402.16683).
- [BlackJAX](https://github.com/blackjax-devs/blackjax) — NUTS, HMC, MCLMC
  and friends. Cabezas et al. (2024),
  [arXiv:2402.10797](https://arxiv.org/abs/2402.10797).
- Ray Tracing Sampler — noise-robust MCMC for high-dimensional fits.
  Behroozi (2025), [arXiv:2510.25824](https://arxiv.org/abs/2510.25824).
- Nested Slice Sampling — vectorized nested sampling for Bayesian evidence.
  Yallup, Kroupa & Handley (2026),
  [arXiv:2601.23252](https://arxiv.org/abs/2601.23252).

### Stellar populations, isochrones, and spectral libraries

- Starburst99 — ionizing-spectrum synthesis. Leitherer et al. (1999),
  [arXiv:astro-ph/9902334](https://arxiv.org/abs/astro-ph/9902334).
- BaSeL — theoretical stellar spectral library. Westera et al. (2002),
  [arXiv:astro-ph/0111175](https://arxiv.org/abs/astro-ph/0111175).
- BC03 — stellar population synthesis, used as a grid generator. Bruzual
  & Charlot (2003),
  [arXiv:astro-ph/0309134](https://arxiv.org/abs/astro-ph/0309134).
- STELIB — empirical stellar spectral library. Le Borgne et al. (2003),
  [arXiv:astro-ph/0302334](https://arxiv.org/abs/astro-ph/0302334).
- BaSTI — stellar evolution isochrones. Pietrinferni et al. (2004),
  [arXiv:astro-ph/0405193](https://arxiv.org/abs/astro-ph/0405193).
- Padova — stellar evolution isochrones. Marigo et al. (2008),
  [arXiv:0711.4922](https://arxiv.org/abs/0711.4922).
- [FSPS](https://github.com/cconroy20/fsps) — flexible stellar population
  synthesis, used as a grid generator. Conroy, Gunn & White (2009),
  [arXiv:0809.4261](https://arxiv.org/abs/0809.4261); and the reference
  implementation, Conroy & Gunn (2010),
  [arXiv:0911.3151](https://arxiv.org/abs/0911.3151).
- Carbon-star spectral library extending the FSPS TP-AGB stars redward of
  K. Aringer et al. (2009),
  [arXiv:0905.4415](https://arxiv.org/abs/0905.4415).
- MILES — empirical stellar spectral library. Falcón-Barroso et al. (2011),
  [arXiv:1107.2303](https://arxiv.org/abs/1107.2303).
- PARSEC — stellar evolution isochrones. Bressan et al. (2012),
  [arXiv:1208.4498](https://arxiv.org/abs/1208.4498).
- [python-fsps](https://github.com/dfm/python-fsps) — the Python interface
  used to generate the SSP grids. Foreman-Mackey et al. (2014),
  [doi:10.5281/zenodo.12157](https://doi.org/10.5281/zenodo.12157).
- Circumstellar AGB dust-shell SEDs in FSPS (`add_agb_dust_model`).
  Villaume et al. (2015),
  [arXiv:1504.00900](https://arxiv.org/abs/1504.00900).
- MIST — stellar isochrones. Choi et al. (2016),
  [arXiv:1604.08592](https://arxiv.org/abs/1604.08592); isochrone
  construction method, Dotter (2016),
  [arXiv:1601.05144](https://arxiv.org/abs/1601.05144).
- BPASS — binary stellar population and spectral synthesis. Eldridge,
  Stanway et al. (2017), [arXiv:1710.02154](https://arxiv.org/abs/1710.02154).
- [DSPS](https://github.com/ArgonneCPAC/dsps) — differentiable stellar
  population synthesis, the engine tengri evaluates SSPs with. Hearin et al.
  (2023), [arXiv:2112.06830](https://arxiv.org/abs/2112.06830).
- ProGeny — stellar population spectra generator. Robotham & Bellstedt
  (2025), [arXiv:2410.17697](https://arxiv.org/abs/2410.17697).

### Initial mass functions

- Salpeter (1955), [doi:10.1086/145971](https://doi.org/10.1086/145971).
- Kroupa (2001),
  [arXiv:astro-ph/0009005](https://arxiv.org/abs/astro-ph/0009005).
- Chabrier (2003),
  [arXiv:astro-ph/0304382](https://arxiv.org/abs/astro-ph/0304382).

### Star formation histories

- Dense Basis non-parametric SFH reconstruction. Iyer & Gawiser (2017),
  [arXiv:1702.04371](https://arxiv.org/abs/1702.04371).
- Prospector model description; source of the continuity SFH prior. Leja et
  al. (2017),
  [doi:10.3847/1538-4357/aa5ffe](https://doi.org/10.3847/1538-4357/aa5ffe).
- Non-parametric SFH priors — Dirichlet, continuity, and bursty continuity.
  Leja et al. (2019), [arXiv:1811.03637](https://arxiv.org/abs/1811.03637).
- Stochastic SFH power-spectral-density model for main-sequence scatter.
  Caplar & Tacchella (2019),
  [arXiv:1901.07556](https://arxiv.org/abs/1901.07556).
- Forensic SED reconstruction of cosmic star formation and metallicity
  history, by galaxy type. Bellstedt et al. (2020),
  [doi:10.1093/mnras/staa2620](https://doi.org/10.1093/mnras/staa2620).
- PSD-governed SFH variability in cosmological simulations. Iyer et al.
  (2020), [arXiv:2007.07916](https://arxiv.org/abs/2007.07916).

### Nebular emission

- MAPPINGS III — radiative shock and nebular models. Allen et al. (2008),
  [arXiv:0805.0204](https://arxiv.org/abs/0805.0204).
- [CLOUDY](https://gitlab.nublado.org/cloudy/cloudy) — the photoionization
  code behind the nebular grids. Ferland et al. (2017),
  [arXiv:1705.10877](https://arxiv.org/abs/1705.10877).
- FSPS nebular emission grids baked into the with-nebular SSP files. Byler
  et al. (2017), [arXiv:1611.08305](https://arxiv.org/abs/1611.08305).
- [Cue](https://github.com/yi-jia-li/cue) — neural emulator for nebular
  emission lines. Li et al. (2025),
  [arXiv:2405.04598](https://arxiv.org/abs/2405.04598).
- [Synthesizer](https://github.com/synthesizer-project/synthesizer) —
  synthetic observables package, source of the Cloudy AGN NLR/BLR grids and
  much nebular and SSP machinery. Lovell et al. (2025),
  [doi:10.33232/001c.145766](https://doi.org/10.33232/001c.145766); and the
  software paper, Roper et al. (2026),
  [doi:10.21105/joss.09436](https://doi.org/10.21105/joss.09436).

### Dust attenuation

- Milky-Way extinction curve (CCM). Cardelli et al. (1989),
  [doi:10.1086/167900](https://doi.org/10.1086/167900).
- Starburst attenuation law. Calzetti et al. (2000),
  [arXiv:astro-ph/9911459](https://arxiv.org/abs/astro-ph/9911459).
- Two-component birth-cloud plus ISM attenuation. Charlot & Fall (2000),
  [arXiv:astro-ph/0003128](https://arxiv.org/abs/astro-ph/0003128).
- Clumpy-medium radiative transfer attenuation. Witt & Gordon (2000),
  [arXiv:astro-ph/9907342](https://arxiv.org/abs/astro-ph/9907342).
- SMC, LMC, and MW extinction curves. Gordon et al. (2003),
  [arXiv:astro-ph/0305257](https://arxiv.org/abs/astro-ph/0305257).
- Four-coefficient analytical extinction curve, tengri's `li08` law. Li et
  al. (2008), [arXiv:0808.4115](https://arxiv.org/abs/0808.4115).
- Modified Calzetti with variable UV slope and bump, the CIGALE form. Noll
  et al. (2009), [arXiv:0909.5439](https://arxiv.org/abs/0909.5439).
- Variable UV-slope and UV-bump attenuation law. Kriek & Conroy (2013),
  [arXiv:1308.1099](https://arxiv.org/abs/1308.1099).
- High-redshift attenuation curve from MOSDEF. Reddy et al. (2015),
  [arXiv:1504.02782](https://arxiv.org/abs/1504.02782).
- Modified-Calzetti attenuation with variable slope, the DSPS default.
  Salim et al. (2018), [arXiv:1804.05850](https://arxiv.org/abs/1804.05850).

### Dust emission

- Optically-thin modified-blackbody dust emission. Hildebrand (1983).
- Luminosity-dependent IR SED templates. Chary & Elbaz (2001),
  [doi:10.1086/321609](https://doi.org/10.1086/321609).
- IR SED template family indexed by IR luminosity. Dale & Helou (2002),
  [doi:10.1086/341632](https://doi.org/10.1086/341632).
- Silicate-graphite-PAH dust emission. Draine & Li (2007),
  [arXiv:astro-ph/0608003](https://arxiv.org/abs/astro-ph/0608003).
- PAH emission feature spectra as Drude profiles. Smith et al. (2007),
  [arXiv:astro-ph/0610913](https://arxiv.org/abs/astro-ph/0610913).
- Thermal continuum emission from dust grains. Draine (2011),
  *Physics of the Interstellar and Intergalactic Medium*, Ch. 22.
- Modified-blackbody plus power-law FIR SED model. Casey (2012),
  [arXiv:1206.1595](https://arxiv.org/abs/1206.1595).
- CMB-heating and CMB-contrast corrections for high-redshift dust. da Cunha
  et al. (2013), [arXiv:1302.0844](https://arxiv.org/abs/1302.0844).
- THEMIS — amorphous-hydrocarbon dust foundation. Jones et al. (2013),
  [arXiv:1411.6293](https://arxiv.org/abs/1411.6293); and the global dust
  modeling framework, Jones et al. (2017),
  [arXiv:1703.00775](https://arxiv.org/abs/1703.00775).
- Two-parameter IR, submillimeter, and radio SED templates. Dale et al.
  (2014), [arXiv:1402.1495](https://arxiv.org/abs/1402.1495).
- Updated Draine & Li templates calibrated on Andromeda. Draine et al.
  (2014), [arXiv:1306.2304](https://arxiv.org/abs/1306.2304).
- Modified-blackbody plus PAH SED template. Schreiber et al. (2016),
  [arXiv:1601.02642](https://arxiv.org/abs/1601.02642); and the tabulated
  cold-dust library with real PAH features, Schreiber et al. (2018),
  [doi:10.1051/0004-6361/201731506](https://doi.org/10.1051/0004-6361/201731506).
- BOSA dust SED templates parameterized by total IR luminosity and specific
  SFR. Boquien & Salim (2021),
  [doi:10.1051/0004-6361/202140810](https://doi.org/10.1051/0004-6361/202140810).
- Warm plus cold two-temperature energy-balance dust split. Kokorev et al.
  (2021).
- PAHspec emission grid over grain size distribution, ionization, and
  starlight intensity. Draine et al. (2021),
  [doi:10.3847/1538-4357/abff51](https://doi.org/10.3847/1538-4357/abff51).
- Astrodust plus PAH unified grain model. Hensley & Draine (2023),
  [arXiv:2208.12365](https://arxiv.org/abs/2208.12365).

### AGN

- Standard thin accretion disc, multicolor blackbody. Shakura & Sunyaev
  (1973).
- Balmer continuum and 3000 Å bump. Grandi (1982).
- Empirical optical Fe II template from I Zw 1-like QSOs. Boroson & Green
  (1992), [doi:10.1086/191679](https://doi.org/10.1086/191679).
- ADAF scaling laws for low-luminosity AGN. Mahadevan (1997),
  [doi:10.1086/303727](https://doi.org/10.1086/303727).
- SDSS composite quasar spectrum, the BLR template reference. Vanden Berk
  et al. (2001),
  [arXiv:astro-ph/0105231](https://arxiv.org/abs/astro-ph/0105231).
- KYCONV relativistic convolution kernel for accretion disks. Dovciak et al.
  (2004), [arXiv:astro-ph/0403541](https://arxiv.org/abs/astro-ph/0403541).
- Optical Fe II empirical template from I Zw 1. Véron-Cetty, Joly & Véron
  (2004),
  [doi:10.1051/0004-6361:20035714](https://doi.org/10.1051/0004-6361:20035714).
- UV Fe II emission template from I Zw 1. Bruhweiler & Verner (2008),
  [doi:10.1086/525557](https://doi.org/10.1086/525557).
- CLUMPY — clumpy-medium AGN torus model. Nenkova et al. (2008),
  [arXiv:0806.0511](https://arxiv.org/abs/0806.0511).
- [SKIRTOR](https://sites.google.com/site/skirtorus/) — 3D AGN torus
  radiative transfer, isotropic case. Stalevski et al. (2012),
  [arXiv:1109.1286](https://arxiv.org/abs/1109.1286); and the dust covering
  factor treatment, Stalevski et al. (2016),
  [arXiv:1602.06954](https://arxiv.org/abs/1602.06954).
- Template torus with BLR and NLR lines. Mor & Netzer (2012),
  [doi:10.1111/j.1365-2966.2011.20060.x](https://doi.org/10.1111/j.1365-2966.2011.20060.x).
- Thin and slim accretion-disc bolometric relation. Netzer & Trakhtenbrot
  (2014), [arXiv:1311.4215](https://arxiv.org/abs/1311.4215).
- AGNSED — accretion disc with warm and hot Comptonization. Kubota & Done
  (2018), [arXiv:1804.00171](https://arxiv.org/abs/1804.00171).
- QSOgen — composite-quasar SED for BLR emission. Temple, Hewett & Banerji
  (2021),
  [doi:10.1093/mnras/stab2586](https://doi.org/10.1093/mnras/stab2586).
- RELAGN relativistic accretion disc with general-relativistic ray tracing.
  Hagen & Done (2023), [arXiv:2304.01253](https://arxiv.org/abs/2304.01253).
- GRAHSP — composable AGN and host SED model. Buchner et al. (2024),
  [arXiv:2405.19297](https://arxiv.org/abs/2405.19297).

### Radio and X-ray

- Radio synchrotron and free-free continuum from star formation. Condon
  (1992),
  [doi:10.1146/annurev.aa.30.090192.003043](https://doi.org/10.1146/annurev.aa.30.090192.003043).
- IR-radio correlation and SFR calibration. Bell (2003),
  [arXiv:astro-ph/0212121](https://arxiv.org/abs/astro-ph/0212121).
- X-ray binary luminosity scalings with stellar mass and SFR. Lehmer et al.
  (2016), [arXiv:1604.06461](https://arxiv.org/abs/1604.06461).
- X-ray AGN corona, the X-CIGALE alpha-ox relation. Yang et al. (2020),
  [doi:10.1093/mnras/stz3001](https://doi.org/10.1093/mnras/stz3001).

### IGM and the Galactic foreground

- IGM attenuation, retained as the legacy law. Madau (1995),
  [doi:10.1086/175332](https://doi.org/10.1086/175332).
- IGM transmission as implemented by CIGALE. Meiksin (2006),
  [arXiv:astro-ph/0512435](https://arxiv.org/abs/astro-ph/0512435).
- Voigt-Hjerting approximation for damped Lyman-alpha profiles.
  Tepper-García (2006),
  [arXiv:astro-ph/0602124](https://arxiv.org/abs/astro-ph/0602124).
- IGM Lyman-series attenuation. Inoue et al. (2014),
  [arXiv:1402.0677](https://arxiv.org/abs/1402.0677).
- CGM damping-wing transmission, which removes a photometric-redshift bias
  at z > 7. Asada et al. (2025),
  [arXiv:2410.21543](https://arxiv.org/abs/2410.21543).
- 3D Galactic dust map used for Milky-Way extinction preprocessing.
  Edenhofer et al. (2024),
  [arXiv:2308.01295](https://arxiv.org/abs/2308.01295).

### Photometry and spectral indices

- AB magnitude system and absolute spectrophotometric zero point. Oke &
  Gunn (1983), [doi:10.1086/160817](https://doi.org/10.1086/160817).
- Photon-counting AB magnitude definition, as used by the FSPS `getmags`
  routine. Fukugita et al. (1996),
  [doi:10.1086/117915](https://doi.org/10.1086/117915).
- Photon-counting bandpass AB flux and the K-correction formalism. Hogg et
  al. (2002),
  [arXiv:astro-ph/0210394](https://arxiv.org/abs/astro-ph/0210394).
- Photonic passbands and pivot wavelength. Bessell & Murphy (2012),
  [arXiv:1112.2698](https://arxiv.org/abs/1112.2698).
- Filter preintegration for differentiable photometry on GPUs. Zacharegkas
  et al. (2025), [arXiv:2506.19919](https://arxiv.org/abs/2506.19919).
- D4000 and Lick indices, and the BPT star-forming/AGN selection. Kauffmann
  et al. (2003),
  [arXiv:astro-ph/0205070](https://arxiv.org/abs/astro-ph/0205070).

### Comparison and reproduction

Tengri implements a superset of the union of the physics these codes
cover, so the reproduction studies configure the public dict API to mimic
each of them to the extent the models overlap. They are not ports, and
residual differences are expected.

- [BAGPIPES](https://github.com/ACCarnall/bagpipes) — reference SED fitting
  framework. Carnall et al. (2018),
  [arXiv:1712.04452](https://arxiv.org/abs/1712.04452).
- [CIGALE](https://cigale.lam.fr/) — reference panchromatic SED fitting
  code. Boquien et al. (2019),
  [arXiv:1811.03094](https://arxiv.org/abs/1811.03094). See
  [Reproduction → CIGALE](reproduction/cigale).
- [ProSpect](https://github.com/asgr/ProSpect) — SED generator with complex
  star formation and metallicity histories. Robotham et al. (2020),
  [arXiv:2002.06980](https://arxiv.org/abs/2002.06980).
- [Prospector](https://github.com/bd-j/prospector) — Bayesian SED inference
  on FSPS. Johnson et al. (2021),
  [arXiv:2012.01426](https://arxiv.org/abs/2012.01426). See
  [Reproduction → Prospector](reproduction/prospector).
- AGNfitter-rx — radio-to-X-ray AGN SED fitting. Martínez-Ramírez et al.
  (2024), [arXiv:2405.12111](https://arxiv.org/abs/2405.12111).

## Inheriting credit automatically

```python
import tengri
fitter = ...
result = fitter.run("mcmc_nuts")
tengri.print_components_bibtex(result)   # BibTeX for every component that ran
```

`print_components_bibtex` walks the model and inference call graph and emits
BibTeX for everything that contributed, including the SSP grid — the easiest
way to keep a paper's acknowledgements in sync with the fit. (For the table
form, use `tengri.cite_components(result)`.)

# Citation Audit for tengri

> **Archived, and partly wrong — do not copy citations out of this file.**
>
> This is a snapshot of the reference strings *as they appeared in the code*
> when the audit ran. Several of those strings were themselves fabricated, and
> at least two were copied outward from here into `src/`, `tests/` and
> `examples/` before anyone checked them. Known bad entries retained below for
> the record:
>
> * Entry 22 — "Li Z. et al. 2024, ApJ, 969, 28, *Cue: An Emulator for
>   AGN-Dominated Emission*" — no such paper. The real reference is Li, Yijia
>   et al. 2025, ApJ, **986**, 9, *"Cue: A Fast and Flexible Photoionization
>   Emulator for Modeling Nebular Emission Powered by Almost Any Ionizing
>   Source"*, arXiv:2405.04598, doi:10.3847/1538-4357/adcab4.
> * Line 624 — Feltre+2016 given as doi:10.1093/mnras/stw2180, which resolves
>   to an unrelated exoplanet-dynamics paper. The real DOI is
>   10.1093/mnras/stv2794.
>
> Corrected in #1801. The curated list is `src/tengri/citations/references.bib`.

## Summary

- **Total unique references found in code**: 158
- **Matched to workspace bibtex (99-references.bib)**: 95
- **References with DOI/arXiv in source (need online verification)**: 40
- **References not in workspace, unverifiable without DOI/arXiv**: 23

## 1. Verified References (Matched to Workspace)

The following 95 references are found in `<private-paper-drafts>`.

| Reference Label | BibTeX Key | File(s) |
|-----------------|-----------|---------|
| Calzetti, D., et al. 2000, ApJ 533, 682 Attenuation and dust... | `Calzetti_2000` | presets.py |
| Charlot, S., & Fall, S. M. 2000, ApJ 539, 718 A simple model... | `Charlot_2000` | presets.py |
| G. Calistro Rivera et al., "AGNfitter: A Bayesian MCMC Appro... | `He_2016` | agn_priors.py |
| Martínez-Ramírez / Zhuang et al. 2024, MNRAS 535, 2961 — the... | `Mart_2024` | agn_priors.py |
| B. D. Johnson et al., "Prospector: Stellar Population Infere... | `C_2021` | fitter.py |
| L. Zhang et al., "Pathfinder: Parallel quasi-Newton variatio... | `Zhang_2022` | fitter.py |
| A. Zacharegkas et al., "Fast Photometry with Precomputed Ste... | `A_2025` | sed_model.py |
| Y. Asada et al., "Improving Photometric Redshifts of Epoch o... | `A_2025` | sed_model.py |
| C. A. Mason et al., "The Universe Is Reionizing at z ~ 7: Ba... | `Mason_2018` | sed_model.py |
| C. Leitherer et al., "Starburst99: Synthesis Models for Gala... | `Leitherer_1999` | sed_model.py |
| Moustakas, J., Scholte, D., Dey, B., Khederlarian, A., 2023,... | `Li_2023` | line_list.py |
| Calzetti, D., Kinney, A. L., Storchi-Bergmann, T., 1994, ApJ... | `Calzetti_1994` | spectral.py |
| Balogh, M. L., Morris, S. L., Yee, H. K. C., Carlberg, R. G.... | `Balogh_1999` | spectral.py |
| Vollmann, K. & Eversberg, T., 2006, AN, 327, 862. Standard d... | `Vale_2006` | spectral.py |
| P. Madau, "Radiative Transfer in a Clumpy Universe: The Colo... | `Madau_1995` | igm.py |
| B. D. Johnson et al., "Stellar Population Inference," ApJS, ... | `C_2021` | igm.py |
| M. Boquien et al., "CIGALE," A&A, 622, A103 (2019). arXiv:18... | `A_2019` | skirtor.py |
| M. Nenkova et al., "AGN Dusty Tori. I. Handling of Clumpy Me... | `Li_2008` | torus.py |
| Lovell C. C. et al. 2025, Open Journal of Astrophysics, "Syn... | `A_2025` | unified.py |
| Roper W. J. et al. 2025, arXiv:2506.15811, "Synthesizer: Syn... | `A_2025` | unified.py |
| G. Yang et al. 2020, MNRAS, 491, 740, "X-CIGALE: Fitting AGN... | `Yang_2020` | unified.py |
| Li Z. et al. 2024, ApJ, 969, 28, "Cue: An Emulator for AGN-D... | `Li_2024` | unified.py |
| Y. Tsuzuki et al., "Very Large Array Imaging of Submillimete... | `Suzuki_2006` | blr.py |
| J. C. Richardson, et al., "Optical Spectroscopy of Post-Star... | `Lu_2014` | nlr.py |
| B. D. Johnson, et al., "Prospector: Inferring the Star Forma... | `C_2021` | nlr.py |
| S. F. Hönig & M. Kishimoto, "The dusty heart of nearby activ... | `D_2017` | cat3d_wind.py |
| J. M. Bardeen, W. H. Press, and S. A. Teukolsky, "Rotating b... | `Bardeen_1972` | disc.py |
| A. M. Beloborodov, ApJL, 510, L123 (1999). """ # ── Zone 1: ... | `Bo_1999` | disc.py |
| A. M. Beloborodov, "Plasma Ejection from Magnetic Flares and... | `Bo_1999` | disc.py |
| R. Mahadevan, "Scaling Laws for Advection-dominated Flows: A... | `Mahadevan_1997` | disc.py |

... and 65 more (see full list below)


## 2. References with DOI/arXiv in Source Code (Candidate for Online Verification)

These 40 references appear in tengri's docstrings with DOI or arXiv identifiers but are not found in the workspace bibtex file. A human should verify these online and add to the workspace file if appropriate.

| Reference Label | arXiv/DOI | File(s) | Status |
|-----------------|-----------|---------|--------|
| P. Behroozi, "The Ray Tracing Sampler," arXiv:2504.20029 (20... | 2504.20029 | fitter.py | needs_lookup |
| A. K. Inoue et al., "An updated analytic model for attenuati... | 10.1093/mnras/stu936 | sed_model.py | needs_lookup |
| A. K. Inoue, I. Shimizu, I. Iwata, and M. Tanaka, "An update... | 10.1093/mnras/stu936 | igm.py | needs_lookup |
| M. Stalevski et al., "3D radiative transfer modeling of the... | 10.1111/j.1365-2966.2011.19775.x | skirtor.py | needs_lookup |
| M. Stalevski et al., "The dust covering factor in AGN — comb... | 1602.01954 | skirtor.py | needs_lookup |
| M. Stalevski et al., "The dust covering factor in AGN," MNRA... | 1602.01954 | skirtor.py | needs_lookup |
| M. A. Vanden Berk et al., "The SDSS Quasar Catalog," AJ, 122... | 10.1086/321167 | blr.py | needs_lookup |
| A. N. Gaskell, J. E. Proga, M. A. Malkan, and Y. Gaskell, "I... | 10.1086/183869 | blr.py | needs_lookup |
| Z. Zhuang, Martínez-Ramírez et al., "AGNfitter-rx: ..." arXi... | 2405.12111 | cat3d_wind.py | needs_lookup |
| A. Kubota and C. Done, "A physical model of the broad-band c... | 1804.00171 | disc.py | needs_lookup |
| A. Laor and H. Netzer, "Massive thin accretion discs – I. Ca... | 10.1093/mnras/238.3.897 | disc.py | needs_lookup |
| C. Done et al., "Intrinsic disc emission and the soft X-ray ... | 10.1111/j.1365-2966.2011.19779.x | disc.py | needs_lookup |
| R. Nemmen et al., "Spectral models for low-luminosity active... | 10.1093/mnras/stt2388 | disc.py | needs_lookup |
| C. M. Gaskell et al., "A Redetermination of the Reddening of... | 10.1086/423885 | polar_dust.py | needs_lookup |
| M. Schartmann et al., "Three-dimensional radiative transfer ... | 10.1051/0004-6361:20042363 | disc_cigale.py | needs_lookup |
| C. Leitherer et al., "Global Far-Ultraviolet (912-1800 Å) Pr... | 10.1086/342486 | attenuation.py | needs_lookup |
| J. Calistro Rivera et al., "AGNfitter — A Bayesian MCMC appr... | 1808.04989 | attenuation.py | needs_lookup |
| C. Leitherer et al., "Global Far-Ultraviolet (912–1800 Å) Pr... | 10.1086/342486 | attenuation.py | needs_lookup |
| S. Noll, S. Pierini, B. Coles, et al., "On the link between ... | 10.1051/0004-6361/200912497 | attenuation.py | needs_lookup |
| A. Haskell, C. L. Steinhardt, C. Conselice, et al., "The Evo... | 2401.11007 | attenuation.py | needs_lookup |

... and 20 more


## 3. References Not in Workspace and Without DOI/arXiv

These 23 references appear in tengri's code without sufficient metadata to verify. These require either:
1. Adding the DOI/arXiv to the source docstring, OR
2. Manual lookup and addition to workspace bibtex

| Reference Label | Year | File(s) | Action Required |
|-----------------|------|---------|-----------------|
| D. W. Just et al., "The X-Ray Properties of the Most Luminou... | 1004 | agn_priors.py | add_metadata_or_lookup |
| Standardized parameterization derivation and Jacobian cancel... | None | loss_functions.py | add_metadata_or_lookup |
| M. D. Hoffman and A. Gelman, "The No-U-Turn Sampler: Adaptiv... | 1593 | fitter.py | add_metadata_or_lookup |
| S. Cooray et al., "Forward Model for Differentiable SED Fitt... | 2026 | sed_model.py | add_metadata_or_lookup |
| Steidel, C. C., et al., 1996, ApJ, 462, L17.... | 1996 | lines.py | add_metadata_or_lookup |
| M. Stalevski et al., MNRAS, 420, 2756 (2012). arXiv:1109.128... | 2756 | skirtor.py | add_metadata_or_lookup |
| synthesizer source: https://github.com/synthesizer-project/s... | None | unified.py | add_metadata_or_lookup |
| See :mod:`tengri.components.agn.disc` and :mod:`tengri.compo... | None | unified.py | add_metadata_or_lookup |
| Synthesizer ``torus_edgeon_condition``: https://github.com/s... | None | unified.py | add_metadata_or_lookup |
| See :mod:`tengri.components.agn` module documentation for ci... | None | _protocol.py | add_metadata_or_lookup |
| D. N. Page and K. S. Thorne, "Disk-Accretion onto a Black Ho... | 1974 | disc.py | add_metadata_or_lookup |
| A. Laor and B. Netzer, "Dust Sublimation Depth in the Infrar... | 1989 | disc.py | add_metadata_or_lookup |
| A. Kubota and C. Done, MNRAS, 480, 1247 (2018).... | 1247 | disc.py | add_metadata_or_lookup |
| G. B. Rybicki and A. P. Lightman, "Radiative Processes in As... | 1979 | disc.py | add_metadata_or_lookup |
| M. Planck, "Zur Theorie des Gesetzes der Energieverteilung i... | 1900 | _phys.py | add_metadata_or_lookup |
| L. Silva, R. Maiolino & G. L. Granato, "The nature of the Co... | 1068 | silva04.py | add_metadata_or_lookup |
| M. Prevot et al., "The Ultraviolet Extinction Curve in the S... | 1200 | attenuation.py | add_metadata_or_lookup |
| See :mod:`tengri.components.dust.attenuation` for specific d... | None | _protocol.py | add_metadata_or_lookup |
| See :mod:`tengri.components.dust.emission` for specific dust... | None | _protocol.py | add_metadata_or_lookup |
| V. Buat et al., "Far-Infrared Observations of Extremely Lumi... | 1989 | fesc_model.py | add_metadata_or_lookup |
| J. Chisholm et al., "The far-ultraviolet continuum slope as ... | 5104 | fesc_model.py | add_metadata_or_lookup |
| This protocol is used internally by tengri to enable swappab... | None | _protocol.py | add_metadata_or_lookup |
| H. Nussbaumer and W. Schmutz, "The two-photon continuum of H... | 1984 | _shared.py | add_metadata_or_lookup |


## 4. Complete List of Matched References

### All 95 Matched References

1. **Calzetti, D., et al. 2000, ApJ 533, 682 Attenuation and dust extinction in starb**  
   → BibTeX key: `Calzetti_2000`  
   → File: presets.py

2. **Charlot, S., & Fall, S. M. 2000, ApJ 539, 718 A simple model for the absorption **  
   → BibTeX key: `Charlot_2000`  
   → File: presets.py

3. **G. Calistro Rivera et al., "AGNfitter: A Bayesian MCMC Approach to Fitting Spect**  
   → BibTeX key: `He_2016`  
   → File: agn_priors.py

4. **Martínez-Ramírez / Zhuang et al. 2024, MNRAS 535, 2961 — the extended AGNfitter-**  
   → BibTeX key: `Mart_2024`  
   → File: agn_priors.py

5. **B. D. Johnson et al., "Prospector: Stellar Population Inference from Spectra and**  
   → BibTeX key: `C_2021`  
   → File: fitter.py

6. **L. Zhang et al., "Pathfinder: Parallel quasi-Newton variational inference," JMLR**  
   → BibTeX key: `Zhang_2022`  
   → File: fitter.py

7. **A. Zacharegkas et al., "Fast Photometry with Precomputed Stellar Population Grid**  
   → BibTeX key: `A_2025`  
   → File: sed_model.py

8. **Y. Asada et al., "Improving Photometric Redshifts of Epoch of Reionization Galax**  
   → BibTeX key: `A_2025`  
   → File: sed_model.py

9. **C. A. Mason et al., "The Universe Is Reionizing at z ~ 7: Bayesian Inference of **  
   → BibTeX key: `Mason_2018`  
   → File: sed_model.py

10. **C. Leitherer et al., "Starburst99: Synthesis Models for Galaxies with Active Sta**  
   → BibTeX key: `Leitherer_1999`  
   → File: sed_model.py

11. **Moustakas, J., Scholte, D., Dey, B., Khederlarian, A., 2023, "FastSpecFit: Fast **  
   → BibTeX key: `Li_2023`  
   → File: line_list.py

12. **Calzetti, D., Kinney, A. L., Storchi-Bergmann, T., 1994, ApJ, 429, 582. https://**  
   → BibTeX key: `Calzetti_1994`  
   → File: spectral.py

13. **Balogh, M. L., Morris, S. L., Yee, H. K. C., Carlberg, R. G., Ellingson, E., 199**  
   → BibTeX key: `Balogh_1999`  
   → File: spectral.py

14. **Vollmann, K. & Eversberg, T., 2006, AN, 327, 862. Standard definition of spectro**  
   → BibTeX key: `Vale_2006`  
   → File: spectral.py

15. **P. Madau, "Radiative Transfer in a Clumpy Universe: The Colors of High-Redshift **  
   → BibTeX key: `Madau_1995`  
   → File: igm.py

16. **B. D. Johnson et al., "Stellar Population Inference," ApJS, 254, 22 (2021). arXi**  
   → BibTeX key: `C_2021`  
   → File: igm.py

17. **M. Boquien et al., "CIGALE," A&A, 622, A103 (2019). arXiv:1811.03094. https://do**  
   → BibTeX key: `A_2019`  
   → File: skirtor.py

18. **M. Nenkova et al., "AGN Dusty Tori. I. Handling of Clumpy Media," ApJ, 685, 147 **  
   → BibTeX key: `Li_2008`  
   → File: torus.py

19. **Lovell C. C. et al. 2025, Open Journal of Astrophysics, "Synthesizer: a Software**  
   → BibTeX key: `A_2025`  
   → File: unified.py

20. **Roper W. J. et al. 2025, arXiv:2506.15811, "Synthesizer: Synthetic Observables F**  
   → BibTeX key: `A_2025`  
   → File: unified.py

21. **G. Yang et al. 2020, MNRAS, 491, 740, "X-CIGALE: Fitting AGN/galaxy SEDs from X-**  
   → BibTeX key: `Yang_2020`  
   → File: unified.py

22. **Li Z. et al. 2024, ApJ, 969, 28, "Cue: An Emulator for AGN-Dominated Emission", **  
   → BibTeX key: `Li_2024`  
   → File: unified.py

23. **Y. Tsuzuki et al., "Very Large Array Imaging of Submillimeter Galaxies," ApJ, 65**  
   → BibTeX key: `Suzuki_2006`  
   → File: blr.py

24. **J. C. Richardson, et al., "Optical Spectroscopy of Post-Starburst Galaxies," ApJ**  
   → BibTeX key: `Lu_2014`  
   → File: nlr.py

25. **B. D. Johnson, et al., "Prospector: Inferring the Star Formation Histories of Ga**  
   → BibTeX key: `C_2021`  
   → File: nlr.py

26. **S. F. Hönig & M. Kishimoto, "The dusty heart of nearby active galaxies. II. From**  
   → BibTeX key: `D_2017`  
   → File: cat3d_wind.py

27. **J. M. Bardeen, W. H. Press, and S. A. Teukolsky, "Rotating black holes: Locally **  
   → BibTeX key: `Bardeen_1972`  
   → File: disc.py

28. **A. M. Beloborodov, ApJL, 510, L123 (1999). """ # ── Zone 1: Outer standard disc **  
   → BibTeX key: `Bo_1999`  
   → File: disc.py

29. **A. M. Beloborodov, "Plasma Ejection from Magnetic Flares and the X-Ray Spectrum **  
   → BibTeX key: `Bo_1999`  
   → File: disc.py

30. **R. Mahadevan, "Scaling Laws for Advection-dominated Flows: Applications to Low-L**  
   → BibTeX key: `Mahadevan_1997`  
   → File: disc.py

31. **A. A. Zdziarski, G. M. Johnson, and M. Magdziarz, "Inverse Compton dominance in **  
   → BibTeX key: `Zdziarski_1996`  
   → File: _nthcomp.py

32. **D. Calzetti et al., "The Dust Content and Opacity of Actively Star-forming Galax**  
   → BibTeX key: `Calzetti_2000`  
   → File: polar_dust.py

33. **M. Boquien et al., "CIGALE: a python Code Investigating GALaxy Emission," A&A, 6**  
   → BibTeX key: `A_2019`  
   → File: polar_dust.py

34. **I. E. Lopez et al., "Modeling the X-ray emission of AGN in CIGALE and applicati**  
   → BibTeX key: `Li_2024`  
   → File: disc_cigale.py

35. **G. Calistro Rivera et al., "AGNfitter: a Bayesian MCMC approach to fitting spect**  
   → BibTeX key: `Li_2016`  
   → File: silva04.py

36. **M. Kriek and C. Conroy, "The Dust Attenuation Law in Distant Galaxies: Evidence **  
   → BibTeX key: `Conroy_2013`  
   → File: attenuation.py

37. **S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by Du**  
   → BibTeX key: `Charlot_2000`  
   → File: attenuation.py

38. **V. Wild, S. Charlot, and P. Diminic, "CANDELS/CDF-S: Unveiling the Nature of Dis**  
   → BibTeX key: `Li_2007`  
   → File: attenuation.py

39. **S. Calzetti et al., "The Dust Content and Opacity of Star-Forming Galaxies," ApJ**  
   → BibTeX key: `Calzetti_2000`  
   → File: attenuation.py

40. **P. G. Pei, "Interstellar Dust from the Ultraviolet to the Infrared," ApJ, 395, 1**  
   → BibTeX key: `Pei_1992`  
   → File: attenuation.py

41. **D. E. Cardelli, G. C. Clayton, and J. S. Mathis, "The Relationship between Infra**  
   → BibTeX key: `Cardelli_1989`  
   → File: attenuation.py

42. **S. Salim, M. Boquien, and J. C. Lee, "CANDELS: Constraining the AGN Contribution**  
   → BibTeX key: `Lee_2018`  
   → File: attenuation.py

43. **D. Narayanan, K. Kriek, C. C. Hayward, et al., "A Theory for the Variation of Du**  
   → BibTeX key: `Conroy_2018`  
   → File: attenuation.py

44. **C. Conroy, R. H. White, and J. S. Gunn, "Recovering the Intergalactic Dust from **  
   → BibTeX key: `Conroy_2010`  
   → File: attenuation.py

45. **S. Lower et al., "How Well Can We Measure Galaxy Dust Attenuation Curves? The Im**  
   → BibTeX key: `Lower_2022`  
   → File: attenuation.py

46. **A. N. Witt and K. D. Gordon, "Multiple Scattering in Clumpy Media. I. Point Sour**  
   → BibTeX key: `Gordon_2000`  
   → File: attenuation.py

47. **D. Calzetti, A. L. Kinney, and T. Storchi-Bergmann, "Dust Extinction of the Stel**  
   → BibTeX key: `Calzetti_1994`  
   → File: attenuation.py

48. **A. N. Witt and K. D. Gordon, "Multiple Scattering in Clumpy Media. II. Galactic **  
   → BibTeX key: `Gordon_2000`  
   → File: attenuation.py

49. **S. Rémy-Ruyer, I. Miville-Deschênes, T. Siebel, et al., "Dust and Gas Relationsh**  
   → BibTeX key: `An_2014`  
   → File: attenuation.py

50. **J. C. Weingartner & B. T. Draine, "Dust Grain-Size Distributions and Extinction **  
   → BibTeX key: `Draine_2001`  
   → File: attenuation.py

51. **B. T. Draine, "Interstellar Dust Grains," ARA&A, 41, 241 (2003). arXiv:astro-ph/**  
   → BibTeX key: `Draine_2003`  
   → File: attenuation.py

52. **B. S. Hensley & B. T. Draine, "The Astrodust+PAH Model: A Unified Description of**  
   → BibTeX key: `Hensley_2023`  
   → File: attenuation.py

53. **D. M. Smith et al., "The Mid-Infrared Emission of Ultraluminous Infrared Galaxie**  
   → BibTeX key: `Smith_2007`  
   → File: drude_profiles.py

54. **Smith, J. D. T., Draine, B. T., Dale, D. A., et al., 2007, ApJ, 656, 770 (PAH pr**  
   → BibTeX key: `Draine_2007`  
   → File: emission.py

55. **V. Kokorev et al., "STARDUST: Spectral Template Analysis and Recovery of Dust an**  
   → BibTeX key: `C_2021`  
   → File: emission.py

56. **C. Johnson et al., "Prospector: Bayesian Stellar Population Inference with Separ**  
   → BibTeX key: `C_2021`  
   → File: dsps_wrapper.py

57. **G. R. Meurer, T. M. Heckman, and D. Calzetti, "Dust Absorption and the Ultraviol**  
   → BibTeX key: `Meurer_1999`  
   → File: fesc_model.py

58. **R. P. Naidu et al., "Rapid Reionization by the Oligarchs: The Case for Massive, **  
   → BibTeX key: `J_2020`  
   → File: fesc_model.py

59. **D. A. Allen et al., "The Distance and Metallicity of the Galaxy M33," ApJS, 178,**  
   → BibTeX key: `Allen_2008`  
   → File: shock.py

60. **R. J. R. Sutherland and M. A. Dopita, "Spectral Synthesis Modeling of AGN Heati**  
   → BibTeX key: `Sutherland_2017`  
   → File: shock.py

61. **D. Alarie and C. Morisset, "Synthetic Narrow-Line Emission from a Large Grid of **  
   → BibTeX key: `A_2019`  
   → File: shock.py

62. **D. E. Osterbrock and G. J. Ferland, "Astrophysics of Gaseous Nebulae and Active **  
   → BibTeX key: `Coe_2006`  
   → File: cloudy_cb19.py

63. **M. Li et al., "The Cue Nebular Emulator: Fast, Interpretable Predictions of Emis**  
   → BibTeX key: `A_2025`  
   → File: ionizing_spectrum.py

64. **N. Byler et al., "Nebular Continuum and Line Emission in Stellar Population Synt**  
   → BibTeX key: `Byler_2017`  
   → File: cloudy_grid.py

65. **B. T. Draine, "Physics of the Interstellar and Intergalactic Medium" (Princeton **  
   → BibTeX key: `Draine_2011`  
   → File: _shared.py

66. **V. Luridiana, C. Morisset, and R. A. Shaw, "PyNeb: A Python Package for Analysin**  
   → BibTeX key: `Ho_2015`  
   → File: _shared.py

67. **L. M. Haffner et al., "The warm ionized medium in spiral galaxies," Rev. Mod. Ph**  
   → BibTeX key: `O_2009`  
   → File: dig.py

68. **Lovell et al. 2025, MNRAS (Synthesizer; arXiv:2004.07283).**  
   → BibTeX key: `A_2025`  
   → File: agn_nebular.py

69. **Flury et al. 2024, "MAPPINGS V photoionization grids for nebular emission predic**  
   → BibTeX key: `Ho_2024`  
   → File: mappings_photo.py

70. **R. S. Sutherland & M. A. Dopita 2017, "Effects of Preionization in Radiative Sho**  
   → BibTeX key: `Sutherland_2017`  
   → File: mappings_photo.py

71. **Li et al. 2025, "Cue: A fast neural network emulator for nebular emission line a**  
   → BibTeX key: `A_2025`  
   → File: cue.py

72. **Charlot & Fall 2000, "A simple model for the absorption of starlight by dust gra**  
   → BibTeX key: `Charlot_2000`  
   → File: cue.py

73. **S. Tacchella et al., "Star Formation Histories from SEDs and Spectra," ApJ, 926,**  
   → BibTeX key: `Pe_2022`  
   → File: nonparametric.py

74. **J. Leja et al., "How to Measure Galaxy Star Formation Histories. II. Nonparametr**  
   → BibTeX key: `A_2019`  
   → File: nonparametric.py

75. **J. Leja et al., "Deriving Physical Properties from Broadband Photometry with Pro**  
   → BibTeX key: `D_2017`  
   → File: nonparametric.py

76. **J. Leja et al., "How to Measure Galaxy Star Formation Histories, I. Parametric M**  
   → BibTeX key: `A_2019`  
   → File: nonparametric.py

77. **B. D. Johnson et al., "Stellar Population Inference from the Spectral Energy Dis**  
   → BibTeX key: `C_2021`  
   → File: nonparametric.py

78. **K. Iyer and E. Gawiser, "Reconstruction of Galaxy Star Formation Histories throu**  
   → BibTeX key: `D_2017`  
   → File: dense_basis.py

79. **K. Iyer et al., "Nonparametric Star Formation History Reconstruction with Gaussi**  
   → BibTeX key: `A_2019`  
   → File: dense_basis.py

80. **Selig et al., "NIFTY - Numerical Information Field Theory in Python," A&A, 554, **  
   → BibTeX key: `Li_2013`  
   → File: gp_sfh.py

81. **A. S. G. Robotham et al., "ProSpect: generating spectral energy distributions wi**  
   → BibTeX key: `Pe_2020`  
   → File: mean_sfh.py

82. **P. S. Behroozi, R. H. Wechsler, C. Conroy, "The Average Star Formation Histories**  
   → BibTeX key: `Be_2013`  
   → File: mean_sfh.py

83. **G. Zacharegkas, A. Hearin, and A. Benson, "Bayesian Posteriors with Stellar Popu**  
   → BibTeX key: `A_2025`  
   → File: mean_sfh.py

84. **L. Ciesla et al., "The SFR-M* main sequence archetypal star-formation history an**  
   → BibTeX key: `Ciesla_2017`  
   → File: mean_sfh.py

85. **C. B. Moler, "Numerical Computing with MATLAB," SIAM, Ch. 3 (2004).**  
   → BibTeX key: `Ma_2004`  
   → File: mean_sfh.py

86. **B. Tinsley, "Fundamentals of Cosmic Physics," in Fundamentals of Cosmic Physics,**  
   → BibTeX key: `Tinsley_1980`  
   → File: chemical_evolution.py

87. **Rasmussen & Williams, "Gaussian Processes for Machine Learning," MIT Press (2006**  
   → BibTeX key: `Rasmussen_2006`  
   → File: psd_models.py

88. **S. Tacchella et al., "A Redshift-independent Efficiency Model: Star Formation an**  
   → BibTeX key: `Ma_2018`  
   → File: psd_models.py

89. **H.-J. Grimm et al., "High-mass X-ray binaries as a star formation rate indicator**  
   → BibTeX key: `Grimm_2003`  
   → File: xray.py

90. **M. Gilfanov, "Low-mass X-ray binaries as a stellar mass indicator for the host g**  
   → BibTeX key: `Ma_2004`  
   → File: xray.py

91. **G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with CIGALE and improvem**  
   → BibTeX key: `Yan_2022`  
   → File: xray.py

92. **I. E. Lopez et al., "IRX-CIGALE: a tailored module for Low-Luminosity AGN," A&A,**  
   → BibTeX key: `Lu_2024`  
   → File: xray.py

93. **P. Gandhi et al., "Resolving the mid-infrared cores of local Seyferts," A&A, 502**  
   → BibTeX key: `O_2009`  
   → File: xray.py

94. **D. Asmus et al., "Local AGN survey (LASr): I. Galaxy sample, infrared and X-ray **  
   → BibTeX key: `Gal_2015`  
   → File: xray.py

95. **Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3), 705-767. ""**  
   → BibTeX key: `He_2003`  
   → File: slice_sampling.py



## 5. Complete List of References with DOI/arXiv

### All 40 References with Metadata Hints

1. **P. Behroozi, "The Ray Tracing Sampler," arXiv:2504.20029 (2025). https://arxiv.o**  
   → Hint: arXiv:2504.20029  
   → File: fitter.py

2. **A. K. Inoue et al., "An updated analytic model for attenuation by the intergalac**  
   → Hint: DOI:10.1093/mnras/stu936  
   → File: sed_model.py

3. **A. K. Inoue, I. Shimizu, I. Iwata, and M. Tanaka, "An updated analytic model for**  
   → Hint: DOI:10.1093/mnras/stu936  
   → File: igm.py

4. **M. Stalevski et al., "3D radiative transfer modeling of the dusty torus around **  
   → Hint: DOI:10.1111/j.1365-2966.2011.19775.x  
   → File: skirtor.py

5. **M. Stalevski et al., "The dust covering factor in AGN — combining the IR torus e**  
   → Hint: arXiv:1602.01954 | DOI:10.1093/mnras/stw444  
   → File: skirtor.py

6. **M. Stalevski et al., "The dust covering factor in AGN," MNRAS, 458, 2288 (2016).**  
   → Hint: arXiv:1602.01954 | DOI:10.1093/mnras/stw444  
   → File: skirtor.py

7. **M. A. Vanden Berk et al., "The SDSS Quasar Catalog," AJ, 122, 549 (2001). arXiv:**  
   → Hint: DOI:10.1086/321167  
   → File: blr.py

8. **A. N. Gaskell, J. E. Proga, M. A. Malkan, and Y. Gaskell, "Iron emission in Seyf**  
   → Hint: DOI:10.1086/183869  
   → File: blr.py

9. **Z. Zhuang, Martínez-Ramírez et al., "AGNfitter-rx: ..." arXiv:2405.12111. """**  
   → Hint: arXiv:2405.12111  
   → File: cat3d_wind.py

10. **A. Kubota and C. Done, "A physical model of the broad-band continuum of AGN and **  
   → Hint: arXiv:1804.00171 | DOI:10.1093/mnras/sty1890  
   → File: disc.py

11. **A. Laor and H. Netzer, "Massive thin accretion discs – I. Calculated spectra," M**  
   → Hint: DOI:10.1093/mnras/238.3.897  
   → File: disc.py

12. **C. Done et al., "Intrinsic disc emission and the soft X-ray excess in active gal**  
   → Hint: DOI:10.1111/j.1365-2966.2011.19779.x  
   → File: disc.py

13. **R. Nemmen et al., "Spectral models for low-luminosity active galactic nuclei in **  
   → Hint: DOI:10.1093/mnras/stt2388  
   → File: disc.py

14. **C. M. Gaskell et al., "A Redetermination of the Reddening of AGNs," ApJ, 616, 14**  
   → Hint: DOI:10.1086/423885  
   → File: polar_dust.py

15. **M. Schartmann et al., "Three-dimensional radiative transfer models of clumpy tor**  
   → Hint: DOI:10.1051/0004-6361:20042363  
   → File: disc_cigale.py

16. **C. Leitherer et al., "Global Far-Ultraviolet (912-1800 Å) Properties of Star-for**  
   → Hint: DOI:10.1086/342486  
   → File: attenuation.py

17. **J. Calistro Rivera et al., "AGNfitter — A Bayesian MCMC approach to fitting spec**  
   → Hint: arXiv:1808.04989 | DOI:10.3847/1538-4357/aad235  
   → File: attenuation.py

18. **C. Leitherer et al., "Global Far-Ultraviolet (912–1800 Å) Properties of Star-for**  
   → Hint: DOI:10.1086/342486  
   → File: attenuation.py

19. **S. Noll, S. Pierini, B. Coles, et al., "On the link between galaxy morphology an**  
   → Hint: DOI:10.1051/0004-6361/200912497  
   → File: attenuation.py

20. **A. Haskell, C. L. Steinhardt, C. Conselice, et al., "The Evolution of the Dust A**  
   → Hint: arXiv:2401.11007  
   → File: attenuation.py

21. **A. Natta and N. Panagia, "Extinction in inhomogeneous clouds," ApJ, 287, 228 (19**  
   → Hint: DOI:10.1086/162686  
   → File: attenuation.py

22. **A. Natta and N. Panagia, "Extinction in Inhomogeneous Clouds," ApJ, 287, 228 (19**  
   → Hint: DOI:10.1086/162686  
   → File: attenuation.py

23. **M. P. Hobson and L. Padman, "A Probabilistic Approach to Extinction in Irregular**  
   → Hint: DOI:10.1093/mnras/264.1.161  
   → File: attenuation.py

24. **S. C. Wilkins et al., "Synthesizer: A Python Package for Generating Synthetic Ga**  
   → Hint: arXiv:2508.03888  
   → File: attenuation.py

25. **Schreiber, C., Elbaz, D., Sparre, M., et al., 2016, A&A, 589, A35 (https://doi.o**  
   → Hint: DOI:10.1051/0004-6361/201527923  
   → File: emission.py

26. **E. da Cunha et al., "MAGPHYS: a new code to compute and interpret the Spectral E**  
   → Hint: DOI:10.1111/j.1365-2966.2008.13535.x  
   → File: emission.py

27. **M. Martinez-Paredes et al., "The 3MdB stellar and AGN library of CLOUDY models,"**  
   → Hint: arXiv:2308.05604 | DOI:10.1093/mnras/stad3891  
   → File: cloudy_cb19.py

28. **P. J. Storey and D. G. Hummer, "Recombination coefficients for H II and HeII," M**  
   → Hint: DOI:10.1093/mnras/272.1.41  
   → File: _shared.py

29. **S. Tacchella et al., "H-alpha emission in local galaxies: star formation, time v**  
   → Hint: arXiv:2112.00027 | DOI:10.1093/mnras/stac818  
   → File: dig.py

30. **S. P. Reynolds, "Supernova Remnants as Cosmic Ray Sources," ApJ, 282, 191 (1984)**  
   → Hint: DOI:10.1086/162189  
   → File: dig.py

31. **A. Feltre, S. Charlot, and J. Gutkin, "Updated photoionization models of the CLO**  
   → Hint: arXiv:1511.08217 | DOI:10.1093/mnras/stw2180  
   → File: agn_nebular.py

32. **S. Wang et al., "Prospector-β," arXiv:2401.12198 (2024).**  
   → Hint: arXiv:2401.12198  
   → File: nonparametric.py

33. **K. A. Suess et al., "Half-mass Radii for ~7000 Galaxies," ApJ, 915, 87 (2021). a**  
   → Hint: arXiv:2101.03177 | DOI:10.3847/1538-4357/ac062c  
   → File: nonparametric.py

34. **F. N. Fritsch and R. E. Carlson, "Monotone Piecewise Cubic Interpolation," SIAM **  
   → Hint: DOI:10.1137/0717021  
   → File: dense_basis.py

35. **S. Bellstedt et al., "Galaxy And Mass Assembly (GAMA): a forensic SED reconstruc**  
   → Hint: arXiv:2005.11917 | DOI:10.1093/mnras/staa2620  
   → File: mean_sfh.py

36. **A. C. Carnall et al., "Inferring the star formation histories of massive quiesce**  
   → Hint: arXiv:1712.04452 | DOI:10.1093/mnras/sty2169  
   → File: mean_sfh.py

37. **V. Buat et al., "Star formation history of galaxies from z = 0 to z = 0.7. A bac**  
   → Hint: DOI:10.1051/0004-6361:20078829  
   → File: mean_sfh.py

38. **J. B. Munoz and K. G. Iyer, "Measuring the Power Spectral Density of Star Format**  
   → Hint: arXiv:2601.07912  
   → File: psd_models.py

39. **N. Caplar and S. Tacchella, "Stochastic modeling of star-formation histories I:**  
   → Hint: arXiv:1901.07556 | DOI:10.1093/mnras/stz1449  
   → File: psd_models.py

40. **K. Duras et al., "Universal bolometric corrections for active galactic nuclei ov**  
   → Hint: DOI:10.1051/0004-6361/201936817  
   → File: xray.py


## Notes

- This audit was generated by scanning all Python files in `src/tengri/` for numpydoc-style `.. [N] Author, Year, ...` references.
- Matching to workspace bibtex was done using BibTeX key patterns (Author_Year).
- References with DOI/arXiv in the source should be verified against ADS or arXiv to obtain complete citations for workspace bibtex.
- References marked "unverifiable" lack sufficient metadata and should be updated in the source code with DOI/arXiv if available, or added to workspace bibtex manually.

---

Generated: 2026-04-23


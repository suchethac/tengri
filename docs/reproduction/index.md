# Physics reproduction

A new code should earn trust by reproducing the physics in the codes
already in use before claiming anything new. The notebooks here do that
for tengri. Each one implements the same models as one external code and
puts the two on the same axes at matched parameters, block by block:
stellar populations, star formation history, dust, nebular, AGN, IGM.
Where they agree, that is stated; where they differ, the reason is
tracked down. Having the same physics in one framework also means every
assumption can be checked and different models compared directly. Each
notebook ends with a full-SED head-to-head with a residual panel.

- **{doc}`cigale`**: CIGALE (Boquien et al. 2019). The widest stack:
  stellar, SFH, dust attenuation with Dale 2014 IR, nebular, AGN
  (SKIRTOR), X-ray, radio, and the Meiksin IGM.
- **{doc}`bagpipes`**: BAGPIPES (Carnall et al. 2018). JWST cosmic-noon
  focus: parametric and non-parametric SFHs, metallicity, and the Inoue
  2014 IGM with the Asada 2025 CGM damping wing.
- **{doc}`prospector`**: Prospector / FSPS (Johnson et al. 2021). The
  core forward model: SSPs, delayed-τ SFH, Calzetti and Kriek & Conroy
  attenuation, Draine & Li 2007 IR, the Byler nebular grid, and Madau
  IGM.
- **{doc}`agnfitter`**: AGNfitter-RX (Martínez-Ramírez et al. 2024). An
  AGN-first, radio-to-X-ray deep dive: four accretion-disk libraries
  (R06, SN12, KD18, THB21) and four torus libraries (S04, NK08, SKIRTOR,
  CAT3D-Wind) head to head, plus the X-ray corona and radio jets.
- **{doc}`prospect_r`**: ProSpect (Robotham et al. 2020), the R-based
  GAMA code, driven live through `rpy2`: BC03 SSPs, the skew-normal SFH,
  a metallicity history tied to the stellar mass formed, Charlot & Fall
  attenuation, Dale 2014 IR, emission lines, the SKIRTOR torus, a radio
  continuum, and the Inoue 2014 IGM.
```{toctree}
:maxdepth: 1

cigale
bagpipes
prospector
agnfitter
prospect_r
```

Comparisons with BEAGLE, MAGPHYS, GRAHSP, and GalaPy are scoped on the
[issue tracker](https://github.com/suchethac/tengri/issues) and will
land as their notebooks come together.

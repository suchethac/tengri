# Forward model

The differentiable forward chain follows paper **§3** (SFH → SPS → nebular/AGN → dust → IGM → observables). **Narrative notebooks** at the repo root (Jupytext `.py`):

| Notebook | Focus |
|----------|-------|
| [`01_sed_anatomy.py`](https://github.com/suchethac/tengri/blob/main/notebooks/01_sed_anatomy.py) | SED anatomy, wavelength ↔ physics |
| [`02_sfh_gallery.py`](https://github.com/suchethac/tengri/blob/main/notebooks/02_sfh_gallery.py) | SFH parametrization + stochastic field / PSD |
| [`13_tabulated_sfh_to_mock_sed.py`](https://github.com/suchethac/tengri/blob/main/notebooks/13_tabulated_sfh_to_mock_sed.py) | Tabulated SFH → SED / mock photometry (`simulate`) |
| [`03_dust_gallery.py`](https://github.com/suchethac/tengri/blob/main/notebooks/03_dust_gallery.py) | Dust attenuation + IR emission |
| [`04_nebular_gallery.py`](https://github.com/suchethac/tengri/blob/main/notebooks/04_nebular_gallery.py) | Nebular |
| [`05_agn_gallery.py`](https://github.com/suchethac/tengri/blob/main/notebooks/05_agn_gallery.py) | AGN |
| [`06_multiwavelength_gallery.py`](https://github.com/suchethac/tengri/blob/main/notebooks/06_multiwavelength_gallery.py) | IGM, radio, X-ray |

Pedagogical order in the spine is **SFH → tabulated SFH option → dust → nebular → AGN → multi-λ**, not necessarily the strict internal call order.

Internal refactor notes for contributors: [DOCS_REFACTOR_REFINED.md](https://github.com/suchethac/tengri/blob/main/docs/dev/DOCS_REFACTOR_REFINED.md).

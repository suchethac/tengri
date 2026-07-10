# AGNFITTER-RX ↔ tengri

A component-by-component reproduction of **AGNFITTER-RX** (Martínez-Ramírez
et al. 2024, A&A 688, A46; arXiv:2405.12111) against tengri, with the focus on
the AGN model — the four accretion-disk libraries, the four torus libraries,
the α_ox–L₂₅₀₀ X-ray corona, and the radio AGN/star-formation components that
make AGNFITTER-RX a radio-to-X-ray AGN-physics laboratory.

AGNFITTER-RX is the natural AGN counterpart to the CIGALE, BAGPIPES, and
Prospector reproductions in this series: where those are galaxy-centric, this
one places tengri next to a code built specifically to characterize active
nuclei across `8 < log ν/Hz < 20`.

## What this notebook compares

| Block | AGNFITTER-RX libraries | tengri |
|-------|------------------------|--------|
| Accretion disk | R06, SN12, KD18, THB21 | `richards2006`, `slone_netzer`, `kubota_done`, `qsogen` (+ lines + FeII) |
| Disk reddening | Prevot SMC `EBVbbb` | `agn_ebv_disc` / `agn.atten = "smc_prevot"` |
| Torus | S04, NK08, SKIRTOR, CAT3D-Wind | `silva04`, `nenkova`, `skirtor` + `skirtor_agnfitter`, `cat3d_wind` |
| Cold dust | DH02_CE01, S17, S17_radio (Schreiber+2018) | `schreiber2018` (also `schreiber2016`, `dale2014`) |
| X-ray corona | α_ox–L₂₅₀₀ (Just+2007) | `xray_agn_corona_from_disc`, `alpha_ox_from_l2500` |
| Radio | SPL / DPL (Eqs. 9–10), Bell-2003 SF | `radio_agn`, `radio_agn_dpl`, `radio_sfr_bell2003` |

tengri's `slone_netzer`, `silva04`, `cat3d_wind`, `skirtor_agnfitter`, and
`schreiber2018` blocks reimplement the physics of these libraries
independently and are validated against the vendored AGNFITTER-RX references
(`data/agnfitter_*_reference.h5`), both in `tests/crossval/` and visually in
the notebook. Every tengri model in the notebook is built through the public
`SEDModel.build` grammar, so the comparisons double as end-to-end wiring
checks of the composable AGN API.

## Prerequisites

The AGNFITTER-RX reference templates the notebook overlays are **committed**
to `data/` (`agnfitter_bbb_reference.h5`, `agnfitter_torus_reference.h5`,
`agnfitter_cold_dust_reference.h5`), so the notebook runs on a clean checkout
with no AGNfitter clone. The clone is needed only to *regenerate* those
references (`scripts/build_agnfitter_bbb_reference.py`,
`scripts/build_agnfitter_s17_reference.py`, and the per-model grid builders):

```bash
git clone --depth 1 --branch AGNfitter-rX_v0.1 \
    https://github.com/GabrielaCR/AGNfitter /tmp/AGNfitter-rX
```

Point the build scripts elsewhere with `export AGNFITTER_HOME=/path/to/...`.
The upstream template pickles are loaded through a restricted unpickler
(numpy/pandas primitives only, with a preflight opcode scan) — they are
untrusted external data.

## Running

```bash
cd reproduction/agnfitter
jupytext --to ipynb 01_agnfitter.py
jupyter nbconvert --to html --execute 01_agnfitter.ipynb
```

The figures are written to `_figs/agnfitter_*.png`.

## References

- Martínez-Ramírez, L. N., et al. 2024, A&A 688, A46 (AGNFITTER-RX).
- Calistro Rivera, G., et al. 2016, ApJ 833, 98 (original AGNfitter).
- Richards, G. T., et al. 2006, ApJS 166, 470 (R06 disk).
- Slone, O. & Netzer, H. 2012, MNRAS 426, 656 (SN12 disk).
- Kubota, A. & Done, C. 2018, MNRAS 480, 1247 (KD18 disk).
- Temple, M. J., Hewett, P. C. & Banerji, M. 2021, MNRAS 508, 737 (THB21 disk).
- Prevot, M. L., et al. 1984, A&A 132, 389 (SMC reddening).
- Silva, L., et al. 2004, MNRAS 355, 973 (S04 torus).
- Nenkova, M., et al. 2008, ApJ 685, 147 (NK08 / CLUMPY torus).
- Stalevski, M., et al. 2016, MNRAS 458, 2288 (SKIRTOR torus).
- Hönig, S. F. & Kishimoto, M. 2017, ApJL 838, L20 (CAT3D-Wind torus).
- Schreiber, C., et al. 2018, A&A 609, A30 (S17 cold dust).
- Dale, D. A. & Helou, G. 2002, ApJ 576, 159; Chary, R. & Elbaz, D. 2001,
  ApJ 556, 562 (DH02_CE01 cold dust).
- Just, A., et al. 2007, ApJ 665, 1004; Lusso, E. & Risaliti, G. 2016, ApJ 819,
  154; 2017, A&A 602, A79 (α_ox–L₂₅₀₀).
- Azadi, M., et al. 2023, ApJ 945, 145 (radio SPL/DPL); Bell, E. F. 2003, ApJ
  586, 794 (IR–radio correlation).
- Stern, D. 2015, ApJ 807, 129 (6 µm ↔ 2–10 keV; AGNFITTER-RX X-ray prior).

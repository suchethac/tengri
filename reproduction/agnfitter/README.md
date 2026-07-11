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
| Accretion disk | R06, SN12, KD18, THB21 | `richards2006`, `slone_netzer`, `multicolor`/`kubota_done`, `qsogen` |
| Torus | S04, NK08, SKIRTOR, CAT3D-Wind | `silva04`, `nenkova`, `skirtor`, `cat3d_wind` |
| Cold dust | DH02_CE01, S17 (Schreiber+2018) | `schreiber2016` |
| X-ray corona | α_ox–L₂₅₀₀ (Just+2007 / Lusso&Risaliti) | `xray_agn_corona_from_disc` |
| Radio | SPL / DPL (Azadi+2023), Bell-2003 SF | `radio_agn`, `radio_agn_dpl`, `radio_sfr_bell2003` |

The `slone_netzer` disc block was added to tengri as part of this work;
its SN12 α-disc template grid is repackaged from AGNFITTER-RX's published
library (`scripts/build_slone_netzer_grid.py`). The `cat3d_wind` and
`silva04` torus blocks likewise evaluate the template libraries published
with AGNFITTER-RX (`scripts/build_cat3d_wind_grid.py`,
`scripts/build_silva04_grid.py`), so §9c doubles as a direct check against
the originals.

## Prerequisites: the AGNFITTER-RX template libraries

The notebook reads AGNFITTER-RX's disk/torus/cold-dust template libraries
directly from a checkout (it never runs the fitter). Clone the tagged release:

```bash
git clone --depth 1 --branch AGNfitter-rX_v0.1 \
    https://github.com/GabrielaCR/AGNfitter /tmp/AGNfitter-rX
```

Point the driver elsewhere with `export AGNFITTER_HOME=/path/to/AGNfitter-rX`.
The template pickles are loaded through a restricted unpickler (numpy/pandas
primitives only, with a preflight opcode scan) — they are untrusted external
data.

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

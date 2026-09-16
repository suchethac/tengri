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
| Accretion disk | R06, SN12, KD18, THB21 | `richards2006`, `slone_netzer`, `kd18_agnfitter` (+ `kd18_agnfitter_warmindex`), `qsogen` (+ `blr`/`feii`) |
| Disk reddening | Prevot SMC `EBVbbb` | `agn_ebv_disc` (top-level); `agn.atten={'type': 'qsogen'}` for qsogen's own curve |
| Galaxy attenuation | SMC / Calzetti (fit-time) | `dust_attenuation={'law': 'calzetti', ...}` |
| Torus | S04, NK08, SKIRTOR, CAT3D-Wind, + NK0_mean_2p/3p, SKIRTOR_mean_1p/2p, CAT3D low-`f_wd` | `silva04`, `nenkova_agnfitter` (+`_2p`/`_3p`), `skirtor` + `skirtor_agnfitter` (+`_1p`/`_2p`), `cat3d_wind` (+`_lowfwd`) |
| Cold dust | DH02_CE01, S17, S17_radio (Schreiber+2018) | `schreiber2018` (also `schreiber2016`, `dale2014`, `dh02_ce01`) |
| X-ray corona | α_ox–L₂₅₀₀ (Just+2007) | `xray_agn_corona_from_disc`, `alpha_ox_from_l2500` |
| Radio | SPL / DPL (Eqs. 9–10), Bell-2003 SF (90/10 split) | `radio_agn`, `radio_agn_dpl`, `radio_sfr_bell2003_split`, `sfr_from_lir` |
| Priors | Eight informative priors (`PRIORS_AGNfitter.py`) | `tengri.agn.priors.agnfitter_priors`; `Fitter(..., extra_log_prior=...)` |

Beyond the single-node face-offs, the notebook sweeps several node grids
directly off each library's own axes: SN12 and KD18 disc `(log M_BH, log
λ_Edd)` nodes; S04 log N_H and NK08 inclination nodes; SKIRTOR `(oa, incl,
τ)` index triples plus the full X-CIGALE grid at the fiducial; CAT3D-Wind
`(incl, a, f_wd)` triples alongside the existing wind-fraction sweep; five
S17 cold-dust `(T_dust, f_PAH)` nodes and three DH02_CE01 log L_IR nodes;
an X-ray corona grid over Δα_ox and Γ; and a radio SPL `alpha x log ν_cut`
grid plus a DPL `log ν_t` grid.

tengri's `slone_netzer`, `silva04`, `cat3d_wind` (+ `cat3d_wind_lowfwd`),
`skirtor_agnfitter` (+ `_1p`/`_2p`), `nenkova_agnfitter` (+ `_2p`/`_3p`),
`kd18_agnfitter` (+ `_warmindex`), and `schreiber2018` blocks evaluate the
template libraries published with AGNFITTER-RX (repackaged by the
`scripts/build_*_grid.py`/`build_agnfitter_torus_reductions.py` builders) and
are validated against the vendored references (`data/agnfitter_*_reference.h5`),
both in `tests/crossval/` and visually in the notebook. Every tengri model in
the notebook is built through the public `SEDModel.build` grammar, so the
comparisons double as end-to-end wiring checks of the composable AGN API.

## Prerequisites

The AGNFITTER-RX reference templates the notebook overlays are **committed**
to `data/` (`agnfitter_bbb_reference.h5`, `agnfitter_torus_reference.h5`,
`agnfitter_cold_dust_reference.h5`), so the notebook runs on a clean checkout
with no AGNfitter clone. The clone is needed only to *regenerate* those
references (`scripts/build_agnfitter_bbb_reference.py`,
`scripts/build_agnfitter_s17_reference.py`, and the per-model grid builders).

The BC03 + Chabrier SSP grid tengri's own side needs is **not** required to
pre-exist: the notebook's Setup cell calls
`tengri.download_ssp("bc03_pdva_stelib_chabrier", dest=...)`, which fetches
it on first run and is a no-op on every run after (the file is cached
alongside the driver).

The build scripts fetch what they need straight from the pinned
`AGNfitter-rX_v0.1` tag (cached under `~/.cache/tengri_agnfitter`), so a
regeneration needs no manual clone; pass `--input` to read a local checkout
instead. They load the upstream pickles through a restricted unpickler
(numpy/pandas primitives only, with a preflight opcode scan) — they are
untrusted external data. The driver itself reads only the committed h5:
`tests/contract/test_reproduction_driver_no_clone.py` pins that.

## Running

```bash
python scripts/render_reproduction_notebook.py agnfitter
```

This runs `reproduction/CONTRACT.md` §7's recipe end to end (`jupytext --to
ipynb`, then a headless `PYTHONHASHSEED=0 jupyter nbconvert --execute
--inplace`), fails loudly on any error-output cell or a `SystemExit`-truncated
run, stamps the render with the SHA-256 of the source `.py` it ran from, and
publishes the result to `docs/reproduction/`. The figures are written to
`_figs/agnfitter_*.png`.

`validate_matched_physics.py` is the strict companion check: it removes every
input difference between tengri and AGNFITTER-RX and compares the cold-dust
and accretion-disk SEDs pixel by pixel at matched template nodes, rather than
the notebook's own configuration-level comparisons.

```bash
JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \
    .venv/bin/python reproduction/agnfitter/validate_matched_physics.py [--figdir DIR]
```

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

# #1206 unit-change regression gate: HEAD vs merge base (2026-09-11, revised)

Both sides measured with the identical `qh_census_probe.py`, in pure float64 (`JAX_ENABLE_X64=1`, `TENGRI_DISABLE_JAX_CACHE=1`), same process type, on this worktree: **base** = `origin/fix/1206-line-fluxes-float32` at `f355e61360b0b5c1d189ad89a2670aad1abf55b7` (the merge base of this branch, i.e. the exact commit this branch forked from -- WI-1's tip at fork time), **head** = this branch's tip. This isolates exactly this PR's own diff, with no dependence on how far `origin/main` has since moved. `rel_change` is `abs(head*L_SUN/base - 1)` for the re-united properties (line luminosities + X-ray luminosities, must be <= 1e-12) and `abs(head/base - 1)` for every other property (must be 0, i.e. bit-identical to rtol 1e-12) -- except `q_h` (retired with no alias, raises `KeyError` on head) and `civ_1549`/`log_civ_1549` (NaN on both sides, a pre-existing Cue legacy-subset gap unrelated to #1206).

## `neb_cue`

| property | base (f64) | head (f64) | rel_change | verdict |
|---|---|---|---|---|
| `balmer_break` | 2.117163e+00 | 2.117163e+00 | 0.000e+00 | OK |
| `balmer_decrement` | 3.529520e+00 | 3.529520e+00 | 0.000e+00 | OK |
| `bpt_nii` | -5.046204e-01 | -5.046204e-01 | 0.000e+00 | OK |
| `bpt_sii` | -4.547346e-01 | -4.547346e-01 | 0.000e+00 | OK |
| `civ_1549` | nan | nan | n/a | OK (NaN both sides, pre-existing) |
| `halpha` | 4.601334e+40 | 1.202020e+07 | 1.199e-14 | OK |
| `hbeta` | 1.303671e+40 | 3.405619e+06 | 1.377e-14 | OK |
| `l_bol` | 7.350815e+09 | 7.350815e+09 | 0.000e+00 | OK |
| `l_tir` | 1.807617e+08 | 1.807617e+08 | 0.000e+00 | OK |
| `log_civ_1549` | nan | nan | n/a | OK (NaN both sides, pre-existing) |
| `log_halpha` | 4.066288e+01 | 4.066288e+01 | 0.000e+00 | OK |
| `log_hbeta` | 4.011517e+01 | 4.011517e+01 | 0.000e+00 | OK |
| `log_lya` | 4.089588e+01 | 4.089588e+01 | 0.000e+00 | OK |
| `log_nii_6548` | 3.970065e+01 | 3.970065e+01 | 0.000e+00 | OK |
| `log_nii_6584` | 4.015826e+01 | 4.015826e+01 | 0.000e+00 | OK |
| `log_oii` | 4.054076e+01 | 4.054076e+01 | 0.000e+00 | OK |
| `log_oiii_4959` | 3.972558e+01 | 3.972558e+01 | 0.000e+00 | OK |
| `log_oiii_5007` | 4.020247e+01 | 4.020247e+01 | 0.000e+00 | OK |
| `log_q_h` | 5.280971e+01 | 5.280971e+01 | 0.000e+00 | OK |
| `log_sii_6717` | 3.996210e+01 | 3.996210e+01 | 0.000e+00 | OK |
| `log_sii_6731` | 3.984416e+01 | 3.984416e+01 | 0.000e+00 | OK |
| `lya` | 7.868372e+40 | 2.055478e+07 | 8.549e-15 | OK |
| `nii_6548` | 5.019422e+39 | 1.311239e+06 | 4.885e-15 | OK |
| `nii_6584` | 1.439671e+40 | 3.760896e+06 | 1.732e-14 | OK |
| `o32` | -3.382910e-01 | -3.382910e-01 | 0.000e+00 | OK |
| `o3hb` | 8.730245e-02 | 8.730245e-02 | 0.000e+00 | OK |
| `oii` | 3.473454e+40 | 9.073808e+06 | 1.754e-14 | OK |
| `oiii_4959` | 5.315914e+39 | 1.388692e+06 | 6.661e-15 | OK |
| `oiii_5007` | 1.593935e+40 | 4.163884e+06 | 1.199e-14 | OK |
| `q_h` | 6.452192e+52 | RETIRED (`KeyError`) | n/a | OK (retired, no alias, #1206 §C -- see log_q_h) |
| `r23` | 6.329408e-01 | 6.329408e-01 | 0.000e+00 | OK |
| `sii_6717` | 9.164243e+39 | 2.394003e+06 | 1.210e-14 | OK |
| `sii_6731` | 6.984886e+39 | 1.824683e+06 | 1.388e-14 | OK |
| `xi_ion` | 4.837000e+10 | 4.837000e+10 | 0.000e+00 | OK |

## `panchromatic`

| property | base (f64) | head (f64) | rel_change | verdict |
|---|---|---|---|---|
| `balmer_break` | 1.979643e+00 | 1.979643e+00 | 0.000e+00 | OK |
| `balmer_decrement` | 3.529520e+00 | 3.529520e+00 | 0.000e+00 | OK |
| `bpt_nii` | -5.046204e-01 | -5.046204e-01 | 0.000e+00 | OK |
| `bpt_sii` | -4.547346e-01 | -4.547346e-01 | 0.000e+00 | OK |
| `civ_1549` | nan | nan | n/a | OK (NaN both sides, pre-existing) |
| `halpha` | 4.601334e+40 | 1.202020e+07 | 1.199e-14 | OK |
| `hbeta` | 1.303671e+40 | 3.405619e+06 | 1.377e-14 | OK |
| `l_bol` | 2.381921e+10 | 2.381921e+10 | 0.000e+00 | OK |
| `l_tir` | 6.199776e+09 | 6.199776e+09 | 0.000e+00 | OK |
| `l_x_agn` | 6.390659e+42 | 1.669451e+09 | 1.399e-14 | OK |
| `l_x_total` | 6.392150e+42 | 1.669841e+09 | 1.454e-14 | OK |
| `l_x_xrb` | 1.491100e+39 | 3.895246e+05 | 1.077e-14 | OK |
| `log_civ_1549` | nan | nan | n/a | OK (NaN both sides, pre-existing) |
| `log_halpha` | 4.066288e+01 | 4.066288e+01 | 0.000e+00 | OK |
| `log_hbeta` | 4.011517e+01 | 4.011517e+01 | 0.000e+00 | OK |
| `log_l_x_agn` | 4.280555e+01 | 4.280555e+01 | 0.000e+00 | OK |
| `log_l_x_total` | 4.280565e+01 | 4.280565e+01 | 0.000e+00 | OK |
| `log_l_x_xrb` | 3.917351e+01 | 3.917351e+01 | 0.000e+00 | OK |
| `log_lya` | 4.089588e+01 | 4.089588e+01 | 0.000e+00 | OK |
| `log_nii_6548` | 3.970065e+01 | 3.970065e+01 | 0.000e+00 | OK |
| `log_nii_6584` | 4.015826e+01 | 4.015826e+01 | 0.000e+00 | OK |
| `log_oii` | 4.054076e+01 | 4.054076e+01 | 0.000e+00 | OK |
| `log_oiii_4959` | 3.972558e+01 | 3.972558e+01 | 0.000e+00 | OK |
| `log_oiii_5007` | 4.020247e+01 | 4.020247e+01 | 0.000e+00 | OK |
| `log_q_h` | 5.280971e+01 | 5.280971e+01 | 0.000e+00 | OK |
| `log_sii_6717` | 3.996210e+01 | 3.996210e+01 | 0.000e+00 | OK |
| `log_sii_6731` | 3.984416e+01 | 3.984416e+01 | 0.000e+00 | OK |
| `lya` | 7.868372e+40 | 2.055478e+07 | 8.549e-15 | OK |
| `nii_6548` | 5.019422e+39 | 1.311239e+06 | 4.885e-15 | OK |
| `nii_6584` | 1.439671e+40 | 3.760896e+06 | 1.732e-14 | OK |
| `o32` | -3.382910e-01 | -3.382910e-01 | 0.000e+00 | OK |
| `o3hb` | 8.730245e-02 | 8.730245e-02 | 0.000e+00 | OK |
| `oii` | 3.473454e+40 | 9.073808e+06 | 1.754e-14 | OK |
| `oiii_4959` | 5.315914e+39 | 1.388692e+06 | 6.661e-15 | OK |
| `oiii_5007` | 1.593935e+40 | 4.163884e+06 | 1.199e-14 | OK |
| `q_h` | 6.452192e+52 | RETIRED (`KeyError`) | n/a | OK (retired, no alias, #1206 §C -- see log_q_h) |
| `r23` | 6.329408e-01 | 6.329408e-01 | 0.000e+00 | OK |
| `sii_6717` | 9.164243e+39 | 2.394003e+06 | 1.210e-14 | OK |
| `sii_6731` | 6.984886e+39 | 1.824683e+06 | 1.388e-14 | OK |
| `xi_ion` | 2.924687e+10 | 2.924687e+10 | 0.000e+00 | OK |

## Summary

- Every non-re-united property bit-identical (rtol 1e-12) between base and head, and every re-united property equal to `base / L_sun` to 1e-12: **YES**.

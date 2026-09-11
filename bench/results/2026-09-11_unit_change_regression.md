# #1206 unit-change regression gate (2026-09-11)

Measured with the orchestrator's standardized fixture (`qh_census_probe.py`, two models: `neb_cue`, `panchromatic`) on this branch (`fix/1206-luminosity-units-lsun`), compared against the same fixture captured on untouched `origin/main`. `rel_change` is `abs(after * L_SUN / before - 1)` for the eleven line properties and the three X-ray luminosities that moved to Lsun (#1206 §A/§B; must be <= 1e-12), and `abs(after / before - 1)` for every other (unaffected) property (must be 0, i.e. bit-identical to 1e-12) -- except `q_h`, retired with no alias (#1206 §C, no `after` row: `pred.properties['q_h']` now raises `KeyError` naming `log_q_h`), and `civ_1549` / `log_civ_1549`, `NaN` in float64 before this PR too (Cue legacy subset gap, unrelated to #1206).

## `neb_cue`

| property | f64 before | f64 after | rel change | f32 after | f32-vs-f64 rel | f32 finite |
|---|---|---|---|---|---|---|
| `balmer_break` | 2.117163e+00 | 2.117163e+00 | 0.000e+00 | 2.117160e+00 | 1.173e-06 | True |
| `balmer_decrement` | 3.529520e+00 | 3.529520e+00 | 0.000e+00 | 3.529354e+00 | 4.712e-05 | True |
| `bpt_nii` | -5.046204e-01 | -5.046204e-01 | 0.000e+00 | -5.046120e-01 | 1.664e-05 | True |
| `bpt_sii` | -4.547346e-01 | -4.547346e-01 | 0.000e+00 | -4.547310e-01 | 7.999e-06 | True |
| `civ_1549` | nan | nan | nan | nan | nan | False |
| `halpha` | 4.601334e+40 | 1.202020e+07 | 1.199e-14 | 1.202006e+07 | 1.153e-05 | True |
| `hbeta` | 1.303671e+40 | 3.405619e+06 | 1.377e-14 | 3.405740e+06 | 3.532e-05 | True |
| `l_bol` | 7.350815e+09 | 7.350815e+09 | 0.000e+00 | 7.350782e+09 | 4.516e-06 | True |
| `l_tir` | 1.807617e+08 | 1.807617e+08 | 0.000e+00 | 1.807608e+08 | 4.842e-06 | True |
| `log_civ_1549` | nan | nan | nan | nan | nan | False |
| `log_halpha` | 4.066288e+01 | 4.066288e+01 | 0.000e+00 | 4.066288e+01 | 9.280e-08 | True |
| `log_hbeta` | 4.011517e+01 | 4.011517e+01 | 0.000e+00 | 4.011518e+01 | 4.171e-07 | True |
| `log_lya` | 4.089588e+01 | 4.089588e+01 | 0.000e+00 | 4.089589e+01 | 1.483e-08 | True |
| `log_nii_6548` | 3.970065e+01 | 3.970065e+01 | 0.000e+00 | 3.970066e+01 | 8.032e-08 | True |
| `log_nii_6584` | 4.015826e+01 | 4.015826e+01 | 0.000e+00 | 4.015827e+01 | 1.166e-07 | True |
| `log_oii` | 4.054076e+01 | 4.054076e+01 | 0.000e+00 | 4.054076e+01 | 5.812e-08 | True |
| `log_oiii_4959` | 3.972558e+01 | 3.972558e+01 | 0.000e+00 | 3.972558e+01 | 8.535e-09 | True |
| `log_oiii_5007` | 4.020247e+01 | 4.020247e+01 | 0.000e+00 | 4.020246e+01 | 2.303e-07 | True |
| `log_q_h` | 5.280971e+01 | 5.280971e+01 | 0.000e+00 | 5.280971e+01 | 7.907e-08 | True |
| `log_sii_6717` | 3.996210e+01 | 3.996210e+01 | 0.000e+00 | 3.996210e+01 | 1.454e-08 | True |
| `log_sii_6731` | 3.984416e+01 | 3.984416e+01 | 0.000e+00 | 3.984416e+01 | 2.852e-08 | True |
| `lya` | 7.868372e+40 | 2.055478e+07 | 8.549e-15 | 2.055474e+07 | 2.272e-06 | True |
| `nii_6548` | 5.019422e+39 | 1.311239e+06 | 4.885e-15 | 1.311244e+06 | 4.009e-06 | True |
| `nii_6584` | 1.439671e+40 | 3.760896e+06 | 1.732e-14 | 3.760922e+06 | 6.954e-06 | True |
| `o32` | -3.382910e-01 | -3.382910e-01 | 0.000e+00 | -3.383023e-01 | 3.345e-05 | True |
| `o3hb` | 8.730245e-02 | 8.730245e-02 | 0.000e+00 | 8.727647e-02 | 2.975e-04 | True |
| `oii` | 3.473454e+40 | 9.073808e+06 | 1.754e-14 | 9.073826e+06 | 2.020e-06 | True |
| `oiii_4959` | 5.315914e+39 | 1.388692e+06 | 6.661e-15 | 1.388688e+06 | 2.765e-06 | True |
| `oiii_5007` | 1.593935e+40 | 4.163884e+06 | 1.199e-14 | 4.163780e+06 | 2.492e-05 | True |
| `q_h` | 6.452192e+52 | RETIRED (raises `KeyError`) | n/a | RETIRED (raises `KeyError`) | n/a | n/a -- see `log_q_h` |
| `r23` | 6.329408e-01 | 6.329408e-01 | 0.000e+00 | 6.329228e-01 | 2.845e-05 | True |
| `sii_6717` | 9.164243e+39 | 2.394003e+06 | 1.210e-14 | 2.393999e+06 | 1.700e-06 | True |
| `sii_6731` | 6.984886e+39 | 1.824683e+06 | 1.388e-14 | 1.824673e+06 | 5.580e-06 | True |
| `xi_ion` | 4.837000e+10 | 4.837000e+10 | 0.000e+00 | 4.837067e+10 | 1.391e-05 | True |

## `panchromatic`

| property | f64 before | f64 after | rel change | f32 after | f32-vs-f64 rel | f32 finite |
|---|---|---|---|---|---|---|
| `balmer_break` | 1.979643e+00 | 1.979643e+00 | 0.000e+00 | 1.979641e+00 | 1.005e-06 | True |
| `balmer_decrement` | 3.529520e+00 | 3.529520e+00 | 0.000e+00 | 3.529354e+00 | 4.712e-05 | True |
| `bpt_nii` | -5.046204e-01 | -5.046204e-01 | 0.000e+00 | -5.046120e-01 | 1.664e-05 | True |
| `bpt_sii` | -4.547346e-01 | -4.547346e-01 | 0.000e+00 | -4.547310e-01 | 7.999e-06 | True |
| `civ_1549` | nan | nan | nan | nan | nan | False |
| `halpha` | 4.601334e+40 | 1.202020e+07 | 1.199e-14 | 1.202006e+07 | 1.153e-05 | True |
| `hbeta` | 1.303671e+40 | 3.405619e+06 | 1.377e-14 | 3.405740e+06 | 3.532e-05 | True |
| `l_bol` | 2.381920e+10 | 2.381921e+10 | 4.228e-07 | 2.381921e+10 | 2.450e-07 | True |
| `l_tir` | 6.199776e+09 | 6.199776e+09 | 0.000e+00 | 6.199773e+09 | 4.692e-07 | True |
| `l_x_agn` | 6.390659e+42 | 1.669451e+09 | 1.399e-14 | 1.669446e+09 | 2.878e-06 | True |
| `l_x_total` | 6.392150e+42 | 1.669841e+09 | 1.454e-14 | 1.669841e+09 | 3.301e-07 | True |
| `l_x_xrb` | 1.491100e+39 | 3.895246e+05 | 1.077e-14 | 3.895239e+05 | 1.587e-06 | True |
| `log_civ_1549` | nan | nan | nan | nan | nan | False |
| `log_halpha` | 4.066288e+01 | 4.066288e+01 | 0.000e+00 | 4.066288e+01 | 9.280e-08 | True |
| `log_hbeta` | 4.011517e+01 | 4.011517e+01 | 0.000e+00 | 4.011518e+01 | 4.171e-07 | True |
| `log_l_x_agn` | 4.280555e+01 | 4.280555e+01 | 0.000e+00 | 4.280555e+01 | 4.237e-09 | True |
| `log_l_x_total` | 4.280565e+01 | 4.280565e+01 | 0.000e+00 | 4.280565e+01 | 4.341e-08 | True |
| `log_l_x_xrb` | 3.917351e+01 | 3.917351e+01 | 0.000e+00 | 3.917351e+01 | 2.350e-08 | True |
| `log_lya` | 4.089588e+01 | 4.089588e+01 | 0.000e+00 | 4.089589e+01 | 1.483e-08 | True |
| `log_nii_6548` | 3.970065e+01 | 3.970065e+01 | 0.000e+00 | 3.970066e+01 | 8.032e-08 | True |
| `log_nii_6584` | 4.015826e+01 | 4.015826e+01 | 0.000e+00 | 4.015827e+01 | 1.166e-07 | True |
| `log_oii` | 4.054076e+01 | 4.054076e+01 | 0.000e+00 | 4.054076e+01 | 5.812e-08 | True |
| `log_oiii_4959` | 3.972558e+01 | 3.972558e+01 | 0.000e+00 | 3.972558e+01 | 8.535e-09 | True |
| `log_oiii_5007` | 4.020247e+01 | 4.020247e+01 | 0.000e+00 | 4.020246e+01 | 2.303e-07 | True |
| `log_q_h` | 5.280971e+01 | 5.280971e+01 | 0.000e+00 | 5.280971e+01 | 7.907e-08 | True |
| `log_sii_6717` | 3.996210e+01 | 3.996210e+01 | 0.000e+00 | 3.996210e+01 | 1.454e-08 | True |
| `log_sii_6731` | 3.984416e+01 | 3.984416e+01 | 0.000e+00 | 3.984416e+01 | 2.852e-08 | True |
| `lya` | 7.868372e+40 | 2.055478e+07 | 8.549e-15 | 2.055474e+07 | 2.272e-06 | True |
| `nii_6548` | 5.019422e+39 | 1.311239e+06 | 4.885e-15 | 1.311244e+06 | 4.009e-06 | True |
| `nii_6584` | 1.439671e+40 | 3.760896e+06 | 1.732e-14 | 3.760922e+06 | 6.954e-06 | True |
| `o32` | -3.382910e-01 | -3.382910e-01 | 0.000e+00 | -3.383023e-01 | 3.345e-05 | True |
| `o3hb` | 8.730245e-02 | 8.730245e-02 | 0.000e+00 | 8.727647e-02 | 2.975e-04 | True |
| `oii` | 3.473454e+40 | 9.073808e+06 | 1.754e-14 | 9.073826e+06 | 2.020e-06 | True |
| `oiii_4959` | 5.315914e+39 | 1.388692e+06 | 6.661e-15 | 1.388688e+06 | 2.765e-06 | True |
| `oiii_5007` | 1.593935e+40 | 4.163884e+06 | 1.199e-14 | 4.163780e+06 | 2.492e-05 | True |
| `q_h` | 6.452192e+52 | RETIRED (raises `KeyError`) | n/a | RETIRED (raises `KeyError`) | n/a | n/a -- see `log_q_h` |
| `r23` | 6.329408e-01 | 6.329408e-01 | 0.000e+00 | 6.329228e-01 | 2.845e-05 | True |
| `sii_6717` | 9.164243e+39 | 2.394003e+06 | 1.210e-14 | 2.393999e+06 | 1.700e-06 | True |
| `sii_6731` | 6.984886e+39 | 1.824683e+06 | 1.388e-14 | 1.824673e+06 | 5.580e-06 | True |
| `xi_ion` | 2.924624e+10 | 2.924687e+10 | 2.137e-05 | 2.924717e+10 | 1.030e-05 | True |

## Summary

- Every previously-working row unchanged in float64 to 1e-12: **NO, two exceptions, both diagnosed as pre-existing and unrelated to this PR (see below): `panchromatic.l_bol` (4.2e-07) and `panchromatic.xi_ion` (2.1e-05).**
- Every previously-broken (non-finite in float32) row now finite: **YES.**

### `l_bol` / `xi_ion` in `panchromatic`: diagnosed, not fixed here

Two rows in the `panchromatic` model moved in float64 by more than the 1e-12
this gate otherwise holds to: `l_bol` (4.228e-07) and `xi_ion` (2.137e-05).
Diagnosis, not a "fix" (per the brief: report rather than paper over a moved
working row):

1. **Not caused by this branch's diff.** `git diff origin/fix/1206-line-fluxes-float32
   HEAD --stat` touches no dust, SFH, or core SED-assembly file; the only
   `agn/component.py` change removes a dead `warnings.warn(...)` call inside a
   branch this model's disc (`multicolor`) never took even before the removal
   (`_NON_FLOAT32_SAFE_DISCS` never contained `"multicolor"`). Confirmed
   directly: re-running the float64 probe with `TENGRI_DISABLE_JAX_CACHE=1`
   (ruling out JAX persistent-cache contamination, #1392's failure mode)
   reproduced `l_bol` and `xi_ion` bit-for-bit identical to the cached run.
2. **Caused by the branch point, not this PR.** This branch forks from
   `origin/fix/1206-line-fluxes-float32` (WI-1), per the brief, not from
   `origin/main` -- but the orchestrator's "before" census was captured on
   `origin/main`. `git diff origin/main origin/fix/1206-line-fluxes-float32
   --stat -- src/tengri/components/dust src/tengri/parameters` shows
   `dust/two_component.py` (253 lines), `parameters/_dust_keys.py` (179,
   deleted going main -> WI-1 branch), `parameters/groups.py` and
   `parameters/parameters.py` all differ: `origin/main` carries
   `1f9ff84e4 feat(dust): each emission source chooses its dust screen; shock
   defaults to diffuse; fast-nebular snaps to catalog lines (#2234, #2235)
   (#2260)`, which `origin/fix/1206-line-fluxes-float32`'s last touch to that
   file (`6cc1a8b25`, an earlier commit) does not have. "Each emission source
   chooses its dust screen" plausibly explains why only `panchromatic`
   (AGN + `dust_emission` together) moves while `neb_cue` (no AGN, no
   `dust_emission`) does not: the per-source screen choice is presumably a
   no-op with only one emission source in play.
3. **Not fixable without rebasing onto current `main`**, which the brief did
   not ask for and which would pull in unrelated `main` history (inference
   backend and parameter-grammar changes well outside this PR's scope, per
   the same `--stat`). Reported here rather than "fixed" by, e.g., forcing a
   match against the stale baseline.

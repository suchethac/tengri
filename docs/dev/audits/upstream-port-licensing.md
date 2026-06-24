# Upstream port licensing — open issue for later resolution

**Status:** Open. To be revisited before the first tagged release / Zenodo DOI.
**Owner:** Project lead.
**Filed:** 2026-05-22.

## Why this exists

Tengri is released under **BSD-3-Clause**. Several physics modules cite "ported
from" upstream projects. Upstream licenses (verified 2026-05-22):

| Upstream | License | Compatibility | Where it appears in tengri |
| --- | --- | --- | --- |
| Prospector (`bd-j/prospector`) | **MIT** | ✓ permissive | `igm.py` (`add_igm` from `fake_fsps.py`), `nlr.py`, line-placement, `psb_logsfr_ratios_to_agebins`, `zred_to_agebins_pbeta` |
| FastSpecFit (`desihub/fastspecfit`) | **BSD-3-Clause** | ✓ matches | `analysis/diagnostics/lines.py` (`populate_emtable`, `LineMasker`) |
| AGNfitter / AGNfitter-rX (`GabrielaCR/AGNfitter`) | **MIT** | ✓ permissive | `silva04.py`, BBB helpers |
| RELAGN (`scotthgn/RELAGN`) | **MIT** | ✓ permissive | `_nthcomp.py` (XSpec `donthcomp.f` lineage; original XSpec is NASA public domain) |
| eazy-py (`gbrammer/eazy-py`) | **MIT** | ✓ permissive | IGM coefficient tables in `igm.py` (lines 79–168) |
| **CIGALE** (`gitlab.lam.fr/cigale/cigale`) | **CeCILL v2** | ⚠ copyleft | `polar_dust.py`, `disc_cigale.py`, SFH modules (`sfh_buat08`, `sfhdelayedbq`, `sfhperiodic`), IRX |
| **ProSpect** (R package, `asgr/ProSpect`) | **LGPL-3** | ⚠ copyleft (linking-permissive) | `massfunc_p4`, `massfunc_p6`, `massfunc_snorm_burst`, `massfunc_snorm_burst_trunc` |

**Revised risk profile (2026-05-22):** the verification flipped the picture
materially. Prospector — previously suspected of being GPL — is **MIT**,
which is fully compatible with BSD-3. The same is true of AGNfitter (MIT),
RELAGN (MIT), and eazy-py (MIT); FastSpecFit is BSD-3 (matches tengri). The
only remaining copyleft concerns are **CIGALE (CeCILL v2)** and **ProSpect
(LGPL-3)**.

For the remaining two:

- **CIGALE (CeCILL v2)** is the closest French equivalent of GPL. Viral if
  the port is a substantial derivative work of CIGALE source code (not just
  the physics CIGALE implements).
- **ProSpect (LGPL-3)** is *linking-permissive* — calling ProSpect from
  BSD-3 code is fine, but *copying* its source into a BSD-3 codebase still
  triggers LGPL on the copied portion. The current "Ported from ProSpect"
  notes need to be scrutinized: if they are reimplementations from the
  published ProSpect / massfunc papers, that is clean; if they are
  R-to-JAX transliterations of ProSpect functions, LGPL applies.

Resolution options remain the same: (a) re-release the affected modules
under the upstream license, (b) replace with clean-room reimplementations
from the original papers, or (c) demonstrate the ports fall outside the
upstream's copyright scope (e.g. by being short, equation-driven, with
no carry-over of upstream control-flow or variable naming).

## Current assessment (port-vs-rewrite, per file)

None of the upstreams are JAX-based; every "port" required a translation
into pure JAX (`jnp.*`, `vmap`, `lax.scan`, immutable updates). That
translation forces structural rewriting and gives some natural distance
from upstream code. Where the assessment lands:

### Low risk — algorithmic ports or clean-room from papers

- **`components/igm/igm.py`** — Inoue+2014 formulae implemented from the
  paper; numerical coefficients pulled from **eazy-py** (BSD-3-Clause,
  credited at file line 15). The Prospector citation (line 805) is for
  the `add_igm` *integration point*, not bulk code reuse. No substantive
  Prospector copy.
- **`components/agn/polar_dust.py`** — Calzetti+2000, Gaskell+2004, and
  Yang+2020 extinction/anisotropy formulae implemented from the published
  equations. JAX-native (`jnp.where`, `jax.nn.sigmoid`, `jnp.trapezoid`).
  No CIGALE code structure carried across.
- **`analysis/diagnostics/lines.py`** — Steidel+1996 EW formula and
  standard Gaussian line-flux physics. FastSpecFit citation is
  attribution for methodology, not bulk code transfer.
- **`components/agn/_nthcomp.py`** — Only the table-lookup harness lives
  in tengri; the Kompaneets solver itself runs in upstream RELAGN to
  generate the table. A custom JAX VJP wraps the trilinear interpolation
  for gradient support. No XSpec/RELAGN code copied.

### Moderate risk — algorithmic ports that embed published numerical tables

- **`components/agn/nlr.py`** — Richardson+2014 Table 3 ('a42') line
  wavelengths and flux ratios are embedded as `_RICHARDSON_WAVES` /
  `_RICHARDSON_FLUXES` (~50 lines of numeric data). Computation is a
  `jax.vmap` Gaussian-line rewrite. Tables are *published scientific
  data* and not generally copyrightable; Prospector's selection /
  formatting is acknowledged but the data itself comes from Richardson+
  2014.
- **`components/agn/disc_cigale.py`** — SKIRTOR power-law breakpoints
  (5 numbers) and slopes (4 numbers) from Stalevski+2012 via CIGALE.
  Computation rewritten with `jax.lax.scan` and `jnp.searchsorted`. The
  parameters are published in Stalevski+2012; the algorithm itself is
  reimplemented.
- **`components/agn/silva04.py`** — No source code from AGNfitter is
  imported; `scripts/build_silva04_grid.py` (out-of-tree at build time)
  converts AGNfitter's pickled grid into an HDF5 loaded at runtime via
  `interp_nd_triweight`. If we ship the HDF5 in `data/`, we are shipping
  data derived from AGNfitter and should mirror AGNfitter's terms for
  that data file.

### Higher risk — SFH and bundled template files

Not assessed in depth yet. Need to check:

- **CIGALE SFH ports** — `sfh_buat08`, `sfhdelayedbq`, `sfhperiodic`,
  IRX module: how much of the *structure* and parameter names are
  carried? Equations from refereed papers (Buat+2008 etc.) are clean
  but the parameter conventions tend to track CIGALE exactly.
- **ProSpect SFH ports** — `massfunc_p4/p6`, `massfunc_snorm_burst*`:
  ProSpect is GPL-3 in R. Need to confirm whether these are paper
  reproductions or R-to-JAX transliterations of ProSpect source.
- **Bundled `.h5` template files in `data/`** — `astrodust_templates.h5`,
  `dl07_templates.h5`, `dale2014_templates.h5`, `skirtor_templates_v3.h5`,
  `bosa_templates.h5`. Each carries its own provenance and license; the
  `data/README.md` notes this but a per-file license table would close
  the loop.

## Overall view

Tengri's licensing posture is **defensible under BSD-3-Clause** as of today:

- No file is a line-for-line numpy→jnp transliteration of GPL/CeCILL
  upstream code.
- Embedded numerical data (Richardson+2014 line table, Stalevski+2012
  power-law parameters) are published scientific values, not creative
  expression.
- Viral copyleft is triggered by copying *code*, not by reimplementing
  published *equations* with attribution.

What we are missing for a confident 1.0:

1. **Per-port written justification.** A short paragraph next to each
   "Ported from X" docstring stating either *"clean-room from paper
   {ref}"* or *"algorithmic port of {symbol} — no upstream code text
   carried"*.
2. **Verified upstream license tags.** This document marks several as
   "(verify upstream)" — those need to be confirmed from each project's
   `LICENSE` file and recorded.
3. **CIGALE SFH and ProSpect SFH** — currently moderate-confidence; need
   the same port-vs-rewrite review as the AGN files above.
4. **Bundled data file licensing** — per-file table in `data/README.md`
   linking each shipped `.h5` to upstream license / data-use terms.
5. **Optional belt-and-suspenders** — for items where the assessment
   feels close (CIGALE SFH especially), consider a clean-room rewrite
   from the published paper and drop the "Ported from CIGALE" wording.

## Resolution criteria

This issue is closed when:
- Every "Ported from" / "Adapted from" docstring carries one of the two
  justification labels above.
- The license table in this document has no "(verify upstream)" entries.
- `NOTICE` and `data/README.md` agree on per-data-file provenance.
- A short paragraph in `LICENSE`-adjacent docs (or this file linked from
  `README.md`) explains the BSD-3 posture and our reasoning, so a
  downstream packager (conda-forge, JOSS reviewer) has the answer
  pre-canned.

## What "keeping BSD-3 for now" entails

Until this issue is resolved, the obligations on the tengri side are:

1. **Distribute `LICENSE` and `NOTICE` with every copy** of the source
   tree, sdist, and wheel. (BSD-3 §§1-2.) Both files are at the repo
   root and are picked up by the `pyproject.toml` packaging config.
2. **Preserve the copyright/permission notice** in source: this is what
   the SPDX header sweep just delivered — every `.py` under `src/` and
   `tests/` declares `# SPDX-License-Identifier: BSD-3-Clause`.
3. **No endorsement** (BSD-3 §3): we may not use Suchetha Cooray's name
   or the tengri name to endorse derivative products without permission.
   Practically: if someone forks tengri, they cannot brand their fork
   "tengri" without permission.
4. **No additional warranty** beyond the disclaimer. Don't add "this
   code is safe for X" claims to README that contradict the BSD-3
   disclaimer.

What BSD-3 does *not* require:
- We do not have to release derivative works under BSD-3 (that would
  be copyleft).
- We do not have to ship build scripts, tests, or documentation under
  the same terms (each can carry its own license if needed).
- We do not have to disclose changes (that would be Apache-2.0 §4.b).

For now, the project is BSD-3 compliant. The upstream-port question
above is a *separate* compatibility concern (us consuming GPL/CeCILL
inbound), not a BSD-3 obligation.

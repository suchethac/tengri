# Wave 1 Object Model Implementation Plan (epic #1322 — #1321, absorbing #1010 API home + #919)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Observation` a pure instrument schema and introduce the `Data` measurement record, with `mode=` on `ForwardModel.build` — the vocabulary every later wave speaks.

**Architecture:** Five tasks. T1 creates `Data` (new file, pure container + validation). T2 routes `Data` through `fit()` (unpack to the existing data/noise/censor/line plumbing — no likelihood changes). T3 applies the razor to `Observation` (add `lines=`, deprecate value-carrying fields). T4 adds `mode=`. T5 extends `NoiseModel` with per-band floors (#919). T1→T2→T3 are sequential; T4 and T5 are independent of them and of each other.

**Tech Stack:** Python 3.12, JAX (CPU tests), pytest + chex, ruff.

**Depends on:** Wave 0 merged — specifically T10/#1315 (`observation` optional on `ForwardModel.build`).

## Global Constraints

Identical to `docs/internal/plans/2026-07-23-wave0-api-fixes.md` §Global Constraints (worktree per task, `PYTHONPATH=$PWD/src` pytest, `-n 2`, ruff, taxonomy markers, pasted-RED anti-no-op rule, STOP-on-anchor-drift, no pushes — orchestrator bundles). The orchestrator includes that section verbatim in every dispatch.

Additional wave-1 rule: **the spec is the API authority.** `docs/internal/specs/2026-07-23-inference-prediction-api-final.md` §3–§4 defines every public name and field in this plan. If a task step conflicts with the spec, the spec wins — STOP and report the conflict.

---

### Task T1: the `Data` record

**Executor:** sonnet
**Branch:** `feat/data-record`

**Files:**
- Create: `src/tengri/observation/data.py`
- Modify: `src/tengri/observation/__init__.py` and `src/tengri/__init__.py` (export `Data`)
- Test: `tests/unit/observation/test_data_record.py` (create)

**Interfaces:**
- Produces: `tengri.Data` — frozen dataclass with fields `photometry: tuple | None`, `spectrum: tuple | None`, `lines: dict | None`, `censor: array | None`, and one method `validate_against(observation) -> ValidatedData` (a plain namedtuple: `flux, noise, censor, line_values` as jnp arrays / dicts, shapes normalized). T2 consumes exactly this.

- [ ] **Step 1: Failing tests.** Cover the schema:record contract (spec §3.2–§3.3):

```python
# tests/unit/observation/test_data_record.py
"""#1321: Data is the measurement record validated against the
Observation schema. One seam for shape checks, censor alignment,
and line-name subsetting (spec 2026-07-23, sections 3.2-3.3)."""
import jax.numpy as jnp
import numpy as np
import pytest


def _obs(n=3):
    from tengri.observation import Observation, Photometry
    names = ["sdss_g", "sdss_r", "sdss_i"][:n]
    return Observation(photometry=Photometry.from_names(names))


def test_photometry_shapes_validate():
    from tengri import Data
    d = Data(photometry=(jnp.ones(3), jnp.full(3, 0.1)))
    v = d.validate_against(_obs(3))
    assert v.flux.shape == (3,) and v.noise.shape == (3,)


def test_wrong_band_count_raises_naming_both():
    from tengri import Data
    d = Data(photometry=(jnp.ones(4), jnp.full(4, 0.1)))
    with pytest.raises(ValueError, match=r"4.*3|3.*4"):
        d.validate_against(_obs(3))


def test_censor_must_align_and_be_flags_not_bool():
    from tengri import Data
    ok = Data(photometry=(jnp.ones(3), jnp.full(3, 0.1)),
              censor=jnp.array([0, 1, -1]))
    ok.validate_against(_obs(3))                      # 0/1/-1 fine
    boolean = Data(photometry=(jnp.ones(3), jnp.full(3, 0.1)),
                   censor=jnp.array([True, False, True]))
    with pytest.raises(ValueError, match="censor"):   # bool rejected (spec 3.3)
        boolean.validate_against(_obs(3))


def test_lines_must_be_subset_of_schema_linelist():
    from tengri import Data
    d = Data(photometry=(jnp.ones(3), jnp.full(3, 0.1)),
             lines={"Halpha": (3.2e-17, 0.4e-17)})
    with pytest.raises(ValueError, match="Halpha"):   # obs declares no LineList
        d.validate_against(_obs(3))


def test_nan_in_single_galaxy_data_raises():
    from tengri import Data
    d = Data(photometry=(jnp.array([1.0, np.nan, 1.0]), jnp.full(3, 0.1)))
    with pytest.raises(ValueError, match=r"NaN.*sdss_r|sdss_r.*NaN"):
        d.validate_against(_obs(3))                   # single-galaxy Data is complete (spec 3.3)


def test_empty_data_rejected():
    from tengri import Data
    with pytest.raises(ValueError):
        Data().validate_against(_obs(3))
```

- [ ] **Step 2: Run — FAIL (`ImportError: cannot import name 'Data'`).** Paste.
- [ ] **Step 3: Implement `src/tengri/observation/data.py`.**

```python
# SPDX-License-Identifier: BSD-3-Clause
"""The measurement record: what came back from the telescope.

``Observation`` is the schema (instrument: filters, wave grid, noise
character, which lines); ``Data`` is one record conforming to it.
Validation happens in exactly one place — ``validate_against`` — so
shape errors, boolean-censor traps, NaNs, and unknown line names all
fail loudly with the offending channel named. See the API spec
(2026-07-23) sections 3.2-3.3 and issue #1321.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class ValidatedData(NamedTuple):
    flux: jnp.ndarray | None          # (n_filters,) [erg/s/cm^2/Hz]
    noise: jnp.ndarray | None         # (n_filters,)
    spec_flux: jnp.ndarray | None     # (n_pix,)
    spec_noise: jnp.ndarray | None    # (n_pix,)
    censor: jnp.ndarray | None        # (n_filters,) in {0, 1, -1}
    line_values: dict | None          # name -> (value, err)


@dataclasses.dataclass(frozen=True)
class Data:
    """One galaxy's measurements, validated against an ``Observation``.

    Parameters
    ----------
    photometry : tuple of (flux, err) or None
        Each ``array_like, shape (n_filters,)`` [erg/s/cm^2/Hz].
    spectrum : tuple of (flux, err) or None
        Each ``array_like, shape (n_pix,)`` [erg/s/cm^2/Hz].
    lines : dict or None
        ``{line_name: (value, err)}`` [erg/s/cm^2]; names must be a
        subset of the observation's ``LineList``.
    censor : array_like or None
        Per-band censoring flags, shape ``(n_filters,)``: ``0`` =
        detected, ``1`` = upper limit, ``-1`` = lower limit. Boolean
        arrays are rejected (they silently invert the semantics).
    """

    photometry: tuple | None = None
    spectrum: tuple | None = None
    lines: dict | None = None
    censor: object | None = None

    def validate_against(self, observation) -> ValidatedData:
        if (self.photometry is None and self.spectrum is None
                and not self.lines):
            raise ValueError("Data is empty: provide photometry=, "
                             "spectrum=, or lines=.")
        flux = noise = spec_flux = spec_noise = censor = None
        if self.photometry is not None:
            phot_schema = getattr(observation, "photometry", None)
            if phot_schema is None:
                raise ValueError("Data has photometry but the Observation "
                                 "declares no photometric filters.")
            flux, noise = (jnp.asarray(a) for a in self.photometry)
            n = phot_schema.n_filters
            if flux.shape != (n,) or noise.shape != (n,):
                raise ValueError(
                    f"photometry shape {flux.shape} does not match the "
                    f"observation's {n} filters "
                    f"({', '.join(phot_schema.names)}).")
            bad = np.flatnonzero(~np.isfinite(np.asarray(flux)))
            if bad.size:
                names = [phot_schema.names[i] for i in bad]
                raise ValueError(
                    f"NaN/inf flux in bands {names}: a single-galaxy Data "
                    "must be complete — drop the filter from the "
                    "Observation instead (spec 3.3).")
        if self.censor is not None:
            c = np.asarray(self.censor)
            if c.dtype == bool:
                raise ValueError(
                    "censor must use flags 0/1/-1 (0=detected, 1=upper, "
                    "-1=lower); boolean arrays are rejected because True "
                    "would silently mean 'upper limit'.")
            if flux is None or c.shape != flux.shape:
                raise ValueError("censor must align with photometry, "
                                 f"got shape {c.shape}.")
            censor = jnp.asarray(c)
        if self.spectrum is not None:
            spec_schema = getattr(observation, "spectroscopy", None)
            if spec_schema is None:
                raise ValueError("Data has a spectrum but the Observation "
                                 "declares no spectroscopy.")
            spec_flux, spec_noise = (jnp.asarray(a) for a in self.spectrum)
            npix = spec_schema.wave_obs.shape[0]
            if spec_flux.shape != (npix,):
                raise ValueError(f"spectrum shape {spec_flux.shape} does "
                                 f"not match the wave grid ({npix} pix).")
        if self.lines:
            declared = getattr(observation, "lines", None)
            declared_names = set(getattr(declared, "names", []) or [])
            unknown = set(self.lines) - declared_names
            if unknown:
                raise ValueError(
                    f"lines {sorted(unknown)} are not declared in the "
                    "Observation's LineList — declare WHICH lines on the "
                    "schema; supply their VALUES here (spec 3.2).")
        return ValidatedData(flux, noise, spec_flux, spec_noise,
                             censor, self.lines)
```

Adapt attribute names (`n_filters`, `names`, `wave_obs`) to the real `Photometry`/`Spectroscopy` fields — verify with `rg -n "n_filters|names" src/tengri/observation/photometry_config.py | head`; STOP if they differ beyond renames. `observation.lines` does not exist until T3 — the subset test passes today because `getattr(..., "lines", None)` yields the unknown-name error.

- [ ] **Step 4: Exports.** Add `Data` to `src/tengri/observation/__init__.py` and the top-level `src/tengri/__init__.py` `__all__` (Tier 1 docstring rules apply — the class docstring above already carries Parameters with units).
- [ ] **Step 5: Run — all PASS.** Paste. Lint. Commit `feat(observation): Data measurement record with single-seam validation. Refs #1321`.

---

### Task T2: `fit()` accepts `Data` (bare arrays remain sugar)

**Executor:** sonnet
**Branch:** `feat/fit-accepts-data`
**Depends on:** T1 merged (imports `Data`, `ValidatedData`).

**Files:**
- Modify: `src/tengri/forward/forward_model.py:647-705` (`ForwardModel.fit`)
- Test: `tests/unit/inference/test_fit_data_record.py` (create)

**Interfaces:**
- Consumes: `Data.validate_against(observation) -> ValidatedData` (T1).
- Produces: `fwd.fit(data_or_flux, noise=None, **kw)` — first positional may be a `Data`; unpacking maps `flux/noise → Fitter(data=, noise=)`, `censor → Fitter(data_mask=)`, `spectrum → the joint photometry=/spectrum= kwargs`, `line_values →` the existing line-flux route.

- [ ] **Step 1: Locate the line-flux route.** Today measured line fluxes ride on `Observation.line_fluxes` (`LineFluxData`). Run and paste: `rg -n "line_fluxes|LineFluxData" src/tengri/inference/fitter.py src/tengri/observation/observation.py | head -15`. Identify how the Fitter reads them. T2 routes `Data.lines` to the same consumption point **without** touching the likelihood: construct the `LineFluxData` internally from `Data.lines` (wavelengths resolved from the schema's LineList once T3 lands; until then, if the observation has no LineList, `Data.lines` raises in validation — acceptable, tested in T1).
- [ ] **Step 2: Failing tests:**

```python
# tests/unit/inference/test_fit_data_record.py
"""#1321: fit() accepts a Data record; bare arrays stay as sugar."""
import jax
import jax.numpy as jnp
import pytest


def test_bare_arrays_and_data_record_agree(synthetic_ssp, minimal_obs, mock_flux):
    from tengri import SEDModel, ForwardModel, Data
    sed = SEDModel.build(ssp_data=synthetic_ssp, observation=minimal_obs,
                         sfh={"type": "dpl"})
    fwd = ForwardModel.build(sed=sed)
    flux, err = mock_flux
    key = jax.random.PRNGKey(0)
    p_arrays = fwd.fit(flux, err, method="map", key=key)
    p_record = fwd.fit(Data(photometry=(flux, err)), method="map", key=key)
    assert jnp.allclose(p_arrays.map_params_array(), p_record.map_params_array())
    # adapt the comparison accessor: use whatever Posterior exposes for the
    # MAP point (rg -n "def map|best" src/tengri/inference/posterior.py | head)


def test_data_censor_reaches_data_mask(synthetic_ssp, minimal_obs, mock_flux, monkeypatch):
    from tengri import SEDModel, ForwardModel, Data
    from tengri.inference import fitter as fitter_mod
    seen = {}
    orig = fitter_mod.Fitter.__init__
    def spy(self, model, data=None, noise=None, **kw):
        seen.update(kw)
        return orig(self, model, data=data, noise=noise, **kw)
    monkeypatch.setattr(fitter_mod.Fitter, "__init__", spy)
    sed = SEDModel.build(ssp_data=synthetic_ssp, observation=minimal_obs,
                         sfh={"type": "dpl"})
    fwd = ForwardModel.build(sed=sed)
    flux, err = mock_flux
    fwd.fit(Data(photometry=(flux, err), censor=jnp.array([0, 1, 0])),
            method="map", key=jax.random.PRNGKey(0))
    assert "data_mask" in seen and seen["data_mask"] is not None


def test_data_plus_noise_kwarg_is_an_error(synthetic_ssp, minimal_obs, mock_flux):
    from tengri import SEDModel, ForwardModel, Data
    sed = SEDModel.build(ssp_data=synthetic_ssp, observation=minimal_obs,
                         sfh={"type": "dpl"})
    fwd = ForwardModel.build(sed=sed)
    flux, err = mock_flux
    with pytest.raises(TypeError, match="Data.*noise"):
        fwd.fit(Data(photometry=(flux, err)), err, method="map",
                key=jax.random.PRNGKey(0))
```

Fixtures: reuse the wave-0 `synthetic_ssp`/`minimal_obs` fixtures (they exist after Wave 0; locate with `rg -n "def synthetic_ssp|def minimal_obs" tests/`); `mock_flux` = a 3-band flux/err pair generated from `sed.predict_photometry` on defaults + 5% noise, built as a local fixture.

- [ ] **Step 3: Run — FAIL** (Data not accepted / TypeError from array coercion). Paste.
- [ ] **Step 4: Implement** at the top of `ForwardModel.fit`:

```python
from tengri.observation.data import Data as _Data
if isinstance(data, _Data):
    if noise is not None:
        raise TypeError("fit(Data, noise=...) is ambiguous: the Data "
                        "record already carries its uncertainties.")
    v = data.validate_against(self.observation)
    kwargs.setdefault("data_mask", v.censor)
    if v.spec_flux is not None and v.flux is not None:
        kwargs.setdefault("photometry", (v.flux, v.noise))
        kwargs.setdefault("spectrum", (v.spec_flux, v.spec_noise))
        data, noise = None, None
    elif v.spec_flux is not None:
        data, noise = v.spec_flux, v.spec_noise
    else:
        data, noise = v.flux, v.noise
    # v.line_values routing: see Step 1 findings; wire to the same
    # entry point Observation.line_fluxes reaches, or raise
    # NotImplementedError("Data.lines lands with the T3 schema change")
    # if the route requires the LineList field that T3 adds.
```

Match the surrounding delegation style (`ForwardModel.fit` already forwards to `Fitter(...).run(...)`); keep the docstring's Parameters section updated (`data : array_like or Data`).

- [ ] **Step 5: Run — PASS.** Paste. Sweep `-k "fit and not slow"`. Lint. Commit `feat(forward): fit() accepts the Data record; censor routes to data_mask. Refs #1321`.

---

### Task T3: the razor on `Observation` — `lines=` in, measured values deprecated

**Executor:** sonnet
**Branch:** `feat/observation-razor`
**Depends on:** T1, T2 merged.

**Files:**
- Modify: `src/tengri/observation/observation.py` (`Observation` dataclass, fields at ~189-196, `__post_init__`)
- Test: `tests/unit/observation/test_observation_razor.py` (create)

**Interfaces:**
- Produces: `Observation(lines=LineList([...]))` — schema-side declaration; `Observation(line_fluxes=…)` / `spectral_indices=` / `line_ratios=` keep working but emit one-shot `DeprecationWarning` pointing at `Data`; `Data.lines` line names resolve wavelengths through the schema `LineList`.

- [ ] **Step 1: Locate `LineList` and the deprecation helper.** Paste: `rg -n "class LineList" src/tengri/ | head -3` and `rg -n "def deprecated_attribute|def deprecated_alias" src/tengri/_deprecated.py`.
- [ ] **Step 2: Failing tests:**

```python
# tests/unit/observation/test_observation_razor.py
"""#1321: Observation is schema-only. lines= declares WHICH lines;
value-carrying fields (line_fluxes, spectral_indices, line_ratios)
deprecate toward Data with a one-shot warning."""
import pytest


def test_lines_field_accepts_linelist():
    from tengri.observation import Observation, Photometry
    from tengri import LineList     # adapt import from Step 1
    obs = Observation(photometry=Photometry.from_names(["sdss_r"]),
                      lines=LineList(["Halpha"]))
    assert "Halpha" in obs.lines.names


def test_line_fluxes_field_warns_deprecation(minimal_line_flux_data):
    from tengri.observation import Observation, Photometry
    with pytest.warns(DeprecationWarning, match="Data"):
        Observation(photometry=Photometry.from_names(["sdss_r"]),
                    line_fluxes=minimal_line_flux_data)


def test_data_lines_resolve_through_schema(minimal_obs_with_lines):
    from tengri import Data
    import jax.numpy as jnp
    d = Data(photometry=(jnp.ones(1), jnp.full(1, 0.1)),
             lines={"Halpha": (3.2e-17, 0.4e-17)})
    v = d.validate_against(minimal_obs_with_lines)
    assert v.line_values == {"Halpha": (3.2e-17, 0.4e-17)}
```

`minimal_line_flux_data`: construct the smallest valid `LineFluxData` (read its class for required fields; local fixture). `minimal_obs_with_lines`: `Observation(photometry=…, lines=LineList(["Halpha"]))`.

- [ ] **Step 3: Run — FAIL** (`Observation` has no `lines` field; no warning emitted). Paste.
- [ ] **Step 4: Implement.** Add `lines: object | None = None` to the frozen dataclass (after `spectral_indices`); in `__post_init__`, emit the one-shot `DeprecationWarning` when any of `line_fluxes` / `spectral_indices` / `line_ratios` is not None: `"Observation(<field>=...) carries measured values on the instrument schema and is deprecated: declare WHICH lines with lines=LineList([...]) and supply the VALUES per galaxy via Data(lines=...). See #1321."` Use the repo's one-shot-warning idiom (find it: `rg -n "one-shot|_warned" src/tengri/config/deprecation.py src/tengri/_deprecated.py | head`). Do NOT remove the fields or change how the Fitter consumes them — this wave adds the new path and marks the old; removal is a later major-version task.
- [ ] **Step 5:** finish T2's line routing if it was left `NotImplementedError`: `Data.lines` + schema `LineList` → construct the internal `LineFluxData` (wavelengths from the LineList's vacuum-wavelength lookup — verify: `rg -n "vacuum|wavelength" <linelist file> | head`) and pass it down the exact path Step 1 of T2 found.
- [ ] **Step 6: Run — PASS; the full observation test tree stays green** (`…pytest tests/ -q -n 2 -k "observation"`). Paste. Lint. Commit `feat(observation): schema-side lines=LineList; deprecate value-carrying fields toward Data. Refs #1321`.

---

### Task T4: `mode=` on `ForwardModel.build` — inferred, assertable, validated

**Executor:** haiku
**Branch:** `feat/build-mode`
**Depends on:** Wave 0 T10 (#1315) merged. Independent of T1–T3.

**Files:**
- Modify: `src/tengri/forward/forward_model.py:311-402` (`build`)
- Test: `tests/unit/forward/test_build_mode.py` (create)

- [ ] **Step 1: Failing tests:**

```python
"""#1321: mode= is inferred from kwargs by default, assertable
explicitly, and validated with ONE mode-aware error."""
import pytest


def _sed(synthetic_ssp, minimal_obs):
    from tengri import SEDModel
    return SEDModel.build(ssp_data=synthetic_ssp, observation=minimal_obs,
                          sfh={"type": "dpl"})


def test_mode_inferred_single(synthetic_ssp, minimal_obs):
    from tengri import ForwardModel
    fwd = ForwardModel.build(sed=_sed(synthetic_ssp, minimal_obs))
    assert fwd.mode == "single"


def test_mode_asserted_matches(synthetic_ssp, minimal_obs):
    from tengri import ForwardModel
    fwd = ForwardModel.build(mode="single", sed=_sed(synthetic_ssp, minimal_obs))
    assert fwd.mode == "single"


def test_mode_mismatch_is_one_clear_error(synthetic_ssp, minimal_obs):
    from tengri import ForwardModel
    with pytest.raises(ValueError, match="mode='multi_population'.*populations="):
        ForwardModel.build(mode="multi_population",
                           sed=_sed(synthetic_ssp, minimal_obs))


def test_mode_hierarchical_reserved(synthetic_ssp, minimal_obs):
    from tengri import ForwardModel
    with pytest.raises(NotImplementedError, match="1319"):
        ForwardModel.build(mode="hierarchical",
                           sed=_sed(synthetic_ssp, minimal_obs))


def test_unknown_mode_lists_valid_ones(synthetic_ssp, minimal_obs):
    from tengri import ForwardModel
    with pytest.raises(ValueError, match="single.*multi_population.*hierarchical"):
        ForwardModel.build(mode="banana", sed=_sed(synthetic_ssp, minimal_obs))
```

- [ ] **Step 2: Run — FAIL (`build` rejects `mode` kwarg; no `.mode` attribute).** Paste.
- [ ] **Step 3: Implement.** `mode: str | None = None` in the signature. Inference: `"multi_population"` if `populations is not None`, `"hierarchical"` if `population is not None` (the current PopulationSEDModel branch — it stays functional; the *reserved* error applies only to `mode="hierarchical"` asserted WITHOUT `population=`, since the #1319 `shared=` construction does not exist yet), else `"single"`. Assertion: if `mode` given and ≠ inferred → `ValueError(f"mode={mode!r} requires {needed}= (got {what_was_given}); inferred mode would be {inferred!r}")`. Valid-mode check first: `{"single", "hierarchical", "multi_population"}`. Store `object.__setattr__`-free — `ForwardModel` is a plain dataclass with `populations`/`observation`; add `mode: str = "single"` field. Error messages must name the missing kwarg exactly (the grammar-error-that-raises memory class: every suggested form in an error must itself be valid).
- [ ] **Step 4: Run — PASS; sweep `-k "forward_model or build"`.** Paste. Lint. Commit `feat(forward): mode= on ForwardModel.build — inferred, assertable, one mode-aware error. Refs #1321`.

---

### Task T5: per-band error floors on `NoiseModel` (#919, first half)

**Executor:** haiku
**Branch:** `feat/noise-per-band-floors`
**Independent** of T1–T4.

**Files:**
- Modify: `src/tengri/observation/noise_model.py:16-40` (`NoiseModel`)
- Locate + modify: the single place `calibration_floor` enters `sigma_eff` (`rg -n "calibration_floor|sigma_eff" src/tengri/inference/ src/tengri/observation/ | head -15` — STOP if more than one consumption site).
- Test: `tests/unit/observation/test_noise_per_band_floor.py` (create)

- [ ] **Step 1: Failing test:**

```python
"""#919: calibration_floor accepts a per-band array — sigma_eff must
apply each band's own floor: sqrt(sigma^2 + (f_b * model_b)^2)."""
import jax.numpy as jnp
import numpy as np


def test_per_band_floor_shapes_and_values():
    from tengri.observation import NoiseModel
    nm = NoiseModel(calibration_floor=jnp.array([0.02, 0.10, 0.05]))
    model_flux = jnp.array([1.0, 1.0, 2.0])
    sigma_obs = jnp.array([0.1, 0.1, 0.1])
    sigma_eff = nm.effective_sigma(sigma_obs, model_flux)   # adapt to the real seam name from Step 1 if a method exists; else test through the located consumption function
    expected = np.sqrt(sigma_obs**2 + (np.array([0.02, 0.10, 0.05]) * model_flux)**2)
    np.testing.assert_allclose(np.asarray(sigma_eff), expected, rtol=1e-12)
```

- [ ] **Step 2: Run — FAIL** (scalar-only floor / no array support). Paste.
- [ ] **Step 3: Implement:** widen the field type (`float | Distribution | jnp.ndarray`), thread the array through the one consumption site (broadcasting handles scalar vs `(n_filters,)` identically — the formula does not change), and validate length against the data at the existing validation point (mismatched length → `ValueError` naming both). A per-band **free** floor (array of Distributions) is OUT of scope — reject with `TypeError` and a message saying so.
- [ ] **Step 4: Run — PASS; sweep `-k noise`.** Paste. Lint. Commit `feat(observation): per-band calibration floors on NoiseModel. Refs #919`.

---

## Review protocol additions for this wave (orchestrator)

Beyond the Wave 0 gate: (1) T2's agreement test is the load-bearing one — re-run it myself and also diff the two `Fitter.__init__` kwarg dicts (spy) to confirm the record path adds nothing the array path lacks; (2) T3's deprecation must be ONE-shot (run a loop of 3 constructions, count warnings); (3) API-reference check: `python tools/check_doc_examples.py` after T1/T3 (new public names in docstrings).

## Self-review record

- Spec coverage: §3.1 (T3), §3.2 (T1+T2), §3.3 (T1 censor/NaN + T2 data_mask), §4 mode= (T4), #919 absorption (T5). §3.2's Catalog-side "N records" is Wave 2 by design.
- Placeholder scan: all "adapt" points carry a locator command + STOP clause; every code step shows code.
- Type consistency: `ValidatedData` fields (T1) are consumed by exactly those names in T2; `lines=LineList` (T3) matches T1's `getattr(observation, "lines")` probe.

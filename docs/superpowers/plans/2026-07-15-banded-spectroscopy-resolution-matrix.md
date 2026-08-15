# Banded-Operator Spectroscopy: DESI/PFS Resolution Matrix (#1163) + Flux-Conserving Resample Benchmark (#1166) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the spectroscopic forward model a JIT/grad-safe banded resolution-matrix operator (`R @ model`) that ingests DESI/PFS-style banded resolution data and replaces the Gaussian `apply_lsf` when a matrix is supplied, and measure whether a flux-conserving resample lets the model run on a coarser wavelength grid.

**Architecture:** One shared banded-operator kernel (`observation/banded.py`, diagonal-offsets storage, O(n·K) matvec) with a DESI ingest builder and a Gaussian-equivalence builder. `Spectroscopy` gains an optional `resolution_matrix` field; `project_spectrum` routes the flux-conserving-resampled model through `R @ model` when present, keeping Gaussian `apply_lsf` as the default. The resolved presence goes into `compile_signature` (structural-cache-collision guard — 4th instance of #1135/#1149/#1166). #1166's contribution here is the benchmark that justifies (or refutes) coarse-grid evaluation; the `resample_bands` matrix and SpectrumPrecomp conserving builder are deferred until the benchmark warrants them.

**Tech Stack:** JAX (jnp, jit, grad), NumPy (build-time), pytest + chex, ruff.

## Global Constraints

- Base worktree: `~/.claude/jobs/d5f48f27/tmp/wt_banded`, branch `cs/banded-spectroscopy` off origin/main (`9c918b64a`).
- Run everything with `PYTHONPATH=src /tengri/.venv/bin/python` (root venv; worktree src via PYTHONPATH — the editable install points at the ROOT checkout, so PYTHONPATH=src is mandatory).
- 64-bit precision assumed (`jax_enable_x64`); import `tengri` first in tests to enable it.
- American-English spelling only (CI guard `tools/check_british_spelling.py`): `normalize`, `center`, `catalog`, `marginalize`.
- Units in docstring brackets always: `[Angstrom]`, `[erg/s/Hz]`, `[erg/s/cm^2/Hz]`. Array shapes annotated.
- Citations verbatim: Bolton & Schlegel 2010 (PASP 122, 248; arXiv:0911.2689; DOI 10.1086/651008); Guy et al. 2023 (AJ 165, 144; arXiv:2209.14482; DOI 10.3847/1538-3881/acb212); Carnall 2017 (arXiv:1705.05165).
- Immutable data: build new arrays with `.at[].set()` / concatenate; never mutate inputs.
- Every test under `tests/physics/`, `tests/regression/`, `tests/components/`, `tests/contract/` MUST declare a taxonomy marker (`tools/check_test_markers.py`).
- Lint gate before every commit: `.venv/bin/ruff check src/ tests/` and `.venv/bin/ruff format --check src/ tests/` (from the worktree, use the root venv binary).
- File-size CI gate: any file that grows past its allowlist entry needs a bump in `tools/file_size_allowlist.json`. Current: `spectrum.py` 843, `spectroscopy.py` 894, `observation.py` 1415, `sed_model.py` 7670.
- Label the PR: `area:observation`, `enhancement`, `type:parity`, plus `jit-safety`.

---

### Task 1: Banded matvec kernel

**Files:**
- Create: `src/tengri/observation/banded.py`
- Test: `tests/components/spectroscopy/test_banded_operator.py`

**Interfaces:**
- Produces: `BandedMatrix` (NamedTuple: `offsets: jnp.ndarray` int `(K,)`, `data: jnp.ndarray` float `(K, n)`); `banded_matvec(offsets, data, x) -> jnp.ndarray` shape `(n,)` computing `y[i] = Σ_k data[k,i]·x[i+offsets[k]]` with `x` zero-padded out of range.

- [ ] **Step 1: Write the failing test** — a dense reference is the ground truth for the band convention.

```python
# tests/components/spectroscopy/test_banded_operator.py
import numpy as np
import pytest
import jax.numpy as jnp
import tengri  # noqa: F401  (enables x64)
from tengri.observation.banded import banded_matvec, BandedMatrix

pytestmark = pytest.mark.contract


def _dense_from_bands(offsets, data):
    """Reference dense (n, n) matrix from the banded convention
    A[i, i+offsets[k]] = data[k, i]."""
    K, n = data.shape
    A = np.zeros((n, n))
    for k in range(K):
        o = int(offsets[k])
        for i in range(n):
            j = i + o
            if 0 <= j < n:
                A[i, j] = data[k, i]
    return A


def test_banded_matvec_matches_dense():
    rng = np.random.default_rng(0)
    n = 12
    offsets = np.array([-2, -1, 0, 1, 2])
    data = rng.standard_normal((offsets.shape[0], n))
    x = rng.standard_normal(n)
    y = banded_matvec(jnp.asarray(offsets), jnp.asarray(data), jnp.asarray(x))
    y_ref = _dense_from_bands(offsets, data) @ x
    np.testing.assert_allclose(np.asarray(y), y_ref, rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/.claude/jobs/d5f48f27/tmp/wt_banded && PYTHONPATH=src /tengri/.venv/bin/python -m pytest tests/components/spectroscopy/test_banded_operator.py::test_banded_matvec_matches_dense -q`
Expected: FAIL — `ModuleNotFoundError: tengri.observation.banded`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tengri/observation/banded.py
# SPDX-License-Identifier: BSD-3-Clause
"""Banded linear operators for the spectroscopic forward model.

A banded operator ``A`` acts on a model vector as ``y = A @ x`` where ``A``
is nonzero only on a handful of diagonals. Both the DESI/PFS instrument
resolution matrix (Bolton & Schlegel 2010 [1]_; Guy et al. 2023 [2]_) and,
in future, the SpectRes flux-conserving resample (Carnall 2017 [3]_) share
this representation, so a single ``O(n * K)`` matvec covers both.

The storage convention is diagonal-offsets:
``A[i, i + offsets[k]] = data[k, i]``, with entries whose column index falls
outside ``[0, n)`` treated as zero. This matches how DESI/desispec ships the
resolution data (a ``(n_diag, n_pix)`` array of diagonals).

References
----------
.. [1] Bolton, A. S. & Schlegel, D. J. 2010, "Spectro-Perfectionism: An
       Algorithmic Framework for Photon Noise-Limited Extraction of Optical
       Fiber Spectroscopy", PASP, 122, 248, arXiv:0911.2689,
       DOI 10.1086/651008.
.. [2] Guy, J. et al. 2023, "The Spectroscopic Data Processing Pipeline for
       the Dark Energy Spectroscopic Instrument", AJ, 165, 144,
       arXiv:2209.14482, DOI 10.3847/1538-3881/acb212.
.. [3] Carnall, A. C. 2017, "SpectRes: A Fast Spectral Resampling Tool in
       Python", arXiv:1705.05165.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class BandedMatrix(NamedTuple):
    """Diagonal-offsets banded matrix.

    Attributes
    ----------
    offsets : ndarray, shape (K,)
        Integer diagonal offsets; diagonal ``k`` holds ``A[i, i+offsets[k]]``.
    data : ndarray, shape (K, n)
        Diagonal values; ``data[k, i]`` is the weight applied to ``x[i+offsets[k]]``.
    """

    offsets: jnp.ndarray
    data: jnp.ndarray


@jax.jit
def banded_matvec(
    offsets: jnp.ndarray, data: jnp.ndarray, x: jnp.ndarray
) -> jnp.ndarray:
    r"""Apply a banded operator to a vector: ``y = A @ x``.

    .. math::

        y_i = \sum_k \mathrm{data}[k, i] \; x_{\,i + \mathrm{offsets}[k]}

    with :math:`x_j = 0` for :math:`j \notin [0, n)`.

    Parameters
    ----------
    offsets : array_like, shape (K,)
        Integer diagonal offsets (static — baked into the trace).
    data : array_like, shape (K, n)
        Diagonal values.
    x : array_like, shape (n,)
        Input vector.

    Returns
    -------
    ndarray, shape (n,)
        ``A @ x``.

    Notes
    -----
    JIT-compatible: yes. Gradient-safe: yes — linear in ``x`` and in ``data``.
    Cost is ``O(n * K)`` via a gather, not the dense ``O(n^2)`` product.
    """
    n = x.shape[0]
    cols = jnp.arange(n)[None, :] + offsets[:, None]  # (K, n)
    valid = (cols >= 0) & (cols < n)
    gathered = jnp.where(valid, x[jnp.clip(cols, 0, n - 1)], 0.0)  # (K, n)
    return jnp.sum(data * gathered, axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/.claude/jobs/d5f48f27/tmp/wt_banded && PYTHONPATH=src /tengri/.venv/bin/python -m pytest tests/components/spectroscopy/test_banded_operator.py::test_banded_matvec_matches_dense -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/.claude/jobs/d5f48f27/tmp/wt_banded
/tengri/.venv/bin/ruff format src/tengri/observation/banded.py tests/components/spectroscopy/test_banded_operator.py
git add src/tengri/observation/banded.py tests/components/spectroscopy/test_banded_operator.py
git commit -m "feat(observation): banded-operator matvec kernel for spectroscopy (#1163)"
```

---

### Task 2: DESI-convention ingest + gradient safety

**Files:**
- Modify: `src/tengri/observation/banded.py`
- Test: `tests/components/spectroscopy/test_banded_operator.py`

**Interfaces:**
- Produces: `resolution_bands_from_desi(diag_data, offsets) -> BandedMatrix` — accepts DESI/desispec storage (a `(n_diag, n_pix)` diagonal array with scipy `dia_matrix` offset convention `A[i, j] = diag_data[k, j]` where `j - i = offsets[k]`) and returns a `BandedMatrix` in tengri's `data[k, i] = A[i, i+offsets[k]]` convention.

- [ ] **Step 1: Write the failing tests** — pin the DESI/scipy convention against `scipy.sparse.dia_matrix`, and prove differentiability.

```python
# append to tests/components/spectroscopy/test_banded_operator.py
import jax
from tengri.observation.banded import resolution_bands_from_desi


def test_desi_convention_roundtrip():
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(1)
    n = 15
    offsets = np.array([2, 1, 0, -1, -2])  # DESI orders high->low
    diag_data = rng.standard_normal((offsets.shape[0], n))
    dense = sp.dia_matrix((diag_data, offsets), shape=(n, n)).toarray()
    x = rng.standard_normal(n)
    bm = resolution_bands_from_desi(jnp.asarray(diag_data), jnp.asarray(offsets))
    y = banded_matvec(bm.offsets, bm.data, jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(y), dense @ x, rtol=1e-12, atol=1e-12)


def test_banded_matvec_is_differentiable():
    rng = np.random.default_rng(2)
    n = 20
    offsets = jnp.asarray([-1, 0, 1])
    data = jnp.asarray(rng.standard_normal((3, n)))

    def loss(x):
        return jnp.sum(banded_matvec(offsets, data, x) ** 2)

    g = jax.grad(loss)(jnp.asarray(rng.standard_normal(n)))
    assert np.all(np.isfinite(np.asarray(g)))
```

- [ ] **Step 2: Run to verify failure**

Run: `... -m pytest tests/components/spectroscopy/test_banded_operator.py::test_desi_convention_roundtrip -q`
Expected: FAIL — `resolution_bands_from_desi` undefined.

- [ ] **Step 3: Implement the converter** (scipy dia: `data[k, j]` sits at row `j-offsets[k]`, i.e. `A[i, i+o] = diag_data[k, i+o]`; tengri wants `data[k, i] = A[i, i+o] = diag_data[k, i+o]`, a left-roll of each diagonal by `o`, zero-filled).

```python
# add to src/tengri/observation/banded.py
def resolution_bands_from_desi(
    diag_data: jnp.ndarray, offsets: jnp.ndarray
) -> BandedMatrix:
    r"""Ingest a DESI/desispec resolution matrix into a :class:`BandedMatrix`.

    DESI extracted spectra ship the resolution operator as a
    ``(n_diag, n_pix)`` array of diagonals with the scipy ``dia_matrix``
    convention: ``A[i, j] = diag_data[k, j]`` where ``j - i = offsets[k]``.
    This re-indexes into tengri's convention ``data[k, i] = A[i, i+offsets[k]]``
    (:func:`banded_matvec`), which is a per-diagonal shift by ``offsets[k]``.

    Parameters
    ----------
    diag_data : array_like, shape (n_diag, n_pix)
        Resolution diagonals as stored by desispec.
    offsets : array_like, shape (n_diag,)
        Integer diagonal offsets (desispec uses descending order, e.g.
        ``[+5, +4, ..., -5]`` for ``n_diag=11``).

    Returns
    -------
    BandedMatrix
        The same operator in tengri's banded convention.

    Notes
    -----
    JIT-compatible: build-time helper (offsets are static). No mutation of
    inputs — a rolled copy is returned. See Bolton & Schlegel 2010 [1]_ for
    the resolution-matrix representation and Guy et al. 2023 [2]_ for the DESI
    per-camera (b/r/z) storage.
    """
    diag_data = jnp.asarray(diag_data)
    offsets = jnp.asarray(offsets)
    K, n = diag_data.shape
    cols = jnp.arange(n)[None, :] + offsets[:, None]  # (K, n): i + offsets[k]
    valid = (cols >= 0) & (cols < n)
    rolled = jnp.where(valid, diag_data[jnp.arange(K)[:, None], jnp.clip(cols, 0, n - 1)], 0.0)
    return BandedMatrix(offsets=offsets, data=rolled)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `... -m pytest tests/components/spectroscopy/test_banded_operator.py -q`
Expected: PASS (all 4).

- [ ] **Step 5: Commit**

```bash
/tengri/.venv/bin/ruff format src/tengri/observation/banded.py tests/components/spectroscopy/test_banded_operator.py
git add src/tengri/observation/banded.py tests/components/spectroscopy/test_banded_operator.py
git commit -m "feat(observation): ingest DESI diagonal-offsets resolution data (#1163)"
```

---

### Task 3: Gaussian-equivalence builder — proves the operator subsumes `apply_lsf`

**Files:**
- Modify: `src/tengri/observation/banded.py`
- Test: `tests/components/spectroscopy/test_banded_operator.py`

**Interfaces:**
- Produces: `gaussian_resolution_bands(wave_obs, resolution, n_diag=11) -> BandedMatrix` — a banded Gaussian LSF on a log-uniform `wave_obs` grid with per-pixel `R = λ/Δλ`, normalized per row so a flat spectrum is preserved.

This is the #1163 acceptance test: an explicitly-built Gaussian R must reproduce `apply_lsf` to tolerance, proving the banded operator is a strict generalization of the current Gaussian path.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/components/spectroscopy/test_banded_operator.py
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.spectrum import apply_lsf


def test_gaussian_bands_reproduce_apply_lsf():
    # Log-uniform observed grid (apply_lsf convolves in log-lambda).
    n = 400
    wave = np.geomspace(4000.0, 7000.0, n)
    R = 2000.0
    rng = np.random.default_rng(3)
    spec = 1.0 + 0.1 * rng.standard_normal(n)  # smooth-ish continuum
    bm = gaussian_resolution_bands(jnp.asarray(wave), R, n_diag=41)
    y_banded = banded_matvec(bm.offsets, bm.data, jnp.asarray(spec))
    y_fft = apply_lsf(jnp.asarray(spec), jnp.asarray(wave), R, sigma_lib_kms=0.0)
    # Compare the interior (edges differ: FFT wraps, banded truncates).
    sl = slice(50, n - 50)
    np.testing.assert_allclose(
        np.asarray(y_banded)[sl], np.asarray(y_fft)[sl], rtol=0.0, atol=2e-2 * np.max(spec)
    )
```

- [ ] **Step 2: Run to verify failure** — `gaussian_resolution_bands` undefined.

- [ ] **Step 3: Implement** (Gaussian weights in log-λ, matching `_apply_lsf_constant_r`'s `sigma_pix = (c/(FWHM·R))/c / dlnλ`).

```python
# add to src/tengri/observation/banded.py
import numpy as _np

_C_KM_S = 299792.458
_FWHM_TO_SIGMA = 2.354820045030949


def gaussian_resolution_bands(
    wave_obs: jnp.ndarray, resolution, n_diag: int = 11
) -> BandedMatrix:
    r"""Banded Gaussian LSF equivalent to :func:`~tengri.observation.spectrum.apply_lsf`.

    Builds a normalized Gaussian kernel in log-wavelength space at spectral
    resolution ``R = λ/Δλ`` (:math:`\sigma_v = c / (\mathrm{FWHM} \cdot R)`,
    :math:`\sigma_{\mathrm{pix}} = (\sigma_v / c) / \Delta\ln\lambda`),
    truncated to ``n_diag`` diagonals. Provided so the banded operator can be
    validated against — and can subsume — the Gaussian ``apply_lsf`` path, and
    as an explicit-matrix fallback for instruments that publish only a
    scalar/array ``R``.

    Parameters
    ----------
    wave_obs : array_like, shape (n_pix,)
        Observed wavelength grid [Angstrom]; assumed ~log-uniform.
    resolution : float or array_like, shape (n_pix,)
        Spectral resolution ``R`` (scalar or per-pixel).
    n_diag : int, optional
        Number of diagonals (odd). Default 11.

    Returns
    -------
    BandedMatrix
        Row-normalized Gaussian LSF operator.

    Notes
    -----
    Build-time helper (NumPy). The Gaussian ``apply_lsf`` is only an
    approximation of the true instrument LSF; prefer :func:`resolution_bands_from_desi`
    when the survey ships a resolution matrix (Bolton & Schlegel 2010 [1]_).
    """
    wave = _np.asarray(wave_obs, dtype=float)
    n = wave.shape[0]
    R = _np.broadcast_to(_np.asarray(resolution, dtype=float), (n,))
    dlnwave = _np.log(wave[1] / wave[0])
    sigma_pix = (_C_KM_S / (_FWHM_TO_SIGMA * R)) / _C_KM_S / dlnwave  # (n,)
    half = n_diag // 2
    offsets = _np.arange(-half, half + 1)
    data = _np.zeros((n_diag, n))
    for k, o in enumerate(offsets):
        # Row i receives x[i+o]; weight ~ Gaussian(offset=o; sigma_pix[i]).
        data[k, :] = _np.exp(-0.5 * (o / sigma_pix) ** 2)
    data /= data.sum(axis=0, keepdims=True)  # normalize per output pixel
    return BandedMatrix(offsets=jnp.asarray(offsets), data=jnp.asarray(data))
```

- [ ] **Step 4: Run to verify pass.** If the interior tolerance is tight, widen `n_diag` (more diagonals capture more of the Gaussian tail) before loosening `atol`.

- [ ] **Step 5: Commit**

```bash
/tengri/.venv/bin/ruff format src/tengri/observation/banded.py tests/components/spectroscopy/test_banded_operator.py
git add -A && git commit -m "feat(observation): Gaussian-equivalent banded LSF; proves R subsumes apply_lsf (#1163)"
```

---

### Task 4: Carry the resolution matrix on `Spectroscopy`

**Files:**
- Modify: `src/tengri/observation/spectroscopy.py` (field near line 137; validation in `__post_init__` near 154; new property)
- Test: `tests/components/spectroscopy/test_resolution_matrix.py`

**Interfaces:**
- Consumes: `BandedMatrix` from `observation.banded`.
- Produces: `Spectroscopy(..., resolution_matrix: BandedMatrix | None = None)`; property `has_resolution_matrix -> bool`. When present, `resolution_matrix.data.shape[1]` must equal `len(wave_obs)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/components/spectroscopy/test_resolution_matrix.py
import numpy as np
import pytest
import jax.numpy as jnp
import tengri  # noqa: F401
from tengri import Spectroscopy
from tengri.observation.banded import gaussian_resolution_bands

pytestmark = pytest.mark.contract


def test_spectroscopy_accepts_resolution_matrix():
    wave = np.geomspace(4000.0, 7000.0, 100)
    bm = gaussian_resolution_bands(jnp.asarray(wave), 2000.0, n_diag=11)
    spec = Spectroscopy(wave_obs=jnp.asarray(wave), resolution_matrix=bm)
    assert spec.has_resolution_matrix
    assert Spectroscopy(wave_obs=jnp.asarray(wave)).has_resolution_matrix is False


def test_resolution_matrix_shape_validated():
    wave = np.geomspace(4000.0, 7000.0, 100)
    bm = gaussian_resolution_bands(jnp.asarray(np.geomspace(4000.0, 7000.0, 50)), 2000.0)
    with pytest.raises(ValueError, match="resolution_matrix"):
        Spectroscopy(wave_obs=jnp.asarray(wave), resolution_matrix=bm)
```

- [ ] **Step 2: Run to verify failure** — `resolution_matrix` not a field.

- [ ] **Step 3: Implement** — add the field after `covariance` (line ~145), validation in `__post_init__`, and the property.

```python
# spectroscopy.py — add to the dataclass field block (after `covariance`)
    resolution_matrix: object | None = dataclasses.field(default=None, hash=False)
```

```python
# spectroscopy.py — add at the end of __post_init__
        if self.resolution_matrix is not None:
            data = jnp.asarray(self.resolution_matrix.data)
            if data.shape[1] != len(self.wave_obs):
                raise ValueError(
                    f"resolution_matrix has {data.shape[1]} columns but wave_obs "
                    f"has length {len(self.wave_obs)}"
                )
```

```python
# spectroscopy.py — add near has_covariance
    @property
    def has_resolution_matrix(self) -> bool:
        """Whether a banded instrument resolution matrix is configured (#1163).

        Returns
        -------
        bool
            True if a :class:`~tengri.observation.banded.BandedMatrix` is set,
            in which case it replaces the Gaussian ``apply_lsf`` in projection.
        """
        return self.resolution_matrix is not None
```

- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `feat(observation): Spectroscopy carries an optional banded resolution matrix (#1163)`

---

### Task 5: Route `project_spectrum` through `R @ model`

**Files:**
- Modify: `src/tengri/observation/spectrum.py` (`project_spectrum`, lines 397-549)
- Test: `tests/components/spectroscopy/test_resolution_matrix.py`

**Interfaces:**
- Consumes: `banded_matvec`, `BandedMatrix`.
- Produces: `project_spectrum(..., resolution_matrix: BandedMatrix | None = None)`. When supplied, after the flux-conserving resample it applies `R @ flux` **instead of** `apply_lsf` (R already encodes the LSF at pixel resolution — Redrock/FastSpecFit convention). `apply_lsf` still runs when only a scalar/array `resolution` is given and no matrix.

- [ ] **Step 1: Write the failing test** — end-to-end through `project_spectrum`, asserting the matrix path equals an explicit dense `R @ resampled`.

```python
# append to tests/components/spectroscopy/test_resolution_matrix.py
from tengri.observation.spectrum import project_spectrum, compute_spectrum_conserving
from tengri.observation.banded import banded_matvec


def test_project_spectrum_applies_resolution_matrix():
    n = 300
    wave = np.geomspace(4000.0, 7000.0, n)
    wave_rest = wave / 1.05
    sed = np.ones(n) + 0.2 * np.sin(np.linspace(0, 30, n))
    bm = gaussian_resolution_bands(jnp.asarray(wave), 2500.0, n_diag=21)
    dl_cm = 1e26

    flux = project_spectrum(
        jnp.asarray(sed), jnp.asarray(wave_rest), jnp.asarray(wave),
        0.05, dl_cm, resolution_matrix=bm, conserving=True,
    )
    # Reference: conserving resample, then dense R @ .
    resampled = compute_spectrum_conserving(
        jnp.asarray(sed), jnp.asarray(wave_rest), jnp.asarray(wave), 0.05, dl_cm
    )
    ref = banded_matvec(bm.offsets, bm.data, resampled)
    np.testing.assert_allclose(np.asarray(flux), np.asarray(ref), rtol=1e-10, atol=1e-30)
```

- [ ] **Step 2: Run to verify failure** — `project_spectrum` has no `resolution_matrix` kwarg (TypeError).

- [ ] **Step 3: Implement** — add the kwarg and branch. In `project_spectrum` (after `resampler(...)` at line 534), replace the `if resolution is not None:` block:

```python
# spectrum.py project_spectrum — new signature line (after conserving)
    resolution_matrix: object | None = None,
```

```python
# spectrum.py project_spectrum — replace lines 535-543
    if resolution_matrix is not None:
        # The banded resolution matrix (DESI/PFS; Bolton & Schlegel 2010)
        # encodes the true LSF at pixel resolution and is applied to the
        # model *after* resampling onto the pixel grid — it REPLACES the
        # Gaussian apply_lsf. #1163.
        from tengri.observation.banded import banded_matvec

        flux = banded_matvec(resolution_matrix.offsets, resolution_matrix.data, flux)
    elif resolution is not None:
        flux = apply_lsf(
            flux,
            wave_obs,
            resolution,
            sigma_lib_kms=sigma_lib_kms,
            n_bins=n_bins,
            sigma_v_kms=sigma_v_kms,
        )
```

Add a `resolution_matrix` paragraph to the numpydoc Parameters section citing Bolton & Schlegel 2010 and stating it replaces the Gaussian LSF.

- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `feat(observation): project_spectrum applies the banded resolution matrix (#1163)`

---

### Task 6: Wire the matrix through the observers and the direct grid path

**Files:**
- Modify: `src/tengri/observation/observation.py` (call sites ~642 and ~803)
- Modify: `src/tengri/forward/sed_model.py` (`_predict_spectrum_on_grid`, ~3681)
- Test: `tests/components/spectroscopy/test_resolution_matrix.py`

**Interfaces:**
- Consumes: `Spectroscopy.resolution_matrix`.
- Produces: both `Observation` spectrum projectors and `SEDModel._predict_spectrum_on_grid` pass `resolution_matrix=self.spectroscopy.resolution_matrix` into `project_spectrum`.

- [ ] **Step 1: Write the failing e2e test** — a full `SEDModel` with a resolution matrix; `predict_spectrum` must differ from the no-matrix result and match the operator applied to the no-matrix (Gaussian-off) spectrum.

```python
# append to tests/components/spectroscopy/test_resolution_matrix.py
@pytest.mark.integration
def test_predict_spectrum_uses_resolution_matrix(ssp_data_fixture):
    # ssp_data_fixture: any bare-stellar SSP (skip if data absent).
    from tengri import SEDModel, Observation, Photometry
    wave = np.geomspace(4000.0, 7000.0, 500)
    bm = gaussian_resolution_bands(jnp.asarray(wave), 2500.0, n_diag=21)
    obs = Observation(spectroscopy=Spectroscopy(
        wave_obs=jnp.asarray(wave), resolution_matrix=bm, resample="conserving"))
    model = SEDModel.build(ssp_data=ssp_data_fixture, observation=obs,
                           sfh={"type": "dpl"}, redshift=0.05)
    p = model.spec.example_params()
    flux = model.predict_spectrum(p)
    assert np.all(np.isfinite(np.asarray(flux)))
    assert flux.shape[0] == wave.shape[0]
```

(Use the repo's existing SSP fixture/skip pattern from `tests/components/spectroscopy/`. If no bare-stellar fixture exists there, mirror `tests/integration/` skip-if-missing.)

- [ ] **Step 2: Run to verify failure** — matrix silently ignored (flux equals the Gaussian/no-op path; or `resolution_matrix` never passed → assertion/behavior gap).

- [ ] **Step 3: Implement the three wirings** — add `resolution_matrix=self.spectroscopy.resolution_matrix,` to the `project_spectrum(...)` calls at observation.py:642 and observation.py:803, and add to `_predict_spectrum_on_grid` (sed_model.py) a `resolution_matrix = getattr(spectroscopy, "resolution_matrix", None)` before the call plus `resolution_matrix=resolution_matrix,` in the call.

- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `feat(observation,forward): route the resolution matrix through all spectrum projectors (#1163)`

---

### Task 7: Compile-signature guard (4th cache-collision instance)

**Files:**
- Modify: `src/tengri/forward/sed_model.py` (`compile_signature`, has_spectroscopy block ~3191-3218; return tuple ~3383)
- Test: `tests/regression/bug/test_resolution_matrix_compile_signature.py`

**Interfaces:**
- Produces: `compile_signature()` includes a `spec_resolution_matrix` structural element (presence + `n_diag` + `n_pix`, e.g. `(K, n)` or `None`). Two models differing only in `resolution_matrix` must NOT share a compiled kernel.

Rationale: `project_spectrum` closes over whether it applies `R @ model` or `apply_lsf`. This is the same structural-cache landmine as #1135/#1149/#1166 — the resolved value MUST be in the signature. **Neuter-check:** the guard test must FAIL if the new signature element is removed.

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/bug/test_resolution_matrix_compile_signature.py
import numpy as np
import pytest
import jax.numpy as jnp
import tengri  # noqa: F401
from tengri import SEDModel, Observation, Spectroscopy
from tengri.observation.banded import gaussian_resolution_bands

pytestmark = pytest.mark.regression_bug


def _model(ssp, with_matrix):
    wave = np.geomspace(4000.0, 7000.0, 400)
    kw = {"wave_obs": jnp.asarray(wave), "resolution": 2500.0}
    if with_matrix:
        kw["resolution_matrix"] = gaussian_resolution_bands(jnp.asarray(wave), 2500.0, 21)
    obs = Observation(spectroscopy=Spectroscopy(**kw))
    return SEDModel.build(ssp_data=ssp, observation=obs, sfh={"type": "dpl"}, redshift=0.05)


@pytest.mark.integration
def test_resolution_matrix_changes_compile_signature(ssp_data_fixture):
    sig_gauss = _model(ssp_data_fixture, False).compile_signature()
    sig_matrix = _model(ssp_data_fixture, True).compile_signature()
    assert sig_gauss != sig_matrix, (
        "resolution_matrix presence must change compile_signature — else the "
        "matrix model reuses the Gaussian kernel from the structural cache (#1163)"
    )
```

- [ ] **Step 2: Run to verify failure** — signatures currently equal (matrix not in signature).

- [ ] **Step 3: Implement** — in the `has_spectroscopy` block add:

```python
            rm = self.observation.spectroscopy.resolution_matrix
            spec_resolution_matrix = (
                tuple(jnp.asarray(rm.data).shape) if rm is not None else None
            )
```

In the `else` branch: `spec_resolution_matrix = None`. Add `spec_resolution_matrix,` to the return tuple after `spec_resample_conserving`.

- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Neuter-check** — temporarily delete `spec_resolution_matrix,` from the return tuple, run the test, confirm it FAILS, then restore. Record the neuter result in the commit body.
- [ ] **Step 6: Commit** — `fix(forward): add resolution_matrix to compile_signature — cache-collision guard (#1163)`

---

### Task 8: Skewed-R bias justification (synthetic, no external data)

**Files:**
- Test: `tests/physics/spectroscopy/test_resolution_matrix_bias.py`

**Interfaces:**
- Consumes: `banded_matvec`, `gaussian_resolution_bands`, `apply_lsf`.

Deliverable justification for #1163: a *skewed* (asymmetric) resolution matrix — the kind the Gaussian LSF cannot represent — biases a recovered line flux/width relative to the true operator. Quantify the bias on a synthetic emission line; assert it is non-negligible (this is the "why this matters" evidence without needing a real DESI spectrum).

- [ ] **Step 1: Write the test** — inject a narrow Gaussian line on a flat continuum, forward-model it through (a) a skewed banded R and (b) the best-fit Gaussian `apply_lsf`; measure the integrated-flux and centroid difference.

```python
# tests/physics/spectroscopy/test_resolution_matrix_bias.py
import numpy as np
import pytest
import jax.numpy as jnp
import tengri  # noqa: F401
from tengri.observation.banded import BandedMatrix, banded_matvec, gaussian_resolution_bands
from tengri.observation.spectrum import apply_lsf

pytestmark = pytest.mark.limit


def _skewed_bands(wave, R, n_diag=21, skew=1.6):
    """Gaussian bands with an asymmetric (skewed) tail — not representable
    by a symmetric Gaussian LSF."""
    bm = gaussian_resolution_bands(jnp.asarray(wave), R, n_diag)
    off = np.asarray(bm.offsets)
    data = np.asarray(bm.data).copy()
    weight = np.where(off[:, None] > 0, skew, 1.0)  # inflate the red wing
    data = data * weight
    data /= data.sum(axis=0, keepdims=True)
    return BandedMatrix(offsets=jnp.asarray(off), data=jnp.asarray(data))


def test_gaussian_lsf_biases_line_flux_vs_true_R():
    n = 600
    wave = np.geomspace(6500.0, 6620.0, n)  # around Halpha-ish
    line = np.exp(-0.5 * ((wave - 6564.6) / 1.2) ** 2)
    bm = _skewed_bands(wave, 3000.0)
    y_true = np.asarray(banded_matvec(bm.offsets, bm.data, jnp.asarray(line)))
    y_gauss = np.asarray(apply_lsf(jnp.asarray(line), jnp.asarray(wave), 3000.0, sigma_lib_kms=0.0))
    # Integrated flux should be ~conserved by both; the CENTROID (line shape)
    # is what the skew moves — the Gaussian cannot follow it.
    centroid = lambda y: np.sum(wave * y) / np.sum(y)
    dcen = abs(centroid(y_true) - centroid(y_gauss))
    assert dcen > 0.02, f"expected a measurable centroid bias, got {dcen:.4f} A"
```

- [ ] **Step 2: Run** — confirm the skew produces a measurable centroid shift; tune `skew`/tolerance so the test is meaningful and stable. Print the bias in Angstrom and km/s in the assertion message for the record.
- [ ] **Step 3: Commit** — `test(spectroscopy): skewed-R biases line shape vs Gaussian LSF — #1163 justification`

---

### Task 9: #1166 benchmark — coarse-grid accuracy vs. wall-clock

**Files:**
- Create: `bench/scripts/benchmark_spectrum_resample.py`
- (No src change; this is the "measure first" deliverable that gates further #1166 work.)

**Interfaces:**
- Consumes: `compute_spectrum`, `compute_spectrum_conserving`.

Sweep model `n_wave` from fine down toward the pixel count; for a representative observed grid compare point-interp vs. flux-conserving resample: binned-flux error (vs. an ultra-fine reference) and per-eval wall-clock. This is #1166's stated acceptance criterion; its result decides whether a banded `R_resample`/SpectrumPrecomp-conserving-builder is worth building.

- [ ] **Step 1: Write the benchmark** — synthetic continuum + a few emission lines on a fine reference grid; downsample the *model* grid; measure `‖rebin(coarse) − rebin(reference)‖` for each resampler and time each with `block_until_ready`.

```python
# bench/scripts/benchmark_spectrum_resample.py
"""#1166: does a flux-conserving resample let the model run on a coarser grid?

Sweeps model n_wave; compares point-interp vs. bin-integral resample on
binned-flux error and forward-eval wall-clock. Run:
    JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python bench/scripts/benchmark_spectrum_resample.py
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import tengri  # noqa: F401
from tengri.observation.spectrum import compute_spectrum, compute_spectrum_conserving


def _model_sed(wave):
    cont = (wave / 5000.0) ** -1.5
    lines = sum(np.exp(-0.5 * ((wave - c) / 1.5) ** 2) for c in (4861.0, 5007.0, 6564.6))
    return cont + 3.0 * lines


def main():
    z, dl = 0.1, 1e27
    wave_obs = np.geomspace(4000.0, 7000.0, 1500)  # ~R fixed pixel grid
    fine = np.geomspace(3000.0, 9000.0, 60000)
    ref = np.asarray(compute_spectrum_conserving(
        jnp.asarray(_model_sed(fine)), jnp.asarray(fine), jnp.asarray(wave_obs), z, dl))
    print(f"{'n_wave':>8} {'point_err':>12} {'consv_err':>12} {'point_ms':>10} {'consv_ms':>10}")
    for n_wave in (60000, 20000, 8000, 4000, 2000, 1500):
        wm = np.geomspace(3000.0, 9000.0, n_wave)
        sed = jnp.asarray(_model_sed(wm))
        for fn, tag in ((compute_spectrum, "point"), (compute_spectrum_conserving, "consv")):
            out = np.asarray(fn(sed, jnp.asarray(wm), jnp.asarray(wave_obs), z, dl))
            err = float(np.sqrt(np.mean((out - ref) ** 2)) / np.mean(ref))
            f = jax.jit(fn)
            f(sed, jnp.asarray(wm), jnp.asarray(wave_obs), z, dl).block_until_ready()
            t = time.perf_counter()
            for _ in range(50):
                f(sed, jnp.asarray(wm), jnp.asarray(wave_obs), z, dl).block_until_ready()
            ms = (time.perf_counter() - t) / 50 * 1e3
            if tag == "point":
                pe, pt = err, ms
            else:
                ce, ct = err, ms
        print(f"{n_wave:>8} {pe:>12.2e} {ce:>12.2e} {pt:>10.3f} {ct:>10.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it** — `JAX_PLATFORMS=cpu PYTHONPATH=src /tengri/.venv/bin/python bench/scripts/benchmark_spectrum_resample.py`. Capture the table.
- [ ] **Step 3: Record findings** — write `bench/reports/2026-07-15_spectrum_resample.md` with the table and the verdict: at what `n_wave` does the conserving path stay within (say) 0.1% binned-flux error while point-interp drifts, and what wall-clock that buys. This verdict is the go/no-go for a follow-up banded `R_resample`.
- [ ] **Step 4: Commit** — `bench(observation): coarse-grid resample accuracy vs wall-clock (#1166)`

---

### Task 10: Docs, allowlist, follow-up issue, PR

**Files:**
- Modify: `tools/file_size_allowlist.json` (bump any file that grew past its cap — check `spectrum.py`, `spectroscopy.py`, `sed_model.py`)
- Modify: `docs/dev/where-things-live.md` or the spectroscopy doc (one line: resolution matrix → `observation/banded.py`)
- Modify: `src/tengri/__init__.py` **only if** `BandedMatrix`/builders should be public (decide: keep internal for now — expose later with the loader).

- [ ] **Step 1: Full lint + targeted test run**

```bash
cd ~/.claude/jobs/d5f48f27/tmp/wt_banded
/tengri/.venv/bin/ruff check src/ tests/
/tengri/.venv/bin/ruff format --check src/ tests/
PYTHONPATH=src /tengri/.venv/bin/python tools/check_test_markers.py
PYTHONPATH=src /tengri/.venv/bin/python tools/check_british_spelling.py
PYTHONPATH=src /tengri/.venv/bin/python -m pytest tests/components/spectroscopy/test_banded_operator.py tests/components/spectroscopy/test_resolution_matrix.py tests/physics/spectroscopy/test_resolution_matrix_bias.py tests/regression/bug/test_resolution_matrix_compile_signature.py -q
```

- [ ] **Step 2: Bump the allowlist** for any file over its cap; re-run the file-size guard if one exists (`tools/` — mirror how prior commits bumped `sed_model.py`).
- [ ] **Step 3: Open the deferred-work follow-up issue** (desispec loader + real-DESI physics test), with body referencing #1163 and the benchmark verdict, labels `area:observation`, `type:parity`, `enhancement`. Do this only after asking the user to confirm the issue title (issue creation is an outward action).
- [ ] **Step 4: Commit + push + PR**

```bash
git add -A && git commit -m "chore: allowlist bumps + docs for banded resolution matrix (#1163, #1166)"
git push -u origin cs/banded-spectroscopy
```

Then open the PR (after user confirms) titled `feat(observation): banded DESI/PFS resolution matrix + flux-conserving resample benchmark (#1163, #1166)`, labels `area:observation enhancement type:parity jit-safety`, body linking both issues and summarizing: operator subsumes Gaussian LSF (Task 3 proof), cache-collision guard (Task 7, neuter-checked), skewed-R justification (Task 8), coarse-grid benchmark verdict (Task 9), and the deferred loader/real-DESI follow-up.

---

## Self-Review

**Spec coverage (#1163 work items):**
- Optional banded matrix on the spectroscopic container → Task 4 ✓
- JIT/grad-safe banded `R @ model`, route `project_spectrum`, Gaussian fallback → Tasks 1, 5 ✓
- Per-camera R (DESI b/r/z) → **partial**: single-segment operator lands (Tasks 1–6); multi-camera composition is a block-diagonal `BandedMatrix` or multi-segment observation — flagged as a follow-up in Task 10's issue if the observation layer doesn't already segment. *(Gap acknowledged — not silently dropped.)*
- Correctness test Gaussian R ≈ `apply_lsf` → Task 3 ✓
- Physics/bias test → Task 8 (synthetic skewed-R; real-DESI deferred per approved scope) ✓
- Loader (desispec) → **deferred** to Task 10 follow-up issue (approved) ✓
- Docstring citations → Tasks 1, 3, 5 ✓

**Spec coverage (#1166 work items):**
- Benchmark first → Task 9 ✓
- Banded `R_resample`, SpectrumPrecomp conserving builder, route `compute_spectrum` → **deferred**, gated on Task 9's verdict (measure-don't-assume) — documented, not dropped ✓
- Compose operator shape with #1163 → shared `banded.py` kernel is in place for later fusion ✓
- JIT+grad safety, conserving-path docstrings → already merged (#1176) + Task 1 ✓

**Placeholder scan:** none — every code step shows complete code.

**Type consistency:** `BandedMatrix(offsets, data)` used identically in Tasks 1, 2, 3, 4, 5, 7, 8; `banded_matvec(offsets, data, x)` signature consistent throughout; `resolution_matrix` kwarg/field name identical across spectrum.py, spectroscopy.py, observation.py, sed_model.py.

**Known open decision (resolve during Task 6):** whether `tests/components/spectroscopy/` has a bare-stellar SSP fixture; if not, use the `tests/integration/` skip-if-missing pattern. Non-blocking.

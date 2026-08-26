# Contributing to tengri

Tengri is differentiable SED fitting in JAX. Bug reports, fixes, new physics
modules, docs, and review are all welcome. Before opening a PR, read this file
and [`docs/dev/NAMING_CONTRACT.md`](docs/dev/NAMING_CONTRACT.md).

Open issues and the roadmap track ongoing work. See
[`ROADMAP.md`](ROADMAP.md) and the [issue tracker](https://github.com/suchethac/tengri/issues).

## Quick start for developers

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suchethac/tengri.git
   cd tengri
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate  # Windows
   ```

3. **Install in editable mode with dev dependencies:**
   ```bash
   .venv/bin/pip install -e ".[dev]"
   ```

4. **Run the test suite:**
   ```bash
   .venv/bin/pytest tests/ -q
   ```

5. **Check code style:**
   ```bash
   .venv/bin/ruff check src/ tests/
   .venv/bin/ruff format --check src/ tests/
   ```

6. **Auto-fix style issues:**
   ```bash
   .venv/bin/ruff check --fix src/ tests/
   .venv/bin/ruff format src/ tests/
   ```

## Filing a bug report

Include:
- a minimum reproducer (smallest snippet that triggers the bug),
- expected vs actual behavior,
- `pip show astro-tengri`, `python --version`, OS, JAX backend + version, and the value of `JAX_PLATFORMS`,
- the full traceback.

Template: `.github/ISSUE_TEMPLATE/bug_report.md`.

## Proposing a feature

Open an issue describing what and why before implementing. Agree on scope first
for larger changes; it is cheaper than rewriting after review.
Template: `.github/ISSUE_TEMPLATE/feature_request.md`.

## Adding a new alternative within a component

Tengri's physics blocks (AGN models, dust attenuation laws, SFH variants, nebular backends) each have a registry of swappable alternatives. To add yours:

1. **Pick the component.** Run `tengri.list_agn_models()` (or `list_dust_laws()`, `list_sfh_models()`, etc.) to see what's already there and what status each is at. Avoid duplicating an existing alternative.
2. **Copy the worked example.** Start from [`docs/dev/example_new_agn_torus.py`](docs/dev/example_new_agn_torus.py). It registers a model, declares metadata (citation + status), and exercises introspection end-to-end.
3. **Register with metadata.** Use the relevant decorator with `citation`, `status`, `short_doc` filled in:
   ```python
   @register_agn_model(
       "my_model",
       citation="Author+Year, ADS bibcode",
       status="production",   # or "experimental", "demo", "deprecated"
       short_doc="One-line description",
   )
   def my_model(wavelength, agn_log_lbol, **params): ...
   ```
   `citation` and `short_doc` are recommended but not required; fill what you have.
4. **Declare new free parameters via the component's `declared_parameters()` method.** Translation entries are auto-derived for identity (no-unit-conversion) cases: you typically don't need to edit `src/tengri/parameters/translate.py`.

   For genuinely new physics components (a whole new `SEDComponent` class, not just a new alternative inside an existing menu), register the class so its `declared_parameters()` is discovered by the param map:
   ```python
   from tengri import register_component

   @register_component
   @dataclass(frozen=True)
   class MyAGNComponent:
       name = "my_agn"
       parameter_prefix = "my_agn_"
       def declared_parameters(self):
           return [ParamDeclaration("my_agn_eddington",
                                    Uniform(0, 1),
                                    "Eddington ratio")]
       ...
   ```
   `tengri.Parameters(my_agn_eddington=Uniform(0, 1))` then works without editing `src/tengri/parameters/translate.py`.
5. **Verify discoverability.** After your registration loads, your model should appear in:
   ```python
   tengri.summary()                # count goes up by one
   tengri.list_agn_models()        # your row in the table, with citation
   tengri.describe("my_model")     # full metadata block
   tengri.search("part_of_name")   # cross-menu search finds it
   ```
   Counts in `summary()` and `help()` are live from registries; no doc edit needed.
6. **Open a PR.** CI runs the `lint` and `smoke` gate jobs, about forty steps between them, before the full suite. The pull-request template shows how to run that list locally.

Status conventions: `production` (validated), `experimental` (works but not yet validated against observations or another code), `demo` (toy model, not for science), `deprecated` (slated for removal).

## Development workflow

**Branch naming:**
- Feature work: `feat/description` (e.g., `feat/agn-xray-emission`)
- Bug fixes: `fix/description` (e.g., `fix/dust-attenuation-edge-case`)
- Documentation: `docs/description`
- Testing: `test/description`

**Commit messages:**
Follow conventional commits format:
```
<type>: <description>

<optional body with details>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

Example:
```
feat: add SKIRTOR AGN torus model with anisotropic scattering

Implements disk-torus geometry from Stalevski et al. (2016).
Anisotropy parameter theta_torus controls opening angle.
```

## Coding standards

- Ruff for lint + format (`pyproject.toml`).
- Naming: `docs/dev/NAMING_CONTRACT.md`. Canonical names are non-negotiable.
- Docstrings: numpydoc (`docs/dev/docstring-standard.md`) with units, shapes, equations, references.
- Immutable arrays: `.at[].set()` rather than in-place writes.
- Pure JAX in `components/` and `forward/`: `jit`, `vmap`, `grad` must compose.

## Tests

Every change ships with a test. Layout:

- `tests/unit/`: fast, no SSP data.
- `tests/integration/`: needs `data/ssp_*.h5`, skips if missing.
- `tests/crossval/`: against bagpipes / FSPS; opt-in with `-m crossval`.

Run before pushing:

```bash
.venv/bin/pytest tests/ -q
```

## Scientific standards

Every new physics module must:

1. Cite a primary source (peer-reviewed paper, arXiv preprint, or well-documented code).
2. Ship a regression test pinning one prediction to 1% against the paper or upstream reference (SED shape, line flux, attenuation curve, …).
3. Register the citation in `src/tengri/citations/registry.py` so `tengri.cite_all()` picks it up.

Example registration in `registry.py`:
```python
CITATIONS["dust_charlot_fall_2000"] = Citation(
    title="Dust attenuation laws for galaxy models",
    authors="Charlot & Fall",
    year=2000,
    arxiv="N/A",
    doi="10.1086/308936"
)
```

Paper-value regression tests live in `tests/regression/paper/`; the attenuation
laws are covered in `tests/regression/paper/test_dust_attenuation_laws.py`, which
is the file to read for the real shape of this. The sketch below is the pattern,
not a copy of any test in the tree:

```python
def test_charlot_fall_attenuation_calibration():
    """Verify Charlot & Fall (2000) attenuation law at known wavelength."""
    tau_v = 1.0
    wave = 5500  # V-band, Angstrom
    atten = dust_attenuation_charlot_fall(wave, tau_v)
    # Known value from Charlot & Fall (2000), Table 1
    expected = 1.0  # magnitude of attenuation
    assert abs(atten - expected) < 0.01  # <1% error
```

See [docs/dev/verification-protocol.md](docs/dev/verification-protocol.md) for a status table of all physics modules.

## Docstring expectations

Numpydoc. Every parameter description carries units in brackets. Compare:

```python
# Not enough:
def sfr_to_mass(sfr, duration):
    """Convert SFR to stellar mass."""

# What we expect:
def sfr_to_mass(sfr, duration):
    """
    Integrate star formation rate to total stellar mass.

    Parameters
    ----------
    sfr : array_like, shape (n_time,)
        Star formation rate [Msun/yr].
    duration : float
        Integration time [yr].

    Returns
    -------
    ndarray, shape ()
        Integrated stellar mass [Msun].
    """
```

For equations, use `.. math::` directive:
```python
def planck(nu, T):
    """
    Planck function.

    .. math::

        B_\\nu(T) = \\frac{2 h \\nu^3}{c^2} \\frac{1}{e^{h\\nu/k_B T} - 1}

    where:
    - $h$ = Planck constant [erg·s]
    - $\\nu$ = frequency [Hz]
    - $T$ = temperature [K]
    - $k_B$ = Boltzmann constant [erg/K]
    - $c$ = speed of light [cm/s]

    Parameters
    ----------
    nu : float
        Frequency [Hz].
    T : float
        Temperature [K].

    Returns
    -------
    float
        Specific intensity [erg/s/Hz/sr/cm^2].
    """
```

## PR review

Reviews land within ~2 weeks. They look at correctness, tests, style, docs,
primary-source citations for new physics, and any obvious performance traps.
Expect requested changes; that is the point of review.

## PR checklist

The pull-request template carries the list, including the commands that extract
the gate steps CI actually runs. Fill it in rather than working from memory.

## AI-assisted contributions

Tengri was initially drafted with AI assistance and continues to accept
AI-assisted PRs. The bar is the same as for any other PR:

- read every line you submit and own its correctness,
- verify every equation and citation against a primary source; never trust an AI-generated reference,
- run `pytest tests/ -q` and `ruff check src/ tests/` before pushing,
- tick the AI-assistance box in the PR template so reviewers know the context.

## Release process

Releases are cut by the maintainer (see GOVERNANCE.md). The steps live in one
place, [`docs/dev/release-checklist.md`](docs/dev/release-checklist.md): the
one-time PyPI trusted-publisher setup, the gates to clear before tagging, the
four files the version string is hand-copied into, and the fact that uploading
is triggered by publishing a GitHub Release rather than by pushing the tag.

They are deliberately not summarized here. This section used to carry its own
four-step copy, and both of the places it disagreed with `publish.yml` —
the version files and the trigger — were wrong.

## Governance

- [GOVERNANCE.md](GOVERNANCE.md): decision-making and maintainer responsibilities
- [CONTRIBUTORS.md](CONTRIBUTORS.md): contributor list
- [.github/CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md)

## Questions

How-to questions go to GitHub Discussions, bug reports to Issues. Use public
channels rather than mailing the maintainer directly; others benefit too.

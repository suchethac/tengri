# Release checklist — publishing `astro-tengri` to PyPI

The publish workflow (`.github/workflows/publish.yml`) is already wired for
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/). It fires
on `release: [published]` — when a GitHub Release is created, **not** when a
version tag is pushed. What it cannot do is create the PyPI-side and repo-side
configuration — those are one-time, maintainer-only actions (#1818). Do them in
this order.

## One-time setup (maintainer, ~10 minutes)

1. **PyPI pending publisher.** At
   <https://pypi.org/manage/account/publishing/> → *Add a new pending
   publisher*:
   - PyPI project name: `astro-tengri`
   - Owner: `suchethac`
   - Repository: `tengri`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

   ("Pending" because the project does not exist yet — the first successful
   publish claims the name and converts it to a normal trusted publisher.)

2. **GitHub environment.** Repo *Settings → Environments → New environment*,
   named exactly `pypi`. No secrets are needed — trusted publishing uses the
   OIDC `id-token: write` permission the workflow already declares. Optionally
   add yourself as a required reviewer so publishing a Release cannot upload
   without a click of approval.

3. *(Optional but recommended)* Repeat both steps against
   <https://test.pypi.org> with a duplicate `publish-testpypi.yml`, and do one
   dry-run release there first.

## Per-release steps

4. **Verify the sdist/wheel does not ship `data/`.** The repository tracks
   ~0.5 GiB of SSP grids and template atlases; a wheel that includes them is a
   release bug. Locally:

   ```bash
   .venv/bin/python -m build
   .venv/bin/python -m twine check dist/*
   tar tzf dist/astro_tengri-*.tar.gz | grep -c '^.*data/' # expect 0 (or only tiny fixtures)
   unzip -l dist/astro_tengri-*.whl | awk '{s+=$1} END {print s/1e6 " MB"}'
   ```

   If `data/` leaks in, fix the packaging excludes in `pyproject.toml` before
   tagging.

5. **Gates before tagging** (all on current `main`):
   - fast tier green: `.venv/bin/pytest tests/ -q`
   - slow tier green (the PR gate deselects it — run it): `.venv/bin/pytest tests/ -q -m slow`
   - docs build green (CI `build` job)
   - `python tools/check_param_ranges.py`, `check_test_markers.py`,
     `check_numeric_guards.py`, `check_doc_examples.py` all exit 0
   - CHANGELOG (or release notes) entry for the version, including any
     behavior changes flagged `breaking-change`

6. **Version bump.** Nothing derives the version — there is no
   `[tool.setuptools_scm]` section — so it is hand-copied into four files that
   have to move together:

   | File | Form |
   |---|---|
   | `pyproject.toml` | `version = "0.1.0"` |
   | `src/tengri/__init__.py` | `__version__ = "0.1.0"` |
   | `docs/conf.py` | `release = "0.1.0"` |
   | `CITATION.cff` | `version: 0.1.0` |

   ```bash
   grep -rn '0\.1\.0' pyproject.toml src/tengri/__init__.py docs/conf.py CITATION.cff
   ```

   Missing `src/tengri/__init__.py` is the quiet one. Wheel metadata comes from
   `pyproject.toml`, so the build still succeeds and `twine check` still passes;
   the only thing that notices is `publish.yml`'s own import smoke test, which
   prints a `tengri.__version__` disagreeing with the version just published.
   PyPI does not allow re-uploading a version, so that is not recoverable
   in place (#1818).

   Commit the bump together with the `CHANGELOG.md` update — `## [Unreleased]`
   becomes the version heading, per Keep a Changelog.

7. **Tag, then publish a Release.** The tag on its own uploads nothing:

   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   gh release create v0.1.0 --title v0.1.0 --generate-notes
   ```

   (`--generate-notes` rather than `--notes-from-tag`: the tag above is
   lightweight and carries no message for the latter to read.)

   *Publishing the Release* is the trigger — `publish.yml` → build →
   `twine check --strict` → clean-venv wheel import → upload to PyPI under the
   `pypi` environment. If steps 1–2 were skipped, this is where it fails with
   an OIDC/trusted-publisher error — that failure mode is #1818.

8. **Post-publish sanity.** `pip install astro-tengri` in a clean venv;
   `python -c "import tengri; print(tengri.__version__)"`; confirm the
   [project page](https://pypi.org/p/astro-tengri) renders the README.

## After the first release only — PyPI and download badges

Neither the README nor the docs landing page carries a PyPI or download badge,
deliberately: until the first upload every one of them renders `not found`.

Do not add them the moment the upload succeeds either. Download figures derive
from PyPI's public BigQuery download logs, which lag by roughly a day, so a
badge committed alongside the release still reads `not found` through the
most-read day the project will have. Wait for this to return a count:

```bash
curl -s https://img.shields.io/pepy/dt/astro-tengri | grep -o 'aria-label="[^"]*"'
```

Then, in one commit:

9. **`README.md`** — extend the badge row at the top of the file. Note
   `pyproject.toml` sets `readme = "README.md"`, so this row renders on the
   PyPI project page too.

   ```markdown
   [![PyPI](https://img.shields.io/pypi/v/astro-tengri)](https://pypi.org/project/astro-tengri/)
   [![Downloads](https://img.shields.io/pepy/dt/astro-tengri)](https://pepy.tech/project/astro-tengri)
   ```

   Cumulative downloads (`pepy/dt`) rather than monthly (`pypi/dm`) on purpose:
   the monthly figure for a young package is mostly CI traffic, and it visibly
   sags in a quiet month rather than accumulating.

10. **`docs/index.md`** — a badge row inside the hero, after the
    `<p class="tg-hero__tagline">` block and before the closing `</div>`, so it
    sits above the hero's bottom border and inherits its centering.

    **Raw HTML, not Markdown.** The hero `<div>` carries `markdown="0"`, so
    MyST leaves a `[![…](…)](…)` in there as literal text.

    ```html
      <p class="tg-hero__badges">
        <a href="https://pypi.org/project/astro-tengri/"><img
          src="https://img.shields.io/pypi/v/astro-tengri" alt="PyPI" /></a>
        <a href="https://pepy.tech/project/astro-tengri"><img
          src="https://img.shields.io/pepy/dt/astro-tengri" alt="Downloads" /></a>
      </p>
    ```

11. **`docs/_static/custom.css`** — one rule, after the `.tg-hero__tagline`
    block. The badges are external SVGs, so they need layout only, no theming.

    ```css
    .tg-hero__badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        justify-content: center;
        margin: 1.25rem 0 0;
    }
    ```

12. **`docs/installation.md`** — it documents source installs only. Add the
    `pip install astro-tengri` route, or the badges advertise a path the
    install page never describes.

Rebuild (`cd docs && make html SPHINXOPTS="-W --keep-going"`) and check the row
is centered within the hero in both light and dark mode. No `linkcheck` runs in
CI, so the external badge URLs cannot redden the docs build.

## Known release-adjacent decisions already made

- **#1817** — the 1.8 GiB clone weight is accepted (repo public since
  2026-03-21 with third-party forks; a history rewrite buys nothing at this
  point). `tools/check_file_sizes.py` ratchets further growth. PyPI users are
  unaffected as long as step 4 holds.

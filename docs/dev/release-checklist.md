# Release checklist — publishing `astro-tengri` to PyPI

The publish workflow (`.github/workflows/publish.yml`) is already wired for
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) and fires
on a version tag. What it cannot do is create the PyPI-side and repo-side
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
   add yourself as a required reviewer so a tag push cannot publish without a
   click of approval.

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

6. **Version + tag.** Bump `version` in `pyproject.toml` if needed (currently
   `0.1.0`), commit, then:

   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   ```

   The tag triggers `publish.yml` → build → publish to PyPI under the `pypi`
   environment. If step 1–2 were skipped, this is where it fails with an
   OIDC/trusted-publisher error — that failure mode is #1818.

7. **Post-publish sanity.** `pip install astro-tengri` in a clean venv;
   `python -c "import tengri; print(tengri.__version__)"`; confirm the
   [project page](https://pypi.org/p/astro-tengri) renders the README.

## Known release-adjacent decisions already made

- **#1817** — the 1.8 GiB clone weight is accepted (repo public since
  2026-03-21 with third-party forks; a history rewrite buys nothing at this
  point). `tools/check_file_sizes.py` ratchets further growth. PyPI users are
  unaffected as long as step 4 holds.

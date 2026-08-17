SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Jupytext --sync spine .py ↔ notebooks/*.ipynb, copy into docs/spine/, nbconvert --inplace
spine-ipynb:
	@python ../scripts/sync_spine_notebooks_for_docs.py

# Optional: after `make html`, run this to dump nbconvert HTML next to the site
export-notebooks-html: spine-ipynb
	@mkdir -p "$(BUILDDIR)/html/nbconvert"
	@jupyter nbconvert spine/*.ipynb --to html --output-dir="$(BUILDDIR)/html/nbconvert"

html: spine-ipynb
	@python ../tools/check_gallery_fresh.py --strict
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)

livehtml: spine-ipynb
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) --watch ../src/tengri --watch ../notebooks

.PHONY: help spine-ipynb export-notebooks-html html clean livehtml

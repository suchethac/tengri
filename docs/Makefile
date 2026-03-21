SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
NOTEBOOKDIR   = ../notebooks
NBCOPYDIR     = _notebooks

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

notebooks:
	@mkdir -p $(NBCOPYDIR)/tutorials $(NBCOPYDIR)/demonstrations $(NBCOPYDIR)/reference
	@cp $(NOTEBOOKDIR)/tutorials/*.ipynb $(NBCOPYDIR)/tutorials/ 2>/dev/null || true
	@cp $(NOTEBOOKDIR)/demonstrations/*.ipynb $(NBCOPYDIR)/demonstrations/ 2>/dev/null || true
	@cp $(NOTEBOOKDIR)/reference/*.ipynb $(NBCOPYDIR)/reference/ 2>/dev/null || true

html: notebooks
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR) $(NBCOPYDIR)

livehtml: notebooks
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) --watch ../src/tengri --watch ../notebooks

.PHONY: help notebooks html clean livehtml

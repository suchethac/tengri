SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
NOTEBOOKDIR   = ../notebooks
NBCOPYDIR     = _notebooks

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

notebooks:
	@mkdir -p $(NBCOPYDIR)/quickstart $(NBCOPYDIR)/fitting $(NBCOPYDIR)/theory \
	           $(NBCOPYDIR)/models $(NBCOPYDIR)/specialist
	@cp $(NOTEBOOKDIR)/quickstart/*.ipynb $(NBCOPYDIR)/quickstart/ 2>/dev/null || true
	@cp $(NOTEBOOKDIR)/fitting/*.ipynb $(NBCOPYDIR)/fitting/ 2>/dev/null || true
	@cp $(NOTEBOOKDIR)/theory/*.ipynb $(NBCOPYDIR)/theory/ 2>/dev/null || true
	@cp $(NOTEBOOKDIR)/models/*.ipynb $(NBCOPYDIR)/models/ 2>/dev/null || true
	@cp $(NOTEBOOKDIR)/specialist/*.ipynb $(NBCOPYDIR)/specialist/ 2>/dev/null || true

html: notebooks
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR) $(NBCOPYDIR)

livehtml: notebooks
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) --watch ../src/tengri --watch ../notebooks

.PHONY: help notebooks html clean livehtml

Examples gallery
================

170+ standalone scripts demonstrating tengri's physics components, fitting
workflows, and end-to-end use cases. Each card below links to a per-script
page with the rendered figure, the full source, and a downloadable Jupyter
notebook.

**Browse by category.** Cards are organised into sections — quickstart and
workflows for end-to-end recipes, physics components for one-knob sweeps,
inference for fitter behaviour, and use cases for paper-style figures.

How to run an example locally
-----------------------------

Each script is a normal Python program::

    python examples/quickstart/plot_first_fit.py

Most physics examples (dust curves, SFH shapes, AGN spectra) need only
tengri's core dependencies. Fitting examples additionally require an SSP
grid — fetch one with::

    import tengri
    tengri.download_ssp()  # default fsps_prsc_miles_chabrier; see list_known_ssps()

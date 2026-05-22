Examples gallery
================

140+ standalone scripts demonstrating tengri's physics components, fitting
workflows, and end-to-end use cases. Each card below links to a per-script
page with the rendered figure, the full source, and a downloadable Jupyter
notebook.

Sections are ordered to follow the reading curve of a working astronomer:
**start here → build the forward model → observe it → fit data → end-to-end
workflows → paper-style use cases → advanced**.

.. rst-class:: gallery-section-order

   1. ``quickstart/`` — start here
   2. ``sps/`` — the stellar spectral library
   3. ``sfh/`` — star formation history
   4. ``metallicity/`` — chemical evolution
   5. ``dust_attenuation/`` — attenuation curves
   6. ``dust_emission/`` — thermal re-emission
   7. ``nebular/`` — line + continuum emission
   8. ``igm/`` — Lyα forest and DLA
   9. ``agn/`` — accretion disc + torus + lines
   10. ``radio/`` — synchrotron + free-free
   11. ``xray/`` — corona + XRB + AGN
   12. ``photometry/`` — filters and broadbands
   13. ``spectroscopy/`` — LSF and velocity dispersion
   14. ``multiwavelength/`` — panchromatic SEDs
   15. ``inference/`` — MAP, NUTS, VI, NSS
   16. ``recipes/`` — end-to-end recipe scripts
   17. ``workflows/`` — full data → posterior workflows
   18. ``usecases/`` — paper-style figures
   19. ``advanced/`` and ``contrib/`` — Fisher, orchestrator, custom models

How to run an example locally
-----------------------------

Each script is a normal Python program::

    python examples/quickstart/plot_first_fit.py

Most physics examples (dust curves, SFH shapes, AGN spectra) need only
tengri's core dependencies. Fitting examples additionally require an SSP
grid — fetch one with::

    import tengri
    tengri.download_ssp()  # default fsps_prsc_miles_chabrier; see list_known_ssps()

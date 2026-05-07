

.. _sphx_glr_auto_examples_nebular:

Nebular Emission
================

Nebular emission backends comparison.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Cue (Li+2025) emulates a 12-dimensional photoionization grid. This figure sweeps each Cue parameter individually at fixed fiducial values for the rest, showing how each one moves a single galaxy&#x27;s locus on the BPT-N plane (``log [O III]/Hβ`` vs log [N II]/Hα).">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_cue_flexibility_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_cue_flexibility`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT with Cue: Every Knob, One Panel Each</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute three classical BPT line-ratio diagrams using the Cue emulator (Li+2025) and overplot the standard 2D nebular grid: lines of constant log U (varying gas metallicity) and lines of constant log Z_gas (varying ionization parameter). This is the canonical view in Kewley+2001/2013, Dopita+2013, and similar nebular-grid papers.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_cue_grid_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_cue_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT Diagrams with Cue: 2D Grid in (log U, log Z_gas)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The BPT diagram ([OIII]/Hβ vs [NII]/Hα) separates three sources of ionizing photons. Shocks (MAPPINGS V, Allen+2008) move emission-line galaxies into the composite and Seyfert regions as shock velocity increases.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_diagnostics_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_diagnostics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT Diagnostics: Star Formation, Shocks, and AGN</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Diffuse ionized gas (DIG) has lower ionization parameter than HII regions. When present, DIG shifts galaxies toward the LINER region on the BPT diagram by suppressing [OIII] relative to [NII]. f_DIG = 0 is pure HII gas; f_DIG = 1 is pure DIG.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_dig_frac_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_dig_frac_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diffuse Ionized Gas Fraction (f_DIG)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The escape fraction sets how many ionizing photons reach the ISM vs escape into the IGM. Higher f_esc suppresses all nebular emission lines since fewer photons remain to ionize the interstellar gas. f_esc = 0 (all photons stay), f_esc = 1 (all photons escape).">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_fesc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_fesc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionizing Photon Escape Fraction (f_esc)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Emission line velocity dispersion broadens lines from a few km/s (narrow, kinematically resolved) to hundreds of km/s (unresolved at typical spectroscopic resolution). Line broadening is crucial for fitting restframe UV emission lines and measuring dynamics in high-redshift galaxies.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_line_sigma_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_line_sigma_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Emission Line Width (σ in km/s)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does the ionization parameter sweep the emission line strengths and SED shape? Higher logU drives stronger [OIII] emission and shifts galaxies toward the Seyfert region on the BPT diagram.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_logu_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_logu_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionization Parameter (logU)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Gas metallicity controls [NII]/Hα and [OIII]/Hβ — the primary optical metallicity diagnostics. These ratios move galaxies on the BPT diagram and are used to measure oxygen abundances in star-forming galaxies.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_logz_gas_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_logz_gas_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gas Metallicity (log Z/Zsun)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Ionizing photon production declines rapidly with stellar population age (~t^-1). Compare young vs old populations to see how nebular line strength evolves.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_age_dependence_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_age_dependence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular Emission: Dependence on Stellar Population Age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare three nebular emission models: BakedIn (embedded in SSP), CloudyGrid (photoionization tables), and Cue (neural emulator). Shows how backend choice affects emission line strengths.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_backend_compare_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_backend_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular Backends: BakedIn vs CloudyGrid vs Cue</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The BPT diagram ([OIII]/Hβ vs [NII]/Hα) classifies ionization sources. Shows how the ionization parameter (logU) and metallicity drive emission galaxies along the SF→composite→Seyfert sequence.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_bpt_logu_grid_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_bpt_logu_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT Diagram: Ionization Parameter Sequence</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Gas phase metallicity affects ionization balance and emission line strengths. Higher metallicity increases cooling efficiency, affecting the nebular continuum and emission-line ratios through recombination rate changes.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_density_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_density_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular Gas Density: Metallicity Variation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare tengri&#x27;s three nebular emission backends: BakedIn (SSP-embedded), CloudyGrid (tabulated photoionization), and Cue (neural emulator). Shows how each backend predicts emission lines in the optical window.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_nebular_backends_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_nebular_backends`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular Emission Backends</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Plot shock emission line diagnostics from MAPPINGS V photoionization models. Shows how shock velocity, density, and magnetic field affect line ratios and can mimic AGN-like emission in diagnostic diagrams.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_shock_emission_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_shock_emission`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular Shock Emission (MAPPINGS V)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/nebular/plot_bpt_cue_flexibility
   /auto_examples/nebular/plot_bpt_cue_grid
   /auto_examples/nebular/plot_bpt_diagnostics
   /auto_examples/nebular/plot_dig_frac_sweep
   /auto_examples/nebular/plot_fesc_sweep
   /auto_examples/nebular/plot_line_sigma_sweep
   /auto_examples/nebular/plot_logu_sweep
   /auto_examples/nebular/plot_logz_gas_sweep
   /auto_examples/nebular/plot_neb_age_dependence
   /auto_examples/nebular/plot_neb_backend_compare
   /auto_examples/nebular/plot_neb_bpt_logu_grid
   /auto_examples/nebular/plot_neb_density_sweep
   /auto_examples/nebular/plot_nebular_backends
   /auto_examples/nebular/plot_shock_emission


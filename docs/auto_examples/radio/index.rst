:orphan:

.. _sphx_glr_auto_examples_radio:

Radio
=====

Star formation (free-free and synchrotron) and AGN (radio-loud) components. Far-infrared–radio correlation and non-thermal spectral slopes. Model-family comparison included.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The synchrotron spectral index α_sf controls how steeply the radio spectrum falls with frequency. Star-forming galaxies typically have α_sf ≈ 0.7–0.8. Flat spectra (α ≈ 0) signal strong free-free contribution; steep spectra (α &gt; 1) indicate cosmic-ray electron aging. We vary α_sf ∈ [0.3, 1.2] at fixed L_IR = 10^11 L_sun and show normalized spectra (reference 1.4 GHz).">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_alpha_sf_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_alpha_sf_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Synchrotron spectral index: steeper α_sf dims the high-frequency tail</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The dimensionless parameter q_IR characterizes the FIR-radio correlation, linking far-infrared luminosity to 1.4 GHz synchrotron emission. Higher q_IR means relatively weaker radio per unit star formation. We vary q_IR across the observationally motivated range 2.0–3.3 at fixed L_IR = 10^11 L_sun, demonstrating how radio loudness evolves (Bell 2003).">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_q_ir_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_q_ir_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FIR-radio correlation: q_IR sets radio loudness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A star-forming galaxy&#x27;s GHz continuum is set by two components: non-thermal synchrotron from supernova remnants (steep, L_ν ∝ ν^{-α_sf}) and thermal free-free from H II regions (flat, L_ν ∝ ν^{-0.1}). Their ratio at fixed frequency depends sensitively on the synchrotron spectral index α_sf — flatter spectra leave more of the GHz luminosity to free-free, steeper spectra are synchrotron-dominated until the (sub-mm) crossover.">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_radio_crossover_frequency_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_radio_crossover_frequency`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Synchrotron / free-free balance vs synchrotron slope α_sf</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The FIR-radio correlation links far-infrared luminosity (dust-reprocessed star-formation energy) to 1.4 GHz synchrotron emission. The dimensionless q_IR parameter relates the two via L_IR ∝ L_1.4GHz^(10^q_IR/2.5). Brighter starbursts emit stronger radio across all frequencies. We sweep L_IR over 10^10–10^13 L_sun at fixed q_IR = 2.64 (canonical; Bell 2003).">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_radio_lir_relation_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_radio_lir_relation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FIR-radio correlation: L_IR × q_IR sets radio loudness scale</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Radio loudness R = log₁₀(L_5GHz / L_B) quantifies the ratio of AGN radio to optical luminosity. Radio-quiet AGN have R ≲ 1; radio-loud sources (FR I/II, blazars) reach R ∼ 3–5. Each decade in R corresponds to an order of magnitude increase in jet radio luminosity at fixed bolometric AGN power. We sweep R ∈ [0, 4] at fixed L_bol = 10^44 erg/s (Seyfert-1-like) and α_agn = 0.7.">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_radio_loudness_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_radio_loudness_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN radio loudness R: orders of magnitude in jet power</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The radio group is two independent choices — a star-forming block tied to the FIR-radio correlation, and an AGN block — so this compares them one at a time on the same galaxy.">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_radio_model_family_compare_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_radio_model_family_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radio blocks: which q_IR calibration, and which AGN synchrotron shape</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed host (constant SFR = 3 M☉/yr, Condon-92 synchrotron + free-free) we sweep the composable AGN&#x27;s bolometric luminosity agn_log_lbol from 9 to 13 (in log L_sun). The host alone produces a power-law GHz continuum; the AGN superposes a flatter-spectrum jet component that takes over above log L_bol ≳ 11.5 — the classic radio-loud / radio-quiet division emerges from this competition.">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_radio_vs_agn_lbol_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_radio_vs_agn_lbol`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radio SED response to AGN bolometric luminosity</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/radio/plot_alpha_sf_sweep
   /auto_examples/radio/plot_q_ir_sweep
   /auto_examples/radio/plot_radio_crossover_frequency
   /auto_examples/radio/plot_radio_lir_relation
   /auto_examples/radio/plot_radio_loudness_sweep
   /auto_examples/radio/plot_radio_model_family_compare
   /auto_examples/radio/plot_radio_vs_agn_lbol


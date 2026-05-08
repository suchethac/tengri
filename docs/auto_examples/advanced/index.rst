

.. _sphx_glr_auto_examples_advanced:

Advanced Topics
===============

Hierarchical inference, gradient sensitivity, batch fitting, panchromatic SED
with radio and X-ray components, and joint photometry + spectroscopy fitting.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cramér-Rao bound from the Fisher Information Matrix shows that SDSS 5-band photometry alone cannot separately constrain age, dust, and metallicity. Adding NIR or MIR bands breaks the degeneracy by factors of 2–5×, quantifying the information gain from multiwavelength coverage.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_fisher_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_fisher_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-Dust-Metallicity Degeneracy: Fisher Analysis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Computes the Jacobian d(flux)/d(theta) of the forward model and displays it as a heatmap showing which photometric bands are sensitive to which physical parameters.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_gradient_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_gradient_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient Sensitivity Heatmap</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sets up a small population of mock galaxies sharing the same burstiness PSD parameters (sigma, tau), runs PopulationFitter briefly, and displays the shared PSD posterior vs truth.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_hierarchical_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_hierarchical`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hierarchical PSD Inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates tengri&#x27;s Observation API for joint fitting. Creates a mock galaxy with both SDSS photometry and a low-resolution spectrum, fits with MAP, and shows the combined constraint power. Requires SSP data.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_joint_fit_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_joint_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Joint Photometry + Spectroscopy Fit</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a full multiwavelength SED — stellar + nebular + AGN + dust + radio + X-ray + IGM — without going through tengri.SEDModel, using tengri.forward.build_components and tengri.forward.run_components (Phase II-2.6 public API).">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_orchestrator_demo_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_orchestrator_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Component orchestrator end-to-end</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a full galaxy SED spanning X-ray to radio wavelengths. Shows stellar emission, dust attenuation, dust IR emission, radio synchrotron, and X-ray binary contributions. No SSP data required for the multi-wavelength components.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_radio_xray_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_radio_xray`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Panchromatic SED: UV to Radio</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/advanced/plot_fisher_degeneracy
   /auto_examples/advanced/plot_gradient_sensitivity
   /auto_examples/advanced/plot_hierarchical
   /auto_examples/advanced/plot_joint_fit
   /auto_examples/advanced/plot_orchestrator_demo
   /auto_examples/advanced/plot_radio_xray


:orphan:



.. _sphx_glr_auto_examples_advanced:

Advanced Topics
===============

Hierarchical inference, gradient sensitivity, batch fitting, panchromatic SED
with radio and X-ray components, and joint photometry + spectroscopy fitting.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


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

    <div class="sphx-glr-thumbcontainer" tooltip="Computes the Jacobian d(flux)/d(theta) of the forward model and displays it as a heatmap showing which photometric bands are sensitive to which physical parameters. Each column shows normalized sensitivity to one parameter; dark blue/red indicates strong dependence.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_gradient_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_gradient_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient Sensitivity Heatmap</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates tengri&#x27;s Observation API for joint fitting across two data streams. Creates a mock galaxy with SDSS photometry and low-resolution spectroscopy, then recovers parameters via MAP. Shows how spectroscopy breaks photometric degeneracies.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_joint_fit_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_joint_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Joint Photometry + Spectroscopy Fit</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="TODO[examples-sweep]: This script uses low-level component orchestration (build_components, run_components) which is experimental Phase II-2.6 API intended for infrastructure use, not recommended for user-facing examples.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_orchestrator_demo_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_orchestrator_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Component Orchestrator End-to-End</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a full galaxy SED spanning X-ray to radio wavelengths. Shows stellar emission, dust attenuation, dust IR emission, radio synchrotron, and X-ray binary contributions. Demonstrates tengri&#x27;s multiwavelength physics modules for radio and X-ray—no SSP data required for these components.">

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
   /auto_examples/advanced/plot_joint_fit
   /auto_examples/advanced/plot_orchestrator_demo
   /auto_examples/advanced/plot_radio_xray


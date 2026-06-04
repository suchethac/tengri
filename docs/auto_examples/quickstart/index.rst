:orphan:

.. _sphx_glr_auto_examples_quickstart:

Quick Start
===========

Getting started with tengri — first fit and SED visualization.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The simplest end-to-end tengri workflow. We build a model with a truncated-skew-normal SFH and a two-component Calzetti dust attenuation, mock SDSS ugriz photometry at S/N = 20, then run a MAP fit to recover the input parameters. The figure shows the full rest-frame SED behind the five observed bands and the residuals of the MAP fit relative to the noise level.">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_first_fit_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_first_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recovering a star-forming galaxy from 5-band SDSS photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The model.spec.summary() method displays each parameter&#x27;s source through provenance tags: [user] for explicit overrides, [ FREE] and [ FIXED] for wildcard expansions, and [default] for registry defaults. We build a model with mixed constraints, display the annotated summary as a figure caption, and show the predicted SED.">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_model_summary_walkthrough_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_model_summary_walkthrough`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Understanding model structure through parameter provenance tags</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a model with both stellar and dust components. Predict the full SED with attenuation, then predict without dust absorption to isolate the absorbed UV-optical flux. The filled region shows how much light dust removes from the intrinsic stellar continuum.">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_sed_components_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_sed_components`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation across the SED: intrinsic, attenuated, and absorbed</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hα and [O III]+Hβ are produced by gas reprocessing the ionising continuum from O/B stars. Whether they appear in the predicted SED depends entirely on the nebular backend the model is built with.">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_swap_nebular_backend_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_swap_nebular_backend`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Swapping the nebular backend on, then off, on a young starburst</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/quickstart/plot_first_fit
   /auto_examples/quickstart/plot_model_summary_walkthrough
   /auto_examples/quickstart/plot_sed_components
   /auto_examples/quickstart/plot_swap_nebular_backend


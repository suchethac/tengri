

.. _sphx_glr_auto_examples_inference:

Inference Methods
=================

Comparing inference methods and convergence diagnostics.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The convergence diagnostic shows how the negative log posterior (loss) decays across optimizer iterations. We fit mock photometry using MAP (maximum a posteriori) optimization with Adam and display the loss curve, showing when the optimizer has effectively converged. The right panel overlays the recovered SFH against the truth.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_convergence_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_convergence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MAP fit convergence: loss decay across iterations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates parameter degeneracies and individual 1-D marginalized posteriors after fitting mock 5-band SDSS photometry. The corner plot shows the full 2-D covariance structure between parameters; blue lines mark the injected truth. Note: for demonstration scale; production runs use 10× more VI iterations and samples.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_corner_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_corner`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Posterior corner plot from variational inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates convergence behavior of two inference methods: MAP (point-estimate via optimization) and pure-JAX geometric variational inference (native VI). Both are initialized from the same MAP fit, then evolve independently to show how they explore the posterior. The SFH panel on the right shows the recovered star-formation history from each method overlaid on the truth.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_method_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_method_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparing MAP and pure-JAX variational inference</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/inference/plot_convergence
   /auto_examples/inference/plot_corner
   /auto_examples/inference/plot_method_comparison


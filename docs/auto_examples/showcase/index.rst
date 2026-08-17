:orphan:

.. _sphx_glr_auto_examples_showcase:

Showcase
========

Full-stack demonstrations: population forward modeling, gradient diagnostics, end-to-end workflows.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Fisher Information Matrix quantifies which parameter combinations are constrained by data and which are degenerate. Age-dust degeneracy: at fixed stellar mass, older stars + more dust produce the same multiwavelength SED as younger stars + less dust.">

.. only:: html

  .. image:: /auto_examples/showcase/images/thumb/sphx_glr_plot_gradient_degeneracy_direction_thumb.png
    :alt:

  :doc:`/auto_examples/showcase/plot_gradient_degeneracy_direction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fisher Information Ellipses from the Hessian</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute logarithmic sensitivities ∂(log F) / ∂(log θ) for each photometric band. Finite-difference methods (∂F/∂θ ≈ [F(θ+δ) − F(θ−δ)] / (2δ)) are slow and fragile; JAX autodiff computes exact sensitivities via one forward and reverse pass per parameter.">

.. only:: html

  .. image:: /auto_examples/showcase/images/thumb/sphx_glr_plot_jax_gradient_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/showcase/plot_jax_gradient_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Automatic differentiation: parameter sensitivities via jax.grad</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Recipes for common science cases">

.. only:: html

  .. image:: /auto_examples/showcase/images/thumb/sphx_glr_plot_recipes_gallery_thumb.png
    :alt:

  :doc:`/auto_examples/showcase/plot_recipes_gallery`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recipes for common science cases</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/showcase/plot_gradient_degeneracy_direction
   /auto_examples/showcase/plot_jax_gradient_sensitivity
   /auto_examples/showcase/plot_recipes_gallery


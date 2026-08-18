:orphan:

.. _sphx_glr_auto_examples_igm:

IGM
===

Intergalactic-medium absorption: Madau vs Inoue prescriptions, Lyα forest, damped Lyα systems. Lyman-break/dropout signature in high-z photometric selection. IGM `igm_transmission(wave_obs, z)` takes observed-frame wavelengths (not rest-frame).



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Damped Lyman-alpha (DLA) systems imprint strong absorption features blueward of the Lyman-alpha line (1216 Å rest-frame). We sweep column density log(N_H) ∈ {19.0, 19.5, 20.0, 20.3, 20.8} cm^{-2} at fixed redshift z=3, showing how higher column density systems deepen the Lyman forest and suppress flux in the UV-to-optical SED.">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_dla_absorption_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_dla_absorption`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DLA column density sculpts the Lyman alpha forest at z=3</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Damped Lyman-alpha (DLA) systems imprint deep absorption troughs across the UV-to-optical range, with the strength and profile shape depending sensitively on the absorber&#x27;s redshift. We hold column density at the classic DLA threshold log(N_H) = 20.3 cm⁻² and sweep the absorber redshift over z ∈ {1, 2, 3, 4, 5, 6}, showing how the damping wing pattern shifts to longer observed wavelengths and the Lyman-alpha forest structure evolves. This complements the fixed-z, variable-N_H absorption pattern by isolating the redshift dependence.">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_dla_redshift_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_dla_redshift_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DLA damping wing evolves with absorber redshift at fixed column density</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Five IGM transmission variants available in tengri are compared at z=7, applied to a young star-forming SED. This diagnostic isolates the differences between models around the Lyman-alpha forest:">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_igm_models_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_igm_models_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparison of IGM absorption models at high redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The intergalactic medium (IGM) imprints wavelength-dependent opacity on observed galaxy SEDs via Lyman-series and Lyman-continuum absorption. The Lyman break at 912 Å rest-frame shifts to longer observed wavelengths at higher z, enabling photometric redshift estimation via the dropout technique. We vary redshift z ∈ {0.5, 1, 2, 3, 4, 6, 8} across the Inoue et al. (2014) transmission model to show how IGM opacity increases with z.">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_igm_redshift_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_igm_redshift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">IGM transmission curves evolve sharply with redshift as Lyman forest deepens</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Lyman-alpha (Lyα) emission line at rest-frame 1216 Å is one of the strongest hydrogen recombination features in star-forming galaxies. As the redshift increases from z = 2 to z = 7, the IGM becomes progressively opaque at wavelengths shortward of Lyα (the &quot;blue wing&quot;), due to cumulative Lyman-series absorption from neutral hydrogen in the intergalactic medium.">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_lyman_alpha_igm_attenuation_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_lyman_alpha_igm_attenuation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-alpha profile and IGM blue-wing absorption across redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A young Lyman-break galaxy SED is built once at rest frame, then redshifted to a sequence of observed-frame epochs (``z = 1, 3, 5, 7``) with the Inoue et al. 2014 IGM transmission stamped on top. The characteristic spectral signatures move with redshift:">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_sed_with_igm_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_sed_with_igm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Full galaxy SED with IGM absorption applied at multiple redshifts</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/igm/plot_dla_absorption
   /auto_examples/igm/plot_dla_redshift_evolution
   /auto_examples/igm/plot_igm_models_comparison
   /auto_examples/igm/plot_igm_redshift
   /auto_examples/igm/plot_lyman_alpha_igm_attenuation
   /auto_examples/igm/plot_sed_with_igm


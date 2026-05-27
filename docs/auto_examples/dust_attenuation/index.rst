:orphan:

.. _sphx_glr_auto_examples_dust_attenuation:

Dust Attenuation
================

How starlight is extincted on its way out of the galaxy — Calzetti vs
power-law slopes, the 2175 Å UV bump, birth-cloud and diffuse-ISM optical
depths, two-component geometry, and law comparisons.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The tengri library offers six attenuation laws covering the morphology-geometry spectrum: Milky Way (Cardelli), SMC (Pei), starburst (Calzetti, Conroy), and theoretical models (Kriek &amp; Conroy, power law). At fixed τ_V = 1, their curves expose the 2175 Å bump (MW/Cardelli), slope differences (SMC is greyer, Calzetti is redder), and parametric extensions (Kriek &amp; Conroy).">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_attenuation_law_compare_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_attenuation_law_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The six headline dust attenuation laws span MW, SMC, and starburst geometries</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust attenuation laws encode how interstellar dust preferentially absorbs short-wavelength (UV) starlight relative to optical/IR. The wavelength dependence is empirically calibrated to extinction measurements in the Milky Way (Cardelli+1989), Large/Small Magellanic Clouds (Pei 1992), and starburst galaxies (Calzetti+2000, Kriek+Conroy 2013).">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_attenuation_law_family_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_attenuation_law_family`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation laws: family comparison across UV to near-IR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Charlot &amp; Fall 2000 two-component dust model splits attenuation into a birth-cloud component (``τ_bc``) that only the youngest stellar ages see, and a diffuse-ISM component (``τ_diff``) that attenuates all stellar light. The two are degenerate for an old population (every star is &quot;old&quot; by the BC clock, so τ_bc has no effect) but separate cleanly for a young one.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_birth_cloud_vs_diffuse_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_birth_cloud_vs_diffuse`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Birth-cloud vs diffuse-ISM dust: which knob does what?</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Charlot &amp; Fall 2000 two-component dust model splits attenuation into: - τ_bc (birth-cloud): attenuates only young stellar ages (&lt; 10 Myr) - τ_diff (diffuse ISM): attenuates all stellar light">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_birth_cloud_vs_diffuse_age_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_birth_cloud_vs_diffuse_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Birth-cloud attenuation age dependence: Charlot & Fall 2000</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cardelli+1989 Milky Way attenuation curve is a family parameterised by R_V = A_V / E(B-V). Smaller R_V (≲ 3) gives a steeper UV rise and stronger 2175 Å bump (denser lines of sight, small grains dominate); larger R_V (≳ 4.5) flattens the UV slope (processed grains, larger sizes).">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_cardelli_rv_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_cardelli_rv_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cardelli MW attenuation: sweeping R_V</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The six headline dust attenuation laws plotted over the full UV-through-NIR range (0.1–3 μm), extending beyond the 2175 Å bump region to show how curves flatten in the infrared. Red-shifted galaxies observe longer wavelengths at rest frame, so the IR slope controls K-correction factors and SED fitting degeneracies.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_curves_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_curves`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation laws from UV through near-infrared</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust geometry determines how dust affects starlight. A screen (foreground dust) filters the light as it leaves the galaxy: transmission = exp(-τ_λ). A mixed geometry (dust uniformly distributed with stars) is more gentle: transmission = (1 - exp(-τ_λ)) / τ_λ.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_geometry_screen_vs_mixed_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_geometry_screen_vs_mixed`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Screen vs. mixed dust geometry: identical optical depths, different SEDs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three dust geometries—foreground screen (power-law), mixed slab (Calzetti), and clumpy two-phase (SMC)—proxy different physical arrangements via their attenuation laws. At fixed τ_V = 1, geometry controls the spectral shape: screens are reddest, clumpy geometries are greyest. Transmission curves show how each law transforms a stellar continuum.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_geometry_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_geometry_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust geometry shapes the extinction: screen vs mixed vs clumpy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each of the bundled dust-attenuation laws applied to the same intrinsic SED at the same V-band optical depth — so the differences between the curves are entirely in the wavelength dependence of the attenuation. The intrinsic (unreddened) SED is shown in black for reference.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_law_application_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_law_application`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The same galaxy reddened by every attenuation law in the registry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="For a fixed star-forming galaxy with τ_V = 1 (a moderate attenuation), six common attenuation laws produce six visibly different reddened UV slopes β. The intrinsic SED has β ≈ −2.3; SMC steepens β to ≈ +0.4; Calzetti / Salim leave a flatter β ≈ −0.5. The spread (~1 mag of UV slope at fixed τ_V) is the systematic an SED fitter inherits if its dust-law assumption is wrong.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_law_uv_slope_response_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_law_uv_slope_response`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Attenuation law leaves a distinct UV-slope fingerprint</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The power-law slope δ steepens (negative) or flattens (positive) UV attenuation relative to the optical, controlling whether dust absorbs more or less light at short wavelengths. We vary δ with elevated τ_bc and τ_diff to make slope effects visible (low dust opacities wash out the continuum slope).">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_slope_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_slope_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation curve slope controls UV vs optical hardness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four named attenuation laws applied to the same intrinsic SED at the same V-band optical depth (τ_V = 1.0), illustrating how dust geometry and grain-size composition vary across the local universe.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_galactic_zoo_dust_laws_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_galactic_zoo_dust_laws`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation laws across the galaxy zoo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Birth-cloud dust optical depth τ_bc attenuates only the youngest stellar light (age &lt; ~10 Myr), controlling nebular emission from embedded HII regions. τ_bc effects are clearest on young star-forming populations; we use a 500 Myr starburst and vary τ_bc across the prior range.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_tau_bc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_tau_bc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Birth cloud dust suppresses young-stellar UV and nebular emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Diffuse ISM dust optical depth τ_diff attenuates all stellar light (young + old). Higher τ_diff reddens the optical continuum and weakens the 4000 Å break, signaling aging stellar populations. We vary τ_diff across a range with every other parameter fixed on a typical star-forming galaxy.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_tau_diff_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_tau_diff_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diffuse ISM dust attenuates all stellar populations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Charlot &amp; Fall two-component dust model separates birth-cloud dust (young stars only, age &lt; ~10 Myr) from diffuse ISM dust (all stars). Two panels show: (left) V-band transmission versus age for three (τ_bc, τ_diff) combinations, revealing the sharp ~10 Myr transition; (right) full transmission spectra for 1 Myr and 1 Gyr stars under the same dust column.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_two_component_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_two_component`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Two-component dust: birth cloud obscures only young stars</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The IRX–β relation connects the UV continuum slope β (1250–2600 Å) with the infrared excess IRX = log₁₀(L_IR / L_UV). This diagram reveals dust reddening and star formation rate indicators in galaxies. Here we:">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_usecase_irx_beta_meurer_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_usecase_irx_beta_meurer`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">IRX–β diagram: infra-red excess vs UV slope (Meurer+1999)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 2175 Å UV bump from PAHs and small graphite grains sweeps from absent to Milky-Way strength via the dust_bump_strength knob. At zero, the attenuation curve is a smooth power law; at MW-like values, the bump dominates the UV. We show the attenuation law (not a galaxy SED) to isolate the curve shape.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_uv_bump_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_uv_bump_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The 2175 Å UV bump traces small-grain dust populations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Charlot &amp; Fall 2000 two-component dust model conserves energy: every UV photon attenuated by the dust must come back out as IR re-emission. We sweep τ_diff from 0 to 2 mag and on each step plot two quantities — the absorbed UV power L_abs(λ&lt;3000 Å) inferred from the difference of (no-dust) minus (with-dust) attenuated SEDs, and the integrated IR luminosity L_IR(8–1000 μm) from the IR re-emission template (Dale+2014 here).">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_uv_ir_energy_balance_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_uv_ir_energy_balance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV-IR energy balance: absorbed = re-emitted</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/dust_attenuation/plot_attenuation_law_compare
   /auto_examples/dust_attenuation/plot_attenuation_law_family
   /auto_examples/dust_attenuation/plot_birth_cloud_vs_diffuse
   /auto_examples/dust_attenuation/plot_birth_cloud_vs_diffuse_age
   /auto_examples/dust_attenuation/plot_cardelli_rv_sweep
   /auto_examples/dust_attenuation/plot_dust_curves
   /auto_examples/dust_attenuation/plot_dust_geometry_screen_vs_mixed
   /auto_examples/dust_attenuation/plot_dust_geometry_sweep
   /auto_examples/dust_attenuation/plot_dust_law_application
   /auto_examples/dust_attenuation/plot_dust_law_uv_slope_response
   /auto_examples/dust_attenuation/plot_dust_slope_sweep
   /auto_examples/dust_attenuation/plot_galactic_zoo_dust_laws
   /auto_examples/dust_attenuation/plot_tau_bc_sweep
   /auto_examples/dust_attenuation/plot_tau_diff_sweep
   /auto_examples/dust_attenuation/plot_two_component
   /auto_examples/dust_attenuation/plot_usecase_irx_beta_meurer
   /auto_examples/dust_attenuation/plot_uv_bump_sweep
   /auto_examples/dust_attenuation/plot_uv_ir_energy_balance


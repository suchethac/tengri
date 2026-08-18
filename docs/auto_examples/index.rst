Examples gallery
================

120 standalone scripts demonstrating tengri's physics components and end-to-end use cases. Fitting and inference live in the tutorial notebooks (docs "Notebooks" section), not the gallery — every gallery script is a forward-model figure.

Run a script locally with ``python examples/quickstart/plot_model_summary_walkthrough.py``. Physics examples (dust curves, SFH shapes, AGN spectra) require only core dependencies. Fetch an SSP grid via ``import tengri; tengri.download_ssp()``.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Quick Start
===========

Model-summary walkthrough, SED dust anatomy, nebular-backend swap, and components-isolated anatomy tour. Fitting and inference examples live in the tutorial notebooks.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six physics blocks added cumulatively to the same star-forming host so the contribution of each is visible at every wavelength.">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_components_isolated_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_components_isolated`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Each tengri SED component shown in isolation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Understanding model structure through parameter provenance tags">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_model_summary_walkthrough_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_model_summary_walkthrough`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Understanding model structure through parameter provenance tags</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust attenuation across the SED: intrinsic, attenuated, and absorbed">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_sed_components_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_sed_components`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation across the SED: intrinsic, attenuated, and absorbed</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hα and [O III]+Hβ are produced by gas reprocessing the ionizing continuum from O/B stars. The SFH is a young starburst (peak age ≈ 30 Myr).">

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

Recipes
=======

Recipe comparison, introspection tour, and custom filter design.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six curated recipes span galaxy populations: star-forming at 0–6 (bare-stellar SSP), quiescent at z ≈ 0.05 (bare-stellar, τ_diff-free to trace dust), AGN panchromatic (bare-stellar, full AGN composite with disc+torus+radio+xray), stochastic JWST high-z with burstiness (bare-stellar, DPL+field at 0.5–12), mock-recovery minimal (any SSP, 4–5 free params for benchmarking), and dust-demo (wNE only — baked nebular emission visualized). All use WavePrecomp() except photoz (ztable does not cover z &gt; 12). Use load_ssp(&quot;*.wNE&quot;) only for dust_demo; others silently under-predict if fed wNE.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_compare_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">What each shipped tengri recipe produces</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a FilterCurve from a Gaussian transmission profile and combine it with standard filters. The Photometry object merges them, then SEDModel predicts photometry on all bands at once — custom filters compose naturally with the standard library.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_custom_filter_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_custom_filter`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Register and use a custom photometric filter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Call tengri.list_recipes() to see the shipped menu with SSP requirements (bare-stellar, wNE, or any) and tengri.describe_recipe(name) to fetch a recipe&#x27;s docstring. Three models showcase the morphological diversity: star-forming (DPL+Cue nebular, free z to 6), quiescent at z=0.05 (dexp, lower dust ceiling), and AGN-panchromatic (full composite, z to 6). All require bare-stellar SSP (Cue backend).">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_introspection_tour_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_introspection_tour`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recipe introspection and SED morphology comparison</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Stellar Population Synthesis
============================

SSP grid and age/metallicity sweeps, IMF choice, mass-to-light band comparison, and library shootout.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Initial Mass Function (IMF) parameterizes the fraction of massive versus low-mass stars born during star formation. Chabrier, Kroupa, and Salpeter IMFs differ most in the high-mass end: Salpeter has more massive stars, producing a higher M/L ratio (more mass per unit light) and harder UV continua. We vary IMF while fixing SFH, age, and metallicity, overlaying rest-frame νL_ν to reveal the IMF signature in the SED continuum shape and M/L.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_imf_choice_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_imf_choice_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Initial Mass Function choice and stellar mass-to-light ratio</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The M_★/L_band ratio depends on population age, but the sensitivity varies dramatically by band. At short wavelengths (u, V), M/L is very age-sensitive: young starbursts are bright, so M/L is small; old populations are faint in the UV, so M/L grows rapidly (factor ~100 over 10 Gyr).">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_mass_to_light_band_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_mass_to_light_band_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar mass-to-light ratios across bands: age sensitivity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A single stellar population transitions from UV-dominated (young, hot) to NIR-dominated (old, red) with age. Peak-normalized λF_λ on log-log axes makes the temperature inversion visible across five representative ages at solar metallicity.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_age_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_age_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Population Aging: SSP at Solar Metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Per-SSP-library solar metallicity differs: MIST Z☉ = 0.0142, BC03/Padova Z☉ = 0.0190, PARSEC Z☉ = 0.0152, BASTI Z☉ = 0.0200.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP Grid: Age and Metallicity Evolution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Stellar population synthesis templates differ across stellar evolution codes and isochrone libraries, producing measurable offsets in predicted spectra even at fixed age and metallicity. This gallery script loads four representative SSP libraries shipped with tengri (BC03, FSPS MILES, FSPS C3K, BPASS, ProGeny), constructs minimal SEDModel instances at age = 5 Gyr and Z = 0 (solar), and overlays rest-frame SED predictions (νL_ν) on log-log axes to reveal template-dependent uncertainties and continuum shape differences.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_library_shootout_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_library_shootout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP Library Shootout: Comparing Spectral Predictions at 5 Gyr, Z=0</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Metallicity reddens the optical continuum and shifts iron-peak absorption features in the near-IR. We show five metallicity points spanning the SSP grid at fixed age (1 Gyr). Peak-normalized λF_λ makes spectral shape variations visible without large luminosity differences obscuring them.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_metallicity_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_metallicity_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Metallicity Effects on SED</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Star Formation Histories
========================

Parametric forms (DPL, delayed-exponential, lognormal) and non-parametric (PSD-governed stochastic). Quenching pathways, burst observability, and SFH form comparison.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed mean SFH and stellar mass, continuity (Leja+2019) and field (PSD-governed) priors yield strikingly different stochastic realizations: continuity produces smooth log-normal transitions; field produces controlled burstiness governed by σ_field.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_continuity_vs_bursty_psd_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_continuity_vs_bursty_psd`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Continuity Prior vs PSD-Governed Prior: Stochastic Structure at Fixed Mean</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 3×3 grid showing how the rising slope α (columns) and falling slope β (rows) together control the full SFH morphology. Early-time α determines assembly speed; late-time β sets the post-peak decay. The optical SED responds across each cell. Bottom panels show representative 1D sweeps: α alone (left, at fixed β) and β alone (right, at fixed α), illustrating how each parameter independently shapes the full UV-to-IR SED.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dpl_alpha_beta_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dpl_alpha_beta_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Double power-law SFH parameter space: early growth α vs late quenching β</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 3×3 grid showing five stochastic-SFH realizations for each combination of amplitude σ (vertical axis) and damping timescale τ (horizontal axis). Larger σ produces more dramatic bursts; longer τ sustains those bursts. Each panel shows the mean smooth SFH (dashed) and colored realizations. Bottom panels show representative SEDs for σ alone (left) and τ alone (right), illustrating how each parameter independently shapes the UV continuum and optical colors.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_burstiness_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_burstiness`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PSD parameter space: amplitude σ and timescale τ control burstiness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare three star-formation histories representing distinct quenching scenarios: (1) Constantly star-forming (no quenching), (2) Slowly quenched exponential decay (tau=4 Gyr, peak 6 Gyr ago), and (3) Rapidly quenched post-starburst (truncated skew-normal, peak 2 Gyr ago, width 0.3 Gyr). The resulting rest-frame SEDs exhibit markedly different colors, equivalent widths (Hα), and spectral slopes, highlighting how quenching timescale imprints on observable photometry and spectroscopy.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_quenching_pathway_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_quenching_pathway_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quenching pathways: fast vs slow termination of star formation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="CIGALE&#x27;s sfh2exp star-formation history superposes an old, exponentially declining main population with a second, more recent exponential burst that contributes a fixed fraction f_burst of the total stellar mass formed. It is the classic parametrization for post-starburst and rejuvenated systems.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh2exp_main_plus_burst_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh2exp_main_plus_burst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">sfh2exp: double declining exponential (old population + recent burst)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each parametric SFH in tengri encodes a different prior on when a galaxy forms its stars. We overlay the SFR(t) shape of nine production-status forms at their default parameter values, all integrated to the same total stellar mass, so the differences are entirely in the shape — not the normalization.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh_form_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh_form_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parametric SFH form atlas</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The parametric SFH atlas (``plot_sfh_form_compare.py``) shows seven classical analytic SFH shapes. Beyond those, tengri ships three non-parametric families that bin the mass formed in successive lookback intervals — useful when the data resolve more than ~5 SFR bins and you want a flexible prior that doesn&#x27;t impose a strong shape.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh_nonparametric_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh_nonparametric_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Non-parametric SFH families compared</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Generate stochastic star-formation histories using the Fourier-space GP correlated field model, governed by a damped-random-walk power spectrum. Left panel shows mild burstiness (σ=0.3, τ=300 Myr); right shows strong burstiness (σ=1.0, τ=100 Myr). Five realizations appear in each panel, with the smooth mean SFH overlaid.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_stochastic_sfh_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_stochastic_sfh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stochastic SFH samples from GP-correlated fields with different burstiness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How observable is an underlying ancient burst (10 Gyr ago) beneath a young (300 Myr) starburst? outshining problem in broadband photometry (Trager+ 2000, Renzini 2006): the young burst&#x27;s UV emission completely dominates over the ancient burst&#x27;s optical/IR, rendering the ancient population invisible to broadband SED fitting.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_two_burst_observability_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_two_burst_observability`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The outshining problem: young bursts eclipse ancient populations</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Metallicity
===========

Per-SSP-library Z☉ differs: MIST 0.0142, BC03/Padova 0.0190, PARSEC 0.0152, BASTI 0.0200. Cross-code comparisons must reason in absolute log(Z). Stellar and gas-phase Z are separate knobs. Age–metallicity degeneracy in broadband data. α-element enhancement.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The [α/Fe] abundance ratio encodes the chemical enrichment history: rapid enrichment by core-collapse supernovae before Type Ia SNe begin leads to high [α/Fe]. In the SED, enhanced alpha-elements suppress iron absorption lines in the optical (especially around 4000–5000 Å) because the higher abundance of alpha elements shifts the line-blanketing opacity. We sweep [α/Fe] on a quiescent passively evolving galaxy where iron features dominate the continuum absorption.">

.. only:: html

  .. image:: /auto_examples/metallicity/images/thumb/sphx_glr_plot_alpha_fe_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/metallicity/plot_alpha_fe_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Alpha-element enhancement suppresses iron absorption features</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Stellar metallicity affects the stellar continuum shape and overall energy balance. Dust emission responds to absorbed stellar photons: metal-poor hot stars emit bluer light with less IR-absorbed energy, while metal-rich cooler stars are less bright in the UV but more absorbed in the optical/NIR. We sweep stellar metallicity on a young star-forming galaxy at z = 0.2 with dust attenuation and thermal emission from warm dust.">

.. only:: html

  .. image:: /auto_examples/metallicity/images/thumb/sphx_glr_plot_logzsol_panchromatic_thumb.png
    :alt:

  :doc:`/auto_examples/metallicity/plot_logzsol_panchromatic`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Metallicity shapes panchromatic SED with dust emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Metal-rich young populations and metal-poor old populations can produce similar optical colors — a fundamental degeneracy in galaxy fitting. This 3×4 grid shows normalized rest-frame continua at nine points in the age–metallicity plane, with each row fixed at one lookback-formation age and each column fixed at one metallicity. Dust is zeroed to expose the clean stellar continuum shape.">

.. only:: html

  .. image:: /auto_examples/metallicity/images/thumb/sphx_glr_plot_metallicity_age_grid_thumb.png
    :alt:

  :doc:`/auto_examples/metallicity/plot_metallicity_age_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-metallicity degeneracy in the stellar continuum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Metallicity evolution Z(t) depends on the balance between metal production (in supernovae) and metal removal (via outflows). This four-panel figure shows how different star formation timescales and outflow efficiencies η alter the enrichment history relative to a closed box (zero outflow). Top-left: closed-box enrichment timescale dependence. Top-right: impact of variable outflow rates. Bottom-left: closed vs leaky enrichment under constant SFR. Bottom-right: age-metallicity relation analog — how different assembly epochs lead to different final metal content.">

.. only:: html

  .. image:: /auto_examples/metallicity/images/thumb/sphx_glr_plot_zh_evolution_compare_thumb.png
    :alt:

  :doc:`/auto_examples/metallicity/plot_zh_evolution_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Chemical evolution: closed-box vs leaky-box enrichment histories</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Nebular Emission
================

Emission lines are vacuum throughout: Hα is 6564.61 Å, not the 6562.8 Å air
value. Mixing the two shifts every line centroid.

``neb={'type': ...}`` takes ``ssp``, ``cue``, ``cb19``, ``cloudy`` or ``none``.
The default, ``ssp``, uses the emission already baked into a with-nebular (wNE)
SSP grid. The live backends instead compute it, and expect a bare stellar grid.
Feed a bare grid to the baked-in path and both continuum and line fluxes come
out low, with no error raised.

Gas-phase metallicity is its own knob and does not follow the stellar one. Shock emission examples include line-ratio diagnostics; a composable shock-group sweep is in development (Task 4).


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Emission lines are vacuum throughout: [OIII] = 5008.24 Å, [NII] = 6585.28 Å, Hα = 6564.61 Å, Hβ = 4862.68 Å. Overlays Kewley+2001 SF/AGN demarcation and Kauffmann+2003 SF/composite line.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_diagram_population_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_diagram_population`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT diagram: star-forming galaxies, AGN, and shocks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Cue (Li, Leja &amp; Speagle 2023) maps a four-dimensional HII region control space — ionization parameter log U, gas-phase metallicity log Z_gas, ionizing-spectrum shape, and dust-to-metal ratio — onto an emission-line spectrum. A two-dimensional sweep over the two knobs most users will turn (``log U`` and log Z_gas) is shown for four diagnostic line ratios.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_cue_parameter_atlas_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_cue_parameter_atlas`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comprehensive 2D sweep of ionization parameter and metallicity (Cue)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We sweep the ionizing-photon escape fraction f_esc from 0 to 1.0 at fixed log U and metallicity, showing both the broadband SED response and a zoomed view of the critical Lyman-continuum (912 A) region. The Lyman edge deepens as ionizing photons escape the ISM unabsorbed, suppressing optical line ratios simultaneously.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_fesc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_fesc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Escape fraction reshapes the SED from the Lyman continuum to optical lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Murphy+2011 SFR-Hα relation requires ionizing photons from stars younger than ~10 Myr. Constant-SFR models at ages 1–300 Myr show the calibration breaks at young (&lt;10 Myr; insufficient ionizing photons) and old (&gt;100 Myr; all stars too old to ionize) populations. We sweep stellar metallicity to show the calibration validity range is weakly sensitive to Z: higher Z reduces ionizing photon production, compressing the valid age window slightly toward older ages.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_halpha_sfr_calibration_age_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_halpha_sfr_calibration_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hα SFR calibration breaks at young ages, weakly dependent on metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Lyα rest-frame wavelength is 1216 Å (vacuum). EW peaks at 3–5 Myr when O-type stars dominate ionization, then decays past 10 Myr. Higher metallicity suppresses ionizing photon production, reducing peak EW.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_lyalpha_ew_vs_age_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_lyalpha_ew_vs_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-alpha equivalent width peaks at young ages, varies with gas metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare four nebular emission backends on identical star-forming spectra:">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_nebular_backends_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_nebular_backends`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular backends: Cue, CloudyGrid, SSP-embedded, and BakedIn</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hydrogen-ionizing photon production (Q_H, photons/s per solar mass) depends critically on stellar population age. Young starbursts (age ≈ 3–5 Myr) produce ionizing photons at peak rates; by 100 Myr, Q_H drops by ~3 orders of magnitude. We show how this evolution varies across metallicity Z = [-1.0, -0.5, 0.0, +0.3] using FSPS bare-stellar (non-nebular) SSP templates, as ionizing photons are consumed by CLOUDY during wNE SSP generation and would appear suppressed.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_qh_vs_age_metallicity_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_qh_vs_age_metallicity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionizing photon production rate Q_H peaks sharply with stellar age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Shock emission (MAPPINGS V models) can mimic AGN on the BPT diagram. We show how shock velocity, gas density, and magnetic field strength affect line ratios and diagnostic positions. Four-panel layout shows velocity and density sequences on BPT, line ratios vs velocity, and magnetic field strength.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_shock_emission_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_shock_emission`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MAPPINGS V shocks: velocity, density, and magnetic field effects</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The shock group adds MAPPINGS V shock emission as an additive component to any photoionized nebular backend (Cue, CloudyGrid, CB19, or baked-in). Here we show how increasing shock contamination (shock_frac) moves a star-forming galaxy from the SF locus toward the LI(N)ER/shock region of the BPT diagnostic plane.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_shock_frac_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_shock_frac_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Shock-group sweep: composable shock contamination on the BPT diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Five widely-used optical strong-line metallicity diagnostics evaluated across the Cue logZ_gas prior. Each one carries a different systematic — Pettini &amp; Pagel 2004 O3N2 saturates at high Z, the R23 ratio is double-valued (Pagel+1979), N2 (Marino+2013) is monotonic but small dynamic range, the [O III]/[O II] diagnostic tracks ionization, and [S II]/[O II] is a low-ion proxy.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_strong_line_metallicity_diagnostics_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_strong_line_metallicity_diagnostics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Strong-line gas-phase metallicity diagnostics</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Dust Attenuation
================

Two-component Charlot & Fall geometry: ``dust_tau_bc`` on the birth clouds,
``dust_tau_diff`` on the diffuse ISM. ``dust_slope`` defaults to -0.7, the
diffuse-ISM value; -1.3 is the birth-cloud one. The 2175 Å bump is a separate
always-on modifier, ``dust_bump_strength``, defaulting to 0.0 — Calzetti
carries no bump unless you ask for one.

Dust emission templates load from ``data/``. There is no analytic fallback: a
missing template raises ``FileNotFoundError`` rather than quietly substituting
a worse model.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Charlot &amp; Fall 2000 two-component dust model splits attenuation into a birth-cloud component (``τ_bc``) that only the youngest stellar ages see, and a diffuse-ISM component (``τ_diff``) that attenuates all stellar light. The two are degenerate for an old population but separate cleanly for a young one.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_birth_cloud_vs_diffuse_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_birth_cloud_vs_diffuse`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Birth-cloud vs diffuse-ISM dust: age dependence and parameter degeneracies</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Kriek &amp; Conroy attenuation law has two degrees of freedom: bump strength and UV slope (δ). Varying both reveals how steeper UV slopes suppress the apparent prominence of the 2175 Å bump relative to the surrounding continuum. We show a 2×2 grid: rows sweep bump strength (0–2 at fixed δ), columns sweep δ slope (−1, +0.5 at fixed bump), revealing the synergy — a steep negative slope (blue wing) enhances bump visibility, while shallow positive slopes (flattened UV) bury the bump in the continuum.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_bump_delta_joint_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_bump_delta_joint_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2175 Å bump × UV slope interaction in Kriek & Conroy attenuation</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduction of Fig. 7 of Buchner et al. (2024, GRAHSP): a star-forming galaxy SED from intrinsic (dark blue) to strongly attenuated (dark red) as the diffuse color excess E(B-V) is swept from 0.01 to 10. Energy balance routes the attenuated UV/optical light into the far-IR dust bump (Dale 2014), so the curves pivot about the FIR peak while the UV is progressively suppressed.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_grahsp_paper_fig7_galaxy_attenuation_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_grahsp_paper_fig7_galaxy_attenuation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP Fig. 7 reproduction: attenuation of the galaxy model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed UV slope β_UV (the observable astronomers measure), many (R_V, A_V) pairs produce identical colors — this is a classical dust modeling pitfall. Shows β_UV as contours on the (R_V, A_V) grid for Cardelli MW attenuation. Standard reference points (SMC, LMC, Milky Way diffuse, Calzetti starburst) sit on different iso-β_UV contours, illustrating why dust-law assumptions strongly bias inferred properties.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_rv_av_uv_slope_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_rv_av_uv_slope_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rv and Av degeneracy in UV slope: the Calzetti trap</div>
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


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="For a foreground dust screen the attenuation curve has a fixed shape — its amplitude scales with tau_V but the UV-to-optical ratio is constant, so a single k(lambda) law captures it. Witt &amp; Gordon (2000) showed this breaks down once dust and stars are mixed: high-``tau_V`` sightlines self-shield, the short-wavelength photons preferentially escape through low-opacity channels, and the effective curve greys (flattens) as tau_V rises. The curve shape is therefore a function of tau_V — which is exactly why tengri ships WG00 as a radiative-transfer table (FSPS dust_type=3), interpolated in tau_V, rather than a fixed-shape law.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_wg00_tau_v_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_wg00_tau_v_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Witt & Gordon 2000: the attenuation shape greys with optical depth</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Dust Emission
=============

Dust emission templates auto-load from ``data/``; analytic fallbacks are not suitable for science. PAH features in Draine & Li templates (q_PAH and U_min sweeps). Mid-IR PAH diagnostics distinguish star-forming, AGN, and composite systems. Temperature sweeps. Template libraries: BOSA, THEMIS, PAHspec, Astrodust (HD23).


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The BOSA infrared template library is parametrized jointly by total infrared luminosity log L_TIR and specific star formation rate log sSFR. Neither axis alone tells the full story: at fixed sSFR the FIR peak migrates with L_TIR (dust temperature), while at fixed L_TIR the PAH mid-IR forest brightens with sSFR. Three side-by-side panels at fixed sSFR overlay three L_TIR values each, making the 2-D dependence legible in a single figure rather than two skinny 1-D loops.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_bosa_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_bosa_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA library: PAH features and FIR peak depend on both sSFR and L_TIR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 2-D grid on the Draine &amp; Li 2007 template library: rows step through PAH mass fraction q_PAH (controls mid-IR PAH-feature strength), columns through the minimum radiation field U_min (sets the diffuse dust temperature, i.e. the FIR peak position). The two axes act nearly orthogonally — a surprise for anyone who would lump them together as &quot;PAH knobs.&quot;">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_qpah_umin_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_qpah_umin_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The q_PAH and U_min knobs move PAH amplitude and FIR peak independently</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All dust IR-emission libraries shipped in tengri, shown on two scales:">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_ir_library_compare_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_ir_library_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR-emission library comparison: models and templates</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Casey 2012 modified blackbody dust SEDs across the canonical fitter&#x27;s two knobs — dust temperature T_dust and emissivity index β. Each curve in the top panel is a fixed β = 1.8 MBB swept in T; the bottom panel fixes T = 30 K and sweeps β. The peak shifts by ~40 μm per 10 K of warming; the sub-mm slope steepens by one power-law index per Δβ = 1.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_mbb_temperature_beta_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_mbb_temperature_beta_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Modified blackbody: T_dust × β grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 5–30 μm rest-frame spectrum showcases distinct infrared tracers: dust polycyclic aromatic hydrocarbon (PAH) emission peaks at 6.2, 7.7, 8.6, 11.3, and 12.7 μm in star-forming galaxies, while silicate absorption (9.7 μm Si–O stretch) and AGN heating suppress PAH and introduce continuum growth in AGN-dominated systems. We model three templates: (a) pure starburst (no AGN), (b) pure AGN (no star formation), and (c) composite with AGN fraction = 0.5. the diagnostic power of mid-IR spectroscopy: PAH strength probes star formation rate, while continuum slope and silicate depth reveal AGN heating and dust temperature.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_mid_ir_pah_features_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_mid_ir_pah_features`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mid-IR PAH features in star-forming, AGN, and composite galaxies</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep across the 13 published PAHspec starlight spectra (mMMP, m31bulge, BC03/BPASS SSPs) at fixed ionization parameter. Demonstrates strong dependence of PAH features on starlight hardness.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_pahspec_starlight_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_pahspec_starlight_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine+2021 PAHspec: starlight-spectrum sweep at fixed log U</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="For Draine &amp; Li (2007) dust at fixed mass, raising the diffuse radiation field intensity U_min does two things at once: it shifts the SED peak blueward (warmer dust) and proportionally boosts the total far-IR luminosity (``L_IR`` ∝ U_min). The standard T_peak–``L_IR`` correlation seen in observations is the joint footprint of these two effects.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_tdust_vs_lir_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_tdust_vs_lir`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radiation field strength sets both dust peak temperature and L_IR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Jones et al. (2017) THEMIS dust model distributes grains over a range of starlight intensities U with a power law dU/dM \propto U^{-\alpha}. The slope alpha controls how much warm, intensely-illuminated dust contributes relative to the cold diffuse component: a smaller alpha puts more mass at high U, shifting the FIR peak blueward and filling in the mid-IR.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_themis_alpha_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_themis_alpha_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS dust IR: radiation-field slope (alpha)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust re-radiates absorbed starlight across a broad range of temperatures: colder dust (e.g., diffuse cirrus at ~20 K) peaks in the far-infrared (~250 μm), while warmer dust grains (e.g., starburst regions at ~40 K) peak at shorter wavelengths (~50–100 μm).">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_warm_cold_dust_decomposition_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_warm_cold_dust_decomposition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR SED: Warm and cold dust decomposition</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

AGN Models
==========

Torus models in `components/agn/torus.py` are toy models; SKIRTOR is the one for science. Disc continua (multicolor, KD18, relagn, qsogen), narrow-/broad-line and FeII emission, polar-dust and Type 1/2 attenuation. Cross-validated against CIGALE and AGNfitter.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All thirteen accretion-disc backbones registered under agn.disc.type, at fixed bolometric luminosity log L_bol = 12.5 (in log L_sun), evaluated in isolation with the host suppressed and no torus/lines/dust. The differences between the curves are entirely how each model partitions the disc power across wavelength: pure blackbody vs warm Comptonization, relativistic vs Newtonian potential, radiatively efficient thin disc vs inefficient ADAF, empirical composite vs first-principles continuum.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_disc_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_disc_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN disc continuum: every registered model at fixed L_bol</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Iron pseudo-continuum (Fe II) emission in AGN produces characteristic humps in the near-UV and optical bands. The strength and shape are governed by the Fe II equivalent width and ionization state, parameterized in tengri by the agn_fe2_strength parameter relative to H-beta (Balmer lines).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_feii_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_feii_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fe II pseudo-continuum strength evolution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Until recently the agn_ parameters were declared with fixed* defaults and no prior range, so the build grammar&#x27;s FREE controls (``agn={&#x27;all_params&#x27;: FREE}``, recipes.agn_panchromatic()) silently resolved every AGN parameter to a constant — a fit would freeze the entire AGN sector with no error. The registry now gives each parameter a physically-motivated Uniform/``LogUniform`` prior (Nenkova+2008, Kubota &amp; Done 2018, Stalevski+2016 grid extents), so FREE actually frees them.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_free_param_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_free_param_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN parameters are free-able now — and every one moves the SED</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four AGN configurations of increasing physical complexity at the same bolometric luminosity (log L_bol = 12.5 in L_sun units) — bare multicolor disc, +SKIRTOR torus, +NLR narrow-line forest, and an empirical QSOgen template that bundles all of the above. The reader sees which spectral feature each block introduces (mid-IR torus bump, optical narrow lines, broad UV continuum) and which are essentially universal across the modeling choice.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_hierarchy_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_hierarchy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Building up an AGN SED: disc, then torus, then lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The renderable line backbones registered under the three composable line selectors — agn.nlr (narrow-line region), agn.blr (broad-line region), and agn.feii (iron pseudo-continuum) — each layered on the same disc + torus at fixed log L_bol = 12.5. The backbone controls which optical/UV features the model produces: narrow forbidden lines, broad permitted lines, or the blended Fe II forest.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_lines_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_lines_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN emission-line backbones compared</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The QSOgen model (Temple+2021) includes empirical UV/optical emission-line forest and broad Balmer continuum. The relative strength of these line features with respect to the continuum obeys the Baldwin effect: luminous quasars show weaker equivalent-width emission lines (the line flux grows sublinearly with continuum). This sweep shows the Baldwin effect in the QSOgen template across six decades of bolometric luminosity (log L_bol = 9 to 13 L_sun), revealing the Ly-alpha + C IV feature cluster around 1000–1600 Å and optical hydrogen Balmer lines (Hα, Hβ).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_qsogen_emline_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_qsogen_emline_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen emission lines: Baldwin effect across AGN luminosity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All ten dusty-torus libraries registered under agn.torus.type, reprocessing the same accretion-disc continuum at fixed log L_bol = 12.5 (in log L_sun) and standard inclination. The disc is held at multicolor (Kubota &amp; Done 2018) so the differences in the curves are entirely how each torus library geometrically distributes hot grains and re-emits the absorbed UV in the MIR — clumpy radiative transfer (SKIRTOR, CLUMPY, CAT3D-WIND) vs smooth-dust grids (Fritz, Silva) vs phenomenological graybodies.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_torus_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_torus_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN dusty torus: library comparison at fixed L_bol</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The agn.disc, agn.lines, agn.feii, agn.torus, agn.atten sub-blocks of SEDModel.build are composable: turning one on at a time and overlaying the all-on reference (dashed gray) shows which features each sub-block contributes. Five panels at fixed log L_bol = 12.0, all built via the public nested-dict grammar:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_composable_block_toggles_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_composable_block_toggles`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cumulative buildup of the GRAHSP AGN recipe, one sub-block at a time</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A toy single-temperature blackbody torus implemented as a modern SEDModelComponent subclass, discoverable through SEDModel.build and composable with other AGN blocks. The SEDModelComponent pattern is the recommended path for any new SED physics — AGN, dust, or stellar.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_custom_torus_extension_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_custom_torus_extension`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Custom AGN torus model via SEDModelComponent and direct integration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Kubota &amp; Done (2018) three-zone accretion disc model shows how the big blue bump (BBB) peaks at different wavelengths depending on black-hole mass and Eddington ratio. Sweeping across the accretion-state plane from low-luminosity advection-dominated (ADAF-like) to high-Eddington thin-disc reveals the transition: high mass + low Eddington gives cool outer discs peaking in the NIR; low mass + high Eddington gives hot inner zones peaking in the FUV/UV.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_kd18_disc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_kd18_disc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kubota & Done 2018 disc: Accretion state effects on continuum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Polar dust disc attenuation applies only to Type 1 (face-on) sightlines — the equatorial torus already screens the disc for Type 2. The bi-conical polar dust absorbs disc photons regardless of viewing angle, however, and re-emits them isotropically as a FIR graybody (Casey 2012). So both Type 1 and Type 2 sweeps show the FIR re-emission bump growing with E(B-V); only the UV/optical attenuation is gated by sightline.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_polar_dust_ebv_type12_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_polar_dust_ebv_type12_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polar dust E(B-V) reddens Type 1 & 2 AGN differently</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In a relativistic accretion-disc model the inner boundary sits at the innermost stable circular orbit (ISCO). Higher spin shrinks the ISCO, raises the inner-disc temperature, and shifts disc power blueward — the UV spectral slope alpha (L_nu ~ nu^alpha across 912 to 3000 Å) hardens monotonically with spin. We sweep a_spin from 0 to 0.998 on the Kubota &amp; Done (2018) disc backbone, the public-API entry point for spin-sensitive disc physics in tengri, and report alpha alongside the SEDs.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_relagn_spin_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_relagn_spin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Black-hole spin hardens the UV slope through ISCO migration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is tengri&#x27;s full-grid SKIRTOR torus (Stalevski+2012, 2016), following the X-CIGALE skirtor2016 conventions: a 5-D clumpy two-phase library indexed by equatorial optical depth tau, radial and polar density gradients p / q, half-opening angle oa, and inclination cos i (plus an optional Casey-2012 polar-dust graybody). It is the science-grade counterpart to the parameter-averaged skirtor_agnfitter library — and, having the full grid, it responds strongly to its parameters.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_xcigale_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_xcigale_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus (full X-CIGALE grid): optical depth and inclination</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Unified AGN models explain the Type 1/Type 2 dichotomy as a purely geometric effect — the same accretion disc + dusty torus system appears as:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_type1_type2_unified_model_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_type1_type2_unified_model`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Type 1 vs Type 2 AGN: Unified viewing-angle classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sanders et al. (1988) proposed that Ultra-Luminous Infrared Galaxies (ULIRGs) are the dust-shrouded precursors to optical QSOs. This sequence traces progressive unveiling of a buried AGN through five stages:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_ulirg_to_qso_transition_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_ulirg_to_qso_transition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">ULIRG→QSO evolutionary sequence: dust-obscured starburst to bare quasar</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

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

X-ray Emission
==============

X-ray binaries (HMXB, LMXB) scaled with SFR and stellar mass. AGN coronae: luminosity, photon index γ, exponential cutoff E_cut, UV-to-X-ray slope α_ox. Model-family comparison included.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The CIGALE-faithful corona derives the X-ray normalization from L_2500 via the empirical alpha_OX-L_2500 correlation. tengri ships three published parametrizations:">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_alpha_ox_relations_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_alpha_ox_relations`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Three empirical alpha_OX-L_2500 prescriptions diverge at the quasar peak</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The UV-to-X-ray spectral slope alpha_OX (defined as log F_X minus log F_UV divided by log nu_X minus log nu_UV) separates X-ray-loud quasars (alpha_OX around -1.2, strong X-ray relative to the UV continuum) from X-ray-quiet systems (alpha_OX around -1.8, suppressed X-ray). The CIGALE-faithful corona derives alpha_OX from L_2500 via the Just+2007 relation by default; here we sweep delta_alpha_ox to apply offsets from -0.4 to +0.4 around that empirical value, at fixed L_2500 (= L_bol = 1e45 erg/s through the standard Hopkins+2007 bolometric correction). More positive delta brightens the corona; more negative suppresses it.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_alpha_ox_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_alpha_ox_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN UV-to-X-ray power-law slope alpha_OX controls X-ray normalization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The X-CIGALE X-ray module (Yang et al. 2020) sums four physically distinct emitters: the AGN corona (a cut-off power law normalized through the α_OX–L_2500 relation), low- and high-mass X-ray binaries (LMXB ∝ M⋆, HMXB ∝ SFR; Lehmer et al. 2016 metallicity/age scalings), and a hot interstellar-gas term (∝ SFR). This reproduces Yang+2020 Figure 1 for a typical AGN host: L_2–10 keV = 10⁴³ erg s⁻¹, M⋆ = 10¹¹ M⊙, SFR = 10 M⊙ yr⁻¹, T = 1 Gyr, Z = 0.02.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_component_decomposition_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_component_decomposition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray SED decomposition: AGN, LMXB, HMXB, hot gas</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Every X-ray model tengri registers, on one host galaxy, with only the xray block changing. At default parameters they collapse onto two curves, and that is the point of the figure rather than a defect in it:">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_model_family_compare_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_model_family_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray model family: five names, two prescriptions</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The line-of-sight column density N_H reshapes the AGN X-ray spectrum in two regimes: photoelectric absorption (``zphabs``) suppresses the soft band roughly as \exp(-\sigma(E)\,N_H) with cross-section \sigma \propto E^{-3}, while Compton down-scattering (``cabs``) adds an energy-independent suppression \exp(-\sigma_T\,N_H) that becomes dominant once log N_H ≳ 24 (the Compton-thick boundary). A constant warm-electron scattered fraction (~1 % of the intrinsic continuum) is added back, which is the only flux observable in the soft band for nearly opaque columns and explains why Compton-thick AGN are still marginally detectable in soft-band stacks (Matsumoto et al. 2026 Fig. 11/12).">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_nh_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_nh_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">N_H column density sweep: from unobscured to Compton-thick</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In Compton-thick AGN (log N_H ≳ 24 cm⁻²), the line-of-sight obscurer extinguishes the primary AGN corona below ~ 10 keV. What&#x27;s left is the reflected component — the fraction of corona photons that hit the cold accretion disc, Compton-scatter off bound electrons, and emerge along the line of sight without being photoelectrically absorbed. The resulting spectrum peaks around 30 keV (the famous Compton hump) and is the smoking-gun signature that NuSTAR / Swift-BAT surveys use to confirm buried supermassive black holes (Ricci+2017, Matsumoto+26).">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_pexrav_compton_hump_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_pexrav_compton_hump`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compton hump in obscured AGN: pexrav reflection across log N_H</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="X-ray binaries (XRBs) are the dominant X-ray sources in star-forming galaxies once an AGN is excluded. High-mass XRBs trace the recent star-formation rate (Mineo+2012), while low-mass XRBs trace the integrated stellar mass (Lehmer+2019). The two scalings have different spectral shapes too: HMXBs are slightly harder, LMXBs slightly softer. Two side-by-side sweeps — SFR (left) at fixed M_star = 1e11 M☉, and M_star (right) at fixed SFR = 10 M☉/yr — separate the two channels on the same axes.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_sf_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_sf`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray binary luminosity scales with SFR (HMXB) and stellar mass (LMXB)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

IGM
===

Intergalactic-medium absorption: Madau vs Inoue prescriptions, Lyα forest, damped Lyα systems. Lyman-break/dropout signature in high-z photometric selection. IGM `igm_transmission(wave_obs, z)` takes observed-frame wavelengths (not rest-frame).


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


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

Photometry
==========

Broadband filter selection, cosmological dimming, color tracks and redshift evolution. Diagnostic planes: WISE/IRAC AGN wedges, red sequence/blue cloud. Photometric-redshift color degeneracies.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 4000 Å break is a sharp discontinuity in the stellar continuum at the boundary between the Balmer and Paschen series, caused by hydrogen Lyman absorption blanketing in the overlying atmosphere. In the rest frame it sits at 4000 Å for all galaxies. In the observer frame, the break shifts to longer wavelengths with increasing redshift: z × 4000 Å. This is why different photometric bands probe the break at different redshifts — the fundamental principle behind photo-z estimation and dust/age degeneracies.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_balmer_break_redshift_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_balmer_break_redshift_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Balmer break (4000 Å) position in observed-frame filters vs redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does a galaxy&#x27;s location in color–color space evolve with redshift? We compute SDSS g − r and r − z colors for two galaxy populations — a young star-forming and an old quiescent — across z = 0 to 3, with arrows marking the integer redshift stops. This is the reference picture for photometric redshift classifiers and for stellar-template grids.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_color_tracks_redshift_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_color_tracks_redshift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photometric color tracks vs redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does the observed photometric flux of a FIXED-luminosity galaxy decline with redshift? We track a star-forming galaxy (log M* = 10.5, SFR = 10 M☉/yr) across z = 0.1 to 6 in three optical/infrared bands (SDSS r, JWST J, JWST H), visualizing the three physical effects:">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_cosmic_dimming_observed_flux_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_cosmic_dimming_observed_flux`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cosmic dimming and K-correction with redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Show a typical star-forming galaxy SED at z=1 with observed-frame filter throughputs overlaid as semi-transparent fills from 0.3 to 25 μm. This helps visualize which rest-frame stellar and dust features each photometric system samples across the spectrum.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_filter_throughput_overlay_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_filter_throughput_overlay`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">HST+JWST+LSST+Spitzer Filter Overlay on Star-Forming SED at z=1</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two galaxies with very different star formation histories and dust can collide in color–color space, making photo-z ambiguous. Here, a young dusty star-forming galaxy at z≈0.5 and an old quiescent galaxy at z≈2 follow nearly identical (u-g, g-r) tracks and intersect at a single point. This shows why intermediate-wavelength photometry is essential for robust photo-z classification.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_photoz_color_degeneracy_grid_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_photoz_color_degeneracy_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photo-z degeneracy in color–color space: low-z dusty vs high-z quiescent</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Galaxy color–magnitude diagram showing the distinct red and blue populations. We model two populations — 25 quiescent old galaxies (peak SFH ~8 Gyr) and 25 star-forming galaxies (continuous SFR) — varying stellar mass via log_total_mass. Each sample is placed at z = 0.05, computing u − r color and rest-frame M_r magnitude. The color bimodality and green valley are key signatures of galaxy assembly across cosmic time (Strateva et al. 2001 SDSS, Baldry et al. 2004).">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_red_sequence_blue_cloud_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_red_sequence_blue_cloud`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Red Sequence vs Blue Cloud Bimodality</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The WISE color-color diagram (Stern et al. 2012) is a tool for separating AGN from star-forming galaxies using mid-infrared colors. The diagnostic exploits the fact that AGN emit power-law SEDs (flat in νLν) while star-forming galaxies have cooler dust emission (Rayleigh-Jeans slope at long wavelengths).">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_wise_agn_color_color_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_wise_agn_color_color`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">WISE W1–W2 vs W2–W3 Color-Color Diagram with Stern+2012 AGN Wedge</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Spectroscopy
============

Absorption-line indices (D4000, Hδ) from stellar age and metallicity. Spectral indices vs age. Velocity dispersion, line broadening, and velocity offset. Instrumental resolution effects. High-redshift example: z ≈ 6 Lyα emitter.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Stellar absorption features, especially the Mg b and Fe5270 line strengths, encode both age and metallicity in a classical anti-correlation pattern: at fixed metallicity, both features strengthen with age (population becomes older, cooler); at fixed age, increasing metallicity also strengthens the features (enhanced α-element abundances + stronger metal absorption).">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_bandheads_age_metallicity_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_bandheads_age_metallicity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar absorption bandheads: age and metallicity anti-correlation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Population diagnostics: single-burst SSP populations (3 SFH shapes × 5 ages × 3 metallicities = 45 points) colored by SFH shape and marked by metallicity. The Hδ_A vs D_n(4000) diagram discriminates starburst (high Hδ_A, low D_n(4000)) from quiescent (low Hδ_A, high D_n(4000)) populations and is sensitive to recent star formation and metal enrichment.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_d4000_hdelta_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_d4000_hdelta_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kauffmann+2003 D_n(4000) vs Hδ_A Diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="High-redshift Lyα emitter at z=6 with young age (~10 Myr), low metallicity (Z~0.1 Z☉), and minimal dust. The observed-frame spectrum (7000–13000 Å) reveals the redshifted Lyα emission line at 8512 Å, the Lyman break at 6384 Å, characteristic IGM blue-wing absorption, and the rest-UV continuum. Demonstrates Lyα radiative transfer and reionization-era observability.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_lae_spectrum_z6_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_lae_spectrum_z6`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-alpha emitter spectrum at z=6: IGM absorption and Lyα escape</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spectral resolution R determines whether the Hα + [N II] emission-line complex appears as a single blended feature (low R) or resolves into three distinct lines (high R). Varying R from 100 to 10000 reveals the transition from kinematically degenerate at R~100 (SDSS/DESI-like) to fully resolved at R~5000 (JWST-like).">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_resolution_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_resolution_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Instrumental resolution controls Hα + [N II] line blending</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three of the most-used optical absorption / emission diagnostics evaluated on a single-burst stellar population from 30 Myr to 13 Gyr, at solar metallicity, no dust. The figure makes obvious which diagnostic responds on which timescale:">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_spectral_indices_vs_age_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_spectral_indices_vs_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Classic spectral indices vs single-burst age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep stellar velocity dispersion σ_v ∈ {50, 100, 150, 250, 400} km/s to show how line broadening increases with dynamical heating. The Mg b absorption feature (~5170 Å) widens progressively, demonstrating the kinematic signature of higher-velocity stellar populations.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_velocity_dispersion_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_velocity_dispersion_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Velocity Dispersion Sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Narrow-line regions sit at σ_v ≈ 50–300 km/s; a broad Hα component from the AGN accretion disk reaches thousands of km/s. The [NII] doublet is separated by 35.4 Å (6549.86 and 6585.28 Å vacuum), which corresponds to σ_v ≈ 1600 km/s — above that the two lines merge into the wing of Hα.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_velocity_offset_lines_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_velocity_offset_lines`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Emission-line velocity dispersion: narrow [NII] to broad Hα</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Use Cases
=========

Paper-style diagnostics: UVJ, JWST color-color, SFR indicators, age–dust degeneracy, main sequence evolution, dropout selection, spectral indices. Simulated-population Catalog examples.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Optical photometry alone cannot uniquely break the degeneracy between stellar age, dust attenuation, and redshift — a fundamental limitation in photo-z and SED fitting. Three physically distinct galaxy populations can produce nearly identical SDSS ugriz photometry:">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_age_dust_redshift_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_age_dust_redshift_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The age-dust-redshift degeneracy in photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Balmer decrement measures dust attenuation via hydrogen recombination line ratios: Hα / H-beta is sensitive to extinction (Calzetti et al. 2000). Without dust, the intrinsic ratio is ~2.78–2.86 (Case B). Here we sweep dust optical depth (τ_diff ∈ [0, 2]) and measure how the predicted Hα and H-beta change. We derive A_V = 1.086 × τ_diff and compare against the Calzetti+2000 expectation.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_balmer_decrement_av_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_balmer_decrement_av`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Balmer Decrement Tests Dust Attenuation on Emission Lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Kauffmann+2003 separation of star-forming and quiescent SDSS galaxies plotted as a sample track: stellar-burst age varied from 30 Myr to 11 Gyr (single-burst SSP), with each model giving a (``D_n(4000)``, sSFR) pair.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_d4000_vs_ssfr_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_d4000_vs_ssfr`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">D_n(4000) – specific SFR: the Kauffmann+2003 sequence</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Steidel+1996 U-dropout box is calibrated for a specific filter set and does not transfer to arbitrary filters: (U − G) &gt; 1.0, (G − R) &lt; 1.5, (U − G) &gt; 1.5(G − R) + 0.3. True z~3 galaxies cluster inside; lower-redshift galaxies fall outside.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_dropout_selection_z3_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_dropout_selection_z3`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">z~3 Lyman-break galaxy U-dropout selection: color-color diagnosis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Coleman, Wu &amp; Weedman 1980 spectral templates remain the textbook illustration of how the integrated SED morphs along the Hubble sequence — from quiescent ellipticals with deep 4000 Å breaks to gas-rich irregulars dominated by ongoing star formation and nebular emission.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_hubble_sequence_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_hubble_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A morphological atlas: E, Sa, Sb, Sc, Im galaxy SEDs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="JWST NIRCam color-color diagnostics for high-z galaxy classification">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_jwst_color_color_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_jwst_color_color`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">JWST NIRCam color-color diagnostics for high-z galaxy classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The star-forming main sequence (MS) defines a tight relation between stellar mass (M*) and star formation rate (SFR) for actively forming galaxies. This example demonstrates how the MS shifts upward by ~0.7 dex from z=0 to z=2, reflecting the Universe&#x27;s peak epoch of star formation. The left panel shows recovery of the z~0 MS from mock SEDModel photometry; the right panel reveals MS evolution to high-z.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_main_sequence_cosmic_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_main_sequence_cosmic_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Star-forming main sequence: z = 0 → 2 cosmic evolution + recovery</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a population of N=200 quiescent galaxies replicating the SDSS Luminous Red Galaxy (LRG) sample selection (Eisenstein et al. 2001, SDSS-I): old, massive systems at z~0.3 with log M* ≈ 11 and ages sampling the red-sequence range Uniform(6, 11) Gyr (Thomas et al. 2005).">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sdss_lrg_stack_template_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sdss_lrg_stack_template`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SDSS Luminous Red Galaxy Stacked Template Spectrum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Star formation rate calibrations depend on which wavelengths we observe. At high dust optical depth, UV-only SFR estimators severely underestimate the true SFR because dusty starbursts radiate most energy in the infrared. The hybrid SFR(UV+IR) recipe recovers the true SFR by combining both tracers.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sfr_uv_ir_consistency_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sfr_uv_ir_consistency`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SFR calibrations: UV only vs UV+IR hybrid estimators vs dust optical depth</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Replacing metallicity history Z(t) with its mass-weighted mean introduces 10–23% flux errors in u and 1–6% in z. The SED is a nonlinear mass-weighted sum of SSP templates; young metal-rich stars (dominant in UV) and old metal-poor stars do not average.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_simulation_seds_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_simulation_seds`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Predicting SEDs for a simulated population: what collapsing Z(t) costs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The infrared excess (IRX = L_IR / L_FUV) versus UV-continuum slope β diagram is the standard tool for inferring attenuation in star-forming galaxies. However, β is degenerate between dust and stellar age: young dusty and old dust-free populations both exhibit red UV continua.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uv_slope_beta_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uv_slope_beta`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV slope β degeneracy: dust optical depth and stellar age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Rest-frame U−V vs V−J colors separate star-forming from quiescent galaxies. The Williams+2009 quiescent wedge marks the boundary between dusty star-forming and passive systems.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uvj_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uvj_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UVJ diagram: rest-frame colors separate star-forming from quiescent</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Advanced Topics
===============

Extension-point demonstration (SEDModelComponent), Fisher degeneracy, and validation techniques: gradient vs finite-difference, mass conservation, redshift-frame invariance, WavePrecomp accuracy.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The on-ramp for adding a custom physics block to tengri. Subclass SEDModelComponent, declare name, parameter_prefix, priors as class attributes, and implement predict(p, sed_in, wave). __init_subclass__ registers the new variant and auto-fills the inputs() / outputs() contracts.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_custom_attenuation_component_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_custom_attenuation_component`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Authoring a new physics block with SEDModelComponent</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="tengri is a differentiable JAX package. Every model gradient ∂L/∂θ computed via jax.grad() should numerically match a central finite-difference approximation. This diagnostic builds a star-forming model with several free parameters, defines a chi-squared loss, and compares autodiff vs FD gradients for each parameter. A mismatch (&gt;1e-3) indicates a non-differentiable operation.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_gradient_finite_difference_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_gradient_finite_difference`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Autodiff gradients vs. finite-difference derivatives: diagnostic verification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Internal consistency check: the cumulative SFR integral ∫₀ᵗ SFR(t) dt should equal the stellar mass returned by predict_properties(). This diagnostic varies the DPL SFH parameters and verifies that the two pathways (manual trapz of the trajectory vs library integration) agree to ~0.1%. Discrepancies &gt; 5% trigger a warning and would indicate a bug in either the SFH trajectory or the mass integration kernel.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_mass_conservation_sfh_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_mass_conservation_sfh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mass conservation in SFH: manual integration vs predict_properties</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The rest-frame SED depends only on intrinsic galaxy properties (SFH, dust, metallicity, nebular, AGN) and is independent of redshift. Redshift only enters via the observation (wavelength shift, distance dimming, IGM attenuation). This diagnostic verifies that Prediction.rest_sed returns bit-identical SEDs across a range of redshifts for identical intrinsic parameters. Age-of-the-Universe constraints at high-z may truncate the SFH legitimately, producing smooth variation; any non-smooth jump signals a coupling bug.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_redshift_rest_invariance_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_redshift_rest_invariance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rest-frame SED Redshift Invariance</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The WavePrecomp approximation pre-integrates SSP × filter LUTs and interpolates photometry through a redshift table, trading exact calculations for speed. This diagnostic compares exact-wave-grid photometry against WavePrecomp variants at different ztable densities n_z, showing how fractional errors decrease with finer redshift grids.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_waveprecomp_accuracy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_waveprecomp_accuracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">WavePrecomp photometric accuracy across redshift grids</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cramér-Rao bound from the Fisher Information Matrix shows that SDSS 5-band photometry alone cannot separately constrain age, dust, and metallicity. Adding NIR or MIR bands breaks the degeneracy by factors of 2–5×, quantifying the information gain from multiwavelength coverage.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_fisher_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_fisher_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-Dust-Metallicity Degeneracy: Fisher Analysis</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Showcase
========

Full-stack demonstrations: population forward modeling, gradient diagnostics, end-to-end workflows.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

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


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

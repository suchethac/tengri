AGN-SED and stellar mass determination
Hagai Netzer 20.5.13

How to use the various emission components in this directory for creating SED templates
There are three major components:

1. Intrinsic continuum: "table_of_disk_models" gives 10 generic accretion disk spectra that cover the parameter range
 MBH=10^6 10^9
 Mdot/Mdot(Eddington)=0.03-0.3 (this is like L/Ledd)
 a (spin) = 0.0 -0.998
The table can easily be reduced to about 6-7 generic models - let me know if this is required.

The table lists frequency, wavelength and L_nu (erg/sec/Hz) for face-on standard accretion disks
with Comptonization and relativistic corrections included.
The recommendation is to use normalized SEDs, i.e. ignore the absolute flux level.

Normalization:
All normalization is relative to the disk continuum at 5100A. To do this calculate 
   L(5100)=lambda_lambda(5100) from the values of L_nu listed in the table

2. Emission lines:
   There are three separate components:
 -  Broad emission lines  
 -  Narrow emission lines
    either high ionization (Seyfert 1s and Seyfert 2s)
    or low ionization = LINERs (both type 1 and type 2)
 -  Broad FeII emission line - a template of the complete spectrum:  fe2_ferland_convolved 
     col1 - wavelength col2 theoretical flux (F_lambda erg/sec/A)
 This spectrum has to be integrated over wavelength to obtain the total normalized luminosity) L(FeII)

How to combine the various emission line components:
For type 1 AGN - combine broad+narrow(S2) + FeII 
For high-ionization type 2s, use only narrow(S2)
For LINERs use only LINER column

Normalization for all emission line components is done by fixing L(Hb)/L(5100) in the following way 
Broad emission lines:          L(Hb)/L(5100)=0.02 and then L(FeII)/L(Hb)=A(FeII) where 2<A(FeII)<10 (free parameter)
Narrow high ionization lines:  L(Hb)/L(5100)=0.002
Liners:                        L(Hb)/L(5100)=0.002

Further refinements (I am not sure that this is necessary)
Scale the narrow component with intrinsic luminosity e.g.
For L(5100)>10^45 erg/sec  L(Hb)/L(5100)=0.0015
    L(5100)<10^44 erg/sec  L(Hb)/L(5100)=0.003
 
3. NIR-MIR continuum:
The Mor and Netzer 2012 template is provided in
"mor_netzer_2012_AGN_SED_combined_with_100K_grey_body"
The first few points with lambda<1mic do not belong to this component and can be used for normalization.
The normalization, relative to the disk continuum is done by
L(12 mic)=2.5xL(5100)xf(cover) where L(12mic)=lambda_L_lambda(12mic)
and f(cover) is the covering factor. For low and intermediate luminosity sources, f(cover)=0.5.
For high luminosity sources it is smaller.
I suggest to treat it as a free parameter in the range 0.1<f(cover)<0.7




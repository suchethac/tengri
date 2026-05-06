#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:50:31 2022

@author: Scott Hagen

Relativistic versions of AGNSED (RELAGN) and QSOSED (RELQSO) (Kubota & Done 2018).
The relativistic corrections are calculated using the convolution code
KYCONV (Dovciak, Karas & Yaqoob 2004)

For the Comptonised region of the disc we use the pyNTHCOMP routine, adapted
from the XSPEC model NTHCOMP (Zdziarski, Johnson & Magdziarz, 1996; Zycki,
Done & Smith, 1999) by Thomas et al. 2016
"""

import numpy as np
from scipy.integrate import quad
import xspec
import warnings

import astropy.constants as const
import astropy.units as u

from pyNTHCOMP import donthcomp

#Stop all the run-time warnings (we know why they happen - doesn't affect the output!)
warnings.filterwarnings('ignore') 



#Constants
G = (const.G * const.M_sun).value #Grav const in units m^3 s^-1 Msol^-1
c = const.c.value #speed of light, m/s
h = const.h.value #planck constant, Js
sigma_T = const.sigma_T.value #Thompson cross section, m^2
sigma_sb = const.sigma_sb.value #Stefan-Boltzmann const, W m^-2 K^-4
m_p = const.m_p.value #Proton mass, kg
k_B = const.k_B.value #Boltzmann const, J K^-1



#Relevant funcitons

def do_black_body(T, nu):
    pre_fac = (2 * h * nu**3)/(c**2)
    exp_fac = np.exp((h * nu)/(k_B * T)) - 1
    Bnu = pre_fac / exp_fac

    return np.pi * Bnu



def interp_spec(Ei, Emod, flxs):
    """
    Linearly interpolates spectra between energy grids
    Needed for tunring into xspec model...

    Parameters
    ----------
    Ei : float
        Point on new energy grid.
    Emod : 1D-array
        Old energy grid
    flxs : 1D-array
        Model spectrum

    Returns
    -------
    fi : float
        Flux at new grid point.

    """
    idx_1 = np.abs(Ei - Emod).argmin()
    
    
    if Ei - Emod[idx_1] > 0:
        if Emod[idx_1] != Emod[-1]: #ensuring we dont fall off array
            E1 = Emod[idx_1]
            E2 = Emod[idx_1 + 1]
            f1 = flxs[idx_1]
            f2 = flxs[idx_1 + 1]
        
        else:
            E1 = Emod[idx_1 - 1]
            E2 = Emod[idx_1]
            f1 = flxs[idx_1 -1]
            f2 = flxs[idx_1]
        
        df_dE = (f2 - f1)/(E2 - E1)
        fi = df_dE * (Ei - E1) + f1
    
    elif Ei - Emod[idx_1] < 0:
        if Emod[idx_1] != Emod[0]:
            E1 = Emod[idx_1 - 1]
            E2 = Emod[idx_1]
            f1 = flxs[idx_1 -1]
            f2 = flxs[idx_1]
        
        else:
            E1 = Emod[idx_1]
            E2 = Emod[idx_1 + 1]
            f1 = flxs[idx_1]
            f2 = flxs[idx_1 + 1]
            
        df_dE = (f2 - f1)/(E2 - E1)
        fi = df_dE * (Ei - E1) + f1
    
    else:
        fi = flxs[idx_1]
        
    return fi



"""
The relagnsed class
"""

class relagn:
    
    """
    relagn - relativistic SED model for AGN
    Assumes a radially stratified flow consisting of:
        - An outer standard accretion disc
        - A warm Comptonisation region (where the disc is having a bad day)
        - A hot Comptonisation region (i.e inner X-ray Corona)
    
    For more details on the model - see Hagen & Done (in prep)
    
    
     
    Attributes 
    ----------    
    risco : float
        Innremost stable circular orbit - units : Dimensionless (Rg)
    
    r_sg : float
        Self-Gravity radius of the disc, following Laor & Netzer 1989
        Units : Dimensionless (Rg)
    
    eta : float
        Accretion efficiency. Used to convert from mass accretion rate to 
        luminosity. i.e L = eta * Mdot * c^2
        Units : dimensionless
    
    Egrid : array
        Energy grid used for calculations (in frame of black hole) - units : keV
    
    nu_grid : array
        Frequency grid corresponding to Egrid - units : Hz
    
    wave_grid : array
        Wavelength grid corresponding to Egrid - units : Angstrom
    
    E_obs : array
        Energy grid converted to observer frame (i.e redshift corrected) - units : keV
    
    nu_obs : array
        Frequency grid converted to observer frame - units : Hz
    
    wave_obs : array
        Wavelength grid converted to observer frame - units : Angstrom
    
    ###########################################################################
    There are other attributes, however it is recomended that you 
    extract these using the built in methods in order to deal with unit choices 
    correctly. The remaining ones after this are only necessary for the internal 
    calculations, so no need to worry about them!
    ###########################################################################
    
    
    Important Methods
    -----------------
    get_totSED(rel=True)
        Extracts the total SED (disc + warm + hot components) in whatever
        units are set
    
    get_DiscComponent(rel=True)
        Extracts disc component from SED (after checking if it exists in the
        current model geometry) in whatever units are set
    
    get_WarmComponent(rel=True)
        Extracts warm Compton component from SED (after checking if it exists
        in the current model geometry) in whatever units are set
    
    get_HotComponent(rel=True)
        Extracts hot Compton component from SED (after checking if it exists
        in the current model geometry) in whatever units are set
    
    set_units(new_unit='cgs')
        Sets the system units to use when extracting spectra / system
        properties
    
    set_flux()
        Sets a flag s.t all spectra are given in terms of flux rather than
        luminosity
    
    set_lum()
        Sets a flag s.t all spectra are given in luminosity (this is the default)
    
    get_Ledd()
        Gives Eddington luminosity in whatever units are set
        (Note: in frame of black hole!!)
    
    get_Rg()
        Gives scale of gravitaional radius (Rg = GM/c^2) in whatever units 
        are set
    
    get_Mdot()
        Gives PHYSICAL mass accretion rate, in either g/s or kg/s
        (depending on what units are set)
    
    
    ###########################################################################
    
    """
    
    
    Emin = 1e-4
    Emax = 1e4
    numE = 2000
    mu = 0.55 #mean particle mass - fixed at solar abundances
    A = 0.3 #Disc albedo = fixed at 0.3 for now
    
    dr_dex = 50 #grid spacing - N points per decade
    
    default_units = 'cgs'
    units = 'cgs'
    as_flux = False #This flags whether to return luminosity or flux
    
    def __init__(self,
                 M=1e8,
                 dist=100,
                 log_mdot=-1,
                 a=0,
                 cos_inc=0.5,
                 kTe_hot=100,
                 kTe_warm=0.2,
                 gamma_hot=1.7,
                 gamma_warm=2.7,
                 r_hot=10,
                 r_warm=20,
                 log_rout=-1,
                 fcol=1,
                 h_max=10,
                 z=0):
        """
        Parameters
        ----------
        M : float
            Black hole mass - units : Msol      
        dist : float
            Co-Moving Distance - units : Mpc
        log_mdot : float
            log mass accretion rate - units : Eddington
        a : float
            Dimensionless Black Hole spin - units : Dimensionless
        cos_inc : float
            cos inclination angle
        kTe_hot : float
            Electron temp for hot Compton region - units : keV
        kTe_warm : float
            Electron temp for warm Compton region - units : keV
        gamma_hot : float
            Spectral index for hot Compton region
        gamma_warm : float
            Spectral index for warm Compton region
        r_hot : float
            Outer radius of hot Compton region - units : Rg
        r_warm : float
            Outer radius of warm Compton region - units : Rg
        log_rout : float
            log of outer disc radius - units : Rg
        fcol : float
            Colour temperature correction as described in Done et al. (2012)
            If -ve then follows equation 1 and 2 in Done et al. (2012).
            If +ve then assumes this to be constant correction over entire disc region
        h_max : float
            Scale height of hot Compton region - units : Rg
        z : float
            Redshift
        """
        
        #Read params
        self.M = M
        self.D, self.d = dist, (dist * u.Mpc).to(u.cm).value
        self.mdot = 10**(log_mdot)
        self.a = np.float64(a)
        self.inc = np.arccos(cos_inc)
        self.kTe_h = kTe_hot
        self.kTe_w = kTe_warm
        self.gamma_h = gamma_hot
        self.gamma_w = gamma_warm
        self.r_h = r_hot
        self.r_w = r_warm
        self.r_out = 10**(log_rout)
        self.fcol = fcol
        self.hmax = h_max 
        self.z = z
        
        self.cosinc = cos_inc
        
        
        #Performing checks
        self._check_spin()
        self._check_inc()
        
        #Calculating disc params 
        self._calc_risco()
        self._calc_r_selfGravity()
        self._calc_Ledd()
        self._calc_efficiency()

        
        if log_rout < 0:
            self.r_out = self.r_sg #setting to self gravity if log_rout < 0
        
        if r_warm == -1:
            self.r_w = self.risco
        
        if r_hot == -1:
            self.r_h = self.risco
        
        self._check_rw()
        self._check_risco()
        self._check_hmax()
        
        #physical conversion factors
        self.Mdot_edd = self.L_edd/(self.eta * c**2)
        self.Rg = (G * self.M)/(c**2)
        
        
        #Energy/frequency grid
        self.Ebins = np.geomspace(self.Emin, self.Emax, self.numE)
        self.dEs = self.Ebins[1:] - self.Ebins[:-1]

        self.Egrid = self.Ebins[:-1] + 0.5 * self.dEs
        self.nu_grid = (self.Egrid * u.keV).to(u.Hz,
                                equivalencies=u.spectral()).value
        self.wave_grid = (self.Egrid * u.keV).to(u.AA,
                                equivalencies=u.spectral()).value
        
        self.nu_obs = self.nu_grid/(1 + self.z) #Observers frame
        self.E_obs = self.Egrid/(1 + self.z)
        self.wave_obs = self.wave_grid * (1+self.z)
        
        #Creating radal grid over disc and warm compton regions
        #using spacing of dr_dex
        self.dlog_r = 1/self.dr_dex
        self.logr_ad_bins = self._make_rbins(np.log10(self.r_w), np.log10(self.r_out))
        self.logr_wc_bins = self._make_rbins(np.log10(self.r_h), np.log10(self.r_w))
        self.logr_hc_bins = self._make_rbins(np.log10(self.risco), np.log10(self.r_h))
        
        #calculating coronal luminosity IF it exists
        if self.r_h != self.risco:
            self._hot_specShape()

            
        

    
    
    
    ###########################################################################
    #---- Performing checks on certain parameters. 
    #     To ensure that we are within both physical limits 
    #     (i.e -0.998 <= a <= 0.998) AND that we dont wander off acceptable 
    #     values of kyconv (i.e 3 <= inc <= 85 deg)
    ###########################################################################
    
    def _check_spin(self):
        if self.a >= -0.998 and self.a <= 0.998:
            pass
        else:
            raise ValueError('Spin ' + str(self.a) + ' not physical! \n'
                             'Must be within: -0.998 <= a_star <= 0.998')
    
    
    def _check_inc(self):
        if self.cosinc <= 1 and self.cosinc >= 0.09:
            pass
        else:
            raise ValueError('Inclination out of bounds - will not work with kyconv! \n'
                             'Require: 0.09 <= cos(inc) <= 0.98 \n'
                             'Translates to: 11.5 <= inc <= 85 deg')
    
    def _check_rw(self):
        if self.r_w >= self.r_h:
            pass
        else:
            print('WARNING r_warm < r_hot ---- Setting r_warm = r_hot')
            self.r_w = self.r_h
        
        if self.r_w <= self.r_out:
            pass
        else:
            print('WARNING r_warm > r_out ----- Setting r_warm = r_out')
            self.r_w = self.r_out
    
    
    def _check_risco(self):
        if self.r_h >= self.risco:
            pass
        else:
            print('WARNING r_hot < r_isco ----- Setting r_hot = r_isco')
            self.r_h = self.risco
        
        if self.r_w >= self.risco:
            pass
        else:
            print('WARNING! r_warm < r_isco ----- Settin r_warm = r_isco')
            self.r_w = self.risco
    
    
    def _check_hmax(self):
        if self.hmax <= self.r_h:
            pass
        else:
            print('WARNING! hmax > r_h ------- Setting hmax = r_h')
            self.hmax = self.r_h
    

        
    
    
    ###########################################################################
    #---- Section for dealing with units. 
    #     Essentially just methods to change the unit flag, which sets the 
    #     spectral output units - and then methods to convert calculated spectrum 
    #     to desired units.
    #     Also includes method for changing the energy grid to something other
    #     than the default
    ###########################################################################
    
    
    def set_units(self, new_unit='cgs'):
        """
        Re-sets default units. ONLY affects attributes extracted through the
        getter methods
        
        Note, the only difference between setting cgs vs counts is in spectra
        Any integrated luminosities (e.g. Ledd) will be given in 
        erg/s IF counts is  set

        Parameters
        ----------
        new_unit : {'cgs','cgs_wave', 'SI', 'counts'}, optional
            The default unit to use. The default is 'cgs'.
            NOTE, the main cgs_wave will give spectra in erg/s/Angstrom,
            while cgs gives in erg/s/Hz

        """
        #Checking valid units
        unit_lst = ['cgs', 'cgs_wave', 'SI', 'SI_wave', 'counts']
        if new_unit not in unit_lst:
            print('Invalid Unit!!!')
            print(f'Valid options are: {unit_lst}')
            print('Setting as default: cgs')
            new_unit = 'cgs'
            
        self.units = new_unit
    
    def set_flux(self):
        """
        Sets default output as a flux
        This ONLY affects spectra! Things like Eddington luminosity, or
        Bolometric luminosity remain as Luminosity!!
        
        Note: This will also take the redshift into account!!
        
        /cm^2 IF cgs or counts, /m^2 if SI

        """
        self.as_flux = True
        
        
    def set_lum(self):
        """
        Sets defualt output as luminosity (only necessary IF previously set
        as flux)

        """
        self.as_flux = False
    
    
    def _to_newUnit(self, L, as_spec=True):
        """
        Sets input luminosity/spectrum to new output units

        Parameters
        ----------
        L : float OR array
            Input lum/spectrum.

        Returns
        -------
        Lnew : float OR array
            In new units.
        unit : str
            new unit (in case the currently set unit is not desired)
        as_spec : bool
            If True, then input should be W/Hz
            If false, then input should be W

        """
        #If spectral density
        if as_spec:
            if self.units == 'cgs':
                Lnew = L*1e7
                
            elif self.units == 'counts':
                Lnew = (L*u.W/u.Hz).to(u.keV/u.s/u.keV,
                                             equivalencies=u.spectral()).value
                
                if np.ndim(L) == 1:
                    Lnew /= self.Egrid
                else:
                    Lnew /= self.Egrid[:, np.newaxis]
            
            elif self.units == 'cgs_wave':
                Lnew = (L*u.W/u.Hz).to(u.erg/u.s/u.AA,
                                equivalencies=u.spectral_density(self.nu_grid*u.Hz)).value
            
            elif self.units == 'SI_wave':
                Lnew = (L*u.W/u.Hz).to(u.W/u.AA,
                                equivalencies=u.spectral_density(self.nu_grid*u.Hz)).value 
            
            else:
                Lnew = L
        
        #If just a luminosity
        else:
            if self.units == 'cgs' or self.units == 'counts' or self.units == 'cgs_wave':
                Lnew = L*1e7
            else:
                Lnew = L
        
        return Lnew
    
    
    def _to_flux(self, L):
        """
        Converts to a flux - takes redshift into accounts    

        Parameters
        ----------
        L : float OR array
            Luminosity to be converted.

        Returns
        -------
        f : float OR array
            Flux seen by observer

        """
        
        if self.units == 'cgs' or self.units == 'counts' or self.units == 'cgs_wave':
            d = self.d #distance in cm
        else:
            d = self.d/100 #distance in m
        
        f = L/(4*np.pi*d**2 * (1+self.z))
        return f
    
        
    
    
    ###########################################################################
    #---- Calculating disc properties
    #     i.e r_isco, L_edd, NT temp, etc
    ###########################################################################
    
    def _calc_Ledd(self):
        """
        Caclulate eddington Luminosity

        """
        Ledd = 1.39e31 * self.M 
        self.L_edd = Ledd
    
    
    def _calc_risco(self):
        """
        Calculating innermost stable circular orbit for a spinning
        black hole. Follows Page and Thorne (1974). Note, in the litterature
        this is also reffered to as r_ms, for marginally stable orbit.
        We will stick to risco throughout!

        """
        Z1 = 1 + (1 - self.a**2)**(1/3) * (
            (1 + self.a)**(1/3) + (1 - self.a)**(1/3))
        Z2 = np.sqrt(3 * self.a**2 + Z1**2)

        self.risco = 3 + Z2 - np.sign(self.a) * np.sqrt(
            (3 - Z1) * (3 + Z1 + 2*Z2))
    
    
    def _calc_r_selfGravity(self):
        """
        Calcultes the self gravity radius according to Laor & Netzer 1989
        
        NOTE: Assuming that \alpha=0.1 

        """
        alpha = 0.1 #assuming turbulence NOT comparable to sound speed
        #See Laor & Netzer 1989 for more details on constraining this parameter
        m9 = self.M/1e9
        self.r_sg = 2150 * m9**(-2/9) * self.mdot**(4/9) * alpha**(2/9)
    
    
    def _calc_efficiency(self):
        """
        Calculates the accretion efficiency eta, s.t L_bol = eta Mdot c^2
        Using the GR case, where eta = 1 - sqrt(1 - 2/(3 r_isco)) 
            Taken from: The Physcis and Evolution of Active Galactic Nuceli,
            H. Netzer, 2013, p.38

        """
        
        self.eta = 1 - np.sqrt(1 - 2/(3*self.risco))
    
    
    def _calc_NTparams(self, r):
        """
        Calculates the Novikov-Thorne relativistic factors.
        see Active Galactic Nuclei, J. H. Krolik, p.151-154
        and Page & Thorne (1974)
        
        Parameters
        ----------
        r : float OR array
            Disc radius (as measured from black hole)
            Units : Dimensionless (Rg)

        """
        y = np.sqrt(r)
        y_isc = np.sqrt(self.risco)
        y1 = 2 * np.cos((1/3) * np.arccos(self.a) - (np.pi/3))
        y2 = 2 * np.cos((1/3) * np.arccos(self.a) + (np.pi/3))
        y3 = -2 * np.cos((1/3) * np.arccos(self.a))

        
        B = 1 - (3/r) + ((2 * self.a)/(r**(3/2)))
        
        C1 = 1 - (y_isc/y) - ((3 * self.a)/(2 * y)) * np.log(y/y_isc)
        
        C2 = ((3 * (y1 - self.a)**2)/(y*y1 * (y1 - y2) * (y1 - y3))) * np.log(
            (y - y1)/(y_isc - y1))
        C2 += ((3 * (y2 - self.a)**2)/(y*y2 * (y2 - y1) * (y2 - y3))) * np.log(
            (y - y2)/(y_isc - y2))
        C2 += ((3 * (y3 - self.a)**2)/(y*y3 * (y3 - y1) * (y3 - y2))) * np.log(
            (y - y3)/(y_isc - y3))
        
        C = C1 - C2
        
        return C/B
        
        
    
    
    def calc_Tnt(self, r):
        """
        Calculates Novikov-Thorne disc temperature^4 at radius r. 
        
        Parameters
        ----------
        r : float OR array
            Disc radius (as measured from black hole)
            Units : Dimensionless (Rg)
        
        Returns
        -------
        T4 : float OR array
            Novikov-Thorne disc temperature at r (to the power 4)
            Units : K^4 (Kelvin to power 4)
        
        """
        Rt = self._calc_NTparams(r)
        const_fac = (3 * G * self.M * self.mdot * self.Mdot_edd)/(
            8 * np.pi * sigma_sb * (r * self.Rg)**3)
        
        T4 = const_fac * Rt

        return T4
    
        
    
    def calc_fcol(self, Tm):
        """
        Calculates colour temperature correction following Eqn. (1) and 
        Eqn. (2) in Done et al. (2012)

        Parameters
        ----------
        Tm : float
            Max temperature at annulus (ie T(r)) - units : K.

        Returns
        -------
        fcol_d : float
            colour temperature correction at temperature T

        """
        if Tm > 1e5:
            #For this region follows Eqn. (1)
            Tm_j = k_B * Tm
            Tm_keV = (Tm_j * u.J).to(u.keV).value #convert to units consitant with equation
            
            fcol_d = (72/Tm_keV)**(1/9)
        
        elif Tm < 1e5 and Tm > 3e4:
            #For this region follows Eqn. (2)
            fcol_d = (Tm/(3e4))**(0.82)
        
        else:
            fcol_d = 1
        
        return fcol_d
    
    
    
    
    
    
    ###########################################################################
    #---- Creating/Updating binning
    ###########################################################################


    def _make_rbins(self, logr_in, logr_out, dlog_r=None):
        """
        Creates an array of radial bin edges, with spacing defined by dr_dex
        Calculates the bin edges from r_out and down to r_in. IF the bin
        between r_in and r_in+dr is less than dlog_r defined by dr_dex, then
        we simply create a slightly wider bin at this point to accomodate
        for the difference

        Parameters
        ----------
        logr_in : float
            Inner radius of model section - units : Rg.
        logr_out : float
            Outer radius of model section - units : Rg.

        Returns
        -------
        logr_bins : 1D-array
            Radial bin edges for section - units : Rg.

        """
        if dlog_r is not None:
            dlr = dlog_r
        else:
            dlr = self.dlog_r
        
        
        i = logr_out
        logr_bins = np.array([np.float64(logr_out)]) 
        while i > logr_in:
            r_next_edge = i - dlr
            logr_bins = np.insert(logr_bins, 0, r_next_edge)
            i = r_next_edge

       
        if logr_bins[0] != logr_in:
            if logr_bins[0] < logr_in:
                if len(logr_bins) > 1:
                    logr_bins = np.delete(logr_bins, 0)
                    logr_bins[0] = logr_in
                else:
                    logr_bins[0] = logr_in
            else:
                logr_bins[0] = logr_in
        
        return logr_bins
        
    
    
    def new_ear(self, new_es):
        """
        Defines new energy grid if necessary

        Parameters
        ----------
        ear : 1D-array
            New energy grid - units : keV.

        """
        
        self.Ebins = new_es
        self.dEs = self.Ebins[1:] - self.Ebins[:-1]
        
        self.Egrid = self.Ebins[:-1] + 0.5 * self.dEs
        self.nu_grid = (self.Egrid * u.keV).to(u.Hz,
                                equivalencies=u.spectral()).value
        self.nu_obs = self.nu_grid/(1 + self.z) #Observers frame
        self.E_obs = self.Egrid/(1 + self.z)
        
        self.Emin = min(self.Egrid)
        self.Emax = max(self.Egrid)
        self.numE = len(self.Egrid)
        
        if self.r_h != self.risco:
            self._hot_specShape()
    
    
    
    def change_rBins(self, new_drdex):
        """
        JUST FOR TESTING PURPOSES!!!! Allows changing of radial bin-width
        makes testing easier...

        Parameters
        ----------
        new_drdex : float
            New number of steps per decade.

        """
        self.dr_dex = new_drdex
        self.dlog_r = 1/self.dr_dex
        
        self.logr_ad_bins = self._make_rbins(np.log10(self.r_w), np.log10(self.r_out))
        self.logr_wc_bins = self._make_rbins(np.log10(self.r_h), np.log10(self.r_w))
        self.logr_hc_bins = self._make_rbins(np.log10(self.risco), np.log10(self.r_h))
        
        #calculating coronal spec shape
        if self.r_h != self.risco:
            self._hot_specShape()
    
    
    
    
    
    ###########################################################################
    #---- Annular emission from disc and warm Compton region
    ###########################################################################
    
    def disc_annuli(self, r, dr):
        """
        Calculates disc spectrum for annulus at position r with width dr. Note
        that r is taken to be the center of the bin!

        Parameters
        ----------
        r : float
            Inner radius of annulus - units : Rg.
        dr : float
            Width of annulus - units : Rg.
        
        Returns
        -------
        Lnu_ann : 1D-array
            Disc black-body at annulus - units : W/Hz

        """
        T4_ann = self.calc_Tnt(r)
        Tann = T4_ann**(1/4)
        if self.fcol < 0:
            fcol_r = self.calc_fcol(Tann)
        else:
            fcol_r = self.fcol
        
        Tann *= fcol_r

        #Extend nu/energy grid to avoid inaccuracies at large radii
        #Only do this if internal Emin > 1e-4 keV
        if self.Emin < 1e-4:
            nu_ext = np.insert(self.nu_grid, 0, np.geomspace(1e12, min(self.nu_grid)-100, 50))
            nu_use = nu_ext
        
        else:
            nu_use = self.nu_grid
        
        B = do_black_body(Tann, nu_use)
        
        #Annulus luminosity - normalised as black body
        #NOTE! kyconv integrates over the annulus within the code, 
        #hence why we are NOT multiplying with pi r dr - as this would 
        #effectively overestimate the actual normalisation post convolution.
        #The final result is still the same as this - it's just done somewhere else...
        #kyconv also deal with the inclination - so no cos(inc) correction factor needs to be applied
        norm = sigma_sb * (Tann/fcol_r)**4 * self.Rg**2

        radiance = np.trapezoid(B, nu_use)
        if radiance == 0:
            Lnu_ann = np.zeros(len(nu_use))
        else:
            Lnu_ann = norm * (B/radiance)
        
        #Re-casting onto original grid in order to be consistent with kyconv
        #i.e, just slicing away entries below Emin/nu_grid_min
        if self.Emin < 1e-4:
            Lnu_ann = Lnu_ann[50:]
            
        return Lnu_ann
        
    
    
    def warmComp_annuli(self, r, dr):
        """
        Calculates comptonised spectrum for annulus at r with width dr. Note,
        r taken to be in center of bin!
        Uses pyNTHCOMP

        Parameters
        ----------
        r : float
            Inner radius of annulus - units : Rg.
        dr : float
            Width of annulus - units : Rg.
        
        Returns
        -------
        Lnu_ann : 1D-array
            Warm Comptonised spectrum at annulus - units : W/Hz
        """
        
        T4_ann = self.calc_Tnt(r)
        Tann = T4_ann**(1/4)
        
        kTann = k_B * Tann
        #kTann = (kTann * u.J).to(u.keV).value #converting T to keV for nthcomp
        kTann = Tann/1.16048e7
        
        ph_nth = donthcomp(self.Egrid, [self.gamma_w, self.kTe_w,
                                        kTann, 0, 0])
        
        ph_nth = (ph_nth * u.W/u.keV).to(u.W/u.Hz, 
                                            equivalencies=u.spectral()).value
        
        norm = sigma_sb * (Tann**4) * self.Rg**2
        radiance = np.trapezoid(ph_nth, self.nu_grid)
        if radiance == 0:
            Lnu_ann = np.zeros(len(self.nu_grid))
        else:
            Lnu_ann = norm * (ph_nth/radiance)# * self.cosinc/0.5
        
        return Lnu_ann
    
    
    
    
    
    ###########################################################################
    #---- Total emission from disc/warm Compton region
    ###########################################################################
    
    
    def do_relDiscSpec(self):
        """
        Calculates contribution from entire disc section - for relativistic 
        case
        
        Returns
        -------
        Lnu_disc_rel : array
            Total RELATIVISTIC spectrum for standard disc region
            Units : W/Hz
        
        """    
        for i in range(len(self.logr_ad_bins) - 1):
            dr_bin = 10**self.logr_ad_bins[i+1] - 10**self.logr_ad_bins[i] #width in lin space
            rmid = 10**(self.logr_ad_bins[i] + self.dlog_r/2) #geometric center of bin
            Lnu_ann = self.disc_annuli(rmid, dr_bin)

        
            #Multiply by 4 because kyconv only does pi r dr
            #We want 4 pi r dr
            fluxs = (Lnu_ann*4 * u.W/u.Hz).to(u.keV/u.s/u.keV,
                                              equivalencies=u.spectral()).value
                
            fluxs = fluxs/(4*np.pi * self.d**2)
            phs = list((self.dEs*fluxs)/self.Egrid)

                        
            
            #Convolving annulus with kyconv
            #kyconv params are:
                #a, spin
                #inc - deg
                #r_in - Rg
                #ms - set to 0 for integration to start at r_in
                #r_out - Rg
                #alph - Emissivity power law - set to 0 as this already hardwired in own code
                #beta - Same as alpha
                #rb - break radius for emissivity - dummy parameter in this scenario
                #z - redshift, assume 0 for now
                #ntable - set to 0 for isotropic emission
                #nE - Energy resolution - set to same as rest of model
                #norm - set to -1 for NO renormalisation - maintain physical units
            if self.logr_ad_bins[i] <= 3 and self.logr_ad_bins[i+1] <= 3:
                r_in = 10**self.logr_ad_bins[i]
                r_out = 10**self.logr_ad_bins[i+1]
                r_br = rmid
                
                kyparams = [self.a, np.rad2deg(self.inc), r_in, 0, r_out, 0, 0,
                         r_br, 0, 0, self.numE, -1]
                
                Ein = list(self.Ebins)
                xspec.callModelFunction('kyconv', Ein, kyparams, phs)
                
                phs_r = np.array(phs)/self.dEs
                
                L_kev = phs_r * self.Egrid * 4 * np.pi * self.d**2 
                Lnu_r = (L_kev * u.keV/u.s/u.keV).to(u.W/u.Hz, equivalencies=u.spectral()).value
                
                
            else:
                #If out of bounds for tables do non-relativistic
                #This is fine - as beyon 1000Rg Gr effects tiny!
                #Since no kyconv - we now apply the normalisation here instead
                Lnu_r = Lnu_ann * 2*np.pi*2 * rmid * dr_bin   * self.cosinc/0.5
            

            if i == 0:
                Lnu_all = Lnu_r
            else:
                Lnu_all = np.column_stack((Lnu_all, Lnu_r))
                
        if np.shape(Lnu_all) != np.shape(self.Egrid):
            Lnu_tot = np.sum(Lnu_all, axis=-1)
        else:
            Lnu_tot = Lnu_all
        
        self.Lnu_disc_rel = Lnu_tot 
        return self.Lnu_disc_rel
    
    
    def do_nonrelDiscSpec(self):
        """
        Calculates contribution from entire disc section - for non-relativisitc
        case. Usefull for comparison...
        
        Returns
        -------
        Lnu_disc_norel : array
            Total NON-RELATIVISTIC spectum from standard disc region
            Units ; W/Hz

        """
        for i in range(len(self.logr_ad_bins) - 1):
            dr_bin = 10**self.logr_ad_bins[i+1] - 10**self.logr_ad_bins[i]
            rmid = 10**(self.logr_ad_bins[i] + self.dlog_r/2)
            
            Lnu_r = self.disc_annuli(rmid, dr_bin) * 2*np.pi * 2 * rmid * dr_bin
            
            if i == 0:
                Lnu_all = Lnu_r
            else:
                Lnu_all = np.column_stack((Lnu_all, Lnu_r))
                
        if np.shape(Lnu_all) != np.shape(self.Egrid):
            Lnu_tot = np.sum(Lnu_all, axis=-1)
        else:
            Lnu_tot = Lnu_all
        
        self.Lnu_disc_norel = Lnu_tot  * self.cosinc/0.5
        return self.Lnu_disc_norel
            
    
    
    def do_relWarmCompSpec(self):
        """
        Calculates contribution from entire warm Compton region - for 
        relativistic case
        
        Returns
        -------
        Lnu_warm_rel : array
            Total RELATIVISTIC spectrum from hot Compton region
            Units : W/Hz
        
        """
        for i in range(len(self.logr_wc_bins) - 1):
            dr_bin = 10**self.logr_wc_bins[i+1] - 10**self.logr_wc_bins[i]
            rmid = 10**(self.logr_wc_bins[i] + self.dlog_r/2)
            
            Lnu_ann = self.warmComp_annuli(rmid, dr_bin)
            
            fluxs = (Lnu_ann*4 * u.W/u.Hz).to(u.keV/u.s/u.keV,
                                              equivalencies=u.spectral()).value
            
            fluxs = fluxs/(4*np.pi * self.d**2)
            
            phs = list((fluxs*self.dEs)/self.Egrid)

        
            #Convolving annulus with kyconv
            #See do_relDiscSpec for explanation of kyparams
            if self.logr_wc_bins[i] <= 3 and self.logr_wc_bins[i+1] <= 3:
                r_in = 10**self.logr_wc_bins[i]
                r_out = 10**self.logr_wc_bins[i+1]
                r_br = rmid
                kyparams = [self.a, np.rad2deg(self.inc), r_in, 0, r_out, 0, 0,
                            r_br, 0, 0, self.numE, -1]

                xspec.callModelFunction('kyconv', list(self.Ebins), 
                                                kyparams, phs)
                phs_r = np.array(phs)/self.dEs
                
                
                L_kev = phs_r * self.Egrid * 4 * np.pi * self.d**2
                Lnu_r = (L_kev * u.keV/u.s/u.keV).to(u.W/u.Hz, equivalencies=u.spectral()).value
            
            else:
                #If out of bounds for tables do non-relativistic
                #This is fine - as beyon 1000Rg Gr effects tiny!
                #Since no kyconv - we now apply the normalisation here instead
                Lnu_r = Lnu_ann * 2*np.pi * 2 * rmid * dr_bin   * self.cosinc/0.5
                

            if i == 0:
                Lnu_all = Lnu_r
            else:
                Lnu_all = np.column_stack((Lnu_all, Lnu_r))
        
        if np.shape(Lnu_all) != np.shape(self.Egrid):
            Lnu_tot = np.sum(Lnu_all, axis=-1)
        else:
            Lnu_tot = Lnu_all
            
        self.Lnu_warm_rel = Lnu_tot
        return self.Lnu_warm_rel
    
    
    def do_nonrelWarmCompSpec(self):
        """
        Calculates contribution from entire warm Compton region - for 
        non-relativistic case
        
        Returns
        -------
        Lnu_warm_norel : array
            Total NON-RELATIVISTIC spectrum from warm Compton region
            Units : W/Hz
        
        """
        for i in range(len(self.logr_wc_bins) - 1):
            dr_bin = 10**self.logr_wc_bins[i+1] - 10**self.logr_wc_bins[i]
            rmid = 10**(self.logr_wc_bins[i] + self.dlog_r/2)
            
            Lnu_r = self.warmComp_annuli(rmid, dr_bin) * 4*np.pi*rmid*dr_bin
            
            if i == 0:
                Lnu_all = Lnu_r
            else:
                Lnu_all = np.column_stack((Lnu_all, Lnu_r))
        
        if np.shape(Lnu_all) != np.shape(self.Egrid):
            Lnu_tot = np.sum(Lnu_all, axis=-1)
        else:
            Lnu_tot = Lnu_all
            
        self.Lnu_warm_norel = Lnu_tot * self.cosinc/0.5
        return self.Lnu_warm_norel
        
        
    
    ###########################################################################
    #---- Hot Comptonisatin region (corona)
    ###########################################################################
    
    def seed_tempHot(self):
        """
        Calculated seed photon temperature for the hot compton region.
        Follows xspec model agnsed, from Kubota & Done (2018)
        
        Returns
        -------
        kT_seed : float
            Seed photon temperature for hot compton - units : keV

        """
        T4_edge = self.calc_Tnt(self.r_h) #inner disc T in K
        Tedge = T4_edge**(1/4)
        
        kT_edge = k_B * Tedge #units J
        kT_edge = (kT_edge * u.J).to(u.keV).value
        if self.r_w != self.r_h:
            #If there is a warm compton region then seed photons mostly from
            #here. Will then need to include Compton y-param
            ysb = (self.gamma_w * (4/9))**(-4.5)
            kT_seed = np.exp(ysb) * kT_edge
            #kT_seed = kT_edge
            
        else:
            #If only disc down to r_hot seed temp will be same as inner disc temp
            kT_seed = kT_edge
            if self.fcol < 0:
                fcol_r = self.calc_fcol(Tedge)
            else:
                fcol_r = self.fcol
            
            kT_seed *= fcol_r
        
        return kT_seed
    
    
    def _hot_specShape(self):
        """
        Calls pyNTHCOMP for the hot compton region, as here spectral shape
        will remain the same in non-relativistic case accross entire corona
        (at least in this slightly simplified model...)
        
        Calling here saves repeatedly calling nthcomp when calculating 
        relativsitc spec.

        """
        
        kTseed = self.seed_tempHot()
        if kTseed < self.kTe_h:
            #Ensures we get energetics right!
            Fnu_hot = donthcomp(self.Egrid, [self.gamma_h, self.kTe_h, kTseed,
                                             0, 0])
            
            Fnu_hot = (Fnu_hot * u.W/u.keV).to(u.W/u.Hz, 
                                            equivalencies=u.spectral()).value
        
        else:
            Fnu_hot = np.zeros(len(self.Egrid))
        
        self.Fnu_seed_hot = Fnu_hot
        
    
    
    def Lseed_hotCorona(self):
        """
        Calculates luminsoty of seed photons emitted at radius r, intercepted
        by corona

        Returns
        -------
        Lseed_tot : float
            Total seed photon luminosity seen by corona - units : W

        """
        
        logr_tot_bins = self._make_rbins(np.log10(self.r_h), np.log10(self.r_out))
        Lseed_tot = 0
        hc = min(self.r_h, self.hmax) #corona height can't be > r_h!!
        for i in range(len(logr_tot_bins) - 1):
            dr = 10**logr_tot_bins[i+1] - 10**logr_tot_bins[i]
            rmid = 10**(logr_tot_bins[i] + self.dlog_r/2)
            
            
            
            if hc <= rmid:
                theta_0 = np.arcsin(hc/rmid)
                cov_frac = theta_0 - 0.5 * np.sin(2*theta_0)
            else:
                cov_frac = 0
            
    
            T4_ann = self.calc_Tnt(rmid)
            
            Fr = sigma_sb * T4_ann
            Lr = 2 * 2*np.pi*rmid*dr * Fr * cov_frac/np.pi * self.Rg**2


            Lseed_tot += Lr
            
        
        return Lseed_tot
        
    
    
    def hotCorona_lumin(self):
        """
        Calculates the coronal luminosity - used as normalisaton for the 
        hot compton spectral component.
        
        Calculated as Lhot = Ldiss + Lseed
        
        where Ldiss is the energy dissipated from the accretion flow, and
        Lseed is the seed photon luminosity intercpted by the corona
        
        Returns
        -------
        Lhot : float
            Total luminosity of hot Comtpon corona - units : W

        """
        
        self.Ldiss, err = quad(lambda rc: 2*sigma_sb*self.calc_Tnt(rc) * 2*np.pi*rc * self.Rg**2,
                    self.risco, self.r_h)
        
        
        self.Lseed = self.Lseed_hotCorona()
        Lhot = self.Ldiss + self.Lseed

        return Lhot 
    
    
    def hotComp_annuli(self, r, dr):
        """
        Calculates spectrum from radial slice of hot comp region
        Neccessary in order to apply kyconv correctly!
        
        Note - this is in FRAME of the BLACK HOLE!
        
        Returns
        -------
        Lnu_ann : array
            Spectrum from annular slice of hot Compton region - units : W/Hz

        """

        T4ann = self.calc_Tnt(r)
        Ldiss_ann = sigma_sb * T4ann * self.Rg**2
        #For Lseed need to /4pi r dr so that normalise correctly in kyconv
        #Also need to /N_bins - assuming the corona has no radial dependence in its emissivity
        Lseed_ann = self.Lseed_hotCorona()/(4*np.pi*r*dr * (len(self.logr_hc_bins) - 1))
        Ltot_ann = Ldiss_ann + Lseed_ann
        
        Lnu_ann = Ltot_ann * (self.Fnu_seed_hot/np.trapezoid(self.Fnu_seed_hot, self.nu_grid))
            
        return Lnu_ann
        
    
    
    def do_relHotCompSpec(self):
        """
        Calculates spectrum of hot compton region - with relativity!
        
        Returns
        -------
        Lnu_hot_rel : array
            Total RELATIVISTIC spectrum from hot Compton region - units: W/Hz
            
        """

        for i in range(len(self.logr_hc_bins) - 1):
            dr_bin = 10**self.logr_hc_bins[i+1] - 10**self.logr_hc_bins[i]
            rmid = 10**(self.logr_hc_bins[i] + self.dlog_r/2)
            
            Lnu_ann = self.hotComp_annuli(rmid, dr_bin)
                
            fluxs = (Lnu_ann*4 * u.W/u.Hz).to(u.keV/u.s/u.keV,
                                              equivalencies=u.spectral()).value
            
        
            fluxs = fluxs/(4*np.pi * self.d**2)
            phs = list((fluxs*self.dEs)/self.Egrid)
                
            
            if self.logr_hc_bins[i] <= 3 and self.logr_hc_bins[i+1] <= 3:
                r_in = 10**self.logr_hc_bins[i]
                r_out = 10**self.logr_hc_bins[i+1]
                r_br = rmid
                
                #Convolving annulus with kyconv
                kyparams = [self.a, np.rad2deg(self.inc), r_in, 0, r_out, 0, 0,
                            r_br, 0, 0, self.numE, -1]
                
                xspec.callModelFunction('kyconv', list(self.Ebins), 
                                                kyparams, phs)
                phs_r = np.array(phs)/self.dEs
                
                L_kev = phs_r * self.Egrid * 4 * np.pi * self.d**2
                Lnu_r = (L_kev * u.keV/u.s/u.keV).to(u.W/u.Hz, equivalencies=u.spectral()).value
                Lnu_r /= (self.cosinc/0.5) #kyconv includes inclination factor
                                           #However, since spherical geometry
                                           #This needs to be removed!
            
            else:
                #If out of bounds for tables do non-relativistic
                #This is fine - as beyon 1000Rg Gr effects tiny!
                #Since no kyconv - we now apply the normalisation here instead
                Lnu_r = Lnu_ann * 2*np.pi * 2 * rmid * dr_bin

            if i == 0:
                Lnu_all = Lnu_r
            else:
                Lnu_all = np.column_stack((Lnu_all, Lnu_r))
        
        
        if np.shape(Lnu_all) != np.shape(self.Egrid):
            Lnu_tot = np.sum(Lnu_all, axis=-1)
        else:
            Lnu_tot = Lnu_all

        self.Lnu_hot_rel = Lnu_tot
        return self.Lnu_hot_rel
        
    
    def do_nonrelHotCompSpec(self):
        """
        Calculates spectrum of hot comptonised region - no relativity
        
        Returns
        -------
        Lnu_hot_norel : array
            Total NON-RELATIVISTIC spectrum from hot Compton region
            Units : W/Hz
        """
        Lum = self.hotCorona_lumin()
        self.Lnu_hot_norel = Lum * (self.Fnu_seed_hot/np.trapezoid(
            self.Fnu_seed_hot, self.nu_grid))
            
        return self.Lnu_hot_norel 
        
            
    
    
    ###########################################################################
    #---- Methods for extracting spectra
    #     This is the recommended way, as it takes into account unit choices
    #     and region limits
    ###########################################################################
    
    def get_DiscComponent(self, rel=True):
        """
        Extracts disc component from SED
        
        First checks if disc region exists in current geometry - if not then
        returns 0 array

        Parameters
        ----------
        rel : Bool, optional
            Flag for whether to include relativistic correction. 
                - True: Full GR is used
                - False: Relativistic transfer to the observer is ignored
                         (i.e non-relativistic SED)
            The default is True.

        Returns
        -------
        Ld : array
            Disc spectral component in whatever units are currently set.

        """
        
        if rel == True:
            sfix = 'rel'
        else:
            sfix = 'nonrel'
        
        #Checking if disc attribute already exists. If not, then checks disc
        #region exists in current geometry, and if so then calculates spectrum
        if hasattr(self, f'Lnu_disc_{sfix}'):
            Ld = getattr(self, f'Lnu_disc_{sfix}')
        elif self.r_w != self.r_out and len(self.logr_ad_bins) != 1:
            Ld_func = getattr(self, f'do_{sfix}DiscSpec')
            Ld = Ld_func()
        else:
            Ld = np.zeros(len(self.nu_grid))
            
        #Converting to currently set units
        Ld = self._to_newUnit(Ld, as_spec=True)
        if self.as_flux == True:
            Ld = self._to_flux(Ld)
        
        return Ld

    
    def get_WarmComponent(self, rel=True):
        """
        Extracts warm Comptonised component from SED
        
        First checks if warm Compton region exists in current geometry - if not
        then returns 0 array

        Parameters
        ----------
        rel : Bool, optional
            Flag for whether to include relativistic correction. 
                - True: Full GR is used
                - False: Relativistic transfer to the observer is ignored
                         (i.e non-relativistic SED)
            The default is True.

        Returns
        -------
        Lw : array
            Warm Compton spectral component in whatever units are currently set.

        """
        
        if rel == True:
            sfix = 'rel'
        else:
            sfix = 'nonrel'
        
        #Checking if disc attribute already exists. If not, then checks disc
        #region exists in current geometry, and if so then calculates spectrum
        if hasattr(self, f'Lnu_warm_{sfix}'):
            Lw = getattr(self, f'Lnu_warm_{sfix}')
        elif self.r_w != self.r_h and len(self.logr_wc_bins) != 1:
            Lw_func = getattr(self, f'do_{sfix}WarmCompSpec')
            Lw = Lw_func()
        else:
            Lw = np.zeros(len(self.nu_grid))
            
        #Converting to currently set units
        Lw = self._to_newUnit(Lw, as_spec=True)
        if self.as_flux == True:
            Lw = self._to_flux(Lw)
        
        return Lw
    
    
    def get_HotComponent(self, rel=True):
        """
        Extracts hot Comptonised component from SED
        
        First checks if hot Compton region exists in current geometry - if not
        then returns 0 array

        Parameters
        ----------
        rel : Bool, optional
            Flag for whether to include relativistic correction. 
                - True: Full GR is used
                - False: Relativistic transfer to the observer is ignored
                         (i.e non-relativistic SED)
            The default is True.

        Returns
        -------
        Lh : array
            Hot Compton spectral component in whatever units are currently set.

        """
        
        if rel == True:
            sfix = 'rel'
        else:
            sfix = 'nonrel'
        
        #Checking if disc attribute already exists. If not, then checks disc
        #region exists in current geometry, and if so then calculates spectrum
        if hasattr(self, f'Lnu_hot_{sfix}'):
            Lh = getattr(self, f'Lnu_hot_{sfix}')
        elif self.r_h != self.risco and len(self.logr_hc_bins) != 1:
            Lh_func = getattr(self, f'do_{sfix}HotCompSpec')
            Lh = Lh_func()
        else:
            Lh = np.zeros(len(self.nu_grid))
            
        #Converting to currently set units
        Lh = self._to_newUnit(Lh, as_spec=True)
        if self.as_flux == True:
            Lh = self._to_flux(Lh)
        
        return Lh 
        
    
    
    
    def get_totSED(self, rel=True):
        """
        Extracts the total SED (i.e Ldisc + Lwarm + Lhot)

        Parameters
        ----------
        rel : Bool, optional
            Flag for whether to include relativistic correction. 
                - True: Full GR is used
                - False: Relativistic transfer to the observer is ignored
                         (i.e non-relativistic SED)
            The default is True.

        Returns
        -------
        Ltot : array
            Total SED in whatever units are currently set.

        """
        
        Ld = self.get_DiscComponent(rel=rel)
        Lw = self.get_WarmComponent(rel=rel)
        Lh = self.get_HotComponent(rel=rel)
        
        Ltot = Ld + Lw + Lh
        return Ltot
        
    
    
    ###########################################################################
    #---- Methods for extracting other useful attributes
    ###########################################################################
    
    def get_Ledd(self):
        """
        Gives system Eddington luminosity in whatever units are currently set
        
        Ignores flux flag - ALWAYS as luminosity 
        
        Returns
        -------
        Ledd : float
            Eddington luminosity.

        """
        
        Ledd = self._to_newUnit(self.L_edd, as_spec=False)
        return Ledd
    
    def get_Rg(self):
        """
        Gives scale of one gravitational radius
        
        If units are: cgs, cgs_wave, or counts - then returns in cm
        If units are: SI or SI_wave - then returns in m

        Returns
        -------
        R_G : float
            Gravitational radius.

        """
        
        if self.units == 'SI' or self.units == 'SI_wave':
            R_G = self.Rg
        else:
            R_G = self.R_G * 100
        
        return R_G
    
    def get_Mdot(self):
        """
        Gives PHYSICAL mass accretion rate
        
        If units are: cgs, cgs_wave, or counts - then returns in g/s
        If units are: SI or SI_wave - then returns in kg/s

        Returns
        -------
        Mdot : float
            Physical mass accretion rate.

        """
        
        Mdot = self.mdot * self.Mdot_edd #kg/s
        if self.units == 'SI' or self.units == 'SI_wave':
            pass
        else:
            Mdot *= 1e3 #g/s
        
        return Mdot
        
        
    
    
    
    
    
    
    




class relqso(relagn):
    
    """
    relqso - A simplified version of relagn, where we fix some parameters to
    typical values (kTe_hot = 100keV, kTe_warm=0.2keV, Gamma_warm = 2.5,
    r_warm = 2r_hot, and h_max = 10 (maybe 100 - change later))
    
    Here we also calculate r_hot based off the condition that L_diss = 0.02Ledd,
    which also sets \Gamma_hot, based of the ratio to Ldiss and Lseed
    Finally, the outer disc radius, r_out, is set to the self-gravity radius
    
    For more details on the model - see Hagen & Done (in prep)
    
    
    Attributes 
    ----------
    r_h : float
        Outer radius of hot Compton region
    
    r_w : float
        Outer radius of warm Compton region
    
    gamma_h : float
        Spectral index of hot Compton component
    
    risco : float
        Innremost stable circular orbit - units : Dimensionless (Rg)
    
    r_sg : float
        Self-Gravity radius of the disc, following Laor & Netzer 1989
        Units : Dimensionless (Rg)
    
    eta : float
        Accretion efficiency. Used to convert from mass accretion rate to 
        luminosity. i.e L = eta * Mdot * c^2
        Units : dimensionless
    
    Egrid : array
        Energy grid used for calculations (in frame of black hole) - units : keV
    
    nu_grid : array
        Frequency grid corresponding to Egrid - units : Hz
    
    wave_grid : array
        Wavelength grid corresponding to Egrid - units : Angstrom
    
    E_obs : array
        Energy grid converted to observer frame (i.e redshift corrected) - units : keV
    
    nu_obs : array
        Frequency grid converted to observer frame - units : Hz
    
    wave_obs : array
        Wavelength grid converted to observer frame - units : Angstrom
    
    ###########################################################################
    There are other attributes, however it is recomended that you 
    extract these using the built in methods in order to deal with unit choices 
    correctly. The remaining ones after this are only necessary for the internal 
    calculations, so no need to worry about them!
    ###########################################################################
    
    
    Important Methods
    -----------------
    get_totSED(rel=True)
        Extracts the total SED (disc + warm + hot components) in whatever
        units are set
    
    get_DiscComponent(rel=True)
        Extracts disc component from SED (after checking if it exists in the
        current model geometry) in whatever units are set
    
    get_WarmComponent(rel=True)
        Extracts warm Compton component from SED (after checking if it exists
        in the current model geometry) in whatever units are set
    
    get_HotComponent(rel=True)
        Extracts hot Compton component from SED (after checking if it exists
        in the current model geometry) in whatever units are set
    
    set_units(new_unit='cgs')
        Sets the system units to use when extracting spectra / system
        properties
    
    set_flux()
        Sets a flag s.t all spectra are given in terms of flux rather than
        luminosity
    
    set_lum()
        Sets a flag s.t all spectra are given in luminosity (this is the default)
    
    get_Ledd()
        Gives Eddington luminosity in whatever units are set
        (Note: in frame of black hole!!)
    
    get_Rg()
        Gives scale of gravitaional radius (Rg = GM/c^2) in whatever units 
        are set
    
    get_Mdot()
        Gives PHYSICAL mass accretion rate, in either g/s or kg/s
        (depending on what units are set)
    
    
    ###########################################################################
    
    
    """
    
    
    def __init__(self,
                 M=1e5,
                 dist=1,
                 log_mdot=-1,
                 a=0,
                 cos_inc=0.5,
                 fcol=1,
                 z=0):
        """
        Parameters
        ----------
        M : float
            Black hole mass - units : Msol
        dist : float
            Co-Moving Distance - units : Mpc
        log_mdot : float
            log mass accretion rate - units : Eddington
        a : float
            Dimensionless Black Hole spin - units : Dimensionless
        cos_inc : float
            cos inclination angle, as measured from the z-axis, with disc in the 
            x-y plane - units : Dimensionless
        fcol : float
            Colour temperature correction as described in Done et al. (2012)
            If -ve then follows equation 1 and 2 in Done et al. (2012).
            If +ve then assumes this to be constant correction over entire disc region
        z : float
            Redshift
        """
        
        #Read params
        self.M = M
        self.D, self.d = dist, (dist * u.Mpc).to(u.cm).value
        self.mdot = 10**(log_mdot)
        self.a = np.float64(a)
        self.inc = np.arccos(cos_inc)
        self.cosinc = cos_inc
        self.fcol = fcol
        self.z = z
        
        
        #Performing checks
        self._check_spin()
        self._check_inc()
        self._check_mdot()
        
        #Calculating disc params 
        self._calc_risco()
        self._calc_r_selfGravity()
        self._calc_Ledd()
        self._calc_efficiency()
        
        #physical conversion factors
        self.Mdot_edd = self.L_edd/(self.eta * c**2)
        self.Rg = (G * self.M)/(c**2)
        
        #Calculating disc regions
        self.dlog_r = 1/self.dr_dex
        self._set_rhot()
        self.r_w = 2*self.r_h
        self.r_out = self.r_sg
        
        if self.r_w > self.r_sg:
            self.r_w = self.r_sg
        
        
        #Setting other parameters
        self.kTe_h = 100 #keV
        self.kTe_w = 0.2 #keV
        self.gamma_w = 2.5
        self.hmax = min(100.0, self.r_h)
        self._set_gammah()
        
        
        #Setting up grids
        self.Ebins = np.geomspace(self.Emin, self.Emax, self.numE)
        self.dEs = self.Ebins[1:] - self.Ebins[:-1]

        self.Egrid = self.Ebins[:-1] + 0.5 * self.dEs
        self.nu_grid = (self.Egrid * u.keV).to(u.Hz,
                                equivalencies=u.spectral()).value
        self.nu_obs = self.nu_grid/(1 + self.z) #Observers frame
        self.E_obs = self.Egrid/(1 + self.z)
        
        #Creating radal grid over disc and warm compton regions
        self.logr_ad_bins = self._make_rbins(np.log10(self.r_w), np.log10(self.r_out))
        self.logr_wc_bins = self._make_rbins(np.log10(self.r_h), np.log10(self.r_w))
        self.logr_hc_bins = self._make_rbins(np.log10(self.risco), np.log10(self.r_h))
            
        #Shape of hot corona (rest frame)
        self._hot_specShape()
        
        
    
    def _check_mdot(self):
        """
        Checks is mdot within bounds.
        If log mdot < -1.65 then too small (The ENTIRE flow will be the corona..)
        If log mdot > 0.5 then the model is probably invalid (this is NOT a 
        super Eddington model...)
        """
        
        lmdot = np.log10(self.mdot)
        if lmdot >= -1.65 and lmdot <= 0.5:
            pass
        else:
            raise ValueError('mdot is out of bounds! \n'
                             'Require: -1.65 <= log mdot <= 0.5')
    
    
    def _set_rhot(self):
        """
        Finds r_hot based on the condition L_diss = 0.02 Ledd, in the frame
        of the black hole
        Uses a refined radial grid, s.t we have more confidence in answer
        
        """

        dlr = 1/1000 #Somewhat refined grid for this calculation
        log_rall = self._make_rbins(np.log10(self.risco), np.log10(self.r_sg),
                                        dlog_r=dlr)
        Ldiss = 0.0
        i = 0
        while Ldiss < 0.02 * self.L_edd and i < len(log_rall) - 1:
            rmid = 10**(log_rall[i] + dlr/2)            
            dr = 10**(log_rall[i + 1]) - 10**(log_rall[i])
            
            Tnt4 = self.calc_Tnt(rmid)
            Ldiss += sigma_sb*Tnt4 * 4*np.pi*rmid*dr * self.Rg**2
            i += 1
            
        self.r_h = rmid

    
    def _set_gammah(self):
        """
        Calculates the spectral index of the hot Compton region

        """
        self.hotCorona_lumin()
        self.gamma_h = (7/3) * (self.Ldiss/self.Lseed)**(-0.1)
        
        
    





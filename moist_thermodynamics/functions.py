# -*- coding: utf-8 -*-
"""
Provides accurate thermodynamic functions for moist atmosphere

Author: Bjorn Stevens (bjorn.stevens@mpimet.mpg.de)
copygright, bjorn stevens Max Planck Institute for Meteorology, Hamburg

License: BSD-3C
"""
#
from . import constants
import numpy as np
from scipy import interpolate, optimize

def planck(T,nu):
    """Planck source function (J/m2 per steradian per Hz)
    
    Args:
        T: temperature in kelvin
        nu: frequency in Hz
        
    Returns:
        Returns the radiance in the differential frequency interval per unit steradian. Usually we
        multiply by $\pi$ to convert to irradiances 
        
    >>> planck(300,1000*constants.c)
    8.086837160291128e-15
    """
    c = constants.speed_of_light
    h = constants.planck_constant
    kB= constants.boltzmann_constant
    return (2 * h * nu**3 / c**2) / (np.exp(h*nu/(kB*T))-1)

def es_liq(T):
    """Returns saturation vapor pressure (Pa) over planer liquid water
    
    Encodes the empirical fits of Wagner and Pruss (2002), Eq 2.5a (page 399). Their formulation
    is compared to other fits in the example scripts used in this package, and deemed to be the 
    best reference.
        
    Args:
        T: temperature in kelvin
                
    Reference: 
        W. Wagner and A. Pruß , "The IAPWS Formulation 1995 for the Thermodynamic Properties
    of Ordinary Water Substance for General and Scientific Use", Journal of Physical and Chemical
    Reference Data 31, 387-535 (2002) https://doi.org/10.1063/1.1461829
    
    >>> es_liq(np.asarray([273.16,305.]))
    array([ 611.65706974, 4719.32683147])
    """
    TvC = constants.temperature_water_vapor_critical_point
    PvC = constants.pressure_water_vapor_critical_point

    vt = 1.-T/TvC
    es = PvC * np.exp(TvC/T * (-7.85951783*vt       + 1.84408259*vt**1.5 - 11.7866497*vt**3 
                               + 22.6807411*vt**3.5 - 15.9618719*vt**4   + 1.80122502*vt**7.5))
    return es

def es_ice(T):
    """Returns sublimation vapor pressure (Pa) over simple (Ih) ice
    
    Encodes the emperical fits of Wagner et al., (2011) which also define the IAPWS standard for
    sublimation vapor pressure over ice-Ih
    
    Args:
        T: temperature in kelvin
    
    Reference:
        Wagner, W., Riethmann, T., Feistel, R. & Harvey, A. H. New Equations for the Sublimation
        Pressure and Melting Pressure of H 2 O Ice Ih. Journal of Physical and Chemical Reference
        Data 40, 043103 (2011).


    >>> es_ice(np.asarray([273.16,260.]))
    array([611.655     , 195.80103377])
    """
    TvT = constants.temperature_water_vapor_triple_point
    PvT = constants.pressure_water_vapor_triple_point

    a1 = -0.212144006e+2
    a2 =  0.273203819e+2
    a3 = -0.610598130e+1
    b1 =  0.333333333e-2
    b2 =  0.120666667e+1
    b3 =  0.170333333e+1
    theta = T/TvT
    es =  PvT * np.exp((a1*theta**b1 + a2 * theta**b2 + a3 * theta**b3)/theta)
    return es

def es_mxd(T):
    """Returns the minimum of the sublimation and saturation vapor pressure
    
    Calculates both the sublimation vapor pressure over ice Ih using es_ice and that over planar
    water using es_liq, and returns the minimum of the two quantities.
    
    Args:
        T: temperature in kelvin
        
    Returns:
        value of es_ice(T) for T < 273.15 and es_liq(T) otherwise
    
    >>> es_mxd(np.asarray([305.,260.]))
    array([4719.32683147,  195.80103377])
    """
    return np.minimum(es_liq(T),es_ice(T))

def es_liq_analytic(T, delta_cl=constants.delta_cl):
    """Analytic approximation for saturation vapor pressure over iquid
    
    Uses the rankine (constant specific heat, negligible condensate volume) approximations to 
    calculate the saturation vapor pressure over liquid.  The procedure is described in Eq(4) of
    Romps (2017) and best approximates the actual value for specific heats that differ slightly
    from the best estimates of these quantities which are provided as default quantities.  
    Romps recommends cl = 4119 J/kg/K, and cpv = 1861 J/kg/K.
    
    Args:
        T: temperature in kelvin
        delta_cl: differnce between isobaric specific heat capacity of vapor and that of liquid.
    
    Returns:
        value of saturation vapor pressure over liquid water in Pa
        
    Reference:
        Romps, D. M. Exact Expression for the Lifting Condensation Level. Journal of the Atmospheric
        Sciences 74, 3891–3900 (2017).
        Romps, D. M. Accurate expressions for the dew point and frost point derived from the Rankine-
        Kirchhoff approximations. Journal of the Atmospheric Sciences (2021) doi:10.1175/JAS-D-20-0301.1.
        
    >>> es_liq_analytic(np.asarray([273.16,305.]))
    array([ 611.655     , 4711.13161169])
    """
    TvT = constants.temperature_water_vapor_triple_point
    PvT = constants.pressure_water_vapor_triple_point
    lvT = constants.vaporization_enthalpy_triple_point
    Rv  = constants.water_vapor_gas_constant

    c1  = delta_cl/Rv
    c2  = lvT/(Rv*TvT) - c1
    es  = PvT * np.exp(c2*(1.-TvT/T)) * (T/TvT)**c1
    return es

def es_ice_analytic(T, delta_ci=constants.delta_ci):
    """Analytic approximation for saturation vapor pressure over ice
    
    Uses the rankine (constant specific heat, negligible condensate volume) approximations to 
    calculate the saturation vapor pressure over ice.  The procedure is described in Eq(4) of
    Romps (2017) and best approximates the actual value for specific heats that differ slightly
    from the best estimates of these quantities which are provided as default quantities.  
    Romps recommends ci = 1861 J/kg/K, and cpv = 1879 J/kg/K.
    
    Args:
        T: temperature in kelvin
        delta_cl: differnce between isobaric specific heat capacity of vapor and that of liquid.
    
    Returns:
        value of saturation vapor pressure over liquid water in Pa
        
    Reference:
        Romps, D. M. Exact Expression for the Lifting Condensation Level. Journal of the Atmospheric
        Sciences 74, 3891–3900 (2017).
        Romps, D. M. Accurate expressions for the dew point and frost point derived from the Rankine-
        Kirchhoff approximations. Journal of the Atmospheric Sciences (2021) doi:10.1175/JAS-D-20-0301.1.


    >>> es_ice_analytic(np.asarray([273.16,260.]))
    array([611.655     , 195.99959431])
    """
    TvT = constants.temperature_water_vapor_triple_point
    PvT = constants.pressure_water_vapor_triple_point
    lsT = constants.sublimation_enthalpy_triple_point
    Rv  = constants.water_vapor_gas_constant

    c1  = delta_ci/Rv
    c2  = lsT/(Rv*TvT) - c1
    es  = PvT * np.exp(c2*(1.-TvT/T)) * (T/TvT)**c1
    return es

def es_mxd_analytic(T, delta_cl=constants.delta_cl, delta_ci=constants.delta_ci):
    """Returns the minimum of the analytic sublimation and saturation vapor pressure
    
    Calculates both the sublimation vapor pressure over ice Ih using es_ice_analytic and
    that over planar water using es_liq_analytic, and returns the minimum of the two
    quantities.
    
    Args:
        T: temperature in kelvin
        
    Returns:
        value of es_ice_analytic(T) for T < 273.15 and es_liq_analytic(T) otherwise
    
    >>> es_ice_analytic(np.asarray([273.16,260.]))
    array([611.655     , 195.99959431])
    """
    return np.minimum(es_liq_analytic(T,delta_cl),es_ice_analytic(T,delta_ci))

def vaporization_enthalpy(TK,delta_cl=constants.delta_cl):
    """Returns the vaporization enthlapy of water (J/kg)
    
    The vaporization enthalpy is calculated from a linear depdence on temperature about a 
    reference value valid at the melting temperature.  This approximation is consistent with the 
    assumption of a Rankine fluid.
    
    Args:
        T: temperature in kelvin
        delta_cl: differnce between isobaric specific heat capacity of vapor and that of liquid.

    >>> vaporization_enthalpy(np.asarray([305.,273.15]))
    array([2427211.264, 2500930.   ])
    """
    T0  = constants.standard_temperature
    lv0 = constants.vaporization_enthalpy_stp
    return lv0 + delta_cl*(TK-T0)

def sublimation_enthalpy(TK,delta_ci=constants.delta_ci):
    """Returns the sublimation enthlapy of water (J/kg)
    
    The sublimation enthalpy is calculated from a linear depdence on temperature about a 
    reference value valid at the melting temperature.  This approximation is consistent with the 
    assumption of a Rankine fluid.
    
    Args:
        T: temperature in kelvin
        delta_cl: differnce between isobaric specific heat capacity of vapor and that of liquid.


    >>> sublimation_enthalpy(273.15)
    2834350.0
    """
    T0  = constants.standard_temperature
    ls0 = constants.sublimation_enthalpy_stp  
    return ls0 + delta_ci*(TK-T0)

def partial_pressure_to_mixing_ratio(pp,p):
    """Returns the mass mixing ratio given the partial pressure and pressure
    
    >>> partial_pressure_to_mixing_ratio(es_liq(300.),60000.)
    0.0389569254590098
    """
    eps1 = constants.rd_over_rv
    return eps1*pp/(p-pp)

def mixing_ratio_to_partial_pressure(r,p):
    """Returns the partial pressure (pp in units of p) from a gas' mixing ratio
    
    Args:
        r: mass mixing ratio (unitless)
        p: pressure in same units as desired return value 
    

    >>> mixing_ratio_to_partial_pressure(2e-5,60000.)
    1.929375975915276
    """
    eps1 = constants.rd_over_rv
    return r*p/(eps1+r)

def partial_pressure_to_specific_humidity(pp,p):
    """Returns the specific mass given the partial pressure and pressure.  
    
    The specific mass can be written in terms of partial pressure and pressure as
    expressed here only if the gas quanta contains no condensate phases.  In this 
    case the specific humidity is the same as the co-dryair specific mass. In
    situations where condensate is present one should instead calculate 
    $q = r*(1-qt)$ which would require an additional argument

    >>> partial_pressure_to_specific_humidity(es_liq(300.),60000.)
    0.037496189210922945
    """   
    r    = partial_pressure_to_mixing_ratio(pp,p)
    return r/(1+r)

def theta_e_bolton(TK,PPa,qt,es=es_liq):
    """Returns the pseudo equivalent potential temperature.
    
    Following Eq. 43 in Bolton (1980) the (pseudo) equivalent potential temperature
    is calculated and returned by this function
    
    Args:
        TK: temperature in kelvin
        PPa: pressure in pascal
        qt: specific total water mass
        es: form of the saturation vapor pressure to use
    
    Reference:
        Bolton, D. The Computation of Equivalent Potential Temperature. Monthly Weather 
        Review 108, 1046–1053 (1980).
    """
    P0    = constants.standard_pressure
    p2r   = partial_pressure_to_mixing_ratio
    r2p   = mixing_ratio_to_partial_pressure
    
    rv = np.minimum(qt/(1.-qt), p2r(es(TK),PPa))  # mixing ratio of vapor (not gas Rv)
    pv = r2p(rv,PPa)

    TL     = 55.0 + 2840./(3.5*np.log(TK) - np.log(pv/100.) - 4.805)    
    return  TK*(P0/PPa)**(0.2854*(1.0 - 0.28*rv)) * np.exp((3376./TL - 2.54)*rv*(1+0.81*rv))

def theta_e(TK,PPa,qt,es=es_liq):
    """Returns the equivalent potential temperature
    
    Follows Eq. 11 in Marquet and Stevens (2022). The closed form solutionis derived for a
    Rankine-Kirchoff fluid (constant specific heats).  Differences arising from its
    calculation using more accurate expressions (such as the default) as opposed to less 
    accurate, but more consistent, formulations are on the order of millikelvin
    
    Args:
        TK: temperature in kelvin
        PPa: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure
        
    Reference:
        Marquet, P. & Stevens, B. On Moist Potential Temperatures and Their Ability to
        Characterize Differences in the Properties of Air Parcels. Journal of the Atmospheric
        Sciences 79, 1089–1103 (2022).
    """
    P0   = constants.standard_pressure
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    cpd  = constants.isobaric_dry_air_specific_heat
    cl   = constants.liquid_water_specific_heat
    p2r  = partial_pressure_to_mixing_ratio
    lv   = vaporization_enthalpy

    ps = es(TK)
    qs = p2r(ps,PPa) * (1.0 - qt)
    qv = np.minimum(qt,qs)

    Re = (1.0-qt)*Rd
    R  = Re + qv*Rv
    pv = qv * (Rv/R) *PPa
    RH = pv/ps
    cpe= cpd + qt*(cl-cpd)
    omega_e = RH**(-qv*Rv/cpe) * (R/Re)**(Re/cpe)
    theta_e = TK*(P0/PPa)**(Re/cpe)*omega_e*np.exp(qv*lv(TK)/(cpe*TK))
    return(theta_e)

def theta_l(TK,PPa,qt,es=es_liq):
    """Returns the liquid-water potential temperature
    
    Follows Eq. 16 in Marquet and Stevens (2022). The closed form solutionis derived for a
    Rankine-Kirchoff fluid (constant specific heats).  Differences arising from its
    calculation using more accurate expressions (such as the default) as opposed to less 
    accurate, but more consistent, formulations are on the order of millikelvin
    
    Args:
        TK: temperature in kelvin
        PPa: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure
        
    Reference:
        Marquet, P. & Stevens, B. On Moist Potential Temperatures and Their Ability to
        Characterize Differences in the Properties of Air Parcels. Journal of the Atmospheric
        Sciences 79, 1089–1103 (2022).
    """
    P0   = constants.standard_pressure
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    cpd  = constants.isobaric_dry_air_specific_heat
    cpv  = constants.isobaric_water_vapor_specific_heat
    p2r  = partial_pressure_to_mixing_ratio
    lv   = vaporization_enthalpy

    ps = es(TK)
    qs = p2r(ps,PPa) * (1. - qt)
    qv = np.minimum(qt,qs)
    ql = qt-qv

    R  = Rd*(1-qt) + qv*Rv
    Rl = Rd + qt*(Rv - Rd)
    cpl= cpd + qt*(cpv-cpd)
    
    omega_l = (R/Rl)**(Rl/cpl) * (qt/(qv+1.e-15))**(qt*Rv/cpl)
    theta_l = (TK*(P0/PPa)**(Rl/cpl)) *omega_l*np.exp(-ql*lv(TK)/(cpl*TK))
    return(theta_l)

def theta_s(TK,PPa,qt,es=es_liq):
    """Returns the entropy potential temperature
    
    Follows Eq. 18 in Marquet and Stevens (2022). The closed form solutionis derived for a
    Rankine-Kirchoff fluid (constant specific heats).  Differences arising from its
    calculation using more accurate expressions (such as the default) as opposed to less 
    accurate, but more consistent, formulations are on the order of millikelvin
    
    Args:
        TK: temperature in kelvin
        PPa: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure
        
    Reference:
        Marquet, P. & Stevens, B. On Moist Potential Temperatures and Their Ability to
        Characterize Differences in the Properties of Air Parcels. Journal of the Atmospheric
        Sciences 79, 1089–1103 (2022).
        
        Marquet, P. Definition of a moist entropy potential temperature: application to FIRE-I
        data flights: Moist Entropy Potential Temperature. Q.J.R. Meteorol. Soc. 137, 768–791 (2011).
    """
    P0   = constants.standard_pressure
    T0   = constants.standard_temperature
    sd00 = constants.entropy_dry_air_satmt
    sv00 = constants.entropy_water_vapor_satmt
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    cpd  = constants.isobaric_dry_air_specific_heat
    cpv  = constants.isobaric_water_vapor_specific_heat
    eps1 = constants.rd_over_rv
    eps2 = constants.rv_over_rd_minus_one
    p2r  = partial_pressure_to_mixing_ratio
    lv   = vaporization_enthalpy

    kappa = Rd/cpd
    e0    = es(T0)
    Lmbd  = ((sv00 - Rv*np.log(e0/P0)) - (sd00 - Rd*np.log(1-e0/P0)))/cpd
    lmbd  = cpv/cpd - 1.
    eta   = 1/eps1
    delta = eps2
    gamma = kappa/eps1
    r0    = e0/(P0-e0)/eta

    ps = es(TK)
    qs = p2r(ps,PPa) * (1. - qt)
    qv = np.minimum(qt,qs)
    ql = qt-qv

    R  = Rd + qv*(Rv - Rd)
    pv = qv * (Rv/R) *PPa
    RH = pv/ps
    rv = qv/(1-qv)

    x1 = (TK/T0)**(lmbd*qt) * (P0/PPa)**(kappa*delta*qt) * (rv/r0)**(-gamma*qt) * RH**(gamma*ql)
    x2 = (1.+eta*rv)**(kappa*(1.+delta*qt)) * (1.+eta*r0)**(-kappa*delta*qt)
    theta_s = (TK*(P0/PPa)**(kappa)) * np.exp(-ql*lv(TK)/(cpd*TK)) * np.exp(qt*Lmbd) * x1 * x2
    return(theta_s)

def theta_es(TK,PPa,es=es_liq):
    """Returns the saturated equivalent potential temperature
    
    Adapted from Eq. 11 in Marquet and Stevens (2022) with the assumption that the gas quanta is
    everywhere just saturated.
    
    Args:
        TK: temperature in kelvin
        PPa: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure
        
    Reference:
        Characterize Differences in the Properties of Air Parcels. Journal of the Atmospheric
        Sciences 79, 1089–1103 (2022).
    """
    P0   = constants.standard_pressure
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    cpd  = constants.isobaric_dry_air_specific_heat
    cl   = constants.liquid_water_specific_heat
    p2q  = partial_pressure_to_specific_humidity
    lv   = vaporization_enthalpy

    ps = es(TK)
    qs = p2q(ps,PPa)

    Re = (1.0-qs)*Rd
    R  = Re + qs*Rv
    cpe= cpd + qs*(cl-cpd)
    omega_e  = (R/Re)**(Re/cpe)
    theta_es = TK*(P0/PPa)**(Re/cpe)*omega_e*np.exp(qs*lv(TK)/(cpe*TK))
    return(theta_es)

def theta_rho(TK,PPa,qt,es=es_liq):
    """Returns the density liquid-water potential temperature
    
    calculates $\theta_\mathrm{l} R/R_\mathrm{d}$ where $R$ is the gas constant of a
    most fluid.  For an unsaturated fluid this is identical to the density potential
    temperature baswed on the two component fluid thermodynamic constants.
    
    Args:
        TK: temperature in kelvin
        PPa: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure
    """
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    p2r  = partial_pressure_to_mixing_ratio

    ps = es(TK)
    qs = p2r(ps,PPa) * (1. - qt)
    qv = np.minimum(qt,qs)
    theta_rho = theta_l(TK,PPa,qt,es) * (1.-qt + qv*Rv/Rd)
    return(theta_rho)

def T_from_Te(Te,P,qt,es=es_liq):
    """Returns temperature for an atmosphere whose state is given by theta_e
    
    This  equation allows the temperature to be inferred from a state description
    in terms of theta_e.  It derives temperature by numerically inverting the
    expression for theta_e provided in this package.  Uses a least-squares non-linear
    optimization to find the value of T such that $theta_e(T,P,q) = theta_e$

Args:
        Te: equivalent potential temperature in kelvin
        P: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure

	>>> T_from_Te(350.,100000.,17.e-3)
	array([304.49321301])
    """
    def zero(T,Te,P,qt,es):
        return  np.abs(Te-theta_e(T,P,qt,es))
    return optimize.fsolve(zero,   280., args=(Te,P,qt,es), xtol=1.e-10)

def T_from_Tl(Tl,P,qt,es=es_liq):
    """returns temperature for an atmosphere whose state is given by theta_l
    
    This  equation allows the temperature to be inferred from a state description
    in terms of theta_l.  It derives temperature by numerically inverting the
    expression for theta_l provided in this package.  Uses a least-squares non-linear
    optimization to find the value of T such that $theta_l(T,P,q) = theta_l$

Args:
        Tl: liquid-water potential temperature in kelvin
        P: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure

	>>> T_from_Tl(282., 90000, 20.e-3)
	array([289.73684039])
    """
    def zero(T,Tl,P,qt,es):
        return  np.abs(Tl-theta_l(T,P,qt,es))
    
    return optimize.fsolve(zero,   280., args=(Tl,P,qt,es), xtol=1.e-10)

def T_from_Ts(Ts,P,qt,es=es_liq):
    """Returns temperature for an atmosphere whose state is given by theta_s
    
    This  equation allows the temperature to be inferred from a state description
    in terms of theta_s.  It derives temperature by numerically inverting the
    expression for theta_s provided in this package.  Uses a least-squares non-linear
    optimization to find the value of T such that $theta_s(T,P,q) = theta_s$

Args:
        Ts: entropy potential temperature in kelvin
        P: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure

	>>> T_from_Tl(282.75436951,90000,20.e-3)
	array([289.98864293])
    """
    def zero(T,Ts,P,qt,es):
        return  np.abs(Ts-theta_s(T,P,qt,es))  
    
    return optimize.fsolve(zero,   280., args=(Ts,P,qt,es), xtol=1.e-10)

def P_from_Te(Te,T,qt,es=es_liq):
    """Returns pressure for an atmosphere whose state is given by theta_e
    
    This  equation allows the pressure to be inferred from a state description
    in terms of theta_e.  It derives pressure by numerically inverting the
    expression for theta_e provided in this package.  Uses a least-squares non-linear
    optimization to find the value of P such that $theta_e(T,P,q) = theta_e$
    
Args:
        Tl: liquid-water potential temperature in kelvin
        T:  temperature in kelvin
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure

	>>> P_from_Te(350.,305.,17e-3)
	array([100586.3357635])
    """
    def zero(P,Te,T,qt,es):
        return np.abs(Te-theta_e(T,P,qt,es))
    
    return optimize.fsolve(zero, 90000., args=(Te,T,qt,es), xtol=1.e-10)

def P_from_Tl(Tl,T,qt,es=es_liq):
    """Returns pressure for an atmosphere whose state is given by theta_l
    
    This  equation allows the pressure to be inferred from a state description
    in terms of theta_l.  It derives pressure by numerically inverting the
    expression for theta_l provided in this package.  Uses a least-squares non-linear
    optimization to find the value of P such that $theta_l(T,P,q) = theta_l$
    
    Args:
        Tl: liquid-water potential temperature in kelvin
        T:  temperature in kelvin
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure

	>>> P_from_Tl(282.75436951,290,20.e-3)
	array([90027.65146427])
    """
    def zero(P,Tl,T,qt,es):
        return np.abs(Tl-theta_l(T,P,qt,es))
    
    return optimize.fsolve(zero, 90000., args=(Tl,T,qt,es), xtol=1.e-10)

def plcl(TK,PPa,qt,es=es_liq):
    """Returns the pressure at the lifting condensation level
    
    Calculates the lifting condensation level pressure using an interative solution under the
    constraint of constant theta-l. Exact to within the accuracy of the expression of theta-l
    which depends on the expression for the saturation vapor pressure
    
    Args:
        TK: temperature in kelvin
        PPa: pressure in pascal
        qt: specific total water mass
        
	>>> plcl(300.,102000.,17e-3)
	array([95971.69750248])
    """
    
    def zero(P,Tl,qt,es):
        p2r   = partial_pressure_to_mixing_ratio
        TK = T_from_Tl(Tl,P,qt)
        qs = p2r(es(TK),P) * (1. - qt)
        return np.abs(qs/qt-1.)

    Tl   = theta_l(TK,PPa,qt,es)
    return optimize.fsolve(zero, 80000., args=(Tl,qt,es), xtol=1.e-5)

def plcl_bolton(TK,PPa,qt):
    """Returns the pressure at the lifting condensation level
    
    Following Bolton (1980) the lifting condensation level pressure is derived from the state
    of an air parcel.  Usually accurate to within about 10 Pa, or about 1 m
    
    Args:
        TK: temperature in kelvin
        PPa: pressure in pascal
        qt: specific total water mass
    
    Reference:
        Bolton, D. The Computation of Equivalent Potential Temperature. Monthly Weather 
        Review 108, 1046–1053 (1980).
        
	>>> plcl_bolton(300.,102000.,17e-3)
	95980.41895404423
    """
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    cpd  = constants.isobaric_dry_air_specific_heat
    cpv  = constants.isobaric_water_vapor_specific_heat
    r2p  = mixing_ratio_to_partial_pressure

    cp = cpd + qt*(cpv-cpd)
    R  = Rd  + qt*(Rv-Rd)
    pv = r2p(qt/(1.-qt),PPa)
    Tl = 55 + 2840./(3.5*np.log(TK) - np.log(pv/100.) - 4.805)
    return PPa * (Tl/TK)**(cp/R)

def zlcl(Plcl,T,P,qt,z):
    """Returns the height of the LCL above mean sea-level
    
    Given the Plcl, calculate its height in meters given the height of the ambient state
    from which it (Plcl) was calculated.  This is accomplished by assuming temperature
    changes following a dry adiabat with vertical displacements between the ambient 
    temperature and the ambient LCL
    
    Args:
        Plcl: lifting condensation level in Pa
        T: ambient temperature in kelvin
        P: ambient pressure in pascal
        qt: specific total water mass
        z: height at ambient temperature and pressure 
        
	>>> zlcl(95000.,300.,90000.,17.e-3,500.)
	16.621174077862747
    """
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    cpd  = constants.isobaric_dry_air_specific_heat
    cpv  = constants.isobaric_water_vapor_specific_heat
    g    = constants.gravity_earth

    cp = cpd + qt*(cpv-cpd)
    R  = Rd  + qt*(Rv-Rd)
    return T*(1. - (Plcl/P)**(R/cp)) * cp / g + z

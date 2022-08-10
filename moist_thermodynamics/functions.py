# -*- coding: utf-8 -*-
"""
Author: Bjorn Stevens (bjorn.stevens@mpimet.mpg.de)
"""
#
from . import constants
import numpy as np
from scipy import interpolate, optimize

def planck(T,nu):
    """returns the Planck source function for the given temperature (K) and frequency (Hz)
    """
    
    c = constants.speed_of_light
    h = constants.planck_constant
    kB= constants.boltzmann_constant

    return (2 * h * nu**3 / c**2) / (np.exp(h*nu/(kB*T))-1)

def saturation_vapor_pressure_water_liq(T):
    """ Returns the saturation vapor pressure of water over liquid or ice, or the minimum of the two,
    depending on the specificaiton of the state variable.  The calculation follows Wagner and Pruss (2002)
    fits for saturation over planar liquid. The Wagner and Pruss formulations were found to be the most 
    accurate as compared to the IAPWS standard for warm temperatures
    """
    TvC = constants.temperature_water_vapor_critical_point
    PvC = constants.pressure_water_vapor_critical_point

    vt = 1.-T/TvC
    es = PvC * np.exp(TvC/T * (-7.85951783*vt + 1.84408259*vt**1.5 - 11.7866497*vt**3 + 22.6807411*vt**3.5 - 15.9618719*vt**4 + 1.80122502*vt**7.5))

    return es

def saturation_vapor_pressure_water_ice(T):
    """ Returns the saturation vapor pressure of water over liquid or ice, or the minimum of the two,
    depending on the specificaiton of the state variable.  The calculation follows Wagner et al., 2011 
    fits for saturation over ice. The Wagner et al 2011 form is the IAPWS standard for ice.
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

def saturation_vapor_pressure_water_mxd(T):
    """ Returns the minimum of the saturation vapor pressure of water over liquid or ice, which should
    default to the ice value at temperatures below the freezing point and the liquid value otherwise
    """
    esl = saturation_vapor_pressure_water_liq
    esi = saturation_vapor_pressure_water_ice

    return np.minimum(esl(x),esi(x))

def analytic_saturation_vapor_pressure_water_liq(T):
    """ Returns an analytic approximation to the saturation vapor pressure over liquid.  Useful for
    computations that require consisntency with assumption of cp's being constant.  The analytic expressions
    become identical to Romps in the case when the specific heats are adjusted to his suggested values.
    >>> es([273.16,290.])
    [611.65706974 1919.87719485]
    """
    TvT = constants.temperature_water_vapor_triple_point
    PvT = constants.pressure_water_vapor_triple_point
    lvT = constants.vaporization_enthalpy_triple_point
    Rv  = constants.water_vapor_gas_constant
    cpv = constants.isobaric_water_vapor_specific_heat
    cl  = constants.liquid_water_specific_heat

    c1 = (cpv-cl)/Rv
    c2 = lvT/(Rv*TvT) - c1
    es = PvT * np.exp(c2*(1.-TvT/T)) * (T/TvT)**c1
    return es

def analytic_saturation_vapor_pressure_water_ice(T):
    """ Returns an analytic approximation to the saturation vapor pressure over ice.  Useful for
    computations that require consisntency with assumption of cp's being constant.  The analytic expressions
    become identical to Romps in the case when the specific heats are adjusted to his suggested values.
    >>> es([273.16,290.])
    [611.65706974 1919.87719485]
    """
    TvT = constants.temperature_water_vapor_triple_point
    PvT = constants.pressure_water_vapor_triple_point
    lsT = constants.sublimation_enthalpy_triple_point
    Rv  = constants.water_vapor_gas_constant
    cpv = constants.isobaric_water_vapor_specific_heat
    ci  = constants.frozen_water_specific_heat

    c1 = (cpv-ci)/Rv
    c2 = lsT/(Rv*TvT) - c1
    es = PvT * np.exp(c2*(1.-TvT/T)) * (T/TvT)**c1
    return es

def analytic_saturation_vapor_pressure_water_mxd(T):
    """ Returns the minimum of the saturation vapor pressure of water over liquid or ice, which should
    default to the ice value at temperatures below the freezing point and the liquid value otherwise
    """
    esl = analytic_saturation_vapor_pressure_water_liq
    esi = analytic_saturation_vapor_pressure_water_ice

    return np.minimum(esl(x),esi(x))

def phase_change_enthalpy(TK,fusion=False):
    """ Returns the enthlapy [J/g] of vaporization (default) of water vapor or
    (if fusion=True) the fusion anthalpy.  Input temperature can be in degC or Kelvin
    >>> phase_change_enthalpy(273.15)
    2500.8e3
    """

    T0  = constants.standard_temperature
    lv0 = constants.vaporization_enthalpy_stp
    lf0 = constants.melting_enthalpy_stp
    cpv = constants.isobaric_water_vapor_specific_heat
    cl  = constants.liquid_water_specific_heat
    ci  = constants.frozen_water_specific_heat

    if (fusion):
        el = lf0 + (cl-ci)*(TK-T0)
    else:
        el = lv0 + (cpv-cl)*(TK-T0)

    return el

def partial_pressure_to_specific_humidity(pp,p):
    """ Calculates specific mass from the partial and total pressure
    assuming both have same units and no condensate is present.  Returns value
    in units of kg/kg. checked 15.06.20
    >>> pp2sm(es(273.16),60000.)
    0.00636529
    """
    
    eps1 = constants.rd_over_rv
    x    = eps1*pp/(p-pp)
    return x/(1+x)

def partial_pressure_to_mixing_ratio(pp,p):
    """ Calculates mixing ratio from the partial and total pressure
    assuming both have same unitsa nd no condensate is present. Returns value
    in units of kg/kg.
    """
    
    eps1 = constants.rd_over_rv
    return eps1*pp/(p-pp)

def mixing_ratio_to_partial_pressure(r,p):
    """ Calculates partial pressure from mixing ratio and pressure, if mixing ratio
    units are greater than 1 they are normalized by 1000.
    """
    
    eps1 = constants.rd_over_rv
    return r*p/(eps1+r)

def pseudo_theta_e(TK,PPa,qt,es=saturation_vapor_pressure_water_liq):
    """ Calculates pseudo equivalent potential temperature. following Bolton
    checked 31.07.20
    """

    P0    = constants.standard_pressure
    p2r   = partial_pressure_to_mixing_ratio
    r2p   = mixing_ratio_to_partial_pressure
    
    rv = np.minimum(qt/(1.-qt), p2r(es(TK),PPa))  # mixing ratio of vapor (not gas Rv)
    pv = r2p(rv,PPa)

    TL     = 55.0 + 2840./(3.5*np.log(TK) - np.log(pv/100.) - 4.805)
    theta_e = TK*(P0/PPa)**(0.2854*(1.0 - 0.28*rv)) * np.exp((3376./TL - 2.54)*rv*(1+0.81*rv))
    
    return(theta_e)

def theta_e(TK,PPa,qt,es=saturation_vapor_pressure_water_liq):
    """ Calculates equivalent potential temperature corresponding to Eq. 2.42 in the Clouds
    and Climate book.
    """

    P0   = constants.standard_pressure
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    lv   = phase_change_enthalpy
    p2r  = partial_pressure_to_mixing_ratio

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

def theta_es(TK,PPa,es=saturation_vapor_pressure_water_liq):
    """ Calculates equivalent potential temperature corresponding to Eq. 2.42 in the Clouds
    and Climate book but assuming that the water amount is just saturated
    """
    
    P0   = constants.standard_pressure
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    cpd  = constants.isobaric_dry_air_specific_heat
    cpv  = constants.isobaric_water_vapor_specific_heat
    p2q  = partial_pressure_to_specific_humidity

    ps = es(TK)
    qs = p2q(ps,PPa)

    Re = (1.0-qs)*Rd
    R  = Re + qs*Rv
    cpe= cpd + qs*(cl-cpd)
    omega_e  = (R/Re)**(Re/cpe)
    theta_es = TK*(P0/PPa)**(Re/cpe)*omega_e*np.exp(qs*lv(TK)/(cpe*TK))

    return(theta_es)


def theta_l(TK,PPa,qt,es=saturation_vapor_pressure_water_liq):
    """ Calculates liquid-water potential temperature.  Following Stevens and Siebesma
    Eq. 2.44-2.45 in the Clouds and Climate book
    """

    P0   = constants.standard_pressure
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    cpd  = constants.isobaric_dry_air_specific_heat
    cpv  = constants.isobaric_water_vapor_specific_heat
    eps1 = constants.rd_over_rv

    lv   = phase_change_enthalpy

    ps = es(TK)
    qs = (ps/(PPa-ps)) * eps1 * (1. - qt)
    qv = np.minimum(qt,qs)
    ql = qt-qv

    R  = Rd*(1-qt) + qv*Rv
    Rl = Rd + qt*(Rv - Rd)
    cpl= cpd + qt*(cpv-cpd)
    
    omega_l = (R/Rl)**(Rl/cpl) * (qt/(qv+1.e-15))**(qt*Rv/cpl)
    theta_l = (TK*(P0/PPa)**(Rl/cpl)) *omega_l*np.exp(-ql*lv(TK)/(cpl*TK))

    return(theta_l)

def theta_s(TK,PPa,qt,es=saturation_vapor_pressure_water_liq):
    """ Calculates entropy potential temperature. This follows the formulation of Pascal
    Marquet and ensures that parcels with different theta-s have a different entropy
    """
    P0   = constants.standard_pressure
    T0   = constants.standard_temperature
    sd00 = constants.entropy_dry_air_satp
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    cpd  = constants.isobaric_dry_air_specific_heat
    cpv  = constants.isobaric_water_vapor_specific_heat
    eps1 = constants.rd_over_rv
    eps2 = constants.rv_over_rd_minus_one
    
    p2r  = partial_pressure_to_mixing_ratio
    lv   = phase_change_enthalpy

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

    x1 = 1
    x1 = (T/T0)**(lmbd*qt) * (P0/PPa)**(kappa*delta*qt) * (rv/r0)**(-gamma*qt) * RH**(gamma*ql)
    x2 = (1.+eta*rv)**(kappa*(1+delta*qt)) * (1+eta*r0)**(-kappa*delta*qt)
    theta_s = (TK*(P0/PPa)**(kappa)) * np.exp(-ql*lv(TK)/(cpd*TK)) * np.exp(qt*Lmbd) * x1 * x2

    return(theta_s)

def theta_rho(TK,PPa,qt,es=saturation_vapor_pressure_water_liq):
    """ Calculates theta_rho as theta_l * (1+Rd/Rv qv - qt)
    """

    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    eps1 = constants.rd_over_rv
    p2r  = partial_pressure_to_mixing_ratio

    ps = es(TK)
    qs = p2r(ps,PPa) * (1. - qt)
    qv = np.minimum(qt,qs)
    theta_rho = theta_l(TK,PPa,qt,es) * (1.+ qv/eps1 - qt)

    return(theta_rho)

def T_from_Te(Te,P,qt,es=saturation_vapor_pressure_water_liq):
    """ Given theta_e solves implicitly for the temperature at some other pressure,
    so that theta_e(T,P,qt) = Te
	>>> T_from_Te(350.,1000.,17)
	304.4761977
    """

    def zero(T,Te,P,qt):
        _theta_e = np.vectorize(theta_e)
        return  np.abs(Te-_theta_e(T,P,qt,es=es))
    
    return optimize.fsolve(zero,   280., args=(Te,P,qt), xtol=1.e-10)

def T_from_Tl(Tl,P,qt,es=saturation_vapor_pressure_water_liq):
    """ Given theta_e solves implicitly for the temperature at some other pressure,
    so that theta_e(T,P,qt) = Te
	>>> T_from_Tl(282.75436951,90000,20.e-3)
	290.00
    """
    def zero(T,Tl,P,qt):
        _theta_l = np.vectorize(theta_l)
        return  np.abs(Tl-_theta_l(T,P,qt,es=es))
    
    return optimize.fsolve(zero,   280., args=(Tl,P,qt), xtol=1.e-10)

def T_from_Ts(Ts,P,qt,es=saturation_vapor_pressure_water_liq):
    """ Given theta_e solves implicitly for the temperature at some other pressure,
    so that theta_e(T,P,qt) = Te
	>>> T_from_Tl(282.75436951,90000,20.e-3)
	290.00
    """
    def zero(T,Ts,P,qt):
        _theta_s = np.vectorize(theta_s)
        return  np.abs(Ts-_theta_s(T,P,qt,es=es))
    
    return optimize.fsolve(zero,   280., args=(Ts,P,qt), xtol=1.e-10)

def P_from_Te(Te,T,qt,es=saturation_vapor_pressure_water_liq):
    """ Given Te solves implicitly for the pressure at some temperature and qt
    so that theta_e(T,P,qt) = Te
	>>> P_from_Te(350.,305.,17)
	100464.71590478
    """
    def zero(P,Te,T,qt):
        _theta_e = np.vectorize(theta_e)
        return np.abs(Te-_theta_e(T,P,qt,es=es))
    return optimize.fsolve(zero, 90000., args=(Te,T,qt), xtol=1.e-10)

def P_from_Tl(Tl,T,qt,es=saturation_vapor_pressure_water_liq):
    """ Given Tl solves implicitly for the pressure at some temperature and qt
    so that theta_l(T,P,qt) = Tl
	>>> T_from_Tl(282.75436951,290,20.e-3)
	90000
    """
    def zero(P,Tl,T,qt):
        _theta_l = np.vectorize(theta_l)
        return np.abs(Tl-_theta_l(T,P,qt,es=es))
    
    return optimize.fsolve(zero, 90000., args=(Tl,T,qt), xtol=1.e-10)

def Plcl(TK,PPa,qt,es=saturation_vapor_pressure_water_liq,iterate=False):
    """ Returns the pressure [Pa] of the LCL.  The routine gives as a default the
    LCL using the Bolton formula.  If iterate is true uses a nested optimization to
    estimate at what pressure, Px and temperature, Tx, qt = qs(Tx,Px), subject to
    theta_e(Tx,Px,qt) = theta_e(T,P,qt).  This works for saturated air.
	>>> Plcl(300.,1020.,17)
	96007.495
    """

    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    cpd  = constants.isobaric_dry_air_specific_heat
    cpv  = constants.isobaric_water_vapor_specific_heat
    eps1 = constants.rd_over_rv
    r2p  = mixing_ratio_to_partial_pressure

    def delta_qs(P,Te,qt,es=saturation_vapor_pressure_water_liq):
        TK = T_from_Te(Te,P,qt)
        ps = es(TK)
        qs = (1./(P/ps-1.)) * eps1 * (1. - qt)
        return np.abs(qs/qt-1.)

    if (iterate):
        Te   = theta_e(TK,PPa,qt,es)
        zero = np.vectorize(delta_qs)
        if scalar_input:
            Plcl = optimize.fsolve(zero, 80000., args=(Te,qt), xtol=1.e-10)
            return np.squeeze(Plcl)
        else:
            if (scalar_input3):
                qx =np.empty(np.shape(Te)); qx.fill(np.squeeze(qt)); qt = qx
            elif len(Te) != len(qt):
                print('Error in Plcl: badly shaped input')

        Plcl = np.zeros(np.shape(Te))
        for i,x in enumerate(Te):
            Plcl[i] = optimize.fsolve(zero, 80000., args=(x,qt[i]), xtol=1.e-10)
    else: # Bolton
        cp = cpd + qt*(cpv-cpd)
        R  = Rd  + qt*(Rv-Rd)
        pv = r2p(qt/(1.-qt),PPa)
        Tl = 55 + 2840./(3.5*np.log(TK) - np.log(pv/100.) - 4.805)
        Plcl = PPa * (Tl/TK)**(cp/R)

    return Plcl

def Zlcl(Plcl,T,P,qt,Z,):
    """ Returns the height of the LCL assuming temperature changes following a
    dry adiabat with vertical displacements from the height where the ambient
    temperature is measured.
	>>> Zlcl(300.,1020.,17)
	96007.495
    """
    Rd   = constants.dry_air_gas_constant
    Rv   = constants.water_vapor_gas_constant
    cpd  = constants.isobaric_dry_air_specific_heat
    cpv  = constants.isobaric_water_vapor_specific_heat

    cp = cpd + qt*(cpv-cpd)
    R  = Rd  + qt*(Rv-Rd)

    return T*(1. - (Plcl/P)**(R/cp)) * cp/earth_gravity + Z

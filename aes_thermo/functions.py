# -*- coding: utf-8 -*-
"""
Author: Bjorn Stevens (bjorn.stevens@mpimet.mpg.de)
"""
#
import numpy as np
import aes_thermo.constants 
from scipy import interpolate, optimize


es_default = 'analytic-liq'
c,h,kB,N_avo        = aes_thermo.constants.fundamental()
TvT,PvT,lvT,lfT,lsT = aes_thermo.constants.triple()
lv0,lf0,P0,T0       = aes_thermo.constants.standard()
Rd,Rv,cpd,cpv,cl,ci = aes_thermo.constants.rankine()
PvC,TvC             = aes_thermo.constants.critical()
eps1                =  aes_thermo.constants.eps1


def Planck(T,nu):
    """returns the Planck source function for the given temperature (K) and frequency (Hz)
    """
    return (2 * h * nu**3 / c**2) / (np.exp(h*nu/(kB*T))-1)

def es(T,es_formula=es_default):
    """ Returns the saturation vapor pressure of water over liquid or ice, or the minimum of the two,
    depending on the specificaiton of the state variable.  The calculation follows Wagner and Pruss (2002)
    fits (es[li]f) for saturation over planar liquid, and Wagner et al., 2011 for saturation over ice.  The choice
    choice of formulation was based on a comparision of many many formulae, among them those by Sonntag, Hardy,
    Romps, Murphy and Koop, and others (e.g., Bolton) just over liquid. The Wagner and Pruss and Wagner
    formulations were found to be the most accurate as cmpared to the IAPWS standard for warm temperatures,
    and the Wagner et al 2011 form is the IAPWS standard for ice.  Additionally an 'analytic' expression es[li]a
    for computations that require consisntency with assumption of cp's being constant can be selected.  The analytic
    expressions become identical to Romps in the case when the specific heats are adjusted to his suggested values.
    >>> es([273.16,290.])
    [611.65706974 1919.87719485]
    """

    def esif(T):
        a1 = -0.212144006e+2
        a2 =  0.273203819e+2
        a3 = -0.610598130e+1
        b1 =  0.333333333e-2
        b2 =  0.120666667e+1
        b3 =  0.170333333e+1
        theta = T/TvT
        return PvT * np.exp((a1*theta**b1 + a2 * theta**b2 + a3 * theta**b3)/theta)

    def eslf(T):
        vt = 1.-T/TvC
        return PvC * np.exp(TvC/T * (-7.85951783*vt + 1.84408259*vt**1.5 - 11.7866497*vt**3 + 22.6807411*vt**3.5 - 15.9618719*vt**4 + 1.80122502*vt**7.5))

    def esla(T):
        c1 = (cpv-cl)/Rv
        c2 = lvT/(Rv*TvT) - c1
        return PvT * np.exp(c2*(1.-TvT/x)) * (x/TvT)**c1

    def esia(T):
        c1 = (cpv-ci)/Rv
        c2 = lsT/(Rv*TvT) - c1
        return PvT * np.exp(c2*(1.-TvT/x)) * (x/TvT)**c1

    x = T

    if (es_formula == 'liq'):
        es = eslf(x)
    if (es_formula == 'ice'):
        es = esif(x)
    if (es_formula == 'mxd'):
        es = np.minimum(esif(x),eslf(x))
    if (es_formula == 'analytic-liq'):
        es = esla(x)
    if (es_formula == 'analytic-ice'):
        es = esia(x)
    if (es_formula == 'analytic-mxd'):
        es = np.minimum(esia(x),esla(x))

    return es

def phase_change_enthalpy(TK,fusion=False):
    """ Returns the enthlapy [J/g] of vaporization (default) of water vapor or
    (if fusion=True) the fusion anthalpy.  Input temperature can be in degC or Kelvin
    >>> phase_change_enthalpy(273.15)
    2500.8e3
    """

    if (fusion):
        el = lf0 + (cl-ci)*(TK-T0)
    else:
        el = lv0 + (cpv-cl)*(TK-T0)

    return el

def pp2sm(pv,p):
    """ Calculates specific mass from the partial and total pressure
    assuming both have same units and no condensate is present.  Returns value
    in units of kg/kg. checked 15.06.20
    >>> pp2sm(es(273.16),60000.)
    0.00636529
    """
    x   = eps1*pv/(p-pv)
    return x/(1+x)

def pp2mr(pv,p):
    """ Calculates mixing ratio from the partial and total pressure
    assuming both have same unitsa nd no condensate is present. Returns value
    in units of kg/kg. Checked 20.03.20
    """

    return eps1*pv/(p-pv)

def mr2pp(mr,p):
    """ Calculates partial pressure from mixing ratio and pressure, if mixing ratio
    units are greater than 1 they are normalized by 1000.
    checked 20.03.20
    """

    return mr*p/(eps1+mr)

def pseudo_theta_e(TK,PPa,qt,es_formula=es_default):
    """ Calculates pseudo equivalent potential temperature. following Bolton
    checked 31.07.20
    """

    rv = np.minimum(qt/(1.-qt), pp2mr(es(TK,es_formula),PPa))
    pv = mr2pp(rv,PPa)

    TL     = 55.0 + 2840./(3.5*np.log(TK) - np.log(pv/100.) - 4.805)
    theta_e = TK*(P0/PPa)**(0.2854*(1.0 - 0.28*rv)) * np.exp((3376./TL - 2.54)*rv*(1+0.81*rv))
    
    return(theta_e)

def theta_e(TK,PPa,qt,es_formula=es_default):
    """ Calculates equivalent potential temperature corresponding to Eq. 2.42 in the Clouds
    and Climate book.
    checked 19.03.20
    """

    ps = es(TK,es_formula)
    qs = (ps/(PPa-ps)) * eps1 * (1.0 - qt)
    qv = np.minimum(qt,qs)

    Re = (1.0-qt)*Rd
    R  = Re + qv*Rv
    pv = qv * (Rv/R) *PPa
    RH = pv/ps
    lv = phase_change_enthalpy(TK)
    cpe= cpd + qt*(cl-cpd)
    omega_e = RH**(-qv*Rv/cpe) * (R/Re)**(Re/cpe)
    theta_e = TK*(P0/PPa)**(Re/cpe)*omega_e*np.exp(qv*lv/(cpe*TK))

    return(theta_e)

def theta_es(TK,PPa,es_formula=es_default):
    """ Calculates equivalent potential temperature corresponding to Eq. 2.42 in the Clouds
    and Climate book but assuming that the water amount is just saturated
    checked 19.03.20
    """

    ps = es(TK,es_formula)
    xx = (ps/(PPa-ps)) * eps1
    qs = xx/(1.0+xx)

    Re = (1.0-qs)*Rd
    R  = Re + qs*Rv
    lv = phase_change_enthalpy(TK)
    cpe= cpd + qs*(cl-cpd)
    omega_e  = (R/Re)**(Re/cpe)
    theta_es = TK*(P0/PPa)**(Re/cpe)*omega_e*np.exp(qs*lv/(cpe*TK))

    return(theta_es)


def theta_l(TK,PPa,qt,es_formula=es_default):
#   """ Calculates liquid-water potential temperature.  Following Stevens and Siebesma
#   Eq. 2.44-2.45 in the Clouds and Climate book
#   """

    ps = es(TK,es_formula)
    qs = (ps/(PPa-ps)) * eps1 * (1. - qt)
    qv = np.minimum(qt,qs)
    ql = qt-qv

    R  = Rd*(1-qt) + qv*Rv
    Rl = Rd + qt*(Rv - Rd)
    cpl= cpd + qt*(cpv-cpd)
    lv = phase_change_enthalpy(TK)

    omega_l = (R/Rl)**(Rl/cpl) * (qt/(qv+1.e-15))**(qt*Rv/cpl)
    theta_l = (TK*(P0/PPa)**(Rl/cpl)) *omega_l*np.exp(-ql*lv/(cpl*TK))

    return(theta_l)

def theta_s(TK,PPa,qt,es_formula=es_default):
#   """ Calculates entropy potential temperature. This follows the formulation of Pascal
#   Marquet and ensures that parcels with different theta-s have a different entropy
#   """

    kappa = Rd/cpd
    e0    = es(T0,es_formula)
    Lmbd  = ((sv00 - Rv*np.log(e0/P0)) - (sd00 - Rd*np.log(1-e0/P0)))/cpd
    lmbd  = cpv/cpd - 1.
    eta   = 1/eps1
    delta = eps2
    gamma = kappa/eps1
    r0    = e0/(P0-e0)/eta

    ps = es(TK,es_formula)
    qs = (ps/(PPa-ps)) * eps1 * (1. - qt)
    qv = np.minimum(qt,qs)
    ql = qt-qv

    lv = phase_change_enthalpy(TK)

    R  = Rd + qv*(Rv - Rd)
    pv = qv * (Rv/R) *PPa
    RH = pv/ps
    rv = qv/(1-qv)

    x1 = 1
    x1 = (T/T0)**(lmbd*qt) * (P0/PPa)**(kappa*delta*qt) * (rv/r0)**(-gamma*qt) * RH**(gamma*ql)
    x2 = (1.+eta*rv)**(kappa*(1+delta*qt)) * (1+eta*r0)**(-kappa*delta*qt)
    theta_s = (TK*(P0/PPa)**(kappa)) * np.exp(-ql*lv/(cpd*TK)) * np.exp(qt*Lmbd) * x1 * x2

    return(theta_s)

def theta_rho(TK,PPa,qt,es_formula=es_default):
#   """ Calculates theta_rho as theta_l * (1+Rd/Rv qv - qt)
#   """

    ps = es(TK,es_formula)
    qs = (ps/(PPa-ps)) * (Rd/Rv) * (1. - qt)
    qv = np.minimum(qt,qs)
    theta_rho = theta_l(TK,PPa,qt,es_formula) * (1.+ qv/eps1 - qt)

    return(theta_rho)

def T_from_Te(Te,P,qt,es_formula=es_default):
    """ Given theta_e solves implicitly for the temperature at some other pressure,
    so that theta_e(T,P,qt) = Te
	>>> T_from_Te(350.,1000.,17)
	304.4761977
    """

    def zero(T,Te,P,qt):
        return  np.abs(Te-theta_e(T,P,qt,es_formula))
    return optimize.fsolve(zero,   280., args=(Te,P,qt), xtol=1.e-10)

def T_from_Tl(Tl,P,qt,es_formula=es_default):
    """ Given theta_e solves implicitly for the temperature at some other pressure,
    so that theta_e(T,P,qt) = Te
	>>> T_from_Tl(282.75436951,90000,20.e-3)
	290.00
    """
    def zero(T,Tl,P,qt):
        return  np.abs(Tl-theta_l(T,P,qt,es_formula))
    return optimize.fsolve(zero,   280., args=(Tl,P,qt), xtol=1.e-10)

def T_from_Ts(Ts,P,qt,es_formula=es_default):
    """ Given theta_e solves implicitly for the temperature at some other pressure,
    so that theta_e(T,P,qt) = Te
	>>> T_from_Tl(282.75436951,90000,20.e-3)
	290.00
    """
    def zero(T,Ts,P,qt):
        return  np.abs(Ts-theta_s(T,P,qt,es_formula))
    return optimize.fsolve(zero,   280., args=(Ts,P,qt), xtol=1.e-10)

def P_from_Te(Te,T,qt,es_formula=es_default):
    """ Given Te solves implicitly for the pressure at some temperature and qt
    so that theta_e(T,P,qt) = Te
	>>> P_from_Te(350.,305.,17)
	100464.71590478
    """
    def zero(P,Te,T,qt):
        return np.abs(Te-theta_e(T,P,qt,es_formula))
    return optimize.fsolve(zero, 90000., args=(Te,T,qt), xtol=1.e-10)

def P_from_Tl(Tl,T,qt,es_formula=es_default):
    """ Given Tl solves implicitly for the pressure at some temperature and qt
    so that theta_l(T,P,qt) = Tl
	>>> T_from_Tl(282.75436951,290,20.e-3)
	90000
    """
    def zero(P,Tl,T,qt):
        return np.abs(Tl-theta_l(T,P,qt,es_formula))
    return optimize.fsolve(zero, 90000., args=(Tl,T,qt), xtol=1.e-10)

def Plcl(TK,PPa,qt,es_formula=es_default,iterate=False):
    """ Returns the pressure [Pa] of the LCL.  The routine gives as a default the
    LCL using the Bolton formula.  If iterate is true uses a nested optimization to
    estimate at what pressure, Px and temperature, Tx, qt = qs(Tx,Px), subject to
    theta_e(Tx,Px,qt) = theta_e(T,P,qt).  This works for saturated air.
	>>> Plcl(300.,1020.,17)
	96007.495
    """

    def delta_qs(P,Te,qt,es_formula=es_default):
        TK = T_from_Te(Te,P,qt)
        ps = es(TK,es_formula)
        qs = (1./(P/ps-1.)) * eps1 * (1. - qt)
        return np.abs(qs/qt-1.)

    if (iterate):
        Te   = theta_e(TK,PPa,qt,es_formula)
        if scalar_input:
            Plcl = optimize.fsolve(delta_qs, 80000., args=(Te,qt), xtol=1.e-10)
            return np.squeeze(Plcl)
        else:
            if (scalar_input3):
                qx =np.empty(np.shape(Te)); qx.fill(np.squeeze(qt)); qt = qx
            elif len(Te) != len(qt):
                print('Error in Plcl: badly shaped input')

        Plcl = np.zeros(np.shape(Te))
        for i,x in enumerate(Te):
            Plcl[i] = optimize.fsolve(delta_qs, 80000., args=(x,qt[i]), xtol=1.e-10)
    else: # Bolton
        cp = cpd + qt*(cpv-cpd)
        R  = Rd  + qt*(Rv-Rd)
        pv = mr2pp(qt/(1.-qt),PPa)
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
    cp = cpd + qt*(cpv-cpd)
    R  = Rd  + qt*(Rv-Rd)

    return T*(1. - (Plcl/P)**(R/cp)) * cp/earth_grav + Z

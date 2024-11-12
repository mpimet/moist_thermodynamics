# -*- coding: utf-8 -*-
"""
Provides accurate thermodynamic functions for moist atmosphere

Author: Bjorn Stevens (bjorn.stevens@mpimet.mpg.de)
copygright, bjorn stevens Max Planck Institute for Meteorology, Hamburg

License: BSD-3C
"""

#
import numpy as np
from scipy import optimize
from scipy.integrate import ode

from . import constants as constants
from .saturation_vapor_pressures import es_default


def make_es_mxd(es_liq, es_ice):
    """Closure to construct a mixed form of the saturation vapor pressure

    To provide a single function that provides the saturation vapor pressure
    over ice when this is lower, and over water when this is lower, we use a
    closure function which accepts different choices of the individual saturation
    vapor pressures.

    Args:
        es_liq: function call for saturation vapor pressure over liquid
        es_ice: function call for saturation vapor pressure over ice

    Returns:
        function that selects the minimum of es_ice(T) and es_liq(T)
    """

    def es(T):
        return np.minimum(es_liq(T), es_ice(T))

    return es


def make_static_energy(hv0):
    """Closure function to construct moist static energies

    When including the effects of composition on the specific heat to calcuate the moist enthalpy, a
    constitutent part of the static energy different reference states can be adopted.  These reference
    states effectively weight the contribution of another invariant of the closed system (total water)
    differently, leading to moist static energies which give different weights to the thermal and phase
    change energies of the system.  The closure funciton allows one to construct one or the other of
    these static energies by choosing the approriate reference state vapor enthalpy, which then
    determines the reference enthalpies of the other phases given the reference state phase change
    enthalpies.  How the choices affec the construct of the static energy are outlined as follows:
        - hv0 = cpv*T0      -> frozen, liquid moist static energy
        - hv0 = ls0 + ci*T0 -> frozen moist static energy
        - hv0 = cpv*T0      -> liquid water static energy if qi= 0 (default if qv /= 0)
        - hv0 = lv0 + cl*T0 -> moist static energy if qi= 0.
        - qv=ql=q0=0        -> dry static energy (default)

    Args:
        hv0: reference vapor enthalpy

    Returns:
        h: a function for the moist static energy

    """

    def h(T, Z, qv=0, ql=0, qi=0):
        """Returns moist static energies given the closure

        This function returns the static energy subject to the vapor reference state enthalpy as
        given through the closure  I

        Args:
            T: temperature in kelvin
            Z: altitude (above mean sea-level) in meters
            qv: specific vapor mass
            ql: specific liquid mass
            qi: specific ice mass
        """

        qt = qv + ql + qi
        x = (
            T * (1.0 - qt) * constants.cpd
            + hv0 * qt
            + (T - constants.T0)
            * (qv * constants.cpv + ql * constants.cl + qi * constants.ci)
            - ql * constants.lv0
            - qi * constants.ls0
            + constants.gravity_earth * Z
        )
        return x

    return h


def planck(T, nu):
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
    kB = constants.boltzmann_constant
    return (2 * h * nu**3 / c**2) / (np.exp(h * nu / (kB * T)) - 1)


def vaporization_enthalpy(T, delta_cl=constants.delta_cl):
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
    T0 = constants.standard_temperature
    lv0 = constants.vaporization_enthalpy_stp
    return lv0 + delta_cl * (T - T0)


def sublimation_enthalpy(T, delta_ci=constants.delta_ci):
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
    T0 = constants.standard_temperature
    ls0 = constants.sublimation_enthalpy_stp
    return ls0 + delta_ci * (T - T0)


def partial_pressure_to_mixing_ratio(pp, p):
    """Returns the mass mixing ratio given the partial pressure and pressure

    >>> partial_pressure_to_mixing_ratio(es_default(300.),60000.)
    0.038901996260228
    """
    eps1 = constants.rd_over_rv
    return eps1 * pp / (p - pp)


def mixing_ratio_to_partial_pressure(r, p):
    """Returns the partial pressure (pp in units of p) from a gas' mixing ratio

    Args:
        r: mass mixing ratio (unitless)
        p: pressure in same units as desired return value


    >>> mixing_ratio_to_partial_pressure(2e-5,60000.)
    1.929375975915276
    """
    eps1 = constants.rd_over_rv
    return r * p / (eps1 + r)


def partial_pressure_to_specific_humidity(pp, p):
    """Returns the specific mass given the partial pressure and pressure.

    The specific mass can be written in terms of partial pressure and pressure as
    expressed here only if the gas quanta contains no condensate phases.  In this
    case the specific humidity is the same as the co-dryair specific mass. In
    situations where condensate is present one should instead calculate
    $q = r*(1-qt)$ which would require an additional argument

    >>> partial_pressure_to_specific_humidity(es_default(300.),60000.)
    0.03744529936439133
    """
    r = partial_pressure_to_mixing_ratio(pp, p)
    return r / (1 + r)


def saturation_partition(P, ps, qt):
    """Returns the water vapor specific humidity given saturation vapor presure

    When condensate is present the saturation specific humidity and the total
    specific humidity differ, and the latter weights the mixing ratio when
    calculating the former from the saturation mixing ratio.  In subsaturated air
    the vapor speecific humidity is just the total specific humidity

    """
    qs = partial_pressure_to_mixing_ratio(ps, P) * (1.0 - qt)
    return np.minimum(qt, qs)


def theta(T, P, qv=0.0, ql=0.0, qi=0.0):
    """Returns the potential temperature for an unsaturated moist fluid

    This expressed the potential temperature in away that makes it possible to account
    for the influence of the specific water mass (in different phases) to influence the
    adiabatic factor R/cp.  The default is the usualy dry potential temperature.

    Args:
        T: temperature in kelvin
        P: pressure in pascal
        qv: specific vapor mass
        ql: specific liquid mass
        qi: specific ice mass

    """
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    cpd = constants.isobaric_dry_air_specific_heat
    cpv = constants.isobaric_water_vapor_specific_heat
    cl = constants.liquid_water_specific_heat
    ci = constants.frozen_water_specific_heat
    P0 = constants.P0

    qd = 1.0 - qv - ql - qi
    kappa = (qd * Rd + qv * Rv) / (qd * cpd + qv * cpv + ql * cl + qi * ci)
    return T * (P0 / P) ** kappa


def theta_e_bolton(T, P, qt, es=es_default):
    """Returns the pseudo equivalent potential temperature.

    Following Eq. 43 in Bolton (1980) the (pseudo) equivalent potential temperature
    is calculated and returned by this function

    Args:
        T: temperature in kelvin
        P: pressure in pascal
        qt: specific total water mass
        es: form of the saturation vapor pressure to use

    Reference:
        Bolton, D. The Computation of Equivalent Potential Temperature. Monthly Weather
        Review 108, 1046–1053 (1980).
    """
    P0 = constants.standard_pressure
    p2r = partial_pressure_to_mixing_ratio
    r2p = mixing_ratio_to_partial_pressure

    rv = np.minimum(
        qt / (1.0 - qt), p2r(es(T), P)
    )  # mixing ratio of vapor (not gas Rv)
    pv = r2p(rv, P)

    TL = 55.0 + 2840.0 / (3.5 * np.log(T) - np.log(pv / 100.0) - 4.805)
    return (
        T
        * (P0 / P) ** (0.2854 * (1.0 - 0.28 * rv))
        * np.exp((3376.0 / TL - 2.54) * rv * (1 + 0.81 * rv))
    )


def theta_e(T, P, qt, es=es_default):
    """Returns the equivalent potential temperature

    Follows Eq. 11 in Marquet and Stevens (2022). The closed form solutionis derived for a
    Rankine-Kirchoff fluid (constant specific heats).  Differences arising from its
    calculation using more accurate expressions (such as the default) as opposed to less
    accurate, but more consistent, formulations are on the order of millikelvin

    Args:
        T: temperature in kelvin
        P: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure

    Reference:
        Marquet, P. & Stevens, B. On Moist Potential Temperatures and Their Ability to
        Characterize Differences in the Properties of Air Parcels. Journal of the Atmospheric
        Sciences 79, 1089–1103 (2022).
    """
    P0 = constants.standard_pressure
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    cpd = constants.isobaric_dry_air_specific_heat
    cl = constants.liquid_water_specific_heat
    lv = vaporization_enthalpy

    ps = es(T)
    qv = saturation_partition(P, ps, qt)

    Re = (1.0 - qt) * Rd
    R = Re + qv * Rv
    pv = qv * (Rv / R) * P
    RH = pv / ps
    cpe = cpd + qt * (cl - cpd)
    omega_e = RH ** (-qv * Rv / cpe) * (R / Re) ** (Re / cpe)
    theta_e = T * (P0 / P) ** (Re / cpe) * omega_e * np.exp(qv * lv(T) / (cpe * T))
    return theta_e


def theta_l(T, P, qt, es=es_default):
    """Returns the liquid-water potential temperature

    Follows Eq. 16 in Marquet and Stevens (2022). The closed form solutionis derived for a
    Rankine-Kirchoff fluid (constant specific heats).  Differences arising from its
    calculation using more accurate expressions (such as the default) as opposed to less
    accurate, but more consistent, formulations are on the order of millikelvin

    Args:
        T: temperature in kelvin
        P: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure

    Reference:
        Marquet, P. & Stevens, B. On Moist Potential Temperatures and Their Ability to
        Characterize Differences in the Properties of Air Parcels. Journal of the Atmospheric
        Sciences 79, 1089–1103 (2022).
    """
    P0 = constants.standard_pressure
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    cpd = constants.isobaric_dry_air_specific_heat
    cpv = constants.isobaric_water_vapor_specific_heat
    lv = vaporization_enthalpy

    ps = es(T)
    qv = saturation_partition(P, ps, qt)
    ql = qt - qv

    R = Rd * (1 - qt) + qv * Rv
    Rl = Rd + qt * (Rv - Rd)
    cpl = cpd + qt * (cpv - cpd)

    omega_l = (R / Rl) ** (Rl / cpl) * (qt / (qv + 1.0e-15)) ** (qt * Rv / cpl)
    theta_l = (T * (P0 / P) ** (Rl / cpl)) * omega_l * np.exp(-ql * lv(T) / (cpl * T))
    return theta_l


def theta_s(T, P, qt, es=es_default):
    """Returns the entropy potential temperature

    Follows Eq. 18 in Marquet and Stevens (2022). The closed form solutionis derived for a
    Rankine-Kirchoff fluid (constant specific heats).  Differences arising from its
    calculation using more accurate expressions (such as the default) as opposed to less
    accurate, but more consistent, formulations are on the order of millikelvin

    Args:
        T: temperature in kelvin
        P: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure

    Reference:
        Marquet, P. & Stevens, B. On Moist Potential Temperatures and Their Ability to
        Characterize Differences in the Properties of Air Parcels. Journal of the Atmospheric
        Sciences 79, 1089–1103 (2022).

        Marquet, P. Definition of a moist entropy potential temperature: application to FIRE-I
        data flights: Moist Entropy Potential Temperature. Q.J.R. Meteorol. Soc. 137, 768–791 (2011).
    """
    P0 = constants.standard_pressure
    T0 = constants.standard_temperature
    sd00 = constants.entropy_dry_air_satmt
    sv00 = constants.entropy_water_vapor_satmt
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    cpd = constants.isobaric_dry_air_specific_heat
    cpv = constants.isobaric_water_vapor_specific_heat
    eps1 = constants.rd_over_rv
    eps2 = constants.rv_over_rd_minus_one
    lv = vaporization_enthalpy

    kappa = Rd / cpd
    e0 = es(T0)
    Lmbd = ((sv00 - Rv * np.log(e0 / P0)) - (sd00 - Rd * np.log(1 - e0 / P0))) / cpd
    lmbd = cpv / cpd - 1.0
    eta = 1 / eps1
    delta = eps2
    gamma = kappa / eps1
    r0 = e0 / (P0 - e0) / eta

    ps = es(T)
    qv = saturation_partition(P, ps, qt)
    ql = qt - qv

    R = Rd + qv * (Rv - Rd)
    pv = qv * (Rv / R) * P
    RH = pv / ps
    rv = qv / (1 - qv)

    x1 = (
        (T / T0) ** (lmbd * qt)
        * (P0 / P) ** (kappa * delta * qt)
        * (rv / r0) ** (-gamma * qt)
        * RH ** (gamma * ql)
    )
    x2 = (1.0 + eta * rv) ** (kappa * (1.0 + delta * qt)) * (1.0 + eta * r0) ** (
        -kappa * delta * qt
    )
    theta_s = (
        (T * (P0 / P) ** (kappa))
        * np.exp(-ql * lv(T) / (cpd * T))
        * np.exp(qt * Lmbd)
        * x1
        * x2
    )
    return theta_s


def theta_es(T, P, es=es_default):
    """Returns the saturated equivalent potential temperature

    Adapted from Eq. 11 in Marquet and Stevens (2022) with the assumption that the gas quanta is
    everywhere just saturated.

    Args:
        T: temperature in kelvin
        P: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure

    Reference:
        Characterize Differences in the Properties of Air Parcels. Journal of the Atmospheric
        Sciences 79, 1089–1103 (2022).
    """
    P0 = constants.standard_pressure
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    cpd = constants.isobaric_dry_air_specific_heat
    cl = constants.liquid_water_specific_heat
    p2q = partial_pressure_to_specific_humidity
    lv = vaporization_enthalpy

    ps = es(T)
    qs = p2q(ps, P)

    Re = (1.0 - qs) * Rd
    R = Re + qs * Rv
    cpe = cpd + qs * (cl - cpd)
    omega_e = (R / Re) ** (Re / cpe)
    theta_es = T * (P0 / P) ** (Re / cpe) * omega_e * np.exp(qs * lv(T) / (cpe * T))
    return theta_es


def theta_rho(T, P, qt, es=es_default):
    """Returns the density liquid-water potential temperature

    calculates $\theta_\mathrm{l} R/R_\mathrm{d}$ where $R$ is the gas constant of a
    most fluid.  For an unsaturated fluid this is identical to the density potential
    temperature baswed on the two component fluid thermodynamic constants.

    Args:
        T: temperature in kelvin
        P: pressure in pascal
        qt: total water specific humidity (unitless)
        es: form of the saturation vapor pressure
    """
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant

    ps = es(T)
    qv = saturation_partition(P, ps, qt)
    theta_rho = theta_l(T, P, qt, es) * (1.0 - qt + qv * Rv / Rd)
    return theta_rho


def invert_for_temperature(f, f_val, P, qt, es=es_default):
    """Returns temperature for an atmosphere whose state is given by f, P and qt

        Infers the temperature from a state description (f,P,qt), where
        f(T,P,qt) = fval.  Uses a newton raphson method. This function only
        works on scalar quantities due to the state dependent number of iterations
        needed for convergence

    Args:
            f(T,P,qt): specified thermodynamice function, i.e., theta_l
            f_val: value of f for which T in kelvin is sought
            P: pressure in pascal
            qt: total water specific humidity (unitless)
            es: form of the saturation vapor pressure, passed to f

            >>> invert_for_temperature(theta_e, 350.,100000.,17.e-3)
            304.49714304228814
    """

    def zero(T, f_val):
        return f_val - f(T, P, qt, es=es)

    return optimize.newton(zero, 280.0, args=(f_val,))


def invert_for_pressure(f, f_val, T, qt, es=es_default):
    """Returns pressure for an atmosphere whose state is given by f, T and qt

        Infers the pressure from a state description (f,T,qt), where
        f(T,P,qt) = fval.  Uses a newton raphson method.  This function only
        works on scalar quantities due to the state dependent number of iterations
        needed for convergence.

    Args:
            f(T,P,qt): specified thermodynamice funcint, i.e., theta_l
            f_val: value of f for which P in Pa is sought
            T: temperature in kelvin
            qt: total water specific humidity (unitless)
            es: form of the saturation vapor pressure, passed to f

            >>> invert_for_pressure(theta_e, 350.,300.,17.e-3)
            94904.59555001547
    """

    def zero(P, f_val):
        return f_val - f(T, P, qt, es=es)

    return optimize.newton(zero, 80000.0, args=(f_val,))


def plcl(T, P, qt, es=es_default):
    """Returns the pressure at the lifting condensation level

    Calculates the lifting condensation level pressure using an interative solution under the
    constraint of constant theta-l. Exact to within the accuracy of the expression of theta-l
    which depends on the expression for the saturation vapor pressure

    Args:
        T: temperature in kelvin
        P: pressure in pascal
        qt: specific total water mass

        >>> plcl(300.,102000.,17e-3)
        array([95994.43612848])
    """

    def zero(P, Tl):
        p2r = partial_pressure_to_mixing_ratio
        T = invert_for_temperature(theta_l, Tl, P, qt, es=es)
        qs = p2r(es(T), P) * (1.0 - qt)
        return np.abs(qs / qt - 1.0)

    Tl = theta_l(T, P, qt, es=es)
    return optimize.fsolve(zero, 80000.0, args=(Tl,))


def plcl_bolton(T, P, qt):
    """Returns the pressure at the lifting condensation level

    Following Bolton (1980) the lifting condensation level pressure is derived from the state
    of an air parcel.  Usually accurate to within about 10 Pa, or about 1 m

    Args:
        T: temperature in kelvin
        P: pressure in pascal
        qt: specific total water mass

    Reference:
        Bolton, D. The Computation of Equivalent Potential Temperature. Monthly Weather
        Review 108, 1046–1053 (1980).

        >>> plcl_bolton(300.,102000.,17e-3)
        95980.41895404423
    """
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    cpd = constants.isobaric_dry_air_specific_heat
    cpv = constants.isobaric_water_vapor_specific_heat
    r2p = mixing_ratio_to_partial_pressure

    cp = cpd + qt * (cpv - cpd)
    R = Rd + qt * (Rv - Rd)
    pv = r2p(qt / (1.0 - qt), P)
    Tl = 55 + 2840.0 / (3.5 * np.log(T) - np.log(pv / 100.0) - 4.805)
    return P * (Tl / T) ** (cp / R)


def zlcl(Plcl, T, P, qt, z):
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
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    cpd = constants.isobaric_dry_air_specific_heat
    cpv = constants.isobaric_water_vapor_specific_heat
    g = constants.gravity_earth

    cp = cpd + qt * (cpv - cpd)
    R = Rd + qt * (Rv - Rd)
    return T * (1.0 - (Plcl / P) ** (R / cp)) * cp / g + z


def moist_adiabat(
    Tbeg,
    Pbeg,
    Pend,
    dP,
    qt,
    cc=constants.cl,
    lv=vaporization_enthalpy,
    es=es_default,
):
    """Returns the temperature and pressure by integrating along a moist adiabat

    Deriving the moist adiabats by assuming a constant moist potential temperature
    provides a Rankine-Kirchoff approximation to the moist adiabat.  If thermodynamic
    constants are allowed to vary with temperature then the intergation must be
    performed numerically, as outlined here for the case of constant thermodynamic
    constants and no accounting for the emergence of a solid condensage phase (ice).

    The introduction of this function allows one to estimate, for instance, the effect of
    isentropic freezing on the moist adiabat as follows:

    Tliq,Px= moist_adiabat(Tsfc,Psfc,Ptop,dP,qt,cc=constants.cl,lv=mt.vaporization_enthalpy,es = mt.es_mxd)
    Tice,Py= moist_adiabat(Tsfc,Psfc,Ptop,dP,qt,cc=constants.ci,lv=mt.sublimation_enthalpy ,es = mt.es_mxd)

    T  = np.ones(len(Tx))*constants.T0
    T[Tliq>constants.T0] = Tliq[Tliq>constants.T0]
    T[Tice<constants.T0] = Tice[Tice<constants.T0]

    which introduces an isothermal layer in the region where the fusion enthalpy is sufficient to do
    the expansional work

    Args:
        Tbeg: temperature at P0 in kelvin
        Pbeg: starting pressure in pascal
        Pend: pressure to which to integrate to in pascal
        dP:   integration step
        qt:   specific mass of total water
        es:   saturation vapor expression

    """

    def f(P, T, qt, cc, lv):
        Rd = constants.Rd
        Rv = constants.Rv
        cpd = constants.cpd
        cpv = constants.cpv

        qv = saturation_partition(P, es(T), qt)
        qc = qt - qv
        qd = 1.0 - qt

        R = qd * Rd + qv * Rv
        cp = qd * cpd + qv * cpv + qc * cc
        vol = R * T / P

        dX_dT = cp
        dX_dP = vol
        if qc > 0.0:
            beta_P = R / (qd * Rd)
            beta_T = beta_P * lv(T) / (Rv * T)

            dX_dT += lv(T) * qv * beta_T / T
            dX_dP *= 1.0 + lv(T) * qv * beta_P / (R * T)
        return dX_dP / dX_dT

    Tx = []
    Px = []
    r = ode(f).set_integrator("lsoda", atol=0.0001)
    r.set_initial_value(Tbeg, Pbeg).set_f_params(qt, cc, lv)
    while r.successful() and r.t > Pend:
        r.integrate(r.t + dP)
        Tx.append(r.y[0])
        Px.append(r.t)

    return np.asarray(Tx), np.asarray(Px)


moist_static_energy = make_static_energy(
    hv0=constants.lv0 + constants.cl * constants.T0
)

liquid_water_static_energy = make_static_energy(hv0=constants.cpv * constants.T0)

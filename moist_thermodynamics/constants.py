# -*- coding: utf-8 -*-
"""
Author: Bjorn Stevens (bjorn.stevens@mpimet.mpg.de)
"""

#
import numpy as np

c = speed_of_light = 299792458
kB = boltzmann_constant = 1.380649e-23
N_avo = avogadro_number = 6.02214076e23
h = planck_constant = 6.62607015e-34
G = gravitational_constant = 6.67430e-11

mass_earth = 5.9722e24
area_earth = 510065623e6
gravity_earth = 9.80665  # 4*np.pi*G*mass_earth/area_earth gives a value of 9.8203
radius_earth = np.sqrt(
    G * mass_earth / gravity_earth
)  # 6375.4, where standard equatorial radius is given as 6,378.1 km

Rstar = ideal_gas_constant = kB * N_avo
sigma = stefan_boltzmann_constant = 2 * (np.pi**5) * kB**4 / (15 * h**3 * c**2)

Tsfc = mean_earth_surface_temperature = 288.0  # Kelvin
Psfc = mean_earth_surface_pressure = 98443  # Pa
mass_atm = mass_earth_atmosphere = Psfc / gravity_earth  # kg/m^2

P0 = standard_pressure = 100000.0  # Pa
T0 = standard_temperature = 273.15  # K
P00 = standard_atmosphere_pressure = 101325.0  # Pa
T00 = standard_atmosphere_temperature = 298.15  # K
Ttriple = temperature_triple_point = 273.16  # K
#
# Based on Park et al (2004) Meteorlogia, O2 levels are declining as CO2 levels rise, but at a tiny rate.
#
x_ar = earth_atmosphere_ar_mass_fraction = 9.332e-3
x_o2 = earth_atmosphere_o2_mass_fraction = 0.20944
x_n2 = earth_atmosphere_n2_mass_fraction = 0.78083

x_co2 = earth_atmosphere_co2_mass_fraction = 0.415e-3
x_ch4 = earth_atmosphere_methane_mass_fraction = 772.0e-9
x_n2o = earth_atmosphere_nitrous_oxide_mass_fraction = 334.0e-9
x_o3 = earth_atmosphere_ozone_mass_fraction = 200.0e-9

#
# Based on Chase (1998) J Phys Chem Ref Data
#
m_ar = molar_mass_ar = 39.948
m_o2 = molar_mass_o2 = 15.9994 * 2
m_n2 = molar_mass_n2 = 14.0067 * 2
m_co2 = molar_mass_co2 = 44.011
m_h2o = molar_mass_h2o = 18.01528

m_ch4 = molar_mass_methane = m_co2 + 4 * m_h2o - 3.0 * m_o2
m_n2o = molar_mass_nitroush_oxide = m_n2 + m_o2 / 2
m_o3 = molar_mass_nitroush_ozone = m_o2 * 3 / 2

# molar mass of dry air
md = atomic_mass_dry_air = (
    x_ar * m_ar
    + x_o2 * m_o2
    + x_n2 * m_n2
    + x_co2 * m_co2
    + x_ch4 * m_ch4
    + x_n2o * m_n2o
    + x_o3 * m_o3
)

cp_ar = isobaric_specific_heat_capacity_a = 20.786  # 298.15K
cp_o2 = isobaric_specific_heat_capacity_o2 = (
    29.376  # isobaric_specific_heat_capacity_oxygen =  298.15K or 29.126 @ 200K
)
cp_n2 = isobaric_specific_heat_capacity_n2 = 29.124  # 298.15K or 29.107 @ 200K
cp_co2 = isobaric_specific_heat_capacity_co2 = 37.129  # 298.15K or 32.359 @ 200K
cp_h2o = isobaric_specific_heat_capacity_h2o = 33.349 + (33.590 - 33.349) / 98.15 * (
    T0 - 200
)  # Interpolated to T0 from Chase values (but not used)

s0_ar = entropy_argon_stp = 154.845  # J/mol*K
s0_o2 = entropy_oxygen_stp = 205.147  # J/mol*K
s0_n2 = entropy_nitrogen_stp = 191.609  # J/mol*K
s0_co2 = entropy_co2_stp = 213.795  # J/mol*K
s0_h2o = entropy_h2o_stp = 188.854  # J/mol*K

q_ar = argon_mass_mixing_fraction = x_ar * m_ar / md
q_o2 = o2_mass_mixing_fraction = x_o2 * m_o2 / md
q_n2 = n2_mass_mixing_fraction = x_n2 * m_n2 / md
q_co2 = water_vapor_mixing_fraction = x_co2 * m_co2 / md

Rd = dry_air_gas_constant = (
    (Rstar / md) * (x_ar + x_o2 + x_n2 + x_co2) * 1000.0
)  # J/kg/K
cpd = isobaric_dry_air_specific_heat = (
    (1.0 / md) * (x_ar * cp_ar + x_o2 * cp_o2 + x_n2 * cp_n2 + x_co2 * cp_co2) * 1000.0
)  # J/kg/K
sd0 = entropy_dry_air_stp = (
    (1.0 / md) * (x_ar * s0_ar + x_o2 * s0_o2 + x_n2 * s0_n2 + x_co2 * s0_co2) * 1000.0
)  # J/kg*K
sd00 = entropy_dry_air_satmt = sd0 + cpd * np.log(T0 / T00)
#
# cl and ci, especially ci, varies considerably with temperature.  Consider that
# cl = 4273 J/kg/K at 263 K decreases sharply to 4220 J/kg/K by 273 K and ever more slowly to
#      4179 J/kg/K at 313 K with most variation at lower temperatures
# ci = 1450 J/kg/K at 183 K and increases progressively to a value of 2132 J/kg/K at 278K
#
# At standard temperature and pressure they have the values
#    cl      = 4219.32   # ''
#    ci      = 2096.70   # ''
cpv = isobaric_water_vapor_specific_heat = 1865.01  # IAPWS97 at 273.15
cl = liquid_water_specific_heat = (
    4179.57  # IAPWS97 at 305 and P=0.1 MPa (gives a good fit for es over ice)
)
ci = frozen_water_specific_heat = (
    1905.43  # IAPWS97 at 247.065 and P=0.1 MPa (gives a good fit for es over ice)
)
delta_cl = cpv - cl
delta_ci = cpv - ci

lv0 = vaporization_enthalpy_stp = 2500.93e3  # IAPWS97 at 273.15
lf0 = melting_enthalpy_stp = 333.42e3  # ''
ls0 = sublimation_enthalpy_stp = lv0 + lf0  # ''

Rv = water_vapor_gas_constant = (Rstar / m_h2o) * 1000.0  # J/kg/K
sv00 = entropy_water_vapor_satmt = (s0_h2o / m_h2o) * 1000.0 + cpv * np.log(T0 / 298.15)

eps1 = rd_over_rv = Rd / Rv
eps2 = rv_over_rd_minus_one = Rv / Rd - 1.0

TvC = temperature_water_vapor_critical_point = (
    647.096  # Critical temperature [K] of water vapor
)
PvC = pressure_water_vapor_critical_point = (
    22.064e6  # Critical pressure [Pa] of water vapor
)

TvT = temperature_water_vapor_triple_point = (
    273.16  # Triple point temperature [K] of water
)
PvT = pressure_water_vapor_triple_point = 611.655
lvT = vaporization_enthalpy_triple_point = lv0 + (cpv - cl) * (TvT - T0)
lfT = melting_enthalpy_triple_point = lf0 + (cpv - ci) * (TvT - T0)
lsT = sublimation_enthalpy_triple_point = lvT + lfT

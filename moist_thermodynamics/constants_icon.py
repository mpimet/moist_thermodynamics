# -*- coding: utf-8 -*-
"""
Author: Bjorn Stevens (bjorn.stevens@mpimet.mpg.de)
"""

#
cpd = isobaric_dry_air_specific_heat = 1004.64  # J/kg/K
rd = dry_air_gas_constant = 287.04  # J/kg/K
cpv = isobaric_water_vapor_specific_heat = 1869.46  # J/kg/K
rv = water_vapor_gas_constant = 461.51  # J/kg/K
clw = liquid_water_specific_heat = (3.1733 + 1.0) * cpd  # J/kg/K
ci = forzen_water_specific_heat = 2108.00  # J/kg/K
g = earths_gravitational_acceleration = 9.80665  # m/s2
lv = vaporization_enthalpy_at_melting_point = 2.5008e6  # J/kg
ls = sublimation_enthalpy_at_melting_point = 2.8345e6  # J/kg
Tmelt = melting_point_temperature = 273.15  # Kelvin

cvd = isometric_dry_air_specific_heat = cpd - rd  # J/kg/K
cvv = isometric_water_vapor_specific_heat = cpv - rv  # J/kg/K
lf = melting_enthalpy = ls - lv  # J/kg

# -*- coding: utf-8 -*-
"""
Author: Bjorn Stevens (bjorn.stevens@mpimet.mpg.de)
"""
#

# dry air
cpd = 1004.64  # J/kg/K  specific heat at constant pressure
rd = 287.04    # J/kg/K  gas constant
cvd = cpd-rd   # J/kg/K  specific heat at constant volume

#h2o vapor
cpv = 1869.46  # J/kg/K  specific heat at constant pressure
rv = 461.51    # J/kg/K  gas constant
cvv = cpv-rv   # J/kg/K  specific heat at constant volume
#
rcpl = 3.1733        # cp_d / cp_l - 1
clw = (rcpl + 1.0) * cpd # specific heat capacity of liquid water
ci  = 2108.00
#
g = 9.80665    # m/s2    gravitational acceleration
lv = 2.5008e6  # J/kg    latent heat of vaporization
ls = 2.8345e6  # J/kg    latent heat of sublimation
lf = ls-lv     # J/kg    latent heat of fusion
Tmelt = 273.15


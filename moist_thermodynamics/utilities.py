import numpy as np

from . import functions as mt
from . import constants
from . import saturation_vapor_pressures as svp


def moist_adiabat_with_ice(
    P, Tx=301.0, qx=17e-5, Tmin=195.0, thx=mt.theta_l, integrate=False
):
    """Returns the liq-ice moist adiabat along a pressure dimension.

    Cacluates the moist adiabat with freezing.

    Args:
        P: pressure
        Tx: starting (value at P[0]) temperature
        qx: starting (value at P[0]) specific humidity
        Tmin: minimum temperature of adiabat
        thx: function to calculate isentrope if integrate = False
        integrate: determines if explicit integration will be used.
    """

    if integrate:
        T0 = constants.T0
        X = {
            "Tliq": {"cx": constants.cl, "lx": mt.vaporization_enthalpy},
            "Tice": {"cx": constants.ci, "lx": mt.sublimation_enthalpy},
        }
        es = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)
        for y in X.values():
            y["T"], Px = mt.moist_adiabat(
                Tx,
                P,
                qx,
                cc=y["cx"],
                lv=y["lx"],
                es=es,
            )
        TK = np.ones(len(Px)) * T0
        TK[X["Tliq"]["T"] > T0] = X["Tliq"]["T"][X["Tliq"]["T"] > T0]
        TK[X["Tice"]["T"] < T0] = X["Tice"]["T"][X["Tice"]["T"] < T0]
    else:
        TK = np.vectorize(mt.invert_for_temperature)(thx, thx(Tx, P[0], qx), P, qx)

    return np.maximum(TK, Tmin)

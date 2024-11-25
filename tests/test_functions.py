import pytest
import numpy as np

from moist_thermodynamics import functions as mtf
from moist_thermodynamics.saturation_vapor_pressures import es_default

es = es_default

data = [
    [285, 80000, 6.6e-3],
    [
        np.array([210, 285, 300]),
        np.array([20000, 80000, 102000]),
        np.array([0.2e-3, 6.6e-3, 17e-3]),
    ],
]


@pytest.mark.parametrize("T, p, qt", data)
def test_invert_T(T, p, qt):
    Tl = mtf.theta_l(T, p, qt, es=es)
    temp = mtf.invert_for_temperature(mtf.theta_l, Tl, p, qt, es=es)

    np.testing.assert_array_equal(temp, T)


@pytest.mark.parametrize("T, p, qt", data)
def test_plcl(T, p, qt):
    res = mtf.plcl(T, p, qt)
    if res.shape[0] > 1:
        print(res)
        assert np.all(res[:-1] - res[1:] < 0)
        assert abs(res[-1] - 95994.43612848) < 1

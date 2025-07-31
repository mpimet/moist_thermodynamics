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

stability_data = [
    [300, 315, 320],
    [0.016, 0.008, 0.004],
    [0, 2000, 4000],
    [0.01468398, 0.01185031, 0.00808245],
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


def test_n2():
    th = np.array(stability_data[0])
    qv = np.array(stability_data[1])
    z = np.array(stability_data[2])
    expected_n2 = np.array(stability_data[3])
    n2 = mtf.get_n2(th, qv, z)
    assert pytest.approx(n2, 1e-5) == expected_n2

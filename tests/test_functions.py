import pytest
import numpy as np

from moist_thermodynamics import functions as mtf
from moist_thermodynamics.saturation_vapor_pressures import es_default, liq_wagner_pruss

es = es_default

data = [
    [285, 80000, 6.6e-3],
    [
        np.array([210, 285, 300]),
        np.array([20000, 80000, 102000]),
        np.array([0.2e-5, 6.6e-3, 17e-3]),
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


@pytest.mark.parametrize(
    "Tbeg, Pbeg, Pend, dP, qt",
    [
        (300, 100000, 10000, 10000, 0.018),
        (
            np.array([300]),
            np.array([100000]),
            np.array([10000]),
            np.array([10000]),
            np.array([0.018]),
        ),
        (
            np.array([[300]]),
            np.array([[100000]]),
            np.array([[10000]]),
            np.array([[10000]]),
            np.array([[0.018]]),
        ),
    ],
)
def test_moist_adiabat(Tbeg, Pbeg, Pend, dP, qt):
    T, p = mtf.moist_adiabat(Tbeg, Pbeg, Pend, dP, qt, es=liq_wagner_pruss)
    assert T.shape == (9,)
    assert np.all(p == [100000, 90000, 80000, 70000, 60000, 50000, 40000, 30000, 20000])
    assert np.all(np.diff(T) < 0)


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


def test_hydrostatic_altitude():
    P = np.linspace(100000, 80000, 5)
    T = np.full(5, 299.5)
    q = np.full(5, 0.018)
    expected = np.array([0.0, 454.57587378, 933.73508251, 1440.28932562, 1977.56209712])
    z = mtf.hydrostatic_altitude_np(P, T, q)
    assert pytest.approx(z, 1e-5) == expected


@pytest.mark.parametrize("T, p, qt", data)
def test_rh_q(T, p, qt):
    """Test the conversion of relative humidity to specific humidity."""
    rh = mtf.specific_humidity_to_relative_humidity(qt, p, T, es=es_default)
    q = mtf.relative_humidity_to_specific_humidity(rh, p, T, es=es_default)
    np.testing.assert_allclose(qt, q, rtol=1e-5)
    assert np.all(rh >= 0) and np.all(rh <= 1)

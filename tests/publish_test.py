"""Check that basic features work.

Catch cases where e.g. files are missing so the import doesn't work. It is
recommended to check that e.g. assets are included."""

import moist_thermodynamics.functions as mtf
import moist_thermodynamics.constants as constants
import moist_thermodynamics.saturation_vapor_pressures as svp


def test_import():
    """Check that the import works."""
    assert mtf.planck is not None
    assert constants.c is not None
    assert svp.es_default is not None

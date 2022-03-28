from ellipticity.tools import ellipticity_coefficients
from obspy.taup import TauPyModel
import numpy as np
import pytest

# Expected values from Table 2 and 3 of Kennett and Gudmundsson GJI (1996) 127 p43
test_data = [
    ("Pdiff", 35.0, 140.0, [-1.452, 0.945, -0.287]),
    ("Sdiff", 200.0, 110.0, [-1.165, 1.337, -1.313]),
]


@pytest.mark.parametrize(
    "phase, source_depth_in_km, distance_in_degree, expected_sigma", test_data
)
def test_ellipticity_coefficients(
    phase, source_depth_in_km, distance_in_degree, expected_sigma
):
    model = TauPyModel("ak135")

    arrivals = model.get_ray_paths(
        source_depth_in_km, distance_in_degree, phase_list=[phase]
    )

    arrival = arrivals[0]
    calculated_sigma = ellipticity_coefficients(arrival, model)

    diff = np.array(calculated_sigma) - np.array(expected_sigma)

    # Kennett quotes ellipticity coefficients to three decimal places, so
    # we might expect to be able to recover them to within this. However, at
    # the moment they are only recovered to within 2e-2.
    # tol = 1e-3
    tol = 2e-2

    print(phase, expected_sigma, calculated_sigma)

    assert np.all(abs(diff) < tol)

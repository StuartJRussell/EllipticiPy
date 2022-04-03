from ellipticity.tools import ellipticity_coefficients
from obspy.taup import TauPyModel
import numpy as np
import pytest

# Expected values from ak135 reference tables (or for phases where these are wrong from our prior calculations)
test_data = [
    ("Pdiff", 100.0, 140.0, [-1.42, 0.94, -0.29]),
    ("Sdiff", 200.0, 110.0, [-1.16, 1.34, -1.32]),
    ("PcP", 700.0, 10.0, [-1.19, -0.24, -0.03]),
    ("ScP", 0.0, 45.0, [-1.49, -0.53, -0.45]),
    ("PP", 300.0, 70.0, [-0.69, -0.96, -0.72]),
    ("SS", 0.0, 130.0, [-1.11, -0.40, -2.59]),
    ("SKiKP", 500.0, 0.0, [-2.60, 0.00, 0.00]),
    ("PKIKP", 0.0, 180.0, [-2.70, 0.00, 0.00]),
    ("p", 200.0, 0.0, [-0.09, 0.00, 0.00]),
    ("s", 500.0, 5.0, [-0.41, -0.33, -0.02]),
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

    calculated_sigma = ellipticity_coefficients(arrivals)[0]

    diff = np.array(calculated_sigma) - np.array(expected_sigma)

    # Allow a tolerence of 2e-2
    tol = 3e-2

    print(phase, expected_sigma, calculated_sigma)

    assert np.all(abs(diff) < tol)

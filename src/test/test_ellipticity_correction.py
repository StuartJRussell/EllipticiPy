from ellipticity import ellipticity_correction
from obspy.taup import TauPyModel


def test_correction():
    """Test code against some expected values of corrections."""
    model = TauPyModel("ak135")

    source_depth_in_km = 124
    distance_in_degree = 65

    source_latitude = 45
    azimuth = 39
    phase_list = ["P", "S"]
    expected_correction = [-0.37, -0.70]  # seconds

    tol = 1e-2

    arrivals = model.get_ray_paths(source_depth_in_km, distance_in_degree, phase_list)
    calculated_correction = ellipticity_correction(
        arrivals, azimuth, source_latitude, model
    )

    for phase, expected, calculated in zip(
        phase_list, expected_correction, calculated_correction
    ):
        print(phase, expected, calculated)
        assert abs(expected - calculated) < tol

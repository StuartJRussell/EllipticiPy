from ellipticity import ellipticity_correction
from obspy.taup import TauPyModel
import pytest

test_data = [
    ("ak135", "P", 124, 65, 45, 39, -0.37),
    ("ak135", "S", 124, 65, 45, 39, -0.70),
]


@pytest.mark.parametrize(
    "model_name, phase, source_depth_in_km, distance_in_degree, source_latitude, azimuth, expected_correction",
    test_data,
)
def test_correction(
    model_name,
    phase,
    source_depth_in_km,
    distance_in_degree,
    source_latitude,
    azimuth,
    expected_correction,
):
    """Test code against some expected values of corrections."""
    model = TauPyModel(model_name)

    arrivals = model.get_ray_paths(source_depth_in_km, distance_in_degree, [phase])
    calculated_correction = ellipticity_correction(
        arrivals, azimuth, source_latitude, model
    )

    tol = 1e-2

    print(phase, expected_correction, calculated_correction)

    assert abs(expected_correction - calculated_correction) < tol

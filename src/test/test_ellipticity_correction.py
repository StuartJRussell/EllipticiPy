from ellipticity import ellipticity_correction
from obspy.taup import TauPyModel
import pytest

# Expected corrections are from prior calculation
test_data = [
    ("ak135", "P", 124., 65., 45., 39., -0.37),
    ("ak135", "S", 124., 65., 45., 39., -0.70),
    ("prem", "PKKP", 320., 90., 10., 15., -0.55),
    ("prem", "PcS", 10., 40., -50., 260., -0.41),
    ("iasp91", "sPKiKP", 540., 75., -80., 210., -1.16),
    ("iasp91", "SKSSKS", 400., 260., 0., 80., 3.48)
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
    calculated_correction = ellipticity_correction(arrivals, azimuth, source_latitude)

    tol = 1e-2

    print(phase, expected_correction, calculated_correction)

    assert abs(expected_correction - calculated_correction) < tol

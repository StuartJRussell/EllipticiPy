from ellipticity import ellipticity_correction
from obspy.taup import TauPyModel
import pytest

# Expected corrections are from prior calculation
test_data = [
    ("ak135", "P", 124.0, 65.0, 45.0, 39.0, -0.37),
    ("ak135", "S", 124.0, 65.0, 45.0, 39.0, -0.70),
    ("prem", "PKKP", 320.0, 90.0, 10.0, 15.0, -0.55),
    ("prem", "PcS", 10.0, 40.0, -50.0, 260.0, -0.41),
    ("iasp91", "sPKiKP", 540.0, 75.0, -80.0, 210.0, -1.16),
    ("iasp91", "SKSSKS", 400.0, 260.0, 0.0, 80.0, 3.48),
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
    calculated_correction = ellipticity_correction(arrivals, azimuth, source_latitude)[
        0
    ]

    tol = 1e-2

    print(phase, expected_correction, calculated_correction)

    assert abs(expected_correction - calculated_correction) < tol

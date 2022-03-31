from ellipticity.tools import model_epsilon, get_epsilon
from obspy.taup import TauPyModel


def test_epsilon():
    """Test the numerical integration for ellipticity (epsilon)"""
    model = TauPyModel("prem")
    model_epsilon(model.model)

    # Calculate ellipticity at the core mantle boundary (CMB)
    cmb_radius = 3480. # km
    calculated_epsilon = get_epsilon(model.model, cmb_radius)

    # Expected ellipticity for PREM at the CMB, see e.g. Huang et al GJI (2001) 146 p130
    expected_epsilon = 0.002547

    print("epsilon_CMB", expected_epsilon, calculated_epsilon)

    assert abs(calculated_epsilon - expected_epsilon) < 1e-6

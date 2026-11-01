import numpy as np
import pytest

from portfolio_simulation.monte_carlo import value_at_risk


def test_value_at_risk():
    assert value_at_risk(np.arange(100.0), 100.0, alpha=0.05) == pytest.approx(95.05)
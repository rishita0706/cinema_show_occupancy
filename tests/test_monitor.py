"""PSI tests on tiny data built inside the test."""
import numpy as np

from src.cinema.monitor import psi


def test_psi_same_data_is_zero():
    x = np.random.default_rng(0).normal(size=500)
    assert psi(x, x) < 1e-6


def test_psi_shifted_data_is_large():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 1000), rng.normal(3, 1, 1000)) > 0.25


def test_psi_works_on_discrete_values():
    old = np.array([9, 12, 15, 18, 21] * 40)
    new = np.array([9, 9, 9, 12, 12] * 40)
    assert psi(old, new) > 0
"""Unit tests for forecast NEES helpers without segment data."""

import numpy as np
from forecast_consistency import (
    CHI_SQUARED_95_TWO_DOF,
    assign_speed_bands,
    nees_values,
)


def nees(error_x, error_y, cov_xx, cov_xy, cov_yy):
    """Returns e^T S^-1 e for one 2D error and covariance."""
    errors = np.array([[error_x, error_y]], dtype = np.float64)
    return float(
        nees_values(errors,
                    np.array([cov_xx]),
                    np.array([cov_xy]),
                    np.array([cov_yy]))[0])


def test_speed_band_edges():
    speeds = np.array([0.2, 0.5, 4.9, 5.0, 20.0])
    assert assign_speed_bands(speeds).tolist() == [0, 1, 1, 2, 3]


def test_nees_identity():
    assert nees(1.0, 0.0, 1.0, 0.0, 1.0) == 1.0


def test_nees_diagonal():
    assert nees(2.0, 0.0, 4.0, 0.0, 1.0) == 1.0


def test_nees_correlated():
    assert nees(1.0, 1.0, 2.0, 1.0, 2.0) == 2.0 / 3.0


def test_nees_monte_carlo_mean_near_two():
    count = 1000
    covariance = np.array([[2.5, 0.4], [0.4, 1.2]], dtype = np.float64)
    samples = np.random.default_rng(0).multivariate_normal(
        np.zeros(2),
        covariance,
        size = count,
    )
    draws = nees_values(
        samples,
        np.full(count, covariance[0, 0]),
        np.full(count, covariance[0, 1]),
        np.full(count, covariance[1, 1]),
    )
    mean = float(np.mean(draws))
    standard_error = float(np.std(draws, ddof = 1) / np.sqrt(count))
    assert abs(mean - 2.0) <= 5.0 * standard_error
    assert float(np.mean(draws <= CHI_SQUARED_95_TWO_DOF)) > 0.9

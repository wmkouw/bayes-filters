"""
Implementations of pure, extended and unscented Kalman filters.

Algorithms based on Ch.5 of Bayesian Filtering & Smoothing (Särkka, 2014)

Wouter Kouw
04-06-2020
"""

using Distributions
using Random

function kalman_filter(observations)
    """
    Kalman filter (Th. 4.2)

    This filter is built for a linear Gaussian dynamical system with known
    transition coefficients, process and measurement noise.
    """

    # Time horizon
    time_horizon = length(observations)


end

function extended_kalman_filter(observations)
    """
    Extended Kalman filter with additive noise (Alg. 5.4)

    This filter is built for a linear Gaussian dynamical system with known
    transition coefficients, process and measurement noise.
    """

    # Time horizon
    time_horizon = length(observations)

    return
end

function unscented_kalman_filter(observations)
    """
    Unscented Kalman filter with additive noise (Alg. 5.14)

    This filter is built for a linear Gaussian dynamical system with unknown
    transition function, but with known process and measurement noise.
    """

    # Time horizon
    time_horizon = length(observations)

    return
end

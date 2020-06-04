"""
Implementations of pure, extended and unscented Kalman filters.

Algorithms based on Ch.5 of Bayesian Filtering & Smoothing (Särkka, 2014)

Wouter Kouw
04-06-2020
"""

using Distributions
using Random

function kalman_filter(observations,
                       transition_matrix,
                       emission_matrix,
                       process_noise,
                       measurement_noise,
                       state0)
    """
    Kalman filter (Th. 4.2)

    This filter is built for a linear Gaussian dynamical system with known
    transition coefficients, process and measurement noise.
    """

    # Dimensionality
    Dx = size(process_noise,1)
    Dy = size(measurement_noise,1)

    # Recast process noise to matrix
    if Dx == 1
        if typeof(process_noise) != Array{Float64,2}
            process_noise = reshape([process_noise], 1, 1)
        end
        if typeof(measurement_noise) != Array{Float64,2}
            measurement_noise = reshape([measurement_noise], 1, 1)
        end
    end

    # Time horizon
    time_horizon = length(observations)

    # Initialize estimate arrays
    mx = zeros(Dx, time_horizon)
    Px = zeros(Dx, Dx, time_horizon)

    # Initial state prior
    m_0, P_0 = state0

    # Start previous state variable
    m_tmin = m_0
    P_tmin = P_0

    for t = 1:time_horizon

        # Prediction step
        m_t_pred = transition_matrix*m_tmin
        P_t_pred = transition_matrix*P_tmin*transition_matrix' .+ process_noise

        # Update step
        v_t = observations[:,t] .- emission_matrix*m_t_pred
        S_t = emission_matrix*P_t_pred*emission_matrix' .+ measurement_noise
        K_t = P_t_pred*emission_matrix'*inv(S_t)
        m_t = m_t_pred .+ K_t*v_t
        P_t = P_t_pred .- K_t*S_t*K_t'

        # Store estimates
        mx[:,t] = m_t
        Px[:,:,t] = P_t

        # Update previous state variable
        m_tmin = m_t
        P_tmin = P_t
    end
    return mx, Px
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

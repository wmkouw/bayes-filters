"""
Set of functions for generating signals.
"""

function random_walk(process_noise,
                    measurement_noise,
                    mean_state0,
                    precision_state0;
                    time_horizon=100)
    "Generate data according to a random walk"

    # Dimensionality
    d = size(process_noise, 1)

    # Preallocate
    y = zeros(d, time_horizon)
    x = zeros(d, time_horizon+1)

    # Initialize state
    x[:, 1] = sqrt(inv(precision_state0))*randn(d) + mean_state0

    for t = 1:time_horizon

        # Evolve state
        x[:, t+1] = sqrt(process_noise)*randn(d) + x[:, t]

        # Observe
        y[:, t] = sqrt(measurement_noise)*randn(d) + x[:, t+1]

    end
    return y, x
end

function LGDS(transition_matrix,
              emission_matrix,
              process_noise,
              measurement_noise,
              prior_params;
              time_horizon=100)
    "Generate data according to a linear Gaussian dynamical system"

    # Dimensionality
    d = size(transition_matrix, 1)

    # Preallocate
    x = zeros(d, time_horizon+1)
    y = zeros(d, time_horizon)

    # Prior parameters
    mean0, var0 = prior_params

    # Initialize state
    x[:, 1] = sqrt(var0)*randn(d) .+ mean0

    for t = 1:time_horizon

        # Evolve state
        x[:, t+1] = sqrt(process_noise)*randn(d) .+ transition_matrix*x[:,t]

        # Observe
        y[:, t] = sqrt(measurement_noise)*randn(d) .+ emission_matrix*x[:,t+1]

    end
    return y, x
end

function NLGDS(transition_function,
               emission_function,
               process_noise,
               measurement_noise,
               state0_params;
               time_horizon=100)
    "Generate data according to a nonlinear Gaussian dynamical system"

    # Dimensionality
    d = size(process_noise, 1)

    # Preallocate
    x = zeros(d, time_horizon+1)
    y = zeros(d, time_horizon)

    # Prior parameters
    mean0, var0 = state0_params

    # Initialize state
    x[:, 1] = sqrt(var0)*randn(d) .+ mean0

    for t = 1:time_horizon

        # Evolve state
        x[:, t+1] = sqrt(process_noise)*randn(d) .+ transition_function(x[:,t])

        # Observe
        y[:, t] = sqrt(measurement_noise)*randn(d) .+ emission_function(x[:,t+1])

    end
    return y, x
end

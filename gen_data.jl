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
    y = zeros(time_horizon, d)
    x = zeros(time_horizon+1, d)

    # Initialize state
    x[1, :] = randn(d)./sqrt(precision_state0) .+ mean_state0

    for t = 1:time_horizon

        # Evolve state
        x[t+1, :] = randn(d).*sqrt(process_noise) + x[t, :]

        # Observe
        y[t, :] = randn(d).*sqrt(measurement_noise) + x[t+1, :]

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
    x = zeros(time_horizon+1, d)
    y = zeros(time_horizon, d)

    # Prior parameters
    mean0, var0 = prior_params

    # Initialize state
    x[1, :] = randn(d)./sqrt(var0) .+ mean0

    for t = 1:time_horizon

        # Evolve state
        x[t+1, :] = randn(d).*sqrt(process_noise) + transition_matrix*x[t, :]

        # Observe
        y[t, :] = randn(d).*sqrt(measurement_noise) + emission_matrix*x[t+1, :]

    end
    return y, x
end

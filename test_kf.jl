using Distributions
using Plots

include("./gen_data.jl")
include("./kalman_filters.jl")

"""Experimental parameters"""

# Number of sigma points
N = 10

# Time horizon
T = 100

# Experimental parameters
transition_coeffs = 1.0
emission_coeffs = 1.0
transition_function(x) = .1*x.^3
emission_function(x) = x

# Noises
process_noise = 0.4
measurement_noise = 0.2

# Prior state
state0 = [0., 1e-12]

# Generate signal
observations, states = LGDS(transition_coeffs,
                            emission_coeffs,
                            process_noise,
                            measurement_noise,
                            state0;
                            time_horizon=T)

# Check signal visually
scatter(1:T, observations, color="blue", label="observations")
plot!(1:T, states[2:end], color="red", label="states")

"""Basic Kalman filter"""

# Call filter
include("./kalman_filters.jl")
mx, Px = kalman_filter(observations,
                       transition_coeffs,
                       emission_coeffs,
                       process_noise,
                       measurement_noise,
                       state0)

# Visualize estimates
scatter(1:T, observations[1,:], color="blue", label="observations")
plot!(1:T, states[1,2:end], color="red", label="latent states")
plot!(1:T, mx[:], color="purple", label="inferred")
plot!(1:T, mx[:],
      ribbon=[sqrt.(Px[1,1,:]), sqrt.(Px[1,1,:])],
      color="purple", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig("./kalman-filters/viz/LGDS_kf.png")

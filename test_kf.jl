using Distributions
using Plots

include("./gen_data.jl")
include("./kalman_filters.jl")

"""Experimental parameters"""

# Number of sigma points
N = 10

# Time horizon
T = 50

# Experimental parameters
transition_coeffs = 1.0
emission_coeffs = 1.0

# Noises
process_noise = 0.4
measurement_noise = 0.2

# Prior state
state0 = ([0.], reshape([1e-12], 1,1))

"Generate data"

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

"Generate data"

# Nonlinearities
transition_function(x) = .01*x.^3 + 0.5*x
emission_function(x) = x

# Noises
process_noise = reshape([0.1], 1,1)
measurement_noise = reshape([0.2], 1,1)

# Generate signal
observations, states = NLGDS(transition_function,
                             emission_function,
                             process_noise,
                             measurement_noise,
                             state0;
                             time_horizon=T)

# Check signal visually
scatter(1:T, observations, color="blue", label="observations")
plot!(1:T, states[2:end], color="red", label="states")

"""Basic Kalman filter"""

# Call linear filter
mx1, Px1 = kalman_filter(observations,
                         transition_coeffs,
                         emission_coeffs,
                         process_noise,
                         measurement_noise,
                         state0)

# Call nonlinear filter
mx2, Px2 = unscented_kalman_filter(observations,
                                   transition_function,
                                   emission_function,
                                   process_noise,
                                   measurement_noise,
                                   state0,
                                   α=1., κ=0., β=2.)

# Visualize estimates
p1 = scatter(1:T, observations[1,:], color="blue", label="observations")
plot!(1:T, states[1,2:end], color="red", label="latent states")
plot!(1:T, mx1[:], color="purple", label="linear")
plot!(1:T, mx1[:],
      ribbon=[sqrt.(Px1[1,1,:]), sqrt.(Px1[1,1,:])],
      color="purple", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig(p1, "./kalman-filters/viz/NLGDS_kf.png")

p2 = scatter(1:T, observations[1,:], color="blue", label="observations")
plot!(1:T, states[1,2:end], color="red", label="latent states")
plot!(1:T, mx2[:], color="green", label="unscented")
plot!(1:T, mx2[:],
      ribbon=[sqrt.(Px2[1,1,:]), sqrt.(Px2[1,1,:])],
      color="green", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig(p2, "./kalman-filters/viz/NLGDS_ukf.png")

plot(p1, p2, layout=(2,1), size=(2000,1200))
savefig("./kalman-filters/viz/NLGDS_kf-v-ukf.png")

using LinearAlgebra
using Distributions
using Plots

include("../util/gen_data.jl")
include("../kalman_filters.jl")

"""Experimental parameters"""

# Number of sigma points
N = 10

# Time horizon
T = 100

# Experimental parameters
transition_coeffs = 1.0
emission_coeffs = 1.0

# Noises
process_noise = 0.5
measurement_noise = 0.1

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
include("../kalman_filters.jl")
mx, Px = kalman_filter(observations,
                       transition_coeffs,
                       emission_coeffs,
                       process_noise,
                       measurement_noise,
                       state0)

# Visualize estimates
scatter(1:T, observations[1,:], color="black", label="observations")
plot!(1:T, states[1,2:end], color="red", label="latent states")
plot!(1:T, mx[:], color="purple", label="inferred")
plot!(1:T, mx[:],
      ribbon=[sqrt.(Px[1,1,:]), sqrt.(Px[1,1,:])],
      color="purple", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig("figures/LGDS_kf.png")

"Generate data"

# Nonlinearities
transition_function(x) = .1*x.^3 + 0.5*x
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
scatter(1:T, observations, color="black", label="observations")
plot!(1:T, states[2:end], color="red", label="states")

"""Nonlinear Kalman filter"""

# Call basic filter
mx1, Px1 = kalman_filter(observations,
                         transition_coeffs,
                         emission_coeffs,
                         process_noise,
                         measurement_noise,
                         state0)

# Visualize estimates
p1 = scatter(1:T, observations[1,:], color="black", label="observations", size=(1200,500))
plot!(1:T, states[1,2:end], color="red", label="latent states")
plot!(1:T, mx1[:], color="purple", label="linear")
plot!(1:T, mx1[:],
      ribbon=[sqrt.(Px1[1,1,:]), sqrt.(Px1[1,1,:])],
      color="purple", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig(p1, "figures/NLGDS_KF.png")

# Call extended filter with first-order Taylor
mx2, Px2 = extended_kalman_filter(observations,
                                  transition_function,
                                  emission_function,
                                  process_noise,
                                  measurement_noise,
                                  state0)

p2 = scatter(1:T, observations[1,:], color="black", label="observations", size=(1200,500))
plot!(1:T, states[1,2:end], color="red", label="latent states")
plot!(1:T, mx2[:], color="blue", label="extended")
plot!(1:T, mx2[:],
      ribbon=[sqrt.(Px2[1,1,:]), sqrt.(Px2[1,1,:])],
      color="blue", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig(p2, "figures/NLGDS_E1KF.png")

# Call extended filter with second-order Taylor
mx3, Px3 = extended_kalman_filter(observations,
                                  transition_function,
                                  emission_function,
                                  process_noise,
                                  measurement_noise,
                                  state0,
                                  second_order=true)

p3 = scatter(1:T, observations[1,:], color="black", label="observations", size=(1200,500))
plot!(1:T, states[1,2:end], color="red", label="latent states")
plot!(1:T, mx3[:], color="cyan", label="extended")
plot!(1:T, mx3[:],
    ribbon=[sqrt.(Px3[1,1,:]), sqrt.(Px3[1,1,:])],
    color="cyan", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig(p3, "figures/NLGDS_E2KF.png")

# Call unscented filter
mx4, Px4 = unscented_kalman_filter(observations,
                                   transition_function,
                                   emission_function,
                                   process_noise,
                                   measurement_noise,
                                   state0,
                                   α=1., κ=0., β=2.)

p4 = scatter(1:T, observations[1,:], color="black", label="observations", size=(1200,500))
plot!(1:T, states[1,2:end], color="red", label="latent states")
plot!(1:T, mx4[:], color="green", label="unscented")
plot!(1:T, mx4[:],
      ribbon=[sqrt.(Px4[1,1,:]), sqrt.(Px4[1,1,:])],
      color="green", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig(p4, "figures/NLGDS_UKF.png")

plot(p1, p2, p3, p4, layout=(4,1), size=(2000,1600))
savefig("figures/NLGDS_filters.png")

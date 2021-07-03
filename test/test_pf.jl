using Distributions
using Plots
pyplot()

include("gen_data.jl")
include("particle_filters.jl")

"""Experimental parameters"""

# Visualization options
viz_particles = true

# Number of particles
N = 20

# Time horizon
T = 100

# Experimental parameters
transition = 1.0
emission = 1.0
process_noise = 0.9
measurement_noise = 0.3

# Prior state
prior = [0, 1]

# Generate signal
observations, latent_states = LGDS(transition,
                                   emission,
                                   process_noise,
                                   measurement_noise,
                                   prior;
                                   time_horizon=T)

# Check signal visually
scatter(1:T, observations, color="blue", label="observations")
plot!(1:T, latent_states[2:end], color="red", label="latent states")

"""SIS filter"""

# Nominal distributions
p_transition(a) = Normal(a, sqrt(process_noise))
p_likelihood(b) = Normal(b, sqrt(measurement_noise))

# Call SIS filter
samples, weights = sequential_importance_sampling(observations,
                                                  p_transition,
                                                  p_likelihood,
                                                  num_particles=N,
                                                  imp_processp=inv(process_noise),
                                                  imp_measurementp=inv(measurement_noise))

# Generate posterior distribution
posterior_mean = sum(weights .* samples, dims=2)
posterior_var = sum(weights .* (broadcast(-, samples, posterior_mean)).^2, dims=2)

# Visualize particles
scatter(1:T, observations, color="blue", label="observations")
plot!(1:T, latent_states[2:end], color="red", label="latent states")
plot!(1:T, posterior_mean[2:end], color="purple", label="inferred")
plot!(1:T, posterior_mean[2:end],
      ribbon=[sqrt.(posterior_var[2:end]), sqrt.(posterior_var[2:end])],
      color="purple", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig(pwd()*"/viz/sis.png")

if viz_particles

    # Show sample spreads
    scatter(1:T, observations, color="blue", label="observations")
    for t = 2:T
        scatter!((t-1).*ones(1,N), samples[t,:], markersize=weights[t,:]*50, markercolor="purple", label="")
    end
    savefig(pwd()*"/viz/sis_particles.png")
end

"""SIR filter"""

# Nominal distributions
p_transition(a) = Normal(a, sqrt(process_noise))
p_likelihood(b) = Normal(b, sqrt(measurement_noise))

# Call filter
samples, weights = sequential_importance_resampling(observations,
                                                    p_transition,
                                                    p_likelihood,
                                                    num_particles=N,
                                                    imp_processp=inv(process_noise),
                                                    imp_measurementp=inv(measurement_noise))

# Generate posterior distribution
posterior_mean = sum(weights .* samples, dims=2)
posterior_var = sum(weights .* (broadcast(-, samples, posterior_mean)).^2, dims=2)

# Visualize particles
scatter(1:T, observations, color="blue", label="observations")
plot!(1:T, latent_states[2:end], color="red", label="latent states")
plot!(1:T, posterior_mean[2:end], color="purple", label="inferred")
plot!(1:T, posterior_mean[2:end],
    ribbon=[sqrt.(posterior_var[2:end]), sqrt.(posterior_var[2:end])],
    color="purple", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig(pwd()*"/viz/sir.png")

if viz_particles

    # Show sample spreads
    scatter(1:T, observations, color="blue", label="observations")
    for t = 2:T
        scatter!((t-1).*ones(1,N), samples[t,:], markersize=weights[t,:]*50, markercolor="purple", label="")
    end
    savefig(pwd()*"/viz/sir_particles.png")
end

"""Bootstrap filter"""

# Nominal distributions
p_transition(a) = Normal(a, sqrt(process_noise))
p_likelihood(b) = Normal(b, sqrt(measurement_noise))

# Call filter
samples, weights = bootstrap_filter(observations,
                                    p_transition,
                                    p_likelihood,
                                    num_particles=N)

# Generate posterior distribution
posterior_mean = sum(weights .* samples, dims=2)
posterior_var = sum(weights .* (broadcast(-, samples, posterior_mean)).^2, dims=2)

# Visualize particles
scatter(1:T, observations, color="blue", label="observations")
plot!(1:T, latent_states[2:end], color="red", label="latent states")
plot!(1:T, posterior_mean[2:end], color="purple", label="inferred")
plot!(1:T, posterior_mean[2:end],
    ribbon=[sqrt.(posterior_var[2:end]), sqrt.(posterior_var[2:end])],
    color="purple", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig(pwd()*"/viz/bootstrap.png")

if viz_particles

    # Show sample spreads
    scatter(1:T, observations, color="blue", label="observations")
    for t = 2:T
        scatter!((t-1).*ones(1,N), samples[t,:], markersize=weights[t,:]*50, markercolor="purple", label="")
    end
    savefig(pwd()*"/viz/bootstrap_particles.png")
end

"""RB-SIR filter"""

# Nominal distributions
α = 0.5
imp_pp = inv(process_noise)
imp_mp = inv(measurement_noise)
nominal_dynamics(u) = Normal(u, sqrt(α))
importance_dist(u,y) = Normal(inv(imp_pp + imp_mp)*(imp_pp*u + imp_mp*y), sqrt(inv(imp_pp + imp_mp)))

# Functions
A(x) = x
Q(x) = x^2
H(x) = x
R(x) = x^2

# Call filter
samples, weights, means, cvars = rao_blackwellized_particle_filter(observations,
                                                                   nominal_dynamics,
                                                                   importance_dist,
                                                                   A, Q, H, R,
                                                                   num_particles=N,
                                                                   resampling_threshold=0.5)

# Generate posterior distribution
posterior_mean = sum(weights .* samples .* means, dims=2)
posterior_cvar = sum(weights .* samples.^2 .* cvars, dims=2)

# Visualize particles
scatter(1:T, observations, color="blue", label="observations")
plot!(1:T, latent_states[2:end], color="red", label="latent states")
plot!(1:T, posterior_mean[2:end], color="purple", label="inferred")
plot!(1:T, posterior_mean[2:end],
      ribbon=[sqrt.(posterior_cvar[2:end]), sqrt.(posterior_cvar[2:end])],
      color="purple", alpha=0.1, label="")
xlabel!("time (t)")
ylabel!("signal")
savefig(pwd()*"/viz/rb-sir.png")

if viz_particles

    # Show sample spreads
    scatter(1:T, observations, color="blue", label="observations")
    for t = 2:T
        scatter!((t-1).*ones(1,N), samples[t,:], markersize=weights[t,:]*50, markercolor="purple", label="")
    end
    savefig(pwd()*"/viz/rb-sir_particles.png")
end

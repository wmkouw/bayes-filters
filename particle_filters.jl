"""
Bayesian filters with monte carlo sampling for aproximating posterior distributions.
"""

using Distributions
using Random


function sequential_importance_sampling(observations,
                                        nominal_state_transition,
                                        nominal_likelihood;
                                        imp_processp=1.0,
                                        imp_measurementp=1.0,
                                        num_particles=100)
    """
    Sequential importance sampling (Alg. 7.2)

    This filter is built for the following model:
        p(x_k | x_k-1) ~ nominal_state_transition(x_k-1)
        p(y_k | x_k) ~ nominal_likelihood(x_k)
        π(x_k | x_k-1) ~ N(x_k-1, imp_processp^-1)
        π(y_k | x_k) ~ N(x_k, imp_measurementp^-1)
    """

    # Time horizon
    time_horizon = length(observations)

    # Preallocate importance sample array
    samples = zeros(time_horizon+1, num_particles)

    # Preallocate importance weight array
    weights = zeros(time_horizon+1, num_particles)

    # Initialize importance weight
    weights[1,:] = ones(1, num_particles) / num_particles

    # Draw from prior
    samples[1,:] = rand(Normal(0, 1), num_particles)

    for t = 2:time_horizon+1

        for i = 1:num_particles

            "Update importance distribution"

            # Update parameters
            impp_precision = imp_processp + imp_measurementp
            impp_mean = inv(impp_precision)*(imp_processp*samples[t-1, i] + imp_measurementp*observations[t-1])

            # Define new importance posterior distribution
            pi_ti = Normal(impp_mean, sqrt(inv(impp_precision)))

            # Draw new importance sample
            samples[t, i] = rand(pi_ti, 1)[1]

            "Update importance weights"

            # Evaluate nominal distributions
            px = pdf(nominal_state_transition(samples[t-1, i]), samples[t, i])
            py = pdf(nominal_likelihood(samples[t, i]), observations[t-1])

            # Evaluate probability of importance sample under importance dist.
            pπ = pdf(pi_ti, samples[t,i])

            # Update weight
            weights[t, i] = weights[t-1,i]*py*px/pπ

        end

        # Normalize weights
        weights[t,:] = weights[t,:] / sum(weights[t,:])
    end
    return (samples, weights)
end

function sequential_importance_resampling(observations,
                                          nominal_state_transition,
                                          nominal_likelihood;
                                          imp_processp=1.0,
                                          imp_measurementp=1.0,
                                          num_particles=100,
                                          resampling_threshold=0.5)
    """
    Sequential importance resampling (Alg. 7.4)

    This filter is built for the following model:
        p(x_k | x_k-1) ~ nominal_state_transition(x_k-1)
        p(y_k | x_k) ~ nominal_likelihood(x_k)
        π(x_k | x_k-1) ~ N(x_k-1, imp_processp^-1)
        π(y_k | x_k) ~ N(x_k, imp_measurementp^-1)
    """

    # Time horizon
    time_horizon = length(observations)

    # Preallocate importance sample array
    samples = zeros(time_horizon+1, num_particles)

    # Preallocate importance weight array
    weights = zeros(time_horizon+1, num_particles)

    # Initialize importance weight
    weights[1,:] = ones(1, num_particles) / num_particles

    # Draw from prior
    samples[1,:] = rand(Normal(0, 1), num_particles)

    for t = 2:time_horizon+1

        for i = 1:num_particles

            "Update importance distribution"

            # Update parameters
            impp_precision = imp_processp + imp_measurementp
            impp_mean = inv(impp_precision)*(imp_processp*samples[t-1, i] + imp_measurementp*observations[t-1])

            # Define new importance posterior distribution
            pi_ti = Normal(impp_mean, sqrt(inv(impp_precision)))

            # Draw new importance sample
            samples[t, i] = rand(pi_ti, 1)[1]

            "Update importance weights"

            # Evaluate nominal distributions
            px = pdf(nominal_state_transition(samples[t-1, i]), samples[t, i])
            py = pdf(nominal_likelihood(samples[t, i]), observations[t-1])

            # Evaluate probability of importance sample under importance dist.
            pπ = pdf(pi_ti, samples[t,i])

            # Update weight
            weights[t, i] = weights[t-1,i]*(py * px / pπ)
        end

        # Normalize weights
        weights[t,:] = weights[t,:] / sum(weights[t,:])

        "Resample according to weights"

        # Compute effective sample size
        num_particles_effective = 1 ./ sum(weights[t, :].^2)

        # Check whether resampling is necessary
        if num_particles_effective < resampling_threshold * num_particles

            # Define categorical distribution based on weights
            dd = Categorical(weights[t,:])

            # Draw from discrete distribution
            index_set = rand(dd, num_particles)

            # Select samples according to drawn index set
            samples[t,:] = samples[t, index_set]

            # Weights are reset to 1/N
            weights[t,:] = ones(1,num_particles) / num_particles

        end
    end
    return samples, weights
end

function bootstrap_filter(observations,
                          nominal_state_transition,
                          nominal_likelihood;
                          num_particles=100)
    """
    Bootstrap filter (Alg. 7.5)

    This filter is built for the following model:
        p(x_k | x_k-1) ~ nominal_state_transition(x_k-1)
        w_k ~ nominal_likelihood(x_k)
    """

    # Time horizon
    time_horizon = length(observations)

    # Preallocate importance sample array
    samples = zeros(time_horizon+1, num_particles)

    # Preallocate importance weight array
    weights = zeros(time_horizon+1, num_particles)

    # Initialize importance weight
    weights[1,:] = ones(1, num_particles) / num_particles

    # Draw from prior
    samples[1,:] = rand(Normal(0, 1), num_particles)

    for t = 2:time_horizon+1

        for i = 1:num_particles

            # Sample from state transition
            samples[t, i] = rand(nominal_state_transition(samples[t-1, i]), 1)[1]

            # Compute weights
            weights[t, i] = pdf(nominal_likelihood(samples[t, i]), observations[t-1])

        end

        # Normalize weights
        weights[t,:] = weights[t,:] / sum(weights[t,:])

        # Define categorical distribution based on weights
        dd = Categorical(weights[t,:])

        # Draw indices from discrete distribution defined by weights
        index_set = rand(dd, num_particles)

        # Select samples according to drawn index set
        samples[t,:] = samples[t, index_set]

        # Weights are reset to 1/N
        weights[t,:] = ones(1,num_particles) / num_particles

    end
    return (samples, weights)
end

function rao_blackwellized_particle_filter(observations,
                                           nominal_dynamics,
                                           importance_dist,
                                           A, Q, H, R;
                                           num_particles=100,
                                           resampling_threshold=0.5)
    """
    Rao-Blackwellization of sequential importance resampling.

    This filter is built for the following model:

        p(x_k | x_k-1, u_k-1) ~ N(A_k-1(u_k-1) x_k-1, Q_k-1(u_k-1))
        p(y_k | x_k, u_k) ~ N(H_k(u_k) x_k, R_k(u_k))
        p(u_k | u_k-1) ~ nominal_dynamics(u_k-1)
        π(u_k | u_k-1, y_k) ~ importance_dist(u_k-1, y_k)
        p(u_0) ~ N(0,1)

    where we assume A, Q, H, and R are functions of u_k. For example:

        A_k = A(u_k) = u_k ⋅ 1
        Q_k = Q(u_k) = u_k^2
        H_k = H(u_k) = 1/2 ⋅ u_k
        R_k = R(u_k) = 1/2 ⋅ u_k^2
    """

    # Time horizon
    time_horizon = length(observations)

    # Preallocate particle arrays
    samples = zeros(time_horizon+1, num_particles)
    weights = zeros(time_horizon+1, num_particles)
    means = zeros(time_horizon+1, num_particles)
    cvars = zeros(time_horizon+1, num_particles)

    # Draw initial particles
    weights[1,:] = ones(1, num_particles) / num_particles
    samples[1,:] = rand(Normal(0, 0.1), num_particles)
    means[1,:] = rand(Normal(0,1), num_particles)
    cvars[1,:] = rand(Gamma(1,1), num_particles)

    for t = 2:time_horizon+1

        # Allocate predictions of Kalman filter parameters
        mean_ = zeros(num_particles,)
        cvar_ = zeros(num_particles,)

        for i = 1:num_particles

            # Convenience variables
            A_tmini = A(samples[t-1,i])
            Q_tmini = Q(samples[t-1,i])

            "1. Kalman filter predictions"

            # Move Kalman filter forward
            mean_[i] = A_tmini*means[t-1,i]
            cvar_[i] = A_tmini*cvars[t-1,i]*A_tmini + Q_tmini

            "2. Draw new latent variables u"

            # Draw from given importance distribution
            samples[t,i] = rand(importance_dist(samples[t-1,i], observations[t-1]), 1)[1]

            "3. Calculate new weights"

            # Convencience variable
            H_ti = H(samples[t,i])
            R_ti = R(samples[t,i])

            # Marginalized measurement likelihood
            pyu = pdf(Normal(H_ti*mean_[i], sqrt(H_ti*cvar_[i]*H_ti + R_ti)), observations[t-1])
            puu = pdf(nominal_dynamics(samples[t-1,i]), samples[t,i])
            πuu = pdf(importance_dist(samples[t-1,i], observations[t-1]), samples[t,i])

            # Update weights
            weights[t,i] = (pyu * puu / πuu) * weights[t-1,i]
        end

        # Normalize weights
        weights[t,:] = weights[t,:] / sum(weights[t,:])

        for i = 1:num_particles

            # Convenience variables
            H_ti = H(samples[t,i])
            R_ti = R(samples[t,i])

            "4. Kalman filter updates"

            # Difference between prediction and observation
            v = observations[t-1] - H_ti*mean_[i]

            # Updated precision
            S = H_ti * cvar_[i]* H_ti + R_ti

            # Kalman gain
            K = cvar_[i]*H_ti * inv(S)

            # Update Kalman filter parameters
            means[t,i] = mean_[i] + K*v
            cvars[t,i] = cvar_[i] - K*S*K'
        end

        # Compute effective sample size
        num_particles_effective = 1 ./ sum(weights[t, :].^2)

        # Check whether resampling is necessary
        if num_particles_effective < resampling_threshold * num_particles

            # Define categorical distribution based on weights
            dd = Categorical(weights[t,:])

            # Draw from discrete distribution
            index_set = rand(dd, num_particles)

            # Select samples according to drawn index set
            samples[t,:] = samples[t, index_set]

            # Weights are reset to 1/N
            weights[t,:] = ones(1,num_particles) / num_particles

        end
    end
    return (samples, weights, means, cvars)
end

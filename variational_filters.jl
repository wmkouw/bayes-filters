"""
Variational Bayesian filters.

These are Bayesian filters with variational approximations 
to support simulaneous inference of states and other latent variables.

"""

using Distributions
using Random

include("util.jl")


function robust_students_t_filter(observations,
                                  transition_matrix,
                                  measurement_matrix,
                                  process_noise_cov,
                                  measurement_noise_cov,
                                  prior_pnoise_cov,
                                  prior_pnoise_scale,
                                  prior_mnoise_scale,
                                  state_0;
                                  num_iters=10,
                                  τ=1.0)
    """
    Ref: Huang, Zhang, Li, Wu, Chambers (2017). A novel robust student's t-based 
    Kalman filter. IEEE Transactions on Aerospace and Electronic systems.

    This filter incorporates a student's t distributed noise model, which 
    creates robustness to process and measurement noise outliers.
    """

    # Extract prior parameters
    ω = prior_pnoise_scale
    ν = prior_mnoise_scale
    u_k, U_k = prior_pnoise_cov

    # Shorthand
    y = observations
    F = transition_matrix
    H = measurement_matrix
    Q = process_noise_cov
    R = measurement_noise_cov

    # Dimensionality
    Dx = size(F,1)
    Dy = size(H,1)

    # Time horizon
    time_horizon = maximum(size(observations))

    # Initialize estimate arrays
    mx = zeros(Dx, time_horizon, num_iters)
    Px = zeros(Dx, Dx, time_horizon, num_iters)
    qΣ = zeros(Dx, Dx, time_horizon, num_iters)

    # Start previous state variables
    m_kmin1 = state_0[1]
    P_kmin1 = state_0[2]

    for k = 1:time_horizon

        # Prediction step
        m_k_pred = F*m_kmin1
        P_k_pred = F*P_kmin1*F' .+ Q

        # Initialization
        uh_k = Dy + τ + 1
        Uh_k = τ*P_k_pred
        m_k = m_k_pred
        P_k = P_k_pred
        EiΣ = (uh_k - Dy - 1)*inv(Uh_k)

        for i = 1:num_iters

            # Auxiliary
            D_k = P_k + (m_k - m_k_pred)*(m_k - m_k_pred)'
            E_k = (y[:,k] - H*m_k)*(y[:,k] - H*m_k)' + H*P_k*H'

            # Update recognition factor q(ξ_k) [process noise scale]
            α_k = 0.5*(Dy + ω)
            β_k = 0.5*(ω + tr(D_k*EiΣ))

            # Update recognition factor q(λ_k) [measurement noise scale]
            γ_k = 0.5*(Dx + ν)
            δ_k = 0.5*(ν + tr(E_k*inv(R)))

            # Update recognition factor q(Σ_k) [process noise covariance]
            uh_k = u_k + 1
            Uh_k = U_k + α_k/β_k*D_k
            EiΣ = (uh_k - Dy - 1)*inv(Uh_k)

            # Update recognition factor q(x_k) [state estimate]
            Rt_k = R/(γ_k / δ_k)
            Pt_k = inv(EiΣ)/(α_k/β_k)
            K_k = Pt_k*H'*inv(H*Pt_k*H' + Rt_k) 
            m_k = m_k_pred + K_k*(y[:,k] - H*m_k_pred)
            P_k = Pt_k - K_k*H*Pt_k

            # Store estimates
            qΣ[:,:,k,i] = uh_k*Uh_k
            mx[:,k,i] = m_k
            Px[:,:,k,i] = P_k       

        end

        # Update previous state variable
        m_kmin1 = m_k
        P_kmin1 = P_k
    end

    return mx, Px, qΣ#, qλ, qξ
end
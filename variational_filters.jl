"""
Variational Bayesian filters.

These are Bayesian filters with variational approximations 
to support simulaneous inference of states and other latent variables.

"""

using Distributions
using Random


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

function AX_filter(observations::Vector{Vector{Float64}},
                   Q::Matrix, C::Matrix, R::Matrix,
                   ard_prior::Vector{Gamma{Float64}}, 
                   state_prior::FullNormal;
                   num_iters::Integer=10)
    """
    Ref: Lutinnen (2013). Fast Variational Bayesian Linear State-Space Model. ECML.

    This filter infers both states and state transition matrix elements.
    """

    T = length(observations)
    D = size(Q,1)

    m0,S0 = params(state_prior)
    α0    = shape.(ard_prior)
    β0    = rate.( ard_prior)

    # Preallocate
    mx = zeros(D,T)
    Sx = zeros(D,D,T)
    Cx = zeros(D,D,T)
    M = diagm(ones(D))
    U = cat([diagm(1e-3ones(D)) for d in 1:D]...,dims=3)
    α = ones(D)
    β = ones(D)

    for _ = 1:num_iters

        "State estimation"

        AA = zeros(D,D)
        for i in 1:D
            for j in 1:D
                AA[i,j] = sum([M[i,d]*M[j,d] + U[i,j,d] for d in 1:D]) 
            end
        end

        Ψ_diag = inv(Q) + AA + C*inv(R)*C'
        Ψ_offd = -M'

        Sx_00 = inv(inv(S0) + AA)
        mx_0 = Sx_00*inv(S0)*m0
        Cx[:,:,1] = Sx_00*Ψ_offd
        Sx[:,:,1] = inv(Ψ_diag - Cx[:,:,1]'*Ψ_offd)
        mx[:,1]   = Sx[:,:,1]*(C*inv(R)*observations[1] - Cx[:,:,1]'*mx_0)

        # Forward pass
        for k = 2:T-1
            Cx[:,:,k] = Sx[:,:,k-1]*Ψ_offd
            Sx[:,:,k] = inv(Ψ_diag - Cx[:,:,k]'*Ψ_offd)
            mx[:,k]   = Sx[:,:,k]*(C*inv(R)*observations[k] - Cx[:,:,k]'*mx[:,k-1])
        end

        # Update for final step
        Cx[:,:,T] = Sx[:,:,T-1]*Ψ_offd
        Sx[:,:,T] = inv(inv(Q) + C'*inv(R)*C - Cx[:,:,T]'*Ψ_offd)
        mx[:,T]   = Sx[:,:,T]*(C*inv(R)*observations[T] - Cx[:,:,T]'*mx[:,T-1])

        "Parameter estimation"
        
        # Update relevance variables
        for d in 1:D
            α[d] = α0[d] + D/2
            β[d] = β0[d] + 1/2*sum([M[j,d]^2 + U[j,j,d] for j = 1:D])
        end

        # Update state transition matrix
        for d in 1:D
            U[:,:,d] = inv(diagm(α./β) + (mx_0*mx_0'+Sx_00) + sum([mx[:,k]*mx[:,k]' + Sx[:,:,k] for k in 1:T-1]))

            XX = mx[d,1]*mx_0
            for n = 2:T
                XX += mx[d,n]*mx[:,n-1] + Cx[:,d,n-1]
            end
            M[d,:] = transpose(U[:,:,d]*XX)
        end
    end
    return mx,Sx,Cx, M,U, α,β
end

function AX_filter2(observations::Vector{Vector{Float64}},
                   Q::Matrix, C::Matrix, R::Matrix,
                   ard_prior::Vector{Gamma{Float64}}, 
                   state_prior::FullNormal;
                   num_iters::Integer=10)
    """
    Ref: Lutinnen (2013). Fast Variational Bayesian Linear State-Space Model. ECML.

    This filter infers both states and state transition matrix elements.
    """

    m0,S0 = params(state_prior)
    α0    = shape.(ard_prior)
    β0    = rate.( ard_prior)

    T = length(observations)
    D = length(m0)

    # Preallocate
    mx = zeros(D,T)
    Sx = zeros(D,D,T)
    Cx = zeros(D,D,T)
    M = diagm(ones(D))
    U = diagm(ones(D))
    V = diagm(ones(D))
    α = ones(D)
    β = ones(D)

    for _ = 1:num_iters

        "State estimation"

        AA = M'*M + V*tr(U)
        P_lik = C'*inv(R)*C

        S_00 = inv(inv(S0) + AA)
        m_0 = S_00*inv(S0)*m0
        
        P_prior = inv(M*S_00*M' + Q)
        Cx[:,:,1] = M*S_00
        Sx[:,:,1] = inv(P_prior + P_lik)
        mx[:,1]   = Sx[:,:,1]*(P_prior*M*m_0 + C'*inv(R)*observations[1])

        # Forward pass
        for k = 2:T
            P_prior = inv(M*Sx[:,:,k-1]*M' + Q)
            Cx[:,:,k] = M*Sx[:,:,k-1]
            Sx[:,:,k] = inv(P_prior + P_lik)
            mx[:,k]   = Sx[:,:,k]*(P_prior*M*mx[:,k-1] + C'*inv(R)*observations[k])
        end

        "Parameter estimation"
        
        # Update relevance variables
        for d in 1:D
            α[d] = α0[d] + D/2
            β[d] = β0[d] + 1/2*sum([M[j,d]^2 + U[j,d] for j = 1:D])
        end

        # Matrix normal row covariance
        XX = m_0*m_0' + S_00
        for k = 2:T
            XX += mx[:,k-1]*mx[:,k-1]' + Sx[:,:,k-1]
        end
        U = inv(diagm(α./β) + XX)

        # Matrix normal mean
        XC = zeros(D,D)
        for k = 2:T
            XC += mx[:,k]*mx[:,k-1]' + Cx[:,:,k]
        end
        M = U*XC
    end
    return mx,Sx,Cx, M,U, α,β
end
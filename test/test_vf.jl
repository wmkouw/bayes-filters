using LinearAlgebra
using Distributions
using Plots
default(label="", grid=false, margin=20Plots.pt)

include("../variational_filters.jl")

"""System dynamics"""

# Length of time-series
T = 100
Δt = 1.0

# Dimensionalities
D = 4

# Transition matrix
ω = 0.3
A = [cos(ω) -sin(ω)   0    0;
     sin(ω)  cos(ω)   0    0;
       0       0    0.3    0;
       0       0      0  0.6]

# Process noise covariance matrix
qc = 3rand()
Q = qc.*diagm(ones(D))

# Emission matrix
C = diagm(ones(D));
    
# Measurement noise covariance matrix
σ = 0.1
R = σ^2 .*diagm(ones(D))
     
# Initial states
state_0 = zeros(D)

# Initialize data array
states = zeros(D,T)
observations = zeros(D,T)

# Initialize previous state variable
prev_state = state_0

for k = 1:T
    
    # State transition
    states[:,k] = A*prev_state + cholesky(Q).L*randn(D)
    
    # Observation with added measurement noise
    observations[:,k] = C*states[:,k] + cholesky(R).L*randn(D)
    
    # Update "previous state"
    global prev_state = states[:,k]
    
end    

"""AX filter"""

# Relevance prior
α0 = [1.0, 1.0, 0.1, 0.1]
β0 = [1.0, 1.0, 1.0, 1.0]
γ0 = [Gamma(α0[d], β0[d]) for d in 1:D]

# State prior
m0 = zeros(D)
S0 = diagm(ones(D))
x0 = MvNormal(m0,S0)

# AX filter
mx, Px, μ,Σ,α,β = AX_filter([observations[:,k] for k in 1:T], C, R, γ0, x0, num_iters=10);

# Plot state estimates
plts = []
for d = 1:D
    plt_d = plot(size=(700,400),legend=:topleft)
    plot!((1:T).*Δt, states[d,:], linewidth=3, color="blue", xlabel="time", ylabel="state $d", label="states")
    plot!((1:T).*Δt, mx[d,:], ribbon=sqrt.(Px[d,d,:]), linewidth=3, color="purple", label="inferred")
    scatter!((1:T).*Δt, observations[d,:], color="black", markersize=2, label="observed")

    push!(plts, plt_d)
end
plot(plts..., layout=(D,1), size=(900,D*200))
savefig("test/figures/AX-filter_state-estimates.png")

# Plot transition matrix estimate
cmins = minimum([minimum(A[:]), minimum(μ[:])])
cmaxs = maximum([maximum(A[:]), maximum(μ[:])])

p21 = heatmap(A, yflip=true, clims=(cmins,cmaxs), colormap=:roma, title="true A")
ann = [(i,j, text(round(A[j,i], digits=2), 15, :white, :center)) for i in 1:D for j in 1:D]
annotate!(ann, linecolor=:white)

p22 = heatmap(μ, yflip=true, clims=(cmins,cmaxs), colormap=:roma, title="estimated A")
ann = [(i,j, text(round(μ[j,i], digits=2), 15, :white, :center)) for i in 1:D for j in 1:D]
annotate!(ann, linecolor=:white)

plot(p21,p22, layout=(1,2), size=(900,300))
savefig("test/figures/AX-filter_A-estimate.png")
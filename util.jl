"Utility functions"

using Zygote

function Jacobian(F, x)
    y = F(x)
    n = length(y)
    m = length(x)
    T = eltype(y)
    J = Array{T, 2}(undef, n, m)
    for i in 1:n
        J[i, :] .= gradient(x -> F(x)[i], x)[1]
    end
    return J
end

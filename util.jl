"Utility functions"

import Zygote: gradient
import Zygote: hessian

function Jacobian(F::Function, x::AbstractArray)
    "Jacobian matrix of non-scalar function."
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

function Hessian(F::Function, x::AbstractArray)
    "Hessian matrix of i-th element of function"
    y = F(x)
    n = length(y)
    m = length(x)
    T = eltype(y)
    H = Array{T, 3}(undef, n, m, m)
    for i in 1:n
        H[i, :, :] .= hessian(x -> F(x)[i], x)
    end
    return H
end

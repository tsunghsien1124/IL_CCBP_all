#=========================================#
# JuMP example (1) -- Nested optimization #
#=========================================#
using JuMP
import Ipopt

function solve_lower_level(x::T...) where {T}
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, y[1:2])
    @objective(
        model,
        Max,
        x[1]^2 * y[1] + x[2]^2 * y[2] - x[1] * y[1]^4 - 2 * x[2] * y[2]^4,
    )
    @constraint(model, (y[1] - 10)^2 + (y[2] - 10)^2 <= 25)
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return objective_value(model), value.(y)
end

function V(x::T...) where {T}
    f, _ = solve_lower_level(x...)
    return f
end

function ∇V(g::AbstractVector, x...)
    _, y = solve_lower_level(x...)
    g[1] = 2 * x[1] * y[1] - y[1]^4
    g[2] = 2 * x[2] * y[2] - 2 * y[2]^4
    return
end

function ∇²V(H::AbstractMatrix, x...)
    _, y = solve_lower_level(x...)
    H[1, 1] = 2 * y[1]
    H[2, 2] = 2 * y[2]
    return
end

model = Model(Ipopt.Optimizer)
@variable(model, x[1:2] >= 0)
@operator(model, op_V, 2, V, ∇V, ∇²V)
# @operator(model, op_V, 2, V)
@objective(model, Min, x[1]^2 + x[2]^2 + op_V(x[1], x[2]))
optimize!(model)
@assert is_solved_and_feasible(model)
solution_summary(model)

objective_value(model)

value.(x)

_, y = solve_lower_level(value.(x)...)
y

# Automatic differentiation using ForwardDiff.jl

model = Model(Ipopt.Optimizer)
@variable(model, x[1:2] >= 0)
@operator(model, op_V, 2, V, fdiff_derivatives(V)...)
@objective(model, Min, x[1]^2 + x[2]^2 + op_V(x[1], x[2]))
optimize!(model)
@assert is_solved_and_feasible(model)
solution_summary(model)

#=================================#
# JuMP example (2) -- using Cache #
#=================================#
mutable struct Cache
    x::Any
    f::Float64
    y::Vector{Float64}
end

function _update_if_needed(cache::Cache, x...)
    if cache.x !== x
        cache.f, cache.y = solve_lower_level(x...)
        cache.x = x
    end
    return
end

function cached_f(cache::Cache, x...)
    _update_if_needed(cache, x...)
    return cache.f
end

function cached_∇f(cache::Cache, g::AbstractVector, x...)
    _update_if_needed(cache, x...)
    g[1] = 2 * x[1] * cache.y[1] - cache.y[1]^4
    g[2] = 2 * x[2] * cache.y[2] - 2 * cache.y[2]^4
    return
end

function cached_∇²f(cache::Cache, H::AbstractMatrix, x...)
    _update_if_needed(cache, x...)
    H[1, 1] = 2 * cache.y[1]
    H[2, 2] = 2 * cache.y[2]
    return
end

model = Model(Ipopt.Optimizer)
@variable(model, x[1:2] >= 0)
cache = Cache(Float64[], NaN, Float64[])
@operator(
    model,
    op_cached_f,
    2,
    (x...) -> cached_f(cache, x...),
    (g, x...) -> cached_∇f(cache, g, x...),
    (H, x...) -> cached_∇²f(cache, H, x...),
)
@objective(model, Min, x[1]^2 + x[2]^2 + op_cached_f(x[1], x[2]))
optimize!(model)
@assert is_solved_and_feasible(model)
solution_summary(model)

#===============================================#
# JuMP example (3) -- Automatic differentiation #
#===============================================#
using JuMP
import Enzyme
import ForwardDiff
import Ipopt
import Test

f(x::T...) where {T} = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

x = rand(2)

f(x...)

function analytic_∇f(g::AbstractVector, x...)
    g[1] = 400 * x[1]^3 - 400 * x[1] * x[2] + 2 * x[1] - 2
    g[2] = 200 * (x[2] - x[1]^2)
    return
end

analytic_g = zeros(2)
analytic_∇f(analytic_g, x...)
analytic_g

function analytic_∇²f(H::AbstractMatrix, x...)
    H[1, 1] = 1200 * x[1]^2 - 400 * x[2] + 2
    # H[1, 2] = -400 * x[1] <-- not needed because Hessian is symmetric
    H[2, 1] = -400 * x[1]
    H[2, 2] = 200.0
    return
end

analytic_H = zeros(2, 2)
analytic_∇²f(analytic_H, x...)
analytic_H

function analytic_rosenbrock()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    @operator(model, op_rosenbrock, 2, f, analytic_∇f, analytic_∇²f)
    @objective(model, Min, op_rosenbrock(x[1], x[2]))
    optimize!(model)
    Test.@test is_solved_and_feasible(model)
    return value.(x)
end

analytic_rosenbrock()

function fdiff_∇f(g::AbstractVector{T}, x::Vararg{T,N}) where {T,N}
    ForwardDiff.gradient!(g, y -> f(y...), collect(x))
    return
end

fdiff_g = zeros(2)
fdiff_∇f(fdiff_g, x...)
Test.@test ≈(analytic_g, fdiff_g)

function fdiff_∇²f(H::AbstractMatrix{T}, x::Vararg{T,N}) where {T,N}
    h = ForwardDiff.hessian(y -> f(y...), collect(x))
    for i in 1:N, j in 1:i
        H[i, j] = h[i, j]
    end
    return
end

fdiff_H = zeros(2, 2)
fdiff_∇²f(fdiff_H, x...)
Test.@test ≈(analytic_H, fdiff_H)

"""
    fdiff_derivatives(f::Function) -> Tuple{Function,Function}

Return a tuple of functions that evaluate the gradient and Hessian of `f` using
ForwardDiff.jl.
"""
function fdiff_derivatives(f::Function)
    function ∇f(g::AbstractVector{T}, x::Vararg{T,N}) where {T,N}
        ForwardDiff.gradient!(g, y -> f(y...), collect(x))
        return
    end
    function ∇²f(H::AbstractMatrix{T}, x::Vararg{T,N}) where {T,N}
        h = ForwardDiff.hessian(y -> f(y...), collect(x))
        for i in 1:N, j in 1:i
            H[i, j] = h[i, j]
        end
        return
    end
    return ∇f, ∇²f
end

function fdiff_rosenbrock()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    @operator(model, op_rosenbrock, 2, f, fdiff_derivatives(f)...)
    @objective(model, Min, op_rosenbrock(x[1], x[2]))
    optimize!(model)
    Test.@test is_solved_and_feasible(model)
    return value.(x)
end

fdiff_rosenbrock()
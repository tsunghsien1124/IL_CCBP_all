#=================#
# Import packages #
#=================#
using Parameters
using JuMP
import Ipopt
using ProgressMeter
using PrettyTables
using GLMakie
using CairoMakie
using Dates
using LaTeXStrings
using Base.Iterators

#==============#
# Housekeeping #
#==============#
PWD = pwd()
VER = "V12"
if Sys.iswindows()
    FL = "\\"
else
    FL = "/"
end
# PATH = mkpath(PWD * FL * VER)
PATH_FIG = mkpath(PWD * FL * "figures" * FL * VER)
dg_p, dg_f = 1, 6

#==============#
# BP functions #
#==============#
τ_1(x_1, x_2, μ) = x_1 * μ + (1.0 - x_2) * (1.0 - μ)
μ_1(x_1, x_2, μ) = x_1 * μ / τ_1(x_1, x_2, μ)
τ_2(x_1, x_2, μ) = (1.0 - x_1) * μ + x_2 * (1.0 - μ)
μ_2(x_1, x_2, μ) = (1.0 - x_1) * μ / τ_2(x_1, x_2, μ)
H(μ) = -(μ * log(μ) + (1.0 - μ) * log((1.0 - μ)))
# c(x_1, x_2, μ) = (1.0 / log(2.0)) * (H(μ) - τ_1(x_1, x_2, μ) * H(μ_1(x_1, x_2, μ)) - τ_2(x_1, x_2, μ) * H(μ_2(x_1, x_2, μ)))
χ(μ) = 1 / H(μ)
c(x_1, x_2, μ) = χ(μ) * (H(μ) - τ_1(x_1, x_2, μ) * H(μ_1(x_1, x_2, μ)) - τ_2(x_1, x_2, μ) * H(μ_2(x_1, x_2, μ)))

#=====================#
# inflation functions #
#=====================#
x_r(ω_i, x_T, ν_1, ν_2) = ω_i == 1 ? x_T + ν_1 : x_T - ν_2
μ_a(μ_0, μ, θ) = (1.0 - θ) * μ_0 + θ * μ
x_e(μ_0, μ, x_T, ν_1, ν_2, θ) = μ_a(μ_0, μ, θ) * x_r(1, x_T, ν_1, ν_2) + (1.0 - μ_a(μ_0, μ, θ)) * x_r(2, x_T, ν_1, ν_2) # x_T + ν * (2.0 * μ - 1.0) if symmetric ν

#========================#
# distribution functions #
#========================#
f(x_1, x_2, μ, a, b) = a * b * (c(x_1, x_2, μ)^(a - 1.0)) * (1.0 - c(x_1, x_2, μ)^a)^(b - 1.0)
F(x_1, x_2, μ, a, b) = 1.0 - (1.0 - c(x_1, x_2, μ)^a)^b

#========================#
# CB objective functions #
#========================#
obj_CB_μ_0(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, θ, a, b) = δ * F(x_1, x_2, μ_0, a, b) * (μ_0_c * (ω_1 + γ * (x_e(μ_0, μ_0, x_T, ν_1, ν_2, θ) - x_r(1, x_T, ν_1, ν_2)))^2.0 + (1.0 - μ_0_c) * (ω_2 + γ * (x_e(μ_0, μ_0, x_T, ν_1, ν_2, θ) - x_r(2, x_T, ν_1, ν_2)))^2.0)

obj_CB_1(x_1, x_2, μ_0, μ_0_c, ω_1, δ, γ, x_T, ν_1, ν_2, θ, a, b) = (1.0 - δ * F(x_1, x_2, μ_0, a, b)) * μ_0_c * (x_1 * (ω_1 + γ * (x_e(μ_0, μ_1(x_1, x_2, μ_0), x_T, ν_1, ν_2, θ) - x_r(1, x_T, ν_1, ν_2)))^2.0 + (1.0 - x_1) * (ω_1 + γ * (x_e(μ_0, μ_2(x_1, x_2, μ_0), x_T, ν_1, ν_2, θ) - x_r(1, x_T, ν_1, ν_2)))^2.0)

obj_CB_2(x_1, x_2, μ_0, μ_0_c, ω_2, δ, γ, x_T, ν_1, ν_2, θ, a, b) = (1.0 - δ * F(x_1, x_2, μ_0, a, b)) * (1.0 - μ_0_c) * ((1.0 - x_2) * (ω_2 + γ * (x_e(μ_0, μ_1(x_1, x_2, μ_0), x_T, ν_1, ν_2, θ) - x_r(2, x_T, ν_1, ν_2)))^2.0 + x_2 * (ω_2 + γ * (x_e(μ_0, μ_2(x_1, x_2, μ_0), x_T, ν_1, ν_2, θ) - x_r(2, x_T, ν_1, ν_2)))^2.0)

obj_CB_x(μ_0_c, x_T, ν_1, ν_2, α) = α * (μ_0_c * (x_r(1, x_T, ν_1, ν_2) - x_T)^2 + (1.0 - μ_0_c) * (x_r(2, x_T, ν_1, ν_2) - x_T)^2)

obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α, θ, a, b) = obj_CB_μ_0(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, θ, a, b) + obj_CB_1(x_1, x_2, μ_0, μ_0_c, ω_1, δ, γ, x_T, ν_1, ν_2, θ, a, b) + obj_CB_2(x_1, x_2, μ_0, μ_0_c, ω_2, δ, γ, x_T, ν_1, ν_2, θ, a, b) + obj_CB_x(μ_0_c, x_T, ν_1, ν_2, α)

#======================#
# benchmark parameters #
#======================#
@with_kw struct Benchmark_Parameters
    δ::Float64 = 0.5
    ω_1::Float64 = 1.0
    ω_2::Float64 = -1.0
    μ_0::Float64 = 0.5
    μ_0_diff::Float64 = 0.0
    μ_0_c::Float64 = 0.5 # μ_0 * (1.0 + μ_0_diff / 100)
    γ::Float64 = 10.0
    x_T::Float64 = 2.0
    ν_1::Float64 = 1.0
    ν_2::Float64 = 1.0
    α::Float64 = 1.0
    θ::Float64 = 1.0
    a::Float64 = 1.0
    b::Float64 = 1.0
    ϵ_x::Float64 = 1E-6
    ϵ_x_p::Float64 = 1E-0
    ϵ_tol::Float64 = 1E-6
    max_iter::Int64 = 3000
end
BP = Benchmark_Parameters()
PATH_FIG_para = mkpath(PATH_FIG * FL * "a=$(round(BP.a,digits=dg_p))_b=$(round(BP.b,digits=dg_p))" * FL * "γ=$(round(BP.γ,digits=dg_p))_α=$(round(BP.α,digits=dg_p))" * FL * "θ=$(round(BP.θ,digits=dg_p))_δ=$(round(BP.δ,digits=dg_p))" * FL * "μ_0=$(round(BP.μ_0,digits=dg_p))_μ_0_c=$(round(BP.μ_0_c,digits=dg_p))_ω_1=$(round(BP.ω_1,digits=dg_p))_ω_2=$(round(BP.ω_2,digits=dg_p))")

#==================#
# benchmark result #
#==================#
function optimal_x_func(μ_0::Float64, μ_0_c::Float64, ω_1::Float64, ω_2::Float64, δ::Float64, γ::Float64, x_T::Float64, ν_1::Float64, ν_2::Float64, α::Float64, θ::Float64, a::Float64, b::Float64, ϵ_x::Float64, ϵ_x_p::Float64, ϵ_tol::Float64, max_iter::Int64)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    set_attribute(model, "max_iter", max_iter)
    @variable(model, ϵ_x <= x_1 <= (1.0 * ϵ_x_p - ϵ_x), start = 1.0 * ϵ_x_p / 2.0)
    @variable(model, ϵ_x <= x_2 <= (1.0 * ϵ_x_p - ϵ_x), start = 1.0 * ϵ_x_p / 2.0)
    @constraint(model, c1, x_1 + x_2 >= 1.0 * ϵ_x_p)
    _obj_CB(x_1, x_2) = obj_CB(x_1 / ϵ_x_p, x_2 / ϵ_x_p, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α, θ, a, b)
    @objective(model, Min, _obj_CB(x_1, x_2))
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return objective_value(model), value(x_1) / ϵ_x_p, value(x_2) / ϵ_x_p
end
obj_opt, x_1_opt, x_2_opt = optimal_x_func(BP.μ_0, BP.μ_0_c, BP.ω_1, BP.ω_2, BP.δ, BP.γ, BP.x_T, BP.ν_1, BP.ν_2, BP.α, BP.θ, BP.a, BP.b, BP.ϵ_x, BP.ϵ_x_p, BP.ϵ_tol, BP.max_iter)
println("Find the minimum of $obj_opt at (x_1, x_2) = ($x_1_opt, $x_2_opt)")

function optimal_x_ν_func(μ_0::Float64, μ_0_c::Float64, ω_1::Float64, ω_2::Float64, δ::Float64, γ::Float64, x_T::Float64, α::Float64, θ::Float64, a::Float64, b::Float64, ϵ_x::Float64, ϵ_x_p::Float64, ϵ_tol::Float64, max_iter::Int64)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    set_attribute(model, "max_iter", max_iter)
    @variable(model, ϵ_x <= x_1 <= (1.0 * ϵ_x_p - ϵ_x), start = 1.0 * ϵ_x_p / 2.0)
    @variable(model, ϵ_x <= x_2 <= (1.0 * ϵ_x_p - ϵ_x), start = 1.0 * ϵ_x_p / 2.0)
    @variable(model, ν_1 >= ϵ_x, start = ϵ_x)
    @variable(model, ν_2 >= ϵ_x, start = ϵ_x)
    @constraint(model, c1, x_1 + x_2 >= 1.0 * ϵ_x_p)
    _obj_CB(x_1, x_2, ν_1, ν_2) = obj_CB(x_1 / ϵ_x_p, x_2 / ϵ_x_p, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α, θ, a, b)
    @objective(model, Min, _obj_CB(x_1, x_2, ν_1, ν_2))
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return objective_value(model), value(x_1) / ϵ_x_p, value(x_2) / ϵ_x_p, value(ν_1), value(ν_2)
end
obj_opt, x_1_opt, x_2_opt, ν_1_opt, ν_2_opt = optimal_x_ν_func(BP.μ_0, BP.μ_0_c, BP.ω_1, BP.ω_2, BP.δ, BP.γ, BP.x_T, BP.α, BP.θ, BP.a, BP.b, BP.ϵ_x, BP.ϵ_x_p, BP.ϵ_tol, BP.max_iter)
println("Find the minimum of $obj_opt at (x_1, x_2, ν_1, ν_2) = ($x_1_opt, $x_2_opt, $ν_1_opt, $ν_2_opt)")

#=============================================#
# handy functions for optimal communication x #
#=============================================#
function optimal_communication_func!(BP::Benchmark_Parameters, res::Array{Float64,2}, TA::String, TA_size::Int64, TA_grid::Vector{Float64})

    # unpack benchmark parameters
    @unpack μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α, θ, a, b, ϵ_x, ϵ_x_p, ϵ_tol, max_iter = BP

    # timer
    pp = Progress(TA_size)
    update!(pp, 0)
    jj = Threads.Atomic{Int}(0)
    ll = Threads.SpinLock()

    # start looping
    Threads.@threads for TA_i in 1:TA_size

        # create the parameter dictionary
        obj_CB_para = Dict([("μ_0", μ_0), ("μ_0_c", μ_0_c), ("ω_1", ω_1), ("ω_2", ω_2), ("δ", δ), ("γ", γ), ("x_T", x_T), ("ν_1", ν_1), ("ν_2", ν_2), ("α", α), ("θ", θ), ("a", a), ("b", b), ("ϵ_x", ϵ_x), ("ϵ_x_p", ϵ_x_p), ("ϵ_tol", ϵ_tol), ("max_iter", max_iter)])
        obj_CB_para[TA] = TA_grid[TA_i]

        # slove CB's optimization problem for (x_1, x_2)
        obj_opt, x_1_opt, x_2_opt = optimal_x_func(obj_CB_para["μ_0"], obj_CB_para["μ_0_c"], obj_CB_para["ω_1"], obj_CB_para["ω_2"], obj_CB_para["δ"], obj_CB_para["γ"], obj_CB_para["x_T"], obj_CB_para["ν_1"], obj_CB_para["ν_2"], obj_CB_para["α"], obj_CB_para["θ"], obj_CB_para["a"], obj_CB_para["b"], obj_CB_para["ϵ_x"], obj_CB_para["ϵ_x_p"], obj_CB_para["ϵ_tol"], obj_CB_para["max_iter"])

        # save results
        @inbounds res[TA_i, 1] = obj_opt
        @inbounds res[TA_i, 2] = x_1_opt
        @inbounds res[TA_i, 3] = x_2_opt
        @inbounds res[TA_i, 4] = ν_1
        @inbounds res[TA_i, 5] = ν_2
        @inbounds res[TA_i, 6] = μ_1(x_1_opt, x_2_opt, obj_CB_para["μ_0"])
        @inbounds res[TA_i, 7] = μ_2(x_1_opt, x_2_opt, obj_CB_para["μ_0"])
        @inbounds res[TA_i, 8] = 1.0 - obj_CB_para["δ"] * F(x_1_opt, x_2_opt, obj_CB_para["μ_0"], obj_CB_para["a"], obj_CB_para["b"])
        @inbounds res[TA_i, 9] = obj_CB_para["δ"] * F(x_1_opt, x_2_opt, obj_CB_para["μ_0"], obj_CB_para["a"], obj_CB_para["b"])
        @inbounds res[TA_i, 10] = c(x_1_opt, x_2_opt, obj_CB_para["μ_0"])

        # update timer
        Threads.atomic_add!(jj, 1)
        Threads.lock(ll)
        update!(pp, jj[])
        Threads.unlock(ll)
    end
    return nothing
end

function optimal_flexibility_func!(BP::Benchmark_Parameters, res::Array{Float64,2}, TA::String, TA_size::Int64, TA_grid::Vector{Float64})

    # unpack benchmark parameters
    @unpack μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α, θ, a, b, ϵ_x, ϵ_x_p, ϵ_tol, max_iter = BP

    # timer
    pp = Progress(TA_size)
    update!(pp, 0)
    jj = Threads.Atomic{Int}(0)
    ll = Threads.SpinLock()

    # start looping
    Threads.@threads for TA_i in 1:TA_size

        # create the parameter dictionary
        obj_CB_para = Dict([("μ_0", μ_0), ("μ_0_c", μ_0_c), ("ω_1", ω_1), ("ω_2", ω_2), ("δ", δ), ("γ", γ), ("x_T", x_T), ("ν_1", ν_1), ("ν_2", ν_2), ("α", α), ("θ", θ), ("a", a), ("b", b), ("ϵ_x", ϵ_x), ("ϵ_x_p", ϵ_x_p), ("ϵ_tol", ϵ_tol), ("max_iter", max_iter)])
        obj_CB_para[TA] = TA_grid[TA_i]

        # slove CB's optimization problem for (x_1, x_2)
        obj_opt, x_1_opt, x_2_opt, ν_1_opt, ν_2_opt = optimal_x_ν_func(obj_CB_para["μ_0"], obj_CB_para["μ_0_c"], obj_CB_para["ω_1"], obj_CB_para["ω_2"], obj_CB_para["δ"], obj_CB_para["γ"], obj_CB_para["x_T"], obj_CB_para["α"], obj_CB_para["θ"], obj_CB_para["a"], obj_CB_para["b"], obj_CB_para["ϵ_x"], obj_CB_para["ϵ_x_p"], obj_CB_para["ϵ_tol"], obj_CB_para["max_iter"])

        # save results
        @inbounds res[TA_i, 1] = obj_opt
        @inbounds res[TA_i, 2] = x_1_opt
        @inbounds res[TA_i, 3] = x_2_opt
        @inbounds res[TA_i, 4] = ν_1_opt
        @inbounds res[TA_i, 5] = ν_2_opt
        @inbounds res[TA_i, 6] = μ_1(x_1_opt, x_2_opt, obj_CB_para["μ_0"])
        @inbounds res[TA_i, 7] = μ_2(x_1_opt, x_2_opt, obj_CB_para["μ_0"])
        @inbounds res[TA_i, 8] = 1.0 - obj_CB_para["δ"] * F(x_1_opt, x_2_opt, obj_CB_para["μ_0"], obj_CB_para["a"], obj_CB_para["b"])
        @inbounds res[TA_i, 9] = obj_CB_para["δ"] * F(x_1_opt, x_2_opt, obj_CB_para["μ_0"], obj_CB_para["a"], obj_CB_para["b"])
        @inbounds res[TA_i, 10] = c(x_1_opt, x_2_opt, obj_CB_para["μ_0"])

        # update timer
        Threads.atomic_add!(jj, 1)
        Threads.lock(ll)
        update!(pp, jj[])
        Threads.unlock(ll)
    end
    return nothing
end

function comparative_result_function!(BP::Benchmark_Parameters, TV::String, TV_size::Int64, TV_grid::Array{Float64,1}, TV_res::Array{Float64,2}, dg_f::Int64, PATH_FIG_para_x::String, PATH_FIG_para_ν::String; Optimal_ν::Int64=0)

    # communication
    filename_x = "fig_optimal_x_by_" * TV
    filename_γ = "fig_optimal_γ_by_" * TV
    filename_1_F = "fig_optimal_1-F_by_" * TV
    filename_F = "fig_optimal_F_by_" * TV
    filename_c = "fig_optimal_c_by_" * TV
    optimal_communication_func!(BP, TV_res, TV, TV_size, TV_grid)
    TV_res = round.(TV_res, digits=dg_f)

    # xlabel for targeted variable (TV)
    if TV == "μ_0"
        xlabel_ = L"HH prior $\mu_0$"
    elseif TV == "μ_0_c"
        xlabel_ = L"CB prior $\mu^c_0$"
    elseif TV == "ω_1"
        xlabel_ = L"Unemployment shock $\omega_1$"
    elseif TV == "ω_2"
        xlabel_ = L"Unemployment shock $\omega_2$"
    end

    # plot optimal communication x
    fig = Figure(fontsize=32, size=(600, 500))
    ax = Axis(fig[1, 1], xlabel=xlabel_)
    ylims!(ax, -0.05, 1.05)
    lines!(ax, TV_grid, TV_res[:, 2], label=L"$x_1$", color=:blue, linestyle=nothing, linewidth=4)
    lines!(ax, TV_grid, TV_res[:, 3], label=L"$x_2$", color=:red, linestyle=:dash, linewidth=4)
    axislegend(position=:cb, nbanks=2, patchsize=(40, 20))
    fig
    save(PATH_FIG_para_x * FL * filename_x * ".pdf", fig)
    save(PATH_FIG_para_x * FL * filename_x * ".png", fig)

    # plot inflation surprise γ * (x_e - x_r)
    if TV == "μ_0"
        γ_μ_1_ω_1 = BP.γ .* (x_e.(TV_grid, TV_res[:, 6], BP.x_T, BP.ν_1, BP.ν_2, BP.θ) .- x_r(1, BP.x_T, BP.ν_1, BP.ν_2))
        γ_μ_1_ω_2 = BP.γ .* (x_e.(TV_grid, TV_res[:, 6], BP.x_T, BP.ν_1, BP.ν_2, BP.θ) .- x_r(2, BP.x_T, BP.ν_1, BP.ν_2))
        γ_μ_2_ω_1 = BP.γ .* (x_e.(TV_grid, TV_res[:, 7], BP.x_T, BP.ν_1, BP.ν_2, BP.θ) .- x_r(1, BP.x_T, BP.ν_1, BP.ν_2))
        γ_μ_2_ω_2 = BP.γ .* (x_e.(TV_grid, TV_res[:, 7], BP.x_T, BP.ν_1, BP.ν_2, BP.θ) .- x_r(2, BP.x_T, BP.ν_1, BP.ν_2))
    else
        γ_μ_1_ω_1 = BP.γ .* (x_e.(BP.μ_0, TV_res[:, 6], BP.x_T, BP.ν_1, BP.ν_2, BP.θ) .- x_r(1, BP.x_T, BP.ν_1, BP.ν_2))
        γ_μ_1_ω_2 = BP.γ .* (x_e.(BP.μ_0, TV_res[:, 6], BP.x_T, BP.ν_1, BP.ν_2, BP.θ) .- x_r(2, BP.x_T, BP.ν_1, BP.ν_2))
        γ_μ_2_ω_1 = BP.γ .* (x_e.(BP.μ_0, TV_res[:, 7], BP.x_T, BP.ν_1, BP.ν_2, BP.θ) .- x_r(1, BP.x_T, BP.ν_1, BP.ν_2))
        γ_μ_2_ω_2 = BP.γ .* (x_e.(BP.μ_0, TV_res[:, 7], BP.x_T, BP.ν_1, BP.ν_2, BP.θ) .- x_r(2, BP.x_T, BP.ν_1, BP.ν_2))
    end
    fig = Figure(fontsize=32, size=(600, 500))
    ax = Axis(fig[1, 1], xlabel=xlabel_)
    lines!(ax, TV_grid, γ_μ_1_ω_1, label=L"(\mu_1,\,\omega_1)", color=:blue, linestyle=nothing, linewidth=4)
    lines!(ax, TV_grid, γ_μ_1_ω_2, label=L"(\mu_1,\,\omega_2)", color=:red, linestyle=:dash, linewidth=4)
    lines!(ax, TV_grid, γ_μ_2_ω_1, label=L"(\mu_2,\,\omega_1)", color=:black, linestyle=:dot, linewidth=4)
    lines!(ax, TV_grid, γ_μ_2_ω_2, label=L"(\mu_2,\,\omega_2)", color=:green, linestyle=:dashdot, linewidth=4)
    if BP.γ == 10.0
        axislegend(position=(1.0, 0.25), nbanks=2, patchsize=(40, 20))
    elseif BP.γ == 1.0
        axislegend(position=:rb, nbanks=2, patchsize=(40, 20))
    end
    fig
    save(PATH_FIG_para_x * FL * filename_γ * ".pdf", fig)
    save(PATH_FIG_para_x * FL * filename_γ * ".png", fig)

    # share of attentive receivers
    fig = Figure(fontsize=32, size=(600, 500))
    ax = Axis(fig[1, 1], xlabel=xlabel_)
    ylims!(ax, -0.05, 1.05)
    lines!(ax, TV_grid, TV_res[:, 8], label=L"$1 - \delta F(x_1,\,x_2,\,\mu_0,\,a,\,b)$", color=:blue, linestyle=nothing, linewidth=4)
    axislegend(position=:cb, nbanks=2, patchsize=(40, 20))
    fig
    save(PATH_FIG_para_x * FL * filename_1_F * ".pdf", fig)
    save(PATH_FIG_para_x * FL * filename_1_F * ".png", fig)

    # share of inattentive receivers
    fig = Figure(fontsize=32, size=(600, 500))
    ax = Axis(fig[1, 1], xlabel=xlabel_)
    ylims!(ax, -0.05, 1.05)
    lines!(ax, TV_grid, TV_res[:, 9], label=L"$ \delta F(x_1,\,x_2,\,\mu_0,\,a,\,b)$", color=:blue, linestyle=nothing, linewidth=4)
    axislegend(position=:cb, nbanks=2, patchsize=(40, 20))
    fig
    save(PATH_FIG_para_x * FL * filename_F * ".pdf", fig)
    save(PATH_FIG_para_x * FL * filename_F * ".png", fig)

    # information cost
    fig = Figure(fontsize=32, size=(600, 500))
    ax = Axis(fig[1, 1], xlabel=xlabel_)
    ylims!(ax, -0.05, 1.05)
    lines!(ax, TV_grid, TV_res[:, 10], label=L"$c(x_1,\,x_2,\,\mu_0)$", color=:blue, linestyle=nothing, linewidth=4)
    axislegend(position=:cb, nbanks=2, patchsize=(40, 20))
    fig
    save(PATH_FIG_para_x * FL * filename_c * ".pdf", fig)
    save(PATH_FIG_para_x * FL * filename_c * ".png", fig)
    
    #========================#
    # distribution functions #
    #========================#
    f(x_1, x_2, μ, a, b) = a * b * (c(x_1, x_2, μ)^(a - 1.0)) * (1.0 - c(x_1, x_2, μ)^a)^(b - 1.0)
    F(x_1, x_2, μ, a, b) = 1.0 - (1.0 - c(x_1, x_2, μ)^a)^b

    if Optimal_ν == 1

        # flexibility
        filename_x = "fig_optimal_x_by_" * TV
        filename_γ = "fig_optimal_γ_by_" * TV
        filename_ν = "fig_optimal_ν_by_" * TV
        TV_res_ν = zeros(TV_size, 8)
        optimal_flexibility_func!(BP, TV_res_ν, TV, TV_size, TV_grid)
        TV_res_ν = round.(TV_res_ν, digits=dg_f)

        # plot optimal communication x
        fig = Figure(fontsize=32, size=(600, 500))
        ax = Axis(fig[1, 1], xlabel=xlabel_)
        ylims!(ax, -0.05, 1.05)
        lines!(ax, TV_grid, TV_res_ν[:, 2], label=L"$x_1$", color=:blue, linestyle=nothing, linewidth=4)
        lines!(ax, TV_grid, TV_res_ν[:, 3], label=L"$x_2$", color=:red, linestyle=:dash, linewidth=4)
        axislegend(position=:cb, nbanks=2, patchsize=(40, 20))
        fig
        save(PATH_FIG_para_ν * FL * filename_x * ".pdf", fig)
        save(PATH_FIG_para_ν * FL * filename_x * ".png", fig)

        # plot optimal flexibility ν
        fig = Figure(fontsize=32, size=(600, 500))
        ax = Axis(fig[1, 1], xlabel=xlabel_)
        lines!(ax, TV_grid, TV_res_ν[:, 4], label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=4)
        lines!(ax, TV_grid, TV_res_ν[:, 5], label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=4)
        axislegend(position=:cb, nbanks=2, patchsize=(40, 20))
        fig
        save(PATH_FIG_para_ν * FL * filename_ν * ".pdf", fig)
        save(PATH_FIG_para_ν * FL * filename_ν * ".png", fig)
    end

    return nothing
end

#=============================#
# all counterfactuals results #
#=============================#
function solve_function()
    # benchmark
    # a_b_grid = [(1.0, 1.0)]
    # γ_grid = [10.0]
    # α_grid = [1.0]
    # θ_grid = [1.0]
    # δ_grid = [0.5]
    # μ_0_grid = [0.5]
    # μ_0_c_grid = [0.5]
    # ω_1_grid = [1.0]
    # ω_2_grid = [-1.0]

    a_b_grid = [(1.0, 1.0), (2.0, 5.0), (5.0, 2.0)]
    γ_grid = [10.0, 1.0]
    α_grid = [1.0]
    θ_grid = [1.0, 0.5]
    δ_grid = [0.5, 1.0, 0.0]
    μ_0_grid = [0.5, 0.1]
    μ_0_c_grid = [0.5]
    ω_1_grid = [1.0, 2.0]
    ω_2_grid = [-1.0, -2.0]

    all_combinations = Iterators.product(a_b_grid, γ_grid, α_grid, θ_grid, δ_grid, μ_0_grid, μ_0_c_grid, ω_1_grid, ω_2_grid)
    all_combinations_size = length(all_combinations)
    all_combinations_i = 1

    # plot Kumaraswamy distribution with (a, b)
    f(c, a, b) = a * b * (c^(a - 1.0)) * (1.0 - c^a)^(b - 1.0)
    F(c, a, b) = 1.0 - (1.0 - c^a)^b
    c_grid = collect(0.0:0.0005:1.0)
    color_ = [:blue, :red, :black]
    linestyle_ = [nothing, :dash, :dot]

    # pdf
    i = 1
    fig = Figure(fontsize=32, size=(600, 500))
    ax = Axis(fig[1, 1], xlabel=L"c")
    for (a_i, b_i) in a_b_grid
        lines!(ax, c_grid, f.(c_grid, a_i, b_i), label=L"$a=%$a_i,\,b=%$b_i$", color=color_[i], linestyle=linestyle_[i], linewidth=4)
        i += 1
    end
    axislegend(position=:lt, nbanks=1, patchsize=(40, 20))
    fig
    save(PATH_FIG * FL * "Kumaraswamy_pdf.pdf", fig)
    save(PATH_FIG * FL * "Kumaraswamy_pdf.png", fig)

    # cdf
    i = 1
    fig = Figure(fontsize=32, size=(600, 500))
    ax = Axis(fig[1, 1], xlabel=L"c")
    for (a_i, b_i) in a_b_grid
        lines!(ax, c_grid, F.(c_grid, a_i, b_i), label=L"$a=%$a_i,\,b=%$b_i$", color=color_[i], linestyle=linestyle_[i], linewidth=4)
        i += 1
    end
    axislegend(position=:lt, nbanks=1, patchsize=(40, 20))
    fig
    save(PATH_FIG * FL * "Kumaraswamy_cdf.pdf", fig)
    save(PATH_FIG * FL * "Kumaraswamy_cdf.png", fig)

    # containers
    μ_grid_ = collect(0.001:0.0005:0.999)
    μ_size_ = length(μ_grid_)
    μ_res_ = zeros(μ_size_, 10)
    ω_grid_ = collect(0.5:0.0005:1.5)
    ω_size_ = length(ω_grid_)
    ω_res_ = zeros(ω_size_, 10)

    # loop over all parameter space
    for ((a_i, b_i), γ_i, α_i, θ_i, δ_i, μ_0_i, μ_0_c_i, ω_1_i, ω_2_i) in all_combinations

        # construct parameter bundle
        println("task: $all_combinations_i / $all_combinations_size")
        println("(a, b) = ($a_i, $b_i) => (γ, α) = ($γ_i, $α_i) => (θ, δ) = ($θ_i, $δ_i) => (μ_0, μ_0_c, ω_1, ω_2) = ($μ_0_i, $μ_0_c_i, $ω_1_i, $ω_2_i)")
        BP = Benchmark_Parameters(a=a_i, b=b_i, γ=γ_i, α=α_i, θ=θ_i, δ=δ_i, μ_0=μ_0_i, μ_0_c=μ_0_c_i, ω_1=ω_1_i, ω_2=ω_2_i)

        # create file directories
        PATH_FIG_para = mkpath(PATH_FIG * FL * "a=$(round(BP.a,digits=dg_p))_b=$(round(BP.b,digits=dg_p))" * FL * "γ=$(round(BP.γ,digits=dg_p))_α=$(round(BP.α,digits=dg_p))" * FL * "θ=$(round(BP.θ,digits=dg_p))_δ=$(round(BP.δ,digits=dg_p))" * FL * "μ_0=$(round(BP.μ_0,digits=dg_p))_μ_0_c=$(round(BP.μ_0_c,digits=dg_p))_ω_1=$(round(BP.ω_1,digits=dg_p))_ω_2=$(round(BP.ω_2,digits=dg_p))")
        PATH_FIG_para_x = mkpath(PATH_FIG_para * FL * "communication")
        PATH_FIG_para_ν = mkpath(PATH_FIG_para * FL * "flexibility")

        # compute comparative communication and flexibility results
        comparative_result_function!(BP, "μ_0", μ_size_, μ_grid_, μ_res_, dg_f, PATH_FIG_para_x, PATH_FIG_para_ν)
        GC.gc()
        comparative_result_function!(BP, "μ_0_c", μ_size_, μ_grid_, μ_res_, dg_f, PATH_FIG_para_x, PATH_FIG_para_ν)
        GC.gc()
        comparative_result_function!(BP, "ω_1", ω_size_, ω_grid_, ω_res_, dg_f, PATH_FIG_para_x, PATH_FIG_para_ν)
        GC.gc()
        comparative_result_function!(BP, "ω_2", ω_size_, -ω_grid_, ω_res_, dg_f, PATH_FIG_para_x, PATH_FIG_para_ν)
        GC.gc()

        # output parameters and execution time
        open(PATH_FIG_para * FL * "output.txt", "w") do io
            println(io, "Program name: ", split(@__FILE__, FL)[end])
            println(io, "")
            println(io, BP)
            println(io, "Finished: $(Dates.now())")
            println(io, "")
        end

        # update iteration counter
        all_combinations_i += 1
    end

    return nothing
end
solve_function()

# for (a_i, b_i) in a_b_grid
#     for γ_i in γ_grid, α_i in α_grid
#         for θ_i in θ_grid, δ_i in δ_grid
#             for μ_0_i in μ_0_grid, μ_0_c_i in μ_0_c_grid, ω_1_i in ω_1_grid, ω_2_i in ω_2_grid
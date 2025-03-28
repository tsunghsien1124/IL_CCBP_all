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

#==============#
# Housekeeping #
#==============#
PWD = pwd()
VER = "V9"
if Sys.iswindows()
    FL = "\\"
else
    FL = "/"
end
PATH = mkpath(PWD * FL * VER)
PATH_FIG = mkpath(PATH * FL * "Figures")

#==============#
# BP functions #
#==============#
τ_1(x_1, x_2, μ) = x_1 * μ + (1.0 - x_2) * (1.0 - μ)
μ_1(x_1, x_2, μ) = x_1 * μ / τ_1(x_1, x_2, μ)
τ_2(x_1, x_2, μ) = (1.0 - x_1) * μ + x_2 * (1.0 - μ)
μ_2(x_1, x_2, μ) = (1.0 - x_1) * μ / τ_2(x_1, x_2, μ)
H(μ) = -(μ * log(μ) + (1.0 - μ) * log((1.0 - μ)))
c(x_1, x_2, μ) = (1.0 / log(2.0)) * (H(μ) - τ_1(x_1, x_2, μ) * H(μ_1(x_1, x_2, μ)) - τ_2(x_1, x_2, μ) * H(μ_2(x_1, x_2, μ)))

#=====================#
# inflation functions #
#=====================#
x_r(ω_i, x_T, ν_1, ν_2) = ω_i == 1 ? x_T + ν_1 : x_T - ν_2
x_e(μ, x_T, ν_1, ν_2) = μ * x_r(1, x_T, ν_1, ν_2) + (1.0 - μ) * x_r(2, x_T, ν_1, ν_2) # x_T + ν * (2.0 * μ - 1.0) if symmetric ν

#========================#
# CB objective functions #
#========================#
obj_CB_μ_0(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2) = δ * c(x_1, x_2, μ_0) * (μ_0_c * (ω_1 + γ * (x_e(μ_0, x_T, ν_1, ν_2) - x_r(1, x_T, ν_1, ν_2)))^2.0 + (1.0 - μ_0_c) * (ω_2 + γ * (x_e(μ_0, x_T, ν_1, ν_2) - x_r(2, x_T, ν_1, ν_2)))^2.0)
obj_CB_1(x_1, x_2, μ_0, μ_0_c, ω_1, δ, γ, x_T, ν_1, ν_2) = (1.0 - δ * c(x_1, x_2, μ_0)) * μ_0_c * (x_1 * (ω_1 + γ * (x_e(μ_1(x_1, x_2, μ_0), x_T, ν_1, ν_2) - x_r(1, x_T, ν_1, ν_2)))^2.0 + (1.0 - x_1) * (ω_1 + γ * (x_e(μ_2(x_1, x_2, μ_0), x_T, ν_1, ν_2) - x_r(1, x_T, ν_1, ν_2)))^2.0)
obj_CB_2(x_1, x_2, μ_0, μ_0_c, ω_2, δ, γ, x_T, ν_1, ν_2) = (1.0 - δ * c(x_1, x_2, μ_0)) * (1.0 - μ_0_c) * ((1.0 - x_2) * (ω_2 + γ * (x_e(μ_1(x_1, x_2, μ_0), x_T, ν_1, ν_2) - x_r(2, x_T, ν_1, ν_2)))^2.0 + x_2 * (ω_2 + γ * (x_e(μ_2(x_1, x_2, μ_0), x_T, ν_1, ν_2) - x_r(2, x_T, ν_1, ν_2)))^2.0)
obj_CB_x(μ_0_c, x_T, ν_1, ν_2, α) = α * (μ_0_c * (x_r(1, x_T, ν_1, ν_2) - x_T)^2 + (1.0 - μ_0_c) * (x_r(2, x_T, ν_1, ν_2) - x_T)^2)
obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α) = obj_CB_μ_0(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2) + obj_CB_1(x_1, x_2, μ_0, μ_0_c, ω_1, δ, γ, x_T, ν_1, ν_2) + obj_CB_2(x_1, x_2, μ_0, μ_0_c, ω_2, δ, γ, x_T, ν_1, ν_2) + obj_CB_x(μ_0_c, x_T, ν_1, ν_2, α)

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
    γ::Float64 = 1.0
    x_T::Float64 = 2.0
    ν_1::Float64 = 1.0
    ν_2::Float64 = 1.0
    α::Float64 = 0.0
    ϵ_x::Float64 = 1E-8
    ϵ_x_p::Float64 = 1E-1
    ϵ_tol::Float64 = 1E-10
end
BP = Benchmark_Parameters()
PATH_FIG_para = mkpath(PATH_FIG * FL * "γ=$(round(BP.γ,digits=1))-μ_0=$(round(BP.μ_0,digits=1))-α=$(round(BP.α,digits=1))")

#==================#
# benchmark result #
#==================#
# function optimal_x_func(μ_0::Float64, μ_0_c::Float64, ω_1::Float64, ω_2::Float64, δ::Float64, γ::Float64, x_T::Float64, ν_1::Float64, ν_2::Float64, α::Float64, ϵ_x::Float64, ϵ_x_p::Float64, ϵ_tol::Float64)
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     set_attribute(model, "tol", ϵ_tol)
#     @variable(model, ϵ_x <= x_1 <= (1.0 * ϵ_x_p - ϵ_x), start = 1.0 * ϵ_x_p / 2.0)
#     @variable(model, ϵ_x <= x_2 <= (1.0 * ϵ_x_p - ϵ_x), start = 1.0 * ϵ_x_p / 2.0)
#     @constraint(model, c1, x_1 + x_2 >= 1.0 * ϵ_x_p)
#     _obj_CB(x_1, x_2) = obj_CB(x_1 / ϵ_x_p, x_2 / ϵ_x_p, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α)
#     @objective(model, Min, _obj_CB(x_1, x_2))
#     optimize!(model)
#     @assert is_solved_and_feasible(model)
#     return objective_value(model), value(x_1) / ϵ_x_p, value(x_2) / ϵ_x_p
# end
# obj_opt, x_1_opt, x_2_opt = optimal_x_func(BP.μ_0, BP.μ_0_c, BP.ω_1, BP.ω_2, BP.δ, BP.γ, BP.x_T, BP.ν_1, BP.ν_2, BP.α, BP.ϵ_x, BP.ϵ_x_p, BP.ϵ_tol)
# println("Find the minimum of $obj_opt at (x_1, x_2) = ($x_1_opt, $x_2_opt)")

function optimal_x_ν_func(μ_0::Float64, μ_0_c::Float64, ω_1::Float64, ω_2::Float64, δ::Float64, γ::Float64, x_T::Float64, α::Float64, ϵ_x::Float64, ϵ_x_p::Float64, ϵ_tol::Float64)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    @variable(model, ϵ_x <= x_1 <= (1.0 * ϵ_x_p - ϵ_x), start = 1.0 * ϵ_x_p / 2.0)
    @variable(model, ϵ_x <= x_2 <= (1.0 * ϵ_x_p - ϵ_x), start = 1.0 * ϵ_x_p / 2.0)
    @variable(model, ν_1 >= ϵ_x, start = ϵ_x)
    @variable(model, ν_2 >= ϵ_x, start = ϵ_x)
    @constraint(model, c1, x_1 + x_2 >= 1.0 * ϵ_x_p)
    _obj_CB(x_1, x_2, ν_1, ν_2) = obj_CB(x_1 / ϵ_x_p, x_2 / ϵ_x_p, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α)
    @objective(model, Min, _obj_CB(x_1, x_2, ν_1, ν_2))
    optimize!(model)
    @assert is_solved_and_feasible(model)
    return objective_value(model), value(x_1) / ϵ_x_p, value(x_2) / ϵ_x_p, value(ν_1), value(ν_2)
end
obj_opt, x_1_opt, x_2_opt, ν_1_opt, ν_2_opt = optimal_x_ν_func(BP.μ_0, BP.μ_0_c, BP.ω_1, BP.ω_2, BP.δ, BP.γ, BP.x_T, BP.α, BP.ϵ_x, BP.ϵ_x_p, BP.ϵ_tol)
println("Find the minimum of $obj_opt at (x_1, x_2, ν_1, ν_2) = ($x_1_opt, $x_2_opt, $ν_1_opt, $ν_2_opt)")

#===========================================#
# handy functions for optimal flexibility ν #
#===========================================#
function optimal_flexibility_func!(BP::Benchmark_Parameters, res::Array{Float64,2}, TA::String, TA_size::Int64, TA_grid::Vector{Float64})

    # unpack benchmark parameters
    @unpack μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α, ϵ_x, ϵ_x_p, ϵ_tol = BP

    # timer
    pp = Progress(TA_size)
    update!(pp, 0)
    jj = Threads.Atomic{Int}(0)
    ll = Threads.SpinLock()

    # start looping
    Threads.@threads for TA_i in 1:TA_size

        # create the parameter dictionary
        obj_CB_para = Dict([("μ_0", μ_0), ("μ_0_c", μ_0_c), ("ω_1", ω_1), ("ω_2", ω_2), ("δ", δ), ("γ", γ), ("x_T", x_T), ("ν_1", ν_1), ("ν_2", ν_2), ("α", α), ("ϵ_x", ϵ_x), ("ϵ_x_p", ϵ_x_p), ("ϵ_tol", ϵ_tol)])
        obj_CB_para[TA] = TA_grid[TA_i]

        # slove CB's optimization problem for (x_1, x_2)
        # obj_opt, x_1_opt, x_2_opt = optimal_x_func(obj_CB_para["μ_0"], obj_CB_para["μ_0_c"], obj_CB_para["ω_1"], obj_CB_para["ω_2"], obj_CB_para["δ"], obj_CB_para["γ"], obj_CB_para["x_T"], obj_CB_para["ν_1"], obj_CB_para["ν_2"], obj_CB_para["α"], obj_CB_para["ϵ_x"], obj_CB_para["ϵ_x_p"], obj_CB_para["ϵ_tol"])
        obj_opt, x_1_opt, x_2_opt, ν_1_opt, ν_2_opt = optimal_x_ν_func(obj_CB_para["μ_0"], obj_CB_para["μ_0_c"], obj_CB_para["ω_1"], obj_CB_para["ω_2"], obj_CB_para["δ"], obj_CB_para["γ"], obj_CB_para["x_T"], obj_CB_para["α"], obj_CB_para["ϵ_x"], obj_CB_para["ϵ_x_p"], obj_CB_para["ϵ_tol"])

        # save results
        @inbounds res[TA_i, 1] = obj_opt
        @inbounds res[TA_i, 2] = x_1_opt
        @inbounds res[TA_i, 3] = x_2_opt
        # @inbounds res[TA_i, 4] = ν_1 
        # @inbounds res[TA_i, 5] = ν_2
        @inbounds res[TA_i, 4] = ν_1_opt
        @inbounds res[TA_i, 5] = ν_2_opt
        @inbounds res[TA_i, 6] = μ_1(x_1_opt, x_2_opt, obj_CB_para["μ_0"])
        @inbounds res[TA_i, 7] = μ_2(x_1_opt, x_2_opt, obj_CB_para["μ_0"])
        @inbounds res[TA_i, 8] = 1.0 - obj_CB_para["δ"] * c(x_1_opt, x_2_opt, obj_CB_para["μ_0"])

        # update timer
        Threads.atomic_add!(jj, 1)
        Threads.lock(ll)
        update!(pp, jj[])
        Threads.unlock(ll)
    end
    return nothing
end

#========================#
# benchmark result - μ_0 #
#========================#
filename_x = "fig_optimal_x_by_μ_0"
filename_ν = "fig_optimal_ν_by_μ_0"
μ_0_grid = collect(0.001:0.0005:0.999)
μ_0_size = length(μ_0_grid)
μ_0_res = zeros(μ_0_size, 8)
optimal_flexibility_func!(BP, μ_0_res, "μ_0", μ_0_size, μ_0_grid)
# μ_0_res = round.(μ_0_res, digits=3)

# plot optimal communication x
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"HH prior $\mu_0$")
lines!(ax, μ_0_grid, μ_0_res[:,2], label=L"$x_1$", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_grid, μ_0_res[:,3], label=L"$x_2$", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=:rt, nbanks=2, patchsize=(40, 20))
fig
save(PATH_FIG_para * FL * filename_x * ".pdf", fig)
save(PATH_FIG_para * FL * filename_x * ".png", fig)

# plot optimal flexibility ν
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"HH prior $\mu_0$")
lines!(ax, μ_0_grid, μ_0_res[:,4], label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_grid, μ_0_res[:,5], label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=:rt, nbanks=2, patchsize=(40, 20))
fig
save(PATH_FIG_para * FL * filename_ν * ".pdf", fig)
save(PATH_FIG_para * FL * filename_ν * ".png", fig)

#==========================#
# benchmark result - μ_0_c #
#==========================#
filename_x = "fig_optimal_x_by_μ_0_c"
filename_ν = "fig_optimal_ν_by_μ_0_c"
filename_γ = "fig_optimal_γ_by_μ_0_c"
μ_0_c_grid = collect(0.001:0.0005:0.999)
μ_0_c_size = length(μ_0_c_grid)
μ_0_c_res = zeros(μ_0_c_size, 8)
optimal_flexibility_func!(BP, μ_0_c_res, "μ_0_c", μ_0_c_size, μ_0_c_grid)
# μ_0_c_res = round.(μ_0_c_res, digits=3)

# plot optimal communication x
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"CB prior $\mu^c_0$")
lines!(ax, μ_0_c_grid, μ_0_c_res[:,2], label=L"$x_1$", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_c_grid, μ_0_c_res[:,3], label=L"$x_2$", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=:rt, nbanks=2, patchsize=(40, 20))
fig
save(PATH_FIG_para * FL * filename_x * ".pdf", fig)
save(PATH_FIG_para * FL * filename_x * ".png", fig)

# plot optimal flexibility ν
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"CB prior $\mu^c_0$")
lines!(ax, μ_0_c_grid, μ_0_c_res[:,4], label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_c_grid, μ_0_c_res[:,5], label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=:rt, nbanks=2, patchsize=(40, 20))
fig
save(PATH_FIG_para * FL * filename_ν * ".pdf", fig)
save(PATH_FIG_para * FL * filename_ν * ".png", fig)

# plot inflation surprise γ * (x_e - x_r)
γ_μ_1_ω_1 = BP.γ .* (x_e.(μ_0_c_res[:, 6], BP.x_T, BP.ν_1, BP.ν_2) .- x_r(1, BP.x_T, BP.ν_1, BP.ν_2))
γ_μ_1_ω_2 = BP.γ .* (x_e.(μ_0_c_res[:, 6], BP.x_T, BP.ν_1, BP.ν_2) .- x_r(2, BP.x_T, BP.ν_1, BP.ν_2))
γ_μ_2_ω_1 = BP.γ .* (x_e.(μ_0_c_res[:, 7], BP.x_T, BP.ν_1, BP.ν_2) .- x_r(1, BP.x_T, BP.ν_1, BP.ν_2))
γ_μ_2_ω_2 = BP.γ .* (x_e.(μ_0_c_res[:, 7], BP.x_T, BP.ν_1, BP.ν_2) .- x_r(2, BP.x_T, BP.ν_1, BP.ν_2))
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"CB prior $\mu^c_0$")
lines!(ax, μ_0_c_grid, γ_μ_1_ω_1, label=L"(μ_1, \omega_1)", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_c_grid, γ_μ_1_ω_2, label=L"(μ_1, \omega_2)", color=:red, linestyle=:dash, linewidth=4)
lines!(ax, μ_0_c_grid, γ_μ_2_ω_1, label=L"(μ_2, \omega_1)", color=:black, linestyle=:dot, linewidth=4)
lines!(ax, μ_0_c_grid, γ_μ_2_ω_2, label=L"(μ_2, \omega_2)", color=:green, linestyle=:dashdot, linewidth=4)
axislegend(position=:lc, nbanks=2, patchsize=(40, 20))
fig
save(PATH_FIG_para * FL * filename_γ * ".pdf", fig)
save(PATH_FIG_para * FL * filename_γ * ".png", fig)

#========================#
# benchmark result - ω_1 #
#========================#
filename_x = "fig_optimal_x_by_ω_1"
filename_ν = "fig_optimal_ν_by_ω_1"
ω_1_grid = collect(0.5:0.0005:1.5)
ω_1_size = length(ω_1_grid)
ω_1_res = zeros(ω_1_size, 8)
optimal_flexibility_func!(BP, ω_1_res, "ω_1", ω_1_size, ω_1_grid)
# ω_1_res = round.(ω_1_res, digits=3)

# plot optimal communication x
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"Unemployment shock $\omega_1$")
lines!(ax, ω_1_grid, ω_1_res[:,2], label=L"$x_1$", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, ω_1_grid, ω_1_res[:,3], label=L"$x_2$", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=:rt, nbanks=2, patchsize=(40, 20))
fig
save(PATH_FIG_para * FL * filename_x * ".pdf", fig)
save(PATH_FIG_para * FL * filename_x * ".png", fig)

# plot optimal flexibility ν
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"Unemployment shock $\omega_1$")
lines!(ax, ω_1_grid, ω_1_res[:,4], label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, ω_1_grid, ω_1_res[:,5], label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=:lt, nbanks=2, patchsize=(40, 20))
fig
save(PATH_FIG_para * FL * filename_ν * ".pdf", fig)
save(PATH_FIG_para * FL * filename_ν * ".png", fig)

#========================#
# benchmark result - ω_2 #
#========================#
filename_x = "fig_optimal_x_by_ω_2"
filename_ν = "fig_optimal_ν_by_ω_2"
ω_2_grid = collect(-1.5:0.0005:-0.5)
ω_2_size = length(ω_2_grid)
ω_2_res = zeros(ω_2_size, 8)
optimal_flexibility_func!(BP, ω_2_res, "ω_2", ω_2_size, ω_2_grid)
# ω_2_res = round.(ω_2_res, digits=3)

# plot optimal communication x
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"Unemployment shock $\omega_2$")
lines!(ax, ω_2_grid, ω_2_res[:,2], label=L"$x_1$", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, ω_2_grid, ω_2_res[:,3], label=L"$x_2$", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=:rt, nbanks=2, patchsize=(40, 20))
fig
save(PATH_FIG_para * FL * filename_x * ".pdf", fig)
save(PATH_FIG_para * FL * filename_x * ".png", fig)

# plot optimal flexibility ν
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"Unemployment shock $\omega_2$")
lines!(ax, ω_2_grid, ω_2_res[:,4], label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, ω_2_grid, ω_2_res[:,5], label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=:rt, nbanks=2, patchsize=(40, 20))
fig
save(PATH_FIG_para * FL * filename_ν * ".pdf", fig)
save(PATH_FIG_para * FL * filename_ν * ".png", fig)

#======================================#
# output parameters and execution time #
#======================================#
open(PATH_FIG_para * FL * "output.txt", "w") do io
    println(io, "Program name: ", split(@__FILE__, FL)[end])
    println(io, "")
    println(io, BP)
    println(io, "Finished: $(Dates.now())")
    println(io, "")
end
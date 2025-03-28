#=================#
# Import packages #
#=================#
# using Symbolics
# using Latexify
# using Optim
# using NLopt
using JuMP
import Ipopt
using PrettyTables
using GLMakie
using CairoMakie

#=====================#
# benchmark functions #
#=====================#
τ_s_1(x_1, x_2, μ_0) = x_1 * μ_0 + (1.0 - x_2) * (1.0 - μ_0)
τ_s_2(x_1, x_2, μ_0) = (1.0 - x_1) * μ_0 + x_2 * (1.0 - μ_0)
μ_s_1(x_1, x_2, μ_0) = x_1 * μ_0 / τ_s_1(x_1, x_2, μ_0)
μ_s_2(x_1, x_2, μ_0) = x_2 * (1.0 - μ_0) / τ_s_2(x_1, x_2, μ_0)
H(μ) = -(μ * log(μ) + (1.0 - μ) * log((1.0 - μ))) # H(1.0/2.0) == log(2.0)
c(x_1, x_2, μ_0) = (1.0 / log(2.0)) * (H(μ_0) - τ_s_1(x_1, x_2, μ_0) * H(μ_s_1(x_1, x_2, μ_0)) - τ_s_2(x_1, x_2, μ_0) * H(μ_s_2(x_1, x_2, μ_0)))

ind_heatmap = 0
if ind_heatmap == 1
    #===================#
    # macro application #
    #===================#
    # visualize the objective function to get a sense
    ω_w = 1.0
    ω_s = -1.0
    μ_0 = 1.0 / 2.0
    μ_0_CB = μ_0
    x_T = 2
    γ = -10.0
    ν = 1
    x_realized = [x_T + ν, x_T - ν]
    x_expected(μ) = x_T + ν * (2.0 * μ - 1.0) # note that it holds only if μ is the belief for the first state!!
    obj_CB_μ_0(x_1, x_2) = μ_0_CB * c(x_1, x_2, μ_0) * (ω_w + γ * (x_expected(μ_0) - x_realized[1]))^2.0 + (1.0 - μ_0_CB) * c(x_1, x_2, μ_0) * (ω_s + γ * (x_expected(μ_0) - x_realized[2]))^2.0
    obj_CB_τ_1(x_1, x_2) = τ_s_1(x_1, x_2, μ_0_CB) * (1.0 - c(x_1, x_2, μ_0)) * (μ_0_CB * (ω_w + γ * (x_expected(μ_s_1(x_1, x_2, μ_0)) - x_realized[1]))^2.0 + (1.0 - μ_0_CB) * (ω_s + γ * (x_expected(μ_s_1(x_1, x_2, μ_0)) - x_realized[2]))^2.0)
    obj_CB_τ_2(x_1, x_2) = τ_s_2(x_1, x_2, μ_0_CB) * (1.0 - c(x_1, x_2, μ_0)) * (μ_0_CB * (ω_w + γ * (x_expected(1.0 - μ_s_2(x_1, x_2, μ_0)) - x_realized[1]))^2.0 + (1.0 - μ_0_CB) * (ω_s + γ * (x_expected(1.0 - μ_s_2(x_1, x_2, μ_0)) - x_realized[2]))^2.0)
    obj_CB(x_1, x_2) = obj_CB_μ_0(x_1, x_2) + obj_CB_τ_1(x_1, x_2) + obj_CB_τ_2(x_1, x_2)
    val_x_1 = collect(0.001:0.001:0.999)
    val_x_2 = collect(0.001:0.001:0.999)
    val_x_1_plus_x_2 = [x_1 + x_2 for x_1 in val_x_1, x_2 in val_x_2]
    val_obj_CB = [obj_CB(x_1, x_2) for x_1 in val_x_1, x_2 in val_x_2]
    val_obj_CB[val_x_1_plus_x_2.<1.0] .= Inf
    global_min = findmin(val_obj_CB)[1]
    # val_x_1[findmin(val_obj_CB)[2][1]]
    # val_x_2[findmin(val_obj_CB)[2][2]]
    ind_global_min = val_obj_CB .≈ global_min
    vec_ind_global_min = findall(ind_global_min .== 1)
    n_vec_ind_global_min = length(vec_ind_global_min)
    ind_global_min_x_1 = zeros(n_vec_ind_global_min)
    ind_global_min_x_2 = zeros(n_vec_ind_global_min)
    for global_min_i = 1:n_vec_ind_global_min
        ind_global_min_x_1[global_min_i] = val_x_1[vec_ind_global_min[global_min_i][1]]
        ind_global_min_x_2[global_min_i] = val_x_2[vec_ind_global_min[global_min_i][2]]
    end
    val_obj_CB[val_obj_CB.==Inf] .= NaN
    # surface(val_x_1, val_x_2, val_obj_CB, axis=(type=Axis3,))
    f = Figure(fontsize=18, size=(800, 600))
    ax = Axis(f[1, 1],
        title=L"Objective Function $U(\pi)$",
        xlabel=L"signal $x_1$",
        ylabel=L"signal $x_2$"
    )
    heatmap!(val_x_1, val_x_2, val_obj_CB)
    scatter!(ind_global_min_x_1, ind_global_min_x_2, color=:red)
    filename = "heatmap_" * "homo_μ_" * "$μ_0" * "_" * "ω_w_" * "$ω_w" * "_" * "ω_s_" * "$ω_s" * ".pdf"
    # save("D://Dropbox//Innocenti and Li//" * filename,f)
    save("C://Users//Tsung-Hsien Li//Dropbox//Innocenti and Li//Figures//" * filename, f)
else
    # loop over parameters of interest
    ω_w = 1 # collect(0.0:1:10)
    n_ω_w = length(ω_w)
    ω_s = -10 # collect(0.0:-1:-10)
    n_ω_s = length(ω_s)
    μ_0 = 1.0 / 2.0
    n_μ_0 = length(μ_0)
    μ_0_CB = collect(0.01:0.01:0.99)
    n_μ_0_CB = length(μ_0_CB)
    x_T = 2
    γ = -10.0
    ν = 1
    x_realized = [x_T + ν, x_T - ν]
    x_expected(μ) = x_T + ν * (2.0 * μ - 1.0) # note that it holds only if μ is the belief for the first state!!
    results = zeros(n_μ_0_CB * n_ω_w * n_ω_s, 11)
    results_i = 1
    for μ_0_i = 1:n_μ_0, μ_0_CB_i = 1:n_μ_0_CB, ω_w_i = 1:n_ω_w, ω_s_i = 1:n_ω_s
        # objective functions 
        obj_CB_μ_0(x_1, x_2) = μ_0_CB[μ_0_CB_i] * c(x_1, x_2, μ_0[μ_0_i]) * (ω_w[ω_w_i] + γ * (x_expected(μ_0[μ_0_i]) - x_realized[1]))^2.0 + (1.0 - μ_0_CB[μ_0_CB_i]) * c(x_1, x_2, μ_0[μ_0_i]) * (ω_s[ω_s_i] + γ * (x_expected(μ_0[μ_0_i]) - x_realized[2]))^2.0
        obj_CB_τ_1(x_1, x_2) = τ_s_1(x_1, x_2, μ_0_CB[μ_0_CB_i]) * (1.0 - c(x_1, x_2, μ_0[μ_0_i])) * (μ_0_CB[μ_0_CB_i] * (ω_w[ω_w_i] + γ * (x_expected(μ_s_1(x_1, x_2, μ_0[μ_0_i])) - x_realized[1]))^2.0 + (1.0 - μ_0_CB[μ_0_CB_i]) * (ω_s[ω_s_i] + γ * (x_expected(μ_s_1(x_1, x_2, μ_0[μ_0_i])) - x_realized[2]))^2.0)
        obj_CB_τ_2(x_1, x_2) = τ_s_2(x_1, x_2, μ_0_CB[μ_0_CB_i]) * (1.0 - c(x_1, x_2, μ_0[μ_0_i])) * (μ_0_CB[μ_0_CB_i] * (ω_w[ω_w_i] + γ * (x_expected(1.0 - μ_s_2(x_1, x_2, μ_0[μ_0_i])) - x_realized[1]))^2.0 + (1.0 - μ_0_CB[μ_0_CB_i]) * (ω_s[ω_s_i] + γ * (x_expected(1.0 - μ_s_2(x_1, x_2, μ_0[μ_0_i])) - x_realized[2]))^2.0)
        obj_CB(x_1, x_2) = obj_CB_μ_0(x_1, x_2) + obj_CB_τ_1(x_1, x_2) + obj_CB_τ_2(x_1, x_2)
        # minimization routine
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        set_attribute(model, "tol", 10^-8)
        @variable(model, x_1, lower_bound = 0.0, upper_bound = 1.0)
        @variable(model, x_2, lower_bound = 0.0, upper_bound = 1.0)
        @constraint(model, c1, x_1 + x_2 >= 1.0)
        @objective(model, Min, obj_CB(x_1, x_2))
        optimize!(model)
        # if !is_solved_and_feasible(model)
        #     error("Solver did not find an optimal solution")
        # end
        # println("got ", objective_value(model), " at ", [value(x_1), value(x_2)])
        results[results_i, 1] = μ_0[μ_0_i]
        results[results_i, 2] = μ_0_CB[μ_0_CB_i]
        results[results_i, 3] = ω_w[ω_w_i]
        results[results_i, 4] = ω_s[ω_s_i]
        # results[results_i, 5] = !is_solved_and_feasible(model)
        results[results_i, 6] = objective_value(model)
        results[results_i, 7] = value(x_1)
        results[results_i, 8] = value(x_2)
        results[results_i, 9] = μ_s_1(results[results_i, 7], results[results_i, 8], results[results_i, 1])
        results[results_i, 10] = μ_s_2(results[results_i, 7], results[results_i, 8], results[results_i, 1])
        results[results_i, 11] = 1.0 - c(results[results_i, 7], results[results_i, 8], results[results_i, 1])
        results_i += 1
    end

    # save results as txt
    # open("D://Dropbox//Innocenti and Li//results.txt", "w") do f
    #     pretty_table(f,results; header=["μ_0", "μ_0_CB", "ω_w", "ω_s", "error", "obj", "x_1", "x_2"])
    # end
    
    # optimal information disclosure π = (x_1, x_2)
    f = Figure(fontsize=32, size=(600, 500))
    ax = Axis(f[1, 1],
        # title="Optimal Information Disclosure",
        xlabel=L"central bank's prior $\mu_0^c$",
        # ylabel=L"signal $x$"
    )
    lines!(ax, results[:, 2], results[:, 7], label=L"x_1", color=:blue, linestyle=nothing, linewidth=4)
    lines!(ax, results[:, 2], results[:, 8], label=L"x_2", color=:red, linestyle=:dash, linewidth=4)
    axislegend(position=:ct)
    filename = "fig_" * "hetero_μ_c_" * "ω_w_" * "$ω_w" * "_" * "ω_s_" * "$ω_s" * ".pdf"
    # save("D://Dropbox//Innocenti and Li//" * filename,f)
    save("C://Users//Tsung-Hsien Li//Dropbox//Innocenti and Li//Figures//" * filename, f)

    # posterior μ_s_1 and μ_s_1 
    f = Figure(fontsize=32, size=(600, 500))
    ax = Axis(f[1, 1],
        # title="Household Posterior",
        xlabel=L"central bank's prior $\mu_0^c$",
        # ylabel=L"belief $μ$"
    )
    lines!(ax, results[:, 2], results[:, 9], label=L"μ_{s_1}", color=:blue, linestyle=nothing, linewidth=4)
    lines!(ax, results[:, 2], 1.0.-results[:, 10], label=L"1-μ_{s_2}", color=:red, linestyle=:dash, linewidth=4)
    lines!(ax, results[:, 2], results[:, 11], label=L"1-c(\pi)", color=:black, linestyle=:dot, linewidth=4)
    axislegend(position=:cb)
    filename = "posterior_" * "hetero_μ_c_" * "ω_w_" * "$ω_w" * "_" * "ω_s_" * "$ω_s" * ".pdf"
    # save("D://Dropbox//Innocenti and Li//" * filename,f)
    save("C://Users//Tsung-Hsien Li//Dropbox//Innocenti and Li//Figures//" * filename, f)

    # share of receivers 1-c(π) 
    # f = Figure(fontsize=32, size=(600, 500))
    # ax = Axis(f[1, 1],
    #     title="Optimal Audience",
    #     xlabel=L"central bank's prior $\mu_0^c$",
    #     ylabel=L"share of attentive HHs $1-c(\pi)$"
    # )
    # lines!(ax, results[:, 2], results[:, 11], color=:blue, linestyle=nothing, linewidth=4)
    # filename = "share_" * "hetero_μ_c_" * "ω_w_" * "$ω_w" * "_" * "ω_s_" * "$ω_s" * ".pdf"
    # # save("D://Dropbox//Innocenti and Li//" * filename,f)
    # save("C://Users//Tsung-Hsien Li//Dropbox//Innocenti and Li//Figures//" * filename, f)

    # all in one!
    # f = Figure(fontsize=18, size=(800, 600))
    # ax = Axis(f[1, 1],
    #     xlabel=L"prior of central bank $\mu_0^c$",
    # )
    # lines!(ax, results[:, 2], results[:, 7], label=L"x_1", linestyle=nothing, linewidth=2)
    # lines!(ax, results[:, 2], results[:, 8], label=L"x_2", linestyle=:dash, linewidth=2)
    # lines!(ax, results[:, 2], results[:, 9], label=L"μ_{s_1}", linestyle=:dot, linewidth=2)
    # lines!(ax, results[:, 2], 1.0 .- results[:, 10], label=L"1-μ_{s_2}", linestyle=:dashdot, linewidth=2)
    # lines!(ax, results[:, 2], results[:, 11], label=L"1-c(\pi)", linestyle=:dashdotdot, linewidth=2)
    # axislegend(position=:ct)
    # filename = "all_" * "hetero_μ_c_" * "ω_w_" * "$ω_w" * "_" * "ω_s_" * "$ω_s" * ".pdf"
    # # save("D://Dropbox//Innocenti and Li//" * filename,f)
    # save("C://Users//Tsung-Hsien Li//Dropbox//Innocenti and Li//Figures//" * filename, f)
end

#===============#
# miscellaneous #
#===============#
# τ_s_1(x) = (1 / 2) * (x_1 + 1.0 - x_2);
# τ_s_2(x) = τ_s_1(x_2, x_1); # (1 / 2) * (x_2 + 1.0 - x_1)
# μ_s_1(x) = x_1 / (x_1 + 1.0 - x_2);
# μ_s_2(x) = μ_s_1(x_2, x_1); # x_2 / (x_2 + 1.0 - x_1)
# H_s_1(x) = -((x_1 * log(x_1) + (1.0 - x_2) * log(1.0 - x_2)) / (x_1 + 1.0 - x_2) - log(x_1 + 1.0 - x_2));
# H_s_2(x) = H_s_1(x_2, x_1); # -((x_2 * ln(x_2) + (1.0 - x_1) * ln(1.0 - x_1)) / (x_2 + 1.0 - x_1) - ln(x_2 + 1.0 - x_1))
# c_π(x) = χ * (log(2.0) - τ_s_1(x) * H_s_1(x) - τ_s_2(x) * H_s_2(x)); # c_π(x, x) if symmetry
# x_expected(μ) = x_T + ν * (2 * μ - 1.0);
# obj_CB(x) = μ_0_CB[1] * (c_π(x) * (ω[1] + γ * (x_expected(μ_0_HH[1]) - x_realized[1]))^2.0 + (1.0 - c_π(x)) * (τ_s_1(x) * (ω[1] + γ * (x_expected(μ_s_1(x)) - x_realized[1])^2.0) + τ_s_2(x) * (ω[1] + γ * (x_expected(μ_s_2(x)) - x_realized[1])^2.0)))
# @variables x_1 x_2
# D_x_1 = Differential(x_1)
# D_x_2 = Differential(x_2)
# expand_derivatives(D_x_1(μ_s_1(x_1,x_2)))
# expand_derivatives(D_x_1(c_π(x)))

#==============#
# verification #
#==============#
# @variables x_1 x_2 μ_0 x χ
# D_x_1 = Differential(x_1)
# dc_x_1(x) = D_x_1(c(x, 1.0/2.0))
# dc_x_1 = expand_derivatives(D_x_1(c(x, μ_0)))
# simplify(H(μ_s_2(x, x, 1/2))) is consistent with Eq (31)
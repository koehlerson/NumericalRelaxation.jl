using NumericalRelaxation, Tensors, BenchmarkTools
import LinearAlgebra
using PGFPlotsX

theme = @pgf {"axis background/.style"="{fill={white!89.803921568!black}}", "x grid style"="{white}", "y grid style"="{white}", "xmajorgrids", "ymajorgrids"}

W_KSD(F) = norm(F) ≥ √2 - 1 ? (1+norm(F)^2) : (2*√2*norm(F))
function W_KSD_rc(F)
    D = norm(det(F))
    ρ = sqrt(norm(F)^2 + 2*D)
    if ρ ≥ 1
        return (1+norm(F)^2) 
    else 
        return 2*(ρ - norm(det(F)))
    end
end
W_multi(F::Tensor{2,dim}) where dim = (norm(F)-1)^2
W_multi_rc(F::Tensor{2,dim}) where dim = norm(F) ≤ 1 ? 0.0 : (norm(F)-1)^2
Ns = Int[]
W_KSD_values = Float64[]
W_multi_values = Float64[]
KSD_timings = []
Multi_timings = []
F_start = ones(Tensor{2,2}) * -3
F_stop = ones(Tensor{2,2}) * 3
F_start_mw = ones(Tensor{2,2}) * -3
F_stop_mw = ones(Tensor{2,2}) * 3

for N in (10, 50, 100, 300, 500, 1000, 3000, 5000, 10_000, 50_000, 100_000, 500_000)
    cs = HROC(F_start_mw, F_stop_mw; GLcheck=false, n_convexpoints=N, maxlevel=10)
    buffer = build_buffer(cs)
    F = zero(Tensor{2,2}) # NOTE: was zero(Tensor{2,3}), which DimensionMismatches against the 2D grid

    bt = convexify(cs,buffer,W_multi,F)
    𝔸, 𝐏, W_val = NumericalRelaxation.eval(bt,W_multi)
    bench = @benchmark convexify($cs,$buffer,$W_multi,$F) evals=20 setup=(cs=HROC(F_start_mw, F_stop_mw; GLcheck=false, n_convexpoints=$N, maxlevel=10); buffer = build_buffer(cs))
    push!(Multi_timings,bench)
    push!(W_multi_values, W_val - W_multi_rc(F))

    cs = HROC(F_start, F_stop; GLcheck=false, n_convexpoints=N, maxlevel=10)
    buffer = build_buffer(cs)
    F = Tensor{2,2}([0.2 0.1; 0.1 0.3])
    bt = convexify(cs,buffer,W_KSD,F)
    𝔸, 𝐏, W_val = NumericalRelaxation.eval(bt,W_KSD)
    bench = @benchmark convexify($cs,$buffer,$W_KSD,$F) evals =30 setup=(cs=HROC(F_start, F_stop; GLcheck=false, n_convexpoints=$N, maxlevel=10); buffer = build_buffer(cs))
    push!(KSD_timings,bench)
    push!(Ns, N)
    push!(W_KSD_values, W_val - W_KSD_rc(F))
end

@pgf gp_KSD = GroupPlot({group_style = { group_size = "2 by 1",  "horizontal sep"="2cm",},
                                    height = "6cm", width = "6cm",});
@pgf gp_multi = GroupPlot({group_style = { group_size = "2 by 1",  "horizontal sep"="2cm",},
                                    height = "6cm", width = "6cm",});
figure_KSD = @pgf PGFPlotsX.Axis(
              { theme...,
                  ylabel = "Error",
                  xlabel = "Grid points N",
                  xmode = "log",
                  ymode = "log"
              },
              Plot({mark  = "none","thick","red!40!gray"},Table(Ns, W_KSD_values)),
              )
figure_KSD_timing = @pgf PGFPlotsX.Axis(
              { theme...,
                  ylabel = "time (s)",
                  xlabel = "Grid points N",
                  xmode = "log",
                  ymode = "log",
              },
              Plot({mark  = "none","thick","red!40!gray"},Table(Ns, getproperty.(minimum.(KSD_timings),:time) .* 1e-9)),
              )
figure_multi = @pgf PGFPlotsX.Axis(
              { theme...,
                  ylabel = "Error",
                  xlabel = "Grid points N",
                  xmode = "log",
              },
              Plot({mark  = "none","thick","red!40!gray"},Table(Ns, W_multi_values)),
              )
figure_multi_timing = @pgf PGFPlotsX.Axis(
              { theme...,
                  ylabel = "time (s)",
                  xlabel = "Grid points N",
                  xmode = "log",
                  ymode = "log",
              },
              Plot({mark  = "none","thick","red!40!gray"},Table(Ns, getproperty.(minimum.(Multi_timings),:time) .* 1e-9)),
              )
push!(gp_KSD,figure_KSD)
push!(gp_KSD,figure_KSD_timing)
PGFPlotsX.save("KSD_convergence.pdf",gp_KSD,include_preamble=false)
PGFPlotsX.save("KSD_convergence.tex",gp_KSD,include_preamble=false)
push!(gp_multi,figure_multi)
push!(gp_multi,figure_multi_timing)
PGFPlotsX.save("Multi_convergence.pdf",gp_multi,include_preamble=false)
PGFPlotsX.save("Multi_convergence.tex",gp_multi,include_preamble=false)
function W_neg(F::Tensor{2,dim}) where dim
    sv = Vec{dim}([F[1,1], F[2,2]])
    return sum((sv .- 1).^2 .* (sv .+ 1).^2)
    #return (sv[2] - 1)^2 * (sv[2] + 1)^2 + (sv[1] + 1)^2 * (sv[1] - 1)^2
end

x = -2:0.05:2
y = copy(x)
z = [W_neg(Tensor{2,2}([i 0; 0 j])) for i in x, j in y]
neg_example = @pgf Axis(
        {
           theme...,
           colorbar,
           "colormap/viridis",
           "unbounded coords" = "jump"
        },
        Plot3(
           {
               surf,
               shader = "interp",
           },
           Table(x, y, z)
        ))

PGFPlotsX.save("counter_example.pdf", neg_example, include_preamble=false)
PGFPlotsX.save("counter_example.tex", neg_example, include_preamble=false)

x = -2:0.05:2
y = copy(x)
z = [W_KSD(Tensor{2,2}([i 0; 0 j])) for i in x, j in y]
KSD_example = @pgf Axis(
        {
           width="6cm",
           height="6cm",
           view="{20}{20}",
           theme...,
           colorbar,
           xmin=-2,xmax=2,
           ymin=-2,ymax=2,
           xtick="{-2,-1,0,1,2}",
           ytick="{-2,-1,0,1,2}",
           "colormap/viridis",
           "unbounded coords" = "jump"
        },
        Plot3(
           {
               surf,
               shader = "interp",
           },
           Table(x, y, z)
        ))

PGFPlotsX.save("KSD.pdf", KSD_example, include_preamble=false)
PGFPlotsX.save("KSD.tex", KSD_example, include_preamble=false)

function generate_convexified(x,y,W)
    cs = HROC(F_start, F_stop; GLcheck=false, n_convexpoints=5000, maxlevel=10)
    buffer = build_buffer(cs)
    z = zeros(length(x),length(y))
    for i in eachindex(x), j in eachindex(y)
        F = Tensor{2,2}([x[i] 0; 0 y[j]])
        bt = convexify(cs,buffer,W,F)
        𝔸, 𝐏, W_val = NumericalRelaxation.eval(bt,W)
        z[i,j] = W_val
    end
    return z
end

x = -2:0.05:2
y = copy(x)
z = generate_convexified(x,y,W_KSD)
KSD_hroc = @pgf Axis(
        {
           width="6cm",
           height="6cm",
           view="{20}{20}",
           theme...,
           colorbar,
           xmin=-2,xmax=2,
           ymin=-2,ymax=2,
           xtick="{-2,-1,0,1,2}",
           ytick="{-2,-1,0,1,2}",
           "colormap/viridis",
           "unbounded coords" = "jump"
        },
        Plot3(
           {
               surf,
               shader = "interp",
           },
           Table(x, y, z)
        ))

PGFPlotsX.save("KSD_hroc.pdf", KSD_hroc, include_preamble=false)
PGFPlotsX.save("KSD_hroc.tex", KSD_hroc, include_preamble=false)

x = -2:0.05:2
y = copy(x)
z = [W_multi(Tensor{2,2}([i 0; 0 j])) for i in x, j in y]
Multi_example = @pgf Axis(
        {
           theme...,
           colorbar,
           "colormap/viridis",
           "unbounded coords" = "jump"
        },
        Plot3(
           {
               surf,
               shader = "interp",
           },
           Table(x, y, z)
        ))

PGFPlotsX.save("Multi.pdf", Multi_example, include_preamble=false)
PGFPlotsX.save("Multi.tex", Multi_example, include_preamble=false)

x = -2:0.05:2
y = copy(x)
z = generate_convexified(x,y,W_multi)
Multi_hroc = @pgf Axis(
        {
           theme...,
           colorbar,
           "colormap/viridis",
           "unbounded coords" = "jump"
        },
        Plot3(
           {
               surf,
               shader = "interp",
           },
           Table(x, y, z)
        ))

PGFPlotsX.save("Multi_hroc.pdf", Multi_hroc, include_preamble=false)
PGFPlotsX.save("Multi_hroc.tex", Multi_hroc, include_preamble=false)

############################################################
# Polish-stage comparison: discrete HROC vs polish variants
# (compass = zero order, BFGS-AD = stage 1, BFGS-analytic = stage 2)
# Compares relaxed energy error, W-calls and wall time over N
# for the KSD, multiwell and counter-example energies.
############################################################
WCALLS = Ref(0)

polish_variants = [
    (nothing,                        "discrete",      "black"),
    (CompassSearch(),                "compass",       "gray"),
    (BFGS(gradient=ADGradient()),    "BFGS-AD",       "orange!80!black"),
    (BFGS(),                         "BFGS-analytic", "blue!70!black"),
    (Adam(),                         "Adam+BFGS",     "green!55!black"),
]
Ns_polish = Int[10, 50, 100, 300, 500, 1000, 3000, 5000, 10_000]

function polish_comparison(Wfun::FUN, Wrc, points, Ns) where FUN
    Wcount = F -> (WCALLS[] += 1; Wfun(F))
    results = Dict{Tuple{Int,String},NTuple{3,Vector{Float64}}}()
    for (pidx, (F, pname)) in enumerate(points)
        for (optimizer, label, _) in polish_variants
            errs = Float64[]; wcalls = Float64[]; times = Float64[]
            for N in Ns
                cs = HROC(F_start, F_stop; GLcheck=false, n_convexpoints=N, maxlevel=10, polish=optimizer)
                buffer = build_buffer(cs)
                bt = convexify(cs, buffer, Wfun, F)
                𝔸, 𝐏, W_val = NumericalRelaxation.eval(bt, Wfun)
                push!(errs, max(abs(W_val - Wrc(F)), 1e-16))
                WCALLS[] = 0
                convexify(cs, buffer, Wcount, F)
                push!(wcalls, Float64(WCALLS[]))
                bench = @benchmark convexify($cs, $buffer, $Wfun, $F) samples=5 evals=1 seconds=2
                push!(times, minimum(bench).time * 1e-9)
            end
            results[(pidx, label)] = (errs, wcalls, times)
            println("[$pname] $label: err(N=$(Ns[end])) = $(errs[end]), Wcalls(N=$(Ns[end])) = $(Int(wcalls[end])), t(N=$(Ns[end])) = $(round(times[end]*1000,digits=2)) ms")
        end
    end
    return results
end

function polish_groupplot(results, points, Ns)
    gp = @pgf GroupPlot({group_style = {group_size = "3 by $(length(points))", "horizontal sep"="2.2cm", "vertical sep"="1.8cm"},
                         height = "5.5cm", width = "6cm"})
    for (pidx, (F, pname)) in enumerate(points)
        allerrs = vcat((results[(pidx, label)][1] for (_, label, _) in polish_variants)...)
        # explicit limits: a degenerate log-axis range (all curves at the same level) overflows pgfplots
        e_lo = 10.0^(floor(log10(minimum(allerrs))) - 1)
        e_hi = 10.0^(ceil(log10(maximum(allerrs))) + 1)
        ax_err = pidx == 1 ?
            PGFPlotsX.Axis(@pgf({theme..., ylabel = "Error", xlabel = "Grid points N", xmode="log", ymode="log",
                                 ymin = e_lo, ymax = e_hi,
                                 title = pname, legend_style = raw"{font=\tiny, at={(0.02,0.02)}, anchor=south west}"})) :
            PGFPlotsX.Axis(@pgf({theme..., ylabel = "Error", xlabel = "Grid points N", xmode="log", ymode="log",
                                 ymin = e_lo, ymax = e_hi, title = pname}))
        ax_wcalls = PGFPlotsX.Axis(@pgf({theme..., ylabel = "W-calls", xlabel = "Grid points N", xmode="log", ymode="log"}))
        ax_time = PGFPlotsX.Axis(@pgf({theme..., ylabel = "time (s)", xlabel = "Grid points N", xmode="log", ymode="log"}))
        for (optimizer, label, clr) in polish_variants
            errs, wcalls, times = results[(pidx, label)]
            push!(ax_err, @pgf Plot({mark="none", thick, color=clr}, Table(Ns, errs)))
            pidx == 1 && push!(ax_err, LegendEntry(label))
            push!(ax_wcalls, @pgf Plot({mark="none", thick, color=clr}, Table(Ns, wcalls)))
            push!(ax_time, @pgf Plot({mark="none", thick, color=clr}, Table(Ns, times)))
        end
        push!(gp, ax_err); push!(gp, ax_wcalls); push!(gp, ax_time)
    end
    return gp
end

# --- KSD: one point where the greedy stage finds a tree, one where it does not
KSD_points = [
    (Tensor{2,2}([0.2 0.1; 0.1 0.3]),     "KSD, laminate region"),
    (Tensor{2,2}([-0.35 0.0; 0.0 -0.45]), "KSD, greedy-stuck region"),
]
res_KSD = polish_comparison(W_KSD, W_KSD_rc, KSD_points, Ns_polish)
gp_KSD_polish = polish_groupplot(res_KSD, KSD_points, Ns_polish)
PGFPlotsX.save("KSD_polish_comparison.pdf", gp_KSD_polish, include_preamble=true)
PGFPlotsX.save("KSD_polish_comparison.tex", gp_KSD_polish, include_preamble=false)

# --- multiwell (|F|-1)^2: W_rc = 0 inside the unit ball (first-order laminate to the sphere)
multi_points = [(Tensor{2,2}([0.3 0.1; -0.1 0.4]), "multiwell")]
res_multi = polish_comparison(W_multi, W_multi_rc, multi_points, Ns_polish)
gp_multi_polish = polish_groupplot(res_multi, multi_points, Ns_polish)
PGFPlotsX.save("Multi_polish_comparison.pdf", gp_multi_polish, include_preamble=true)
PGFPlotsX.save("Multi_polish_comparison.tex", gp_multi_polish, include_preamble=false)

# --- counter-example of section 4.3 "Failure of Approximation", eq. (4.7) of the paper:
#     Wfail(F) = ((ν₁-3)²(ν₁+3)² + (ν₂-3)²(ν₂+3)²)·((√(ν₁²+ν₂²)-1)²+1), ν = signed singular values.
#     Even in each νᵢ, hence expressible in invariants of C = FᵀF:
#     (ν₁²-9)²+(ν₂²-9)² = (trC)²-2detC-18trC+162 and ν₁²+ν₂² = trC = |F|².
#     Nested minima: inner local-minimum ring at radius 1, global zeros only at ν=(±3,±3);
#     the rank-one convex envelope is 0 at F̂ = 0, but reaching the corner wells requires an
#     energetically non-optimal intermediate laminate — a global obstruction by construction.
function W_fail(F::Tensor{2,2})
    trC = norm(F)^2
    detC = det(F)^2
    return (trC^2 - 2*detC - 18*trC + 162) * ((sqrt(trC) - 1)^2 + 1)
end
W_fail_rc(F) = 0.0 # valid inside the diagonal box |F₁₁|,|F₂₂| ≤ 3, in particular at F̂ = 0
fail_points = [(zero(Tensor{2,2}), "Wfail (section 4.3) at F = 0")]
res_fail = polish_comparison(W_fail, W_fail_rc, fail_points, Ns_polish)
gp_fail_polish = polish_groupplot(res_fail, fail_points, Ns_polish)
PGFPlotsX.save("counter_example_polish_comparison.pdf", gp_fail_polish, include_preamble=true)
PGFPlotsX.save("counter_example_polish_comparison.tex", gp_fail_polish, include_preamble=false)

# --- StVK-type energy (F₁₁²-1)² + (F₂₂²-1)²: quadratic in the uniaxial Green-Lagrange strains,
#     separable in the diagonal entries, hence W_rc(F) = convf(F₁₁) + convf(F₂₂) with convf the
#     1D convex hull of (x²-1)². The greedy stage plateaus here (level-wise optimal but globally
#     wrong direction choice) while the polish recovers the exact envelope — a good showcase.
W_STVK(F::Tensor{2,2}) = (F[1,1]^2 - 1)^2 + (F[2,2]^2 - 1)^2
convf(x) = abs(x) ≤ 1 ? 0.0 : (x^2 - 1)^2
W_STVK_rc(F) = convf(F[1,1]) + convf(F[2,2])
stvk_points = [(Tensor{2,2}([0.5 0.1; 0.1 -0.3]), "StVK-type")]
res_stvk = polish_comparison(W_STVK, W_STVK_rc, stvk_points, Ns_polish)
gp_stvk_polish = polish_groupplot(res_stvk, stvk_points, Ns_polish)
PGFPlotsX.save("STVK_polish_comparison.pdf", gp_stvk_polish, include_preamble=true)
PGFPlotsX.save("STVK_polish_comparison.tex", gp_stvk_polish, include_preamble=false)

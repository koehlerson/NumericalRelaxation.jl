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
    F = zero(Tensor{2,3})
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

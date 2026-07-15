using NumericalRelaxation
using StaticArrays
using Tensors
using Test
import ForwardDiff

W(F::Number,x1=2,x2=6) = (F-1)^x1 * (42 + 77*F + 15*F^x1 - 102.5*F^3 + 58.89*F^4 - 12.89*F^5 + F^x2)
W(F::Tensor{2,1},x1=2,x2=6) = W(F[1],x1,x2)

W_multi(F::Tensor{2,dim}) where dim = (norm(F)-1)^2
W_multi_rc(F::Tensor{2,dim}) where dim = norm(F) ≤ 1 ? 0.0 : (norm(F)-1)^2

# toy leaf objective, nonlinear in the measure: E(ν) = ⟨ν, W⟩ + ⟨ν, h⟩², with exact duals
struct MeasureToyObjective{WF,HF} <: NumericalRelaxation.AbstractLeafObjective
    W::WF
    h::HF
end
(o::MeasureToyObjective)(leaves) = sum(l[2]*o.W(l[3]) for l in leaves) + sum(l[2]*o.h(l[3]) for l in leaves)^2
function NumericalRelaxation.leafduals(o::MeasureToyObjective, leaves)
    m = sum(l[2]*o.h(l[3]) for l in leaves)
    return Dict(l[1] => (o.W(l[3]) + 2m*o.h(l[3]),
                         Tensors.gradient(o.W, l[3]) + 2m*Tensors.gradient(o.h, l[3])) for l in leaves)
end

@testset "Equidistant Graham Scan" begin
    convexification = GrahamScan(start=0.01,stop=5.0,δ=0.01)
    buffer = build_buffer(convexification)
    W_conv, F⁺, F⁻ = convexify(convexification,buffer,W,Tensor{2,1}((2.0,)))
    @test isapprox(W_conv,7.2,atol=1e-1)
    @test isapprox(F⁺[1],4.0,atol=1e-1)
    @test isapprox(F⁻[1],1.0,atol=1e-1)
    @test @inferred(convexify(convexification,buffer,W,Tensor{2,1}((2.0,)))) == (W_conv,F⁺,F⁻)
end

##### multi-dimensional relaxation unit tests ####
@testset "Gradient Grid Iterator" begin
    # equidistant meshes
    gradientgrid_axes = -2.0:0.5:2
    gradientgrid1 = GradientGrid((gradientgrid_axes, gradientgrid_axes, gradientgrid_axes, gradientgrid_axes))
    @test size(gradientgrid1) == ntuple(x->length(gradientgrid_axes),4)
    @test size(gradientgrid1,1) == size(gradientgrid1,2) == size(gradientgrid1,3) == size(gradientgrid1,4)  == length(gradientgrid_axes)
    @test eltype(gradientgrid1) == Tensor{2,2,Float64,4}
    @test gradientgrid1[2,3,4,5] == Tensor{2,2}((gradientgrid_axes[2],gradientgrid_axes[3],gradientgrid_axes[4],gradientgrid_axes[5]))
    @test gradientgrid1[9,9,9,9] == gradientgrid1[end]
    @test gradientgrid1[1,1,1,1] == gradientgrid1[1]
    @test gradientgrid1[2] == gradientgrid1[2,1,1,1]
    @test gradientgrid1[9] == gradientgrid1[9,1,1,1]
    @test gradientgrid1[10] == gradientgrid1[1,2,1,1]
    @test NumericalRelaxation.center(gradientgrid1) == Tensor{2,2}((0.0,0.0,0.0,0.0))
    @test NumericalRelaxation.δ(gradientgrid1) == 0.5
    @test NumericalRelaxation.radius(gradientgrid1) == 2.0

    gradientgrid_axes = -1.0:0.5:2
    gradientgrid2 = GradientGrid((gradientgrid_axes, gradientgrid_axes, gradientgrid_axes, gradientgrid_axes))
    @test gradientgrid2[2,3,4,5] == Tensor{2,2}((gradientgrid_axes[2],gradientgrid_axes[3],gradientgrid_axes[4],gradientgrid_axes[5]))
    @test size(gradientgrid2) == ntuple(x->length(gradientgrid_axes),4)
    @test size(gradientgrid2,1) == size(gradientgrid2,2) == size(gradientgrid2,3) == size(gradientgrid2,4)  == length(gradientgrid_axes)
    @test NumericalRelaxation.center(gradientgrid2) == Tensor{2,2}((0.5,0.5,0.5,0.5))
    @test NumericalRelaxation.δ(gradientgrid2) == 0.5
    @test NumericalRelaxation.radius(gradientgrid2) == 1.5

    gradientgrid_axes = 1.0:0.1:2
    gradientgrid3 = GradientGrid((gradientgrid_axes, gradientgrid_axes, gradientgrid_axes, gradientgrid_axes))
    @test gradientgrid3[2,3,4,5] == Tensor{2,2}((gradientgrid_axes[2],gradientgrid_axes[3],gradientgrid_axes[4],gradientgrid_axes[5]))
    @test size(gradientgrid3) == ntuple(x->length(gradientgrid_axes),4)
    @test size(gradientgrid3,1) == size(gradientgrid3,2) == size(gradientgrid3,3) == size(gradientgrid3,4)  == length(gradientgrid_axes)
    @test NumericalRelaxation.center(gradientgrid3) == Tensor{2,2}((1.5,1.5,1.5,1.5))
    @test NumericalRelaxation.δ(gradientgrid3) == 0.1
    @test NumericalRelaxation.radius(gradientgrid3) == 0.5

    # non equidistant meshes
    gradientgrid_axes_diag = -2.0:0.5:2; gradientgrid_axes_off = -1.0:0.1:1.0
    gradientgrid1 = GradientGrid((gradientgrid_axes_diag, gradientgrid_axes_off, gradientgrid_axes_off, gradientgrid_axes_diag))
    @test eltype(gradientgrid1) == Tensor{2,2,Float64,4}
    @test size(gradientgrid1) == (length(gradientgrid_axes_diag),length(gradientgrid_axes_off),length(gradientgrid_axes_off),length(gradientgrid_axes_diag))
    @test size(gradientgrid1,1) == size(gradientgrid1,4) == length(gradientgrid_axes_diag)
    @test size(gradientgrid1,2) == size(gradientgrid1,3) == length(gradientgrid_axes_off)
    @test gradientgrid1[2,3,4,5] == Tensor{2,2}((gradientgrid_axes_diag[2],gradientgrid_axes_off[3],gradientgrid_axes_off[4],gradientgrid_axes_diag[5]))
    @test gradientgrid1[9,21,21,9] == gradientgrid1[end]
    @test gradientgrid1[1,1,1,1] == gradientgrid1[1]
    @test gradientgrid1[2] == gradientgrid1[2,1,1,1]
    @test gradientgrid1[9] == gradientgrid1[9,1,1,1]
    @test gradientgrid1[10] == gradientgrid1[1,2,1,1]
    @test NumericalRelaxation.δ(gradientgrid1) == 0.1
    @test NumericalRelaxation.center(gradientgrid1) == Tensor{2,2}((0.0,0.0,0.0,0.0))
    @test NumericalRelaxation.radius(gradientgrid1) == 2.0

    gradientgrid_axes_diag = -1.0:0.5:2; gradientgrid_axes_off = -3.0:0.1:0.0
    gradientgrid2 = GradientGrid((gradientgrid_axes_diag, gradientgrid_axes_off, gradientgrid_axes_off, gradientgrid_axes_diag))
    @test size(gradientgrid2) == (length(gradientgrid_axes_diag),length(gradientgrid_axes_off),length(gradientgrid_axes_off),length(gradientgrid_axes_diag))
    @test size(gradientgrid2,1) == size(gradientgrid2,4) == length(gradientgrid_axes_diag)
    @test size(gradientgrid2,2) == size(gradientgrid2,3) == length(gradientgrid_axes_off)
    @test NumericalRelaxation.center(gradientgrid2) == Tensor{2,2}((0.5,-1.5,-1.5,0.5))
    @test NumericalRelaxation.δ(gradientgrid2) == 0.1
    @test NumericalRelaxation.radius(gradientgrid2) == 1.5

    gradientgrid_axes_diag = 1.0:0.1:2; gradientgrid_axes_off = 0.0:0.1:2.0
    gradientgrid3 = GradientGrid((gradientgrid_axes_diag, gradientgrid_axes_off, gradientgrid_axes_off, gradientgrid_axes_diag))
    @test size(gradientgrid3) == (length(gradientgrid_axes_diag),length(gradientgrid_axes_off),length(gradientgrid_axes_off),length(gradientgrid_axes_diag))
    @test size(gradientgrid3,1) == size(gradientgrid3,4) == length(gradientgrid_axes_diag)
    @test size(gradientgrid3,2) == size(gradientgrid3,3) == length(gradientgrid_axes_off)
    @test NumericalRelaxation.center(gradientgrid3) == Tensor{2,2}((1.5,1.0,1.0,1.5))
    @test NumericalRelaxation.δ(gradientgrid3) == 0.1
    @test NumericalRelaxation.radius(gradientgrid3) == 1.0

    @test @inferred(gradientgrid1[1,1,1,1]) == Tensor{2,2}((-2.0,-1.0,-1.0,-2.0))
    @test @inferred(Union{Tensor{2,2},Nothing}, Base.iterate(gradientgrid1,1)) == (Tensor{2,2}((-2.0,-1.0,-1.0,-2.0)),2)
    @test @inferred(Union{Tensor{2,2},Nothing}, Base.iterate(gradientgrid1,100000)) === nothing
    @test @inferred(NumericalRelaxation.center(gradientgrid3)) == Tensor{2,2}((1.5,1.0,1.0,1.5))
    @test @inferred(NumericalRelaxation.δ(gradientgrid3)) == 0.1
    @test @inferred(NumericalRelaxation.radius(gradientgrid3)) == 1.0
    @test @inferred(size(gradientgrid3,1)) == size(gradientgrid3,4) == length(gradientgrid_axes_diag)
    @test size(gradientgrid3,1) == @inferred(size(gradientgrid3,4)) == length(gradientgrid_axes_diag)
    @test size(gradientgrid3,1) == size(gradientgrid3,4) == @inferred(length(gradientgrid_axes_diag))
end

@testset "Rank One Direction Iterator" begin
    gradientgrid_axes = -2.0:0.5:2
    gradientgrid1 = GradientGrid((gradientgrid_axes, gradientgrid_axes, gradientgrid_axes, gradientgrid_axes))
    dirs = ParametrizedR1Directions(2)
    @test(@inferred Union{Nothing,Tuple{Tuple{Vec{2,Int},Vec{2,Int}},Int}} Base.iterate(dirs,1) == (ones(Tensor{2,2}),2))
    (𝐚,i) = Base.iterate(dirs,1)
    #@test @inferred(NumericalRelaxation.inbounds_𝐚(gradientgrid1,𝐚)) && @inferred(NumericalRelaxation.inbounds_𝐛(gradientgrid1,𝐛))
end

#@testset "R1Convexification" begin
#    d = 2
#    a = -2.0:0.5:2.0
#    r1convexification_reduced = R1Convexification(a,a,dim=d,dirtype=ParametrizedR1Directions)
#    buffer_reduced = build_buffer(r1convexification_reduced)
#    convexify!(r1convexification_reduced,buffer_reduced,W_multi;buildtree=true)
#    @test all(isapprox.(buffer_reduced.W_rk1.itp.itp.coefs .- [W_multi_rc(Tensor{2,2}((x1,x2,y1,y2))) for x1 in a, x2 in a, y1 in a, y2 in a],0.0,atol=1e-8))
#    r1convexification_full = R1Convexification(a,a,dim=d,dirtype=ℛ¹Direction)
#    buffer_full = build_buffer(r1convexification_full)
#    convexify!(r1convexification_full,buffer_full,W_multi;buildtree=false)
#    @test all(isapprox.(buffer_full.W_rk1.itp.itp.coefs .- [W_multi_rc(Tensor{2,2}((x1,x2,y1,y2))) for x1 in a, x2 in a, y1 in a, y2 in a],0.0,atol=1e-8))
#    @test all(isapprox.(buffer_full.W_rk1.itp.itp.coefs .- buffer_reduced.W_rk1.itp.itp.coefs ,0.0,atol=1e-8))
#    # test if subsequent convexifications work
#    convexify!(r1convexification_full,buffer_full,W_multi;buildtree=true)
#    @test all(isapprox.(buffer_full.W_rk1.itp.itp.coefs .- buffer_reduced.W_rk1.itp.itp.coefs ,0.0,atol=1e-8))
#
#    @testset "Tree Construction" begin
#        F1 = Tensor{2,2}((0.1,0.0,0.0,0.0))
#        F2 = Tensor{2,2}((0.1,0.5,0.3,0.2))
#        F3 = Tensor{2,2}((0.5,0.5,0.0,0.0))
#        for F in (F1,F2,F3)
#            flt = FlexibleLaminateTree(F,r1convexification_full,buffer_full,3)
#            @test NumericalRelaxation.checkintegrity(flt,buffer_full.W_rk1)
#            𝔸, 𝐏, W = NumericalRelaxation.eval(flt,W_multi)
#            @test W == 0.0
#            @test 𝐏 == zero(Tensor{2,2})
#        end
#    end
#end

@testset "HROC" begin
    for dim in (2,3)
        convexification = HROC(10,500,ParametrizedR1Directions(dim),false,[-2.0 for _ in 1:dim^2],[2.0 for _ in 1:dim^2])
        buffer = build_buffer(convexification)
        F = zero(Tensor{2,dim})
        bt = convexify(convexification,buffer,W_multi,F)
        𝔸, 𝐏, W_val = NumericalRelaxation.eval(bt,W_multi)
        @test NumericalRelaxation.checkintegrity(bt)
        @test isapprox(W_val,0.0,atol=1e-4)
        @test isapprox(𝐏,F,atol=1e-4)

        @testset "HROCBuffer" begin
            ctr_fw = 4
            ctr_bw = 4
            buffer.forward_initial.grid[1:ctr_fw] .= [0,1,2,3]
            buffer.backward_initial.grid[1:ctr_bw] .= [-1,-2,-3,-4]
            NumericalRelaxation.concat!(buffer,ctr_fw+1,ctr_bw)
            @test all(buffer.initial.grid[1:ctr_fw+ctr_bw] .== [-4,-3,-2,-1,0,1,2,3])
        end

        @testset "Type stability" begin
            laminate = @inferred Nothing NumericalRelaxation.hrockernel(BinaryLaminationTree(F,0.0,1.0,0),convexification,buffer,W_multi,zero(Tensor{2,dim}))
            @test laminate.W⁺ == laminate.W⁻ && isapprox(laminate.W⁺,0.0,atol=1e-4) && isapprox(laminate.W⁻,0.0,atol=1e-4)
            if dim ==2
                @test laminate.F⁺ == Tensor{2,dim}([0.0 0.0; -1.0 0.0])
                @test laminate.F⁻ == Tensor{2,dim}([0.0 0.0; 1.0 0.0])
            else
                @test laminate.F⁺ == Tensor{2,dim}([0.0 0.0 0.0; 0.0 0.0 0.0; -1.0 0.0 0.0])
                @test laminate.F⁻ == Tensor{2,dim}([0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 0.0 0.0])
            end
        end
    end
end

@testset "HROC polish" begin
    W_KSD(F) = norm(F) ≥ √2 - 1 ? (1+norm(F)^2) : (2*√2*norm(F))
    W_KSD_rc(F) = (ρ = sqrt(norm(F)^2 + 2*norm(det(F))); ρ ≥ 1 ? 1+norm(F)^2 : 2*(ρ - norm(det(F))))
    F_start = ones(Tensor{2,2})*-3
    F_stop  = ones(Tensor{2,2})*3
    F_lam   = Tensor{2,2}([0.2 0.1; 0.1 0.3])     # greedy stage finds a lamination tree here
    F_stuck = Tensor{2,2}([-0.35 0.0; 0.0 -0.45]) # greedy stage finds no laminate here (rank-two region)
    for optimizer in (CompassSearch(), BFGS(gradient=ADGradient()), BFGS(), Adam(), NonLocalNewton())
        cs = HROC(F_start,F_stop;GLcheck=false,n_convexpoints=1000,maxlevel=10,polish=optimizer)
        buffer = build_buffer(cs)
        for F in (F_lam, F_stuck)
            bt = convexify(cs,buffer,W_KSD,F)
            𝔸, 𝐏, W_val = NumericalRelaxation.eval(bt,W_KSD)
            @test NumericalRelaxation.checkintegrity(bt)
            @test abs(W_val - W_KSD_rc(F)) < 5e-6
        end
    end
    # polish=true sugar maps to the default BFGS optimizer
    @test HROC(F_start,F_stop;polish=true).polish isa BFGS
    @test HROC(F_start,F_stop;polish=false).polish === nothing

    @testset "rate-constrained polish" begin
        # mimics the convexified-damage usage: convexify(btFₖ, btβₖ, cs, buffer, constraint, irr, W, F, material)
        # with a RateConstraint-style predicate on β(F) = Ψ₀(C); the polish must never place a
        # laminate state outside the constraint-admissible set and stays above the constrained optimum
        cap = 0.01
        Ψ₀(F) = 0.5*norm(F)^2
        W_dmg(F, cap) = W_KSD(F)
        function ratecon(prev_F, prev_β, F, cap)
            β = Ψ₀(F)
            for i in 1:length(prev_β.active)
                (prev_β.active[i] && NumericalRelaxation.isleaf(prev_β, i)) || continue
                β ≤ prev_β.nodes[i].F + cap && return true
            end
            return false
        end
        bndcheck(r, parent, F, cap) = Ψ₀(F) ≥ parent.F + cap - 1e-4
        cs0 = HROC(F_start, F_stop; GLcheck=false, n_convexpoints=500, maxlevel=10)
        buf0 = build_buffer(cs0)
        btFₖ = convexify(cs0, buf0, W_dmg, Tensor{2,2}([0.15 0.05; 0.05 0.2]), cap)
        βnodes = Vector{NumericalRelaxation.BinaryLaminationTreeNode{1,Float64,1}}(undef, length(btFₖ.nodes))
        for i in 1:length(btFₖ.active)
            btFₖ.active[i] || continue
            n = btFₖ.nodes[i]
            βnodes[i] = NumericalRelaxation.BinaryLaminationTreeNode(Ψ₀(n.F), 0.0, n.ξ, n.level)
        end
        btβₖ = NumericalRelaxation.BinaryLaminationTree{1,Float64,1}(βnodes, copy(btFₖ.active))
        W_ref = Dict{String,Float64}()
        for (name, polish) in (("discrete", nothing), ("bfgs", BFGS()), ("compass", CompassSearch()))
            cs = HROC(F_start, F_stop; GLcheck=false, n_convexpoints=500, maxlevel=10, polish=polish)
            buffer = build_buffer(cs)
            btF = convexify(btFₖ, btβₖ, cs, buffer, ratecon, bndcheck, W_dmg, F_lam, cap)
            𝔸, 𝐏, W_val = NumericalRelaxation.eval(btF, W_dmg, cap)
            @test NumericalRelaxation.checkintegrity(btF)
            @test all(ratecon(btFₖ, btβₖ, btF.nodes[i].F, cap) for i in 1:length(btF.active) if btF.active[i])
            @test W_val ≥ W_KSD_rc(F_lam) - 1e-10 # rate cap binds: stays above the unconstrained envelope
            W_ref[name] = W_val
        end
        @test W_ref["bfgs"] ≤ W_ref["discrete"] + 1e-12 # polish never worse than the discrete constrained tree
    end

    @testset "leaf objectives" begin
        cs = HROC(F_start,F_stop;GLcheck=false,n_convexpoints=1000,maxlevel=10)
        buffer = build_buffer(cs)
        bt = convexify(cs,buffer,W_KSD,F_lam)
        p, offsets = NumericalRelaxation.polish_parameters(bt; maxdepth=4)
        p .*= 1 .+ 0.01*sin.(1:length(p))
        admissible = NumericalRelaxation._admissible(cs)
        h_toy(F) = norm(F)^2
        obj = MeasureToyObjective(W_KSD, h_toy)
        E = q -> NumericalRelaxation.polish_energy(obj, q, offsets, F_lam, admissible)
        @test isfinite(E(p))
        # analytic gradient vs central finite differences
        g_an = NumericalRelaxation.polish_gradient!(zero(p), obj, p, offsets, F_lam, admissible)
        δ = 1e-6
        for k in eachindex(p)
            q⁺ = copy(p); q⁺[k] += δ
            q⁻ = copy(p); q⁻[k] -= δ
            @test abs(g_an[k] - (E(q⁺) - E(q⁻))/(2δ)) < 1e-5*max(1.0, abs(g_an[k]))
        end
        # degenerate h ≡ 0 must reproduce the scalar-density gradient exactly
        obj0 = MeasureToyObjective(W_KSD, F -> 0.0)
        g0 = NumericalRelaxation.polish_gradient!(zero(p), obj0, p, offsets, F_lam, admissible)
        g_scalar = zero(p)
        NumericalRelaxation.polish_gradient!(g_scalar, p, offsets, F_lam, W_KSD)
        @test maximum(abs, g0 .- g_scalar) < 1e-12
        # BFGS on the leaf objective is monotone and reduces the energy
        f0 = E(p)
        pB = copy(p)
        fB = NumericalRelaxation.polish_minimize!(BFGS(), obj, pB, offsets, F_lam, admissible)
        @test fB ≤ f0 + 1e-12
    end

    @testset "analytic gradient vs ForwardDiff" begin
        cs = HROC(F_start,F_stop;GLcheck=false,n_convexpoints=1000,maxlevel=10)
        buffer = build_buffer(cs)
        bt = convexify(cs,buffer,W_KSD,F_lam)
        p, offsets = NumericalRelaxation.polish_parameters(bt)
        p .*= 1 .+ 0.01*sin.(1:length(p)) # move off the discrete optimum, keep offsets positive
        admissible = NumericalRelaxation._admissible(cs)
        E = q -> NumericalRelaxation.polish_energy(q, offsets, 1, F_lam, admissible, W_KSD)
        @test isfinite(E(p))
        g_ad = ForwardDiff.gradient(E, p)
        g_an = zero(p)
        NumericalRelaxation.polish_gradient!(g_an, p, offsets, F_lam, W_KSD)
        @test maximum(abs, g_ad .- g_an) < 1e-12
    end
end

@testset "Adaptive Convexification" begin
    ac = AdaptiveGrahamScan(
            interval=[0.001,5.0],
            basegrid_numpoints=50,
            adaptivegrid_numpoints=115,
            exponent=5,
            distribution="fix",
            stepSizeIgnoreHessian=0.05,
            minPointsPerInterval=15,
            radius=3,
            minStepSize=0.03,
            forceAdaptivity=false)
    @testset "build_buffer()" begin
        buf = @inferred NumericalRelaxation.build_buffer(ac)
        @test typeof(buf) == (NumericalRelaxation.AdaptiveConvexificationBuffer1D{Tensor{2,1,Float64,1},Float64,Tensor{4,1,Float64,1}})
        @test length(buf.basebuffer.grid) == ac.basegrid_numpoints
        @test length(buf.basebuffer.values) == ac.basegrid_numpoints
        @test length(buf.basegrid_∂²W) == ac.basegrid_numpoints
        @test length(buf.adaptivebuffer.grid) == ac.adaptivegrid_numpoints
        @test length(buf.adaptivebuffer.values) == ac.adaptivegrid_numpoints
        @test buf.basebuffer.grid[1][1] == ac.interval[1]
        @test buf.basebuffer.grid[end][1] == ac.interval[2]
        δ_sum = 0
        for i in 1:ac.basegrid_numpoints-1 #check for equal distribution
            δ_sum = abs((buf.basebuffer.grid[i+1][1]-buf.basebuffer.grid[i][1])-(ac.interval[2]-ac.interval[1])/(ac.basegrid_numpoints-1))
        end
        @test isapprox(δ_sum,0,atol=1e-10)
    end
    @testset "convexify!()" begin
        buffer = build_buffer(ac)
        F = Tensors.Tensor{2,1}((2.0,))
        W_conv, F⁺, F⁻ = convexify(ac,buffer,W,Tensor{2,1}((2.0,)))
        @test isapprox(W_conv,7.2,atol=1e-1)
        @test isapprox(F⁺[1],4.0,atol=1e-1)
        @test isapprox(F⁻[1],1.0,atol=1e-1)
        @test @inferred(convexify(ac,buffer,W,Tensor{2,1}((2.0,)))) == (W_conv,F⁺,F⁻)
    end
end

@testset "PolyConvexification" begin
    function W(F::Union{Matrix{Float64},SMatrix{d,d,Float64},Tensor{2,d,Float64}},a,b,c) where d
        if !(a == 1 && b == 2 && c == 3)
            error("ARGS not working")
        end
        return (norm(F) <= (sqrt(2) - 1)) ? (2 * sqrt(2) * norm(F)) : (1 + norm(F)^2)
    end

    function DW(F::Union{Matrix{Float64},SMatrix{d,d,Float64},Tensor{2,d,Float64}}) where d
        return (norm(F) <= (sqrt(2) - 1)) ? (2 * sqrt(2) * (norm(F)^(-1)) * F) : 2 .* F
    end

    function Wpc(F::Union{Matrix{Float64},SMatrix{d,d,Float64},Tensor{2,d,Float64}}) where d
        rho = sqrt(norm(F)^2 + 2 * abs(det(F)))
        return (rho <= 1) ? (2 * (rho - abs(det(F)))) : (1 + norm(F)^2)
    end

    function DWpc(F::Union{Matrix{Float64},SMatrix{d,d,Float64},Tensor{2,d,Float64}}) where d
        rho = sqrt(norm(F)^2 + 2 * abs(det(F)))
        Drho = (1 / 2) * (norm(F)^2 + 2 * abs(det(F)))^(-1 / 2) * (2 * F + 2 * sign(det(F)) * inv(F)' * det(F))
        return (rho <= 1) ? (2 * (Drho - sign(det(F)) * inv(F)' * det(F))) : 2 .* F
    end

    function Φ(ν,xargs...)
        return W(diagm(ν),xargs...)
    end

    function Φpc(ν)
        return Wpc(diagm(ν))
    end


    # ##############################################################
    # small convergence test for the polyconvexification approach
    # ##############################################################
    d = 2
    F = (rand(2,2).-0.5) * 2
    ν = ssv(F)
    r = 2
    nrefs = collect(1:10)
    errors = zeros(size(nrefs))
    errorsIso = zeros(size(nrefs))
    errorsDerivative = zeros(size(nrefs))
    errorsDerivativeIso = zeros(size(nrefs))
    for (i, nref) in enumerate(nrefs)
        poly_convexification = PolyConvexification(d, 3.; nref=nref)
        poly_buffer = build_buffer(poly_convexification)

        Φpcνδ, DΦpcνδ, _ = convexify(poly_convexification, poly_buffer, Φ, ν, 1, 2, 3; returnDerivs=true)
        WpcFδ, DWpcFδ, _ = convexify(poly_convexification, poly_buffer, W, F, 1, 2, 3; returnDerivs=true)
        errors[i] = WpcFδ - Wpc(F)
        errorsDerivative[i] = norm(DWpcFδ - DWpc(F))

        errorsIso[i] = Φpcνδ - Wpc(F)
        DWpcFδ = Tensor{3,d}(Dssv(F)) ⋅ Vec{d}(DΦpcνδ)
        errorsDerivativeIso[i] = norm(DWpcFδ - DWpc(F))
    end
    Δ = (2. * r) ./ (2.0 .^ (nrefs))
    @test isapprox(errors[end],0.0,atol=1e-3)
end

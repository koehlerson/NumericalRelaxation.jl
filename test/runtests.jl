using NumericalRelaxation
using Test

#### one-dimensional relaxation unit tests ####
@testset "Equidistant Graham Scan Version" begin
    material = RelaxedDamage(base_material = NeoHooke(), D∞ = 0.99 , D₀ = 0.5, convexstrategy=Bartels1D(start=0.9,stop=20.0,δ=0.01))
    state = ConvexDamage.init_materialstate(material)
    W_conv, F⁺, F⁻ = ConvexDamage.convexify(Tensor{2,1}((2.0,)), material, state)
    @test isapprox(F⁺[1],15.0,atol=0.1)
    @test isapprox(F⁻[1],1.0,atol=0.1)
    # Check if two subsequent calls yield the same result regression test of bug in [MR !12](https://git.noc.ruhr-uni-bochum.de/koehlmsw/convexified-damage/-/merge_requests/12)
    # commit bb09ce59
    W_conv, F⁺, F⁻ = ConvexDamage.convexify(Tensor{2,1}((2.0,)), material, state)
    @test isapprox(F⁺[1],15.0,atol=0.1)
    @test isapprox(F⁻[1],1.0,atol=0.1)

    @test @inferred(ConvexDamage.convexify(Tensor{2,1}((2.0,)),material,state)) == (W_conv, F⁺, F⁻)
end

#### multi-dimensional relaxation unit tests ####
@testset "Deformation Grid Iterator" begin
    # equidistant meshes
    defomesh_axes = -2.0:0.5:2
    defomesh1 = DeformationGrid((defomesh_axes, defomesh_axes, defomesh_axes, defomesh_axes))
    @test size(defomesh1) == ntuple(x->length(defomesh_axes),4)
    @test size(defomesh1,1) == size(defomesh1,2) == size(defomesh1,3) == size(defomesh1,4)  == length(defomesh_axes)
    @test eltype(defomesh1) == Tensor{2,2,Float64,4}
    @test defomesh1[2,3,4,5] == Tensor{2,2}((defomesh_axes[2],defomesh_axes[3],defomesh_axes[4],defomesh_axes[5]))
    @test defomesh1[9,9,9,9] == defomesh1[end]
    @test defomesh1[1,1,1,1] == defomesh1[1]
    @test defomesh1[2] == defomesh1[2,1,1,1]
    @test defomesh1[9] == defomesh1[9,1,1,1]
    @test defomesh1[10] == defomesh1[1,2,1,1]
    @test ConvexDamage.center(defomesh1) == Tensor{2,2}((0.0,0.0,0.0,0.0))
    @test ConvexDamage.δ(defomesh1) == 0.5
    @test ConvexDamage.radius(defomesh1) == 2.0

    defomesh_axes = -1.0:0.5:2
    defomesh2 = DeformationGrid((defomesh_axes, defomesh_axes, defomesh_axes, defomesh_axes))
    @test defomesh2[2,3,4,5] == Tensor{2,2}((defomesh_axes[2],defomesh_axes[3],defomesh_axes[4],defomesh_axes[5]))
    @test size(defomesh2) == ntuple(x->length(defomesh_axes),4)
    @test size(defomesh2,1) == size(defomesh2,2) == size(defomesh2,3) == size(defomesh2,4)  == length(defomesh_axes)
    @test ConvexDamage.center(defomesh2) == Tensor{2,2}((0.5,0.5,0.5,0.5))
    @test ConvexDamage.δ(defomesh2) == 0.5
    @test ConvexDamage.radius(defomesh2) == 1.5

    defomesh_axes = 1.0:0.1:2
    defomesh3 = DeformationGrid((defomesh_axes, defomesh_axes, defomesh_axes, defomesh_axes))
    @test defomesh3[2,3,4,5] == Tensor{2,2}((defomesh_axes[2],defomesh_axes[3],defomesh_axes[4],defomesh_axes[5]))
    @test size(defomesh3) == ntuple(x->length(defomesh_axes),4)
    @test size(defomesh3,1) == size(defomesh3,2) == size(defomesh3,3) == size(defomesh3,4)  == length(defomesh_axes)
    @test ConvexDamage.center(defomesh3) == Tensor{2,2}((1.5,1.5,1.5,1.5))
    @test ConvexDamage.δ(defomesh3) == 0.1
    @test ConvexDamage.radius(defomesh3) == 0.5

    # non equidistant meshes
    defomesh_axes_diag = -2.0:0.5:2; defomesh_axes_off = -1.0:0.1:1.0
    defomesh1 = DeformationGrid((defomesh_axes_diag, defomesh_axes_off, defomesh_axes_off, defomesh_axes_diag))
    @test eltype(defomesh1) == Tensor{2,2,Float64,4}
    @test size(defomesh1) == (length(defomesh_axes_diag),length(defomesh_axes_off),length(defomesh_axes_off),length(defomesh_axes_diag))
    @test size(defomesh1,1) == size(defomesh1,4) == length(defomesh_axes_diag)
    @test size(defomesh1,2) == size(defomesh1,3) == length(defomesh_axes_off)
    @test defomesh1[2,3,4,5] == Tensor{2,2}((defomesh_axes_diag[2],defomesh_axes_off[3],defomesh_axes_off[4],defomesh_axes_diag[5]))
    @test defomesh1[9,21,21,9] == defomesh1[end]
    @test defomesh1[1,1,1,1] == defomesh1[1]
    @test defomesh1[2] == defomesh1[2,1,1,1]
    @test defomesh1[9] == defomesh1[9,1,1,1]
    @test defomesh1[10] == defomesh1[1,2,1,1]
    @test ConvexDamage.δ(defomesh1) == 0.1
    @test ConvexDamage.center(defomesh1) == Tensor{2,2}((0.0,0.0,0.0,0.0))
    @test ConvexDamage.radius(defomesh1) == 2.0

    defomesh_axes_diag = -1.0:0.5:2; defomesh_axes_off = -3.0:0.1:0.0
    defomesh2 = DeformationGrid((defomesh_axes_diag, defomesh_axes_off, defomesh_axes_off, defomesh_axes_diag))
    @test size(defomesh2) == (length(defomesh_axes_diag),length(defomesh_axes_off),length(defomesh_axes_off),length(defomesh_axes_diag))
    @test size(defomesh2,1) == size(defomesh2,4) == length(defomesh_axes_diag)
    @test size(defomesh2,2) == size(defomesh2,3) == length(defomesh_axes_off)
    @test ConvexDamage.center(defomesh2) == Tensor{2,2}((0.5,-1.5,-1.5,0.5))
    @test ConvexDamage.δ(defomesh2) == 0.1
    @test ConvexDamage.radius(defomesh2) == 1.5

    defomesh_axes_diag = 1.0:0.1:2; defomesh_axes_off = 0.0:0.1:2.0
    defomesh3 = DeformationGrid((defomesh_axes_diag, defomesh_axes_off, defomesh_axes_off, defomesh_axes_diag))
    @test size(defomesh3) == (length(defomesh_axes_diag),length(defomesh_axes_off),length(defomesh_axes_off),length(defomesh_axes_diag))
    @test size(defomesh3,1) == size(defomesh3,4) == length(defomesh_axes_diag)
    @test size(defomesh3,2) == size(defomesh3,3) == length(defomesh_axes_off)
    @test ConvexDamage.center(defomesh3) == Tensor{2,2}((1.5,1.0,1.0,1.5))
    @test ConvexDamage.δ(defomesh3) == 0.1
    @test ConvexDamage.radius(defomesh3) == 1.0

    @test @inferred(defomesh1[1,1,1,1]) == Tensor{2,2}((-2.0,-1.0,-1.0,-2.0))
    @test @inferred(Union{Tensor{2,2},Nothing}, Base.iterate(defomesh1,1)) == (Tensor{2,2}((-2.0,-1.0,-1.0,-2.0)),2)
    @test @inferred(Union{Tensor{2,2},Nothing}, Base.iterate(defomesh1,100000)) === nothing
    @test @inferred(ConvexDamage.center(defomesh3)) == Tensor{2,2}((1.5,1.0,1.0,1.5))
    @test @inferred(ConvexDamage.δ(defomesh3)) == 0.1
    @test @inferred(ConvexDamage.radius(defomesh3)) == 1.0
    @test @inferred(size(defomesh3,1)) == size(defomesh3,4) == length(defomesh_axes_diag)
    @test size(defomesh3,1) == @inferred(size(defomesh3,4)) == length(defomesh_axes_diag)
    @test size(defomesh3,1) == size(defomesh3,4) == @inferred(length(defomesh_axes_diag))
end

@testset "Rank One Direction Iterator" begin
    defomesh_axes = -2.0:0.5:2
    defomesh1 = DeformationGrid((defomesh_axes, defomesh_axes, defomesh_axes, defomesh_axes))
    dirs = PrincipalDamageDirections(2)
    @test @inferred Base.iterate(dirs,1) == ((Vec{2}((1,1)),Vec{2}((1,1))),2)
    ((𝐚,𝐛),i) = Base.iterate(dirs,1)
    @test @inferred(ConvexDamage.inbounds_𝐚(defomesh1,𝐚)) && @inferred(ConvexDamage.inbounds_𝐛(defomesh1,𝐛))
end

@testset "Adaptive Convexification" begin
    ac = AdaptiveConvexification(
            interval=[0.001,20.001],
            basegrid_numpoints=50,
            adaptivegrid_numpoints=115,
            exponent=5,
            distribution="fix",
            stepSizeIgnoreHessian=0.05,
            minPointsPerInterval=15,
            radius=3,
            minStepSize=0.03,
            forceAdaptivity=false)
    bm = NeoHooke(
            μ=0.5,
            λ=0.0)
    materialvec = [
                ReconvexifyDamage(;
                    D∞=0.99,
                    D₀=0.5,
                    base_material=bm,
                    convexstrategy=ac)
                RelaxedDamage(;
                    D∞=0.99,
                    D₀=0.5,
                    base_material=bm,
                    convexstrategy=ac)
                ]
    @testset "build_buffer()" begin
        buf = @inferred ConvexDamage.build_buffer(ac)
        @test typeof(buf) == (ConvexDamage.AdaptiveConvexificationBuffer1D{Vector{Tensor{2,1,Float64,1}},Vector{Float64},Vector{Tensor{4,1,Float64,1}}})
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
        for material in materialvec
            buf = ConvexDamage.build_buffer(material.convexstrategy)
            state = ConvexDamage.init_materialstate(material)
            F = Tensors.Tensor{2,1}((3.1,))
            W_min, Fp, Fm = @inferred ConvexDamage.convexify(F, material, state)
            @test isapprox(W_min,0.1537;atol=1e-4)
            @test isapprox(Fm,Tensors.Tensor{2,1,Float64,1}((1.0750,));atol=1e-4)
            @test isapprox(Fp,Tensors.Tensor{2,1,Float64,1}((14.9843,));atol=1e-4)
            # same result for two subsequent calls ??
            @test ConvexDamage.convexify(F,material,state) == (W_min,Fp,Fm)
        end
    end
    @testset "adaptive_1Dgrid!()" begin
        for material in materialvec
            buf = ConvexDamage.build_buffer(material.convexstrategy)
            state = ConvexDamage.init_materialstate(material)
            state.convexificationbuffer.basebuffer.values .= [ConvexDamage.W_energy(x, material, state.damage) for x in state.convexificationbuffer.basebuffer.grid]
            state.convexificationbuffer.basegrid_∂²W .= [Tensors.hessian(i->ConvexDamage.W_energy(i,material,state.damage), x) for x in state.convexificationbuffer.basebuffer.grid]
            Fpm = @inferred ConvexDamage.adaptive_1Dgrid!(material.convexstrategy,state.convexificationbuffer)
            gridnorm = norm(getindex.(state.convexificationbuffer.adaptivebuffer.grid,1))
            @test isapprox(gridnorm,119.80529;atol=1e-5)
            for (i,f) in enumerate([0.001,0.24589,0.7356,1.2254,14.6948,20.001])
                @test isapprox(Fpm[i],Tensors.Tensor{2,1}((f,));atol=1e-4)
            end
            #same result for subsequent calls?
            @test @inferred(ConvexDamage.adaptive_1Dgrid!(material,state)) == Fpm
            @test norm(getindex.(state.convexificationbuffer.adaptivebuffer.grid,1)) == gridnorm
        end
    end
    @testset "check_hessian()"
    begin
        for material in materialvec
            buf = ConvexDamage.build_buffer(material.convexstrategy)
            state = ConvexDamage.init_materialstate(material)
            state.convexificationbuffer.basegrid_∂²W .= [Tensors.hessian(i->ConvexDamage.W_energy(i,material,state.damage), x) for x in state.convexificationbuffer.basebuffer.grid]
            @inferred ConvexDamage.check_hessian(material.convexstrategy,state.convexificationbuffer)
            F_hes = ConvexDamage.check_hessian(state.convexificationbuffer.basegrid_∂²W, state.convexificationbuffer.basebuffer.grid, material.convexstrategy)
            for (i,F) in enumerate([0.40916 2.44997])
                @test isapprox(F_hes[i],Tensors.Tensor{2,1}((F,));atol=1e-4)
            end
            @test ConvexDamage.check_hessian(state.convexificationbuffer.basegrid_∂²W, state.convexificationbuffer.basebuffer.grid, material.convexstrategy) == F_hes
        end
        # check for different data types
        for T_F in [Float64, Tensors.Tensor{2,1,Float64,1}]
            for T_∂²W in [Float64, Tensors.Tensor{4,1,Float64,1}]
                F = ones(T_F,11).*      [0.0, 1.0, 1.05, 1.1, 1.14999, 1.19998, 6.0, 7.0, 8.0, 9.0, 10.0]
                ∂²W = ones(T_∂²W,11).*  [1.0, 2.0, 1.00, 2.0, 1.00000, 2.00000, 2.0, 1.9, 2.0, 1.9, 1.9]
                @test ConvexDamage.check_hessian(∂²W, F,ac) == ones(T_F,2).*[1.05, 7.0]
            end
        end
    end
    @testset "check_slope()" begin
        for material in materialvec
            buf = ConvexDamage.build_buffer(material.convexstrategy)
            state = ConvexDamage.init_materialstate(material)
            state.convexificationbuffer.basebuffer.values .= [ConvexDamage.W_energy(x, material, state.damage) for x in state.convexificationbuffer.basebuffer.grid]
            @inferred ConvexDamage.check_slope(state.convexificationbuffer)
            F_slp = @inferred ConvexDamage.check_slope(state.convexificationbuffer.basebuffer.grid, state.convexificationbuffer.basebuffer.values)
            for (i,F) in enumerate([0.001, 1.2254, 14.6948, 20.001])
                @test isapprox(F_slp[i],Tensors.Tensor{2,1}((F,));atol=1e-4)
            end
            # subsequent call returns same result??
            @test ConvexDamage.check_slope(state.convexificationbuffer.basebuffer.grid, state.convexificationbuffer.basebuffer.values) == F_slp
        end
        # test for different datatypes
        for T_F in [Float64, Tensors.Tensor{2,1,Float64,1}]
            for T_W in [Float64, Tensors.Tensor{4,1,Float64,1}]
                F = ones(T_F,11).*  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
                W = ones(T_W,11).*  [9.0, 8.0, 7.0, 6.1, 5.3, 4.6, 5.0, 6.0, 7.1, 8.0, 9.0]
                @test ConvexDamage.check_slope(F, W) == ones(T_F,4).*[0.0, 7.0, 9.0, 10.0]
            end
        end
        # ====== check helper fcns ==========
        # iterator()
        mask = ones(Bool,10); mask[2:6].=zeros(Bool,5);
        @inferred ConvexDamage.iterator(4,mask;dir=1)
        @test isapprox(ConvexDamage.iterator(1,mask;dir=1),7;atol=0)
        @test isapprox(ConvexDamage.iterator(7,mask;dir=-1),1;atol=0)
        @test isapprox(ConvexDamage.iterator(1,mask;dir=-1),1;atol=0)
        @test isapprox(ConvexDamage.iterator(10,mask;dir=1),10;atol=0)
        # is_convex()
        for T1 in [Tensors.Tensor{2,1}, Tensors.Tensor{4,1}, Float64]
            for T2 in [Tensors.Tensor{2,1}, Tensors.Tensor{4,1}, Float64]
                P1 = (1.0*one(T1), 1.0*one(T2))
                P2o = (2.0*one(T1), 3.0*one(T2))
                P2m = (2.0*one(T1), 2.0*one(T2))
                P2u = (2.0*one(T1), 1.0*one(T2))
                P3 = (3.0*one(T1), 3.0*one(T2))
                @inferred ConvexDamage.is_convex(P1,P2m,P3)
                @test ConvexDamage.is_convex(P1,P2o,P3)==false # non convex
                @test ConvexDamage.is_convex(P1,P2m,P3)==true # convex (line --> not strictly)
                @test ConvexDamage.is_convex(P1,P2u,P3)==true # strictly convex
            end
        end
    end
    @testset "combine()" begin
        for material in materialvec
            buf = ConvexDamage.build_buffer(material.convexstrategy)
            state = ConvexDamage.init_materialstate(material)
            state.convexificationbuffer.basebuffer.values .= [ConvexDamage.W_energy(x, material, state.damage) for x in state.convexificationbuffer.basebuffer.grid]
            state.convexificationbuffer.basegrid_∂²W .= [Tensors.hessian(i->ConvexDamage.W_energy(i,material,state.damage), x) for x in state.convexificationbuffer.basebuffer.grid]
            F_slp = getindex.(ConvexDamage.check_slope(state.convexificationbuffer.basebuffer.grid, state.convexificationbuffer.basebuffer.values),1)
            F_hes = getindex.(ConvexDamage.check_hessian(state.convexificationbuffer.basegrid_∂²W, state.convexificationbuffer.basebuffer.grid, material.convexstrategy),1)
            for T in [Float64, Tensors.Tensor{2,1,Float64,1}, Tensors.Tensor{4,1,Float64,1}]
                F_s = ones(T,length(F_slp)).*F_slp
                # F_hes hält werte
                F_h = ones(T,length(F_hes)).*F_hes
                Fpm = @inferred ConvexDamage.combine(F_s,F_h,0.4)
                for (i,f) in enumerate([0.001,0.24589,0.7356,1.2254,14.6948,20.001])
                    @test isapprox(Fpm[i],f*one(T); atol=1e-4)
                end
                # F_hes ist leer
                F_h = Vector{T}()
                Fpm = ConvexDamage.combine(F_s,F_h)
                for (i,f) in enumerate([0.001,1.2254,14.6948,20.001])
                    @test isapprox(Fpm[i],f*one(T); atol=1e-4)
                end
            end
        end
    end

    @testset "discretize_interval()"  begin
        for material in materialvec
            buf = ConvexDamage.build_buffer(material.convexstrategy)
            state = ConvexDamage.init_materialstate(material)
            state.convexificationbuffer.basebuffer.values .= [ConvexDamage.W_energy(x, material, state.damage) for x in state.convexificationbuffer.basebuffer.grid]
            state.convexificationbuffer.basegrid_∂²W .= [Tensors.hessian(i->ConvexDamage.W_energy(i,material,state.damage), x) for x in state.convexificationbuffer.basebuffer.grid]
            F_slp = getindex.(ConvexDamage.check_slope(state.convexificationbuffer.basebuffer.grid, state.convexificationbuffer.basebuffer.values),1)
            F_hes = getindex.(ConvexDamage.check_hessian(state.convexificationbuffer.basegrid_∂²W, state.convexificationbuffer.basebuffer.grid, material.convexstrategy),1)
            for T in [Float64, Tensors.Tensor{2,1,Float64,1}]
                F_s = ones(T,length(F_slp)).*F_slp
                F_h = ones(T,length(F_hes)).*F_hes
                Fpm = ConvexDamage.combine(F_s,F_h)
                Fret = ones(T,length(buf.adaptivebuffer.grid)).*getindex.(buf.adaptivebuffer.grid,1)
                @inferred ConvexDamage.discretize_interval(Fret, Fpm, ac)
                @test isapprox(norm(getindex.(Fret,1)),119.80529; atol=1e-5)
                # ================= helper fcns =================
                # project() and Polynomial()
                P = @inferred ConvexDamage.Polynomial(Fpm[4], Fpm[5]-Fpm[4], 30, material.convexstrategy)
                @inferred project(P,0)
                @test isapprox(project(P,0), Fpm[4]; atol=1e-4)
                @test isapprox(project(P,30), Fpm[5]; atol=1e-4)
                @test isapprox(project(P,15), (Fpm[4]+Fpm[5])/2; atol=1e-4)
                Fadap = getindex.(map(x->project(P,x),collect(0:30)),1)
                for i in 1:30 # check monotonicity of grid
                    @test Fadap[i]<Fadap[i+1]
                end
                # distribute_gridpoints()
                pntsperinterval = Vector{Int}(zeros(length(Fpm)-1))
                @inferred ConvexDamage.distribute_gridpoints!(pntsperinterval, Fpm, ac)
                @test sum(pntsperinterval)==ac.adaptivegrid_numpoints-1
                for (i,n) in enumerate([8 16 16 40 34])
                    @test pntsperinterval[i]===n
                end
            end
            # inv_m()
            mask = Vector{Bool}([0,1,0,1,0,0,0,0,1,1,1,0,1])
            inv_mask = @inferred ConvexDamage.inv_m(mask)
            for i in 1:length(mask)
                @test mask[i]!=inv_mask[i]
            end
        end
    end
end

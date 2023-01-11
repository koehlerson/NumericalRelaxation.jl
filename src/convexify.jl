@doc raw"""
    GrahamScan{T<:Number} <: AbstractConvexification

Datastructure that implements in `convexify` dispatch the discrete one-dimensional convexification of a line without deletion of memory.
This results in a complexity of $\mathcal{O}(N)$.

# Kwargs
- `δ::T = 0.01`
- `start::T = 0.9`
- `stop::T = 20.0`
"""
Base.@kwdef struct GrahamScan{T<:Number} <: AbstractConvexification
    δ::T = 0.01
    start::T = 0.9
    stop::T = 20.0
end

δ(s::GrahamScan) = s.δ

function build_buffer(convexification::GrahamScan{T}) where T
    basegrid_F = [Tensors.Tensor{2,1}((x,)) for x in range(convexification.start,convexification.stop,step=convexification.δ)]
    #basegrid_F = collect(range(convexification.start,convexification.stop,step=convexification.δ))
    basegrid_W = zeros(T,length(basegrid_F))
    return ConvexificationBuffer1D(basegrid_F,basegrid_W)
end

@doc raw"""
    convexify(graham::GrahamScan{T2}, buffer::ConvexificationBuffer1D{T1,T2}, W::FUN, F, xargs::Vararg{Any,XN}) where {T1,T2,FUN,XN}  -> W_convex::Float64, F⁻::Tensor{2,1}, F⁺::Tensor{2,1}
Function that implements the convexification on equidistant grid without deletion in $\mathcal{O}(N)$.
"""
function convexify(graham::GrahamScan{T2}, buffer::ConvexificationBuffer1D{T1,T2}, W::FUN, F::T1, xargs::Vararg{Any,XN}) where {T1,T2,FUN,XN}
    #init buffer for new convexification run
    for (i,x) in enumerate(graham.start:graham.δ:graham.stop)
        tmp = T1(x)
        buffer.grid[i] = tmp
        buffer.values[i] = W(tmp, xargs...)
    end
    #convexify
    convexgrid_n = convexify_nondeleting!(buffer.grid,buffer.values)
    # return W at F
    id⁺ = findfirst(x -> x >= F, @view(buffer.grid[1:convexgrid_n]))
    id⁻ = findlast(x -> x <= F,  @view(buffer.grid[1:convexgrid_n]))
    id⁺ == id⁻ ? (id⁻ -= 1) : nothing
    # reorder below to be agnostic w.r.t. tension and compression
    support_points = [buffer.grid[id⁺],buffer.grid[id⁻]] #F⁺ F⁻ assumption
    values_support_points = [buffer.values[id⁺],buffer.values[id⁻]] # W⁺ W⁻ assumption
    _perm = sortperm(values_support_points)
    W_conv = values_support_points[_perm[1]] + ((values_support_points[_perm[2]] - values_support_points[_perm[1]])/(support_points[_perm[2]] - support_points[_perm[1]]))*(F - support_points[_perm[1]])
    return W_conv, support_points[_perm[2]], support_points[_perm[1]]
end

####################################################
####################################################
###################  Adaptive 1D ###################
####################################################
####################################################

@doc raw"""
        AdaptiveGrahamScan <: AbstractConvexification

struct that stores all relevant information for adaptive convexification.

# Fields
- `interval::Vector{Float64}`
- `basegrid_numpoints::Int64`
- `adaptivegrid_numpoints::Int64`
- `exponent::Int64`
- `distribution::String`
- `stepSizeIgnoreHessian::Float64`
- `minPointsPerInterval::Int64`
- `radius::Float64`
- `minStepSize::Float64`
- `forceAdaptivity::Bool`


# Constructor
    AdaptiveGrahamScan(interval; basegrid_numpoints=50, adaptivegrid_numpoints=115, exponent=5, distribution="fix", stepSizeIgnoreHessian=0.05, minPointsPerInterval=15, radius=3, minStepSize=0.03, forceAdaptivity=false)
"""
Base.@kwdef struct AdaptiveGrahamScan <: AbstractConvexification
    interval::Vector{Float64}
    basegrid_numpoints::Int64 = 50
    adaptivegrid_numpoints::Int64 = 115
    exponent::Int64 = 5
    distribution::String = "fix"
    stepSizeIgnoreHessian::Float64 = 0.05     # minimale Schrittweite für die Hesse berücksichtigt wird 
    minPointsPerInterval::Int64 = 15
    radius::Float64 = 3                       # nur relevant für: distribution = "fix"
    minStepSize::Float64 = 0.03
    forceAdaptivity::Bool = false
end

δ(s::AdaptiveGrahamScan) = step(range(s.interval[1],s.interval[2],length=s.adaptivegrid_numpoints))

function build_buffer(ac::AdaptiveGrahamScan)
    basegrid_F = [Tensors.Tensor{2,1}((x,)) for x in range(ac.interval[1],ac.interval[2],length=ac.basegrid_numpoints)]
    basegrid_W = zeros(Float64,ac.basegrid_numpoints)
    basegrid_∂²W = [Tensors.Tensor{4,1}((x,)) for x in zeros(Float64,ac.basegrid_numpoints)]
    adaptivegrid_F = [Tensors.Tensor{2,1}((x,)) for x in zeros(Float64,ac.adaptivegrid_numpoints)]
    adaptivegrid_W = zeros(Float64,ac.adaptivegrid_numpoints)
    basebuffer = ConvexificationBuffer1D(basegrid_F,basegrid_W)
    adaptivebuffer = ConvexificationBuffer1D(adaptivegrid_F,adaptivegrid_W)
    return AdaptiveConvexificationBuffer1D(basebuffer,adaptivebuffer,basegrid_∂²W)
end

@doc raw"""
    convexify(adaptivegraham::AdaptiveGrahamScan{T2}, buffer::AdaptiveConvexificationBuffer1D{T1,T2}, W::FUN, F::T1, xargs::Vararg{Any,XN}) where {T1,T2,FUN,XN}  -> W_convex::Float64, F⁻::Tensor{2,1}, F⁺::Tensor{2,1}
Function that implements the adaptive Graham's scan convexification without deletion in $\mathcal{O}(N)$.
"""
function convexify(adaptivegraham::AdaptiveGrahamScan, buffer::AdaptiveConvexificationBuffer1D{T1,T2}, W::FUN, F::T1, xargs::Vararg{Any,XN}) where {T1,T2,FUN,XN}
    #init function values **and grid** on coarse grid
    buffer.basebuffer.values .= [W(F, xargs...) for x in buffer.basebuffer.grid]
    buffer.basegrid_∂²W .= [Tensors.hessian(i->W(i,xargs...), x) for x in buffer.basebuffer.grid]

    #construct adpative grid
    adaptive_1Dgrid!(adaptivegraham, buffer)
    #init function values on adaptive grid
    for (i,x) in enumerate(buffer.adaptivebuffer.grid)
        buffer.adaptivebuffer.values[i] = W(x, xargs...)
    end
    #convexify
    convexgrid_n = convexify_nondeleting!(buffer.adaptivebuffer.grid,buffer.adaptivebuffer.values)
    # return W at F
    id⁺ = findfirst(x -> x >= F, @view(buffer.adaptivebuffer.grid[1:convexgrid_n]))
    id⁻ = findlast(x -> x <= F,  @view(buffer.adaptivebuffer.grid[1:convexgrid_n]))
    # reorder below to be agnostic w.r.t. tension and compression
    support_points = [buffer.adaptivebuffer.grid[id⁺],buffer.adaptivebuffer.grid[id⁻]] #F⁺ F⁻ assumption
    values_support_points = [buffer.adaptivebuffer.values[id⁺],buffer.adaptivebuffer.values[id⁻]] # W⁺ W⁻ assumption
    _perm = sortperm(values_support_points)
    W_conv = values_support_points[_perm[1]] + ((values_support_points[_perm[2]] - values_support_points[_perm[1]])/(support_points[_perm[2]][1] - support_points[_perm[1]][1]))*(F[1] - support_points[_perm[1]][1])
    return W_conv, support_points[_perm[2]], support_points[_perm[1]]
end

"""
    is_convex(P1::Tuple, P2::Tuple, P3::Tuple) -> bool
Checks if the triplet of `P1,P2,P3` are convex.
The triplet are ordered points with the structure: P1 = (x , f(x))
"""
is_convex(P1::Tuple,P2::Tuple,P3::Tuple) = (P3[2]-P2[2])/(P3[1][1]-P2[1][1]) >= (P2[2]-P1[2])/(P2[1][1]-P1[1][1])

@doc raw"""
    convexify_nondeleting!(F, W)
Kernel function that implements the actual convexification without deletion in $\mathcal{O}(N)$.
"""
function convexify_nondeleting!(F, W)
    n = 2
    for i in 3:length(F)
        while n >=2 && ~is_convex((F[n], W[n]),(F[n-1], W[n-1]),(F[i], W[i]))
            n -= 1
        end
        n += 1
        F[n] = F[i];   W[n] = W[i]
    end
    return n
end

####################################################
####################################################
##########  Adaptive 1D utility functions ##########
####################################################
####################################################

struct Polynomial{T1<:Union{Float64,Tensors.Tensor{2,1}}}
    distribution::String
    F::T1
    ΔF::T1
    exponent::Int64
    numpoints::Int64
    hₘᵢₙ::Float64
    r::Float64
    n::Float64
    # Parameters for fcn-fitting
    a::T1
    b::T1
    c::T1
    d::T1
    e::T1
    function Polynomial(F::T, ΔF::T, numpoints::Int, ac::AdaptiveGrahamScan) where {T}#exponent::Int, numpoints, distribution="fix", r=1.0, hₘᵢₙ=0.00001) where {T}
        if ac.distribution == "var"
            c = F
            b = one(T)* ac.minStepSize
            a = one(T)* (2/numpoints)^ac.exponent*(ΔF[1]/2-ac.minStepSize*numpoints/2)
            d = one(T)* 0.0
            e = one(T)* 0.0
            n = 0.0
            rad = copy(ac.radius)
        elseif ac.distribution == "fix"
            rad =  ac.radius<ΔF[1]/2 ? ac.radius/1 : ΔF[1]/2
            c = F
            b = one(T)* ac.minStepSize
            d = one(T)* (1/(2*numpoints)*(sqrt((ΔF[1]-(ac.exponent-1)*(b[1]*numpoints-2*rad))^2+4*b[1]*numpoints*(ac.exponent-1)*(ΔF[1]-2*rad))-b[1]*numpoints*ac.exponent+b[1]*numpoints+ΔF[1]+2*ac.exponent*rad-2*rad))
            n = (rad-ΔF[1]/2)/d[1]+numpoints/2
            e = F+ΔF/2-d*numpoints/2
            a = (d-b)/(ac.exponent*n^(ac.exponent-1))
        end
        return new{T}(ac.distribution, F, ΔF, ac.exponent, numpoints, ac.minStepSize, rad, n, a, b, c, d, e)
    end
end

@doc raw"""
        function adaptive_1Dgrid!(ac::AdaptiveGrahamScan, ac_buffer::AdaptiveConvexificationBuffer1D{T1,T2,T3}) where {T1,T2,T3}
            ...
            return  F⁺⁻
        end

Based on any grid `ac_buffer.basebuffer.grid` and coresponding function values `ac_buffer.basebuffer.values` and 
its second derivative `ac_buffer.basegrid_∂²W`, a
set of points of interest `F⁺⁻` will be determined. Based on this set of points and
different parameters stored in `ac`
an adaptive grid will be constructed such that grid resolution is highest at these points.

The resultiong grid will be broadcasted into `ac_buffer.adaptivebuffer.grid`.

F⁺⁻ will be determined by checking the slope of mathematical function W(F). Start and end
points of non-convex subintervals will be stored. Additionally all minima of ∂²W(F) serve
as points of interest as well (only if step size at this point is greater than
`ac.stepSizeIgnoreHessian`).
"""
function adaptive_1Dgrid!(ac::AdaptiveGrahamScan, ac_buffer::AdaptiveConvexificationBuffer1D{T1,T2,T3}) where {T1,T2,T3}
    Fₕₑₛ = check_hessian(ac, ac_buffer)
    Fₛₗₚ = check_slope(ac_buffer)
    F⁺⁻ = combine(Fₛₗₚ, Fₕₑₛ)
    discretize_interval(ac_buffer.adaptivebuffer.grid, F⁺⁻, ac)
    return F⁺⁻
end

function check_hessian(∂²W::Vector{T2}, F::Vector{T1}, params::AdaptiveGrahamScan) where {T1,T2}
    length(∂²W) == length(F) ? nothing : error("cannot process arguments of different length.")
    Fₕₑₛ = zeros(T1,0)
    for i in 2:length(∂²W)-1
        if (∂²W[i][1] < ∂²W[i-1][1]) && (∂²W[i][1] < ∂²W[i+1][1]) && ((F[i+1][1]-F[i-1][1])/2 > params.stepSizeIgnoreHessian)
            push!(Fₕₑₛ,F[i])
        end
    end
    return Fₕₑₛ
end

function check_hessian(params::AdaptiveGrahamScan, ac_buffer::AdaptiveConvexificationBuffer1D)
    return check_hessian(ac_buffer.basegrid_∂²W, ac_buffer.basebuffer.grid, params)
end

function check_slope(ac_buffer::AdaptiveConvexificationBuffer1D)
    return check_slope(ac_buffer.basebuffer.grid,ac_buffer.basebuffer.values)
end

function check_slope(F::Vector{T2}, W::Vector{T1}) where {T2,T1}
    mask = ones(Bool,length(F))
    i = 1   # linker Iterator
    k = 2   # rechter Iterator
    r = iterator(k, mask; dir=1) # r = 3
    flag_l = false
    flag_r = false
    while r < length(W)
        r = iterator(k, mask; dir=1) # temp iterator
        if ~is_convex((F[i],W[i]), (F[k],W[k]), (F[r],W[r]))
            int_konvex_l = true
            int_konvex_r = false
            while ~(int_konvex_l && int_konvex_r)
                r = iterator(k, mask; dir=1)
                #k nach rechts bis rechte Seite konvex
                if r < length(W)    #falls rand des Intervalls erreicht....
                    r = iterator(k, mask; dir=1)
                    while ~is_convex((F[i],W[i]), (F[k],W[k]), (F[r],W[r]))
                        mask[k] = 0
                        if r == length(W)
                            flag_r = true
                            break
                        end
                        k = iterator(k, mask; dir=1)
                        r = iterator(k, mask; dir=1)
                        int_konvex_l = false
                    end
                elseif ~flag_r      #....Warnung ausgeben
                    mask[k] = 0
                    k = iterator(k, mask; dir=1)
                    flag_r = true
                end
                int_konvex_r = true
                #i nach links bis linke Seite konvex
                if i > 1    #falls rand des Intervalls erreicht....
                    l = iterator(i, mask; dir=-1)
                    while ~is_convex((F[l],W[l]), (F[i],W[i]), (F[k],W[k]))
                        mask[i] = 0
                        if l == 1
                            flag_l = true
                            break
                        end
                        i = iterator(i, mask; dir=-1)
                        l = iterator(i, mask; dir=-1)
                        int_konvex_r = false
                    end
                elseif ~flag_l      #....Warnung ausgeben
                    flag_l = true
                end
                int_konvex_l = true
            end
        else
            i = iterator(i, mask; dir=1)
            k = iterator(k, mask; dir=1)
        end
    end

    F_info = zeros(typeof(F[1]),1)
    F_info[1] = F[1]
    #if flag_l
    #    @info("linker Rand in nicht konveFem Bereich")
    #end

    for i in 1:length(F)-1
        if (mask[i]==0) && (mask[i+1]==1)
            push!(F_info,F[i+1])
        elseif (mask[i]==1) && (mask[i+1]==0)
            push!(F_info,F[i])
        end
    end
    #if flag_r
    #    @info("rechter Rand in nicht konvexem Bereich")
    #end
    push!(F_info,F[end])
    return F_info
end

function combine(X_slp::Array{T}, X_hes::Array{T},d=0.4::AbstractFloat) where {T}
    # X_HEssian --> aus ∂²W∂x² extrahierte Minima.
    # X_Slope --> aus den Funktionswerten herausgefiltertete Start- und Endpunkte nicht konv. Bereiche 
    #=d       --> relative Distanz zwischen Minima in X_Hessian und nächstem/vorherigem Punkt an dem 
                  Intervallgrenze gesetzt werden soll =#
    X_Slp = copy(X_slp)
    X_Hes = copy(X_hes)
    for i in 1:length(X_Hes)
        X_Hes[i]>X_Slp[1] && X_Hes[i]<X_Slp[end] ? nothing : error("X_Hes[$i]=$(X_Hes[i][1]) not within interval [$(X_Slp[1][1]), $(X_Slp[end][1])]")
    end
    if d>0.4999
        d = 0.4999
    end
    if ~isempty(X_Hes)
        X_mtrx_1 = ones(T,length(X_Slp)+length(X_Hes))
        X_mtrx_2 = ones(Int64,length(X_Slp)+length(X_Hes))
        j = 1; # Iterator für X_Slp
        k = 1; # Iterator für X_Hes

        push!(X_Hes,X_Slp[end]+one(T))  # damit Schleife auf Index [end+1] zugreifen kann

         # Konstruktionsmatrix erzeugen
            #= z.B.
            [X_mtrx_1 ^T =   [0.001 0.501 0.701 1.001 1.201 2.901 5.001;
             X_mtrx_2]        1     2     1     0     1     1     1     ]
            -> 1. Zeile: koordinaten relevanter Punkte (Minimum Hesse oder Start/Ende konvexer Berch)
            -> 2. Zeile: 1 -> aus X_Slp; 2/0 -> aus X_Hes;  =#

        for i in 1:length(X_mtrx_1)
            if X_Slp[j][1] > X_Hes[k][1]
                X_mtrx_1[i] = X_Hes[k]
                X_mtrx_2[i] = iseven(j) ? 2 : 0 # Marker -> X_Hes in konvx Bereich (0) sonst (2)
                k += 1
            elseif X_Slp[j][1] < X_Hes[k][1]
                X_mtrx_1[i] = X_Slp[j]
                j += 1
            else
                X_mtrx_1[i] = X_Hes[k]
                X_mtrx_2[i] = 0
                k+=1
            end
        end
        # X_res aus Konstruktionsmatrix zusammensetzen
        X_res = zeros(T,Int(sum(X_mtrx_2[1:end])))
        j = 1
        for i in 1:length(X_mtrx_1)
            if X_mtrx_2[i] == 1
                X_res[j] = X_mtrx_1[i]
                j += 1
            elseif X_mtrx_2[i] == 2
                X_res[j] = X_mtrx_1[i] - d*(X_mtrx_1[i]-X_mtrx_1[i-1])
                X_res[j+1] = X_mtrx_1[i] + d*(X_mtrx_1[i+1]-X_mtrx_1[i])
                j += 2
            end
        end

        return unique(X_res)
    else
        return unique(X_Slp)
    end
end

function discretize_interval(Fₒᵤₜ::Array{T}, F⁺⁻::Array{T}, ac::AdaptiveGrahamScan) where {T}
    if (length(F⁺⁻) > 2) || (ac.forceAdaptivity) # is function convex ?
        numIntervals = length(F⁺⁻)-1
        gridpoints_oninterval = Array{Int64}(undef,numIntervals)
        distribute_gridpoints!(gridpoints_oninterval, F⁺⁻, ac)
        # ================================================================================
        # ===================================  fill vector  ==============================
        # ================================================================================
        ∑gridpoints = sum(gridpoints_oninterval)
        ∑j = 0
        for i=1:numIntervals
            P = Polynomial(F⁺⁻[i],F⁺⁻[i+1]-F⁺⁻[i], gridpoints_oninterval[i], ac)
            j = 0
            while j < gridpoints_oninterval[i]
                Fₒᵤₜ[∑j+j+1] = project(P,j)
                j += 1
            end
            ∑j += gridpoints_oninterval[i]; 
        end
        Fₒᵤₜ[end] = F⁺⁻[end]
        return nothing
    else # if function already convex
        Fₒᵤₜ .= collect(range(F⁺⁻[1],F⁺⁻[2]; length=ac.adaptivegrid_numpoints))
        return nothing
    end
end

function inv_m(mask::Array{T}) where {T}
    return ones(T,size(mask)) - mask
end

function distribute_gridpoints!(vecₒᵤₜ::Array, F⁺⁻::Array, ac::AdaptiveGrahamScan)
    numIntervals = length(F⁺⁻)-1
    gridpoints_oninterval = copy(vecₒᵤₜ)
    if ac.distribution == "var"
        # ================================================================================
        # ================= Stuetzstellen auf Intervalle aufteilen =======================
        # ================================================================================
        for i=1:numIntervals
            gridpoints_oninterval[i] = Int(round((F⁺⁻[i+1]-F⁺⁻[i])/(F⁺⁻[end]-F⁺⁻[1]) * (ac.adaptivegrid_numpoints-1)))                    
        end
        # ================================================================================
        # ======== korrektur --> um vorgegebene Anzahl an Gitterpunkten einzuhalten ======
        # ================================================================================
        # normierung
        norm_gridpoints_oninterval = gridpoints_oninterval/sum(gridpoints_oninterval)
        gridpoints_oninterval = Int.(round.(norm_gridpoints_oninterval*(ac.adaptivegrid_numpoints-1))) 
        # Mindestanzahl eingehlaten?
        for i in 1:length(gridpoints_oninterval)
            gridpoints_oninterval[i] = max(ac.minPointsPerInterval,gridpoints_oninterval[i])
        end
        # differenz ausgleichen
        ∑gridpoints = sum(gridpoints_oninterval)
        if ∑gridpoints != (ac.adaptivegrid_numpoints-1)
            dif = (ac.adaptivegrid_numpoints-1) - ∑gridpoints
            iₘₐₓ = 1
            for i in 2:numIntervals
                gridpoints_oninterval[i]>gridpoints_oninterval[iₘₐₓ] ? iₘₐₓ = i : ()
            end
            gridpoints_oninterval[iₘₐₓ] += dif
        end
        # Einträge übertragen
        vecₒᵤₜ .= gridpoints_oninterval
        return nothing
    elseif ac.distribution == "fix"
        mask_active = ones(Bool, numIntervals)
        mask_active_last = zeros(Bool, numIntervals)
        cnt = 1
        while (sum(mask_active_last-mask_active)!=0) && (cnt<=10)
            # ================================================================================
            # ================= Stuetzstellen auf Intervalle aufteilen =======================
            # ================================================================================ 
            mask_active_last = copy(mask_active)
            activeIntervals = sum(mask_active)
            activeIntervals==0 ? error("Could not distribute grid points among intervalls. Try to reduce number of grid points or decrease minimum step size.") : nothing 
            numGridpointsOnRadius =
                Int(round( (ac.adaptivegrid_numpoints-1-sum(gridpoints_oninterval.*inv_m(mask_active)))
                /(activeIntervals) ))
            radPol = Polynomial(0.0,2*ac.radius, numGridpointsOnRadius, ac)
            hₘₐₓ =
                (project(radPol, numGridpointsOnRadius/2+0.001)
                -project(radPol, numGridpointsOnRadius/2-0.001)) / 0.002
            for i in 1:numIntervals
                if mask_active[i] == 1
                    linPartOfF = max((F⁺⁻[i+1][1]-F⁺⁻[i][1])-2*ac.radius,0)
                    gridpoints_oninterval[i] =
                        Int(round( (ac.adaptivegrid_numpoints-1)/(activeIntervals) + linPartOfF/hₘₐₓ ))
                end
            end
            # ================================================================================
            # ======== korrektur --> um vorgegebene Anzahl an Gitterpunkten einzuhalten ======
            # ================================================================================
            # normierung
            norm_gridpoints_oninterval = gridpoints_oninterval./(sum(mask_active.*gridpoints_oninterval))
            norm_gridpoints_oninterval .*= mask_active
            active_points = ac.adaptivegrid_numpoints - 1 - sum(inv_m(mask_active).*gridpoints_oninterval)
            gridpoints_oninterval = Int.(round.(inv_m(mask_active).*gridpoints_oninterval +     norm_gridpoints_oninterval*active_points))
            # reduktion falls minimale Schrittweite*Stützpunkte > Intervallbreite
            for i in 1:length(gridpoints_oninterval)
                maxnum = floor((F⁺⁻[i+1][1]-F⁺⁻[i][1])/ac.minStepSize)
                if gridpoints_oninterval[i] > maxnum
                    gridpoints_oninterval[i] = maxnum
                    mask_active[i] = 0
                end
            end
            cnt += 1
        end
        # differenz ausgleichen
        ∑gridpoints = sum(gridpoints_oninterval)
        if ∑gridpoints != (ac.adaptivegrid_numpoints-1)
            dif = (ac.adaptivegrid_numpoints-1) - ∑gridpoints
            iₘₐₓ = 1
            for i in 2:numIntervals
                gridpoints_oninterval[i]>gridpoints_oninterval[iₘₐₓ] ? iₘₐₓ = i : ()
            end
            gridpoints_oninterval[iₘₐₓ] += dif
        end
        # Einträge übertragen
        vecₒᵤₜ .= gridpoints_oninterval
        return nothing
    end
end

function iterator(i, mask; dir=1)
    # check input
    dir in [1, -1] ? nothing : error("search direction must be either positive (1) or negative (-1)")
    ((i>=1) && (i<=length(mask))) ? nothing : error("tried to access vector entry at position "*string(i)*". Must lie between 1 and "*string(length(mask))*".")
    ~(mask[1] == 0) ? nothing : error("first entry of mask is not supposed to be set to false")
    ~(mask[end] == 0) ? nothing : error("last entry of mask is not supposed to be set to false")        

    if dir == -1
        id_next = findlast(@view mask[1:(i==1 ? 1 : i-1)])
    else#if dir == 1
        id_next = i + (i==length(mask) ? 0 : findfirst(@view mask[i+1:end]))
    end
    return id_next!=nothing ? id_next : error("findlast/findfirst returned value of type \"nothing\".")
end

function project(P::Polynomial, n)
    if P.distribution == "var"
        if n < P.numpoints/2
            pot = 1
            for i=1:P.exponent
                pot *= (n)
            end
            return (pot*P.a + P.b*n + P.c)
        else
            pot = 1
            for i=1:P.exponent
                pot *= (P.numpoints-n)
            end
            return P.F+P.ΔF - (pot*P.a + P.b*(P.numpoints-n))
        end
    else
        if (n>=0) && (n<P.n)
            pot = 1
            for i=1:P.exponent
                pot *= (n)
            end
            return (P.a*pot + P.b*n + P.c)
        elseif (n>=P.n) && (n<P.numpoints-P.n)
            P.d*n+P.e
        elseif (n>=P.numpoints-P.n) && (n<= P.numpoints)
            pot = 1
            for i=1:P.exponent
                pot *= (P.numpoints-n)
            end
            return P.F+P.ΔF - (P.a*pot+P.b*(P.numpoints-n))
        else
            error("projecion-polynomial only defined for indices 0>=j>=$P.numpoints")
        end
    end
end

"""
    build_buffer(convexstrategy::T) where T<:AbstractConvexification
Maps a given convexification strategy `convexstrategy` to an associated buffer.
"""
build_buffer

####################################################
####################################################
############### Multidimensional  ##################
####################################################
####################################################

@doc raw"""
    GradientGrid{dimc,T,R<:AbstractRange{T}}
Lightweight implementation of a structured convexification grid in multiple dimensions.
Computes the requested convexification grid node adhoc and therefore is especially suited for threading (no cache misses).
Implements the `Base.Iterator` interface and other `Base` functions such as, `length`,`size`,`getindex`,`lastindex`,`firstindex`,`eltype`,`axes`
Within the parameterization `dimc` denote the convexification dimensions, `T` the used number type and `R` the number type of the `start`, `step` and `end` value of the axes ranges.

# Constructor
    GradientGrid(axes::NTuple{dimc,R}) where dimc
- `axes::R` is a tuple of discretizations with the order of Tensor{2,2}([x1 y1;x2 y2])

# Fields
- `axes::NTuple{dimc,R}`
- `indices::CartesianIndices{dimc,NTuple{dimc,Base.OneTo{Int64}}}`
"""
struct GradientGrid{dimc,T,R<:AbstractRange{T}}
    axes::NTuple{dimc,R}
    indices::CartesianIndices{dimc,NTuple{dimc,Base.OneTo{Int64}}}
end

function GradientGrid(axes::NTuple{dimc}) where dimc
    indices = CartesianIndices(ntuple(x->length(axes[x]),dimc))
    return GradientGrid(axes,indices)
end

function Base.size(gradientgrid::GradientGrid{dimc}, axes::Int) where dimc
    @assert dimc ≥ axes
    return length(gradientgrid.axes[axes])
end

function Base.size(gradientgrid::GradientGrid{dimc}) where dimc
    NTuple{dimc,Int}(size(gradientgrid,dim) for dim in 1:dimc)
end

function Base.length(b::GradientGrid{dimc}) where dimc
    _size::Int = size(b,1)
    for i in 2:dimc
        _size *= size(b,i)
    end
    return _size
end

getindex_type(b::GradientGrid{4,T}) where {T} = Tensor{2,2,T,4}
getindex_type(b::GradientGrid{9,T}) where {T} = Tensor{2,3,T,9}

function Base.getindex(b::GradientGrid{dimc,T},args...) where {dimc,T}
    @assert length(args) == dimc
    content = NTuple{dimc,T}(b.axes[x][args[x]] for x in 1:dimc)
    return getindex_type(b)(content)
end

function Base.getindex(b::GradientGrid{dimc,T},idx) where {dimc,T}
    content = NTuple{dimc,T}(b.axes[x][idx[x]] for x in 1:dimc)
    return getindex_type(b)(content)
end

function Base.getindex(b::GradientGrid{dimc,T},idx::Int) where {dimc,T}
    ind = b.indices[idx]
    return b[ind]
end

Base.lastindex(b::GradientGrid) = length(b)
Base.firstindex(b::GradientGrid) = 1
Base.axes(b::GradientGrid,d::Int) = Base.OneTo(length(b.axes[d]))
Base.eltype(b::GradientGrid) = getindex_type(b)

Base.IteratorSize(b::GradientGrid{dimc}) where {dimc} = Base.HasShape{dimc}()

function Base.iterate(b::GradientGrid, state=1)
    if state <= length(b)
        return (b[state], state+1)
    else
        return nothing
    end
end

δ(b::GradientGrid{dimc,T}, axes::Int) where {dimc,T} = T(b.axes[axes].step)
δ(b::GradientGrid{dimc}) where {dimc} = minimum(ntuple(x->δ(b,x),dimc))
center(b::GradientGrid{dimc,T}) where {dimc,T} = getindex_type(b)(ntuple(x->radius(b,x)+b.axes[x][1], dimc))
radius(b::GradientGrid, axes::Int) = (b.axes[axes][end] - b.axes[axes][1])/2
radius(b::GradientGrid{dimc}) where {dimc} = maximum(ntuple(x->radius(b,x),dimc))

function inbounds(𝐱::Tensor{2,dimp,T},b::GradientGrid{dimc,T}) where {dimp,dimc,T}
    inbound = ntuple(i->b.axes[i][1] ≤ 𝐱[i] ≤ b.axes[i][end],dimc)
    return all(inbound)
end

@doc raw"""
    GradientGridBuffered{dimc,T,dimp}
Heavyweight implementation of a structured convexification grid in multiple dimensions.
Computes the requested convexification grid within the constructor and only accesses thereafter the `grid` field.
Implements the `Base.Iterator` interface and other `Base` functions such as, `length`,`size`,`getindex`,`lastindex`,`firstindex`,`eltype`,`axes`
Within the parameterization `dimc` denote the convexification dimensions, `T` the used number type and `dimp` the physical dimensions of the problem.

# Constructor
    GradientGridBuffered(axes::NTuple{dimc}) where dimc
- `axes::StepRangeLen{T,R,R}` is a tuple of discretizations with the order of Tensor{2,2}([x1 y1;x2 y2]

# Fields
- `grid::AbstractArray{Tensor{2,dimp,T,dimc},dimc} `
- `indices::CartesianIndices{dimc,NTuple{dimc,Base.OneTo{Int64}}}`
"""
struct GradientGridBuffered{dimc,T,dimp}
    grid::AbstractArray{Tensor{2,dimp,T,dimc},dimc}
    indices::CartesianIndices{dimc,NTuple{dimc,Base.OneTo{Int64}}}
end

function GradientGridBuffered(axes::NTuple{dimc}) where dimc
    indices = CartesianIndices(ntuple(x->length(axes[x]),dimc))
    grid = collect(GradientGrid(axes,indices))
    return GradientGridBuffered(grid,indices)
end

function GradientGridBuffered(defomesh::GradientGrid)
    return GradientGridBuffered(collect(defomesh),defomesh.indices)
end

Base.size(gradientgrid::GradientGridBuffered, axes::Int) = size(gradientgrid.grid,axes)
Base.size(gradientgrid::GradientGridBuffered) = size(gradientgrid.grid)
Base.length(gradientgrid::GradientGridBuffered) = length(gradientgrid.grid)
Base.getindex(gradientgrid::GradientGridBuffered, idx) = gradientgrid.grid[idx]
Base.getindex(gradientgrid::GradientGridBuffered, args...) = gradientgrid.grid[args...]
Base.lastindex(gradientgrid::GradientGridBuffered) = length(gradientgrid)
Base.firstindex(gradientgrid::GradientGridBuffered) = 1
Base.axes(gradientgrid::GradientGridBuffered,d::Int) = Base.axes(gradientgrid.grid,d)
Base.eltype(gradientgrid::GradientGridBuffered) = eltype(gradientgrid.grid)
Base.IteratorSize(gradientgrid::GradientGridBuffered) = Base.IteratorSize(gradientgrid.grid)
Base.iterate(defomesh::GradientGridBuffered, state=1) = Base.iterate(defomesh.grid,state)

δ(b::GradientGridBuffered{dimc,T}, axes::Int) where {dimc,T} = T(round(b.grid[CartesianIndex(ntuple(x->2,dimc))][axes] - b.grid[CartesianIndex(ntuple(x->1,dimc))][axes],digits=1))
δ(b::GradientGridBuffered{dimc}) where {dimc} = minimum(ntuple(x->δ(b,x),dimc))

function center(b::GradientGridBuffered{dimc}) where dimc
    idx = ceil(Int,size(b,1)/2)
    return b.grid[CartesianIndex(ntuple(x->idx,dimc))]
end

radius(b::GradientGridBuffered, axes::Int) = b.grid[end][axes] - center(b)[axes]
radius(b::GradientGridBuffered{dimc}) where {dimc} = maximum(ntuple(x->radius(b,x),dimc))

function inbounds(𝐱::Tensor{2,dimp,T},b::GradientGridBuffered{dimc,T}) where {dimp,dimc,T}
    inbound = ntuple(i->b[1][i] ≤ 𝐱[i] ≤ b[end][i],dimc)
    return all(inbound)
end

function inbounds_𝐚(b::Union{GradientGrid{dimc},GradientGridBuffered{dimc}},𝐚) where {dimc}
    _δ = δ(b) ##TODO
    dimp = isqrt(dimc)
    return ((norm(𝐚)≤(1+dimp*_δ)/_δ) && (𝐚⋅𝐚 ≥ (1-2*dimp)/_δ^2)) #TODO warum hier die norm ohne Wurzel?
end

function inbounds_𝐛(b::Union{GradientGrid{dimc},GradientGridBuffered{dimc}}, 𝐛) where {dimc}
    _δ = δ(b) ##TODO
    r = radius(b) ##TODO welcher?
    dimp = isqrt(dimc)
    return _δ*norm(𝐛) ≤ 2*dimp*r+dimp*_δ # Hinterer Term nur im paper
end

𝐚_bounds(b::Union{GradientGrid{dimc},GradientGridBuffered{dimc}}) where {dimc} = floor(Int,(1+isqrt(dimc)*δ(b))/δ(b)) #TODO largest delta?
𝐛_bounds(b::Union{GradientGrid{dimc},GradientGridBuffered{dimc}}) where {dimc} = floor(Int,2*isqrt(dimc)*radius(b)+isqrt(dimc)*δ(b)/δ(b))#TODO delta?

getaxes(defomesh::GradientGrid) = defomesh.axes
function getaxes(defomesh::GradientGridBuffered{dimc}) where dimc
    start_ = defomesh[1]
    end_ = defomesh[end]
    step =  δ(defomesh)
    return ntuple(x->(start_[x]:δ(defomesh,x):end_[x]),dimc)
end


##################################
#### Rank One Direction Space ####
##################################
abstract type RankOneDirections{dimp} end

@doc raw"""
    ℛ¹Direction{dimp,dimc} <: RankOneDirections{dimp}
Lightweight implementation that computes all rank-one directions within a `grid::GradientGrid` adhoc.
Therefore, also suited for threading purposes, since this avoids cache misses.
Implements the `Base.Iterator` interface and other utility functions.
Within the parameterization `dimp` and `dimc` denote the physical dimensions of the problem and the convexification dimensions, respectively.

# Constructor
    ℛ¹Direction(b::GradientGrid)
- `b::GradientGrid` is a deformation grid discretization

# Fields
- `a_axes::NTuple{dimp,UnitRange{Int}}`
- `b_axes::NTuple{dimp,UnitRange{Int}}`
- `indices::CartesianIndices{dimc,NTuple{dimc,Base.OneTo{Int64}}}`
"""
struct ℛ¹Direction{dimp,dimc} <: RankOneDirections{dimp}
    a_axes::NTuple{dimp,UnitRange{Int}}
    b_axes::NTuple{dimp,UnitRange{Int}}
    indices::CartesianIndices{dimc,NTuple{dimc,Base.OneTo{Int64}}}
end

Base.eltype(b::ℛ¹Direction{dimp}) where dimp = Tuple{Vec{dimp,Int},Vec{dimp,Int}}
Base.IteratorSize(d::ℛ¹Direction{dimp,dimc}) where {dimp,dimc} = Base.HasShape{dimc}()

function ℛ¹Direction(b::GradientGrid)
    a = 𝐚_bounds(b)
    b = 𝐛_bounds(b)
    return ℛ¹Direction((-a:a,-a:a),(0:b,-b:b),CartesianIndices(((a*2+1),(a*2+1),b+1,(b*2+1))))
end

function Base.size(d::ℛ¹Direction{dimp}, axes::Int) where dimp
    @assert dimp^2 ≥ axes
    if axes ≤ dimp
        return length(d.a_axes[axes])
    else
        return length(d.b_axes[(axes-dimp)])
    end
end

function Base.size(d::ℛ¹Direction{dimp,dimc}) where {dimp,dimc}
    NTuple{dimc,Int}(size(d,dim) for dim in 1:dimc)
end

function Base.length(d::ℛ¹Direction{dimp,dimc}) where {dimp,dimc}
    _size::Int = size(d,1)
    for i in 2:dimc
        _size *= size(d,i)
    end
    return _size
end

Base.lastindex(d::ℛ¹Direction) = size(d)
Base.firstindex(d::ℛ¹Direction) = 1

function Base.getindex(d::ℛ¹Direction{dimp},idx) where {dimp}
    return Vec{dimp,Int}(NTuple{dimp,Int}(d.a_axes[x][idx[x]] for x in 1:dimp)),Vec{dimp,Int}(NTuple{dimp,Int}(d.b_axes[x][idx[(x + dimp)]] for x in 1:dimp))
end

function Base.getindex(d::ℛ¹Direction,idx::Int)
    return d[d.indices[idx]]
end

function Base.getindex(d::ℛ¹Direction,args...)
    return d[args]
end

function Base.iterate(d::ℛ¹Direction{dimp,dimc}, state=1) where {dimp,dimc}
    if state <= length(d)
        return (d[state], state+1)
    else
        return nothing
    end
end

@doc raw"""
    ℛ¹DirectionBuffered{dimp,dimc,T} <: RankOneDirections{dimp}
Heavyweight implementation that computes all rank-one directions within a `grid::GradientGridBuffered` within the constructor.
Implements the `Base.Iterator` interface and other utility functions.
Within the parameterization `dimp` and `dimc` denote the physical dimensions of the problem and the convexification dimensions, respectively.

# Constructor
    ℛ¹DirectionBuffered(dirs::ℛ¹Direction)
- `dirs::ℛ¹Direction` collects the reference `dirs` direction and caches them in the `grid` field

# Fields
- `grid::AbstractArray{Tuple{Vec{dimp,T},Vec{dimp,T}},dimc}`
"""
struct ℛ¹DirectionBuffered{dimp,dimc,T} <: RankOneDirections{dimp}
    grid::AbstractArray{Tuple{Vec{dimp,T},Vec{dimp,T}},dimc}
end
ℛ¹DirectionBuffered(dirs::ℛ¹Direction) = ℛ¹DirectionBuffered(collect(dirs))
Base.iterate(d::ℛ¹DirectionBuffered, state=1) = Base.iterate(d.grid, state)

@doc raw"""
    ParametrizedR1Directions{dimp,T} <: RankOneDirections{dimp}
Direction datastructure that computes the reduced rank-one directions in the `dimp` physical dimensions and `dimp`² convexification dimensions.
Implements the `Base.Iterator` interface and other utility functions.
This datastructure only computes the first neighborhood rank-one directions and utilizes the symmetry of the `dimp`² dimensionality.
Since they are quite small in 2² and 3² the directions are precomputed and stored in `dirs`.

# Constructor
    ParametrizedR1Directions(2)
    ParametrizedR1Directions(3)

# Fields
- `dirs::Vector{Tuple{Vec{dimp,T},Vec{dimp,T}}}`
"""
struct ParametrizedR1Directions{dimp,T} <: RankOneDirections{dimp}
    dirs::Vector{Tuple{Vec{dimp,T},Vec{dimp,T}}}
end

function ParametrizedR1Directions(::Val{2})
    rankdirs = Vector{Tuple{Vec{2,Int},Vec{2,Int}}}()
    for i in -1:1, j in -1:1, m in 0:1, n in -1:1
        if (i==j==0) || (m==n==0)
            continue
        else
            push!(rankdirs,(Vec{2}((i,j)),Vec{2}((m,n))))
        end
    end
    return ParametrizedR1Directions(rankdirs)
end

function ParametrizedR1Directions(::Val{3})
    rankdirs = Vector{Tuple{Vec{3,Int},Vec{3,Int}}}()
    for i in -1:1, j in -1:1, k in -1:1, l in 0:1, m in -1:1, n in -1:1
        if (i==j==k==0) || (l==m==n==0)
            continue
        else
            push!(rankdirs,(Vec{3}((i,j,k)),Vec{3}((l,m,n))))
        end
    end
    return ParametrizedR1Directions(rankdirs)
end

ParametrizedR1Directions(dimp::Int) = ParametrizedR1Directions(Val(dimp))
ParametrizedR1Directions(gradientgrid::GradientGrid{dimc}) where dimc = ParametrizedR1Directions(isqrt(dimc))
Base.iterate(d::ParametrizedR1Directions, state=1) = Base.iterate(d.dirs, state)


@doc raw"""
    R1Convexification{dimp,dimc,dirtype<:RankOneDirections{dimp},T1,T2,R} <: Convexification
Datastructure that is used as an equivalent to `GrahamScan` in the multidimensional rank-one relaxation setting.
Bundles rank-one direction discretization as well as a tolerance and the convexification grid.
# Constructor
    R1Convexification(axes_diag::AbstractRange,axes_off::AbstractRange;dirtype=ℛ¹Direction,dim=2,tol=1e-4)
    R1Convexification(grid::GradientGrid,r1dirs,tol)

# Fields
- `grid::GradientGrid{dimc,T1,R}`
- `dirs::dirtype`
- `tol::T1`
"""
struct R1Convexification{dimp,dimc,dirtype<:RankOneDirections{dimp},T,R} <: AbstractConvexification
    grid::GradientGrid{dimc,T,R}
    dirs::dirtype
    tol::T
end

function R1Convexification(axes_diag::AbstractRange,axes_off::AbstractRange;dirtype=ℛ¹Direction,dim=2,tol=1e-4)
    diag_indices = dim == 2 ? (1,4) : (1, 5, 9)
    gradientgrid = GradientGrid(ntuple(x->x in diag_indices ? axes_diag : axes_off,dim^2))
    dirs = dirtype(gradientgrid)
    return R1Convexification(gradientgrid,dirs,tol)
end

function build_buffer(r1convexification::R1Convexification{dimp,dimc,dirtype,T}) where {dimp,dimc,dirtype,T}
    gradientgrid = r1convexification.grid
    _δ = δ(gradientgrid)
    _r = radius(gradientgrid)
    max_gx = ceil(Int,((2*_r))/_δ^3) + dimp^2
    buffer = [R1ConvexificationThreadBuffer(dimp,max_gx) for i in 1:Threads.nthreads()]
    W_rk1 = linear_interpolation(getaxes(gradientgrid),zeros(size(gradientgrid)),extrapolation_bc=Interpolations.Flat())
    W_rk1_old = deepcopy(W_rk1)
    diff_matrix = zero(W_rk1.itp.itp.coefs)
    laminatetree = Dict{Int,LaminateTree{dimp,T,dimc}}()
    return R1ConvexificationBuffer(buffer,W_rk1,W_rk1_old,diff_matrix,laminatetree)
end

@doc raw"""
    convexify!(r1convexification::R1Convexification,r1buffer::R1ConvexificationBuffer,W::FUN,xargs::Vararg{Any,XN};buildtree=true,maxk=20) where {FUN,XN}
Multi-dimensional parallelized implementation of the rank-one convexification.
If `buildtree=true` the lamination tree is saved in the `r1buffer.laminatetree`.
Note that the interpolation objects within `r1buffer` are overwritten in this routine.
The approximated rank-one convex envelope is saved in `r1buffer.W_rk1`
"""
function convexify!(r1convexification::R1Convexification,r1buffer::R1ConvexificationBuffer,W::FUN,xargs::Vararg{Any,XN};buildtree=false,maxk=20) where {FUN,XN}
    gradientgrid = r1convexification.grid
    directions = r1convexification.dirs
    W_rk1 = r1buffer.W_rk1
    W_rk1.itp.itp.coefs .= [try (isnan(W(F,xargs...)) ? 1000.0 : W(F,xargs...)) catch DomainError 1000.0 end for F in gradientgrid]
    W_rk1_old = r1buffer.W_rk1_old
    diff = r1buffer.diff
    copyto!(W_rk1_old.itp.itp.coefs,W_rk1.itp.itp.coefs)
    copyto!(diff,W_rk1_old.itp.itp.coefs)
    _δ = δ(gradientgrid)
    _r = radius(gradientgrid)
    k = 1
    threadbuffer = r1buffer.threadbuffer
    laminatetree = r1buffer.laminatetree# init full tree
    empty!(laminatetree)
    [empty!(b.partiallaminatetree) for b in r1buffer.threadbuffer]

    while norm(diff, Inf) > r1convexification.tol
        copyto!(W_rk1_old.itp.itp.coefs,W_rk1.itp.itp.coefs)
        Threads.@threads :static for lin_ind_𝐅 in 1:length(gradientgrid)
            𝐅 = gradientgrid[lin_ind_𝐅]
            id = Threads.threadid()
            g_fw = threadbuffer[id].g_fw; g_bw = threadbuffer[id].g_bw; X_fw = threadbuffer[id].X_fw; X_bw = threadbuffer[id].X_bw
            X = threadbuffer[id].X; g = threadbuffer[id].g; h = threadbuffer[id].h; y = threadbuffer[id].y;
            buildtree && (partiallaminatetree = threadbuffer[id].partiallaminatetree)
            for (𝐚,𝐛) in directions
                if inbounds_𝐚(gradientgrid,𝐚) && inbounds_𝐛(gradientgrid,𝐛)
                    𝐀 = _δ^3 * (𝐚 ⊗ 𝐛) # ^3 sollte für jede Dimension richtig sein
                    if norm(𝐀,Inf) > 0
                        ctr_fw = 0
                        ctr_bw = 0
                        for dir in (-1, 1)
                            if dir==-1
                                𝐱 = 𝐅 - 𝐀 # init dir
                                ell = -1 # start bei -1, deswegen -𝐀
                            else
                                𝐱 = 𝐅 # init dir
                                ell = 0 # start bei 0
                            end
                            while inbounds(𝐱,gradientgrid)
                                val = W_rk1_old(𝐱...)
                                if dir == 1
                                    g_fw[ctr_fw+1] = val
                                    X_fw[ctr_fw+1] = ell
                                    ctr_fw += 1
                                else
                                    g_bw[ctr_bw+1] = val
                                    X_bw[ctr_bw+1] = ell
                                    ctr_bw += 1
                                end
                                𝐱 += dir*𝐀
                                ell += dir
                            end
                        end
                        if ((ctr_fw > 0) && (ctr_bw > 0))
                            concat!(g,g_fw,ctr_fw+1,g_bw,ctr_bw) # +1 ctr_fw wegen start bei 0
                            concat!(X,X_fw,ctr_fw+1,X_bw,ctr_bw) # +1 ctr_fw wegen start bei 0
                            g_ss, j = convexify!(g,X, ctr_bw+ctr_fw, h, y)
                            if g_ss < W_rk1.itp.itp.coefs[lin_ind_𝐅]
                                W_rk1.itp.itp.coefs[lin_ind_𝐅] = g_ss
                                if k ≤ maxk && buildtree
                                    l₁ = y[j-1]
                                    l₂ = y[j]
                                    F¯ = 𝐅 + l₁*𝐀
                                    F⁺ = 𝐅 + l₂*𝐀
                                    W¯ = W_rk1_old(F¯...)
                                    W⁺ = W_rk1_old(F⁺...)
                                    laminate = Laminate(F¯,F⁺,W¯,W⁺,𝐀,k)
                                    if haskey(partiallaminatetree,lin_ind_𝐅) # check if thread laminate tree has key 𝐅
                                        if isassigned(partiallaminatetree[lin_ind_𝐅],k) # check if thread laminate tree has already k-level laminates
                                            partiallaminatetree[lin_ind_𝐅][k] = laminate
                                        else
                                            push!(partiallaminatetree[lin_ind_𝐅].laminates,laminate)
                                        end
                                    else # add key with current laminates
                                        partiallaminatetree[lin_ind_𝐅] = LaminateTree([laminate])
                                    end
                                end
                            end
                        end
                    end
                else
                    continue
                end
            end
        end
        diff .= W_rk1.itp.itp.coefs - W_rk1_old.itp.itp.coefs
        k += 1
    end
    buildtree && merge!(laminatetree,getproperty.(r1buffer.threadbuffer,:partiallaminatetree)...)
    nothing
end

@doc raw"""
    convexify!(f, x, ctr, h, y)
Rank-one line convexification algorithm in multiple dimensions without deletion, but in $\mathcal{O}(N)$
"""
function convexify!(f, x, ctr, h, y)
    fill!(h,zero(eltype(h))); fill!(y,zero(eltype(y)))
    last = 2
    h[1] = f[1]; h[2] = f[2];
    y[1] = x[1]; y[2] = x[2];
    for j in 2:ctr-1
        while ((last >=2) && ((x[j+1]-y[last]) * (-h[last]+h[last-1]) + (f[j+1]-h[last]) * (y[last]-y[last-1]) ≤ 0))
            last -= 1
        end
        h[last+1] = f[j+1]
        y[last+1] = x[j+1]
        last += 1
    end
    #last += 1 # TODO ich glaube das gehört auskommentiert, nicht sicher
    y[last] = x[ctr]
    h[last] = f[ctr]
    j = 1
    while (y[j] < 0)
        j += 1
    end
    λ = (h[j]-h[j-1]) / (y[j]-y[j-1])
    g_ss = h[j-1] + λ * -y[j-1]
    return g_ss, j
end

struct BinaryAdaptiveLaminationTree{dimp,dimc,R1Dir<:RankOneDirections{dimp},T,R<:AbstractRange{T}}
    maxlevel::Int
    n_convexpoints::Int
    dirs::R1Dir
    axes::NTuple{dimc,R} 
end

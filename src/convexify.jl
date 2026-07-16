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
struct ParametrizedR1Directions{dimp,T,dimc} <: RankOneDirections{dimp}
    dirs::Vector{Tensor{2,dimp,T,dimc}}
end

function ParametrizedR1Directions(::Val{2};l=1)
    rankdirs = Vector{Tuple{Vec{2,Int},Vec{2,Int}}}()
    for i in -l:l, j in -l:l, m in -l:l, n in -l:l
        if (i==j==0) || (m==n==0)
            continue
        else
            push!(rankdirs,(Vec{2}((i,j)),Vec{2}((m,n))))
        end
    end
    dirs = [𝐚 ⊗ 𝐛 for (𝐚, 𝐛) in rankdirs]
    unique!(dirs)
    return ParametrizedR1Directions(dirs)
end

function ParametrizedR1Directions(::Val{3};l=1)
    rankdirs = Vector{Tuple{Vec{3,Int},Vec{3,Int}}}()
    for i in -l:l, j in -l:l, k in -l:l, m in -l:l, n in -l:l, o in -l:l
        if (i==j==k==0) || (m==n==o==0)
            continue
        else
            push!(rankdirs,(Vec{3}((i,j,k)),Vec{3}((m,n,o))))
        end
    end
    dirs = [𝐚 ⊗ 𝐛 for (𝐚, 𝐛) in rankdirs]
    unique!(dirs)
    #dirs = [Vec{3}([1 0 0]) ⊗ Vec{3}([1 0 0])]
    return ParametrizedR1Directions(dirs)
end

ParametrizedR1Directions(dimp::Int;l=1) = ParametrizedR1Directions(Val(dimp);l=l)
ParametrizedR1Directions(gradientgrid::GradientGrid{dimc}) where dimc = ParametrizedR1Directions(isqrt(dimc))
Base.iterate(d::ParametrizedR1Directions, state=1) = Base.iterate(d.dirs, state)
Base.length(d::ParametrizedR1Directions) = length(d.dirs)

struct ParametrizedDDirections{dimp,T,dimc} <: RankOneDirections{dimp}
    dirs::Vector{Tensor{2,dimp,T,dimc}}
end

function ParametrizedDDirections(::Val{2};l=1)
    rankdirs = Vector{Tensor{2,2,Float64,4}}()
    for i in -l:l, j in -l:l, m in -l:l, n in -l:l
        if (i==j==m==n==0)
            continue
        else
            push!(rankdirs,Tensor{2,2}((i,j,m,n)))
        end
    end
    unique!(rankdirs)
    return ParametrizedDDirections(rankdirs)
end

function ParametrizedDDirections(::Val{3};l=1)
    rankdirs = Vector{Tensor{2,3,Float64,9}}()
    for i in -l:l, j in -l:l, m in -l:l, n in -l:l, k in -l:l, o in -l:l, p in -l:l, q in -l:l, r in -l:l
        if (i==j==m==n==k==o==p==q==r==0)
            continue
        else
            push!(rankdirs,Tensor{2,3}((i,j,m,n,k,o,p,q,r)))
        end
    end
    unique!(rankdirs)
    return ParametrizedDDirections(rankdirs)
end

ParametrizedDDirections(dimp::Int;l=1) = ParametrizedDDirections(Val(dimp);l=l)
ParametrizedDDirections(gradientgrid::GradientGrid{dimc}) where dimc = ParametrizedDDirections(isqrt(dimc))
Base.iterate(d::ParametrizedDDirections, state=1) = Base.iterate(d.dirs, state)
Base.length(d::ParametrizedDDirections) = length(d.dirs)

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
        @show k
        copyto!(W_rk1_old.itp.itp.coefs,W_rk1.itp.itp.coefs)
        Threads.@threads :static for lin_ind_𝐅 in 1:length(gradientgrid)
            𝐅 = gradientgrid[lin_ind_𝐅]
            id = Threads.threadid()
            g_fw = threadbuffer[id].g_fw; g_bw = threadbuffer[id].g_bw; X_fw = threadbuffer[id].X_fw; X_bw = threadbuffer[id].X_bw
            X = threadbuffer[id].X; g = threadbuffer[id].g; h = threadbuffer[id].h; y = threadbuffer[id].y;
            buildtree && (partiallaminatetree = threadbuffer[id].partiallaminatetree)
            for 𝐀 in directions
                if true #inbounds_𝐚(gradientgrid,𝐚) && inbounds_𝐛(gradientgrid,𝐛)
                    𝐀 *= _δ^3 #* (𝐚 ⊗ 𝐛) # ^3 sollte für jede Dimension richtig sein
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
    Base.fill!(h,zero(eltype(h))); Base.fill!(y,zero(eltype(y)))
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

function convexify!(buffer::HROCBuffer,ctr::Int)
    return convexify!(buffer.initial.values,buffer.initial.grid,ctr,buffer.convex.values,buffer.convex.grid)
end

"""
    AbstractPolishOptimizer
Supertype of the optimizers usable in the polish stage of [`HROC`](@ref):
[`CompassSearch`](@ref), [`BFGS`](@ref) and [`Adam`](@ref).
Each optimizer holds its own hyperparameters; the actual minimization dispatches on the type
via `polish_minimize!(optimizer, p, offsets, F, admissible, W, xargs...)`.
"""
abstract type AbstractPolishOptimizer end

"""
    AnalyticGradient()
Gradient backend for [`BFGS`](@ref): assembles the exact gradient in one reverse sweep over
the tree from the first Piola-Kirchhoff stresses at the leaves via [`polish_gradient!`](@ref)
(cost independent of the number of parameters).
"""
struct AnalyticGradient end

"""
    ADGradient()
Gradient backend for [`BFGS`](@ref): forward-mode automatic differentiation over the parameter
vector (ForwardDiff). Mainly useful for verification; cost grows with the parameter count.
"""
struct ADGradient end

@doc raw"""
    CompassSearch(; steptol=1e-6, maxsweeps=10_000)
Zero-order pattern search on the tree parameters. Derivative free and therefore robust for
non-smooth `W`; linear convergence, the achievable accuracy is set by `steptol`.
Monotone: never returns a value above the initial energy.
"""
Base.@kwdef struct CompassSearch <: AbstractPolishOptimizer
    steptol::Float64 = 1e-6
    maxsweeps::Int = 10_000
end

@doc raw"""
    BFGS(; gradient=AnalyticGradient(), gtol=1e-10, maxiter=200, stalliter=3, c_armijo=1e-4, ls_maxiter=50, fallback=CompassSearch())
Dense BFGS with Armijo backtracking on the tree parameters. Infeasible trial points (negative
offsets, out of bounds, constraint violations) evaluate to `Inf` and are rejected by the
backtracking, so no explicit constraint handling is needed. Terminates on the gradient norm
(`gtol`), on `stalliter` consecutive stagnating energies (relevant for non-smooth `W`, where
the gradient does not vanish at the optimum) or after `maxiter` iterations.
If no progress is made from the initial point and a `fallback` optimizer is set, the fallback
is run instead. Monotone: never returns a value above the initial energy.
"""
Base.@kwdef struct BFGS{G<:Union{AnalyticGradient,ADGradient}} <: AbstractPolishOptimizer
    gradient::G = AnalyticGradient()
    gtol::Float64 = 1e-10
    maxiter::Int = 200
    stalliter::Int = 3
    c_armijo::Float64 = 1e-4
    ls_maxiter::Int = 50
    fallback::Union{Nothing,CompassSearch} = CompassSearch()
end

@doc raw"""
    Adam(; α=0.02, β1=0.9, β2=0.999, ϵ=1e-8, maxiter=500, restarts=4, σ_angle=0.3, σ_offset=0.2, seed=0x9e3779b97f4a7c15, finish=BFGS())
Adam (adaptive moment) descent with randomized restarts: besides the unperturbed discrete tree,
`restarts` perturbed copies of the initial parameters are optimized (direction angles perturbed
by up to `±σ_angle` radians, offsets relatively by up to `±σ_offset`), and the best result over
all runs and iterations is kept. The perturbations explore neighbouring laminate basins that
the deterministic descent from the greedy tree cannot reach; the deterministic run guarantees
the result is never worse than the initial tree. Steps leading to infeasible parameters are
rejected with a halved learning rate. If `finish` is set, the best parameters are refined by
that optimizer afterwards (Adam explores, BFGS sharpens).
The perturbations are drawn from an internal xorshift generator seeded by `seed`, so results
are deterministic and reproducible without a dependency on `Random`.
"""
Base.@kwdef struct Adam <: AbstractPolishOptimizer
    α::Float64 = 0.02
    β1::Float64 = 0.9
    β2::Float64 = 0.999
    ϵ::Float64 = 1e-8
    maxiter::Int = 500
    restarts::Int = 4
    σ_angle::Float64 = 0.3
    σ_offset::Float64 = 0.2
    seed::UInt64 = 0x9e3779b97f4a7c15
    finish::Union{Nothing,BFGS} = BFGS()
end

# minimal deterministic xorshift64* generator for the Adam perturbations (avoids a Random dep)
@inline function _xorshift(state::UInt64)
    state ⊻= state << 13
    state ⊻= state >> 7
    state ⊻= state << 17
    return state
end
# returns (new state, uniform number in [-1,1])
@inline function _unitrand(state::UInt64)
    state = _xorshift(state)
    return state, 2.0*(state/typemax(UInt64)) - 1.0
end
# returns (new state, standard normal number) via Box-Muller
@inline function _normrand(state::UInt64)
    state, u1 = _unitrand(state)
    state, u2 = _unitrand(state)
    r = sqrt(-2*log(clamp(0.5*(u1 + 1), 1e-16, 1.0)))
    return state, r*cospi(u2)
end

@doc raw"""
    NonLocalNewton(; σ0=1.0, k=0, maxiter=100, σ_restart=1e-4, steptol=1e-4, ls_base=6/5, ls_range=10, seed=0x2545f4914f6cdd1d, finish=BFGS())
Non-local quasi-Newton method of Müller,
[*A Principle for Global Optimization with Gradients*](https://doi.org/10.1007/s10957-025-02848-5)
(JOTA 2025, arXiv:2308.09556), as a polish optimizer. Instead of a local quadratic Taylor model,
each iteration draws `k` Gaussian gradient samples `∇E(p + σₜ zⱼ)` (default `k = 3n`, the paper's
choice) and fits a *non-local* quadratic model to them by symmetric least squares — the normal
equations are the Lyapunov-type equation `M ẐẐᵀ + ẐẐᵀ M = ĜẐᵀ + ẐĜᵀ` of Corollary 2.1, solved
by eigendecomposition. The search direction is the Newton step of that model (regularized when
the model Hessian is indefinite, cf. Remark 1); the line search follows the paper and evaluates
`p + (6/5)ⁱ Δp` and `p + (6/5)ⁱ (−b)` for `i ∈ -ls_range:ls_range`, the scaling σₜ is halved on
small steps, set to half the step length on large steps and restarted at `σ0` below `σ_restart`.
Because the fitted model averages gradients over a σ-sized neighbourhood, the direction can point
across energy barriers that trap strictly local descent — the iteration itself is non-monotone,
but the best parameters over all iterations are tracked and returned (and refined by `finish` if
set), so the polished result is never worse than the discrete tree.
Gradient samples use the smooth extension of the laminate energy (no feasibility penalties);
the line search uses the true penalized energy, so accepted iterates remain admissible.
"""
Base.@kwdef struct NonLocalNewton <: AbstractPolishOptimizer
    σ0::Float64 = 1.0
    k::Int = 0 # 0 → 3n samples per iteration, following the paper's experiments
    maxiter::Int = 100
    σ_restart::Float64 = 1e-4
    steptol::Float64 = 1e-4
    ls_base::Float64 = 6/5
    ls_range::Int = 10
    seed::UInt64 = 0x2545f4914f6cdd1d
    finish::Union{Nothing,BFGS} = BFGS()
end

@doc raw"""
    HROC{dimp,R1Dir<:RankOneDirections{dimp},T} <: AbstractConvexification
Holds the specification for performing rank-one convexification by the upper bound described in [this paper](https://arxiv.org/abs/2405.16866).
Useable by constructing an instance of this type, as well as a buffer by `build_buffer` and calling `convexify` as usual.

# Constructors
    HROC(maxlevel::Int,n_convexpoints::Int,dir::R1Dir,GLcheck::Bool,start::Tensor{2,dimp,T,dimc},stop::Tensor{2,dimp,T,dimc},polish=nothing)
    HROC(start::Tensor{2,dimp},stop::Tensor{2,dimp};maxlevel=10,l=1,dirs=ParametrizedR1Directions(dimp;l=l),GLcheck=true,n_convexpoints=1000,polish=nothing)

# Fields
- `maxlevel::Int`
- `n_convexpoints::Int`
- `dirs::R1Dir`
- `GLcheck::Bool`
- `startF::Vector{T}`
- `endF::Vector{T}`
- `polish::Union{Nothing,AbstractPolishOptimizer}` if set, the discrete lamination tree is
  post-processed by a joint continuous optimization of all tree parameters (lamination direction
  angles and endpoint offsets per internal node) with the given optimizer, which removes the
  direction-set and line-grid discretization errors of the greedy stage. Available optimizers:
  [`BFGS`](@ref) (default when `polish=true` is passed), [`CompassSearch`](@ref) (zero order,
  robust for non-smooth `W`) and [`Adam`](@ref) (randomized restarts around the discrete tree
  to explore neighbouring laminate basins). The optimizer structs hold their hyperparameters,
  e.g. `HROC(start, stop; polish=BFGS(gradient=ADGradient(), gtol=1e-12))`.

"""
struct HROC{dimp,R1Dir<:RankOneDirections{dimp},T,O<:Union{Nothing,AbstractPolishOptimizer}} <: AbstractConvexification
    maxlevel::Int
    n_convexpoints::Int
    dirs::R1Dir
    GLcheck::Bool
    startF::Vector{T}
    endF::Vector{T}
    polish::O
end

# `polish=true/false` convenience sugar → default BFGS / no polishing
_polish_optimizer(polish::Bool) = polish ? BFGS() : nothing
_polish_optimizer(polish::Union{Nothing,AbstractPolishOptimizer}) = polish

function HROC(maxlevel::Int,n_convexpoints::Int,dir::R1Dir,GLcheck::Bool,startF::Vector{T},endF::Vector{T},polish=nothing) where {dimp,R1Dir<:RankOneDirections{dimp},T}
    HROC(maxlevel,n_convexpoints,dir,GLcheck,startF,endF,_polish_optimizer(polish))
end

function HROC(maxlevel::Int,n_convexpoints::Int,dir::R1Dir,GLcheck::Bool,start::Tensor{2,dimp,T,dimc},stop::Tensor{2,dimp,T,dimc},polish=nothing) where {dimp,R1Dir<:RankOneDirections{dimp},T,dimc}
    HROC(maxlevel,n_convexpoints,dir,GLcheck,collect(start.data),collect(stop.data),_polish_optimizer(polish))
end

HROC(start::Tensor{2,dimp},stop::Tensor{2,dimp};maxlevel=10,l=1,dirs=ParametrizedR1Directions(dimp;l=l),GLcheck=true,n_convexpoints=1000,polish=nothing) where {dimp} = HROC(maxlevel,n_convexpoints,dirs,GLcheck,start,stop,_polish_optimizer(polish))

function build_buffer(convexification::HROC{dimp,R1Dir,T}) where {dimp,R1Dir <: RankOneDirections{dimp}, T}
    F = zeros(Int,convexification.n_convexpoints+2)
    W = zeros(T,convexification.n_convexpoints+2)
    buffer = ConvexificationBuffer1D(F,W)
    return HROCBuffer(buffer,deepcopy(buffer),deepcopy(buffer),deepcopy(buffer),deepcopy(buffer),deepcopy(buffer))
end

function δ(convexification::HROC{dimp,R1Dir,T1},A::Tensor{2,dimp,T2,dimc}) where {dimp,R1Dir<:RankOneDirections{dimp},T1,T2,dimc}
    startF = convexification.startF
    endF = convexification.endF
    newvals = ntuple(i->A[i] != 0 ? (endF[i] - startF[i])/convexification.n_convexpoints : Inf,dimc)
    return Tensor{2,dimp,T1,dimc}(newvals)
end

function inbounds(𝐱::Tensor{2,dimp,T,dimc}, convexification::HROC) where {dimp,T,dimc}
    return all(ntuple(i->convexification.startF[i] ≤ 𝐱[i] ≤ convexification.endF[i],dimc))
end

rotation_matrix(θ) = Tensor{2,2}((cos(θ), sin(θ), -sin(θ), cos(θ)))
function isorthogonal(laminate::Laminate, 𝐀::Tensor{2,2})
    orthogonal = false
    for θ in (π/2, -π/2)
        Q = rotation_matrix(θ)
        if isapprox(Q ⋅ laminate.A ⋅ Q', 𝐀, atol=1e-10)
            orthogonal = true
            break
        end
    end
    return orthogonal
end
isorthogonal(laminate::Nothing, 𝐀::Tensor{2}) = false

struct BinaryLaminationTreeNode{dim,T,N}
    F::Union{T,Tensor{2,dim,T,N}}
    W::T
    ξ::T
    level::Int

    function BinaryLaminationTreeNode(F::Tensor{order,dimp,T,N},W::T,ξ::T,l::Int) where {order,dimp,T,N}
        return new{dimp,T,N}(F,W,ξ,l)
    end

    function BinaryLaminationTreeNode(F::T,W::T,ξ::T,l::Int) where {T<:Number}
        return new{1,T,1}(F,W,ξ,l)
    end
end

struct BinaryLaminationTree{dim,T,N}
    nodes::Vector{BinaryLaminationTreeNode{dim,T,N}}
    active::BitVector
end

# Index convention for implicit binary tree
minus_idx(i::Int) = 2i
plus_idx(i::Int) = 2i + 1
parent_idx(i::Int) = i ÷ 2

function haschildren(bt::BinaryLaminationTree, i::Int)
    mi = minus_idx(i)
    return mi ≤ length(bt.active) && bt.active[mi]
end
isleaf(bt::BinaryLaminationTree, i::Int) = !haschildren(bt, i)

function _make_tree(::Type{Tensor{2,dim,T,N}}, maxlevel) where {dim,T,N}
    maxnodes = 3
    nodes = Vector{BinaryLaminationTreeNode{dim,T,N}}(undef, maxnodes)
    active = falses(maxnodes)
    return BinaryLaminationTree{dim,T,N}(nodes, active)
end

function _setnode!(bt::BinaryLaminationTree, i::Int, node::BinaryLaminationTreeNode)
    if i > length(bt.nodes)
        oldlen = length(bt.nodes)
        newlen = max(2 * oldlen, i)
        resize!(bt.nodes, newlen)
        resize!(bt.active, newlen)
        bt.active[(oldlen+1):newlen] .= false
    end
    bt.nodes[i] = node
    bt.active[i] = true
end

function BinaryLaminationTree(F::Tensor{2,dim,T,N},W::T,ξ::T,l::Int) where {dim,T,N}
    bt = _make_tree(Tensor{2,dim,T,N}, l)
    _setnode!(bt, 1, BinaryLaminationTreeNode(F, W, ξ, l))
    return bt
end

BinaryLaminationTree(cs::HROC{dimp}) where dimp = BinaryLaminationTree(one(Tensor{2,dimp}),0.0,0.0,cs.maxlevel)

function BinaryLaminationTree(convexification::HROC, buffer::HROCBuffer, W::FUN, F::Tensor{2,dim,T,N}, xargs::Vararg{Any,XN}) where {dim,T,N,FUN,XN}
    level = convexification.maxlevel
    bt = _make_tree(Tensor{2,dim,T,N}, level)
    _setnode!(bt, 1, BinaryLaminationTreeNode(F, zero(T), one(T), level + 1))
    laminate = hrockernel(bt,convexification,buffer,W,F,xargs...)
    if laminate === nothing
        return bt
    end
    queue = [(1, laminate)] # (parent_idx, laminate_candidate)

    while !isempty(queue)
        pidx, lc = pop!(queue)
        parent_F = bt.nodes[pidx].F
        ξ = norm(parent_F - lc.F⁻) / norm(lc.F⁺ - lc.F⁻)
        if isapprox(ξ,1.0,atol=1e-10) || isapprox(ξ,0.0,atol=1e-10)
            continue
        end
        level = bt.nodes[pidx].level - 1 # NOTE: must be computed before _setnode!, otherwise a stale level doubles the effective maxlevel
        _setnode!(bt, minus_idx(pidx), BinaryLaminationTreeNode(lc.F⁻, lc.W⁻, (1.0 - ξ), level))
        _setnode!(bt, plus_idx(pidx), BinaryLaminationTreeNode(lc.F⁺, lc.W⁺, ξ, level))
        if level > 0
            laminate⁺ = hrockernel(bt,convexification,buffer,W,lc.F⁺,xargs...)
            laminate⁻ = hrockernel(bt,convexification,buffer,W,lc.F⁻,xargs...)
            !(laminate⁺ === nothing) && push!(queue,(plus_idx(pidx), laminate⁺))
            !(laminate⁻ === nothing) && push!(queue,(minus_idx(pidx), laminate⁻))
        end
    end
    convexification.polish === nothing || polish!(bt, convexification, buffer, _admissible(convexification), W, F, xargs...)
    return bt
end

function BinaryLaminationTree(prev_F,prev_bt::BinaryLaminationTree, convexification::HROC, buffer::HROCBuffer, constraint::CON1, irr::CON2, W::FUN, F::Tensor{2,dim,T,N}, xargs::Vararg{Any,XN}) where {dim,T,N,FUN,CON1,CON2,XN}
    level = convexification.maxlevel
    bt = _make_tree(Tensor{2,dim,T,N}, level)
    _setnode!(bt, 1, BinaryLaminationTreeNode(F, zero(T), one(T), level + 1))
    diss_offset = 0.0
    # CE at root level only: prefer the previous root direction if it still gives energy reduction.
    root_prev_direction = haschildren(prev_F, 1) ? rankonedir(prev_F, 1) : zero(typeof(F))
    laminate = hrockernel(root_prev_direction,prev_F,prev_bt,1,bt,convexification,buffer,constraint,diss_offset,W,F,xargs...)
    if laminate === nothing
        return bt
    end
    queue = [(1, laminate, 1)] # (parent_idx, laminate_candidate, prev_parent_idx)

    while !isempty(queue)
        pidx, lc, prev_pidx = pop!(queue)
        parent_F = bt.nodes[pidx].F
        ξ = norm(parent_F - lc.F⁻) / norm(lc.F⁺ - lc.F⁻)
        if isapprox(ξ,1.0,atol=1e-10) || isapprox(ξ,0.0,atol=1e-10)
            continue
        end
        level = bt.nodes[pidx].level - 1 # NOTE: must be computed before _setnode!, otherwise a stale level doubles the effective maxlevel
        _setnode!(bt, minus_idx(pidx), BinaryLaminationTreeNode(lc.F⁻, lc.W⁻, (1.0 - ξ), level))
        _setnode!(bt, plus_idx(pidx), BinaryLaminationTreeNode(lc.F⁺, lc.W⁺, ξ, level))
        if level > 0
            prev_plus_idx = haschildren(prev_bt, prev_pidx) ? plus_idx(prev_pidx) : prev_pidx
            prev_minus_idx = haschildren(prev_bt, prev_pidx) ? minus_idx(prev_pidx) : prev_pidx
            # Continuity enforcement disabled — global search at child level.
            # prev_direction⁺ = haschildren(prev_F, prev_plus_idx) ? rankonedir(prev_F, prev_plus_idx) : rankonedir(bt, pidx)
            # prev_direction⁻ = haschildren(prev_F, prev_minus_idx) ? rankonedir(prev_F, prev_minus_idx) : rankonedir(bt, pidx)
            laminate⁺ = hrockernel(zero(F),prev_F,prev_bt,prev_plus_idx,bt,convexification,buffer,constraint,diss_offset,W,lc.F⁺,xargs...)
            laminate⁻ = hrockernel(zero(F),prev_F,prev_bt,prev_minus_idx,bt,convexification,buffer,constraint,diss_offset,W,lc.F⁻,xargs...)
            !irr(constraint,NodeView(prev_bt,prev_pidx),lc.F⁺,xargs...) && !(laminate⁺ === nothing) && push!(queue,(plus_idx(pidx), laminate⁺, prev_plus_idx))
            !irr(constraint,NodeView(prev_bt,prev_pidx),lc.F⁻,xargs...) && !(laminate⁻ === nothing) && push!(queue,(minus_idx(pidx), laminate⁻, prev_minus_idx))
        end
    end
    if convexification.polish !== nothing
        # NOTE: with labeled densities/constraints the polish uses the root label as a fallback
        admissible = 𝐱 -> _admissible(convexification)(𝐱) && _eval_constraint(constraint, prev_F, prev_bt, 1, 𝐱, xargs...)
        Wp = W isa LabeledDensity ? (𝐱, args...) -> W.f(𝐱, NodeView(prev_bt, 1), args...) : W
        polish!(bt, convexification, buffer, admissible, Wp, F, xargs...)
    end
    return bt
end

function rankonedir(bt::BinaryLaminationTree{dim}, idx::Int) where dim
    start_𝐀 = bt.nodes[plus_idx(idx)].F - bt.nodes[minus_idx(idx)].F
    start_𝐀 /= minimum(x->isapprox(abs(x),0,atol=1e-10) ? Inf : x, start_𝐀) #normalize direction and filter out zeros
    start_𝐀 = Tensor{2,dim}((i,j)->round(start_𝐀[i,j]))
end

function rankonedir(laminate::Laminate{dim}) where dim
    start_𝐀 = laminate.F⁺ - laminate.F⁻
    start_𝐀 /= minimum(x->isapprox(abs(x),0,atol=1e-10) ? Inf : x, start_𝐀) #normalize direction and filter out zeros
    start_𝐀 = Tensor{2,dim}((i,j)->round(start_𝐀[i,j]))
end

####################################################
################  Tree polishing  ##################
####################################################
# Post-processing of the greedy HROC tree: with fixed tree topology, all lamination
# directions (parametrized by angles, rank-one by construction) and endpoint offsets
# are optimized jointly. This lifts the two accuracy limits of the discrete stage:
# the finite direction set and the sequential (greedy) per-level minimization of the
# unrelaxed W along lines.

@inline _unitvec(::Val{2}, θ::NTuple{1}) = Vec{2}((cos(θ[1]), sin(θ[1])))
@inline _unitvec(::Val{3}, θ::NTuple{2}) = Vec{3}((sin(θ[1])*cos(θ[2]), sin(θ[1])*sin(θ[2]), cos(θ[1])))
_angles(a::Vec{2}) = (atan(a[2], a[1]),)
_angles(a::Vec{3}) = (acos(clamp(a[3], -1.0, 1.0)), atan(a[2], a[1]))
_nangles(::Val{2}) = 1
_nangles(::Val{3}) = 2
# parameters per internal node: [angles(𝐚)..., angles(𝐛)..., s⁻, s⁺]
_nparams(::Val{dim}) where dim = 2*_nangles(Val(dim)) + 2

_heapdepth(i::Int) = floor(Int, log2(i)) + 1

"""
    polish_parameters(bt::BinaryLaminationTree{dim,T}; maxdepth=typemax(Int)) -> p::Vector{T}, offsets::Dict{Int,Int}
Extracts the continuous parametrization of the (fixed-topology) lamination tree:
for every internal node the rank-one direction angles of `𝐚 ⊗ 𝐛` and the two offsets
`s⁻,s⁺` such that `F⁻ = F - s⁻ 𝐚⊗𝐛` and `F⁺ = F + s⁺ 𝐚⊗𝐛`.
`offsets` maps the heap index of an internal node to its position in `p`.
Internal nodes deeper than `maxdepth` are truncated (treated as leaves): the greedy stage
compensates its per-level suboptimality with long refinement chains whose depth grows with the
line-grid resolution, and carrying that redundant, degenerate parametrization into the joint
optimization deteriorates convergence — a shallow tree with jointly optimal parameters
represents the same (or a better) laminate.
"""
function polish_parameters(bt::BinaryLaminationTree{dim,T}; maxdepth::Int=typemax(Int)) where {dim,T}
    offsets = Dict{Int,Int}()
    p = T[]
    for i in 1:length(bt.active)
        (bt.active[i] && haschildren(bt, i) && _heapdepth(i) ≤ maxdepth) || continue
        F0 = bt.nodes[i].F
        F⁻ = bt.nodes[minus_idx(i)].F
        F⁺ = bt.nodes[plus_idx(i)].F
        D = F⁺ - F⁻
        decomp = svd(Array(D))
        𝐚 = Vec{dim,T}(NTuple{dim,T}(decomp.U[:,1]))
        𝐛 = Vec{dim,T}(NTuple{dim,T}(decomp.V[:,1]))
        ((𝐚 ⊗ 𝐛) ⊡ D) < 0 && (𝐛 = -𝐛)
        offsets[i] = length(p) + 1
        push!(p, _angles(𝐚)...)
        push!(p, _angles(𝐛)...)
        push!(p, norm(F0 - F⁻), norm(F⁺ - F0))
    end
    return p, offsets
end

"""
    polish_energy(p, offsets, i, F, admissible, W, xargs...)
Evaluates the laminate energy of the parametrized tree rooted at heap index `i` and point `F`:
the ξ-weighted sum of `W` over the tree leaves. Inadmissible evaluation points (out of bounds,
`GLcheck` violation, constraint violation) and negative offsets return `Inf`.
The parameter eltype `S` is decoupled from the eltype of `F`, so `p` may carry
`ForwardDiff.Dual` numbers — this makes the energy differentiable by automatic differentiation.
"""
function polish_energy(p::AbstractVector{S}, offsets::Dict{Int,Int}, i::Int, F::Tensor{2,dim}, admissible::AFUN, W::FUN, xargs::Vararg{Any,XN}) where {S,dim,AFUN,FUN,XN}
    admissible(F) || return S(Inf)
    o = get(offsets, i, 0)
    o == 0 && return S(W(F, xargs...))
    na = _nangles(Val(dim))
    θ𝐚 = ntuple(k->p[o+k-1], na)
    θ𝐛 = ntuple(k->p[o+na+k-1], na)
    s⁻ = p[o+2na]; s⁺ = p[o+2na+1]
    (s⁻ < 0 || s⁺ < 0) && return S(Inf)
    s⁻ + s⁺ < 1e-12 && return S(W(F, xargs...))
    𝐑 = _unitvec(Val(dim), θ𝐚) ⊗ _unitvec(Val(dim), θ𝐛)
    ξ = s⁻/(s⁻ + s⁺) # weight of the plus branch, consistent with the tree convention
    W⁻ = polish_energy(p, offsets, minus_idx(i), F - s⁻*𝐑, admissible, W, xargs...)
    W⁺ = polish_energy(p, offsets, plus_idx(i), F + s⁺*𝐑, admissible, W, xargs...)
    return ξ*W⁺ + (1-ξ)*W⁻
end

"""
    polish_minimize!(opt::AbstractPolishOptimizer, p, offsets, F, admissible, W, xargs...) -> f
Minimizes [`polish_energy`](@ref) over the tree parameters `p` (mutated in place) with the
optimizer `opt`, which holds all hyperparameters. Every method is monotone: the returned
energy is never above the initial one.
"""
function polish_minimize!(opt::CompassSearch, p::Vector{T}, offsets::Dict{Int,Int}, F::Tensor{2,dim,T}, admissible::AFUN, W::FUN, xargs::Vararg{Any,XN}) where {dim,T,AFUN,FUN,XN}
    E = q -> polish_energy(q, offsets, 1, F, admissible, W, xargs...)
    return _compass_core!(E, p, offsets, _nangles(Val(dim)), opt.steptol, opt.maxsweeps)
end

function _compass_core!(E::EFUN, p::Vector{T}, offsets::Dict{Int,Int}, na::Int, steptol, maxsweeps) where {EFUN,T}
    isempty(p) && return E(p) # trivial tree: nothing to optimize
    steps = similar(p)
    for o in values(offsets)
        for k in 0:2na-1 # angles
            steps[o+k] = T(0.1)
        end
        steps[o+2na] = max(T(0.1)*abs(p[o+2na]), T(0.05))   # s⁻
        steps[o+2na+1] = max(T(0.1)*abs(p[o+2na+1]), T(0.05)) # s⁺
    end
    f = E(p)
    sweep = 0
    while maximum(steps) > steptol && sweep < maxsweeps
        improved = false
        for j in eachindex(p), s in (steps[j], -steps[j])
            pⱼ = p[j]
            p[j] = pⱼ + s
            f_trial = E(p)
            if f_trial < f
                f = f_trial
                improved = true
            else
                p[j] = pⱼ
            end
        end
        improved || (steps .*= T(0.5))
        sweep += 1
    end
    return f
end

# derivatives of the unit-vector parametrization w.r.t. its angles
@inline _dunitvec(::Val{2}, θ::NTuple{1}) = (Vec{2}((-sin(θ[1]), cos(θ[1]))),)
@inline _dunitvec(::Val{3}, θ::NTuple{2}) = (Vec{3}(( cos(θ[1])*cos(θ[2]),  cos(θ[1])*sin(θ[2]), -sin(θ[1]))),
                                             Vec{3}((-sin(θ[1])*sin(θ[2]),  sin(θ[1])*cos(θ[2]),  zero(θ[1]))))

"""
    polish_gradient!(g, p, offsets, F, W, xargs...) -> g
Analytic gradient of [`polish_energy`](@ref) w.r.t. the tree parameters `p`, assembled in one
reverse sweep over the tree from the first Piola-Kirchhoff stresses `P = ∂W/∂F` at the leaves
(one `Tensors.gradient` call per leaf — the cost is independent of the number of parameters).
The offset components are the Weierstrass-Erdmann/traction-balance residuals of the laminate;
the angle components are the stress jumps contracted with `s ∂(𝐚 ⊗ 𝐛)/∂θ`.
Assumes the current point is admissible (interior); the penalty branches of the energy are
handled by rejection in the line search, not here.
"""
function polish_gradient!(g::Vector{T}, p::Vector{T}, offsets::Dict{Int,Int}, F::Tensor{2,dim,T}, W::FUN, xargs::Vararg{Any,XN}) where {dim,T,FUN,XN}
    Base.fill!(g, zero(T))
    _gradsweep!(g, p, offsets, 1, F, one(T), W, xargs...)
    return g
end

function _gradsweep!(g::Vector{T}, p::Vector{T}, offsets::Dict{Int,Int}, i::Int, F::Tensor{2,dim,T}, w_anc::T, W::FUN, xargs::Vararg{Any,XN}) where {dim,T,FUN,XN}
    o = get(offsets, i, 0)
    na = _nangles(Val(dim))
    if o == 0 || p[o+2na] + p[o+2na+1] < 1e-12 # leaf (or degenerate split): weighted energy and stress
        𝐏, Wval = Tensors.gradient(y -> W(y, xargs...), F, :all)
        return Wval, 𝐏
    end
    θ𝐚 = ntuple(k->p[o+k-1], na)
    θ𝐛 = ntuple(k->p[o+na+k-1], na)
    s⁻ = p[o+2na]; s⁺ = p[o+2na+1]
    𝐚 = _unitvec(Val(dim), θ𝐚); 𝐛 = _unitvec(Val(dim), θ𝐛)
    𝐑 = 𝐚 ⊗ 𝐛
    ξ = s⁻/(s⁻ + s⁺) # weight of the plus branch
    S⁻, T⁻ = _gradsweep!(g, p, offsets, minus_idx(i), F - s⁻*𝐑, w_anc*(1-ξ), W, xargs...)
    S⁺, T⁺ = _gradsweep!(g, p, offsets, plus_idx(i), F + s⁺*𝐑, w_anc*ξ, W, xargs...)
    ∂ξ∂s⁻ = s⁺/(s⁻ + s⁺)^2
    ∂ξ∂s⁺ = -s⁻/(s⁻ + s⁺)^2
    # local subtree energy S = ξ S⁺ + (1-ξ) S⁻; T± are the locally-weighted subtree stresses ∂S±/∂F±
    𝐓θ = -s⁻*(1-ξ)*T⁻ + s⁺*ξ*T⁺
    d𝐚 = _dunitvec(Val(dim), θ𝐚); d𝐛 = _dunitvec(Val(dim), θ𝐛)
    for k in 1:na
        g[o+k-1]    += w_anc * (𝐓θ ⊡ (d𝐚[k] ⊗ 𝐛))
        g[o+na+k-1] += w_anc * (𝐓θ ⊡ (𝐚 ⊗ d𝐛[k]))
    end
    g[o+2na]   += w_anc * (∂ξ∂s⁻*(S⁺ - S⁻) - (1-ξ)*(T⁻ ⊡ 𝐑))
    g[o+2na+1] += w_anc * (∂ξ∂s⁺*(S⁺ - S⁻) + ξ*(T⁺ ⊡ 𝐑))
    return ξ*S⁺ + (1-ξ)*S⁻, ξ*T⁺ + (1-ξ)*T⁻
end

function _make_gradient(::AnalyticGradient, E::EFUN, p::Vector, offsets::Dict{Int,Int}, F::Tensor{2}, W::FUN, xargs::Vararg{Any,XN}) where {EFUN,FUN,XN}
    return (g, q) -> polish_gradient!(g, q, offsets, F, W, xargs...)
end

function _make_gradient(::ADGradient, E::EFUN, p::Vector, offsets::Dict{Int,Int}, F::Tensor{2}, W::FUN, xargs::Vararg{Any,XN}) where {EFUN,FUN,XN}
    cfg = ForwardDiff.GradientConfig(E, p)
    return (g, q) -> ForwardDiff.gradient!(g, E, q, cfg)
end

function polish_minimize!(opt::BFGS, p::Vector{T}, offsets::Dict{Int,Int}, F::Tensor{2,dim,T}, admissible::AFUN, W::FUN, xargs::Vararg{Any,XN}) where {dim,T,AFUN,FUN,XN}
    E = q -> polish_energy(q, offsets, 1, F, admissible, W, xargs...)
    grad! = _make_gradient(opt.gradient, E, p, offsets, F, W, xargs...)
    f0 = E(p)
    isfinite(f0) || return f0
    f = _bfgs_core!(E, grad!, p, opt)
    if !(f < f0) && opt.fallback !== nothing
        # no progress from the initial point (e.g. seeded on a kink of a non-smooth W)
        f = polish_minimize!(opt.fallback, p, offsets, F, admissible, W, xargs...)
    end
    return f
end

function _bfgs_core!(E::EFUN, grad!::GFUN, p::Vector{T}, opt::BFGS) where {EFUN,GFUN,T}
    isempty(p) && return E(p) # trivial tree: nothing to optimize
    n = length(p)
    H = Matrix{T}(T(0.01)*I, n, n)
    g = zeros(T, n); g_new = zeros(T, n)
    f0 = E(p)
    isfinite(f0) || return f0
    f = f0
    grad!(g, p)
    stall = 0
    for _ in 1:opt.maxiter
        maximum(abs, g) < opt.gtol && break
        stall ≥ opt.stalliter && break # energy stagnated: converged or non-smooth kink
        d = -H*g
        if dot(d, g) ≥ 0 # not a descent direction: reset curvature
            H .= Matrix{T}(T(0.01)*I, n, n)
            d = -H*g
        end
        α = one(T); f_new = T(Inf); accepted = false
        for _ in 1:opt.ls_maxiter
            f_new = E(p .+ α.*d)
            if isfinite(f_new) && f_new ≤ f + T(opt.c_armijo)*α*dot(g, d)
                accepted = true
                break
            end
            α /= 2
        end
        accepted || break # line search failed (kink or stationary point)
        p_new = p .+ α.*d
        grad!(g_new, p_new)
        s = p_new .- p; y = g_new .- g
        sy = dot(s, y)
        if sy > 1e-12*norm(s)*norm(y) # curvature condition, else skip update
            ρ = 1/sy
            H .= (I - ρ*s*y')*H*(I - ρ*y*s') .+ ρ*s*s'
        end
        stall = (f - f_new) < 1e-14*max(one(T), abs(f)) ? stall + 1 : 0
        copyto!(p, p_new); f = f_new; copyto!(g, g_new)
    end
    return f
end

function polish_minimize!(opt::Adam, p::Vector{T}, offsets::Dict{Int,Int}, F::Tensor{2,dim,T}, admissible::AFUN, W::FUN, xargs::Vararg{Any,XN}) where {dim,T,AFUN,FUN,XN}
    E = q -> polish_energy(q, offsets, 1, F, admissible, W, xargs...)
    na = _nangles(Val(dim))
    n = length(p)
    g = zeros(T, n); m = zeros(T, n); v = zeros(T, n)
    best_p = copy(p)
    best_f = E(p)
    state = opt.seed
    for r in 0:opt.restarts
        q = copy(p)
        if r > 0 # perturb the unpolished tree to explore neighbouring laminate basins
            for o in values(offsets)
                for k in 0:2na-1 # direction angles
                    state, u = _unitrand(state)
                    q[o+k] += opt.σ_angle * u
                end
                for k in (2na, 2na+1) # offsets, kept non-negative
                    state, u = _unitrand(state)
                    q[o+k] = max(q[o+k] * (1 + opt.σ_offset * u), zero(T))
                end
            end
            isfinite(E(q)) || continue # perturbation left the admissible set: skip this restart
        end
        Base.fill!(m, zero(T)); Base.fill!(v, zero(T))
        α = opt.α
        f = E(q)
        for t in 1:opt.maxiter
            polish_gradient!(g, q, offsets, F, W, xargs...)
            all(isfinite, g) || break
            m .= opt.β1 .* m .+ (1 - opt.β1) .* g
            v .= opt.β2 .* v .+ (1 - opt.β2) .* g.^2
            m̂ = m ./ (1 - opt.β1^t)
            v̂ = v ./ (1 - opt.β2^t)
            q_trial = q .- α .* m̂ ./ (sqrt.(v̂) .+ opt.ϵ)
            f_trial = E(q_trial)
            if isfinite(f_trial)
                q = q_trial; f = f_trial
                if f < best_f
                    best_f = f
                    copyto!(best_p, q)
                end
            else
                α /= 2 # step left the admissible set: reject and damp
            end
        end
    end
    copyto!(p, best_p)
    if opt.finish !== nothing # Adam explores, the finisher sharpens
        best_f = polish_minimize!(opt.finish, p, offsets, F, admissible, W, xargs...)
    end
    return best_f
end

function polish_minimize!(opt::NonLocalNewton, p::Vector{T}, offsets::Dict{Int,Int}, F::Tensor{2,dim,T}, admissible::AFUN, W::FUN, xargs::Vararg{Any,XN}) where {dim,T,AFUN,FUN,XN}
    E = q -> polish_energy(q, offsets, 1, F, admissible, W, xargs...)
    n = length(p)
    k = opt.k > 0 ? opt.k : 3n
    Z = zeros(T, n, k); G = zeros(T, n, k)
    g = zeros(T, n)
    x = copy(p)
    best_p = copy(p)
    best_f = E(p)
    σ = opt.σ0
    state = opt.seed
    for _ in 1:opt.maxiter
        # sample a σ-neighbourhood of x and collect gradients of the smooth extension
        nvalid = 0
        for j in 1:k
            for i in 1:n
                state, zij = _normrand(state)
                Z[i, j] = zij
            end
            q = x .+ σ .* @view(Z[:, j])
            ok = try
                polish_gradient!(g, q, offsets, F, W, xargs...)
                all(isfinite, g)
            catch
                false
            end
            ok || (Z[:, j] .= zero(T); G[:, j] .= zero(T); continue)
            G[:, j] .= g
            nvalid += 1
        end
        nvalid ≥ 2 || break # neighbourhood not evaluable: give up on the non-local model
        # symmetric least-squares fit of the non-local quadratic model (Corollary 2.1):
        # M ẐẐᵀ + ẐẐᵀ M = ĜẐᵀ + ẐĜᵀ, b = ḡ - M z̄, solved via eigendecomposition
        z̄ = sum(Z, dims=2) ./ k
        ḡ = sum(G, dims=2) ./ k
        Ẑ = Z .- z̄; Ĝ = G .- ḡ
        P = Symmetric(Ẑ*Ẑ')
        R = Ĝ*Ẑ'; R = R + R'
        eig = eigen(P)
        R̃ = eig.vectors' * R * eig.vectors
        M̃ = similar(R̃)
        for i in 1:n, j in 1:n
            λ = eig.values[i] + eig.values[j]
            M̃[i, j] = λ > 1e-12 ? R̃[i, j]/λ : zero(T)
        end
        M = Symmetric(eig.vectors * M̃ * eig.vectors')
        b = vec(ḡ) .- M*vec(z̄)
        # Newton step of the non-local model; regularized when indefinite (Remark 1)
        λmin = eigmin(M)
        Δ = -(M + (λmin > 0 ? zero(T) : (-λmin + T(1e-8)*max(one(T), abs(λmin))))*I) \ b
        # paper line search: log grid over both candidate directions
        f_move = T(Inf); x_move = x
        for d in (Δ, -b), i in -opt.ls_range:opt.ls_range
            x_trial = x .+ opt.ls_base^i .* d
            f_trial = E(x_trial)
            isfinite(f_trial) || continue
            f_trial < f_move && ((f_move, x_move) = (f_trial, x_trial))
        end
        steplen = isfinite(f_move) ? norm(x_move .- x) : zero(T)
        if isfinite(f_move)
            x = x_move # non-monotone iteration, the best iterate is tracked separately
            if f_move < best_f
                best_f = f_move
                copyto!(best_p, x)
            end
        end
        # scaling adaption of the paper
        if σ < opt.σ_restart
            σ = opt.σ0
        elseif steplen < opt.steptol
            σ /= 2
        elseif steplen > 2σ
            σ = steplen/2
        end
    end
    copyto!(p, best_p)
    if opt.finish !== nothing
        best_f = polish_minimize!(opt.finish, p, offsets, F, admissible, W, xargs...)
    end
    return best_f
end

####################################################
############  Leaf-objective polishing  ############
####################################################
# Polishing of functionals that are NOT ξ-weighted integrals of a single scalar density,
# e.g. cross-relaxation increments that couple the laminate to a history measure through
# an optimal-transport pairing. The tree parametrization, the reverse gradient sweep and
# the optimizers are reused; the objective supplies the energy of a leaf collection and,
# for the analytic gradient, the per-leaf sensitivities with its internal minimizers
# (transport plan, condensed internal variables, ...) frozen at their optimum (Danskin).

@doc raw"""
    AbstractLeafObjective
Interface for polish objectives that are general functionals of the laminate
`ν = Σᵢ ξᵢ δ_{Fᵢ}` instead of a ξ-weighted sum of one scalar density. An objective `obj`
must implement
- `(obj)(leaves)::Real` — the energy of the leaf collection, where `leaves` is a vector of
  `(heapindex, ξᵢ, Fᵢ)` tuples as produced by [`polish_leaves!`](@ref);
- `leafduals(obj, leaves)::Dict{Int,Tuple{T,Tensor{2,dim,T,N}}}` — for every heap index the
  pair `(∂E/∂ξᵢ, ∂E/∂Fᵢ / ξᵢ)`, evaluated with all internal minimizers of the objective
  frozen at their optimum (envelope theorem). For a transport-coupled objective `∂E/∂ξᵢ`
  is the Kantorovich potential of leaf `i` (defined up to a constant, which cancels since
  tree-parameter variations conserve the total mass).
Use via `polish_minimize!(optimizer, obj, p, offsets, F, admissible)`.
"""
abstract type AbstractLeafObjective end

"""
    leafduals(obj::AbstractLeafObjective, leaves) -> Dict{Int,Tuple{T,Tensor{2,dim,T,N}}}
Per-leaf sensitivities `(∂E/∂ξᵢ, ∂E/∂Fᵢ / ξᵢ)` of the objective, internal minimizers frozen.
"""
function leafduals end

"""
    polish_leaves!(out, p, offsets, i, F, w, admissible) -> Bool
Collects the leaves `(heapindex, weight, F)` of the parametrized tree rooted at heap index `i`
into `out`. Returns `false` (and stops) on inadmissible nodes or negative offsets, mirroring
the penalty branches of [`polish_energy`](@ref).
"""
function polish_leaves!(out::Vector{Tuple{Int,T,Tensor{2,dim,T,N}}}, p::AbstractVector, offsets::Dict{Int,Int}, i::Int, F::Tensor{2,dim,T,N}, w::T, admissible::AFUN) where {dim,T,N,AFUN}
    admissible(F) || return false
    o = get(offsets, i, 0)
    na = _nangles(Val(dim))
    if o == 0 || p[o+2na] + p[o+2na+1] < 1e-12
        push!(out, (i, w, F))
        return true
    end
    θ𝐚 = ntuple(k->p[o+k-1], na)
    θ𝐛 = ntuple(k->p[o+na+k-1], na)
    s⁻ = p[o+2na]; s⁺ = p[o+2na+1]
    (s⁻ < 0 || s⁺ < 0) && return false
    𝐑 = _unitvec(Val(dim), θ𝐚) ⊗ _unitvec(Val(dim), θ𝐛)
    ξ = s⁻/(s⁻ + s⁺)
    polish_leaves!(out, p, offsets, minus_idx(i), F - s⁻*𝐑, w*(1-ξ), admissible) || return false
    polish_leaves!(out, p, offsets, plus_idx(i), F + s⁺*𝐑, w*ξ, admissible) || return false
    return true
end

function polish_energy(obj::AbstractLeafObjective, p::AbstractVector, offsets::Dict{Int,Int}, F::Tensor{2,dim,T,N}, admissible::AFUN) where {dim,T,N,AFUN}
    leaves = Tuple{Int,T,Tensor{2,dim,T,N}}[]
    polish_leaves!(leaves, p, offsets, 1, F, one(T), admissible) || return T(Inf)
    return obj(leaves)
end

# reverse sweep as in _gradsweep!, but the leaf pairs (value-dual, stress-dual) come from the objective
function _gradsweep_obj!(g::Vector{T}, p::Vector{T}, offsets::Dict{Int,Int}, i::Int, F::Tensor{2,dim,T}, w_anc::T, duals::Dict{Int,Tuple{T,Tensor{2,dim,T,N}}}) where {dim,T,N}
    o = get(offsets, i, 0)
    na = _nangles(Val(dim))
    if o == 0 || p[o+2na] + p[o+2na+1] < 1e-12
        return duals[i]
    end
    θ𝐚 = ntuple(k->p[o+k-1], na)
    θ𝐛 = ntuple(k->p[o+na+k-1], na)
    s⁻ = p[o+2na]; s⁺ = p[o+2na+1]
    𝐚 = _unitvec(Val(dim), θ𝐚); 𝐛 = _unitvec(Val(dim), θ𝐛)
    𝐑 = 𝐚 ⊗ 𝐛
    ξ = s⁻/(s⁻ + s⁺)
    S⁻, T⁻ = _gradsweep_obj!(g, p, offsets, minus_idx(i), F - s⁻*𝐑, w_anc*(1-ξ), duals)
    S⁺, T⁺ = _gradsweep_obj!(g, p, offsets, plus_idx(i), F + s⁺*𝐑, w_anc*ξ, duals)
    ∂ξ∂s⁻ = s⁺/(s⁻ + s⁺)^2
    ∂ξ∂s⁺ = -s⁻/(s⁻ + s⁺)^2
    𝐓θ = -s⁻*(1-ξ)*T⁻ + s⁺*ξ*T⁺
    d𝐚 = _dunitvec(Val(dim), θ𝐚); d𝐛 = _dunitvec(Val(dim), θ𝐛)
    for k in 1:na
        g[o+k-1]    += w_anc * (𝐓θ ⊡ (d𝐚[k] ⊗ 𝐛))
        g[o+na+k-1] += w_anc * (𝐓θ ⊡ (𝐚 ⊗ d𝐛[k]))
    end
    g[o+2na]   += w_anc * (∂ξ∂s⁻*(S⁺ - S⁻) - (1-ξ)*(T⁻ ⊡ 𝐑))
    g[o+2na+1] += w_anc * (∂ξ∂s⁺*(S⁺ - S⁻) + ξ*(T⁺ ⊡ 𝐑))
    return ξ*S⁺ + (1-ξ)*S⁻, ξ*T⁺ + (1-ξ)*T⁻
end

"""
    polish_gradient!(g, obj::AbstractLeafObjective, p, offsets, F, admissible) -> g
Analytic gradient of the leaf objective w.r.t. the tree parameters: one reverse sweep with the
per-leaf sensitivities from [`leafduals`](@ref) (internal minimizers of the objective frozen).
Assumes the current parameters are admissible (interior).
"""
function polish_gradient!(g::Vector{T}, obj::AbstractLeafObjective, p::Vector{T}, offsets::Dict{Int,Int}, F::Tensor{2,dim,T,N}, admissible::AFUN) where {dim,T,N,AFUN}
    Base.fill!(g, zero(T))
    leaves = Tuple{Int,T,Tensor{2,dim,T,N}}[]
    polish_leaves!(leaves, p, offsets, 1, F, one(T), admissible) || return g
    duals = leafduals(obj, leaves)
    _gradsweep_obj!(g, p, offsets, 1, F, one(T), duals)
    return g
end

function polish_minimize!(opt::CompassSearch, obj::AbstractLeafObjective, p::Vector{T}, offsets::Dict{Int,Int}, F::Tensor{2,dim,T}, admissible::AFUN) where {dim,T,AFUN}
    E = q -> polish_energy(obj, q, offsets, F, admissible)
    return _compass_core!(E, p, offsets, _nangles(Val(dim)), opt.steptol, opt.maxsweeps)
end

function polish_minimize!(opt::BFGS{AnalyticGradient}, obj::AbstractLeafObjective, p::Vector{T}, offsets::Dict{Int,Int}, F::Tensor{2,dim,T}, admissible::AFUN) where {dim,T,AFUN}
    E = q -> polish_energy(obj, q, offsets, F, admissible)
    grad! = (g, q) -> polish_gradient!(g, obj, q, offsets, F, admissible)
    f0 = E(p)
    isfinite(f0) || return f0
    f = _bfgs_core!(E, grad!, p, opt)
    if !(f < f0) && opt.fallback !== nothing
        f = polish_minimize!(opt.fallback, obj, p, offsets, F, admissible)
    end
    return f
end

function _deactivate_subtree!(bt::BinaryLaminationTree, i::Int)
    i ≤ length(bt.active) || return nothing
    bt.active[i] || return nothing
    bt.active[i] = false
    _deactivate_subtree!(bt, minus_idx(i))
    _deactivate_subtree!(bt, plus_idx(i))
    return nothing
end

function polish_apply!(bt::BinaryLaminationTree{dim,T}, p::Vector{T}, offsets::Dict{Int,Int}, i::Int, F::Tensor{2,dim,T}, ξnode::T, W::FUN, xargs::Vararg{Any,XN}) where {dim,T,FUN,XN}
    bt.nodes[i] = BinaryLaminationTreeNode(F, W(F, xargs...), ξnode, bt.nodes[i].level)
    o = get(offsets, i, 0)
    if o == 0
        # node is a leaf of the (possibly depth-truncated) parametrization: drop any deeper
        # greedy structure so the tree stays consistent with the optimized laminate
        _deactivate_subtree!(bt, minus_idx(i))
        _deactivate_subtree!(bt, plus_idx(i))
        return nothing
    end
    na = _nangles(Val(dim))
    θ𝐚 = ntuple(k->p[o+k-1], na)
    θ𝐛 = ntuple(k->p[o+na+k-1], na)
    s⁻ = p[o+2na]; s⁺ = p[o+2na+1]
    if s⁻ + s⁺ < 1e-12 # split degenerated during optimization → node became a leaf
        _deactivate_subtree!(bt, minus_idx(i))
        _deactivate_subtree!(bt, plus_idx(i))
        return nothing
    end
    𝐑 = _unitvec(Val(dim), θ𝐚) ⊗ _unitvec(Val(dim), θ𝐛)
    ξ = s⁻/(s⁻ + s⁺)
    polish_apply!(bt, p, offsets, minus_idx(i), F - s⁻*𝐑, (1-ξ), W, xargs...)
    polish_apply!(bt, p, offsets, plus_idx(i), F + s⁺*𝐑, ξ, W, xargs...)
    return nothing
end

"""
    polish!(bt::BinaryLaminationTree, convexification::HROC, buffer::HROCBuffer, admissible, W, F, xargs...; maxdepth=4) -> bt
Post-processes a discrete HROC lamination tree by jointly optimizing all lamination directions
and endpoint offsets (fixed topology, truncated at tree depth `maxdepth` — greedy refinement
chains beyond that depth are redundant once the remaining parameters are jointly optimal and
only deteriorate the optimization, see [`polish_parameters`](@ref)), then writes the optimized
laminate back into the tree. The optimizer instance (with its hyperparameters) is taken from
`convexification.polish`, see [`AbstractPolishOptimizer`](@ref).
If the discrete stage found no laminate at all (trivial tree),
a speculative rank-two template split is seeded and optimized instead, which resolves regions
where a single lamination of the unrelaxed `W` is not energy-decreasing although the rank-one
convex envelope lies below `W` (e.g. the second-order laminate region of the Kohn-Strang-Dolzmann
example). Enabled via the `polish` option of [`HROC`](@ref).
"""
function polish!(bt::BinaryLaminationTree{dim,T}, convexification::HROC, buffer::HROCBuffer, admissible::AFUN, W::FUN, F::Tensor{2,dim,T}, xargs::Vararg{Any,XN}; maxdepth::Int=4) where {dim,T,AFUN,FUN,XN}
    if haschildren(bt, 1)
        p, offsets = polish_parameters(bt; maxdepth=maxdepth)
        polish_minimize!(convexification.polish, p, offsets, F, admissible, W, xargs...)
        polish_apply!(bt, p, offsets, 1, F, bt.nodes[1].ξ, W, xargs...)
    else
        polish_speculate!(bt, convexification, buffer, admissible, W, F, xargs...)
    end
    return bt
end

function _split_angles(D::Tensor{2,dim,T}) where {dim,T}
    decomp = svd(Array(D))
    𝐚 = Vec{dim,T}(NTuple{dim,T}(decomp.U[:,1]))
    𝐛 = Vec{dim,T}(NTuple{dim,T}(decomp.V[:,1]))
    ((𝐚 ⊗ 𝐛) ⊡ D) < 0 && (𝐛 = -𝐛)
    return (_angles(𝐚)..., _angles(𝐛)...)
end

"""
    polish_speculate!(bt, convexification, buffer, admissible, W, F, xargs...; n_scan=256, n_speculate=2)
Speculative polishing for points where the greedy stage found no laminate. Rank-one lines through `F`
are ranked by the minimum of `W` along them; for the best candidates a rank-two template tree is
seeded (root endpoint `F⁺` at the line minimum, its child split obtained by `hrockernel`) and
jointly optimized with `convexification.polish`. The result is only accepted if it is strictly
below `W(F)`.
"""
function polish_speculate!(bt::BinaryLaminationTree{dim,T,N}, convexification::HROC, buffer::HROCBuffer, admissible::AFUN, W::FUN, F::Tensor{2,dim,T,N}, xargs::Vararg{Any,XN}; n_scan=256, n_speculate=2) where {dim,T,N,AFUN,FUN,XN}
    W_ref = W(F, xargs...)
    span = norm(Tensor{2,dim,T,N}(NTuple{dim*dim,T}(convexification.endF[i] - convexification.startF[i] for i in 1:dim*dim)))
    # rank all directions by the minimum of W along their line through F
    ranked = Tuple{T,T,Tensor{2,dim,T,N}}[] # (W_linemin, t_linemin, unit direction)
    for 𝐀 in convexification.dirs
        Â = 𝐀 / norm(𝐀)
        t_min, W_min = zero(T), W_ref
        for t in range(-span, span; length=2*n_scan+1)
            𝐱 = F + t*Â
            admissible(𝐱) || continue
            w = W(𝐱, xargs...)
            w < W_min && ((t_min, W_min) = (t, w))
        end
        (abs(t_min) > 1e-10 && W_min < W_ref) && push!(ranked, (W_min, t_min, Â))
    end
    isempty(ranked) && return bt # no line reaches below W(F): nothing to speculate on
    sort!(ranked, by=first)
    ppn = _nparams(Val(dim))
    offsets = Dict(1=>1, 2=>1+ppn, 3=>1+2*ppn)
    best_f, best_p = T(Inf), T[]
    for (W_min, t_min, Â) in ranked[1:min(n_speculate,length(ranked))]
        𝐃 = sign(t_min) * Â
        s⁺ = abs(t_min)
        θroot = _split_angles(𝐃)
        F⁺ = F + s⁺*𝐃
        laminate⁺ = hrockernel(bt, convexification, buffer, W, F⁺, xargs...)
        child⁺ = if laminate⁺ === nothing
            (θroot..., zero(T), zero(T))
        else
            (_split_angles(laminate⁺.F⁺ - laminate⁺.F⁻)..., norm(F⁺ - laminate⁺.F⁻), norm(laminate⁺.F⁺ - F⁺))
        end
        child⁻ = (θroot..., zero(T), zero(T))
        # choose the best root balance point among a few candidates, optimize only that one
        p_init, f_init = T[], T(Inf)
        for s⁻ in (s⁺/4, s⁺/2, s⁺, 2s⁺)
            admissible(F - s⁻*𝐃) || continue
            p = T[θroot..., s⁻, s⁺, child⁻..., child⁺...]
            f = polish_energy(p, offsets, 1, F, admissible, W, xargs...)
            f < f_init && ((p_init, f_init) = (p, f))
        end
        isempty(p_init) && continue
        f = polish_minimize!(convexification.polish, p_init, offsets, F, admissible, W, xargs...)
        f < best_f && ((best_f, best_p) = (f, p_init))
    end
    best_f < W_ref || return bt # speculation did not improve upon W(F): keep the trivial tree
    # materialize the template nodes, then write the optimized laminate back
    rootlevel = bt.nodes[1].level
    for i in 2:7
        _setnode!(bt, i, BinaryLaminationTreeNode(F, W_ref, T(0.5), max(rootlevel - (i < 4 ? 1 : 2), 1)))
    end
    polish_apply!(bt, best_p, offsets, 1, F, bt.nodes[1].ξ, W, xargs...)
    return bt
end

_admissible(convexification::HROC) = 𝐱 -> inbounds(𝐱, convexification) && (convexification.GLcheck ? det(𝐱) > 1e-6 : true)

@doc raw"""
    convexify(hroc::HROC, buffer::HROCBuffer, W::FUN, F::T1, xargs::Vararg{Any,XN}) -> bt::BinaryLaminationTree
Performs a hierarchical rank one convexification (HROC) based on the H-sequence characterization of the convex envelope.
Note that the output of the algorithm is only an upper bound. For a class of problems the provided hull matches the rank-one convex envelope.
The return of the algorithm can be used to call `eval` which evaluates the constructed binary lamination tree in terms of semi convex envelope value and its derivatives.
"""
function convexify(hroc::HROC, buffer::HROCBuffer, W::FUN, F::T1, xargs::Vararg{Any,XN}) where {T1,FUN,XN}
    return BinaryLaminationTree(hroc,buffer,W,F,xargs...)
end

@doc raw"""
    convexify(prev_bt::BinaryLaminationTree,hroc::HROC, buffer::HROCBuffer, W::FUN, F::T1, xargs::Vararg{Any,XN}) -> bt::BinaryLaminationTree
Performs a hierarchical rank one convexification (HROC) based and enforces laminate continuity by preferring the previous laminate direction.
"""
function convexify(prev_F,prev_bt::BinaryLaminationTree, hroc::HROC, buffer::HROCBuffer, constraint, irr, W::FUN, F::T1, xargs::Vararg{Any,XN}) where {T1,FUN,XN}
    return BinaryLaminationTree(prev_F,prev_bt,hroc,buffer,constraint,irr,W,F,xargs...)
end

function stretchfilter(F)
    C = tdot(F)
    eigen_ = eigen(C)
    if !all(eigen_.values .> 0)
        return zero(F)
    else
        return sqrt(C)
    end
end

function same_as_previous(A,prev)
    U_p, S_p, Vt_p = svd(prev)
    U, S, Vt = svd(A)
    return all(isapprox.(U_p[:,1] * sqrt(S_p[1]),U[:,1]*sqrt(S[1])))
end

function hrockernel(root::BinaryLaminationTree, convexification::HROC, buffer::HROCBuffer, W::FUN, F::Tensor{2,dim,T,N}, xargs::Vararg{Any,XN}) where {dim,T,N,FUN,XN}
    W_ref = W(F,xargs...)
    𝔸_ref, _, W_glob_ref = eval(root,W,xargs...)
    laminate = nothing
    for 𝐀 in convexification.dirs
        fill!(buffer) # fill buffers with zeros
        _δ = minimum(δ(convexification, 𝐀))
        𝐀 *= _δ
        if norm(𝐀,Inf) > 0
            ctr_fw = 0
            ctr_bw = 0
            for dir in (-1, 1)
                if dir==-1
                    𝐱 = F - 𝐀 # init dir
                    ell = -1 # start at -1, so - 𝐀
                else
                    𝐱 = F # init dir
                    ell = 0 # start at 0
                end
                while inbounds(𝐱,convexification) && (convexification.GLcheck ? det(𝐱) > 1e-6 : true)
                    val = W(𝐱,xargs...)
                    if dir == 1
                        buffer.forward_initial.values[ctr_fw+1] = val
                        buffer.forward_initial.grid[ctr_fw+1] = ell
                        ctr_fw += 1
                    else
                        buffer.backward_initial.values[ctr_bw+1] = val
                        buffer.backward_initial.grid[ctr_bw+1] = ell
                        ctr_bw += 1
                    end
                    𝐱 += dir*𝐀
                    ell += dir
                end
            end
            if ((ctr_fw > 0) && (ctr_bw > 0))
                concat!(buffer,ctr_fw+1,ctr_bw)
                Wᶜ, j = convexify!(buffer,ctr_bw+ctr_fw)
                l₁ = buffer.convex.grid[j-1]
                l₂ = buffer.convex.grid[j]
                F⁻ = F + l₁*𝐀
                F⁺ = F + l₂*𝐀
                W⁻ = W(F⁻,xargs...)
                W⁺ = W(F⁺,xargs...)
                lc = Laminate(F⁻,F⁺,W⁻,W⁺,𝐀,0)
                𝔸, _, W_glob_trial = eval(root,F,lc,W,xargs...)
                if (Wᶜ <= W_ref)
                    W_ref = Wᶜ
                    𝔸_ref = 𝔸
                    W_glob_ref = W_glob_trial
                    laminate = lc
                end
            end
        end
    end
    return laminate
end

@doc raw"""
    LabeledDensity(f)
Wraps an incremental density for the constrained (cross) convexification whose value depends on
the matched node of the previous internal-variable tree: the synchronized descent of the
constrained constructor forwards the matched `NodeView` and the density is evaluated as
`f(𝐱, node, xargs...)`. This realizes label-inherited (tree-lineage) transport: every branch of
the new laminate is condensed against the history of its matched previous phase, which conserves
the previous measure's masses by construction.
"""
struct LabeledDensity{F}
    f::F
end

@doc raw"""
    LabeledConstraint(f)
Wraps an admissibility predicate for the constrained convexification that is tested against the
matched node of the previous internal-variable tree instead of the whole tree:
`f(prev_F, prev, node, 𝐱, xargs...)`. Replaces the disjunction over all previous leaves (which
ignores mass conservation of the history measure) by the label-inherited test.
"""
struct LabeledConstraint{F}
    f::F
end

@inline _eval_density(W::LabeledDensity, 𝐱, prev, prev_idx, xargs...) = W.f(𝐱, NodeView(prev, prev_idx), xargs...)
@inline _eval_density(W, 𝐱, prev, prev_idx, xargs...) = W(𝐱, xargs...)
@inline _eval_constraint(c::LabeledConstraint, prev_F, prev, prev_idx, 𝐱, xargs...) = c.f(prev_F, prev, NodeView(prev, prev_idx), 𝐱, xargs...)
@inline _eval_constraint(c, prev_F, prev, prev_idx, 𝐱, xargs...) = c(prev_F, prev, 𝐱, xargs...)

function hrockernel(prev_direction,prev_F,prev::BinaryLaminationTree, prev_idx::Int, root::BinaryLaminationTree, convexification::HROC, buffer::HROCBuffer, constraint, diss_offset, W::FUN, F::Tensor{2,dim,T,N}, xargs::Vararg{Any,XN}) where {dim,T,N,FUN,XN}
    W_ref = _eval_density(W, F, prev, prev_idx, xargs...)
    laminate = nothing
    # CE: try prev_direction first (only active when non-zero, i.e. at root level).
    # Children pass zero(F), so the iszero guard skips it there → free search at child level.
    for (try_prev, 𝐀_raw) in Iterators.flatten((((true, prev_direction),), ((false, a) for a in convexification.dirs)))
        iszero(𝐀_raw) && continue
        !try_prev && same_as_previous(𝐀_raw, prev_direction) && continue
        fill!(buffer) # fill buffers with zeros
        𝐀 = 𝐀_raw * minimum(δ(convexification, 𝐀_raw))
        if norm(𝐀,Inf) > 0
            ctr_fw = 0
            ctr_bw = 0
            for dir in (-1, 1)
                if dir==-1
                    𝐱 = F - 𝐀 # init dir
                    ell = -1 # start at -1, so - 𝐀
                else
                    𝐱 = F # init dir
                    ell = 0 # start at 0
                end
                while inbounds(𝐱,convexification) && (convexification.GLcheck ? det(𝐱) > 1e-6 : true) && _eval_constraint(constraint, prev_F, prev, prev_idx, 𝐱, xargs...)
                    val = _eval_density(W, 𝐱, prev, prev_idx, xargs...)
                    if dir == 1
                        buffer.forward_initial.values[ctr_fw+1] = val
                        buffer.forward_initial.grid[ctr_fw+1] = ell
                        ctr_fw += 1
                    else
                        buffer.backward_initial.values[ctr_bw+1] = val
                        buffer.backward_initial.grid[ctr_bw+1] = ell
                        ctr_bw += 1
                    end
                    𝐱 += dir*𝐀
                    ell += dir
                end
            end
            if ((ctr_fw > 0) && (ctr_bw > 0))
                concat!(buffer,ctr_fw+1,ctr_bw)
                Wᶜ, j = convexify!(buffer,ctr_bw+ctr_fw)
                l₁ = buffer.convex.grid[j-1]
                l₂ = buffer.convex.grid[j]
                F⁻ = F + l₁*𝐀
                F⁺ = F + l₂*𝐀
                W⁻ = _eval_density(W, F⁻, prev, prev_idx, xargs...)
                W⁺ = _eval_density(W, F⁺, prev, prev_idx, xargs...)
                lc = Laminate(F⁻,F⁺,W⁻,W⁺,𝐀,0)
                if (Wᶜ <= W_ref)
                    W_ref = Wᶜ
                    laminate = lc
                    try_prev && return laminate  # CE: root prev direction works — accept immediately
                end
            end
        end
    end
    return laminate
end

function laminatekernel(𝐀::Tensor{2,dim,T,N},convexification::HROC, buffer::HROCBuffer, W::FUN, F::Tensor{2,dim,T,N}, xargs::Vararg{Any,XN}) where {dim,T,N,FUN,XN}
    W_ref = W(F,xargs...)
    laminate = nothing
    fill!(buffer) # fill buffers with zeros
    _δ = minimum(δ(convexification, 𝐀))
    𝐀 *= _δ
    ctr_fw = 0
    ctr_bw = 0
    for dir in (-1, 1)
        if dir==-1
            𝐱 = F - 𝐀 # init dir
            ell = -1 # start at -1, so - 𝐀
        else
            𝐱 = F # init dir
            ell = 0 # start at 0
        end
        while inbounds(𝐱,convexification) && (convexification.GLcheck ? det(𝐱) > 1e-6 : true)
            val = W(𝐱,xargs...)
            if dir == 1
                buffer.forward_initial.values[ctr_fw+1] = val
                buffer.forward_initial.grid[ctr_fw+1] = ell
                ctr_fw += 1
            else
                buffer.backward_initial.values[ctr_bw+1] = val
                buffer.backward_initial.grid[ctr_bw+1] = ell
                ctr_bw += 1
            end
            𝐱 += dir*𝐀
            ell += dir
        end
    end
    if ((ctr_fw > 0) && (ctr_bw > 0))
        concat!(buffer,ctr_fw+1,ctr_bw)
        Wᶜ, j = convexify!(buffer,ctr_bw+ctr_fw)
        if (Wᶜ < W_ref)
            W_ref = Wᶜ
            l₁ = buffer.convex.grid[j-1]
            l₂ = buffer.convex.grid[j]
            F⁻ = F + l₁*𝐀
            F⁺ = F + l₂*𝐀
            W⁻ = W(F⁻,xargs...)
            W⁺ = W(F⁺,xargs...)
            laminate = Laminate(F⁻,F⁺,W⁻,W⁺,𝐀,0)
        end
    end
    return laminate
end

function NumericalRelaxation.eval(bt::BinaryLaminationTree{dim}, idx::Int, W_nonconvex::FUN, xargs::Vararg{Any,XN}) where {dim,FUN,XN}
    W = 0.0
    𝐏 = zero(Tensor{2,dim})
    𝔸 = zero(Tensor{4,dim})
    if isleaf(bt, idx)
        𝔸_temp, 𝐏_temp, W_temp = Tensors.hessian(y -> W_nonconvex(y, xargs...), bt.nodes[idx].F, :all)
        W += W_temp; 𝐏 += 𝐏_temp; 𝔸 += 𝔸_temp
    else
        𝔸⁻, 𝐏⁻, W⁻ = eval(bt, minus_idx(idx), W_nonconvex, xargs...)
        𝔸⁺, 𝐏⁺, W⁺ = eval(bt, plus_idx(idx), W_nonconvex, xargs...)
        ξ = bt.nodes[plus_idx(idx)].ξ
        W += ξ*W⁺+(1-ξ)*W⁻; 𝐏 += ξ*𝐏⁺+(1-ξ)*𝐏⁻; 𝔸 += ξ*𝔸⁺+(1-ξ)*𝔸⁻
    end
    return 𝔸, 𝐏, W
end

function NumericalRelaxation.eval(bt::BinaryLaminationTree{dim}, W_nonconvex::FUN, xargs::Vararg{Any,XN}) where {dim,FUN,XN}
    return eval(bt, 1, W_nonconvex, xargs...)
end

function NumericalRelaxation.eval(bt::BinaryLaminationTree{dim}, idx::Int, F::Tensor{2,dim}, laminate::Laminate{dim}, W_nonconvex::FUN, xargs::Vararg{Any,XN}) where {dim,FUN,XN}
    W = 0.0
    𝐏 = zero(Tensor{2,dim})
    𝔸 = zero(Tensor{4,dim})
    if isleaf(bt, idx)
        if bt.nodes[idx].F ≈ F
            𝔸⁻, 𝐏⁻, W⁻ = Tensors.hessian(y -> W_nonconvex(y, xargs...), laminate.F⁻, :all)
            𝔸⁺, 𝐏⁺, W⁺ = Tensors.hessian(y -> W_nonconvex(y, xargs...), laminate.F⁺, :all)
            ξ = norm(F - laminate.F⁻) / norm(laminate.F⁺ - laminate.F⁻)
            W += ξ*W⁺+(1-ξ)*W⁻; 𝐏 += ξ*𝐏⁺+(1-ξ)*𝐏⁻; 𝔸 += ξ*𝔸⁺+(1-ξ)*𝔸⁻
        else
            𝔸_temp, 𝐏_temp, W_temp = Tensors.hessian(y -> W_nonconvex(y, xargs...), bt.nodes[idx].F, :all)
            W += W_temp; 𝐏 += 𝐏_temp; 𝔸 += 𝔸_temp
        end
    else
        𝔸⁻, 𝐏⁻, W⁻ = eval(bt, minus_idx(idx), F, laminate, W_nonconvex, xargs...)
        𝔸⁺, 𝐏⁺, W⁺ = eval(bt, plus_idx(idx), F, laminate, W_nonconvex, xargs...)
        ξ = bt.nodes[plus_idx(idx)].ξ
        W += ξ*W⁺+(1-ξ)*W⁻; 𝐏 += ξ*𝐏⁺+(1-ξ)*𝐏⁻; 𝔸 += ξ*𝔸⁺+(1-ξ)*𝔸⁻
    end
    return 𝔸, 𝐏, W
end

function NumericalRelaxation.eval(bt::BinaryLaminationTree{dim}, F::Tensor{2,dim}, laminate::Laminate{dim}, W_nonconvex::FUN, xargs::Vararg{Any,XN}) where {dim,FUN,XN}
    return eval(bt, 1, F, laminate, W_nonconvex, xargs...)
end

function checkintegrity(tree::BinaryLaminationTree,tol=1e-4)
    isintegre = true
    for i in 1:length(tree.active)
        !tree.active[i] && continue
        isleaf(tree, i) && continue
        F = tree.nodes[i].F
        mi = minus_idx(i)
        pi = plus_idx(i)
        points = [tree.nodes[mi].F, tree.nodes[pi].F]
        weights = [tree.nodes[mi].ξ, tree.nodes[pi].ξ]
        # rank-one connectivity checked on the normalized direction with a cancellation-aware
        # tolerance: for splits of small width h the direction computed from the stored endpoints
        # carries relative rounding noise of order eps·|F|/h, which the default rank tolerance flags
        Δ = points[2] - points[1]
        rankone = norm(Δ) < tol || rank(Δ / norm(Δ), rtol=sqrt(eps(Float64))) < 2
        isintegre = isapprox(F,sum(points .* weights),atol=tol) && rankone
        if !isintegre
            break
        end
    end
    return isintegre
end

function rotation_tensor(F::Tensor{2})
    n = 0
    prev = one(F)
    while !isapprox(prev,F)
       n += 1
       prev = F
       F = 0.5*(F + inv(F)')
    end
    return F
end

function rotationangles(R::Tensor{2,2})
    return acos(R[1])
end

function rotationangles(R::Tensor{2,3})
    if R[3,1] != 1 || R[3,1] != -1
        θ = -asin(R[3,1])
        # θ₂ = π - θ ignored
        ψ = atan(R[3,2]/cos(θ),R[3,3]/cos(θ))
        ϕ = atan(R[2,1]/cos(θ),R[1,1]/cos(θ))
    else
        ϕ = 0
        if R[3,1] == -1
            θ = π/2
            ψ = atan(R[1,2],R[1,3])
        else
            θ = -π/2
            ψ = atan(-R[1,2],-R[1,3])
        end
    end
    return (ψ,θ,ϕ)
end

function rotate!(bt::BinaryLaminationTree,args...)
    for i in 1:length(bt.active)
        bt.active[i] || continue
        node = bt.nodes[i]
        bt.nodes[i] = BinaryLaminationTreeNode(Tensors.rotate(node.F,args...), node.W, node.ξ, node.level)
    end
end

function rotate(bt::BinaryLaminationTree,args...)
    new_bt = BinaryLaminationTree(copy(bt.nodes), copy(bt.active))
    rotate!(new_bt,args...)
    return new_bt
end

function rotationaverage(bt::BinaryLaminationTree{2},W::FUN,xargs::Vararg{Any,N}) where {FUN,N}
    𝔸, 𝐏, W_ref = eval(bt, W, xargs...)
    bt_rotate = rotate(bt,0)
    angles = pi/180:pi/180:pi
    counter = 1
    for α in angles
        rotate!(bt_rotate,α)
        𝔸_r, 𝐏_r, W_r = eval(bt_rotate, W, xargs...)
        if isapprox(W_r,W_ref)
            𝔸 += 𝔸_r; 𝐏 += 𝐏_r; W_ref += W_r
            counter += 1
        end
        rotate!(bt_rotate,-α) #rotate back
    end
    return 𝔸/counter, 𝐏/counter, W_ref/counter
end

function rotationaverage(bt::BinaryLaminationTree{3},W::FUN,xargs::Vararg{Any,N}) where {FUN,N}
    𝔸, 𝐏, W_ref = eval(bt, W, xargs...)
    bt_rotate = rotate(bt,0,0,0)
    angles = pi/10:pi/10:pi
    counter = 1
    for α ∈ angles, β ∈ angles, γ ∈ angles
        rotate!(bt_rotate,α,β,γ)
        𝔸_r, 𝐏_r, W_r = eval(bt_rotate, W, xargs...)
        if isapprox(W_r,W_ref)
            𝔸 += 𝔸_r; 𝐏 += 𝐏_r; W_ref += W_r
            counter += 1
        end
        rotate!(bt_rotate,-α,-β,-γ) #rotate back
    end
    return 𝔸/counter, 𝐏/counter, W_ref/counter
end

# NodeView wrapper for AbstractTrees interface and external callbacks
struct NodeView{dim,T,N}
    tree::BinaryLaminationTree{dim,T,N}
    idx::Int
end

Base.getproperty(nv::NodeView, s::Symbol) = _nv_getproperty(nv, Val(s))
_nv_getproperty(nv::NodeView, ::Val{:tree}) = getfield(nv, :tree)
_nv_getproperty(nv::NodeView, ::Val{:idx}) = getfield(nv, :idx)
_nv_getproperty(nv::NodeView, ::Val{:F}) = getfield(nv, :tree).nodes[getfield(nv, :idx)].F
_nv_getproperty(nv::NodeView, ::Val{:W}) = getfield(nv, :tree).nodes[getfield(nv, :idx)].W
_nv_getproperty(nv::NodeView, ::Val{:ξ}) = getfield(nv, :tree).nodes[getfield(nv, :idx)].ξ
_nv_getproperty(nv::NodeView, ::Val{:level}) = getfield(nv, :tree).nodes[getfield(nv, :idx)].level
function _nv_getproperty(nv::NodeView, ::Val{:minus})
    bt = getfield(nv, :tree)
    mi = minus_idx(getfield(nv, :idx))
    mi ≤ length(bt.active) && bt.active[mi] ? NodeView(bt, mi) : nothing
end
function _nv_getproperty(nv::NodeView, ::Val{:plus})
    bt = getfield(nv, :tree)
    pi = plus_idx(getfield(nv, :idx))
    pi ≤ length(bt.active) && bt.active[pi] ? NodeView(bt, pi) : nothing
end
function _nv_getproperty(nv::NodeView, ::Val{:parent})
    idx = getfield(nv, :idx)
    idx == 1 ? nothing : NodeView(getfield(nv, :tree), parent_idx(idx))
end

function equilibrium(nv::NodeView,W::FUN,xargs::Vararg{Any,N}) where {FUN,N}
   nv.idx == 1 && return (0.0,AbstractTrees.children(nv))
   p = nv.parent
   sibling = nv.idx == plus_idx(p.idx) ? p.minus : p.plus
   λ₁ = nv.ξ; λ₂ = sibling.ξ
   A = nv.idx == plus_idx(p.idx) ? nv.F - sibling.F : sibling.F - nv.F
   W⁺ = nv.idx == plus_idx(p.idx) ? nv.W : sibling.W
   W⁻ = nv.idx == minus_idx(p.idx) ? nv.W : sibling.W
   P₁ = Tensors.gradient(y->W(y,xargs...),nv.F)
   P₂ = Tensors.gradient(y->W(y,xargs...),sibling.F)
   return ((W⁺-W⁻)-((λ₁*P₁+λ₂*P₂)⊡(A)),AbstractTrees.children(nv))
end

AbstractTrees.printnode(io::IO, nv::NodeView) = print(io, "$(nv.F) ξ=$(nv.ξ)")
AbstractTrees.printnode(io::IO, bt::BinaryLaminationTree) = AbstractTrees.printnode(io, NodeView(bt, 1))
AbstractTrees.ParentLinks(::Type{<:NodeView}) = AbstractTrees.StoredParents()
AbstractTrees.SiblingLinks(::Type{<:NodeView}) = AbstractTrees.ImplicitSiblings()
Base.show(io::IO, ::MIME"text/plain", tree::BinaryLaminationTree) = AbstractTrees.print_tree(io, NodeView(tree, 1))
Base.eltype(::Type{<:AbstractTrees.TreeIterator{NodeView{dim,T,N}}}) where {dim,T,N} = NodeView{dim,T,N}
Base.IteratorEltype(::Type{<:AbstractTrees.TreeIterator{NodeView{dim,T,N}}}) where {dim,T,N} = Base.HasEltype()

AbstractTrees.parent(nv::NodeView) = nv.parent
function AbstractTrees.children(nv::NodeView)
    m = nv.minus
    p = nv.plus
    if !isnothing(m)
        if !isnothing(p)
            return (m, p)
        end
        return (m,)
    end
    !isnothing(p) && return (p,)
    return ()
end


####################################################
####################################################
##############  Polyconvexification  ###############
####################################################
####################################################

@doc raw"""
    PolyConvexification{} <: AbstractConvexification

Datastructure which holds basic parameters and grid for the Polyconvexification

- `dimp::Int` dimension of the physical problem
- `dimc::Int` lifted dimension, 2D -> 3, 3D -> 7, in this dimension the convexificaiton problem for the polyconvexification is stated 
- `r::Float64` discretization radius
- `nref::Int` number of uniform grid refinements
- `grid::Vector{T1}` grid of the signed singula values
- `liftedGrid::Vector{T1}` lifted grid of signed singular values through application of the minors function
"""
struct PolyConvexification{T1,T2} <: AbstractConvexification
    dimp::Int
    dimc::Int
    r::Float64
    nref::Int
    grid::Vector{T1}
    liftedGrid::Vector{T2}
end


function PolyConvexification(dimp::Int, r::Float64; nref::Int=9, δ::Float64=0.0)
    if δ <= 0.0
        δ = 2 * r / 2 .^ nref
    end
    if dimp == 2
        p = Iterators.product(-r:δ:r, -r:δ:r)
    elseif dimp == 3
        p = Iterators.product(-r:δ:r, -r:δ:r, -r:δ:r)
    end
    grid = vec(collect.(p))
    liftedGrid = minors.(grid)
    return PolyConvexification(dimp, length(minors(ones(dimp))), r, nref, grid, liftedGrid)
end


@doc raw"""
- `Φν_δ::Vector{T1}` holds the values of `Φ` evaluated at the grid `ν_δ`
- `Φactive::Vector{Bool}` marks the grid points involved in the minimization problem
"""
struct PolyConvexificationBuffer{T} <: AbstractConvexificationBuffer
    Φν_δ::Vector{T}
    Φactive::Vector{Bool}
end

function build_buffer(poly_convexification::PolyConvexification)
    nrpoints = length(poly_convexification.grid)
    return PolyConvexificationBuffer(zeros(nrpoints), Base.fill!(Vector{Bool}(undef, nrpoints), false))
end


@doc raw"""
    convexify(poly_convexification::PolyConvexification, poly_buffer::PolyConvexificationBuffer, Φ::FUN, ν::Union{Vec{d},Vector{Float64}}, xargs::Vararg{Any,XN}; returnDerivs::Bool=true) where {FUN,XN,d}
Signed singular value polyconvexification using the linear programming approach.
Compute approximation to the singular value polycovex envelope of the function `Φ` which is the reformulation of the isotropic function `W`
in terms of signed singular values $Φ(ν) = W(diagm(ν))$, at the point `ν` via the linear programming approach as discussed in 
    [^1] Timo Neumeier, Malte A. Peter, Daniel Peterseim, David Wiedemann.
    Computational polyconvexification of isotropic functions, arXiv 2307.15676, 2023.
The parameters `nref` and `r` (stored in poly_convexification struct) discribe the grid by radius `r` (in the ∞ norm) and `nref` uniform mesh refinements.
The points of the lifted grid which are involved in the minimization are marked by the Φactive buffer, and deliver `Φ` values smaller than infinity.

`Φ::FUN` function in terms of signed singular values `Φ(ν) = W(diagm(ν))`
`ν::Vector{Float64}` point of evaluation for the polyconvex hull
`returnDerivs::Bool` return first order derivative information
"""
function convexify(poly_convexification::PolyConvexification, poly_buffer::PolyConvexificationBuffer, Φ::FUN, ν::Union{Vec{d},Vector{Float64}}, xargs::Vararg{Any,XN}; returnDerivs::Bool=true) where {FUN,XN,d}
    ν_δ = poly_convexification.grid
    mν_δ = poly_convexification.liftedGrid

    if length(ν) != poly_convexification.dimp
        display("dimension missmatch")
    end

    poly_buffer.Φν_δ[1:end] = Φ.(ν_δ, xargs...)
    poly_buffer.Φactive[1:end] = poly_buffer.Φν_δ .< Inf

    # delete points from the grid where Φ attends infinity (by the active bool vector)
    mν_δ, Φν_δ = mν_δ[poly_buffer.Φactive], poly_buffer.Φν_δ[poly_buffer.Φactive]
    nΦν_δ = count(!=(0), poly_buffer.Φactive)

    # set up optimization model
    model = Model(HiGHS.Optimizer)
    set_attribute(model, "presolve", "on")
    set_attribute(model, "time_limit", 60.0)
    set_silent(model) # set output silent
    # solution variable
    @variable(model, 1 >= x[1:nΦν_δ] >= 0)
    @objective(model, Min, x' * Φν_δ)
    # constraints
    A = stack([ones(nΦν_δ)'; hcat(mν_δ...)])
    b = stack([1; minors(ν)])
    con = @constraint(model, A * x .== b)
    optimize!(model)

    if returnDerivs
        # first order derivative information
        # DΦpc(ν) = Dminors(ν) * λ (Lagrange Multiplier associated to the optimization problem)
        return objective_value(model), Dminors(ν) * dual.(con)[2:end], zero(Tensor{3, poly_convexification.dimp})
    else
        return objective_value(model)
    end
end


@doc raw"""
Signed singular value polyconvexification using the linear programming approach

takes dxd matrix `F` and function `W`$: \mathbb{R}^{d \times d} \to \mathbb{R}$  (isotropic)
"""
function convexify(poly_convexification::PolyConvexification, poly_buffer::PolyConvexificationBuffer, W::FUN, F::Union{Matrix{T},SMatrix{dim,dim,T},Tensor{2,dim,T}}, xargs::Vararg{Any,XN}; returnDerivs::Bool=true) where {FUN,XN,dim,T}
    d = poly_convexification.dimp
    ν = ssv(F)
    Φ = (x,xargs...) -> W(diagm(x), xargs...)  # TODO: optimize xargs treatment
    if returnDerivs
        Φpcνδ, DΦpcνδ, _ = convexify(poly_convexification, poly_buffer, Φ, ν::Vector{Float64}, xargs...; returnDerivs)
        WpcFδ = Φpcνδ
        DWpcFδ = Tensor{3,d}(Dssv(F)) ⋅ Vec{d}(DΦpcνδ)
        return WpcFδ, DWpcFδ, zero(Tensor{4, d})
    else
        Φpcνδ = convexify(poly_convexification, poly_buffer, Φ, ν::Vector{Float64}, xargs...; returnDerivs)
        WpcFδ = Φpcνδ
        return WpcFδ
    end
end

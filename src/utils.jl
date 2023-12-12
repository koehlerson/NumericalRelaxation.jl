"""
    ConvexificationBuffer1D{T1,T2} <: ConvexificationBuffer

# Constructor
- `build_buffer` is the unified constructor for all `ConvexificationBuffer`
# Fields
- `grid::T1` holds the deformation gradient grid
- `values::T2` holds the incremental stress potential values W(F)
"""
struct ConvexificationBuffer1D{T1,T2} <: AbstractConvexificationBuffer
    grid::Vector{T1}
    values::Vector{T2}
end

"""
    AdaptiveConvexificationBuffer1D{T1,T2,T3} <: ConvexificationBuffer

# Constructor
- `build_buffer` is the unified constructor for all `ConvexificationBuffer`
# Fields
- `basebuffer::ConvexificationBuffer{T1,T2}` holds the coarse grid buffers
- `adaptivebuffer::ConvexificationBuffer{T1,T2}` holds the adapted grid buffers
- `basegrid_∂²W::T3` second derivative information on coarse grid
"""
struct AdaptiveConvexificationBuffer1D{T1,T2,T3} <: AbstractConvexificationBuffer
    basebuffer::ConvexificationBuffer1D{T1,T2}
    adaptivebuffer::ConvexificationBuffer1D{T1,T2}
    basegrid_∂²W::Vector{T3}
end

# type piracy
Base.isless(a::Tensors.Tensor{2,1,T,1}, b::Tensors.Tensor{2,1,T,1}) where T = a[1] < b[1]
Base.isless(a::Tensors.Tensor{4,1,T,1}, b::Tensors.Tensor{4,1,T,1}) where T = a[1] < b[1]
Base.:/(x::T,y::Tensor{order,1}) where {T<:Number, order} = x/y[1]
Base.:+(x::T,y::Tensor{order,1}) where {T<:Number, order} = x + y[1]
Tensors.Tensor{order,1,T,1}(x::T) where {order,T} = Tensor{order,1}((x,))

####################################################
####################################################
############### Multidimensional  ##################
####################################################
####################################################

struct BALTBuffer{T1,T2} <: AbstractConvexificationBuffer
    #concat array for rank-one line convexification input
    initial::ConvexificationBuffer1D{T1,T2}
    #output list of rank-one line convexification
    convex::ConvexificationBuffer1D{T1,T2}
    forward_initial::ConvexificationBuffer1D{T1,T2}
    backward_initial::ConvexificationBuffer1D{T1,T2}
    forward_convex::ConvexificationBuffer1D{T1,T2}
    backward_convex::ConvexificationBuffer1D{T1,T2}
end

function fill!(baltbuffer::BALTBuffer{T1,T2}) where {T1,T2}
    Base.fill!(baltbuffer.initial.grid,zero(T1))
    Base.fill!(baltbuffer.initial.values,zero(T2))
    Base.fill!(baltbuffer.convex.grid,zero(T1))
    Base.fill!(baltbuffer.convex.values,zero(T2))
    Base.fill!(baltbuffer.forward_initial.grid,zero(T1))
    Base.fill!(baltbuffer.forward_initial.values,zero(T2))
    Base.fill!(baltbuffer.forward_convex.grid,zero(T1))
    Base.fill!(baltbuffer.forward_convex.values,zero(T2))
    Base.fill!(baltbuffer.backward_initial.grid,zero(T1))
    Base.fill!(baltbuffer.backward_initial.values,zero(T2))
    Base.fill!(baltbuffer.backward_convex.grid,zero(T1))
    Base.fill!(baltbuffer.backward_convex.values,zero(T2))
end

struct Laminate{dim,T,N}
    F⁺::Tensor{2,dim,T,N}
    F⁻::Tensor{2,dim,T,N}
    A::Tensor{2,dim,T,N}
    W⁺::T
    W⁻::T
    k::Int
    function Laminate(F₁::Tensor{2,dim,T,N}, F₂::Tensor{2,dim,T,N}, W₁::T, W₂::T, A::Tensor{2,dim,T,N}, k::Int) where {dim,T,N}
        F⁺ = zero(Tensor{2,dim})
        F⁻ = zero(Tensor{2,dim})
        W⁺ = zero(W₁)
        W⁻ = zero(W₂)
        if W₁ > W₂ #TODO fix me
            # F₁ > F₂ -> 1 = + und 2 = -
            F⁺ = F₁
            W⁺ = W₁
            F⁻ = F₂
            W⁻ = W₂
        else #1 = - und 2 = +
            F⁺ = F₂
            W⁺ = W₂
            F⁻ = F₁
            W⁻ = W₁
        end
        new{dim,T,N}(F⁺, F⁻, A, W⁺, W⁻, k)
    end
end

function Base.show(io::IO, ::MIME"text/plain", laminate::Laminate{dim}) where {dim}
    println(io,"$(dim)D Laminate: F⁻=$(laminate.F⁻),F⁺=$(laminate.F⁺), level=$(laminate.k)")
end

struct LaminateTree{dim,T,N}
    laminates::Vector{Laminate{dim,T,N}}
end

function Base.show(io::IO, ::MIME"text/plain", laminatetree::LaminateTree{dim}) where dim
    println("$(dim)D Laminatetree")
    for laminate in laminatetree.laminates
        println(io,"\t"^(laminate.k-1)* "k=$(laminate.k) "* "F⁻=$(laminate.F⁻), F⁺=$(laminate.F⁺)")
    end
end

function Base.isassigned(laminatetree::LaminateTree,level::Int)
    idx = findfirst(x->x.k==level,laminatetree.laminates)
    return isa(idx,Int)
end

function Base.setindex!(laminatetree::LaminateTree,x,level)
    idx = findfirst(x->x.k==level,laminatetree.laminates)
    laminatetree.laminates[idx] = x
end

function Base.getindex(laminatetree::LaminateTree,level)
    idx = findfirst(x->x.k==level,laminatetree.laminates)
    return laminatetree.laminates[idx]
end

Base.lastindex(laminatetree::LaminateTree) = maximum(getproperty.(laminatetree.laminates,:k))

struct R1ConvexificationThreadBuffer{T1,T2,dimp,N}
    g_fw::Vector{T1}
    g_bw::Vector{T1}
    X_fw::Vector{T2}
    X_bw::Vector{T2}
    g::Vector{T1}
    X::Vector{T2}
    h::Vector{T1}
    y::Vector{T2}
    partiallaminatetree::Dict{Int,LaminateTree{dimp,T1,N}}
end

function R1ConvexificationThreadBuffer(dim::Int,max_gx::Int,T1=Float64,T2=Int)
    g_fw = zeros(T1,max_gx);  g_bw = zeros(T1,max_gx)
    X_fw = zeros(T2,max_gx);  X_bw = zeros(T2,max_gx)
    X = zeros(T2,max_gx);     g = zeros(T1,max_gx)
    y = zeros(T2,max_gx);     h = zeros(T1,max_gx)
    partiallaminatetree = Dict{Int,LaminateTree{dim,T1,dim^2}}()
    return R1ConvexificationThreadBuffer(g_fw, g_bw, X_fw, X_bw, g, X, h, y, partiallaminatetree)
end

struct R1ConvexificationBuffer{T1,T2,dimp,dimc,N,ITP<:Interpolations.AbstractInterpolation} <: AbstractConvexificationBuffer
    threadbuffer::Vector{R1ConvexificationThreadBuffer{T1,T2,dimp,N}}
    W_rk1::ITP
    W_rk1_old::ITP
    diff::Array{T1,dimc}
    laminatetree::Dict{Int,LaminateTree{dimp,T1,dimc}}
end

"""Helper utils function to concat two arrays in a pretty specific way"""
function concat!(conc_list, a_fw, s_fw, a_bw, s_bw)
    ctr = 1
    for j in s_bw:-1:1
        conc_list[ctr] = a_bw[j]
        ctr += 1
    end
    for j in 1:1:s_fw-1
        conc_list[ctr] = a_fw[j]
        ctr += 1
    end
    return conc_list
end

function concat!(buffer::BALTBuffer, ctr_fw::Int, ctr_bw::Int)
    concat!(buffer.initial.grid,buffer.forward_initial.grid,ctr_fw,buffer.backward_initial.grid,ctr_bw)
    concat!(buffer.initial.values,buffer.forward_initial.values,ctr_fw,buffer.backward_initial.values,ctr_bw)
    return buffer
end

####################################################
####################################################
############ Signed singular values  ###############
####################################################
####################################################

@doc raw"""
singular values computed by the eigenvalues of the right cauchy green tensor
"""
function sv(F)
    return sqrt.(eigvals(F' * F))
end

@doc raw"""
signed singular values
"""
function ssv(F)
    return sqrt.(eigvals(F' * F)) .* [sign(det(F)), ones(size(F)[1] - 1)...]
end

@doc raw"""
Derivative of signed singular values mapping, calculated by forward difference quotients.
TODO: might need improvement
"""
function Dssv(F; eps=1e-8)
    return Tensor{3,size(F)[1]}((i, j, k) -> (ssv(F + sparse([i], [j], [eps], size(F)...)) - ssv(F))[k] / eps)
end

####################################################
####################################################
#################### Minors  #######################
####################################################
####################################################
@doc raw"""
minors for matrices and vectors
"""
function minors(A::Matrix{Float64})
    if size(A)[1] == 2
        return [A..., det(A)]
    elseif size(A)[1] == 3
        return [A..., det(A) * inv(A)..., det(A)]
    end
end

function minors(A::Union{Tensor{2,d,Float64},SMatrix{d,d,Float64}}) where {d}
    if d == 2
        return SVector{5}([A..., det(A)])
    elseif d == 3
        return SVector{19}([A..., det(A) * inv(A)..., det(A)])
    end
end

function minors(ν::Vector{Float64})
    if length(ν) == 2
        return [ν[1], ν[2], ν[1]*ν[2]]
    elseif length(ν) == 3
        return [ν[1], ν[2], ν[3], ν[2]*ν[3], ν[1]*ν[3], ν[1]*ν[2], ν[1]*ν[2]*ν[3]]
    end
end

function minors(ν::Union{Vec{d,T},SVector{d, T}}) where {d,T}
    if d == 2
        return SVector{3}([ν[1], ν[2], ν[1]*ν[2]])
    elseif d == 3
        return SVector{7}([ν[1], ν[2], ν[3], ν[2]*ν[3], ν[1]*ν[3], ν[1]*ν[2], ν[1]*ν[2]*ν[3]])
    end
end


@doc raw"""
Derivative of the minors function
    Dminors$:\mathbb{R}^{d} \to \mathbb{R}^{d \times k_d}$
with $k_d$ denoting the lifted dimension
"""
function Dminors(ν::Vector{Float64})
    if length(ν) == 2
        return [1 0 ν[2]; 0 1 ν[1]]
    elseif length(ν) == 3
        return [1 0 0 0    ν[3] ν[2] ν[2]*ν[3];
                0 1 0 ν[3] 0    ν[1] ν[1]*ν[3];
                0 0 1 ν[2] ν[1] 0    ν[1]*ν[2]]
    end
end

function Dminors(ν::Union{Vec{d,T},SVector{d, T}}) where {d,T}
    if d == 2
        return SMatrix{2, 3}([1 0 ν[2]; 0 1 ν[1]])
    elseif d == 3
        return SMatrix{3, 7}([1 0 0 0    ν[3] ν[2] ν[2]*ν[3];
                              0 1 0 ν[3] 0    ν[1] ν[1]*ν[3];
                              0 0 1 ν[2] ν[1] 0    ν[1]*ν[2]])
    end
end

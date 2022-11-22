"""
    ConvexificationBuffer1D{T1,T2} <: ConvexificationBuffer

# Constructor
- `build_buffer` is the unified constructor for all `ConvexificationBuffer`
# Fields
- `grid::T1` holds the deformation gradient grid
- `values::T2` holds the incremental stress potential values W(F)
"""
struct ConvexificationBuffer1D{T1,T2} <: AbstractConvexificationBuffer
    grid::T1
    values::T2
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
    basegrid_∂²W::T3
end

# type piracy
Base.isless(a::Tensors.Tensor{2,1,T,1}, b::Tensors.Tensor{2,1,T,1}) where T = a[1] < b[1]
Base.isless(a::Tensors.Tensor{4,1,T,1}, b::Tensors.Tensor{4,1,T,1}) where T = a[1] < b[1]

####################################################
####################################################
############### Multidimensional  ##################
####################################################
####################################################

struct ConvexificationThreadBuffer{T1,T2,dimp,N} <: AbstractConvexificationBuffer
    g_fw::Vector{T1}
    g_bw::Vector{T1}
    X_fw::Vector{T2}
    X_bw::Vector{T2}
    g::Vector{T1}
    X::Vector{T2}
    h::Vector{T1}
    y::Vector{T2}
    partialphasestree::Dict{Tensor{2,dimp,T1,N},PhaseTree{dimp,T1,N}}
end

function ConvexificationThreadBuffer(dim::Int,max_gx::Int,T1=Float64,T2=Int)
    g_fw = zeros(T1,max_gx);  g_bw = zeros(T1,max_gx)
    X_fw = zeros(T2,max_gx);  X_bw = zeros(T2,max_gx)
    X = zeros(T2,max_gx);     g = zeros(T1,max_gx)
    y = zeros(T2,max_gx);     h = zeros(T1,max_gx)
    partialphasestree = Dict{Tensor{2,dim,T1,dim^2},PhaseTree{dim,T1,dim^2}}()
    return ConvexificationThreadBuffer(g_fw, g_bw, X_fw, X_bw, g, X, h, y, partialphasestree)
end

"""Helper utils function to concat two arrays in a pretty specific way"""
function concat!(conc_list, a_fw, s_fw, a_bw, s_bw)
    ctr = 1
    for j in s_bw:-1:1
        conc_list[ctr] = a_bw[j]
        ctr += 1
    end
    for j in 1:1:s_fw
        conc_list[ctr] = a_fw[j]
        ctr += 1
    end
    return conc_list
end

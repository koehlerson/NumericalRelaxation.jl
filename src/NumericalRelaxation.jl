module NumericalRelaxation

using Tensors
using AbstractTrees
using Interpolations
using LinearAlgebra

abstract type AbstractConvexification end
abstract type AbstractConvexificationBuffer end

include("utils.jl")
include("convexify.jl")
include("derivative.jl")
include("export.jl")

end

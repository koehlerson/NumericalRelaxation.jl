module NumericalRelaxation

using Tensors

abstract type AbstractConvexification end
abstract type AbstractConvexificationBuffer end

include("discretizations.jl")
include("utils.jl")
include("convexify.jl")

end

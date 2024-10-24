module NumericalRelaxation

using Tensors
using StaticArrays
using SparseArrays
using AbstractTrees
using Interpolations
using LinearAlgebra
using JuMP, HiGHS
import JLD2

abstract type AbstractConvexification end
abstract type AbstractConvexificationBuffer end

include("utils.jl")
include("convexify.jl")
include("derivative.jl")
include("export.jl")

end

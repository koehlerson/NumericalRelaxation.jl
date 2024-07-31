# Multi-Dimensional Signed Singular Value Polyconvexification

## API

```@docs
NumericalRelaxation.PolyConvexification
NumericalRelaxation.PolyConvexificationBuffer
NumericalRelaxation.convexify(poly_convexification::PolyConvexification, poly_buffer::PolyConvexificationBuffer, Φ::FUN, ν::Union{Vec{d},Vector{Float64}}, xargs::Vararg{Any,XN}; returnDerivs::Bool) where {FUN,XN,d}
NumericalRelaxation.ssv
NumericalRelaxation.Dssv
NumericalRelaxation.minors
NumericalRelaxation.Dminors
```

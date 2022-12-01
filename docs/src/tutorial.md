# Tutorial

The main step is to construct one of the possible convexification strategies and an associated buffer:

```julia
convexification = AdaptiveGrahamScan()
buffer = build_buffer(convexification)
```


## One-Dimensional Example

In the one-dimensional case it is sufficient to provide the convex hull supporting points $F^+$ and $F^-$ as well as the value of the convex envelope $W^C(F)$.
With a convexification strategy and a buffer at hand this can be realized by e.g.

```julia
using NumericalRelaxation, Tensors

convexification = GrahamScan(start=0.01,stop=5.0,δ=0.01)
buffer = build_buffer(convexification)

W(F::Number,x1=1,x2=42) = (F-x1)^2 * (x2 + 77*F + 15*F^2 - 102.5*F^3 + 58.89*F^4 - 12.89*F^5 + F^6)
W(F::Tensor{2,1},x1=1,x2=42) = W(F[1],x1,x2)

W_conv, F⁺, F⁻ = convexify(convexification,buffer,W,Tensor{2,1}((2.0,)))
```

Note that the package assumes a `W` that can handle [`Tensors.jl`](https://github.com/Ferrite-FEM/Tensors.jl) objects.
Further, the function `W` can take arbitrary much arguments that are propagated.

```julia
W_conv, F⁺, F⁻ = convexify(convexification,buffer,W,Tensor{2,1}((2.0,)),1.5,43) #x1 =1.5, x2=43 for the convexification
```

## Multi-Dimensional Example

The rank-one convexification can be realized quite similarly.
However, a triplet of the value, and the two gradients is not sufficient.
Therefore, the complete rank-one approximation in terms of the linear interpolation object and (optionally) lamination forrest are stored in the buffer.

```julia
W_multi(F::Tensor{2,dim},x1=1) where dim = (norm(F)-x1)^2
W_multi_rc(F::Tensor{2,dim},x1=1) where dim = norm(F) ≤ x1 ? 0.0 : (norm(F)-x1)^2

d = 2
a = -2.0:0.5:2.0
r1convexification_reduced = R1Convexification(a,a,dim=d,dirtype=ParametrizedR1Directions)
buffer_reduced = build_buffer(r1convexification_reduced)
```

Here, you can specify in which spatial dimension `d` you want to do the convexification and which kind of rank-one direction set should be used.
The latter can either be `ParametrizedR1Directions` or `ℛ¹Direction`.
The first two arguments of the `R1Convexification` constructor describe the individual axis discretization, where the first argument correspond to the diagonal entries of the gradient and the second argument are the offdiagional axes.

Again, the last arguments are splatted into the `W_multi` call.
```julia
convexify!(r1convexification_reduced,buffer_reduced,W_multi,1;buildtree=true) #W_multi x1=1
```

The keyword argument specifies if the lamination tree should be saved.

To test if the procedure approaches the rank-one convex envelope pointwise, you can do:

```julia
julia> all(isapprox.(buffer_reduced.W_rk1.itp.itp.coefs .- [W_multi_rc(Tensor{2,2}((x1,x2,y1,y2))) for x1 in a, x2 in a, y1 in a, y2 in a],0.0,atol=1e-8))
true
```

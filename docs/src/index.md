# NumericalRelaxation.jl

This package is an outcome of the research of the [Chair of Mechanics - Continuum Mechanics Ruhr-University Bochum](http://www.lkm.rub.de/index.html.en) in collaboration with [Chair of Computational Mathematics - University of Augsburg](https://www.uni-augsburg.de/en/fakultaet/mntf/math/prof/numa/).

The goal of this project is the (semi-)convexification of some generalized energy density function $W$, which often becomes non-convex in dissipative continuum mechanics or non-linear elasticity.
In both cases per incremental time step a potential should become stationary

```math
\Pi = \int_{\mathcal{B}} W(\boldsymbol{F}) \ dV - \Pi^{\text{ext}} \rightarrow \text{stationary}.
```
with the corresponding Euler-Lagrange equation

```math
\delta \Pi = \int_{\mathcal{B}} \delta \boldsymbol{F} \cdot \partial_{\boldsymbol{F}} W(\boldsymbol{F}) \ dV - \Pi^{\text{ext}} = 0.
```

The solution to it only possesses minimizers if $W$ is convex in the one-dimensional case and polyconvex in the multi-dimensional case.
This package implements currently three different (semi)convexification procedures.
Two methods implement the convexification in the one-dimensional case and one method for the rank-one convexification in the multi-dimensional case.

- [`GrahamScan`](@ref)
- [`AdaptiveGrahamScan`](@ref)

for one-dimensional convexification and

- [`R1Convexification`](@ref)
- [`HROC`](@ref)

for the rank-one convexification.

This package only provides the (semi)convex envelope of $W$, the minimization by means of finite elements must be realized by e.g. [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl).

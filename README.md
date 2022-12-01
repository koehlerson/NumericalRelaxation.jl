# NumericalRelaxation.jl

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/koehlerson/NumericalRelaxation.jl/blob/main/docs/src/assets/logo-horizontal-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/koehlerson/NumericalRelaxation.jl/blob/main/docs/src/assets/logo-horizontal-dark.svg">
  <img alt="NumiercalRelaxation.jl logo." src="https://github.com/koehlerson/NumericalRelaxation.jl/blob/master/docs/src/assets/logo-horizontal-light.svg">
</picture>

![Build Status](https://github.com/koehlerson/NumericalRelaxation.jl/workflows/CI/badge.svg?event=push)
[![][docs-dev-img]][docs-dev-url]


This package implements various convexification algorithms that can be used for the numerical relaxation of variational problems.

Install the package with

```
pkg> add https://github.com/koehlerson/NumericalRelaxation.jl
```

The implemented methods are research results from the following papers:

1. [Adaptive convexification of microsphere-based incremental damage for stress and strain softening at finite strains](https://link.springer.com/article/10.1007/s00707-022-03332-1)
2. [Multidimensional rank-one convexification of incremental damage models at finite strains](https://arxiv.org/abs/2211.14318)

If you use one-dimensional convexification procedures, please cite the prior.
In case you want to use rank-one convexification refer to the second paper.

## Acknowledgement

This code is a result of the [SPP 2256 Project Group 2](https://spp2256.ur.de/research/members-2020/2023).
We acknowledge the Deutsche Forschungsgemeinschaft (DFG) for funding within the Priority
Program 2256 ("Variational Methods for Predicting Complex Phenomena in Engineering Structures and Materials"),
project ID 441154176, reference ID BA2823/17-1.

![spp-logo](docs/src/assets/spp-logo.png)

[docs-dev-img]: https://img.shields.io/badge/docs-latest%20release-blue
[docs-dev-url]: http://koehlerson.github.io/NumericalRelaxation.jl/dev/

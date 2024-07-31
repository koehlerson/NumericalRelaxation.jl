using Documenter, NumericalRelaxation
import Tensors

makedocs(sitename="NumericalRelaxation.jl",
         modules=[NumericalRelaxation],
         authors="Maximilian Köhler",
         warnonly=true,
         pages=["Home"=> "index.md",
                "Tutorial" => "tutorial.md",
                "One-Dimensional Convexification" => "api/oned-convexification.md",
                "Rank-One Convexification" => "api/r1convexification.md",
                "Polyconvexification" => "api/polyconvexification.md"])
deploydocs(
    repo = "github.com/koehlerson/NumericalRelaxation.jl.git",
    push_preview=true,
)

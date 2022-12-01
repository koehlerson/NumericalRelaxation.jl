var documenterSearchIndex = {"docs":
[{"location":"api/r1convexification/#Multi-Dimensional-Convexification","page":"Rank-One Convexification","title":"Multi-Dimensional Convexification","text":"","category":"section"},{"location":"api/r1convexification/#API","page":"Rank-One Convexification","title":"API","text":"","category":"section"},{"location":"api/r1convexification/","page":"Rank-One Convexification","title":"Rank-One Convexification","text":"NumericalRelaxation.R1Convexification\nNumericalRelaxation.GradientGrid\nNumericalRelaxation.ParametrizedR1Directions\nNumericalRelaxation.ℛ¹Direction\nNumericalRelaxation.ℛ¹DirectionBuffered\nNumericalRelaxation.GradientGridBuffered\nNumericalRelaxation.convexify!","category":"page"},{"location":"api/r1convexification/#NumericalRelaxation.R1Convexification","page":"Rank-One Convexification","title":"NumericalRelaxation.R1Convexification","text":"R1Convexification{dimp,dimc,dirtype<:RankOneDirections{dimp},T1,T2,R} <: Convexification\n\nDatastructure that is used for the actual material struct as an equivalent to GrahamScan in the multidimensional relaxation setting. Bundles parallelization buffers, rank-one direction discretization as well as a tolerance and the convexification grid.\n\nConstructor\n\nR1Convexification(axes_diag::AbstractRange,axes_off::AbstractRange;dirtype=ℛ¹Direction,dim=2,tol=1e-4)\n\nFields\n\ngrid::GradientGrid{dimc,T1,R}\ndirs::dirtype\ntol::T1\n\n\n\n\n\n","category":"type"},{"location":"api/r1convexification/#NumericalRelaxation.GradientGrid","page":"Rank-One Convexification","title":"NumericalRelaxation.GradientGrid","text":"GradientGrid{dimc,T,R<:AbstractRange{T}}\n\nLightweight implementation of a structured convexification grid in multiple dimensions. Computes the requested convexification grid node adhoc and therefore is especially suited for threading (no cache misses). Implements the Base.Iterator interface and other Base functions such as, length,size,getindex,lastindex,firstindex,eltype,axes Within the parameterization dimc denote the convexification dimensions, T the used number type and R the number type of the start, step and end value of the axes ranges.\n\nConstructor\n\nGradientGrid(axes::NTuple{dimc,R}) where dimc\n\naxes::R is a tuple of discretizations with the order of Tensor{2,2}([x1 y1;x2 y2])\n\nFields\n\naxes::NTuple{dimc,R}\nindices::CartesianIndices{dimc,NTuple{dimc,Base.OneTo{Int64}}}\n\n\n\n\n\n","category":"type"},{"location":"api/r1convexification/#NumericalRelaxation.ParametrizedR1Directions","page":"Rank-One Convexification","title":"NumericalRelaxation.ParametrizedR1Directions","text":"ParametrizedR1Directions{dimp,T} <: RankOneDirections{dimp}\n\nDirection datastructure that computes the reduced rank-one directions in the dimp physical dimensions and dimp² convexification dimensions. Implements the Base.Iterator interface and other utility functions. This datastructure only computes the first neighborhood rank-one directions and utilizes the symmetry of the dimp² dimensionality. Since they are quite small in 2² and 3² the directions are precomputed and stored in dirs.\n\nConstructor\n\nParametrizedR1Directions(2)\nParametrizedR1Directions(3)\n\nFields\n\ndirs::Vector{Tuple{Vec{dimp,T},Vec{dimp,T}}}\n\n\n\n\n\n","category":"type"},{"location":"api/r1convexification/#NumericalRelaxation.ℛ¹Direction","page":"Rank-One Convexification","title":"NumericalRelaxation.ℛ¹Direction","text":"ℛ¹Direction{dimp,dimc} <: RankOneDirections{dimp}\n\nLightweight implementation that computes all rank-one directions within a grid::GradientGrid adhoc. Therefore, also suited for threading purposes, since this avoids cache misses. Implements the Base.Iterator interface and other utility functions. Within the parameterization dimp and dimc denote the physical dimensions of the problem and the convexification dimensions, respectively.\n\nConstructor\n\nℛ¹Direction(b::GradientGrid)\n\nb::GradientGrid is a deformation grid discretization\n\nFields\n\na_axes::NTuple{dimp,UnitRange{Int}}\nb_axes::NTuple{dimp,UnitRange{Int}}\nindices::CartesianIndices{dimc,NTuple{dimc,Base.OneTo{Int64}}}\n\n\n\n\n\n","category":"type"},{"location":"api/r1convexification/#NumericalRelaxation.ℛ¹DirectionBuffered","page":"Rank-One Convexification","title":"NumericalRelaxation.ℛ¹DirectionBuffered","text":"ℛ¹DirectionBuffered{dimp,dimc,T} <: RankOneDirections{dimp}\n\nHeavyweight implementation that computes all rank-one directions within a grid::GradientGridBuffered within the constructor. Implements the Base.Iterator interface and other utility functions. Within the parameterization dimp and dimc denote the physical dimensions of the problem and the convexification dimensions, respectively.\n\nConstructor\n\nℛ¹DirectionBuffered(dirs::ℛ¹Direction)\n\ndirs::ℛ¹Direction collects the reference dirs direction and caches them in the grid field\n\nFields\n\ngrid::AbstractArray{Tuple{Vec{dimp,T},Vec{dimp,T}},dimc}\n\n\n\n\n\n","category":"type"},{"location":"api/r1convexification/#NumericalRelaxation.GradientGridBuffered","page":"Rank-One Convexification","title":"NumericalRelaxation.GradientGridBuffered","text":"GradientGridBuffered{dimc,T,dimp}\n\nHeavyweight implementation of a structured convexification grid in multiple dimensions. Computes the requested convexification grid within the constructor and only accesses thereafter the grid field. Implements the Base.Iterator interface and other Base functions such as, length,size,getindex,lastindex,firstindex,eltype,axes Within the parameterization dimc denote the convexification dimensions, T the used number type and dimp the physical dimensions of the problem.\n\nConstructor\n\nGradientGridBuffered(axes::NTuple{dimc}) where dimc\n\naxes::StepRangeLen{T,R,R} is a tuple of discretizations with the order of Tensor{2,2}([x1 y1;x2 y2]\n\nFields\n\ngrid::AbstractArray{Tensor{2,dimp,T,dimc},dimc}\nindices::CartesianIndices{dimc,NTuple{dimc,Base.OneTo{Int64}}}\n\n\n\n\n\n","category":"type"},{"location":"api/r1convexification/#NumericalRelaxation.convexify!","page":"Rank-One Convexification","title":"NumericalRelaxation.convexify!","text":"convexify!(r1convexification::R1Convexification,r1buffer::R1ConvexificationBuffer,W::Function,xargs...;buildtree=true,maxk=20)\n\nMulti-dimensional parallelized implementation of the rank-one convexification. This dispatch stores the lamination tree in the convexification threading buffer and merges them after convergence to a single dictionary. Note that the interpolation objects within r1buffer are overwritten in this routine.\n\n\n\n\n\nconvexify!(f, x, ctr, h, y)\n\nRank-one line convexification algorithm in multiple dimensions without deletion, but in mathcalO(N)\n\n\n\n\n\n","category":"function"},{"location":"api/oned-convexification/#One-Dimensional-Convexification","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"","category":"section"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"This packages's focus is the (semi-)convexification of a generalized energy density W(boldsymbolF) that depends on the gradient of the solution nabla boldsymbolu.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"First, a strategy needs to be instantiated, which only holds the parameters of the convexification, such as discretization interval, size of the discretization etc. This strategy needs to be of subtype <: AbstractConvexification and dispatches build_buffer. The latter mentioned will return a buffer that is needed together with the convexification strategy to call convexify.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"NumericalRelaxation.convexify","category":"page"},{"location":"api/oned-convexification/#NumericalRelaxation.convexify","page":"One-Dimensional Convexification","title":"NumericalRelaxation.convexify","text":"convexify(graham::GrahamScan{T2}, buffer::ConvexificationBuffer1D{T1,T2}, W::Function, F, xargs...) where {T1,T2}  -> W_convex::Float64, F⁻::Tensor{2,1}, F⁺::Tensor{2,1}\n\nFunction that implements the convexification without deletion in mathcalO(N).\n\n\n\n\n\nconvexify(adaptivegraham::AdaptiveGrahamScan{T2}, buffer::AdaptiveConvexificationBuffer1D{T1,T2}, W::Function, F, xargs...) where {T1,T2}  -> W_convex::Float64, F⁻::Tensor{2,1}, F⁺::Tensor{2,1}\n\nFunction that implements the convexification without deletion in mathcalO(N).\n\n\n\n\n\n","category":"function"},{"location":"api/oned-convexification/#Equidistant-Convexificationstrategy","page":"One-Dimensional Convexification","title":"Equidistant Convexificationstrategy","text":"","category":"section"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"NumericalRelaxation.GrahamScan\nNumericalRelaxation.ConvexificationBuffer1D","category":"page"},{"location":"api/oned-convexification/#NumericalRelaxation.GrahamScan","page":"One-Dimensional Convexification","title":"NumericalRelaxation.GrahamScan","text":"GrahamScan{T<:Number} <: AbstractConvexification\n\nDatastructure that implements in convexify dispatch the discrete one-dimensional convexification of a line with actual deletion of memory. This results in a complexity of mathcalO(N). However, with additional costs due to the memory delete process.\n\nKwargs\n\nδ::T = 0.01\nstart::T = 0.9\nstop::T = 20.0\n\n\n\n\n\n","category":"type"},{"location":"api/oned-convexification/#NumericalRelaxation.ConvexificationBuffer1D","page":"One-Dimensional Convexification","title":"NumericalRelaxation.ConvexificationBuffer1D","text":"ConvexificationBuffer1D{T1,T2} <: ConvexificationBuffer\n\nConstructor\n\nbuild_buffer is the unified constructor for all ConvexificationBuffer\n\nFields\n\ngrid::T1 holds the deformation gradient grid\nvalues::T2 holds the incremental stress potential values W(F)\n\n\n\n\n\n","category":"type"},{"location":"api/oned-convexification/#Adaptive-Convexification","page":"One-Dimensional Convexification","title":"Adaptive Convexification","text":"","category":"section"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"NumericalRelaxation.AdaptiveGrahamScan\nNumericalRelaxation.AdaptiveConvexificationBuffer1D","category":"page"},{"location":"api/oned-convexification/#NumericalRelaxation.AdaptiveGrahamScan","page":"One-Dimensional Convexification","title":"NumericalRelaxation.AdaptiveGrahamScan","text":"    AdaptiveGrahamScan <: AbstractConvexification\n\nstruct that stores all relevant information for adaptive convexification.\n\nFields\n\ninterval::Vector{Float64}\nbasegrid_numpoints::Int64\nadaptivegrid_numpoints::Int64\nexponent::Int64\ndistribution::String\nstepSizeIgnoreHessian::Float64\nminPointsPerInterval::Int64\nradius::Float64\nminStepSize::Float64\nforceAdaptivity::Bool\n\nConstructor\n\nAdaptiveGrahamScan(interval; basegrid_numpoints=50, adaptivegrid_numpoints=115, exponent=5, distribution=\"fix\", stepSizeIgnoreHessian=0.05, minPointsPerInterval=15, radius=3, minStepSize=0.03, forceAdaptivity=false)\n\n\n\n\n\n","category":"type"},{"location":"api/oned-convexification/#NumericalRelaxation.AdaptiveConvexificationBuffer1D","page":"One-Dimensional Convexification","title":"NumericalRelaxation.AdaptiveConvexificationBuffer1D","text":"AdaptiveConvexificationBuffer1D{T1,T2,T3} <: ConvexificationBuffer\n\nConstructor\n\nbuild_buffer is the unified constructor for all ConvexificationBuffer\n\nFields\n\nbasebuffer::ConvexificationBuffer{T1,T2} holds the coarse grid buffers\nadaptivebuffer::ConvexificationBuffer{T1,T2} holds the adapted grid buffers\nbasegrid_∂²W::T3 second derivative information on coarse grid\n\n\n\n\n\n","category":"type"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"To reduce the absolute computational cost of the convexification algorithm it is essential to represent the incremental stress potential W(boldsymbolF) on as less grid points as possible. For accuracy reasons of the resulting convex hull, the resolution of the grid must be fairly high around the supporting points F^- and F^+ of non-convex regions of W. In adaptive convexification we first identify a set of points of interest based on a coarse grid to construct an adaptive grid in a second step. The adaptive grid then has a high resolution around the previously identified points and a coarse resolution everywhere inbetween. An example of the coarse and the adaptive grid can be seen in the figure below.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"(Image: Image)","category":"page"},{"location":"api/oned-convexification/#How-to-use-it","page":"One-Dimensional Convexification","title":"How to use it","text":"","category":"section"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"Simply use an instance of AdaptiveGrahamScan and build a buffer with buildbuffer. The convexify function will then dispatch on the assigned strategy.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"adaptiveconvexification = AdaptiveGrahamScan(interval=[0.001, 20.001])\nbuffer = build_buffer(adaptiveconvexification)\nW, F⁺, F⁻ = convexify(adaptiveconvexification,buffer,W,Tensor{2,1}((2.0,)))","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"interval::Vector{Float64}","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"Specifies here the discretization interval for the convexification grid. In some cases default settings for adaptive convexification will not fit for the problem at hand. In this case the default values for the keyword arguments","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"basegrid_numpoints::Int64 = 50\nadaptivegrid_numpoints::Int64 = 1150\nexponent::Int64 = 5\ndistribution::String = \"fix\"\nstepSizeIgnoreHessian::Float64 = 0.05\nminPointsPerInterval::Int64 = 15\nradius::Float64 = 3\nminStepSize::Float64 = 0.03\nforceAdaptivity::Bool = false","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"of struct AdaptiveGrahamScan must be altered.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"In general, select an interval such that it covers all non-convex parts of W(boldsymbolF). basegrid_numpoints and convexgrid_numpoints must be chosen large enough to represent all relevant information of W(boldsymbolF) but small enough to keep computational cost low.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"To understand how to set the parameters minStepSize, radius and exponent, that mainly characterize the distribution of the grid points on a given subinterval F^-F+ (subinterval between two previously identified points of interest) of the main interval, correctly, we need to understand how the distribution of the grid points works mathematically.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"For each of the two distribution types there exists a function that maps a vector entry j to to a point on the given subinterval. This function is a piecewisely defined polynomial of degree p. a, b, c, d, j^-_r and j^+_r are parameters that are automatically fitted for a specific problem. j_max is the number of gridpoints to be distributed on a given subintervall.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"beginalign*\nvar mathbbR  longrightarrow mathbbR \nj  longmapsto\nleftbeginarrayll F^- + a  j^p + b  j   textif  j  fracj_max2 \nF^+ - left(a (j_max - j)^p + b (j_max - j) right)   textif  j geq fracj_max2 endarray right\n endalign*","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"beginalign*\nfix mathbbR  longrightarrow mathbbR \nj  longmapsto\nleftbeginarrayll F^- + a  j^p + b  j   textif  j  j^-_r \nc  j + d  textif  j^-_r leq j leq j^+_r \nF^+ - left(a (j_max - j)^p + b (j_max - j) right)   textif  j  j^+_r endarray right\n endalign*","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"For distribution type \"var\" the function constists of two regions. A polynomial of  degree p on the first half of the interval 0 j_max and a mirrored version  of it on the second half of it. Distribution type \"fix\" is an extension of type \"var\"  that adds a linear part to the middle of the interval. See figures below for examples  polynomials of type \"fix\".","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"Unless the lengths of all subintervals are almost equal,  one should not use distribution type \"var\", since it has the drawback that it results  in an adaptive grid where the step size between two grid points depends heavily on the  length (F^+-F^-) of the subinterval. This way the resulting grid can have a drastically  different resolution on either side of a point of interest.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"Always set the parameters of your grid in the following order:","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"minStepSize","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"Defines the slope of the polinomial at its start/end point. For j_max rightarrow infty  the step size of the interval at its start/end point will converge to this value. Should  be the highest value that still just serves the accuracy requirements.  Try 0.0015*length_of_interval as a starting value.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"radius (only for type \"fix\")","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"Sets the radius Delta F around a point of interest in which the step size increases  to the maximum value. In terms of the mathematical definition of fix this value is used  to set the parameters j^-_r and j^+_r, which are the vector indices that define  the transitions between linear and non-linear parts. See the figure above for the influence of  this parameter on the resulting polynomial. Also check the figure at the beginning of this  section to see the resulting adaptive grid. You will notice that the step size of the grid  increases equally and within a constant radius around all points of interest.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"exponent","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"Sets the exponential \"p\" of the polynomial. It mainly influences the difference between  highest and lowest step size. Or in other words, for increasing polynomial degrees the grid  points will be pushed further towards the start and end points of the interval.  See figure above for the influence of this parameter on the resulting polynomial.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"convexgrid_numpoints","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"The number of grid points of the adaptive grid must now be chosen intuitively.  As a starting value choose 20 points per subinterval.","category":"page"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"(Image: Image) (Image: Image)","category":"page"},{"location":"api/oned-convexification/#Extended-Info","page":"One-Dimensional Convexification","title":"Extended Info","text":"","category":"section"},{"location":"api/oned-convexification/","page":"One-Dimensional Convexification","title":"One-Dimensional Convexification","text":"NumericalRelaxation.adaptive_1Dgrid!\nNumericalRelaxation.build_buffer\nNumericalRelaxation.convexify_nondeleting!","category":"page"},{"location":"api/oned-convexification/#NumericalRelaxation.adaptive_1Dgrid!","page":"One-Dimensional Convexification","title":"NumericalRelaxation.adaptive_1Dgrid!","text":"    function adaptive_1Dgrid!(ac::AdaptiveGrahamScan, ac_buffer::AdaptiveConvexificationBuffer1D{T1,T2,T3}) where {T1,T2,T3}\n        ...\n        return  F⁺⁻\n    end\n\nBased on any grid ac_buffer.basebuffer.grid and coresponding function values ac_buffer.basebuffer.values and  its second derivative ac_buffer.basegrid_∂²W, a set of points of interest F⁺⁻ will be determined. Based on this set of points and different parameters stored in ac an adaptive grid will be constructed such that grid resolution is highest at these points.\n\nThe resultiong grid will be broadcasted into ac_buffer.adaptivebuffer.grid.\n\nF⁺⁻ will be determined by checking the slope of mathematical function W(F). Start and end points of non-convex subintervals will be stored. Additionally all minima of ∂²W(F) serve as points of interest as well (only if step size at this point is greater than ac.stepSizeIgnoreHessian).\n\n\n\n\n\n","category":"function"},{"location":"api/oned-convexification/#NumericalRelaxation.build_buffer","page":"One-Dimensional Convexification","title":"NumericalRelaxation.build_buffer","text":"build_buffer(convexstrategy::T) where T<:AbstractConvexification\n\nMaps a given convexification strategy convexstrategy to an associated buffer.\n\n\n\n\n\n","category":"function"},{"location":"api/oned-convexification/#NumericalRelaxation.convexify_nondeleting!","page":"One-Dimensional Convexification","title":"NumericalRelaxation.convexify_nondeleting!","text":"convexify_nondeleting!(F, W)\n\nFunction that implements the convexification without deletion, but in mathcalO(N).\n\n\n\n\n\n","category":"function"},{"location":"#NumericalRelaxation.jl","page":"Home","title":"NumericalRelaxation.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package is an outcome of the research of the Chair of Mechanics - Continuum Mechanics Ruhr-University Bochum in collaboration with Chair of Computational Mathematics - University of Augsburg","category":"page"},{"location":"","page":"Home","title":"Home","text":"The goal of this project is the (semi-)convexification of some energy function W, which often becomes non-convex in dissipative continuum mechanics.","category":"page"}]
}

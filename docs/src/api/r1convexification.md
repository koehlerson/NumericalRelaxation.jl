# Multi-Dimensional Rank-one Convexification

## API

```@docs
NumericalRelaxation.R1Convexification
NumericalRelaxation.GradientGrid
NumericalRelaxation.ParametrizedR1Directions
NumericalRelaxation.ℛ¹Direction
NumericalRelaxation.ℛ¹DirectionBuffered
NumericalRelaxation.GradientGridBuffered
NumericalRelaxation.convexify!
NumericalRelaxation.HROC
NumericalRelaxation.convexify(hroc::HROC,buffer::HROCBuffer,W::FUN,F::T1,xargs::Vararg{Any,XN}) where {T1,FUN,XN}
NumericalRelaxation.convexify(prev_bt::BinaryLaminationTree,hroc::HROC,buffer::HROCBuffer,W::FUN,F::T1,xargs::Vararg{Any,XN}) where {T1,FUN,XN}
```

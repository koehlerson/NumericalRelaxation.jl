using NumericalRelaxation
using LinearAlgebra
using Tensors
using JLD2

W_multi(F::Tensor{2,dim}) where dim = (norm(F)-1)^2
W_multi_rc(F::Tensor{2,dim}) where dim = norm(F) ≤ 1 ? 0.0 : (norm(F)-1)^2

file = jldopen("convergence_test.jld2", "w")

discretizations = (1.0,0.5,0.25,0.1,0.05)
r1dirs = ParametrizedR1Directions(2;l=1)
a = -2.0:0.5:2.0
for δ in discretizations
    g = SingularValueGrid((-5.0:δ:5.0, -5.0:δ:5.0))
    r1convexification_svd = R1Convexification(g,r1dirs,1e-8)
    buffer_svd = build_buffer(r1convexification_svd)
    convexify!(r1convexification_svd,buffer_svd,W_multi,buildtree=false)
    diff_matrix = [buffer_svd.W_rk1(NumericalRelaxation.ssvd([x1 x2; y1 y2])...) - W_multi_rc(Tensor{2,2}((x1,x2,y1,y2))) for x1 in a, x2 in a, y1 in a, y2 in a]
    println("δ=$δ \t norm Inf = $(norm(diff_matrix,Inf)) \t norm 2 = $(norm(diff_matrix,2))")
    file["$δ"] = buffer_svd.W_rk1
end

close(file)

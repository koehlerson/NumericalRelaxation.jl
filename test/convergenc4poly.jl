# Convergence test for the numerical polyconvexification algorithm
# utilizing the two dimensional example presented in
# [1] Robert V. Kohn and Gilbert Strang. Optimal design and relaxation of variational problems, I. Communications on Pure and Applied Mathematics, 39(1):113-137, 1986.
# [2] Robert V. Kohn and Gilbert Strang. Optimal design and relaxation of variational problems, II. Communications on Pure and Applied Mathematics, 39(2):139-182, 1986.
# [3] Georg Dolzmann. Numerical computation of rank-one convex envelopes. SIAM Journal on Numerical Analysis, 36(5):1621-1635, 1999.
#

using NumericalRelaxation
using StaticArrays
using Tensors
using Plots


function W(F::Matrix{Float64})
    return (norm(F) <= (sqrt(2) - 1)) ? (2 * sqrt(2) * norm(F)) : (1 + norm(F)^2)
end
function W(F::SMatrix{d,d,Float64}) where {d}
    return (norm(F) <= (sqrt(2) - 1)) ? (2 * sqrt(2) * norm(F)) : (1 + norm(F)^2)
end
function W(F::Tensor{2,d,Float64}) where {d}
    return (norm(F) <= (sqrt(2) - 1)) ? (2 * sqrt(2) * norm(F)) : (1 + norm(F)^2)
end


function DW(F::Matrix{Float64})
    return (norm(F) <= (sqrt(2) - 1)) ? (2 * sqrt(2) * (norm(F)^(-1)) * F) : 2 .* F
end
function DW(F::SMatrix{d,d,Float64}) where {d}
    return (norm(F) <= (sqrt(2) - 1)) ? (2 * sqrt(2) * (norm(F)^(-1)) * F) : 2 .* F
end
function DW(F::Tensor{2,d,Float64}) where {d}
    return (norm(F) <= (sqrt(2) - 1)) ? (2 * sqrt(2) * (norm(F)^(-1)) * F) : 2 .* F
end


function Wpc(F::Matrix{Float64})
    rho = sqrt(norm(F)^2 + 2 * abs(det(F)))
    return (rho <= 1) ? (2 * (rho - abs(det(F)))) : (1 + norm(F)^2)
end
function Wpc(F::SMatrix{d,d,Float64}) where {d}
    rho = sqrt(norm(F)^2 + 2 * abs(det(F)))
    return (rho <= 1) ? (2 * (rho - abs(det(F)))) : (1 + norm(F)^2)
end
function Wpc(F::Tensor{2,d,Float64}) where {d}
    rho = sqrt(norm(F)^2 + 2 * abs(det(F)))
    return (rho <= 1) ? (2 * (rho - abs(det(F)))) : (1 + norm(F)^2)
end


function DWpc(F::Matrix{Float64})
    rho = sqrt(norm(F)^2 + 2 * abs(det(F)))
    Drho = (1 / 2) * (norm(F)^2 + 2 * abs(det(F)))^(-1 / 2) * (2 * F + 2 * sign(det(F)) * inv(F)' * det(F))
    return (rho <= 1) ? (2 * (Drho - sign(det(F)) * inv(F)' * det(F))) : 2 .* F
end
function DWpc(F::SMatrix{d,d,Float64}) where {d}
    rho = sqrt(norm(F)^2 + 2 * abs(det(F)))
    Drho = (1 / 2) * (norm(F)^2 + 2 * abs(det(F)))^(-1 / 2) * (2 * F + 2 * sign(det(F)) * inv(F)' * det(F))
    return (rho <= 1) ? (2 * (Drho - sign(det(F)) * inv(F)' * det(F))) : 2 .* F
end
function DWpc(F::Tensor{2,d,Float64}) where {d}
    rho = sqrt(norm(F)^2 + 2 * abs(det(F)))
    Drho = (1 / 2) * (norm(F)^2 + 2 * abs(det(F)))^(-1 / 2) * (2 * F + 2 * sign(det(F)) * inv(F)' * det(F))
    return (rho <= 1) ? (2 * (Drho - sign(det(F)) * inv(F)' * det(F))) : 2 .* F
end


function Φ(ν)
    return W(diagm(ν))
end

function Φpc(ν)
    return Wpc(diagm(ν))
end


# ##############################################################
# small convergence test for the polyconvexification approach 
# ##############################################################
d = 2
F = (rand(2,2).-0.5) * 2
ν = ssv(F)
r = 2
nrefs = collect(1:10)
errors = zeros(size(nrefs))
errorsIso = zeros(size(nrefs))
errorsDerivative = zeros(size(nrefs))
errorsDerivativeIso = zeros(size(nrefs))
for (i, nref) in enumerate(nrefs)
    print("Started SVPC convexify, nref = ", nref)
    poly_convexification = PolyConvexification(d, 3.; nref=nref)
    poly_buffer = build_buffer(poly_convexification)

    Φpcνδ, DΦpcνδ, _ = convexify(poly_convexification, poly_buffer, Φ, ν; returnDerivs=true)
    WpcFδ, DWpcFδ, _ = convexify(poly_convexification, poly_buffer, W, F; returnDerivs=true)
    
    errors[i] = WpcFδ - Wpc(F)
    errorsDerivative[i] = norm(DWpcFδ - DWpc(F))

    errorsIso[i] = Φpcνδ - Wpc(F)
    DWpcFδ = Tensor{3,d}(Dssv(F)) ⋅ Vec{d}(DΦpcνδ) 
    errorsDerivativeIso[i] = norm(DWpcFδ - DWpc(F))

    println(", err = ", errors[i], ", err iso = ", errorsIso[i], " err deriv = ", errorsDerivative[i], " err deriv iso = ", errorsDerivativeIso[i])
end 
Δ = (2. * r) ./ (2.0 .^ (nrefs))
plot(Δ, errors, xscale=:log10, yscale=:log10, title="Convergence SVPC and derivative", label="err", m=:xcross, minorgrid=true, xlabel="δ", ylabel="Φpc_δ(ν) - Wpc(F) / ||DWpc_δ(ν) - DWpc(F)||")
plot!(Δ, errorsIso, xscale=:log10, yscale=:log10, label="err Iso", m=:xcross)
plot!(Δ, errorsDerivative, xscale=:log10, yscale=:log10, label="deriv err", m=:xcross)
plot!(Δ, errorsDerivativeIso, xscale=:log10, yscale=:log10, label="deriv err Iso", m=:xcross)

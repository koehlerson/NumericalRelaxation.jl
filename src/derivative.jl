struct DamagePhase{dim,T,N}
    F⁺::Tensor{2,dim,T,N}
    F¯::Tensor{2,dim,T,N}
    A::Tensor{2,dim,T,N}
    W⁺::T
    W¯::T
    k::Int
    function DamagePhase(F₁::Tensor{2,dim,T,N}, F₂::Tensor{2,dim,T,N}, W₁::T, W₂::T, A::Tensor{2,dim,T,N}, k::Int) where {dim,T,N}
        F⁺ = zero(Tensor{2,dim})
        F¯ = zero(Tensor{2,dim})
        W⁺ = zero(W₁)
        W¯ = zero(W₂)
        normW₁ = abs(W₁)
        normW₂ = abs(W₂)
        if normW₁ > normW₂ #TODO fix me
            # F₁ > F₂ -> 1 = + und 2 = -
            F⁺ = F₁
            W⁺ = W₁
            F¯ = F₂
            W¯ = W₂
        else #1 = - und 2 = +
            F⁺ = F₂
            W⁺ = W₂
            F¯ = F₁
            W¯ = W₁
        end
        new{dim,T,N}(F⁺, F¯, A, W⁺, W¯, k)
    end
end

function Base.show(io::IO, ::MIME"text/plain", phase::DamagePhase{dim}) where {dim}
    println(io,"$(dim)D Damage Phase: F¯=$(phase.F¯),F⁺=$(phase.F⁺), level=$(phase.k)")
end

struct PhaseTree{dim,T,N}
    phases::Vector{DamagePhase{dim,T,N}}
end

function Base.show(io::IO, ::MIME"text/plain", phasetree::PhaseTree{dim}) where dim
    println("$(dim)D Phasetree")
    for phase in phasetree.phases
        println(io,"\t"^(phase.k-1)* "k=$(phase.k) "* "F¯=$(phase.F¯), F⁺=$(phase.F⁺)")
    end
end

function Base.isassigned(phasetree::PhaseTree,level::Int)
    idx = findfirst(x->x.k==level,phasetree.phases)
    return isa(idx,Int)
end

function Base.setindex!(phasetree::PhaseTree,x,level)
    idx = findfirst(x->x.k==level,phasetree.phases)
    phasetree.phases[idx] = x
end

function Base.getindex(phasetree::PhaseTree,level)
    idx = findfirst(x->x.k==level,phasetree.phases)
    return phasetree.phases[idx]
end

Base.lastindex(phasetree::PhaseTree) = maximum(getproperty.(phasetree.phases,:k))

struct FlexiblePhaseTree{dim,T,N}
    F::Tensor{2,dim,T,N}
    W::T
    ξ::T
    level::Int
    children::Vector{FlexiblePhaseTree{dim,T,N}}
end

FlexiblePhaseTree(F::Tensor{2,dim,T,N},W,ξ,l) where {dim,T,N} = FlexiblePhaseTree(F,W,ξ,l,FlexiblePhaseTree{dim,T,N}[])
ongrid(𝐅,grid) = [any(isapprox.((𝐅[i]), grid.axes[i],atol=1-5)) for i in 1:length(𝐅)] |> all

function weightedindexes(arg, scaleditp::Interpolations.ScaledInterpolation, x::NTuple{N}) where {N}
    ind = ntuple(i-> (x[i] - first(scaleditp.ranges[i])) / step(scaleditp.ranges[i]) + 1, N)
    return Interpolations.weightedindexes(arg, Interpolations.itpinfo(scaleditp)..., ind)
end

weightedindexes(arg, extrapolation::Interpolations.Extrapolation, x) = weightedindexes(arg,extrapolation.itp,x)

function decompose(F::Tensor{2,2,T,N}, itp::Interpolations.Extrapolation) where {T,N}
    weightedindices = weightedindexes((Interpolations.value_weights,), itp, F.data)
    weights = Interpolations.weights.(weightedindices)
    indices = Interpolations.indextuple.(weightedindices)
    weights = [weights[1][idxi]*weights[2][idxj]*weights[3][idxk]*weights[4][idxl] for idxi in 1:2 for idxj in 1:2 for idxk in 1:2 for idxl in 1:2]
    points = [Tensor{2,2,T,N}((itp.itp.ranges[1][i], itp.itp.ranges[2][j], itp.itp.ranges[3][k], itp.itp.ranges[4][l])) for i in indices[1] for j in indices[2] for k in indices[3] for l in indices[4]] 
    nnzs = findall(x->!isapprox(x,0.0,atol=1e-15) && !isnan(x),weights) #nzs
    return (points[nnzs], weights[nnzs])
end

function FlexiblePhaseTree(F::Tensor{2,dim,T,N},state::MultiDimRelaxedDamageTreeState,material::MultiDimRelaxedDamage,startdepth) where {dim,T,N}
    phasesforrest = state.phasestree
    _keys = collect(keys(phasesforrest))
    depth = startdepth
    root = FlexiblePhaseTree(F,0.0,1.0,depth+1)
    node = root
    phases = Tuple{Union{DamagePhase{2,T,N},FlexiblePhaseTree{2,T,N}},FlexiblePhaseTree{2,T,N}}[]
    if ongrid(F,material.convexification.grid)
        keyidx = findfirst(x->isapprox(x,F,atol=1e-8),_keys)
        keyidx === nothing && return root
        highest_order_laminate = lastindex(phasesforrest[_keys[keyidx]])
        depth = highest_order_laminate < depth ? highest_order_laminate : depth
        if isassigned(phasesforrest[_keys[keyidx]],depth)
            push!(phases,(phasesforrest[_keys[keyidx]][depth],node))
        end
    else
        points, weights = decompose(F,state.W_rk1_new)
        W_values = [state.W_rk1_new(point.data...) for point in points]
        perm = sortperm(W_values)
        points = points[perm]; weights = weights[perm]
        #points_with_keys = haskey.((phasesforrest,),points)
        #points = points[points_with_keys]; weights = weights[points_with_keys]
        is_on_laminate = false
        #for (pointidx,point) in enumerate(points)
        #    keyidx = findfirst(x->isapprox(x,point,atol=1e-8),_keys)
        #    keyidx === nothing && continue
        #    highest_order_laminate = lastindex(phasesforrest[_keys[keyidx]])
        #    depth = highest_order_laminate #< depth ? highest_order_laminate : depth
        #    _phase = phasesforrest[_keys[keyidx]][depth]
        #    ξ = (norm(F,1) - norm(_phase.F¯,1))/(norm(_phase.F⁺,1) - norm(_phase.F¯,1)) #TODO fix me
        #    if isapprox(norm(F- ξ*_phase.F⁺ - (1-ξ)*_phase.F¯,2),0.0,atol=1e-6)
        #        is_on_laminate = true
        #        F_plus = FlexiblePhaseTree(_phase.F⁺, state.W_rk1_new(_phase.F⁺...), ξ, depth-1)
        #        F_minus = FlexiblePhaseTree(_phase.F¯, state.W_rk1_new(_phase.F¯...),1-ξ, depth-1)
        #        #candidate = FlexiblePhaseTree(child.F, state.W_rk1_new(child.F...), child.ξ, child.level, [F_plus, F_minus])
        #        push!(phases,(F_plus,node))
        #        push!(phases,(F_minus,node))
        #        break
        #    end
        #end
        # TODO gleiches Problem wie Zeile 720
        #phases_fpt = [FlexiblePhaseTree(point, W, weight, depth) for (point,weight,W) in zip(points,weights,W_values)]
        if !is_on_laminate
            _depths = Int[] # depths of decomposition
            for point in points
                if findfirst(x->isapprox(x,point,atol=1e-8),_keys) isa Int
                    push!(_depths,lastindex(phasesforrest[point]) < depth ? lastindex(phasesforrest[point])+1 : depth)
                else
                    push!(_depths,depth)
                end
            end
            phases_fpt = [FlexiblePhaseTree(point, W, weight, _depth) for (point,weight,W,_depth) in zip(points,weights,W_values,_depths)]
            for phase in phases_fpt push!(phases,(phase,node)) end
        end
    end
    while !isempty(phases)
        phase,node = pop!(phases)
        if phase isa DamagePhase
            ξ = (norm(node.F,1) - norm(phase.F¯,1))/(norm(phase.F⁺,1) - norm(phase.F¯,1)) #TODO fix me
            #isnan(ξ) && continue
            push!(node.children, FlexiblePhaseTree(phase.F¯, phase.W¯, (1.0-ξ), phase.k))
            push!(node.children, FlexiblePhaseTree(phase.F⁺, phase.W⁺, ξ, phase.k))
            #depth = phase.k
            #depth -= 1
        elseif phase isa FlexiblePhaseTree
            childidxs = findall(x->x[1] isa FlexiblePhaseTree && x[2] == node,phases)
            childidxs === nothing ? push!(node.children,phase) : append!(node.children, [phase; getindex.(phases,1)[childidxs]])
            #depth = node.children[end].level
            #depth -= 1
            deleteat!(phases,childidxs)
        end
        for child in node.children
            depth = child.level -1
            keyidx = findfirst(x->isapprox(x,child.F,atol=1e-8),_keys)
            if ongrid(child.F,material.convexification.grid) && !(keyidx === nothing) && depth >= 1
                if isassigned(phasesforrest[_keys[keyidx]],depth)
                    damagephases_ongrid = phasesforrest[_keys[keyidx]][depth]
                else
                    #highest_order_laminate = lastindex(phasesforrest[child.F])
                    #depth = highest_order_laminate - 1
                    #damagephases_ongrid = phasesforrest[child.F][depth]
                    continue
                end
                ξ = (norm(child.F,1) - norm(damagephases_ongrid.F¯,1))/(norm(damagephases_ongrid.F⁺,1) - norm(damagephases_ongrid.F¯,1)) #TODO fix me
                (isinf(ξ) || isnan(ξ)) && continue
                push!(phases, (FlexiblePhaseTree(damagephases_ongrid.F¯, damagephases_ongrid.W¯, (1.0-ξ), damagephases_ongrid.k),child))
                push!(phases, (FlexiblePhaseTree(damagephases_ongrid.F⁺, damagephases_ongrid.W⁺, ξ, damagephases_ongrid.k),child))
            elseif depth >= 1 && !ongrid(child.F,material.convexification.grid)
                points, weights = decompose(child.F, state.W_rk1_new)
                W_values = [state.W_rk1_new(point.data...) for point in points]
                perm = sortperm(W_values)
                points = points[perm]; weights = weights[perm]
                # TODO: Was ist hier richtig? gehen wir ein level höher, wenn wir es durch die lineare interpolation aufteilen, oder nicht?
                # auskommentierter geht **nicht** höher, falls eine Aufteilung erfolgt
                is_on_laminate = false
                #for (pointidx,point) in enumerate(points)
                #    keyidx = findfirst(x->isapprox(x,point,atol=1e-8),_keys)
                #    keyidx === nothing && continue
                #    isassigned(phasesforrest[_keys[keyidx]], depth+1) ? nothing : continue
                #    _phase = phasesforrest[_keys[keyidx]][depth+1]
                #    ξ = (norm(point,1) - norm(_phase.F¯,1))/(norm(_phase.F⁺,1) - norm(_phase.F¯,1)) #TODO fix me
                #    if isapprox(norm(point - ξ*_phase.F⁺ - (1-ξ)*_phase.F¯,2),0.0,atol=1e-6)
                #        is_on_laminate = true
                #        F_plus = FlexiblePhaseTree(_phase.F⁺, state.W_rk1_new(_phase.F⁺...), ξ, child.level-1)
                #        F_minus = FlexiblePhaseTree(_phase.F¯, state.W_rk1_new(_phase.F¯...),1-ξ, child.level-1)
                #        #candidate = FlexiblePhaseTree(child.F, state.W_rk1_new(child.F...), child.ξ, child.level, [F_plus, F_minus])
                #        push!(phases,(F_plus,child))
                #        push!(phases,(F_minus,child))
                #        break
                #    end
                #end
                if !is_on_laminate
                    phases_decomposed = [FlexiblePhaseTree(point, W, weight, depth+1) for (point,weight,W) in zip(points,weights,W_values)]
                    #phases_decomposed = [FlexiblePhaseTree(point, W, weight, depth) for (point,weight,W) in zip(points,weights,W_values)]
                    for candidate in phases_decomposed
                        push!(phases,(candidate,child))
                    end
                end
            end
        end
    end
    return root
end

function eval(node::FlexiblePhaseTree{2}, material, state)
    W = 0.0
    𝐏 = zero(Tensor{2,2})
    𝔸 = zero(Tensor{4,2})
    if isempty(node.children)
        𝔸_temp, 𝐏_temp, W_temp = Tensors.hessian(y -> W_energy(y, material, state.damage), node.F, :all)
        W += W_temp; 𝐏 += 𝐏_temp; 𝔸 += 𝔸_temp
    else
        for child in node.children
            𝔸_c, 𝐏_c, W_c = eval(child,material,state)
            W += child.ξ*W_c; 𝐏 += child.ξ*𝐏_c; 𝔸 += child.ξ * 𝔸_c
        end
    end
    return 𝔸, 𝐏, W
end

function checkintegrity(tree::FlexiblePhaseTree,itp)
    isintegre = true
    for node in AbstractTrees.StatelessBFS(tree)
        if isempty(node.children)
            continue
        end
        F = node.F
        W = itp(F.data...)
        points = getproperty.(node.children,:F)
        weights = getproperty.(node.children,:ξ)
        W_values = [itp(point.data...) for point in points]
        isintegre = isapprox(F,sum(points .* weights)) && isapprox(W,sum(W_values .* weights)) #hinterer Teil nicht überprüfbar da wir nur W^endlevel kennen und nicht alle W^k
        if !isintegre
            isintegre = false
            break
        end
    end
    return isintegre
end

nearestnode(F::Tensor{dim},defogrid) where dim = typeof(F)(ntuple(i->closestval(getaxes(defogrid)[i],F[i]),dim^2))
closestval(x,val) = x[closestindex(x,val)]

function closestindex(x,val)
    ibest = first(eachindex(x))
    dxbest = abs(x[ibest]-val)
        for I in eachindex(x)
            dx = abs(x[I]-val)
            if dx < dxbest
                dxbest = dx
                ibest = I
            end
        end
    return ibest
end

function AbstractTrees.children(node::FlexiblePhaseTree)
    return (node.children...,)
end

AbstractTrees.printnode(io::IO, node::FlexiblePhaseTree) = print(io, "$(node.F) ξ=$(node.ξ)")
Base.show(io::IO, ::MIME"text/plain", tree::FlexiblePhaseTree) = AbstractTrees.print_tree(io, tree)
Base.eltype(::Type{<:AbstractTrees.TreeIterator{FlexiblePhaseTree{dim,T,N}}}) where {dim,T,N} = FlexiblePhaseTree{dim,T,N}
Base.IteratorEltype(::Type{<:AbstractTrees.TreeIterator{FlexiblePhaseTree{dim,T,N}}}) where {dim,T,N} = Base.HasEltype()

function neighborhood_average(𝐅::Tensor{2,dim}, material::MultiDimRelaxedDamage{dim}, state::Union{MultiDimRelaxedDamageState{dim},MultiDimRelaxedDamageTreeState{dim}}) where dim
    diff = δ(material.convexification.grid)/4
    gridaxes = getaxes(material.convexification.grid)
    indices_on_axis = Vector{Int}()
    for (i,axis) in enumerate(gridaxes)
        if any(isapprox.((𝐅[i],), axis,atol=1e-5))
            push!(indices_on_axis, i)
        end
    end
    amount_on_axis = length(indices_on_axis)
    dirs = Iterators.product(ntuple(x->-1:2:1,amount_on_axis)...)
    𝔸 = Vector{Tensor{4,dim}}()
    𝐏 = Vector{Tensor{2,dim}}()
    for dir in dirs
        perturbation = zeros(dim,dim)
        perturbation[indices_on_axis] .= dir
        perturbation .*= diff
        𝔸_F, 𝐏_F = Tensors.hessian(y -> state.W_rk1_new(y...), Tensor{2,dim}(𝐅+ perturbation),:all)
        push!(𝐏,𝐏_F)
        push!(𝔸,𝔸_F)
    end
    #handling of diagonal tangent sign switch
    tangent_signs = [getindex.(broadcast.(sign,𝐏),i,j) for i in 1:dim, j in 1:dim] # get sign matrices
    # index_to_overwrite checks where a sign change happened in the tangents
    index_to_overwrite = findall(x->abs(sum(x))!=length(dirs) && !all(x.≈ 0.0),tangent_signs)
    𝐏 = Matrix(sum(𝐏))
    𝔸 = sum(𝔸)
    #𝔸 = Tensors.hessian(y -> state.W_rk1_new(y...), Tensor{2,dim}(𝐅))
    𝔸⁰ = Tensors.hessian(x -> W_energy(x, material, state.damage), 𝐅)
    𝐏 ./= length(dirs)
    𝔸 = 𝔸 / length(dirs)
    for idx in index_to_overwrite
        𝐏[idx] = 0.0
    end
    𝔸 = Tensor{4,dim}((i,j,k,l) -> nominaltangent_subdifferential(i,j,k,l,𝔸,𝔸⁰,index_to_overwrite))
    return 𝔸,Tensor{2,dim}(𝐏)
end

function nominaltangent_subdifferential(i,j,k,l,𝔸,𝔸⁰,idx_overwrite)
    if CartesianIndex(i,j) ∈ idx_overwrite #&& i == j
        return 𝔸⁰[i,j,k,l]
    else
        return 𝔸[i,j,k,l]
    end
end

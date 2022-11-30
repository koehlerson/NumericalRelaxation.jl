struct FlexibleLaminateTree{dim,T,N}
    F::Tensor{2,dim,T,N}
    W::T
    ξ::T
    level::Int
    children::Vector{FlexibleLaminateTree{dim,T,N}}
end

FlexibleLaminateTree(F::Tensor{2,dim,T,N},W,ξ,l) where {dim,T,N} = FlexibleLaminateTree(F,W,ξ,l,FlexibleLaminateTree{dim,T,N}[])
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

function decompose(F::Tensor{2,3,T,N}, itp::Interpolations.Extrapolation) where {T,N}
    weightedindices = weightedindexes((Interpolations.value_weights,), itp, F.data)
    weights = Interpolations.weights.(weightedindices)
    indices = Interpolations.indextuple.(weightedindices)
    weights = [weights[1][idxi]*weights[2][idxj]*weights[3][idxk]*weights[4][idxl]*weights[5][idxm]*weights[6][idxn]*weights[7][idxo]*weights[8][idxp]*weights[9][idxq] for idxi in 1:2 for idxj in 1:2 for idxk in 1:2 for idxl in 1:2 for idxm in 1:2 for idxn in 1:2 for idxo in 1:2 for idxp in 1:2 for idxq in 1:2]
    points = [Tensor{2,3,T,N}((itp.itp.ranges[1][i], itp.itp.ranges[2][j], itp.itp.ranges[3][k], itp.itp.ranges[4][l], itp.itp.ranges[5][m], itp.itp.ranges[6][n], itp.itp.ranges[7][o], itp.itp.ranges[8][p], itp.itp.ranges[9][q])) for i in indices[1] for j in indices[2] for k in indices[3] for l in indices[4] for m in indices[5] for n in indices[6] for o in indices[7] for p in indices[8] for q in indices[9]] 
    nnzs = findall(x->!isapprox(x,0.0,atol=1e-10) && !isnan(x),weights) #nzs 
    return (points[nnzs], weights[nnzs])
end

function linearindex(F::Tensor{2,dim},grid::GradientGrid) where dim
    cartesianindices = ntuple(i->findfirst(x->isapprox(F[i],x),grid.axes[i]),dim^2)
    any(isnothing.(cartesianindices)) && return nothing
    return LinearIndices(size(grid))[cartesianindices...]
end

function FlexibleLaminateTree(F::Tensor{2,dim,T,N},r1convexification::R1Convexification,buffer::R1ConvexificationBuffer,startdepth) where {dim,T,N}
    laminateforrest = buffer.laminatetree
    depth = startdepth
    root = FlexibleLaminateTree(F,0.0,1.0,depth+1)
    node = root
    laminates = Tuple{Union{Laminate{2,T,N},FlexibleLaminateTree{2,T,N}},FlexibleLaminateTree{2,T,N}}[]
    if ongrid(F,r1convexification.grid)
        _key = linearindex(F,r1convexification.grid)
        !haskey(buffer.laminatetree,_key) && return root
        highest_order_laminate = lastindex(laminateforrest[_key])
        depth = highest_order_laminate < depth ? highest_order_laminate : depth
        if isassigned(laminateforrest[_key],depth)
            push!(laminates,(laminateforrest[_key][depth],node))
        end
    else
        points, weights = decompose(F,buffer.W_rk1)
        W_values = [buffer.W_rk1(point.data...) for point in points]
        perm = sortperm(W_values)
        points = points[perm]; weights = weights[perm]
        #points_with_keys = haskey.((laminatesforrest,),points)
        #points = points[points_with_keys]; weights = weights[points_with_keys]
        is_on_laminate = false
        #for (pointidx,point) in enumerate(points)
        #    keyidx = findfirst(x->isapprox(x,point,atol=1e-8),_keys)
        #    keyidx === nothing && continue
        #    highest_order_laminate = lastindex(laminatesforrest[_keys[keyidx]])
        #    depth = highest_order_laminate #< depth ? highest_order_laminate : depth
        #    _laminate = laminatesforrest[_keys[keyidx]][depth]
        #    ξ = (norm(F,1) - norm(_laminate.F¯,1))/(norm(_laminate.F⁺,1) - norm(_laminate.F¯,1)) #TODO fix me
        #    if isapprox(norm(F- ξ*_laminate.F⁺ - (1-ξ)*_laminate.F¯,2),0.0,atol=1e-6)
        #        is_on_laminate = true
        #        F_plus = FlexibleLaminateTree(_laminate.F⁺, state.W_rk1_new(_laminate.F⁺...), ξ, depth-1)
        #        F_minus = FlexibleLaminateTree(_laminate.F¯, state.W_rk1_new(_laminate.F¯...),1-ξ, depth-1)
        #        #candidate = FlexibleLaminateTree(child.F, state.W_rk1_new(child.F...), child.ξ, child.level, [F_plus, F_minus])
        #        push!(laminates,(F_plus,node))
        #        push!(laminates,(F_minus,node))
        #        break
        #    end
        #end
        # TODO gleiches Problem wie Zeile 720
        #laminates_fpt = [FlexibleLaminateTree(point, W, weight, depth) for (point,weight,W) in zip(points,weights,W_values)]
        if !is_on_laminate
            _depths = Int[] # depths of decomposition
            for point in points
                pivot_key = linearindex(point,r1convexification.grid)
                if haskey(laminateforrest,pivot_key)
                    push!(_depths,lastindex(laminateforrest[pivot_key]) < depth ? lastindex(laminateforrest[pivot_key])+1 : depth)
                else
                    push!(_depths,depth)
                end
            end
            laminates_fpt = [FlexibleLaminateTree(point, W, weight, _depth) for (point,weight,W,_depth) in zip(points,weights,W_values,_depths)]
            for laminate in laminates_fpt push!(laminates,(laminate,node)) end
        end
    end
    while !isempty(laminates)
        lc,parent = pop!(laminates)
        if lc isa Laminate
            ξ = (norm(parent.F - lc.F¯,1))/(norm(lc.F⁺ - lc.F¯,1)) #TODO fix me
            #isnan(ξ) && continue
            push!(parent.children, FlexibleLaminateTree(lc.F¯, lc.W¯, (1.0-ξ), lc.k))
            push!(parent.children, FlexibleLaminateTree(lc.F⁺, lc.W⁺, ξ, lc.k))
            #depth = laminate.k
            #depth -= 1
        elseif lc isa FlexibleLaminateTree
            childidxs = findall(x->x[1] isa FlexibleLaminateTree && x[2] == parent,laminates)
            childidxs === nothing ? push!(parent.children,lc) : append!(parent.children, [lc; getindex.(laminates,1)[childidxs]])
            #depth = node.children[end].level
            #depth -= 1
            deleteat!(laminates,childidxs)
        end
        for child in parent.children
            depth = child.level -1
            _key = linearindex(child.F,r1convexification.grid)
            if ongrid(child.F,r1convexification.grid) && !(_key === nothing) && depth >= 1
                if haskey(laminateforrest,_key) && isassigned(laminateforrest[_key],depth)
                    laminates_ongrid = laminateforrest[_key][depth]
                else
                    #highest_order_laminate = lastindex(laminatesforrest[child.F])
                    #depth = highest_order_laminate - 1
                    #laminates_ongrid = laminatesforrest[child.F][depth]
                    continue
                end
                ξ = (norm(child.F - laminates_ongrid.F¯,1))/(norm(laminates_ongrid.F⁺ - laminates_ongrid.F¯,1)) #TODO fix me
                (isinf(ξ) || isnan(ξ)) && continue
                push!(laminates, (FlexibleLaminateTree(laminates_ongrid.F¯, laminates_ongrid.W¯, (1.0-ξ), laminates_ongrid.k),child))
                push!(laminates, (FlexibleLaminateTree(laminates_ongrid.F⁺, laminates_ongrid.W⁺, ξ, laminates_ongrid.k),child))
            elseif depth >= 1 && !ongrid(child.F,r1convexification.grid)
                points, weights = decompose(child.F, buffer.W_rk1)
                W_values = [buffer.W_rk1(point.data...) for point in points]
                perm = sortperm(W_values)
                points = points[perm]; weights = weights[perm]
                # TODO: Was ist hier richtig? gehen wir ein level höher, wenn wir es durch die lineare interpolation aufteilen, oder nicht?
                # auskommentierter geht **nicht** höher, falls eine Aufteilung erfolgt
                is_on_laminate = false
                #for (pointidx,point) in enumerate(points)
                #    keyidx = findfirst(x->isapprox(x,point,atol=1e-8),_keys)
                #    keyidx === nothing && continue
                #    isassigned(laminatesforrest[_keys[keyidx]], depth+1) ? nothing : continue
                #    _laminate = laminatesforrest[_keys[keyidx]][depth+1]
                #    ξ = (norm(point,1) - norm(_laminate.F¯,1))/(norm(_laminate.F⁺,1) - norm(_laminate.F¯,1)) #TODO fix me
                #    if isapprox(norm(point - ξ*_laminate.F⁺ - (1-ξ)*_laminate.F¯,2),0.0,atol=1e-6)
                #        is_on_laminate = true
                #        F_plus = FlexibleLaminateTree(_laminate.F⁺, state.W_rk1_new(_laminate.F⁺...), ξ, child.level-1)
                #        F_minus = FlexibleLaminateTree(_laminate.F¯, state.W_rk1_new(_laminate.F¯...),1-ξ, child.level-1)
                #        #candidate = FlexibleLaminateTree(child.F, state.W_rk1_new(child.F...), child.ξ, child.level, [F_plus, F_minus])
                #        push!(laminates,(F_plus,child))
                #        push!(laminates,(F_minus,child))
                #        break
                #    end
                #end
                if !is_on_laminate
                    laminates_decomposed = [FlexibleLaminateTree(point, W, weight, depth+1) for (point,weight,W) in zip(points,weights,W_values)]
                    #laminates_decomposed = [FlexibleLaminateTree(point, W, weight, depth) for (point,weight,W) in zip(points,weights,W_values)]
                    for candidate in laminates_decomposed
                        push!(laminates,(candidate,child))
                    end
                end
            end
        end
    end
    return root
end

function eval(node::FlexibleLaminateTree{dim}, W_nonconvex::Function, xargs...) where dim
    W = 0.0
    𝐏 = zero(Tensor{2,dim})
    𝔸 = zero(Tensor{4,dim})
    if isempty(node.children)
        𝔸_temp, 𝐏_temp, W_temp = Tensors.hessian(y -> W_nonconvex(y, xargs...), node.F, :all)
        W += W_temp; 𝐏 += 𝐏_temp; 𝔸 += 𝔸_temp
    else
        for child in node.children
            𝔸_c, 𝐏_c, W_c = eval(child,W_nonconvex,xargs...)
            W += child.ξ*W_c; 𝐏 += child.ξ*𝐏_c; 𝔸 += child.ξ * 𝔸_c
        end
    end
    return 𝔸, 𝐏, W
end

function checkintegrity(tree::FlexibleLaminateTree,itp)
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
        isintegre = isapprox(F,sum(points .* weights)) && isapprox(W,sum(W_values .* weights))
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

function AbstractTrees.children(node::FlexibleLaminateTree)
    return (node.children...,)
end

AbstractTrees.printnode(io::IO, node::FlexibleLaminateTree) = print(io, "$(node.F) ξ=$(node.ξ)")
Base.show(io::IO, ::MIME"text/plain", tree::FlexibleLaminateTree) = AbstractTrees.print_tree(io, tree)
Base.eltype(::Type{<:AbstractTrees.TreeIterator{FlexibleLaminateTree{dim,T,N}}}) where {dim,T,N} = FlexibleLaminateTree{dim,T,N}
Base.IteratorEltype(::Type{<:AbstractTrees.TreeIterator{FlexibleLaminateTree{dim,T,N}}}) where {dim,T,N} = Base.HasEltype()

#function neighborhood_average(𝐅::Tensor{2,dim}, material::MultiDimRelaxedDamage{dim}, state::Union{MultiDimRelaxedDamageState{dim},MultiDimRelaxedDamageTreeState{dim}}) where dim
#    diff = δ(material.convexification.grid)/4
#    gridaxes = getaxes(material.convexification.grid)
#    indices_on_axis = Vector{Int}()
#    for (i,axis) in enumerate(gridaxes)
#        if any(isapprox.((𝐅[i],), axis,atol=1e-5))
#            push!(indices_on_axis, i)
#        end
#    end
#    amount_on_axis = length(indices_on_axis)
#    dirs = Iterators.product(ntuple(x->-1:2:1,amount_on_axis)...)
#    𝔸 = Vector{Tensor{4,dim}}()
#    𝐏 = Vector{Tensor{2,dim}}()
#    for dir in dirs
#        perturbation = zeros(dim,dim)
#        perturbation[indices_on_axis] .= dir
#        perturbation .*= diff
#        𝔸_F, 𝐏_F = Tensors.hessian(y -> state.W_rk1_new(y...), Tensor{2,dim}(𝐅+ perturbation),:all)
#        push!(𝐏,𝐏_F)
#        push!(𝔸,𝔸_F)
#    end
#    #handling of diagonal tangent sign switch
#    tangent_signs = [getindex.(broadcast.(sign,𝐏),i,j) for i in 1:dim, j in 1:dim] # get sign matrices
#    # index_to_overwrite checks where a sign change happened in the tangents
#    index_to_overwrite = findall(x->abs(sum(x))!=length(dirs) && !all(x.≈ 0.0),tangent_signs)
#    𝐏 = Matrix(sum(𝐏))
#    𝔸 = sum(𝔸)
#    #𝔸 = Tensors.hessian(y -> state.W_rk1_new(y...), Tensor{2,dim}(𝐅))
#    𝔸⁰ = Tensors.hessian(x -> W_energy(x, material, state.damage), 𝐅)
#    𝐏 ./= length(dirs)
#    𝔸 = 𝔸 / length(dirs)
#    for idx in index_to_overwrite
#        𝐏[idx] = 0.0
#    end
#    𝔸 = Tensor{4,dim}((i,j,k,l) -> nominaltangent_subdifferential(i,j,k,l,𝔸,𝔸⁰,index_to_overwrite))
#    return 𝔸,Tensor{2,dim}(𝐏)
#end
#
#function nominaltangent_subdifferential(i,j,k,l,𝔸,𝔸⁰,idx_overwrite)
#    if CartesianIndex(i,j) ∈ idx_overwrite #&& i == j
#        return 𝔸⁰[i,j,k,l]
#    else
#        return 𝔸[i,j,k,l]
#    end
#end

#Replaces flow.jl, manifolds.jl, loss.gl
#Update tests!


abstract type Flow end
abstract type VectorFlow <: Flow end #State represented by a vector
abstract type MatrixFlow <: Flow end #State represented by a matrix
abstract type FlowState{T, N} <: AbstractArray{T, N} end

export Flow, VectorFlow, MatrixFlow, FlowState

### Flow State Structs ###
struct VectorFlowState{T, A<:AbstractArray{T, 2}, B <: AbstractVector{Bool}} <: FlowState{T, 2}
    x::A
    mask::B
end
function VectorFlowState(A::AbstractArray{T, 2}) where T
    return VectorFlowState(A, fill(true, size(A, 2)))
end

struct MatrixFlowState{T, A<:AbstractArray{T, 3}, B <: AbstractVector{Bool}} <: FlowState{T, 3}
    x::A
    mask::B
end
function MatrixFlowState(A::AbstractArray{T, 3}) where T
    return MatrixFlowState(A, fill(true, size(A, 3)))
end


### Flows ###
struct EuclideanFlow <: VectorFlow
    schedule::Function
end
EuclideanFlow() = EuclideanFlow(t -> t)
statetype(f::EuclideanFlow) = VectorFlowState

struct RelaxedDiscreteFlow <: VectorFlow
    schedule::Function
end
RelaxedDiscreteFlow() = RelaxedDiscreteFlow(t -> t)
statetype(f::RelaxedDiscreteFlow) = VectorFlowState

struct RotationalFlow <: MatrixFlow
    schedule::Function
end
RotationalFlow() = RotationalFlow(t -> t)
statetype(f::RotationalFlow) = MatrixFlowState

struct ManifoldVectorFlow <: VectorFlow #I'm specializing this, in case we want to try and handle others later
    schedule::Function
    manifold::AbstractManifold{ℝ}
end
ManifoldVectorFlow(manifold) = ManifoldVectorFlow(t -> t, manifold)
statetype(f::ManifoldVectorFlow) = VectorFlowState

"""
    batch_flowstate(statetuple::Tuple{Vararg{AbstractArray}}, flowtuple::Tuple{Vararg{Flow}})

Converts a tuple of abstract arrays and a tuple of Flows, into a tuple of Flow-appropriate FlowStates.
"""
#No masking:
batch_flowstate(flowtuple::Tuple{Vararg{Flow}}, statetuple::Tuple{Vararg{AbstractArray}}) = Tuple([c(s) for (c,s) in zip(statetype.(flowtuple),statetuple)]) #ick
#Allows one mask per flow state
#Adding this copy in for a hunch about a GPU issue...
batch_flowstate(flowtuple::Tuple{Vararg{Flow}}, statetuple::Tuple{Vararg{AbstractArray}}, masktuple::Tuple{Vararg{AbstractArray}}) = Tuple([c(s,copy(m)) for (c,s,m) in zip(statetype.(flowtuple),statetuple, masktuple)])
#One mask which gets used for all flow states
batch_flowstate(flowtuple::Tuple{Vararg{Flow}}, statetuple::Tuple{Vararg{AbstractArray}}, m::AbstractArray) = Tuple([c(s,copy(m)) for (c,s) in zip(statetype.(flowtuple),statetuple)])
#Test these second two!
export batch_flowstate

#Making these wrapper types GPU compatible
#Adapt.adapt_structure(to, A::FlowState) = typeof(A)(Adapt.adapt(to, A.x), Adapt.adapt(to, A.mask))
#For some reason the above was triggering scalarindexing!?
Adapt.adapt_structure(to, A::MatrixFlowState) = MatrixFlowState(Adapt.adapt(to, A.x), Adapt.adapt(to, A.mask))
Adapt.adapt_structure(to, A::VectorFlowState) = VectorFlowState(Adapt.adapt(to, A.x), Adapt.adapt(to, A.mask))


### Inheriting from AbstractArray, and definind cat behavior ###
Base.size(A::FlowState) = size(A.x)
Base.copy(A::FlowState) = typeof(A)(copy(A.x), copy(A.mask))
Base.getindex(A::FlowState, i...) = A.x[i...]
#Base.parent(A::FlowState) = A.x #Undecided on this one

function Base.cat(arrays::VectorFlowState...; dims = 2) #Check that default works here!
    if dims != 2
        throw(ArgumentError("Only dims=2 supported for VectorFlowState"))
    end
    return VectorFlowState(cat([a.x for a in arrays]..., dims = 2), vcat([a.mask for a in arrays]...))
end

function Base.cat(arrays::MatrixFlowState...; dims = 3)
    if dims != 3
        throw(ArgumentError("Only dims=3 supported for MatrixFlowState"))
    end
    return MatrixFlowState(cat([a.x for a in arrays]..., dims = 3), vcat([a.mask for a in arrays]...))
end

#####################
### Flow Behavior ###
#####################

### Interpolation ### - Doesn't need to happen on GPU
#The mask is NOT taken into account when interpolating the states. You need to deal with that yourself.
"""
    interpolate(f::Flow, x0::A, x1::A, t::T) where A::FlowState{T}

Geodesic interpolation between two `FlowState`s. `t`` must be a scalar, or row vector.
The flow states are interpolated regardless of the mask, and the output mask is the logical AND of the two input masks.
"""
function interpolate(f::Union{EuclideanFlow,RelaxedDiscreteFlow}, x0::VectorFlowState{T}, x1::VectorFlowState{T}, t) where T
    size(t, 1) != 1 && throw(ArgumentError("t must be a row vector or a scalar"))
    t = f.schedule.(T.(t))
    return VectorFlowState(t .* x1.x .+ (1 .- t) .* x0.x, x0.mask .& x1.mask)
end

function interpolate(f::RotationalFlow, x0::MatrixFlowState{T}, x1::MatrixFlowState{T}, t) where T
    size(t, 1) != 1 && throw(ArgumentError("t must be a row vector or a scalar"))
    t = f.schedule.(T.(t))
    #Slerp interpolation of the states, and logical AND of the masks
    return MatrixFlowState(slerp_stack(x0.x, x1.x, t), x0.mask .& x1.mask)
end

function interpolate(f::ManifoldVectorFlow, x0::VectorFlowState{T}, x1::VectorFlowState{T}, t) where T
    size(t, 1) != 1 && throw(ArgumentError("t must be a row vector or a scalar"))
    t = ones(1,size(x0.x, 2)) .* f.schedule.(T.(t))
    new_x = copy(x1.x)
    for i in 1:size(x0,2)
        γ = shortest_geodesic(f.manifold, x0[:,i], x1[:,i])
        new_x[:,i] .= γ(t[1,i])
    end
    return VectorFlowState(new_x, x0.mask .& x1.mask)
end

#Handles a tuple of FlowStates into interpolate. t is either a scalar, or a Tuple of row vectors
function interpolate(f::Tuple{Vararg{Flow}}, x0::Tuple{Vararg{FlowState}}, x1::Tuple{Vararg{FlowState}}, t::Union{Real, Tuple{Vararg{Real}}})
    return interpolate.(f, x0, x1, t)
end

#=
function x0x1_to_xt(flowtuple::Tuple{Vararg{Flow}}, x1tuple::Tuple{Vararg{AbstractArray}}, x0tuple::Tuple{Vararg{AbstractArray}}, t::Real)
    x1 = batch_flowstate(flowtuple, x1tuple)
    x0 = batch_flowstate(flowtuple, x0tuple)
    return interpolate(x0, x1, t)
end
=#

### Perturbation ### - Doesn't need to happen on GPU
"""
    perturb!(f::Flow, x::A, σ::T) where A::FlowState{T}

Perturb the flow by a random amount, respecting the manifold, but do not change states where mask is false.
"""
function perturb!(f::Union{EuclideanFlow,RelaxedDiscreteFlow}, x::VectorFlowState{T}, σ::Real) where T
    x.x[:,x.mask] .= x.x[:,x.mask] .+ σ .* randn(T, size(x.x[:,x.mask]))
end

function perturb!(f::RotationalFlow, x::MatrixFlowState{T}, σ::Real) where T
    for i in 1:size(x.x,3)
        if x.mask[i]
            x.x[:,:,i] .=  x.x[:,:,i] * Matrix(randrot(σ))
        end
    end
end

#This throws inexact error for probability simplex sometimes.
function perturb!(f::ManifoldVectorFlow, x::VectorFlowState{T}, σ::Real) where T
    for i in 1:size(x.x,2)
        if x.mask[i]
            rv = rand(f.manifold, vector_at=x.x[:,i], σ=σ) #Random vector in the tangent space
            x.x[:,i] .= exp(f.manifold,x.x[:,i],rv) #Exponential map of rv
        end
    end
end


#######################
### Generative Flow ###
#######################

#Flow for tuples, where the model must take a tuple, do joint inference, and return a tuple of data matrices.
function flow(f::Tuple{Vararg{Flow}}, x0::Tuple{Vararg{FlowState}}, model; steps = 100)
    xt = copy.(x0)
    if typeof(steps) <: Int
        t_step = eltype(x0[1].x)(1/steps)
        steps = 0:t_step:1
    end
    for i in 2:length(steps)
        t = (steps[i]+steps[i-1])/2 #midpoint
        step = steps[i] - steps[i-1]
        ts = Tuple([t .* ones(eltype(c.x), 1, size(c.x)[end]) for c in x0])
        x̂1 = copy.(x0)
        res = model(ts,xt)
        for i in 1:length(res)
            x̂1[i].x .= res[i]
        end
        xt = interpolate(f, xt, x̂1, min(1,step/(1-t)))
    end
    return xt
end
flow(f::Flow, x0::FlowState, model; steps = 100) = flow((f,), (x0,), (t,xt) -> (model(t[1],xt[1]), ), steps = steps)[1]
    

######################################################################
### Loss Functions - Need to be stable, GPU-friendly, autodiffable ###
######################################################################

safe_sqrt(x::T) where T = x < 0 ? T(0) : sqrt(x)

### GPU-friendly logarithmic maps for manifolds ###
#=
function Base.log(::ProbabilitySimplex, p_arr::AbstractMatrix, q_arr::AbstractMatrix)
    eps = eltype(p_arr)(1e-6)
    z = safe_sqrt.(p_arr .* q_arr)
    s = clamp.(sum(z, dims=1), -1, 1)
    return 2 .* acos.(s) ./ safe_sqrt.(eps + 1 .- s.^2) .* (z .- s .* p_arr)
end
=#

arccos_approx(x::T) where T = T(1.5707963) - x - x^3/6

floor_eps(x,eps) = x < eps ? eps : x

### Version without the arccos and denominator instability
function approx_stable_log(::ProbabilitySimplex, p_arr::AbstractMatrix, q_arr::AbstractMatrix)
    eps = eltype(p_arr)(0.0001)
    z = safe_sqrt.(p_arr .* q_arr)
    s = sum(z, dims=1)
    return (2 .* arccos_approx.(s) ./ (eps .+ safe_sqrt.((eps + 1) .- s.^2))) .* (z .- s .* p_arr)
end

#---Tested for prabability simplex---
#This version is completely unstable
#=
logrtr1 = log(rt.manifold, rt.x, r1.x)
logrtr̂1 = log(rt.manifold, rt.x, r̂1)
sq = clamp.(sum(abs2.(logrtr1 .- logrtr̂1), dims = 1), 0, 1)
return mean((r1.mask .* sq) ./ ((1+eps) .- t).^2)
=#

#Trying just log CE between targets and predictions
#Not good
#-mean(log.(sum(r̂1 .* r1, dims = 1)))

#Vanilla CE. Worst so far
#return -mean(r̂1 .* r1)

#Using stable log map.
#Not too bad, actually!
#NOTE: got NaNs. Trying another

#=
logrtr1 = approx_stable_log(f.manifold, rt.x, r1.x)
logrtr̂1 = approx_stable_log(f.manifold, rt.x, r̂1)
sq = sum(abs2.(logrtr1 .- logrtr̂1), dims = 1)
return mean((r1.mask .* sq) ./ ((1+eps) .- t).^pow)
=#

#Trying Bhattacharyya
#Does better than log CE, but still not great
#return mean(abs2.(safe_sqrt.(r̂1) .- safe_sqrt.(r1.x)))

#=
sq = sum(abs2.(safe_sqrt.(r̂1) .- safe_sqrt.(r1.x)), dims = 1)
return mean((r1.mask .* sq) ./ ((1+eps) .- t).^pow)
=#

#Trying the manifold distance between p and q.
#=
@inbounds for i in eachindex(p, q)
    sumsqrt += sqrt(p[i] * q[i])
end
return 2 * acos(sumsqrt)
=#
#sq = T(2) .* arccos_approx.(sum(safe_sqrt.(r̂1 .* r1.x), dims = 1))
#return mean((r1.mask .* sq) ./ ((1+eps) .- t).^pow)

#This does not do well on the one real task I've tried it on:
function loss_func(
    m::ProbabilitySimplex, #manifold
    r̂1::AbstractArray{T}, #Predicted end point (as array)
    r1::VectorFlowState{T}, #True end point
    rt::VectorFlowState{T}, #Starting point
    t::Union{T,AbstractArray{T,2}},
    eps,
    pow) where T <: Real
    eps2 = T(0.0001)
    sq = T(2) .* arccos_approx.(sum(sqrt.(floor_eps.(r̂1,eps2) .* floor_eps.(r1.x,eps2)), dims = 1))
    return sq ./ ((1+eps) .- t).^pow
    #return mean((r1.mask .* sq) ./ ((1+eps) .- t).^pow) / (T(mean(r1.mask))  + T(0.0001f0))
end

#Notice how the non-trivial manifold loss requires an extra point
function loss(
    f::ManifoldVectorFlow, #Flow
    r̂1::AbstractArray{T}, #Predicted end point (as array)
    r1::VectorFlowState{T}, #True end point
    rt::VectorFlowState{T}, #Starting point
    t::Union{T,AbstractArray{T,2}};
    masked = false,
    eps = T(0.01), pow = 2
    ) where T <: Real
    size(t, 1) != 1 && throw(ArgumentError("t must be a row vector"))
    ndims(r1.mask) != 1 && throw(ArgumentError("mask must be a columns vector"))

    #Now set up to allow dispatching to different manifolds
    site_losses = loss_func(f.manifold, r̂1, r1, rt, t, eps, pow)

    #@show size(site_losses)
    #@show size(r1.mask' .* site_losses)

    if masked
        return mean(r1.mask' .* site_losses) / (T(mean(r1.mask)) + T(0.0001f0))
    else
        return mean(site_losses)
    end
    #return loss_func(f.manifold, r̂1, r1, rt, t, eps, pow)
end

#In these, t needs to be a vector, because GNNs (our main use case) batch by concatenating
#and we want to batch different t values

#These are reparameterized so that we're predicting the true (t=1) state, not the change in state
#It might have been my ODE, but in my small testing this wasn't nearly as good
#For Euclidean, at least, we can train a model that learns the change in state, but it
#would be confusing to combine that with a model that learns the terminal rotation
#and less amenable to pre-training

#Note: the model estimates are just regular arrays, not FlowStates, because we don't want too
#much casting etc with what comes off the GPU
function loss(
    f::Union{EuclideanFlow,RelaxedDiscreteFlow},
    x̂1::AbstractArray{T},
    x1::VectorFlowState{T},
    t::Union{T,AbstractArray{T,2}};
    masked = false,
    eps = T(0.01), pow = 2
    ) where T <: Real
    size(t, 1) != 1 && throw(ArgumentError("t must be a row vector"))
    ndims(x1.mask) != 1 && throw(ArgumentError("mask must be a columns vector"))

    site_losses = mean((x̂1 .- x1.x).^2, dims = 1) ./ ((1+eps) .- t).^pow
    #@show size(site_losses)
    #@show size(x1.mask' .* site_losses)

    if masked
        return mean(x1.mask' .* site_losses) / (T(mean(x1.mask)) + T(0.0001f0))
    else
        return mean(site_losses)
    end
    #return mean(x1.mask' .* site_losses) / (T(mean(x1.mask)) + T(0.0001f0))
end

#The Euclidean case doesn't actually require the current state (xt) but we include it in case we want things to work when we don't know what kind of Flow we're using
function loss(
    f::Union{EuclideanFlow,RelaxedDiscreteFlow},
    x̂1::AbstractArray{T},
    x1::VectorFlowState{T},
    xt::VectorFlowState{T},
    t::Union{T,AbstractArray{T,2}};
    masked = false,
    eps = T(0.01), pow = 2
    ) where T <: Real
    return loss(f, x̂1, x1, t, eps = eps, pow = pow, masked = masked)
    #mse((x1hat - xt)/(1-t),(x1 - x0))
    #return mean(x1.mask' .* mean((((x̂1 .- xt.x) ./ ((1+eps) .- t)) .- ((x1.x .- xt.x) ./ ((1+eps) .- t))) .^ 2 , dims = 1)) / (T(mean(x1.mask)) + T(0.0001f0))
end



#Trying the axisangle trick from https://github.com/jasonkyuyim/se3_diffusion/blob/53359d71cfabc819ffaa571abd2cef736c871a5d/experiments/train_se3_diffusion.py#L595
#This has given the best performance, empirically, on toy tests and large models
function loss(
    f::RotationalFlow,
    r̂1::AbstractArray{T}, #Predicted end point (as array)
    r1::MatrixFlowState{T}, #True end point
    rt::MatrixFlowState{T}, #Starting point
    t::Union{T,AbstractArray{T,2}};
    masked = false,
    eps = T(0.01), pow = 2
    ) where T <: Real
    size(t, 1) != 1 && throw(ArgumentError("t must be a row vector"))
    ndims(r1.mask) != 1 && throw(ArgumentError("mask must be a columns vector"))

    rtT = batched_transpose(rt.x)
    r̂an,r̂ax = angleaxis_stack(batched_mul(r̂1, rtT))
    ran,rax = angleaxis_stack(batched_mul(r1.x, rtT))
    sq = compute_rot_loss_vec(r̂an,ran,r̂ax,rax)
    #@show size(sq)
    #@show size(r1.mask' .* sq)

    if masked
        return mean((r1.mask' .* sq) ./ ((1+eps) .- t).^pow) / (T(mean(r1.mask)) + T(0.0001f0))
    else
        return mean(sq)
    end
    #return mean((r1.mask .* sq) ./ ((1+eps) .- t).^pow) / (T(mean(r1.mask)) + T(0.0001f0))
end



#########################################################
### Glue between discrete states and continuous Flows ###
##      Note: I've got no idea if this is sensible     ##
#########################################################

#Handles discrete tokens and their conversino to continuous points
struct Relaxation{T, A<:AbstractArray{T, 2}}
    k::Int
    m::A
    #σ::T
    alph2ind::Dict
    ind2alph::Dict
end

#This weirdness on the "m" is to avoid the corners, in case you're using a manifold.
function Relaxation(alph::AbstractVector; T=Float32, m = softmax(Matrix(Float32(5+log(length(alph)))*I, length(alph), length(alph))) )#, σ = T(0.1),)
    k = size(alph, 1) #Alphabet length
    alph2ind = Dict(zip(alph, 1:k))
    ind2alph = Dict(zip(1:k, alph))
    return Relaxation(k, m, #=T(σ),=# alph2ind, ind2alph)
end

function relax(seq, r::Relaxation)
    seq = [r.alph2ind[a] for a in seq]
    c = r.m[:, seq]
    return c # .+ randn(typeof(r.σ),size(c)) .* r.σ
end

#=One hot version
function relax(seq::OneHotArray, r::Relaxation)
    c = AAmap.m * seq #double-check this mat mul
    return c .+ randn(typeof(r.σ),size(c)) .* r.σ
end
=#

#Finds the index of the closest column in r.m to each column in points 
function unrelax(points::AbstractArray, r::Relaxation; L = 2)
    return [r.ind2alph[argmin(sum((r.m .- points[:,i]).^L, dims = 1)[:])] for i in 1:size(points, 2)]
end
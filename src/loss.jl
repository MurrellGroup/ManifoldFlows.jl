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

approx_acos(x::T) where T = T(1.5707963) - x - x^3/6

floor_eps(x,eps) = x < eps ? eps : x

### Version without the arccos and denominator instability
function approx_stable_log(::ProbabilitySimplex, p::AbstractMatrix{T}, q::AbstractMatrix{T}) where T
    eps = T(1f-4)
    z = @. safe_sqrt(p * q)
    s = sum(z, dims=1)
    return @. (2 * approx_acos(s) / (eps + safe_sqrt((eps + 1) - abs2(s)))) * (z - s * p)
end


#This does not do well on the one real task I've tried it on:
function loss_func(
    ::ProbabilitySimplex, # manifold
    r̂1::AbstractArray{T}, # predicted end point (as array)
    r1::BatchedState{T}, # true end point
    rt::BatchedState{T}, # starting point
    t::Union{T,AbstractVector{T}},
    eps, pow
) where T
    eps₂ = T(1f-4)
    sq = @. 2 * approx_acos($dropdims($sum(sqrt(floor_eps(r̂1, eps₂) * floor_eps(flatarray(r1), eps₂)), dims=1), dims=1))
    return @. sq / ((1 + eps) - t)^pow
    #return mean((r1.mask .* sq) ./ ((1+eps) .- t).^pow) / (T(mean(r1.mask))  + T(0.0001f0))
end

#Notice how the non-trivial manifold loss requires an extra point
function loss(
    flow::Flow,
    r̂1::AbstractArray{T}, # predicted end point (as array)
    r1::BatchedState{T}, # true end point
    rt::BatchedState{T}, # starting point
    t::Union{T,AbstractVector{T}};
    masked = false,
    eps = T(0.01), pow = 2
) where T
    site_losses = loss_func(flow.manifold, r̂1, r1, rt, t, eps, pow)
    return masked ?
        mean(r1.mask .* site_losses) / (T(mean(r1.mask)) + T(1f-4)) :
        mean(site_losses)
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
    x1::BatchedState{T},
    t::Union{T,AbstractArray{T,2}};
    masked = false,
    eps = T(0.01), pow = 2
) where T <: Real
    site_losses = dropdims(mean(abs2, x̂1 .- flatarray(x1), dims=1), dims=1) ./ ((1 + eps) .- t) .^ pow
    return masked ?
        mean(x1.mask' .* site_losses) / (T(mean(x1.mask)) + T(1f-4)) :
        mean(site_losses)
end

"""
    loss(f::Flow, x̂1::A, x1::A, xt::A, t::T; masked = false, eps = T(0.01), pow = 2) where A<:FlowState{T}

Compute a loss between the predicted end point x̂1 and the true end point x1, given the starting point xt and the time t.
These should be considered as "default" losses, and you might need to adapt and adjust them for your problem.
"""
function loss(
    f::Union{EuclideanFlow,RelaxedDiscreteFlow},
    x̂1::AbstractArray{T},
    x1::BatchedState{T},
    xt::BatchedState{T}, # The Euclidean case doesn't actually require the current state (xt) but we include it in case we want things to work when we don't know what kind of Flow we're using
    t::Union{T,AbstractArray{T,2}};
    masked = false,
    eps = T(0.01), pow = 2
) where T <: Real
    return loss(f, x̂1, x1, t; eps, pow, masked)
    #mse((x1hat - xt)/(1-t),(x1 - x0))
    #return mean(x1.mask' .* mean((((x̂1 .- xt.x) ./ ((1+eps) .- t)) .- ((x1.x .- xt.x) ./ ((1+eps) .- t))) .^ 2 , dims = 1)) / (T(mean(x1.mask)) + T(0.0001f0))
end

#Trying the axisangle trick from https://github.com/jasonkyuyim/se3_diffusion/blob/53359d71cfabc819ffaa571abd2cef736c871a5d/experiments/train_se3_diffusion.py#L595
#This has given the best performance, empirically, on toy tests and large models
function loss(
    ::RotationalFlow,
    r̂1::AbstractArray{T}, # predicted end point (as array)
    r1::BatchedState{T}, # true end point
    rt::BatchedState{T}, # starting point
    t::Union{T,AbstractVector{T}};
    masked = false,
    eps = T(0.01), pow = 2
) where T <: Real
    rtT = batched_transpose(flatarray(rt))
    r̂an,r̂ax = angleaxis_stack(batched_mul(r̂1, rtT))
    ran,rax = angleaxis_stack(batched_mul(flatarray(r1), rtT))
    sq = compute_rot_loss_vec(r̂an, ran, r̂ax, rax) ./ ((1 + eps) .- t) .^ pow

    return masked ?
        mean(r1.mask' .* sq) / (T(mean(r1.mask)) + T(0.0001f0)) :
        mean(sq)
    #return mean((r1.mask .* sq) ./ ((1+eps) .- t).^pow) / (T(mean(r1.mask)) + T(0.0001f0))
end

#---Tested for probability simplex---
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
#sq = T(2) .* approx_acos.(sum(safe_sqrt.(r̂1 .* r1.x), dims = 1))
#return mean((r1.mask .* sq) ./ ((1+eps) .- t).^pow)

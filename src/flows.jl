using Manifolds
using ArraysOfArrays: VectorOfSimilarArrays, nestedview, flatview, innersize
using FillArrays: Fill

struct Flow{M<:AbstractManifold,S}
    manifold::M
    schedule::S
end

struct State{T,N,A<:AbstractArray{T,N}}
    x::A
end

struct BatchedState{T,N,B<:VectorOfSimilarArrays{T,N},M<:AbstractVector{Bool}} <: AbstractVector{State{T,N}}
    xs::B
    mask::M
end

statesize(flow::Flow) = representation_size(flow.manifold)
statesize(state::State) = size(state.x)
statesize(states::BatchedState) = innersize(states.xs)

flatarray(state::State) = state.x
flatarray(states::BatchedState) = flatview(states.xs)

Flow(manifold::AbstractManifold) = Flow(manifold, identity)

function BatchedState(xs::VectorOfSimilarArrays, mask=fill(true, length(xs)))
    @assert eachindex(xs) == eachindex(mask)
    return BatchedState(xs, mask)
end

function BatchedState(xs::AbstractArray{T,N}, args...) where {T,N}
    return BatchedState(nestedview(xs, N-1), args...)
end

function BatchedState(states::AbstractVector{<:State}, args...)
    return BatchedState(VectorOfSimilarArrays(map(state -> flatarray(state), states)), args...)
end

Base.size(b::BatchedState) = size(b.xs)
Base.getindex(b::BatchedState, i::Integer) = State(b.xs[i])
Base.setindex!(b::BatchedState, i::Integer, v::State) = b.xs[i] = v.x

Base.copy(b::BatchedState) = BatchedState(copy(b.xs), copy(b.mask))


### Flow Behavior

const EuclideanFlow = Flow{<:Euclidean}
const RelaxedDiscreteFlow = Flow{<:ProbabilitySimplex}
const RotationalFlow = Flow{<:SpecialOrthogonal}
const LinearFlow = Union{EuclideanFlow,RelaxedDiscreteFlow}

# for yeeting an array into higher dimensions
shiftdims(x::AbstractArray, n::Integer) = reshape(x, ntuple(Returns(1), n)..., size(x)...)

function interpolate(flow::Flow, x₀::State{T,N}, x₁::State{T,N}, t::Real) where {T,N}
    t′ = flow.schedule(T(t))
    γ = shortest_geodesic(flow.manifold, flatarray(x₀), flatarray(x₁))
    return State(γ(t′))
end

function interpolate(flow::Flow, x₀::BatchedState{T,N}, x₁::BatchedState{T,N}, t::AbstractVector) where {T,N}
    t′ = flow.schedule.(T.(t))
    xₜ = BatchedState(similar(x₁.xs), x₀.mask .& x₁.mask)
    for i in eachindex(xₜ)
        xₜ[i] = interpolate(flow, x₀[i], x₁[i], t′[i])
    end
    return xₜ
end

function interpolate(flow::Union{EuclideanFlow,RelaxedDiscreteFlow}, x₀::BatchedState{T,N}, x₁::BatchedState{T,N}, t::AbstractVector) where {T,N}
    t′ = shiftdims(flow.schedule.(T.(t)), N)
    xₜ = BatchedState(t′ .* flatarray(x₁) + (1 .- t′) .* flatarray(x₀), x₀.mask .& x₁.mask)
    return xₜ
end

function interpolate(flow::Flow{<:SpecialOrthogonal{3}}, x₀::BatchedState{T,N}, x₁::BatchedState{T,N}, t::AbstractVector) where {T,N}
    t′ = flow.schedule.(T.(t))
    display(size(flatarray(x₀)))
    display(size(flatarray(x₁)))
    display(size(t′))
    xₜ = BatchedState(slerp_stack(flatarray(x₀), flatarray(x₁), t′), x₀.mask .& x₁.mask)
    return xₜ
end

interpolate(flow, x₀, x₁, t::Real) = interpolate(flow, x₀, x₁, Fill(t, size(x₀)))


### Perturbation
#This throws inexact error for probability simplex sometimes.

"""
    perturb!([rng=default_rng()], f::Flow, state, σ)

Perturb the flow by a random amount, respecting the manifold, but do not change states where mask is false.
"""
function perturb!(rng::AbstractRNG, flow::Flow, state::State{T}, σ::Real) where T
    # note from old code: this throws inexact error for probability simplex sometimes.
    rv = rand(rng, flow.manifold, vector_at=state.x, σ=T(σ)) # Random vector in the tangent space
    state.x .= exp(flow.manifold, state.x, rv) # Exponential map of rv
    return state
end

function perturb!(rng::AbstractRNG, ::LinearFlow, state::State{T}, σ::Real) where T
    state.x .+= T(σ) * randn(rng, T, size(state.x))
    return state
end

function perturb!(rng::AbstractRNG, ::Flow{<:SpecialOrthogonal{3}}, state::State, σ::Real)
    state.x .= state.x * randrot(rng, σ)
    return state
end

function perturb!(rng::AbstractRNG, ::LinearFlow, states::BatchedState{T,N}, σ::Real) where {T,N}
    flatarray(states) .+= shiftdims(σ * states.mask, N) .* randn(rng, T, size(flatarray(states)))
    return states
end

function perturb!(rng::AbstractRNG, flow::Flow, states::BatchedState,  σ::Real)
    foreach(states, states.mask) do state, m
        m && perturb!(rng, flow, state, σ)
    end
    return states
end

perturb!(flow, state, σ) = perturb!(Random.default_rng(), flow, state, σ)


### Relaxation

struct Relaxation{E<:AbstractMatrix,T}
    k::Int
    embeddings::E
    token_to_index::Dict{T,Int}
    index_to_token::Dict{Int,T}
end

initial_embeddings(n::Integer, T::Type=Float32) = softmax(collect(T(5+log(n))*I(n)))

function Relaxation(
    tokens::AbstractVector,
    embeddings::AbstractMatrix = initial_embeddings(length(tokens))
)
    @assert allunique(tokens)
    k = length(tokens)
    token_to_index = Dict(zip(tokens, 1:k))
    index_to_token = Dict(zip(1:k, tokens))
    return Relaxation(k, embeddings, token_to_index, index_to_token)
end

relax(r::Relaxation, tokens::AbstractVector) =
    BatchedState(r.embeddings[:, [r.token_to_index[token] for token in tokens]])

function unrelax(r::Relaxation, states::BatchedState{T,1}, L::Real=2) where T
    tokens = map(eachindex(states)) do i
        p = flatarray(states[i])
        j = argmin(vec(sum((r.embeddings .- p).^L, dims=1)))
        r.index_to_token[j]
    end
    return tokens
end


### Generative Flow

#Flow for tuples, where the model must take a tuple, do joint inference, and return a tuple of data arrays.
"""
    flow(f::Flow, x₀::State, model, steps=100, tracker=NullTracker())

Samples from the distribution implied by the model under the Flow f, starting from x0. f and x0 can also be tuples, with matches components.
steps can be an integer, in which case a linear schedule is used, or a vector of times to specify the schedule. If a tracker is supplied, the sample paths are tracked.
"""
function flow(
    f::Tuple{Vararg{Flow}}, X₀::Tuple{Vararg{BatchedState{T}}}, model, steps::AbstractVector;
    tracker::Function=Returns(nothing)
) where T
    Xₜ = copy(X₀) # capitalized cause tuple
    steps = map(T, steps)
    for (s₁, s₂) in zip(steps, steps[begin+1:end])
        t = (s₁ + s₂) / 2
        Δt = s₂ - s₁
        ts = Tuple(t .* ones(T, 1, length(x₀)) for x₀ in X₀)
        X̂₁ = copy(X₀)
        res = model(ts, Xₜ)
        foreach(res, X̂₁) do r, x̂₁
            flatarray(x̂₁) .= r
        end
        tracker(t, Xₜ, X̂₁)
        Xₜ = isapprox(t, 1) ? X̂₁ : interpolate(f, Xₜ, X̂₁, min(1, Δt/(1-t)))
    end
    X₁ = Xₜ
    return X₁
end

flow(f, x0, model, steps::Integer=100; kwargs...) =
    flow(f, x0, model, [range(0, 1, steps); 1]; kwargs...)

flow(f::Flow, x0::BatchedState, model, args...; kwargs...) =
    flow((f,), (x0,), (t,xt) -> (model(t[1],xt[1]),), args...; kwargs...)[1]


# GPU compatibility

Adapt.adapt_structure(to, state::State) = State(Adapt.adapt(to, state.x))
Adapt.adapt_structure(to, states::BatchedState) = BatchedState(Adapt.adapt(to, states.xs), Adapt.adapt(to, states.mask))


# Tracking

struct Tracker <: Function
    t::Vector
    xt::Vector
    x̂1::Vector
end

Tracker() = Tracker([], [], [])

function (tracker::Tracker)(t, xt, x̂1)
    push!(tracker.t, t)
    push!(tracker.xt, xt)
    push!(tracker.x̂1, x̂1)
    return nothing
end

function stack_tracker(tracker, field; tuple_index = 1)
    return stack([data[tuple_index] for data in getproperty(tracker, field)])
end

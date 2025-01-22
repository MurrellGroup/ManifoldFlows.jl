#Geodesic rotational interpolation
"""
    slerp(x₀, x₁, t) -> xₜ

Interpolation between two 3×3×N stacks of rotation matrices, performing
Spherical linear interpolation (Slerp) on each pair of rotation matrices.
"""
function slerp_stack(x₀::AbstractArray{T,3}, x₁::AbstractArray{T,3}, t::AbstractVector{T}) where T
    xₜ = similar(x₁)
    @assert (axes(x₀) == axes(x₁) == axes(xₜ)) && (axes(x₀, 3) == axes(t, 1))
    @inbounds @views for i in axes(x₀, 3)
        xₜ[:,:,i] = QuatRotation(slerp(QuatRotation(x₀[:,:,i]), QuatRotation(x₁[:,:,i]), t[i]))
    end
    return xₜ
end

slerp_stack(x₀::AbstractArray{T,N}, x₁::AbstractArray{T,N}, t::T) where {T,N} =
    slerp_stack(x₀, x₁, fill(t, size(x₀)[3:end]))


# Logarithmic map. Needs to work on a GPU, and with Zygote
"""
    log_rot_stack(R::AbstractArray{T,3})

Calculate the logarithmic map of each rotation in a stack of 3×3×N rotation matrices.
log_rot_stack(A) is calculating the same thing as stack([log(A[:,:,i]) for i in 1:size(A,3)])
"""
function log_rot_stack(R::AbstractArray{T,3}) where T
    @assert size(R, 1) == size(R, 2) == 3

    @views tr_R = R[1,1,:] + R[2,2,:] + R[3,3,:]

    Θ = @. acos(clamp((tr_R - 1) / 2, T(-1), T(1)))
    coeff = @. (abs(Θ) > T(1e-5)) * Θ / 2sin(Θ)

    @views omega₁₂ = reshape(coeff .* (R[1,2,:] .- R[2,1,:]), 1, 1, :)
    @views omega₁₃ = reshape(coeff .* (R[1,3,:] .- R[3,1,:]), 1, 1, :)
    @views omega₂₃ = reshape(coeff .* (R[2,3,:] .- R[3,2,:]), 1, 1, :)
    zero_vec = zeros(T, 1, 1, size(R, 3))

    return [
        zero_vec  omega₁₂  omega₁₃
        -omega₁₂ zero_vec  omega₂₃
        -omega₁₃ -omega₂₃ zero_vec
    ]
end


"""
    randrot(σ²)

Generate a random rotation matrix, with each element drawn from the
exponential map of a normal distribution with variance σ².
"""
function randrot(rng::Random.AbstractRNG, σ²::Real)
    σ = √float(σ²)
    T = typeof(σ)
    return QuatRotation(exp(quat(0, randn(rng, T) * σ, randn(rng, T) * σ, randn(rng, T) * σ)))
end

randrot(σ²::Real) = randrot(Random.default_rng(), σ²)


"""
    identity_rot_stack(T,N)

Generate a stack of 3×3×N identity matrices of type T.
"""
function identity_rot_stack(T, N)
    R = zeros(T, 3, 3, N)
    R .= I(3)
    return R
end

rand_rot_stack(rng,T,N) = T.(stack([rand(rng, QuatRotation) for i in 1:N]))
rand_rot_stack(T,N) = rand_rot_stack(Random.default_rng(),T,N)


"""
    quats2rots(q)

Convert a 4×N array of quaternions to a 3×3×N array of rotation matrices.
"""
function quats2rots(q::AbstractMatrix{<:Number})
    @views a, b, c, d = q[1:1, :], q[2:2, :], q[3:3, :], q[4:4, :]

    sx = 2a .* b
    sy = 2a .* c
    sz = 2a .* d
    xx = 2(b.^2)
    xy = 2b .* c
    xz = 2b .* d
    yy = 2(c.^2)
    yz = 2c .* d
    zz = 2(d.^2)

    r1 = 1 - (yy + zz)
    r2 = xy - sz
    r3 = xz + sy
    r4 = xy + sz
    r5 = 1 - (xx + zz)
    r6 = yz - sx
    r7 = xz - sy
    r8 = yz + sx
    r9 = 1 - (xx + yy)

    return [
        r1 r2 r3
        r4 r5 r6
        r7 r8 r9
    ]
end

"""
    bcds2quats(bcd::AbstractMatrix)

Convert a 3×N array of partial quaternions to an array of (flat) unit quaternions.
"""
function bcds2quats(bcd::AbstractMatrix{T}, a::T=T(1)) where T<:Number
    norms = sqrt.(a .+ sum(abs2, bcd, dims=1))
    return vcat(a ./ norms, bcd ./ norms)
end


"""
    angleaxis_stack(R::AbstractArray)

Convert a stack of 3×3×N rotation matrices to a row vector of angles and a 3×N matrix of axes?
"""
function angleaxis_stack(R::AbstractArray{T,3}) where T
    @views tr_R = R[1,1,:] + R[2,2,:] + R[3,3,:]

    Θ = @. acos(clamp((tr_R - 1) / 2, T(-0.99), T(0.99)))
    coeff = @. T(0.5) / (sin(Θ) + T(1e-5))

    @views axis_x = reshape(coeff .* (R[3,2,:] .- R[2,3,:]), 1, :)
    @views axis_y = reshape(coeff .* (R[1,3,:] .- R[3,1,:]), 1, :)
    @views axis_z = reshape(coeff .* (R[2,1,:] .- R[1,2,:]), 1, :)

    axis = [axis_x; axis_y; axis_z]

    return reshape(Θ, 1, :), axis
end

#This gives you a vector of losses, which you can scale, mask, etc
function compute_rot_loss_vec(
    an1hat::AbstractArray{T},
    an1::AbstractArray{T},
    ax1hat::AbstractArray{T},
    ax1::AbstractArray{T};
    an_vs_ax_weight = T(0.5)
) where T    
    axis_loss = mean((ax1hat .- ax1) .^ 2, dims=1) # Summing along rows
    angle_loss = (an1hat .- an1) .^ 2
    rot_loss = (an_vs_ax_weight .* angle_loss) .+ (1 - an_vs_ax_weight) .* axis_loss
    return rot_loss
end


### Optimal transport stuff ###
"""
    sinkhorn(C, λ; iters=50, standardize = true)

Returns the "plan" from the Sinkhorn algorithm for optimal transport, given a cost matrix C and regularization parameter λ.
If standardize is true, then the cost matrix is standardized by dividing by its standard deviation which can help with numerical stability.
"""
function sinkhorn(C, λ::T; iters=50, standardize = true) where T
    if standardize
        C = C ./ std(C)
    end
    r,c = size(C)
    a, b, u, v = ones(T, r)./r, ones(T, c)./c, ones(T, r), ones(T, c)
    K = exp.(-C ./ λ)
    for _ in 1:iters
        u, v = a ./ (K * v), b ./ (K' * u)
    end
    u .* K .* v'
end

#=
#A possible alternative to sinkhorn: Algorithm from Assignment.jl, pointing to this ref:
[2] D. F. Crouse, "Advances in displaying uncertain estimates of multiple
    targets," in Proceedings of SPIE: Signal Processing, Sensor Fusion, and
    Target Recognition XXII, vol. 8745, Baltimore, MD, Apr. 2013

M = distance_matrix(rand_locs, locs)
matches = find_best_assignment(M)
return rand_locs[:,matches.col4row]
=#

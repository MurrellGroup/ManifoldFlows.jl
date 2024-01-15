#Geometry

#Geodesic rotational interpolation
#Off GPU
#Note: could maybe replace this with whatever Manifolds.jl gets you for free.
"""
slerp(a,b,t)

Interpolation between two 3-by-3-by-N stacks of rotation matrices,performing
Spherical linear interpolation (Slerp) on each pair of rotation matrices.
"""
function slerp_stack(x0::AbstractArray{T, 3} , x1::AbstractArray{T, 3}, t::AbstractArray{T}) where T
    @assert (size(x0) == size(x1)) && (length(size(x0)) == 3) && (size(x0)[1:2] == (3,3)) && (size(x0)[3] == length(t))
    new_x = copy(x1)
    for i in 1:size(x0,3)
        new_x[:,:,i] .= Matrix(QuatRotation(slerp(QuatRotation(x0[:,:,i]),QuatRotation(x1[:,:,i]),t[:][i])))
    end
    return new_x
end
slerp_stack(x0::AbstractArray{T, 3} , x1::AbstractArray{T, 3}, t::T) where T = slerp_stack(x0,x1,fill(t, size(x0,3)))


#approxacos(x) = pi/2 - x - (x^3)/6 + (x^5)/120

#Logarithmic map. Needs to work on a GPU, and with Zygote
"""
log_rot_stack(R::AbstractArray{T, 3})

Calculate the logarithmic map of each rotation in a stack of 3-by-3-by-N rotation matrices.

log_rot_stack(A) is calculating the same thing as stack([log(A[:,:,i]) for i in 1:size(A,3)])
"""
function log_rot_stack(R::AbstractArray{T, 3}) where T
    tr_R = R[1,1,:] .+ R[2,2,:] .+ R[3,3,:]

    theta = acos.(clamp.((tr_R .- 1) ./ 2,T(-1),T(1)))

    small_theta = abs.(theta) .< 1e-5
    sin_theta = sin.(theta)
    coeff = @. (1 - small_theta) * theta / (2 * sin_theta)

    omega_12 = reshape(coeff .* (R[1,2,:] .- R[2,1,:]), 1, 1, :)
    omega_13 = reshape(coeff .* (R[1,3,:] .- R[3,1,:]), 1, 1, :)
    omega_23 = reshape(coeff .* (R[2,3,:] .- R[3,2,:]), 1, 1, :)
    zero_vec = reshape(zeros(T, length(theta)), 1, 1, :)
    
    omega = reshape(vcat(zero_vec, -omega_12, -omega_13,
                omega_12, zero_vec, -omega_23,
                omega_13, omega_23, zero_vec) , 3, 3, :)

    return omega
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
rot_identity_stack(T,N)

Generate a stack of 3-by-3-by-N identity matrices of type T.
"""
function rot_identity_stack(T,N)
    R = zeros(T, 3, 3, N)
    R .= Matrix(I, 3, 3)
    return R
end

rand_rot_stack(rng,T,N) = T.(stack([rand(rng, QuatRotation) for i in 1:N]))
rand_rot_stack(T,N) = rand_rot_stack(Random.default_rng(),T,N)

export rand_rot_stack

#T.(stack([rand(FrameDance.QuatRotation) for i in 1:l]))

"""
    quats2rots(q)

Convert a 4-by-N array of quaternions to a 3-by-3-by-N array of rotation matrices.
"""
function quats2rots(q)
 
    sx = 2q[1, :] .* q[2, :]
    sy = 2q[1, :] .* q[3, :]
    sz = 2q[1, :] .* q[4, :]

    xx = 2q[2, :].^2
    xy = 2q[2, :] .* q[3, :]
    xz = 2q[2, :] .* q[4, :]

    yy = 2q[3, :].^2
    yz = 2q[3, :] .* q[4, :]
    zz = 2q[4, :] .^ 2  
    
    r1 = reshape(1 .- (yy .+ zz), 1, :)
    r2 = reshape(xy .- sz, 1, :)
    r3 = reshape(xz .+ sy, 1, :)

    r4 = reshape( xy .+ sz, 1, :)
    r5 = reshape(1 .- (xx .+ zz), 1, :)
    r6 = reshape( yz .- sx, 1, :)

    r7 = reshape(xz .- sy, 1, :)
    r8 = reshape(yz .+ sx, 1, :)
    r9 = reshape(1 .- (xx .+ yy), 1, :)

    return reshape(vcat(r1, r4, r7, r2, r5, r8, r3, r6, r9), 3, 3, :)
end

"""
    bcds2quats(bcd::AbstractArray{<: Real, 2})

Convert a 3xN array of partial quaternions to an array of (flat) unit quaternions.
"""
function bcds2quats(bcd::AbstractArray{<: Real, 2})
    denom = sqrt.(1 .+ bcd[1,:].^2 .+ bcd[2,:].^2 .+ bcd[3,:].^2)
    return vcat((1 ./ denom)', bcd ./ denom')
end


"""
angle_axis_stack(R::AbstractArray{T, 3})

Convert a stack of 3-by-3-by-N rotation matrices to a row vector of angles and a 3-by-N matrix of axis...es?
"""
function angleaxis_stack(R::AbstractArray{T, 3}) where T
    eps = T(0.00001)  # Numerical stability threshold
    tr_R = R[1,1,:] .+ R[2,2,:] .+ R[3,3,:]
    
    # Compute angle
    theta = acos.(clamp.((tr_R .- 1) ./ 2,T(-0.99),T(0.99)))
    sin_theta = sin.(theta)

    # Compute coefficient with conditional for numerical stability
    coeff = T(0.5) ./ (sin_theta .+ eps)
    
    # Compute axis components
    axis_x = reshape(coeff .* (R[3,2,:] .- R[2,3,:]), 1, :)
    axis_y = reshape(coeff .* (R[1,3,:] .- R[3,1,:]), 1, :)
    axis_z = reshape(coeff .* (R[2,1,:] .- R[1,2,:]), 1, :)
    
    # Combine into a single array
    axis = vcat(axis_x, axis_y, axis_z)
    
    # Reshape theta for concatenation
    theta = reshape(theta, 1, :)
    
    return theta, axis
end

#This gives you a vector of losses, which you can scale, mask, etc
function compute_rot_loss_vec(
    an1hat::AbstractArray{T},
    an1::AbstractArray{T},
    ax1hat::AbstractArray{T},
    ax1::AbstractArray{T};
    an_vs_ax_weight = T(0.5)) where T
    
    axis_loss = mean((ax1hat .- ax1) .^ 2, dims=1) # Summing along rows
    angle_loss = (an1hat .- an1) .^ 2
    rot_loss = (an_vs_ax_weight .* angle_loss) .+ (1 - an_vs_ax_weight) .* axis_loss

    return rot_loss
end


### Optimal transport stuff ###
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

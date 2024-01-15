using ManifoldFlows, Plots, Flux, StatsBase

#Allow Plots to run headless
ENV["GKSwstype"] = "100"

#Setting up "target" distribution (which would usually be your data)...
function target_sample()
    l = rand()*14
    return Float32.([exp(-l/7)*sin(l) ,exp(-l/7)*cos(l)] * 5.0)
end
#...and the "zero" distribution (which is the simple distribution that we bridge to).
zero_sample() = randn(Float32,2) .* 2f0

#Plotting the two distributions
sample_size = 2000
xsamp = stack([zero_sample() for i in 1:sample_size])
pl = scatter(xsamp[1,:],xsamp[2,:], markerstrokewidth = 0, color = "blue", alpha = 0.8, label = "q0")
xsamp = stack([target_sample() for i in 1:sample_size])
scatter!(xsamp[1,:],xsamp[2,:], markerstrokewidth = 0, color = "red", alpha = 0.8, label = "q1")
savefig(pl,"spiral_problem.svg")

#Setting up a simple network:
af = swish
hs = 128
net = Chain(
    Dense(21,hs,af),
    Dense(hs,hs,af),
    Dense(hs,hs,af),
    Dense(hs,hs,af),
    Dense(hs,hs,af),Dense(hs,2))
hack_rff(x) = vcat(sin.(x), cos.(x), sin.(x .* 3), cos.(x .* 3), sin.(x .* 7), cos.(x .* 7)) #6 per input
v(t,x) = net(vcat(t,x,hack_rff(t),hack_rff(x)))

#Optimizer...
opt = Flux.Optimiser(Flux.WeightDecay(0.0001f0), Flux.AdamW(0.003f0))
ps = Flux.params(net)
batch_size = 32

#Defining the Flow:
f = EuclideanFlow()

lsum = 0.0f0
for batch in 1:250000
    #Set up the training sample pairs
    x0 = VectorFlowState(stack([zero_sample() for i in 1:batch_size]))
    x1 = VectorFlowState(stack([target_sample() for i in 1:batch_size]))    
    t = rand(Float32, 1, batch_size) #Note: t must be a ROW vector
    xt = interpolate(f,x0,x1,t)
    
    # Calculate gradients and update the model
    l,grads = Flux.withgradient(ps) do
        loss(f,v(t,xt),x1,t)
    end
    Flux.Optimise.update!(opt, ps, grads)

    #Watch the loss bounce around
    lsum += l
    if mod(batch,2000)==0
        opt[2][1].eta *= 0.975f0 #Learning rate decay
        println("Batch: ", batch, "; LR: ", opt[2][1].eta ,"; loss: ", lsum/2000)
        lsum = 0.0f0
    end
end

#Inference under the trained model, starting from the zero distribution
x0 = VectorFlowState(stack([zero_sample() for i in 1:2000]))
sample_paths = Tracker()
draws = flow(f,x0, v, steps = 25, tracker = sample_paths)


#Plotting Flow inference vs the target distribution
pl = scatter(x0[1,:],x0[2,:], markerstrokewidth = 0, color = "blue", alpha = 0.2, label = "q0")
xsamp = stack([target_sample() for i in 1:2500])
scatter!(xsamp[1,:],xsamp[2,:], markerstrokewidth = 0, color = "green", alpha = 0.2, label = "q1")
scatter!(draws[1,:],draws[2,:], markerstrokewidth = 0, color = "red", alpha = 0.2, label = "flow")
savefig(pl,"spiral_solution.svg")

xt_stack = stack_tracker(sample_paths, :xt)
x̂1_stack = stack_tracker(sample_paths, :x̂1)
t_stack = stack_tracker(sample_paths, :t)

anim = @animate for i in vcat([1 for _ in 1:5], 1:size(xt_stack, 3), [size(xt_stack, 3) for i in 1:5], size(xt_stack, 3):-1:1)
    scatter(xt_stack[1,:,i], xt_stack[2,:,i], markerstrokewidth = 0.0, axis = ([], false),
    color = "blue", label = "Xt", alpha = 0.9, 
    markersize = 4.5)
    scatter!(x̂1_stack[1,:,i], x̂1_stack[2,:,i], markerstrokewidth = 0.0, axis = ([], false),
    color = "red", label = "X1|Xt", xlim = (-4,5.2), ylim = (-4.5,6.2), alpha = 0.5, 
    markersize = 3.5, legend = :topleft)
    annotate!(-2.7, 4.5, text("t = $(round(t_stack[i], digits = 2))", :black, :right, 9))
end
gif(anim, "spiral_animation.gif", fps = 15)




#########################################
###### With Optimal Transport (OT) ######
#########################################

#Distance matrix - used for OT pairings
dismat(a,b) = sum((a .- reshape(b,2,1,:)).^2, dims = 1)[1,:,:]

#Entropic regularized OT
function sinkhorn(C, λ::T; iters=50) where T
    r,c = size(C)
    a, b, u, v = ones(T, r)./r, ones(T, c)./c, ones(T, r), ones(T, c)
    K = exp.(-(C ./ std(C)) ./ λ)
    for _ in 1:iters
        u, v = a ./ (K * v), b ./ (K' * u)
    end
    u .* K .* v'
end

#Setting up a simple network:
af = swish
hs = 128
net = Chain(
    Dense(21,hs,af),
    Dense(hs,hs,af),
    Dense(hs,hs,af),
    Dense(hs,hs,af),
    Dense(hs,hs,af),Dense(hs,2))
hack_rff(x) = vcat(sin.(x), cos.(x), sin.(x .* 3), cos.(x .* 3), sin.(x .* 7), cos.(x .* 7)) #6 per input
v(t,x) = net(vcat(t,x,hack_rff(t),hack_rff(x)))

#Optimizer...
opt = Flux.Optimiser(Flux.WeightDecay(0.0001f0), Flux.AdamW(0.003f0))
ps = Flux.params(net)
batch_size = 32

#Defining the Flow:
f = EuclideanFlow()

lsum = 0.0f0
for batch in 1:250000
    #Set up the training sample pairs
    x0 = VectorFlowState(stack([zero_sample() for i in 1:batch_size]))
    x1 = VectorFlowState(stack([target_sample() for i in 1:batch_size]))    
    t = rand(Float32, 1, batch_size) #Note: t must be a ROW vector

    ######  <Optimal Transport Pairings>  ######
    #OT pairings - only needs to be done during training
    OT = sinkhorn(dismat(x0.x, x1.x), 0.01)
    any(isnan.(OT)) && continue #Skip if NaNs
    #Sample pairs from the Sinkhorn plan
    inds = stack(Tuple.(sample(CartesianIndices(OT), Weights(OT[:]), batch_size, replace = false)))
    #Re-org samples to match up the sampled pairs
    x0.x .= x0.x[:,inds[1,:]]
    x1.x .= x1.x[:,inds[2,:]]
    ######  </Optimal Transport Pairings> ######

    xt = interpolate(f,x0,x1,t)
    
    # Calculate gradients and update the model
    l,grads = Flux.withgradient(ps) do
        loss(f,v(t,xt),x1,t)
    end
    Flux.Optimise.update!(opt, ps, grads)

    #Watch the loss bounce around
    lsum += l
    if mod(batch,2000)==0
        opt[2][1].eta *= 0.975f0 #Learning rate decay
        println("Batch: ", batch, "; LR: ", opt[2][1].eta ,"; loss: ", lsum/2000)
        lsum = 0.0f0
    end
end

#Inference under the trained model, starting from the zero distribution
x0 = VectorFlowState(stack([zero_sample() for i in 1:2000]))
sample_paths = Tracker() #This is to be able to extract the sample paths for plotting
draws = flow(f,x0, v, steps = 25, tracker = sample_paths)

#Plotting Flow inference vs the target distribution
pl = scatter(x0[1,:],x0[2,:], markerstrokewidth = 0, color = "blue", alpha = 0.2, label = "q0")
xsamp = stack([target_sample() for i in 1:2500])
scatter!(xsamp[1,:],xsamp[2,:], markerstrokewidth = 0, color = "green", alpha = 0.2, label = "q1")
scatter!(draws[1,:],draws[2,:], markerstrokewidth = 0, color = "red", alpha = 0.2, label = "flow")
savefig(pl,"spiral_solution_OT.svg")

xt_stack = stack_tracker(sample_paths, :xt)
x̂1_stack = stack_tracker(sample_paths, :x̂1)
t_stack = stack_tracker(sample_paths, :t)

anim = @animate for i in vcat([1 for _ in 1:5], 1:size(xt_stack, 3), [size(xt_stack, 3) for i in 1:5], size(xt_stack, 3):-1:1)
    scatter(xt_stack[1,:,i], xt_stack[2,:,i], markerstrokewidth = 0.0, axis = ([], false),
    color = "blue", label = "Xt", alpha = 0.9, 
    markersize = 4.5)
    scatter!(x̂1_stack[1,:,i], x̂1_stack[2,:,i], markerstrokewidth = 0.0, axis = ([], false),
    color = "red", label = "X1|Xt", xlim = (-4,5.2), ylim = (-4.5,6.2), alpha = 0.5, 
    markersize = 3.5, legend = :topleft)
    annotate!(-2.7, 4.5, text("t = $(round(t_stack[i], digits = 2))", :black, :right, 9))
end
gif(anim, "spiral_animation_OT.gif", fps = 15)

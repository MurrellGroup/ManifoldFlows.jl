using ManifoldFlows, Plots, Flux, StatsBase

#Allow Plots to run headless
ENV["GKSwstype"] = "100"

#Setting up "target" distribution (which would usually be your data)...
function target_sample()
    l = rand()*14
    return Float32.([exp(-l/7)*sin(l) ,exp(-l/7)*cos(l)] * 5.0)
end
#...and the "zero" distribution (which is the simple distribution that we bridge to).
zero_sample() = randn(Float32,2) .* 2

#Plotting the two distributions
sample_size = 1000
xsamp = stack([zero_sample() for i in 1:sample_size])
pl = scatter(xsamp[1,:],xsamp[2,:], markerstrokewidth = 0, color = "blue", alpha = 0.2, label = "q0")
xsamp = stack([target_sample() for i in 1:sample_size])
scatter!(xsamp[1,:],xsamp[2,:], markerstrokewidth = 0, color = "red", alpha = 0.2, label = "q1")
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
batch_size = 64

#Defining the Flow:
f = EuclideanFlow()

lsum = 0.0f0
for batch in 1:200000
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
    if mod(batch,2500)==0
        println("Batch: ", batch, "; loss: ", lsum/2500)
        lsum = 0.0f0
    end
end

#Inference under the trained model, starting from the zero distribution
x0 = VectorFlowState(stack([zero_sample() for i in 1:1000]))
draws = flow(f,x0, v, steps = 100)

#Plotting Flow inference vs the target distribution
pl = scatter(x0[1,:],x0[2,:], markerstrokewidth = 0, color = "blue", alpha = 0.2, label = "q0")
xsamp = stack([target_sample() for i in 1:2500])
scatter!(xsamp[1,:],xsamp[2,:], markerstrokewidth = 0, color = "green", alpha = 0.2, label = "q1")
scatter!(draws[1,:],draws[2,:], markerstrokewidth = 0, color = "red", alpha = 0.2, label = "flow")
savefig(pl,"spiral_solution_square.svg")
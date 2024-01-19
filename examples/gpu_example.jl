using Pkg
Pkg.activate(".")
#Pkg.add(["Plots", "Flux", "StatsBase", "CUDA"])
#Pkg.add(url="https://github.com/MurrellGroup/ManifoldFlows.jl")

using ManifoldFlows, Plots, Flux, StatsBase, CUDA
#device!(2) #If you have more than one GPU (they're zero indexed)

ENV["GKSwstype"] = "100" #Allow Plots to run headless

#Convenience function for plotting
scatter_points!(points; label = "", color = "black") = scatter!(points[1,:],points[2,:], markerstrokewidth = 0, color = color, alpha = 0.8, label = label)

#Setting up "target" (which would usually be your data) and "zero" distributions (where the points start).
target_sample() = (l -> Float32.(randn(2).*0.05 .+ [sin(l), cos(l)]))(rand()*2*pi)
zero_sample() = rand(Float32,2)

#Plotting the two distributions
pl = plot()
scatter_points!(stack([zero_sample() for i in 1:1000]), label = "X0", color = "blue")
scatter_points!(stack([target_sample() for i in 1:1000]), label = "X1", color = "red")
savefig(pl,"square_circle_problem.svg")

#Setting up a simple NN:
hs = 256
af = leakyrelu
features(x) = vcat(x, sin.(x), cos.(x), sin.(x .* 3), cos.(x .* 3), sin.(x .* 7), cos.(x .* 7)) #Eh...
net = Chain(
    Dense(21,hs,af),
    [SkipConnection(Dense(hs,hs,af), +) for i in 1:4]...,
    Dense(hs,hs,af),Dense(hs,2)) |> gpu #Note: moved to the GPU

#A ManifoldFlows model must take a (t,Xt) pair as input and return X̂1|Xt
model(t,Xt) = net(vcat(features(t),features(Xt)))
    
ps = Flux.params(net) #For Flux to track the parameters

#Optimizer - Note: the WeightDecay param is fun to play with!
opt = Flux.Optimiser(Flux.WeightDecay(1f-5), Flux.AdamW(1f-3))

#Defining the Flow:
f = EuclideanFlow()

#Training loop
batch_size = 4096
for batch in 1:5000
    #Set up the training sample pairs
    x0 = VectorFlowState(stack([zero_sample() for i in 1:batch_size]))
    x1 = VectorFlowState(stack([target_sample() for i in 1:batch_size]))    
    t = rand(Float32, 1, batch_size) #Note: t must be a ROW vector

    xt = interpolate(f,x0,x1,t) #Interpolate between target and zero samples
    x1, t, xt = (x1, t, xt) |> gpu #Move to GPU
    
    # Calculate gradients and update the model
    l,grads = Flux.withgradient(ps) do
        loss(f,model(t,xt.x),x1,t)
    end
    Flux.Optimise.update!(opt, ps, grads)
    mod(batch, 100) == 1 && println("Batch: ", batch, "; Loss:", l)
end

#Inference under the trained model, starting from the zero distribution
x0 = VectorFlowState(stack([zero_sample() for i in 1:2000]))
#This is where the "Flow" sampling actually happens.
#Note how the model needs move input from the CPU to GPU, and move output back.
#This is because the flow maths happens CPU-side.
draws = flow(f,x0, (t,Xt) -> cpu(model(gpu(t),gpu(Xt.x))), steps = 50) 

#Plot these against the original target distribution
pl = plot()
scatter_points!(stack([target_sample() for i in 1:1000]), label = "Target", color = "red")
scatter_points!(draws, label = "Flow samples", color = "green")
savefig(pl,"square_circle_flow.svg")

#And that's it - everything below here is just visualization, a few different ways:

#If you want to track the sample "paths" during sampling, use a Tracker during the flow:
sample_paths = Tracker()
draws = flow(f,x0, (t,Xt) -> cpu(model(gpu(t),gpu(Xt.x))), steps = 50, tracker = sample_paths) #This is where the "Flow" sampling actually happens

#Plotting Flow inference vs the target distribution
xt_stack = stack_tracker(sample_paths, :xt)
x̂1_stack = stack_tracker(sample_paths, :x̂1)
t_stack = stack_tracker(sample_paths, :t)

pl = scatter(xt_stack[1,1:500,1], xt_stack[2,1:500,1], line_z=zeros(500), c = :rainbow, axis = ([], false), label = :none, markerstrokewidth = 0, markersize = 3)
plot!(xt_stack[1,1:500,:]', xt_stack[2,1:500,:]', line_z=stack([t_stack for i in 1:500]), c = :rainbow, axis = ([], false), label = :none)
scatter!(xt_stack[1,1:500,end], xt_stack[2,1:500,end], label = :none, markerstrokewidth = 0, markersize = 3, color = "red")
savefig(pl,"square_circle_paths.svg")


pl = plot(; xlabel = "t", colorbar = :none)
for i in 1:250
    plot3d!(t_stack, xt_stack[1,i,:], xt_stack[2,i,:], line_z= t_stack, c = :rainbow, label = :none, alpha = 0.5, )
end
savefig(pl,"square_circle_3Dpaths.svg")

anim = @animate for i in vcat([1 for _ in 1:5], 1:size(xt_stack, 3), [size(xt_stack, 3) for i in 1:5], size(xt_stack, 3):-1:1)
    scatter(xt_stack[1,:,i], xt_stack[2,:,i], markerstrokewidth = 0.0, axis = ([], false),
    color = "blue", label = "Xt", alpha = 0.9, 
    markersize = 4.5)
    scatter!(x̂1_stack[1,:,i], x̂1_stack[2,:,i], markerstrokewidth = 0.0, axis = ([], false),
    color = "red", label = "X̂1|Xt", xlim = (-1.2,1.2), ylim = (-1.2,1.2), alpha = 0.5, 
    markersize = 3.5, legend = :topleft)
    annotate!(-0.85, 0.75, text("t = $(round(t_stack[i], digits = 2))", :black, :right, 9))
end
gif(anim, "square_circle.gif", fps = 15)


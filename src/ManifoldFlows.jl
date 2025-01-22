module ManifoldFlows

using Rotations
using Quaternions
using LinearAlgebra
using Random
using Manifolds
using Statistics: mean, std
using Adapt: Adapt
using NNlib

include("flows.jl")
export AbstractFlow, ManifoldFlow, DiscreteFlow
export State, BatchedState
export statesize, flatarray
export Tracker, stack_tracker
export EuclideanFlow, RotationalFlow, RelaxedDiscreteFlow, ManifoldVectorFlow
export interpolate, perturb!, flow
export Relaxation, relax, unrelax

include("geometry.jl")
export rand_rot_stack, identity_rot_stack

include("loss.jl")

end

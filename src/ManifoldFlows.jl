module ManifoldFlows
#Installation
#Pkg.add(["Rotations","Quaternions","Manifolds","StatsBase","Random","LinearAlgebra","Adapt","NNlib","Statistics"])

    using Rotations
    using Quaternions
    using LinearAlgebra
    using StatsBase
    using Random
    using Manifolds
    using Statistics: mean
    using Adapt: Adapt
    using NNlib

include("flows.jl")
include("geometry.jl")

export
    EuclideanFlow,
    RotationalFlow,
    RelaxedDiscreteFlow,
    ManifoldVectorFlow,
    VectorFlowState,
    MatrixFlowState,
    interpolate,
    perturb!,
    rot_identity_stack,
    quats2rots,
    bcds2quats,
    loss,
    flow,
    Relaxation,
    relax,
    unrelax,
    ProbabilitySimplex

end

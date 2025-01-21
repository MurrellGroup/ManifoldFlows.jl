module ManifoldFlows

    using Rotations
    using Quaternions
    using LinearAlgebra
    using StatsBase
    using Random
    using Manifolds
    using Statistics: mean
    using Adapt: Adapt
    using NNlib
    using ForwardDiff: derivative
    using OneHotArrays: onecold, onehotbatch

include("flows.jl")
include("geometry.jl")

export
    EuclideanFlow,
    RotationalFlow,
    RelaxedDiscreteFlow,
    ManifoldVectorFlow,
    DiscreteFlow,
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

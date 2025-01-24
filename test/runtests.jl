using ManifoldFlows
using Test
using Rotations
using Random
using OneHotArrays: onecold, onehotbatch

#=
@testset "Geometry" begin
    Random.seed!(3)
    A = stack([Matrix(rand(QuatRotation)) for i in 1:10])
    B = stack([Matrix(rand(QuatRotation)) for i in 1:10])
    @test isapprox(ManifoldFlows.log_rot_stack(A) , stack([log(A[:,:,i]) for i in 1:size(A,3)]))
    @test isapprox(ManifoldFlows.slerp_stack(A,B,0.0), A)
    @test isapprox(ManifoldFlows.slerp_stack(A,B,1.0), B)
end
=#

@testset "Flows" begin
    #=
    Random.seed!(3)
    f = EuclideanFlow()
    x0 = VectorFlowState(zeros(Float32,2,10))
    x1 = VectorFlowState(zeros(Float32,2,10))
    xt = VectorFlowState(zeros(Float32,2,10))
    perturb!(f,x0, 0.1)
    perturb!(f,x1, 0.1)
    @test isapprox(interpolate(f,x0, x1, 0.5).x, (x0.x .+ x1.x)./2)
    @test isapprox(interpolate(f,x0, x1, 0.0).x, x0.x)
    @test isapprox(interpolate(f,x0, x1, 1.0).x, x1.x)
    @test typeof(loss(f,x0.x,x1,0.1f0)) == Float32
    @test isapprox(loss(f,x0.x,x1,0.1f0) , loss(f,x0.x,x1,xt,0.1f0))
    
    f = RotationalFlow()
    x0 = MatrixFlowState(rot_identity_stack(Float32,10))
    x1 = MatrixFlowState(rot_identity_stack(Float32,10))
    xt = MatrixFlowState(rot_identity_stack(Float32,10))
    perturb!(f,x0, 0.1f0)
    perturb!(f,x1, 0.1f0)
    perturb!(f,xt, 0.1f0)
    interpolate(f,x0, x1, 0.5f0)
    @test isapprox(interpolate(f,x0, x1, 0.0f0).x , x0.x)
    @test isapprox(interpolate(f,x0, x1, 1.0f0).x , x1.x)
    @test typeof(loss(f,x0.x,x1,xt,0.1f0)) == Float32
    
    rel = Relaxation(1:20)
    l = 20
    Mf = ManifoldVectorFlow(ProbabilitySimplex(19))
    Mx0 = VectorFlowState(Float32.(stack(rand(Mf.manifold,l))))
    Mx1 = VectorFlowState(relax(1:l, rel))
    Mxt = VectorFlowState(ManifoldFlows.softmax(zeros(Float32,20,l)))
    perturb!(Mf,Mx0, 0.1)
    perturb!(Mf,Mx1, 0.1)
    perturb!(Mf,Mxt, 0.1)
    @test isapprox(interpolate(Mf,Mx0, Mx1, 0.0).x , Mx0.x)
    @test isapprox(interpolate(Mf,Mx0, Mx1, 1.0).x , Mx1.x)
    @test unrelax(interpolate(Mf,Mx0, Mx1, 1.0).x, rel) == 1:20
    @test typeof(loss(Mf,Mx0.x,Mx1,Mxt,0.1f0)) == Float32

    @test all(isapprox.(interpolate((f,Mf),(x0,Mx0), (x1,Mx1), 0.5) , (interpolate(f,x0, x1, 0.5),interpolate(Mf,Mx0, Mx1, 0.5))))
    =#

    f = DiscreteFlow()
    k = 8  # the number of discrete states
    d = 2  # the dimension of the state
    n = 1000  # the number of samples
    x0 = MatrixFlowState(onehotbatch(rand(1:k, d, n), 1:k))
    x1 = MatrixFlowState(onehotbatch(rand(1:k, d, n), 1:k))
    @test interpolate(f, x0, x1, 0.0).x == x0.x
    @test interpolate(f, x0, x1, 1.0).x == x1.x
    xt = interpolate(f, x0, x1, 0.3)
    # xt is either x0 or x1.
    @test all((onecold(xt.x) .== onecold(x0.x)) .| (onecold(xt.x) .== onecold(x1.x)))
    # xt and x0 are in the same state if x0 and x1 are in the same state.
    @test all((onecold(x0.x) .!= onecold(xt.x)) .| (onecold(xt.x) .== onecold(x0.x)))
    # if x0 and x1 are in different states, the probability of xt being in the same state as x1 ih 0.3.
    diff = onecold(x0.x) .!= onecold(x1.x)
    @test sum(onecold(xt.x)[diff] .== onecold(x1.x)[diff]) / sum(diff) ≈ 0.3 atol=0.05
end

@testset "Samples" begin
    f = EuclideanFlow()
    x0 = VectorFlowState(zeros(Float32, 2, 10))
    model(t, xt) = randn(Float32, size(xt))
    x1 = flow(f, x0, model)
    @test size(x1) == size(x0)

    f = DiscreteFlow()
    x0 = MatrixFlowState(onehotbatch(rand(1:8, 4, 10), 1:8))
    model(t, xt) = randn(Float32, size(xt))
    x1 = flow(f, x0, model)
    @test size(x1) == size(x0)

    f1 = EuclideanFlow()
    x01 = VectorFlowState(zeros(Float32, 2, 10))
    f2 = DiscreteFlow()
    x02 = MatrixFlowState(onehotbatch(rand(1:8, 4, 10), 1:10))
    model(t, (xt1, xt2)) = (randn(Float32, size(xt1)), randn(Float32, size(xt2)))
    x11, x12 = flow((f1, f2), (x01, x02), model)
    @test size(x11) == size(x01)
    @test size(x12) == size(x02)
end

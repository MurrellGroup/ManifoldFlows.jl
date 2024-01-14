using ManifoldFlows
using Test
using Rotations
using Random

@testset "Geometry" begin
    Random.seed!(3)
    A = stack([Matrix(rand(QuatRotation)) for i in 1:10])
    B = stack([Matrix(rand(QuatRotation)) for i in 1:10])
    @test isapprox(ManifoldFlows.log_rot_stack(A) , stack([log(A[:,:,i]) for i in 1:size(A,3)]))
    @test isapprox(ManifoldFlows.slerp_stack(A,B,0.0), A)
    @test isapprox(ManifoldFlows.slerp_stack(A,B,1.0), B)
end

@testset "Flows" begin
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
end

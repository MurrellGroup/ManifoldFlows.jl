using ManifoldFlows
using Test
using Manifolds
using Statistics
using Rotations
using Random

Random.seed!(0)

@testset "ManifoldFlows.jl" begin

    @testset "geometry.jl" begin
        A = stack([Matrix(rand(QuatRotation)) for i in 1:10])
        B = stack([Matrix(rand(QuatRotation)) for i in 1:10])
        @test ManifoldFlows.slerp_stack(A, B, 0.0) ≈ A
        @test ManifoldFlows.slerp_stack(A, B, 1.0) ≈ B
        @test ManifoldFlows.log_rot_stack(A) ≈ stack([log(A[:,:,i]) for i in axes(A, 3)])
    end

    @testset "flows.jl" begin
        for D in [1, 2, 3]
            flow = Flow(Euclidean(D))
            x₀ = BatchedState(zeros(Float32, D, 10))
            x₁ = BatchedState(zeros(Float32, D, 10))
            xₜ = BatchedState(zeros(Float32, D, 10))
            perturb!(flow, x₀, 0.1)
            perturb!(flow, x₁, 0.1)
            @test flatarray(interpolate(flow, x₀, x₁, 0.0)) ≈ flatarray(x₀)
            @test flatarray(interpolate(flow, x₀, x₁, 0.5)) ≈ (flatarray(x₀) .+ flatarray(x₁)) / 2
            @test flatarray(interpolate(flow, x₀, x₁, 1.0)) ≈ flatarray(x₁)
            @test typeof(loss(flow, flatarray(x₀), x₁, 0.1f0)) == Float32
            @test loss(flow, flatarray(x₀), x₁, 0.1f0) ≈ loss(flow, flatarray(x₀), x₁, xₜ, 0.1f0)
        end

        for D in [1, 2, 3]
            f = Flow(SpecialOrthogonal(3))
            x₀ = BatchedState(identity_rot_stack(Float32, 10))
            x₁ = BatchedState(identity_rot_stack(Float32, 10))
            xₜ = BatchedState(identity_rot_stack(Float32, 10))
            perturb!(flow, x₀, 0.1f0)
            perturb!(flow, x₁, 0.1f0)
            perturb!(flow, xₜ, 0.1f0)
            interpolate(flow, x₀, x₁, 0.5f0)
            @test flatarray(interpolate(flow, x₀, x₁, 0.0f0)) ≈ flatarray(x₀)
            @test flatarray(interpolate(flow, x₀, x₁, 1.0f0)) ≈ flatarray(x₁)
            @test typeof(loss(flow, flatarray(x₀), x₁, xₜ, 0.1f0)) == Float32
        end

        k = 20
        n = 10
        relaxation = Relaxation(1:k)
        flow = Flow(ProbabilitySimplex(k-1))
        x₀ = BatchedState(Float32.(stack(rand(flow.manifold, n))))
        x₁ = BatchedState(relax(relaxation, 1:n))
        xₜ = BatchedState(ManifoldFlows.softmax(zeros(Float32, k, n)))
        perturb!(flow, x₀, 0.1)
        perturb!(flow, x₁, 0.1)
        perturb!(flow, xₜ, 0.1)
        @test flatarray(interpolate(flow, x₀, x₁, 0.0)) ≈ flatarray(x₀)
        @test flatarray(interpolate(flow, x₀, x₁, 1.0)) ≈ flatarray(x₁)
        @test unrelax(relaxation, interpolate(flow, x₀, x₁, 1.0)) == 1:n
        @test typeof(loss(flow, flatarray(x₀), x₁, xₜ, 0.1f0)) == Float32
    end

    @testset "loss.jl" begin
        k = 10
        n = 5
        manifold = ProbabilitySimplex(k-1)
        p = stack(rand(manifold, n))
        q = stack(rand(manifold, n))
        @test isapprox(ManifoldFlows.approx_stable_log(manifold, p, q), stack(ManifoldFlows.log.(Ref(manifold), eachcol(p), eachcol(q))), rtol=0.1)
    end

end

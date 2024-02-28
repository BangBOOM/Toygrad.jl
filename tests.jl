using Test

include("toygrad.jl")

w = Tensor{Float32}(
    [
        1.0f0 2.0f0
        3.0f0 4.0f0
        5.0f0 -6.0f0
    ]
)

b = Tensor{Float32}(
    [
        2.0f0
        3.0f0
        4.0f0
    ]
)

w2 = Tensor{Float32}(
    [1.0f0 2.0f0 1.0f0]
)

x = Tensor{Float32}(
    [
        2.0f0
        3.0f0
    ]
)


@testset "test relu and tlog" begin
    o = tlog(w2 * relu(w * x + b) + 0.1f0)
    one_grad!(o)
    backward!(o)

    @test o.data ≈ [3.9532]
    @test w2.grad ≈ [0.1919 0.4031 0.0000]
    @test w.grad ≈ [0.0384 0.0576; 0.0768 0.1152; 0.0 0.0]
end


@testset "test sigmoid" begin
    zero_grad!(w)
    zero_grad!(b)
    zero_grad!(w2)
    o = w2 * sigmoid(w * x + b)
    one_grad!(o)
    backward!(o)

    @test o.data ≈ [3.0179]
    @test w2.grad ≈ [1.0000 1.0000 0.0180]
    @test w.grad ≈ [9.0833e-05 1.3625e-04; 0.0000e+00 0.0000e+00; 3.5325e-02 5.2988e-02]
end

@testset "test tanh" begin
    zero_grad!(w)
    zero_grad!(b)
    zero_grad!(w2)
    o = w2 * tanh(w * x + b)
    one_grad!(o)
    backward!(o)
    @test o.data ≈ [2.0007]
    @test w2.grad ≈ [1.0000 1.0000 -0.9993]
    @test w.grad ≈ [0.0 0.0; 0.0 0.0; 0.0026817322 0.0040225983]
end
abstract type AbstractOP end

mutable struct Tensor{T<:Real}
    data::Array{T}
    grad::Array{T}
    op::Union{AbstractOP,Any}
    Tensor{T}(a::Int, b::Int) where {T<:Real} = new(randn(T, a, b), zeros(T, a, b), nothing)
    Tensor{T}(data::Array{T}) where {T<:Real} = new(data, zeros(T, size(data)), nothing)
    Tensor{T}(data::Array{T}, op::Any) where {T<:Real} = new(data, zeros(T, size(data)), op)
end

zero_grad!(t::Tensor{T}) where {T<:Real} = fill!(t.grad, zero(T))
one_grad!(t::Tensor{T}) where {T<:Real} = fill!(t.grad, one(T))

struct AddOP{T} <: AbstractOP
    left::Tensor{T}
    right::Tensor{T}
    out::Tensor{T}
end

function Base.:+(left::Tensor{T}, right::Tensor{T}) where {T}
    @assert size(left.data) == size(right.data)
    out = Tensor{T}(left.data + right.data)
    out.op = AddOP{T}(left, right, out)
    return out
end

function backward!(op::AddOP)
    op.left.grad += op.out.grad
    op.right.grad += op.out.grad
    nothing
end

struct MulOP{T} <: AbstractOP
    left::Tensor{T}
    right::Tensor{T}
    out::Tensor{T}
end

function Base.:*(left::Tensor{T}, right::Tensor{T}) where {T}
    @assert size(left.data)[2] == size(right.data)[1]
    out = Tensor{T}(left.data * right.data)
    out.op = MulOP{T}(left, right, out)
    return out
end

function backward!(op::MulOP)
    op.left.grad += op.out.grad * op.right.data'
    op.right.grad += op.left.data' * op.out.grad
    nothing
end


function backward!(tensor::Tensor)
    isnothing(tensor.op) && return
    backward!(tensor.op)
end


w = Tensor{Float32}(
    [
        1.0f0 2.0f0
        3.0f0 4.0f0
        5.0f0 6.0f0
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

o = w2 * (w * x + b)

@show o.data
one_grad!(o)
backward!(o)
# one_grad!(o)
# backward!(o)
@show w2.grad


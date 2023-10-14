abstract type AbstractOP end

mutable struct Tensor{T<:Real}
    data::Array{T}
    grad::Array{T}
    op::Union{AbstractOP,Any}
    requires_grad::Bool
    Tensor{T}(a::Int, b::Int; requires_grad::Bool=true) where {T<:Real} = new(randn(T, a, b), zeros(T, a, b), nothing, requires_grad)
    Tensor{T}(data::Array{T}; requires_grad::Bool=true) where {T<:Real} = new(data, zeros(T, size(data)), nothing, requires_grad)
    Tensor{T}(data::Array{T}, op::Any) where {T<:Real} = new(data, zeros(T, size(data)), op, true)
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

struct ReluOP{T} <: AbstractOP
    left::Tensor{T}
    out::Tensor{T}
end

function relu(x::Tensor{T}) where {T}
    out = Tensor{T}(max.(x.data, zero(T)))
    out.op = ReluOP{T}(x, out)
    return out
end

function backward!(op::ReluOP)
    op.left.grad += (op.left.data .> 0) .* op.out.grad
end

struct Log{T} <: AbstractOP
    left::Tensor{T}
    out::Tensor{T}
end

function log(x::Tensor{T}) where {T}
    out = Tensor{T}(log.(x.data))
    out.op = Log{T}(x, out)
    return out
end

function backward!(op::Log)
    op.left.grad += op.out.grad ./ op.left.data
end


# function cross_entropy(p::Tensor{T}, q::Tensor{T}) where {T}
#     out = Tensor{T}(p.data .* log.(y.data))
#     out.op = nothing
#     return out
# end



function backward!(tensor::Tensor)
    isnothing(tensor.op) && return
    backward!(tensor.op)
    hasproperty(tensor.op, :left) && backward!(tensor.op.left)
    hasproperty(tensor.op, :right) && backward!(tensor.op.right)
    nothing
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
@show w2.grad
@show w.grad

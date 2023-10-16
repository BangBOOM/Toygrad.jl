abstract type AbstractOP end
abstract type AbstractTensor end

mutable struct Tensor{T<:Real} <: AbstractTensor
    data::Union{AbstractArray{T},T}
    grad::Union{AbstractArray{T},T}
    op::Union{AbstractOP,Any}
    requires_grad::Bool
    Tensor{T}(a::Int, b::Int; requires_grad::Bool=true) where {T<:Real} = new(randn(T, a, b), zeros(T, a, b), nothing, requires_grad)
    Tensor{T}(data::AbstractArray{T}; requires_grad::Bool=true) where {T<:Real} = new(data, zeros(T, size(data)), nothing, requires_grad)
    Tensor{T}(data::AbstractArray{T}, op::Any) where {T<:Real} = new(data, zeros(T, size(data)), op, true)
    Tensor{T}(data::T) where {T<:Real} = new(data, zeros(T, 0), nothing, false)
    Tensor{T}(data::AbstractArray{T}, grad::AbstractArray{T}, op::Any, requires_grad::Bool) where {T<:Real} = new(data, grad, op, requires_grad)
end

Base.show(io::IO, t::Tensor) = print(io, "data = $(t.data) \ngrad = $(t.grad)")
Base.show(io::IO, t::AbstractOP) = print(io, "op = $(typeof(t))\nleft = $(t.left)\nright = $(t.right)")

zero_grad!(t::Tensor{T}) where {T<:Real} = fill!(t.grad, zero(T))
one_grad!(t::Tensor{T}) where {T<:Real} = fill!(t.grad, one(T))

# basic Operations

struct AddOP{T} <: AbstractOP
    left::Union{Tensor{T},T}
    right::Union{Tensor{T},T}
    out::Tensor{T}
end

function Base.:+(left::Tensor{T}, right::Tensor{T}) where {T}
    @assert size(left.data) == size(right.data)
    out = Tensor{T}(left.data + right.data)
    out.op = AddOP{T}(left, right, out)
    return out
end

function Base.:+(left::T, right::Tensor{T}) where {T}
    out = Tensor{T}(left + right.data)
    out.op = AddOP{T}(Tensor{T}(left), right, out)
    return out
end

function Base.:+(left::Tensor{T}, right::T) where {T}
    out = Tensor{T}(left.data + right)
    out.op = AddOP{T}(left, Tensor{T}(right), out)
    return out
end

function backward!(op::AddOP)
    op.left.requires_grad && (op.left.grad += op.out.grad)
    op.right.requires_grad && (op.right.grad += op.out.grad)
    nothing
end

struct MinusOP{T} <: AbstractOP
    left::Tensor{T}
    right::Tensor{T}
    out::Tensor{T}
end

function Base.:-(left::Tensor{T}, right::Tensor{T}) where {T}
    @assert size(left.data) == size(right.data)
    out = Tensor{T}(left.data - right.data)
    out.op = MinusOP{T}(left, right, out)
    return out
end

function Base.:-(left::T, right::Tensor{T}) where {T}
    out = Tensor{T}(left .- right.data)
    out.op = MinusOP{T}(Tensor{T}(left), right, out)
    return out
end

function Base.:-(left::Tensor{T}, right::T) where {T}
    out = Tensor{T}(left.data .- right)
    out.op = MinusOP{T}(left, Tensor{T}(right), out)
    return out
end

function Base.:-(right::Tensor{T}) where {T}
    out = Tensor{T}(zero(T) .- right.data)
    out.op = MinusOP{T}(Tensor{T}(zero(T)), right, out)
    return out
end

function backward!(op::MinusOP)
    isa(op.left, Tensor) && (op.left.grad += op.out.grad)
    isa(op.right, Tensor) && (op.right.grad -= op.out.grad)
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

function Base.:*(left::Tensor{T}, right::T) where {T}
    out = Tensor{T}(left.data * right)
    out.op = MulOP{T}(left, Tensor{T}(right), out)
    return out
end

function Base.:*(left::T, right::Tensor{T}) where {T}
    out = Tensor{T}(left.data * right)
    out.op = MulOP{T}(Tensor{T}(left), right, out)
    return out
end

function Base.:/(left::Tensor{T}, right::T) where {T}
    @assert right != zero(T)
    out = Tensor{T}(left.data / right)
    out.op = MulOP{T}(left, Tensor{T}(one(T) / right), out)
    return out
end

function backward!(op::MulOP)
    op.left.requires_grad && (op.left.grad += op.out.grad * op.right.data')
    op.right.requires_grad && (op.right.grad += op.left.data' * op.out.grad)
    nothing
end

# Activate functions
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


Base.:transpose(x::Tensor{T}) where {T} = Tensor{T}(x.data', x.grad', x.op, x.requires_grad)

#=
TODO:
Operations
 [-] negative operation
 [-] subtraction
 [-] division
 [-] transpose
Activate functions
 [] sigmoid
 [] tanh
Loss functions
 [] cross entropy
Dropout
Batch Operation
Optimization
 [] SGD
 [] Adam
=#

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

if abspath(PROGRAM_FILE) == @__FILE__

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

end

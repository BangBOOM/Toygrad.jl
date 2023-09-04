mutable struct Tensor{T<:Real}
    data::Array{T}
    grad::Array{T}
    parent::Set{Tensor{T}}
    _op::String
    _backward::Function

    Tensor{T}(data::Array{T}) where {T<:Real} = new(data, zeros(T, size(data)), Set{Tensor{T}}(), "", () -> nothing)
    Tensor{T}(data::Array{T}, parent::Set{Tensor{T}}, op::String) where {T<:Real} = new(data, zeros(T, size(data)), parent, op, () -> nothing)
end

zero_grad!(t::Tensor{T}) where {T<:Real} = fill!(t.grad, zero(T))
one_grad!(t::Tensor{T}) where {T<:Real} = fill!(t.grad, one(T))
update!(t::Tensor{T}, lr::T) where {T<:Real} = t.data .-= lr .* t.grad

function Base.:+(a::Tensor{T}, b::Tensor{T}) where {T<:Real}
    @assert size(a.data) == size(b.data)
    out = Tensor{T}(a.data + b.data, Set{Tensor{T}}([a, b]), "+")
    function _backward()
        a.grad += out.grad
        b.grad += out.grad
        nothing
    end
    out._backward = _backward
    out
end

function Base.:*(a::T, b::Tensor{T}) where {T<:Real}
    out = Tensor{T}(a * b.data, Set{Tensor{T}}([b]), "$a*")
    function _backward()
        b.grad += a * out.grad
        nothing
    end
    out._backward = _backward
    out
end

function Base.:*(a::Tensor{T}, b::Tensor{T}) where {T<:Real}
    @assert size(a.data)[2] == size(b.data)[1]
    out = Tensor{T}(a.data * b.data, Set{Tensor{T}}([a, b]), "*")
    function _backward()
        a.grad += out.grad * b.data'
        b.grad += a.data' * out.grad
        nothing
    end
    out._backward = _backward
    out
end

function Base.show(io::IO, t::Tensor{T}) where {T<:Real}
    println(io, "Tensor{$T}(")
    println(io, "    data: ", t.data)
    println(io, "    grad: ", t.grad)
    println(io, ")")
end

function Base.:tanh(x::Tensor{T}) where {T<:Real}
    out = Tensor{T}(tanh.(x.data), Set{Tensor{T}}([x]), "tanh")
    function _backward()
        #TODO
        nothing
    end
    out._backward = _backward
    out
end

struct Layer{T}
    w::Tensor{T}
    b::Tensor{T}
    # activation::Function 

    Layer{T}(w::Tensor{T}, b::Tensor{T}) where {T<:Real} = new(w, b)
    Layer{T}(nin::Int, nout::Int) where {T<:Real} = new(Tensor{T}(randn(T, nout, nin)), Tensor{T}(zeros(T, nout)))
end

zero_grad!(l::Layer) = begin
    zero_grad!(l.w)
    zero_grad!(l.b)
    nothing
end

(l::Layer)(x) = l.w * x + l.b

struct MLP{T<:Real}
    layers::Vector{Layer{T}}
    MLP{T}(layers::Vector{Layer{T}}) where {T<:Real} = new(layers)
end

function MLP{T}(nin::Int, nouts::Vector{Int}) where {T<:Real}
    @assert length(nouts) > 0
    @assert all(x -> x > 0, nouts)
    @assert nouts[end] == 1

    nouts = [nin; nouts]
    layers = [Layer{T}(i, o) for (i, o) in zip(nouts[1:end-1], nouts[2:end])]
    MLP{T}(layers)
end

zero_grad!(mlp::MLP) = begin
    for l in mlp.layers
        zero_grad!(l)
    end
    nothing
end


forward(mlp::MLP{T}, x::Tensor{T}) where {T<:Real} = begin
    for l in mlp.layers
        x = l(x)
    end
    x
end

backward!(o::Tensor{T}) where {T<:Real} = begin
    one_grad!(o)
    back_list = [o]
    while !isempty(back_list)
        t = pop!(back_list)
        t._backward()
        for p in t.parent
            push!(back_list, p)
        end
    end
end



function main()
    w1 = Tensor{Float32}([1.0f0 2.0f0; 3.0f0 4.0f0; 5.0f0 6.0f0])
    b1 = Tensor{Float32}([2.0f0; 3.0f0; 4.0f0])

    l1 = Layer{Float32}(w1, b1)

    w2 = Tensor{Float32}([5.0f0 2.0f0 3.0f0])
    b2 = Tensor{Float32}([4.0f0])

    l2 = Layer{Float32}(w2, b2)

    x = Tensor{Float32}([2.0f0; 3.0f0])

    layers = [l1, l2]
    mlp = MLP{Float32}(layers)

    o = forward(mlp, x)
    zero_grad!(mlp)

    @show o

    backward!(o)

    @show w2
    @show b2

    @show w1
    @show b1
end


main()

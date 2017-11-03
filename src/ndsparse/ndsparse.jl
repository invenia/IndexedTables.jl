export NDSparse, ndsparse

struct NDSparse{T, D<:Tuple, C<:Columns, V<:AbstractVector}
    index::C
    data::V
    _table::NextTable

    index_buffer::C
    data_buffer::V
end

function NextTable(nds::NDSparse; kwargs...)
    convert(NextTable, nds.index, nds.data; kwargs...)
end

convert(::Type{NextTable}, nd::NDSparse) = NextTable(nd)

Base.@deprecate_binding IndexedTable NDSparse

# optional, non-exported name
Base.@deprecate_binding Table NDSparse


"""
    ndsparse(indices::Columns, data::AbstractVector; agg, presorted, copy, chunks)

Construct an NDSparse array with the given indices and data. Each vector in `indices` represents the index values for one dimension. On construction, the indices and data are sorted in lexicographic order of the indices.

# Arguments:

* `agg::Function`: If `indices` contains duplicate entries, the corresponding data items are reduced using this 2-argument function.
* `presorted::Bool`: If true, the indices are assumed to already be sorted and no sorting is done.
* `copy::Bool`: If true, the storage for the new array will not be shared with the passed indices and data. If false (the default), the passed arrays will be copied only if necessary for sorting. The only way to guarantee sharing of data is to pass `presorted=true`.
* `chunks::Integer`: distribute the table into `chunks` (Integer) chunks (a safe bet is nworkers()). Not distributed by default. See [Distributed](@distributed) docs.
"""
function ndsparse(I::C, d::AbstractVector{T}; agg=nothing, presorted=false, copy=false) where {T,C<:Columns}
    length(I) == length(d) || error("index and data must have the same number of elements")

    if !presorted && !issorted(I)
        p = sortperm(I)
        I = I[p]
        d = d[p]
    elseif copy
        if agg !== nothing
            I, d = groupreduce_to!(agg, I, d, similar(I, 0),
                                   similar(d,0), Base.OneTo(length(I)))
            agg = nothing
        else
            I = Base.copy(I)
            d = Base.copy(d)
        end
    end
    stripnames(x) = rows(astuple(columns(x)))
    _table = convert(NextTable, stripnames(I), stripnames(d); presorted=true, copy=false)
    nd = NDSparse{T,astuple(eltype(C)),C,typeof(d)}(I, d, _table, similar(I,0), similar(d,0))
    agg===nothing || aggregate!(agg, nd)
    return nd
end

# backwards compat
NDSparse(idx::Columns, data; kwargs...) = ndsparse(idx, data; kwargs...)

# TableLike API
Base.@pure function colnames(t::NDSparse)
    dnames = colnames(t.data)
    if all(x->isa(x, Integer), dnames)
        dnames = map(x->x+ncols(t.index), dnames)
    end
    vcat(colnames(t.index), dnames)
end

columns(nd::NDSparse) = concat_tup(columns(nd.index), columns(nd.data))

# IndexedTableLike API

permcache(t::NDSparse) = permcache(t._table)
cacheperm!(t::NDSparse, p) = cacheperm!(t._table, p)
pkeynames(t::NDSparse) = (dimlabels(t)...)

"""
`NDSparse(columns...; names=Symbol[...], kwargs...)`

Construct an NDSparse array from columns. The last argument is the data column, and the rest are index columns. The `names` keyword argument optionally specifies names for the index columns (dimensions).
"""
function NDSparse(columns...; names=nothing, rest...)
    keys, data = columns[1:end-1], columns[end]
    ndsparse(Columns(keys..., names=names), data; rest...)
end

similar(t::NDSparse) = NDSparse(similar(t.index, 0), similar(t.data, 0))

function copy(t::NDSparse)
    flush!(t)
    NDSparse(copy(t.index), copy(t.data), presorted=true)
end

function (==)(a::NDSparse, b::NDSparse)
    flush!(a); flush!(b)
    return a.index == b.index && a.data == b.data
end

function empty!(t::NDSparse)
    empty!(t.index)
    empty!(t.data)
    empty!(t.index_buffer)
    empty!(t.data_buffer)
    return t
end

_convert(::Type{<:Tuple}, tup::Tuple) = tup
_convert{T<:NamedTuple}(::Type{T}, tup::Tuple) = T(tup...)
convertkey(t::NDSparse{V,K,I}, tup::Tuple) where {V,K,I} = _convert(eltype(I), tup)

ndims(t::NDSparse) = length(t.index.columns)
length(t::NDSparse) = (flush!(t);length(t.index))
eltype{T,D,C,V}(::Type{NDSparse{T,D,C,V}}) = T
dimlabels{T,D,C,V}(::Type{NDSparse{T,D,C,V}}) = fieldnames(eltype(C))

# Generic ndsparse constructor that also works with distributed
# arrays in JuliaDB

function ndsparse(keycols::Tup, valuecols::Tup)
    NDSparse(rows(keycols), rows(valuecols))
end

Base.@deprecate itable(x, y) ndsparse(x, y)

# Keys and Values iterators

"""
`keys(t::NDSparse)`

Returns an array of the keys in `t` as tuples or named tuples.
"""
keys(t::NDSparse) = t.index

"""
`keys(t, which...)`

Returns a array of rows from a subset of columns
in the index of `t`. `which` is either an `Int`, `Symbol` or [`As`](@ref)
or a tuple of these types.
"""
keys(t::NDSparse, which...) = rows(keys(t), which...)

# works for both NextTable and NDSparse
primarykeys(t::NDSparse, which...) = keys(t, which...)

"""
`values(t)`

Returns an array of values stored in `t`.
"""
values(t::NDSparse) = t.data

"""
`values(t, which...)`

Returns a array of rows from a subset of columns
of the values in `t`. `which` is either an `Int`, `Symbol` or [`As`](@ref)
or a tuple of these types.
"""
values(t::NDSparse, which...) = rows(values(t), which...)

## Some array-like API

"""
`dimlabels(t::NDSparse)`

Returns an array of integers or symbols giving the labels for the dimensions of `t`.
`ndims(t) == length(dimlabels(t))`.
"""
dimlabels(t::NDSparse) = dimlabels(typeof(t))

start(a::NDSparse) = start(a.data)
next(a::NDSparse, st) = next(a.data, st)
done(a::NDSparse, st) = done(a.data, st)

function permutedims(t::NDSparse, p::AbstractVector)
    if !(length(p) == ndims(t) && isperm(p))
        throw(ArgumentError("argument to permutedims must be a valid permutation"))
    end
    flush!(t)
    NDSparse(Columns(t.index.columns[p]), t.data, copy=true)
end

# showing

import Base.show
function show(io::IO, t::NDSparse{T,D}) where {T,D}
    flush!(t)
    if !(values(t) isa Columns)
        cnames = colnames(keys(t))
        eltypeheader = "$(eltype(t))"
    else
        cnames = colnames(t)
        nf = nfields(eltype(t))
        if eltype(t) <: NamedTuple
            eltypeheader = "$(nf) field named tuples"
        else
            eltypeheader = "$(nf)-tuples"
        end
    end
    header = "$(ndims(t))-d NDSparse with $(length(t)) values (" * eltypeheader * "):"
    showtable(io, t; header=header,
              cnames=cnames, divider=length(columns(keys(t))))
end

import Base: @md_str

function showmeta(io, t::NDSparse, cnames)
    nc = length(columns(t))
    nidx = length(columns(keys(t)))
    nkeys = length(columns(values(t)))

    print(io,"    ")
    with_output_format(:underline, println, io, "Dimensions")
    metat = Columns(([1:nidx;], [Text(get(cnames, i, "<noname>")) for i in 1:nidx],
                     eltype.([columns(keys(t))...])))
    showtable(io, metat, cnames=["#", "colname", "type"], cstyle=fill(:bold, nc), full=true)
    print(io,"\n    ")
    with_output_format(:underline, println, io, "Values")
    if isa(values(t), Columns)
        metat = Columns(([nidx+1:nkeys+nidx;], [Text(get(cnames, i, "<noname>")) for i in nidx+1:nkeys+nidx],
                         eltype.([columns(values(t))...])))
        showtable(io, metat, cnames=["#", "colname", "type"], cstyle=fill(:bold, nc), full=true)
    else
        show(io, eltype(values(t)))
    end
end

abstract type SerializedNDSparse end

function serialize(s::AbstractSerializer, x::NDSparse)
    flush!(x)
    Base.Serializer.serialize_type(s, SerializedNDSparse)
    serialize(s, x.index)
    serialize(s, x.data)
end

function deserialize(s::AbstractSerializer, ::Type{SerializedNDSparse})
    I = deserialize(s)
    d = deserialize(s)
    NDSparse(I, d, presorted=true)
end

# map and convert

function _map(f, xs)
    T = _promote_op(f, eltype(xs))
    if T<:Tup
        out_T = arrayof(T)
        out = similar(out_T, length(xs))
        map!(f, out, xs)
    else
        map(f, xs)
    end
end

function map(f, x::NDSparse)
    NDSparse(copy(x.index), _map(f, x.data), presorted=true)
end

# lift projection on arrays of structs
map(p::Proj, x::NDSparse{T,D,C,V}) where {T,D<:Tuple,C<:Tup,V<:Columns} =
    NDSparse(x.index, p(x.data.columns), presorted=true)

(p::Proj)(x::NDSparse) = map(p, x)

# """
# `columns(x::NDSparse, names...)`
#
# Given an NDSparse array with multiple data columns (its data vector is a `Columns` object), return a
# new array with the specified subset of data columns. Data is shared with the original array.
# """
# columns(x::NDSparse, which...) = NDSparse(x.index, Columns(x.data.columns[[which...]]), presorted=true)

#columns(x::NDSparse, which) = NDSparse(x.index, x.data.columns[which], presorted=true)

#column(x::NDSparse, which) = columns(x, which)

# NDSparse uses lex order, Base arrays use colex order, so we need to
# reorder the data. transpose and permutedims are used for this.
convert(::Type{NDSparse}, m::SparseMatrixCSC) = NDSparse(findnz(m.')[[2,1,3]]..., presorted=true)

function convert{T}(::Type{NDSparse}, a::AbstractArray{T})
    n = length(a)
    nd = ndims(a)
    a = permutedims(a, [nd:-1:1;])
    data = reshape(a, (n,))
    idxs = [ Vector{Int}(n) for i = 1:nd ]
    i = 1
    for I in CartesianRange(size(a))
        for j = 1:nd
            idxs[j][i] = I[j]
        end
        i += 1
    end
    NDSparse(Columns(reverse(idxs)...), data, presorted=true)
end
module ChunkedArrayBase
export eachchunk, copy_chunked, copy_chunked!
"""
    eachchunk(a)

Returns an iterator with `CartesianIndices` elements that mark the index range of each chunk within an array.
"""
function eachchunk end

struct GridChunks{N}
    parentsize::NTuple{N,Int}
    chunksize::NTuple{N,Int}
    chunkgridsize::NTuple{N,Int}
end
GridChunks(a, chunksize) = GridChunks(size(a), chunksize, map(fld1,size(a),chunksize))
function Base.show(io::IO, g::GridChunks)
  print(io,"Regular ",join(g.chunksize,"x")," chunks over a ", join(g.parentsize,"x"), " array.")
end
Base.size(g::GridChunks) = g.chunkgridsize
Base.size(g::GridChunks, dim) = g.chunkgridsize[dim]
Base.IteratorSize(::Type{GridChunks{N}}) where N = Base.HasShape{N}()
Base.eltype(::Type{GridChunks{N}}) where N = CartesianIndices{N,NTuple{N,UnitRange{Int64}}}
Base.length(c) = prod(size(c))
@inline function _iterate(g,r)
    if r === nothing
        return nothing
    else
        ichunk, state = r
        outinds = map(ichunk.I, g.chunksize, g.parentsize) do ic, cs, ps
            (ic-1)*cs+1:min(ic*cs, ps)
        end |> CartesianIndices
        outinds, state
    end
end
function Base.iterate(g::GridChunks)
    r = iterate(CartesianIndices(g.chunkgridsize))
    _iterate(g,r)
end
function Base.iterate(g::GridChunks, state)
    r = iterate(CartesianIndices(g.chunkgridsize), state)
    _iterate(g,r)
end

function copy_chunked!(dest, src)
    for c in eachchunk(src)
        dest[c.indices...] = src[c.indices...]
    end
    dest
end
function copy_chunked(src)
    dest = Array{eltype(src),ndims(src)}(undef,size(src)...)
    copy_chunked!(dest,src)
end

end # module

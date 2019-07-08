# ChunkedArrayBase.jl

Proposal for an interface for chunked array stores. See also the discussion in discourse https://discourse.julialang.org/t/common-interface-for-chunked-arrays/26009

Basically this package only defines a single functions `eachchunk` that other packages can add methods to. Here `eachchunk` an iterator that loops over the `CartesianIndices` of every chunk. For example:

```julia
using ChunkedArrayBase
ds = HDF5.h5open("mydata.h5")["A"]
eachchunk(ds)
```

would return an iterator over the indices of each chunk. Below is a short (executable) demonstration of code that would be possible if this is implemented.  

## 1. Implement the interface for different backends

This is usually just a line of code and would have to be done inside the individual packages


```julia
#HDF5
import HDF5
function ChunkedArrayBase.eachchunk(a::HDF5.HDF5Dataset)
    cs = HDF5.get_chunk(a)
    ChunkedArrayBase.GridChunks(a,cs)
end
#Zarr
import Zarr
ChunkedArrayBase.eachchunk(a::Zarr.ZArray) = ChunkedArrayBase.GridChunks(a, a.metadata.chunks)
#DistributedArrays
import DistributedArrays
ChunkedArrayBase.eachchunk(a::DistributedArrays.DArray) = ChunkedArrayBase.GridChunks(a, map(length,first(a.indices)))
#NetCDF
import NetCDF
ChunkedArrayBase.eachchunk(a::NetCDF.NcVar) = ChunkedArrayBase.GridChunks(a, map(Int64,a.chunksize))
```

### Example 1: Copy a HDF5 file to Zarr chunk by chunk

First we create a dummy HDF5 file as a source


```julia
HDF5.h5open("mydata.h5", "w") do f
    f["A", "chunk", (5,5)] = rand(100,50)
end
```

    100×50 Array{Float64,2}:
     0.487943   0.74488   0.914045   0.398469  …  0.0777673  0.0993469  0.955354
     0.083824   0.539686  0.059137   0.516492     0.384592   0.545417   0.358116
     0.343853   0.501171  0.372518   0.837062     0.885167   0.106328   0.868845
     ⋮                                         ⋱                                 
     0.378929   0.584512  0.127812   0.731926     0.588496   0.823291   0.751509
     0.62482    0.212712  0.186356   0.536405     0.262975   0.44097    0.958594
     0.182579   0.655219  0.629535   0.238136     0.736597   0.272199   0.634228



Create our sink Zarr array


```julia
zout = Zarr.zcreate(Float64, 100,50, path = "output.zarr",chunks = (5,5))
```

    ZArray{Float64} of size 100 x 50


And copy the data chunk by chunk. The current implementation simply iterates over source array chunks, in the future it may be better to do some optimization on source and destination chunkings.


```julia
HDF5.h5open("mydata.h5") do f
    copy_chunked!(zout,f["A"])
end
```

    ZArray{Float64} of size 100 x 50


```julia
all(zout[:,:].==HDF5.h5read("mydata.h5","A"))
```

    true

## Example 2: Packages defining particular sink functions for chunked arrays

The workflow above could in principle be automatised. For example, the Zarr package might want to provide a function that saves any array implementing the `eachchunk` function as a ZArray with appropriate chunk settings. This could look like this:


```julia
function chunked_to_zarr(a; kwargs...)
    cI = eachchunk(a)
    cI isa ChunkedArrayBase.GridChunks || error("Can only converted regular chunk grids to Zarr")
    cs = size(first(cI))
    zout = Zarr.zcreate(Float64, size(a)...; chunks = cs, kwargs...)
    copy_chunked!(zout,a)
end
```




    chunked_to_zarr (generic function with 1 method)



Then we apply the function as:


```julia
a = HDF5.h5open("mydata.h5") do f
    chunked_to_zarr(f["A"], path="output2.zarr")
end
```




    ZArray{Float64} of size 100 x 50



## Use pmap over chunks of a Zarr Array

To show how this simplifies doing computations over chunked arrays: First we add a few workers and open a Zarr dataset that every worker has access to.


```julia
using Distributed
addprocs(4)
@everywhere begin
    using Zarr, OnlineStats
    zar = zopen("output2.zarr")
end
```

And then we fit an Online Histogram to the data:


```julia
r = pmap(i->fit!(KHist(100),zar[i.indices...]),eachchunk(zar))
hist_all = reduce(merge, r)
```




    KHist: n=5000 | value=(x = [0.000205246, 0.00554865, 0.0167523, 0.0254097, 0.0336459, 0.0427167, 0.0534323, 0.0639424, 0.0754509, 0.0852213  …  0.91412, 0.92511, 0.935388, 0.94598, 0.957331, 0.967112, 0.976546, 0.984996, 0.994557, 0.999839], y = [1, 52, 39, 37, 50, 44, 55, 55, 49, 48  …  40, 52, 60, 56, 63, 53, 54, 42, 55, 1])



Note that almost the same code would work for any other array that implements this interface. The only Zarr-specific part is that the file needs to be open on all workers. Maybe this could become part of the interface later

## Interacting with DistributedArrays

DistributedArrays are a special case here, because they are not a pure data backend but it would be very nice to have a generic way to interact with them. For example assume we want to write a DistributedArray into a Zarr dataset. This is exactly what Zarr is made for, you can do concurrent writes as long as each process writes into different chunks. However just applying our function works, but does not do the correct thing:


```julia
@everywhere using DistributedArrays
a = drand(Float64,(100,100),workers(),[2,2]);
znew = chunked_to_zarr(a,name = "output5.zarr")
```




    ZArray{Float64} of size 100 x 100



Because this always uses the main process to write the data. An extension of the interface that includes worker affinities would be something like this:


```julia
is_distributed(::Any) = false
is_distributed(::DArray) = true
eachchunkworker(::Any) = nothing
eachchunkworker(d::DArray) = d.pids

@everywhere function copy_chunked_dist!(dest,src)
   for (c,proc) in zip(eachchunk(src), eachchunkworker(src))
        @spawnat proc r[c.indices...] = src[c.indices...]
   end
    dest
end
```

And then we define the chunked_to_zarr function:


```julia
function chunked_to_zarr(a; kwargs...)
    cI = eachchunk(a)
    cI isa ChunkedArrayBase.GridChunks || error("Can only converted regular chunk grids to Zarr")
    cs = size(first(cI))
    zout = Zarr.zcreate(Float64, size(a)...; chunks = cs, kwargs...)
    if is_distributed(a)
        @everywhere r = $zout
        copy_chunked_dist!(r,a)
    else
        copy_chunked!(zout,a)
    end
end
```




    chunked_to_zarr (generic function with 1 method)




```julia
rr = chunked_to_zarr(a,path="output6.zarr")
```




    ZArray{Float64} of size 100 x 100




```julia
rr[:,:]
```




    100×100 Array{Float64,2}:
     0.240908   0.527813    0.228223  …  0.557935   0.425919     0.708251  
     0.579776   0.230842    0.737792     0.24166    0.58953      0.0565524
     0.347023   0.608021    0.394577     0.0079891  0.572144     0.58169   
     0.693292   0.648999    0.952134     0.772287   0.359551     0.387697  
     0.864584   0.00637337  0.120113     0.308075   0.0377649    0.570447  
     0.670837   0.258396    0.532635  …  0.408991   0.570901     0.617165  
     0.848143   0.310697    0.989882     0.522368   0.136365     0.552576  
     0.655658   0.921542    0.181077     0.98976    0.501077     0.284172  
     0.733931   0.653235    0.269191     0.83299    0.277243     0.127147  
     0.0822896  0.713552    0.185611     0.412192   0.130851     0.0241448
     0.728169   0.950998    0.336277  …  0.303407   0.000289681  0.394205  
     0.850385   0.603225    0.289752     0.416672   0.136694     0.924237  
     0.743653   0.678501    0.312576     0.0593383  0.236865     0.447326  
     ⋮                                ⋱                                    
     0.713929   0.501844    0.963084     0.459341   0.158043     0.785527  
     0.248041   0.600882    0.185266     0.455192   0.558921     0.714065  
     0.167946   0.402836    0.197628  …  0.677688   0.337543     0.653794  
     0.513297   0.560373    0.440744     0.123954   0.430283     0.818652  
     0.708142   0.414104    0.224189     0.376063   0.906001     0.213997  
     0.540566   0.835374    0.510687     0.742785   0.644839     0.549113  
     0.88534    0.266501    0.784352     0.347411   0.193772     0.924712  
     0.476398   0.832003    0.180794  …  0.869917   0.471489     0.00541684
     0.618048   0.572681    0.759495     0.682645   0.876602     0.0382101
     0.504003   0.58133     0.745643     0.353857   0.764372     0.856206  
     0.760044   0.895416    0.913169     0.81009    0.268661     0.152116  
     0.365744   0.370778    0.450759     0.854643   0.485486     0.856109  




```julia
rmprocs(workers())
```




    Task (done) @0x00007fe7b6c38010

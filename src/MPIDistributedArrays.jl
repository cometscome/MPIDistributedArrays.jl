module MPIDistributedArrays
    using MPI
    export MArray,mzeros,distribute,free
# Write your package code here.

    const masterrank = 0

    function initialize()
        MPI.Init()
    end

    function Comm_size()
        return  MPI.Comm_size(MPI.COMM_WORLD)
    end



    struct MArray{T,N,A} <: AbstractArray{T,N}
        localpart::A#Union{Nothing,A}
        localindices::Array{UnitRange{Int64},1}
        myrank::Int64
        nprocs::Int64
        comm::MPI.Comm
        arrayrank::Array{Int64,1}
        win::MPI.Win
        localindices_set::Array{Array{UnitRange{Int64},1},1}
        size::NTuple{N,Int64}
        localsize::NTuple{N,Int64}

        MArray{T}(dims::Dims,parallel;debug= false) where {T} = MArray{T}(MPI.COMM_WORLD,dims,parallel,debug= debug)
        MArray(::Type{T},dims::Dims,parallel;debug= false) where {T} = MArray{T}(MPI.COMM_WORLD,dims,parallel,debug= debug)
        MArray(::Type{T},comm::MPI.Comm,dims::Dims,parallel;debug= false) where {T} = MArray{T}(comm,dims,parallel,debug= debug)

        function MArray{T}(comm::MPI.Comm,dims::Dims,parallel;debug= false) where {T}
            @assert length(dims) == length(parallel) "dimension mismatch! length(dims) = $(length(dims)) but length(parallel) = $(length(parallel))"
            N = length(dims)
            nprocs = MPI.Comm_size(comm)
            myrank = MPI.Comm_rank(comm)

            #println(typeof(comm))
            #exit()
            
            if prod(parallel) != nprocs
                if myrank == masterrank
                    println("number of MPI processes is mismatched. nprocs = $nprocs and parallel = $parallel")
                end     
                MPI.Barrier(comm)     
            end
            arrayrank = get_arrayrank(myrank,parallel)
            localindices = Array{UnitRange{Int64},1}(undef,N)
            localdims = zeros(Int64,N)
            

            for idim =  1:length(dims)
                @assert dims[idim] % parallel[idim] == 0 "$idim-th dimension: dims[$idim] % parallel[$idim] should be 0! now dims[$idim] = $(dims[idim]) and parallel[$idim] = $(parallel[idim])"
                ndim = dims[idim] ÷ parallel[idim]
                istart = arrayrank[idim]*ndim + 1
                iend = istart + ndim-1
                localindices[idim] = istart:iend
                localdims[idim] = ndim
            end

            localindices_set= Array{Array{UnitRange{Int64},1},1}(undef,nprocs)

            for rank=0:nprocs-1
                localindices_set[rank+1] = copy(localindices)
                MPI.Bcast!(localindices_set[rank+1], rank, comm)

                
            end


            if debug
                for rank=0:nprocs-1
                    #isinside = get_local_or_not(localindices_set[rank+1],1,1,1)


                    if rank == myrank
                        println("myrank : $myrank localindices = ",localindices)
                        #println("isinside = $isinside")
                        #println("localindices_set: $localindices_set")
                    end
                    MPI.Barrier(comm)   
                end
            end

            localpart = Array{T,N}(undef,localdims...)
            win = MPI.Win_create(localpart,comm)
            return new{T,N,Array{T,N}}(localpart,localindices,myrank,nprocs,comm,arrayrank,win,
            localindices_set,dims,size(localpart))

        end

        

        
    end

    function free(a::MArray{T,N}) where {T,N}
        MPI.Barrier(a.comm) 
        MPI.free(a.win)
    end

    Base.size(a::MArray) = a.size

    function Base.getindex(a::MArray{T,N,A},i...) where {T,N,A}
        isinside,lenind = get_local_or_not(a.localindices,i...)
        if isinside
            index = [i[idim] - a.localindices[idim][1]+1 for idim=1:lenind]
            return a.localpart[index...]
        else
            result = Ref{T}()
            target_rank,local_index = get_targetrank_and_index(a,i...)
            #println("target_rank ",target_rank)
            if target_rank == a.myrank
                #println("target $local_index",a.localpart[local_index])
                result[] = a.localpart[local_index]
            else
                #println("localindex = $local_index")
                MPI.Win_lock(MPI.LOCK_SHARED, target_rank, 0, a.win)
                MPI.Get(result,  target_rank, local_index-1 , a.win)
                MPI.Win_unlock(target_rank, a.win)
            end
            return result[]
        end
    end

    function get_local_or_not(indices,i...)
        #println("indices, $indices i = $i")
        lenind = length(indices)
        isinside = true
        for idim = 1:lenind
            isinside *= i[idim] in indices[idim]
        end
        return isinside,lenind
    end

    function get_targetrank_and_index(A::MArray,i...)
        for rank=0:A.nprocs
            isinside,lenind = get_local_or_not(A.localindices_set[rank+1],i...)
            if isinside
                index = [i[idim] - A.localindices_set[rank+1][idim][1]+1 for idim=1:lenind]
                #println("index = $index")
                prod = 1
                index_1d = 1
                for idim=1:lenind
                    index_1d += prod*(i[idim] - A.localindices_set[rank+1][idim][1]+1-1)
                    prod *= length(A.localindices_set[rank+1][idim])
                end
                return rank,index_1d
            end
        end
    end

    mzeros(comm::MPI.Comm,dims::Dims,parallel;debug=false)  = mzeros(Float64,comm,dims,parallel,debug=debug)
    mzeros(dims::Dims,parallel;debug=false)  = mzeros(Float64,MPI.COMM_WORLD,dims,parallel,debug=debug)
    mzeros(::Type{T},dims::Dims,parallel;debug=false) where {T} = mzeros(T,MPI.COMM_WORLD,dims,parallel,debug=debug)

    function mzeros(::Type{T},comm::MPI.Comm,dims::Dims,parallel;debug=false) where {T}
        A = MArray{T}(comm,dims,parallel,debug= debug)   
        A.localpart .= zero(T)
        return A
    end

    distribute(rank::I,A::T,parallel;debug = false) where {T <: AbstractArray, I <: Int} = distribute(rank,MPI.COMM_WORLD,A,parallel,debug = debug)
    distribute(A::T,parallel;debug = false) where {T <: AbstractArray} = distribute(masterrank,MPI.COMM_WORLD,A,parallel,debug = debug)
    distribute(comm::MPI.Comm,A::T,parallel;debug = false) where {T <: AbstractArray} =distribute(masterrank,comm,A,parallel,debug = debug)

    function distribute(rank::I,comm::MPI.Comm,A::T,parallel;debug = false) where {T <: AbstractArray, I <:Int}
        MPI.Bcast!(A, rank, comm)
        A_local = mzeros(eltype(A),size(A),parallel,debug = debug)
        local_length = length(A_local.localpart)
        localindex =  zero(A_local.arrayrank)

        globalindex =  zero(A_local.arrayrank)
        for i=1:local_length
            get_localindex!(localindex,A_local ,i)
            get_globalindex!(globalindex,A_local,localindex,)
            A_local.localpart[localindex...] = A[globalindex...]
            #println("local $localindex",A_local.localpart[localindex...])
            #println("global $globalindex",A[globalindex...])
        end
        return A_local
    end

    function get_arrayrank(myrank,parallel)
        arrayrank = zero(parallel)
        psize = 1
        myranktemp = myrank
        for idim = 1:length(parallel)
            psize = parallel[idim]
            arrayrank[idim] = myranktemp % psize
            myranktemp = (myranktemp-arrayrank[idim]) ÷ psize
        end
        return arrayrank
    end

    function get_myrankfromarray(arrayrank,parallel)
        myrank = 0
        psize = 1
        for idim = 1:length(parallel)
            @assert arrayrank[idim] < parallel[idim]
            myrank += arrayrank[idim]*psize
            psize *= parallel[idim]
        end
        return myrank
    end

    function get_globalindex!(globalindex,A::MArray,localindex)
        for idim = 1:length(localindex)
            #println(A.localindices[idim])
            globalindex[idim] = A.localindices[idim][localindex[idim]]
        end
        return
    end

    function get_localindex!(localindex,A::MArray,i)
        #localindex =  zero(A.arrayrank)
        localsize = size(A.localpart)
        psize = 1
        itemp = i-1
        for idim = 1:length(localindex)
            psize = localsize[idim]
            #println(psize)
            localindex[idim] = itemp % psize
            itemp = (itemp-localindex[idim]) ÷ psize
        end
        localindex .+= 1
        return
        #return localindex .+ 1

    end

    function get_localindex(A::MArray,i)
        localindex =  zero(A.arrayrank)
        get_localindex!(localindex,A::MArray,i)
        return localindex
    end
end
